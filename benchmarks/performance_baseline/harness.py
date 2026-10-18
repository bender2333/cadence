"""Multi-scale performance benchmark harness for Cadence on CPU and NPU backends."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

# ponytail: minimal harness helper set; add digest/sources if standalone packaging audit needed.

def host_memory_bytes() -> int:
    """Peak resident set size of process in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1 if sys.platform == "darwin" else 1024))


def run_benchmark(
    backend: str,
    device: str | None = None,
    size: int = 64,
    batch: int = 1,
    iterations: int = 5,
    warmup: int = 1,
    seed: int = 42,
    tolerance: float = 0.003,
    budget: int = 128,
) -> dict[str, Any]:
    """Run performance profiling on a single network configuration.

    Profiles:
    1. Free Relaxation: NeuralGraph.equilibrate latency, steps/sec, residual.
    2. Nudged Phase & Teaching: Contrast computation latency and throughput.
    3. Synaptic Plasticity: Learner.step latency, parameter updates/sec.
    4. Memory & Resource: Host RSS and NPU device memory.
    """
    assert backend in ("cpu", "torch_cpu", "npu"), f"Unsupported backend: {backend}"
    is_npu = backend == "npu"
    is_torch = backend in ("npu", "torch_cpu")

    if device is None:
        device = "npu:0" if is_npu else "cpu"

    torch_module: Any = None
    if is_torch:
        import torch

        torch_module = torch
        if is_npu:
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                raise RuntimeError("Ascend NPU requested but not available")
            torch.npu.set_device(device)
            torch.npu.reset_peak_memory_stats(device)
        torch.manual_seed(seed)

    def sync() -> None:
        if is_npu and torch_module is not None:
            torch_module.npu.synchronize(device)

    # 1. Connectome and model construction
    rng = np.random.default_rng(seed)
    inputs = max(8, size // 32)
    outputs = max(4, size // 64)
    hidden = size - inputs - outputs

    conn = cd.layered(inputs=inputs, hidden=hidden, outputs=outputs, seed=seed)
    model = cd.learning_neuron_model(dt=1.0)
    precision = "float32" if is_torch else None
    graph_backend = "torch" if is_torch else "cpu"

    graph = cd.NeuralGraph(
        conn,
        model,
        backend=graph_backend,
        device=device,
        precision=precision,
    )
    learner = cd.Learner(graph, outputs=list(range(size - outputs, size)))

    # Sensory drive and teacher targets
    drive = np.zeros((batch, size), dtype=float)
    drive[:, :inputs] = rng.normal(0.0, 0.1, size=(batch, inputs))
    labels = rng.integers(0, outputs, size=batch)

    # Memory baseline before workload
    sync()
    rss_before_bytes = host_memory_bytes()

    # --- Profile 1: Free Relaxation ---
    # Cold warmup (first execution may trigger JIT compilation on CANN)
    sync()
    t_free_cold_start = time.perf_counter()
    eq_cold = graph.equilibrate(drive, tolerance=tolerance, budget=budget)
    sync()
    cold_free_ms = (time.perf_counter() - t_free_cold_start) * 1000.0

    cold_residual = float(np.max(eq_cold.residual))
    if cold_residual > tolerance:
        raise AssertionError(
            f"Cold equilibration residual {cold_residual:.6e} exceeded tolerance {tolerance}"
        )

    for _ in range(max(0, warmup - 1)):
        graph.equilibrate(drive, tolerance=tolerance, budget=budget)
        sync()

    free_latencies_ms: list[float] = []
    free_steps: list[int] = []
    free_residuals: list[float] = []
    latest_free_state: Any = None

    for _ in range(iterations):
        sync()
        t0 = time.perf_counter()
        eq = graph.equilibrate(drive, tolerance=tolerance, budget=budget)
        sync()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        free_latencies_ms.append(lat_ms)
        free_steps.append(eq.steps)
        max_res = float(np.max(eq.residual))
        free_residuals.append(max_res)
        latest_free_state = eq.state
        if max_res > tolerance:
            raise AssertionError(
                f"Equilibration residual {max_res:.6e} exceeded tolerance {tolerance}"
            )

    mean_free_ms = float(np.mean(free_latencies_ms))
    avg_steps = float(np.mean(free_steps))
    free_relaxation_report = {
        "cold_latency_ms": cold_free_ms,
        "p50_latency_ms": float(np.percentile(free_latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(free_latencies_ms, 95)),
        "relaxation_steps_per_sec": (
            avg_steps / (mean_free_ms / 1000.0) if mean_free_ms > 0 else 0.0
        ),
        "sample_steps_per_sec": (
            avg_steps * batch / (mean_free_ms / 1000.0) if mean_free_ms > 0 else 0.0
        ),
        "max_residual": float(np.max(free_residuals)),
        "qualified": True,
    }

    # --- Profile 2: Nudged Phase & Teaching Contrast ---
    target = learner.targets(learner._labels(labels))
    nudge = learner.nudge_for(target, learner.config.beta)
    nudged_eq = graph.equilibrate(
        drive, state=latest_free_state, nudge=nudge, tolerance=tolerance, budget=budget
    )
    sync()
    nudged_residual = float(np.max(nudged_eq.residual))
    if nudged_residual > tolerance:
        raise AssertionError(
            f"Nudged equilibration residual {nudged_residual:.6e} exceeded tolerance {tolerance}"
        )
    nudged_state = nudged_eq.state

    # Cold contrast
    sync()
    t_contrast_cold_start = time.perf_counter()
    learner.contrast(latest_free_state, nudged_state)
    sync()
    cold_contrast_ms = (time.perf_counter() - t_contrast_cold_start) * 1000.0

    contrast_latencies_ms: list[float] = []
    for _ in range(iterations):
        sync()
        t0 = time.perf_counter()
        learner.contrast(latest_free_state, nudged_state)
        sync()
        contrast_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_contrast_ms = float(np.mean(contrast_latencies_ms))
    contrast_report = {
        "cold_latency_ms": cold_contrast_ms,
        "p50_latency_ms": float(np.percentile(contrast_latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(contrast_latencies_ms, 95)),
        "synapses_per_sec": (
            conn.synapses / (mean_contrast_ms / 1000.0) if mean_contrast_ms > 0 else 0.0
        ),
        "nudged_residual": nudged_residual,
    }

    # --- Profile 3: Synaptic Plasticity (Learner.step) ---
    sync()
    t_step_cold_start = time.perf_counter()
    _, rep_cold = learner.step(drive, labels)
    sync()
    cold_step_ms = (time.perf_counter() - t_step_cold_start) * 1000.0

    step_latencies_ms: list[float] = []
    last_step_report: dict[str, float] = {}
    for _ in range(iterations):
        sync()
        t0 = time.perf_counter()
        _, rep = learner.step(drive, labels)
        sync()
        step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        last_step_report = rep

    mean_step_ms = float(np.mean(step_latencies_ms))
    total_parameters = conn.synapses + conn.n
    plasticity_report = {
        "cold_latency_ms": cold_step_ms,
        "p50_latency_ms": float(np.percentile(step_latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(step_latencies_ms, 95)),
        "mean_latency_ms": mean_step_ms,
        "min_latency_ms": float(np.min(step_latencies_ms)),
        "param_updates_per_sec": (
            total_parameters / (mean_step_ms / 1000.0) if mean_step_ms > 0 else 0.0
        ),
        "steps_per_sec": (1000.0 / mean_step_ms) if mean_step_ms > 0 else 0.0,
        "sample_updates_per_sec": (batch * 1000.0 / mean_step_ms) if mean_step_ms > 0 else 0.0,
        "scale_step": float(last_step_report.get("scale_step", 0.0)),
        "bias_step": float(last_step_report.get("bias_step", 0.0)),
        "device_moments": bool(
            hasattr(learner, "_device_moments")
            and learner.__dict__.get("_device_moments") is not None
        ),
    }

    # --- Profile 4: Memory & Resources ---
    sync()
    rss_peak_bytes = host_memory_bytes()
    memory_report: dict[str, Any] = {
        "host_rss_before_bytes": rss_before_bytes,
        "host_rss_peak_bytes": rss_peak_bytes,
        "host_rss_peak_mib": round(rss_peak_bytes / (1024 * 1024), 2),
    }
    if is_npu and torch_module is not None:
        alloc_bytes = torch_module.npu.memory_allocated(device)
        max_alloc_bytes = torch_module.npu.max_memory_allocated(device)
        res_bytes = torch_module.npu.memory_reserved(device)
        max_res_bytes = torch_module.npu.max_memory_reserved(device)
        memory_report.update({
            "npu_allocated_bytes": alloc_bytes,
            "npu_allocated_mib": round(alloc_bytes / (1024 * 1024), 3),
            "npu_max_allocated_bytes": max_alloc_bytes,
            "npu_max_allocated_mib": round(max_alloc_bytes / (1024 * 1024), 3),
            "npu_reserved_bytes": res_bytes,
            "npu_reserved_mib": round(res_bytes / (1024 * 1024), 3),
            "npu_max_reserved_bytes": max_res_bytes,
            "npu_max_reserved_mib": round(max_res_bytes / (1024 * 1024), 3),
        })

    return {
        "passed": True,
        "backend": backend,
        "device": device,
        "size": size,
        "batch": batch,
        "neurons": conn.n,
        "synapses": conn.synapses,
        "inputs": inputs,
        "outputs": outputs,
        "hidden": hidden,
        "iterations": iterations,
        "warmup": warmup,
        "seed": seed,
        "free_relaxation": free_relaxation_report,
        "teaching_contrast": contrast_report,
        "synaptic_plasticity": plasticity_report,
        "memory": memory_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["cpu", "torch_cpu", "npu", "numpy", "torch_npu"],
        default="cpu",
        help="Backend to profile ('cpu', 'torch_cpu', 'npu')",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device string (e.g. 'cpu', 'npu:0')"
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size (e.g. 1, 4, 16, 64)")
    parser.add_argument(
        "--size", type=int, default=64, help="Network size N (e.g. 64, 256, 1024, 2048)"
    )
    parser.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tolerance", type=float, default=0.003, help="Residual qualification tolerance"
    )
    parser.add_argument("--budget", type=int, default=128, help="Equilibration budget")
    parser.add_argument("--out", type=Path, default=None, help="Output file for result JSON")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    backend = args.backend
    if backend == "numpy":
        backend = "cpu"
    elif backend == "torch_npu":
        backend = "npu"

    print(
        f"Profiling backend={backend} device={args.device or 'default'} "
        f"size={args.size} batch={args.batch}..."
    )

    t0 = time.perf_counter()
    result = run_benchmark(
        backend=backend,
        device=args.device,
        size=args.size,
        batch=args.batch,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        tolerance=args.tolerance,
        budget=args.budget,
    )
    elapsed = time.perf_counter() - t0

    result["wall_seconds"] = elapsed
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cadence": cd.__version__,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Result written to {args.out}")

    print(f"Finished in {elapsed:.2f}s:")
    print(f"  Free relaxation p50: {result['free_relaxation']['p50_latency_ms']:.2f} ms")
    print(f"  Contrast p50:        {result['teaching_contrast']['p50_latency_ms']:.2f} ms")
    print(f"  Step p50:            {result['synaptic_plasticity']['p50_latency_ms']:.2f} ms")
    print(f"  Max residual:        {result['free_relaxation']['max_residual']:.6e}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
