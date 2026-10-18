"""Matched CPU (NumPy float64) vs NPU (torch_npu float32) benchmark campaign for System 1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sources(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / "src/cadence").rglob("*.py"))
    }


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def memory() -> dict[str, int]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(peak * (1 if sys.platform == "darwin" else 1024))}


def run_worker(args: argparse.Namespace) -> int:
    entered = time.perf_counter()
    root, output = args.source_root.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(protocol.get("threads", 1))
    os.environ["PYTHONUTF8"] = "1"

    source_before = sources(root)
    manifest = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "source_root": str(root),
        "source_sha256_lf": source_before,
        "harness_sha256_lf": digest(Path(__file__)),
        "protocol_sha256_lf": digest(args.protocol),
        "backend": args.backend,
        "seed": args.seed,
        "batch": args.batch,
        "protocol": protocol,
    }
    write(output / "manifest.json", manifest)

    sys.path.insert(0, str(root / "src"))
    result: dict[str, Any] = {"passed": False, "measurements": [], "warnings": []}
    event_file = (output / "events.jsonl").open("w", encoding="utf-8")

    npu = args.backend == "npu"
    torch: Any = None
    torch_npu: Any = None

    try:
        from dataclasses import replace

        import numpy as np

        import cadence as cd

        assert Path(cd.__file__).resolve().is_relative_to(root)

        if npu:
            import torch as _torch
            import torch_npu as _torch_npu

            torch, torch_npu = _torch, _torch_npu
            if not torch_npu.npu.is_available():
                raise RuntimeError("Actual Ascend NPU required; no fallback or skipped receipt")
            torch.set_num_threads(protocol.get("threads", 1))
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()

        def sync() -> None:
            if npu and torch is not None:
                torch.npu.synchronize()

        def plain(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return plain(value.tolist())
            if isinstance(value, np.generic):
                return plain(value.item())
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        work = {
            "settling_calls": 0,
            "failed_settling_calls": 0,
            "sweeps": 0,
            "row_sweeps": 0,
            "residual_calls": 0,
        }
        settle, residual = cd.NeuralGraph.settle_batch, cd.NeuralGraph.residual

        def counted_settle(graph: Any, *a: Any, **kw: Any) -> Any:
            work["settling_calls"] += 1
            try:
                phase = settle(graph, *a, **kw)
            except Exception:
                work["failed_settling_calls"] += 1
                raise
            work["sweeps"] += phase.steps
            work["row_sweeps"] += phase.steps * len(np.atleast_2d(a[0]))
            return phase

        def counted_residual(graph: Any, *a: Any, **kw: Any) -> Any:
            work["residual_calls"] += 1
            return residual(graph, *a, **kw)

        cd.NeuralGraph.settle_batch = counted_settle  # type: ignore[assignment]
        cd.NeuralGraph.residual = counted_residual  # type: ignore[assignment]

        def record(name: str, operation: Any, decisions: int = 0) -> Any:
            before = work.copy()
            sync()
            started = time.perf_counter()
            failure = None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    value = operation()
                except Exception:
                    failure = traceback.format_exc()
                    raise
                finally:
                    sync()
                    row = {
                        "stage": name,
                        "seconds": time.perf_counter() - started,
                        "decisions": decisions,
                        "work": {k: work[k] - before[k] for k in work},
                        "failure": failure,
                    }
                    result["warnings"].extend(str(w.message) for w in caught)
                    result["measurements"].append(row)
                    event_file.write(json.dumps(row, allow_nan=False) + "\n")
                    event_file.flush()
            return value

        packages = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "numba", "scipy")
            if name in sys.modules or importlib.util.find_spec(name) is not None
        }
        if npu:
            packages["torch"] = torch.__version__
            packages["torch_npu"] = torch_npu.__version__

        result["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "cadence": cd.__version__,
            "numpy": np.__version__,
            "threads": protocol.get("threads", 1),
            "packages": packages,
        }
        if npu:
            props = torch.npu.get_device_properties(0)
            result["environment"].update(
                {
                    "device": props.name,
                    "device_index": 0,
                    "total_memory_bytes": props.total_memory,
                    "torch_npu_version": torch_npu.__version__,
                }
            )

        result["imports_and_device_init_seconds"] = time.perf_counter() - entered
        result["memory_after_imports"] = memory()

        def create() -> Any:
            brain = cd.Brain.compose(
                protocol["inputs"],
                protocol["actions"],
                modules=protocol["modules"],
                observers=protocol["observers"],
                seed=args.seed,
                backend="torch" if npu else "cpu",
                device="npu:0" if npu else None,
                precision="float32" if npu else None,
            )
            brain.learner.config = replace(brain.learner.config, **protocol["learning"])
            brain.basal_ganglia.config = replace(brain.basal_ganglia.config, **protocol["reward"])
            return brain

        brain = record("construction", create)
        result["execution"] = {
            "backend": brain.brain.backend,
            "device": "cpu",
            "state_dtype": "float64",
            "parameter_dtype": "float64",
        }
        if npu:
            kernel = brain.brain._torch
            assert kernel is not None
            assert kernel.device.type == "npu"
            assert kernel.dtype == torch.float32
            result["execution"].update(
                {
                    "device": str(kernel.device),
                    "state_dtype": str(kernel.dtype),
                    "parameter_dtype": str(kernel.param_dtype),
                }
            )
        result["graph"] = {
            "neurons": brain.connectome.n,
            "edges": brain.connectome.synapses,
            "learning": brain.learner.config.to_dict(),
            "reward": brain.basal_ganglia.config.to_dict(),
        }

        def samples(
            seed: int, count: int = protocol["training_rows"]
        ) -> tuple[np.ndarray, np.ndarray]:
            rng = np.random.default_rng(seed)
            labels = np.repeat([0, 1], count // 2)
            x = np.zeros((count, 8))
            x[labels == 0, :4] = 1
            x[labels == 1, 4:] = 1
            return np.clip(x + protocol["noise"] * rng.standard_normal(x.shape), 0, 1), labels

        training, labels = samples(protocol["training_data_seed"])
        held, expected = samples(protocol["held_out_data_seed"])
        result["founder_accuracy"] = record(
            "cold_free_query", lambda: brain.accuracy(held, expected)
        )
        result["fit_accuracy"] = record(
            "bootstrap",
            lambda: brain.fit(
                training, labels, epochs=protocol["epochs"], batch=protocol["training_batch"]
            ),
        )
        result["bootstrap_accuracy"] = record(
            "bootstrap_held_out", lambda: brain.accuracy(held, expected)
        )
        brain.save(output / "acquired.npz")

        rng = np.random.default_rng(700 + args.seed)
        last_action: np.ndarray | None = None
        last_labels: np.ndarray | None = None
        live_rows = []
        batch_size = args.batch
        current = None

        for stage in protocol["stages"]:
            for tick in range(stage["calls"]):

                def decision(stage: dict[str, Any] = stage, tick: int = tick) -> np.ndarray:
                    nonlocal last_action, last_labels
                    y = rng.integers(0, 2, batch_size)
                    x = np.zeros((len(y), 8))
                    x[y == 0, :4] = stage["amplitude"]
                    x[y == 1, 4:] = stage["amplitude"]
                    x = np.clip(x + protocol["noise"] * rng.standard_normal(x.shape), 0, 1)
                    feedback: dict[str, Any] = (
                        {}
                        if last_action is None or last_labels is None
                        else {
                            "reward": np.asarray(last_action == last_labels, dtype=float),
                            "done": np.zeros(len(y), dtype=bool),
                        }
                    )
                    action = brain.step(x, teacher=y if stage["teach"] else None, **feedback)
                    row = {
                        "stage": stage["name"],
                        "tick": tick,
                        "correct": int((action == y).sum()),
                        "labels": y.tolist(),
                        "actions": action.tolist(),
                        "settlement": dict(brain.last_settlement),
                        "learning": dict(brain.last_learning),
                    }
                    live_rows.append(plain(row))
                    last_action, last_labels = action, y
                    return np.asarray(x)

                current = record(stage["name"], decision, batch_size)

        result["live"] = live_rows
        saved = record("checkpoint_save", lambda: brain.save(output / "pending.npz"))
        loaded = record(
            "checkpoint_restore",
            lambda: cd.Brain.load(
                saved,
                backend="torch" if npu else "cpu",
                device="npu:0" if npu else None,
                precision="float32" if npu else None,
            ),
        )
        assert last_action is not None and last_labels is not None
        feedback = {
            "reward": np.asarray(last_action == last_labels, dtype=float),
            "done": np.zeros(batch_size, dtype=bool),
        }
        assert current is not None
        first = record(
            "saved_continuation",
            lambda: brain.step(current * 0.9, **feedback),
            batch_size,
        )
        second = record(
            "restored_continuation",
            lambda: loaded.step(current * 0.9, **feedback),
            batch_size,
        )
        result["saved_actions_equal"] = bool(np.array_equal(first, second))
        result["retained_accuracy"] = record(
            "retention_query", lambda: brain.accuracy(held, expected)
        )
        result["memory"] = memory()
        result["npu_memory"] = (
            None
            if not npu
            else {
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                "allocated_bytes": torch.npu.memory_allocated(),
                "reserved_bytes": torch.npu.memory_reserved(),
            }
        )
        result["counts"] = {
            "graph_updates": brain.learner.updates,
            "actor_updates": brain.basal_ganglia.updates,
            "memory_writes": brain.hippocampus.writes,
            "work": work,
        }
        groups = {}
        for stage in protocol["stages"]:
            rows = [r for r in result["measurements"] if r["stage"] == stage["name"]]
            values = np.array([r["seconds"] for r in rows])
            groups[stage["name"]] = {
                "calls": len(rows),
                "seconds": float(values.sum()),
                "p50_seconds": float(np.percentile(values, 50)),
                "p95_seconds": float(np.percentile(values, 95)),
                "p99_seconds": float(np.percentile(values, 99)),
                "max_seconds": float(values.max()),
                "decisions_per_second": sum(r["decisions"] for r in rows) / float(values.sum()),
                "deadline_misses": int((values > protocol["latency"]["deadline_seconds"]).sum()),
            }
        result["latencies"] = groups
        result["passed"] = bool(
            result["bootstrap_accuracy"] >= protocol["gate"]["bootstrap_held_out_accuracy"]
            and result["bootstrap_accuracy"] > result["founder_accuracy"]
            and result["retained_accuracy"] >= protocol["gate"]["retained_held_out_accuracy"]
            and result["saved_actions_equal"]
            and all(row["settlement"]["qualified"] for row in live_rows)
        )
    except Exception:
        result["failure"] = traceback.format_exc()
    finally:
        result["memory"] = memory()
        if npu and torch is not None and torch_npu.npu.is_available():
            result["npu_memory"] = {
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                "allocated_bytes": torch.npu.memory_allocated(),
                "reserved_bytes": torch.npu.memory_reserved(),
            }
        result["source_unchanged"] = sources(root) == source_before
        result["passed"] = result["passed"] and result["source_unchanged"]
        result["worker_seconds"] = time.perf_counter() - entered
        event_file.close()
        write(output / "result.json", result)

    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "passed",
                    "bootstrap_accuracy",
                    "retained_accuracy",
                    "worker_seconds",
                    "failure",
                )
            }
        )
    )
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Run one isolated worker job")
    parser.add_argument("--backend", choices=["cpu", "npu"], help="Backend to execute")
    parser.add_argument("--batch", type=int, help="Live batch size (1 or 16)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    default_out = Path(__file__).resolve().parent / "results/campaign"
    default_proto = Path(__file__).resolve().parent / "protocol.json"
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--protocol", type=Path, default=default_proto)
    args = parser.parse_args()

    if args.worker:
        if args.backend is None or args.batch is None or args.seed is None:
            parser.error("--worker requires --backend, --batch, and --seed")
        return run_worker(args)

    here = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = args.source_root.resolve()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)

    jobs = []
    for batch in protocol["batches"]:
        for seed in protocol["seeds"]:
            order = ["cpu", "npu"] if seed % 2 == 0 else ["npu", "cpu"]
            for backend in order:
                jobs.append({
                    "kind": "nursery",
                    "seed": seed,
                    "backend": backend,
                    "batch": batch,
                })

    frozen = {
        "commit": commit,
        "dirty": dirty,
        "protocol": protocol,
        "jobs": jobs,
        "harness_sha256_lf": {p.name: digest(p) for p in sorted(here.glob("*.py"))},
        "protocol_sha256_lf": digest(protocol_path),
    }
    write(args.out / "campaign.json", frozen)

    outcomes = []
    environment = dict(os.environ, PYTHONUTF8="1")
    if Path("/usr/local/Ascend/ascend-toolkit/latest").is_dir():
        environment.setdefault("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        environment.setdefault("TBE_PARALLEL_COMPILER", "2")
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("OPENBLAS_NUM_THREADS", "2")

    print(f"=== Starting System 1 NPU Campaign: {len(jobs)} jobs ===")
    for index, job in enumerate(jobs):
        kind, seed, backend, batch = job["kind"], job["seed"], job["backend"], job["batch"]
        key = f"{index:02d}-{kind}-{backend}-batch{batch}-seed{seed}"
        folder = args.out / key
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(here / "campaign.py"),
            "--worker",
            "--source-root",
            str(root),
            "--out",
            str(folder),
            "--backend",
            backend,
            "--batch",
            str(batch),
            "--seed",
            str(seed),
            "--protocol",
            str(protocol_path),
        ]
        print(f"[{index + 1}/{len(jobs)}] {key}...", flush=True)
        started = time.perf_counter()
        log_path = args.out / f"{key}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
            )
        elapsed = time.perf_counter() - started
        row = {
            "directory": key,
            "kind": kind,
            "seed": seed,
            "backend": backend,
            "batch": batch,
            "exit_code": process.returncode,
            "process_seconds": elapsed,
            "command": cmd,
        }
        outcomes.append(row)
        write(args.out / "outcomes.json", outcomes)
        print(f"  exit={process.returncode}, wall={elapsed:.2f}s", flush=True)

    unchanged = (
        frozen["harness_sha256_lf"] == {p.name: digest(p) for p in sorted(here.glob("*.py"))}
        and frozen["protocol_sha256_lf"] == digest(protocol_path)
        and commit
        == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )
    write(args.out / "completion.json", {"harness_and_revisions_unchanged": unchanged})
    failed = not unchanged or any(r["exit_code"] != 0 for r in outcomes)
    print(f"=== Campaign finished: {'SUCCESS' if not failed else 'FAILED'} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
