"""Automated multi-scale performance baseline campaign for CPU and NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from benchmarks.performance_baseline.harness import run_benchmark
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.performance_baseline.harness import run_benchmark

def digest(path: Path) -> str:
    """Compute sha256 digest with normalized line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sources(root: Path) -> dict[str, str]:
    """Compute sha256 digests for all Cadence library source files."""
    return {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / "src/cadence").rglob("*.py"))
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_worker_backend(args: argparse.Namespace) -> int:
    """Worker process executing all sizes and batch regimes for one backend."""
    out_dir = args.out.resolve()
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    backend = args.backend
    device = args.device
    sizes = args.sizes or protocol.get("sizes", [64, 256, 1024, 2048])
    batches = args.batches or protocol.get("batches", [1, 4, 16, 64])
    iterations = args.iterations or protocol.get("iterations", 5)
    warmup = args.warmup or protocol.get("warmup", 1)
    seed = args.seed or protocol.get("seed", 42)
    tolerance = protocol.get("tolerance", 0.003)
    budget = protocol.get("budget", 128)

    print(
        f"=== Worker started: backend={backend} device={device or 'default'} "
        f"sizes={sizes} batches={batches} ==="
    )

    success = True
    total_configs = len(sizes) * len(batches)
    count = 0

    for size in sizes:
        for batch in batches:
            count += 1
            key = f"{backend}_s{size}_b{batch}"
            run_file = runs_dir / f"{key}.json"
            print(f"[{count}/{total_configs}] Running {key}...", flush=True)

            t0 = time.perf_counter()
            try:
                res = run_benchmark(
                    backend=backend,
                    device=device,
                    size=size,
                    batch=batch,
                    iterations=iterations,
                    warmup=warmup,
                    seed=seed,
                    tolerance=tolerance,
                    budget=budget,
                )
                res["wall_seconds"] = time.perf_counter() - t0
                res["run_key"] = key
                write_json(run_file, res)
                p50 = res["free_relaxation"]["p50_latency_ms"]
                res_max = res["free_relaxation"]["max_residual"]
                print(
                    f"  OK: p50={p50:.2f}ms, max_residual={res_max:.6e}, "
                    f"wall={res['wall_seconds']:.2f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr, flush=True)
                write_json(
                    run_file,
                    {
                        "passed": False,
                        "backend": backend,
                        "device": device,
                        "size": size,
                        "batch": batch,
                        "run_key": key,
                        "error": str(e),
                    },
                )
                success = False

    return 0 if success else 1


def orchestrate(args: argparse.Namespace) -> int:
    """Orchestrate campaign across all backends."""
    root = args.source_root.resolve()
    here = Path(__file__).resolve().parent
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )

    backends = args.backends or protocol.get("backends", ["cpu", "torch_cpu", "npu"])

    manifest = {
        "commit": commit,
        "dirty": dirty,
        "source_root": str(root),
        "source_sha256_lf": sources(root),
        "harness_sha256_lf": {p.name: digest(p) for p in sorted(here.glob("*.py"))},
        "protocol_sha256_lf": digest(protocol_path),
        "protocol": protocol,
        "backends": backends,
        "platform": platform.platform(),
        "python": sys.version,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "campaign.json", manifest)

    environment = dict(os.environ, PYTHONUTF8="1")
    if Path("/usr/local/Ascend/ascend-toolkit/latest").is_dir():
        environment.setdefault("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        environment.setdefault("TBE_PARALLEL_COMPILER", "2")
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("OPENBLAS_NUM_THREADS", "2")

    print(f"=== Starting Multi-Scale Performance Campaign: backends={backends} ===")
    outcomes: list[dict[str, Any]] = []

    for index, backend in enumerate(backends):
        log_path = out_dir / f"worker_{backend}.log"
        print(f"[{index + 1}/{len(backends)}] Running worker backend={backend}...", flush=True)

        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(here / "campaign.py"),
            "--worker",
            "--backend",
            backend,
            "--source-root",
            str(root),
            "--out",
            str(out_dir),
            "--protocol",
            str(protocol_path),
        ]
        if args.sizes:
            cmd.extend(["--sizes"] + [str(s) for s in args.sizes])
        if args.batches:
            cmd.extend(["--batches"] + [str(b) for b in args.batches])

        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
            )
        elapsed = time.perf_counter() - started

        outcome = {
            "backend": backend,
            "exit_code": proc.returncode,
            "seconds": elapsed,
            "log": str(log_path.relative_to(out_dir)),
            "command": cmd,
        }
        outcomes.append(outcome)
        write_json(out_dir / "outcomes.json", outcomes)
        print(f"  Worker {backend} exit={proc.returncode} in {elapsed:.2f}s", flush=True)

    failed = any(o["exit_code"] != 0 for o in outcomes)
    write_json(out_dir / "completion.json", {"completed": not failed, "outcomes": outcomes})

    # Run verification automatically
    verify_script = here / "verify.py"
    if verify_script.is_file() and not failed:
        print("=== Running verifier ===", flush=True)
        v_cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(verify_script),
            "--campaign-dir",
            str(out_dir),
            "--source-root",
            str(root),
        ]
        v_res = subprocess.run(v_cmd, check=False)
        if v_res.returncode != 0:
            print("Verifier failed!", file=sys.stderr)
            return 1

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Run one isolated worker backend")
    parser.add_argument("--backend", choices=["cpu", "torch_cpu", "npu"], help="Backend to execute")
    parser.add_argument("--device", type=str, default=None, help="Device string")
    parser.add_argument("--backends", nargs="+", help="Backends list for orchestration")
    parser.add_argument("--sizes", type=int, nargs="+", help="Sizes list N")
    parser.add_argument("--batches", type=int, nargs="+", help="Batches list")
    parser.add_argument("--iterations", type=int, default=None, help="Iterations count")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup count")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parent / "protocol.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    if args.worker:
        if not args.backend:
            parser.error("--worker requires --backend")
        return run_worker_backend(args)

    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
