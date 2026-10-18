"""Verify performance baseline results and generate compressed verifiable receipt."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Compute sha256 digest with normalized line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sources(root: Path) -> dict[str, str]:
    """Compute sha256 digests for all Cadence library source files."""
    return {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / "src/cadence").rglob("*.py"))
    }


def verify_campaign(campaign_dir: Path, root: Path) -> dict[str, Any]:
    campaign_file = campaign_dir / "campaign.json"
    manifest_file = campaign_dir / "manifest.json"
    outcomes_file = campaign_dir / "outcomes.json"
    runs_dir = campaign_dir / "runs"

    if not campaign_file.is_file():
        raise FileNotFoundError(f"Missing {campaign_file}")
    if not outcomes_file.is_file():
        raise FileNotFoundError(f"Missing {outcomes_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    outcomes = json.loads(outcomes_file.read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # Check 1: All backend workers succeeded
    worker_success = all(o.get("exit_code") == 0 for o in outcomes)
    check("backend_workers_succeeded", worker_success, f"{len(outcomes)} workers")

    # Check 2: Load run files
    backends = manifest.get("backends", protocol.get("backends", ["cpu", "torch_cpu", "npu"]))
    sizes = protocol.get("sizes", [64, 256, 1024, 2048])
    batches = protocol.get("batches", [1, 4, 16, 64])
    tolerance = protocol.get("tolerance", 0.003)

    records: list[dict[str, Any]] = []
    runs_by_key: dict[str, dict[str, Any]] = {}

    for backend in backends:
        for size in sizes:
            for batch in batches:
                key = f"{backend}_s{size}_b{batch}"
                run_path = runs_dir / f"{key}.json"
                if not run_path.is_file():
                    check(f"run_exists_{key}", False, "File missing")
                    continue
                data = json.loads(run_path.read_text(encoding="utf-8"))
                runs_by_key[key] = data
                records.append(data)

                # Check passed flag
                check(f"run_passed_{key}", bool(data.get("passed", False)))

                # Check equation residual <= tolerance
                max_res = data.get("free_relaxation", {}).get("max_residual", 1.0)
                res_ok = max_res <= tolerance
                check(
                    f"residual_qualified_{key}",
                    res_ok,
                    f"max_res={max_res:.6e} <= {tolerance}",
                )

                # Check valid latencies
                free_p50 = data.get("free_relaxation", {}).get("p50_latency_ms", 0.0)
                contrast_p50 = data.get("teaching_contrast", {}).get("p50_latency_ms", 0.0)
                step_p50 = data.get("synaptic_plasticity", {}).get("p50_latency_ms", 0.0)
                check(
                    f"latencies_valid_{key}",
                    free_p50 > 0 and contrast_p50 > 0 and step_p50 > 0,
                    f"free={free_p50:.2f}ms, contrast={contrast_p50:.2f}ms, step={step_p50:.2f}ms",
                )

    expected_total = len(backends) * len(sizes) * len(batches)
    check(
        "matrix_coverage_complete",
        len(records) == expected_total,
        f"{len(records)}/{expected_total}",
    )
    # Build comparative summaries and crossover analysis
    comparison: list[dict[str, Any]] = []
    crossover_points: list[dict[str, Any]] = []

    for size in sizes:
        for batch in batches:
            cpu_key = f"cpu_s{size}_b{batch}"
            torch_cpu_key = f"torch_cpu_s{size}_b{batch}"
            npu_key = f"npu_s{size}_b{batch}"

            cpu_run = runs_by_key.get(cpu_key)
            torch_cpu_run = runs_by_key.get(torch_cpu_key)
            npu_run = runs_by_key.get(npu_key)

            if cpu_run and npu_run:
                cpu_free_p50 = cpu_run["free_relaxation"]["p50_latency_ms"]
                npu_free_p50 = npu_run["free_relaxation"]["p50_latency_ms"]
                cpu_step_p50 = cpu_run["synaptic_plasticity"]["p50_latency_ms"]
                npu_step_p50 = npu_run["synaptic_plasticity"]["p50_latency_ms"]

                free_speedup = (cpu_free_p50 / npu_free_p50) if npu_free_p50 > 0 else 0.0
                step_speedup = (cpu_step_p50 / npu_step_p50) if npu_step_p50 > 0 else 0.0

                item = {
                    "size": size,
                    "batch": batch,
                    "cpu_free_p50_ms": cpu_free_p50,
                    "torch_cpu_free_p50_ms": (
                        torch_cpu_run["free_relaxation"]["p50_latency_ms"]
                        if torch_cpu_run
                        else None
                    ),
                    "npu_free_p50_ms": npu_free_p50,
                    "free_speedup_npu_vs_cpu": round(free_speedup, 2),
                    "cpu_step_p50_ms": cpu_step_p50,
                    "torch_cpu_step_p50_ms": (
                        torch_cpu_run["synaptic_plasticity"]["p50_latency_ms"]
                        if torch_cpu_run
                        else None
                    ),
                    "npu_step_p50_ms": npu_step_p50,
                    "step_speedup_npu_vs_cpu": round(step_speedup, 2),
                    "cpu_contrast_p50_ms": cpu_run["teaching_contrast"]["p50_latency_ms"],
                    "npu_contrast_p50_ms": npu_run["teaching_contrast"]["p50_latency_ms"],
                    "npu_free_relaxation_steps_per_sec": (
                        npu_run["free_relaxation"]["relaxation_steps_per_sec"]
                    ),
                    "cpu_free_relaxation_steps_per_sec": (
                        cpu_run["free_relaxation"]["relaxation_steps_per_sec"]
                    ),
                    "host_rss_peak_mib_cpu": cpu_run["memory"]["host_rss_peak_mib"],
                    "host_rss_peak_mib_npu": npu_run["memory"]["host_rss_peak_mib"],
                    "npu_allocated_mib": npu_run["memory"].get("npu_allocated_mib", 0.0),
                    "npu_reserved_mib": npu_run["memory"].get("npu_reserved_mib", 0.0),
                }
                comparison.append(item)
                if free_speedup >= 1.0 or step_speedup >= 1.0:
                    crossover_points.append({
                        "size": size,
                        "batch": batch,
                        "free_speedup": round(free_speedup, 2),
                        "step_speedup": round(step_speedup, 2),
                    })

    passed = all(c["passed"] for c in checks)

    receipt = {
        "format": "cadence-performance-baseline/1",
        "passed": passed,
        "checks_count": len(checks),
        "failed_checks": [c for c in checks if not c["passed"]],
        "checks": checks,
        "manifest": manifest,
        "protocol": protocol,
        "records": records,
        "comparison": comparison,
        "crossover_points": crossover_points,
    }

    # Write compressed receipt
    receipt_path = campaign_dir / "baseline-comparison-receipt.json.gz"
    with gzip.open(receipt_path, "wt", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    # Also write plain json summary
    summary_path = campaign_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "checks_count": len(checks),
                "records_count": len(records),
                "comparison": comparison,
                "crossover_points": crossover_points,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Verification complete: {'PASSED' if passed else 'FAILED'}")
    print(f"Total checks: {len(checks)}, failed: {len(receipt['failed_checks'])}")
    print(f"Receipt written to {receipt_path}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Campaign directory containing runs and manifests",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Cadence source repository root",
    )
    args = parser.parse_args()

    receipt = verify_campaign(args.campaign_dir.resolve(), args.source_root.resolve())
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
