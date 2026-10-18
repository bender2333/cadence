"""Tests verifying the performance baseline benchmark harness and published receipt."""

import gzip
import json
from pathlib import Path

from benchmarks.performance_baseline.harness import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_performance_baseline_harness_cpu() -> None:
    res = run_benchmark(
        backend="cpu",
        size=64,
        batch=1,
        iterations=2,
        warmup=1,
        seed=42,
        tolerance=0.003,
        budget=128,
    )
    assert res["passed"] is True
    assert res["neurons"] == 64
    assert res["free_relaxation"]["max_residual"] <= 0.003
    assert res["free_relaxation"]["p50_latency_ms"] > 0
    assert res["teaching_contrast"]["p50_latency_ms"] > 0
    assert res["synaptic_plasticity"]["p50_latency_ms"] > 0
    assert res["memory"]["host_rss_peak_mib"] > 0


def test_performance_baseline_harness_npu_smoke() -> None:
    try:
        import torch
        import torch_npu  # noqa: F401

        npu_available = hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        npu_available = False

    if not npu_available:
        return

    res = run_benchmark(
        backend="npu",
        device="npu:0",
        size=64,
        batch=1,
        iterations=1,
        warmup=1,
        seed=42,
        tolerance=0.003,
        budget=128,
    )
    assert res["passed"] is True
    assert res["neurons"] == 64
    assert res["free_relaxation"]["max_residual"] <= 0.003
    assert res["memory"]["npu_allocated_bytes"] >= 0


def test_published_performance_baseline_receipt() -> None:
    receipt_path = ROOT / "benchmarks/performance_baseline/results/baseline-comparison-receipt.json.gz"
    assert receipt_path.is_file(), "baseline-comparison-receipt.json.gz must exist"

    with gzip.open(receipt_path, "rt", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["passed"] is True
    assert receipt["format"] == "cadence-performance-baseline/1"
    assert len(receipt["records"]) == 48, "Receipt must contain 48 configurations (3 backends x 4 sizes x 4 batches)"
    assert all(c["passed"] for c in receipt["checks"])

    for record in receipt["records"]:
        assert record["passed"] is True
        assert record["free_relaxation"]["max_residual"] <= 0.003
        assert record["free_relaxation"]["p50_latency_ms"] > 0
        assert record["teaching_contrast"]["p50_latency_ms"] > 0
        assert record["synaptic_plasticity"]["p50_latency_ms"] > 0
