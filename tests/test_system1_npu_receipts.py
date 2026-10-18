"""Verify the published Ascend NPU System 1 benchmark receipt and checks."""

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_system1_npu_published_receipt_passed() -> None:
    receipt_path = ROOT / "benchmarks/system1_npu/results/system1-npu-receipt.json.gz"
    assert receipt_path.is_file(), "system1-npu-receipt.json.gz must exist"

    with gzip.open(receipt_path, "rt", encoding="utf-8") as stream:
        receipt = json.load(stream)

    assert receipt["passed"] is True, "Published receipt must have passed all verifier checks"
    assert len(receipt["records"]) == 12, "Receipt must contain exactly 12 jobs (3 seeds x 2 backends x 2 batches)"
    assert len(receipt["checks"]) > 0, "Receipt must contain verifier checks"
    assert all(c["passed"] for c in receipt["checks"]), "All verification checks must pass"

    for record in receipt["records"]:
        res = record["result"]
        assert res["passed"] is True
        assert res["bootstrap_accuracy"] >= 0.95
        assert res["retained_accuracy"] >= 0.90
        assert res["saved_actions_equal"] is True
        assert all(row["settlement"]["qualified"] for row in res["live"])
