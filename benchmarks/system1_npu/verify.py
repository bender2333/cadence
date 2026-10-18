"""Independently verify a System 1 NPU campaign and publish its source-bound receipt."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from cadence import ActorCriticConfig, LearnerConfig

FOUNDERS = {
    "learning": {f.name: f.default for f in fields(LearnerConfig)},
    "reward": {f.name: f.default for f in fields(ActorCriticConfig)},
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(folder: Path) -> dict[str, Any]:
    campaign = read(folder / "campaign.json")
    outcomes = read(folder / "outcomes.json")
    completion = read(folder / "completion.json")
    protocol = campaign["protocol"]

    checks: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "passed": bool(passed)})

    check(
        "ordered job inventory complete",
        [[job[k] for k in ("kind", "seed", "backend", "batch")] for job in outcomes]
        == [[job[k] for k in ("kind", "seed", "backend", "batch")] for job in campaign["jobs"]],
    )
    check("unique job directories", len({job["directory"] for job in outcomes}) == len(outcomes))
    check("harness remained frozen", completion["harness_and_revisions_unchanged"])

    sources: dict[str, Any] = {}

    for job in outcomes:
        directory = folder / job["directory"]
        manifest = read(directory / "manifest.json")
        result = read(directory / "result.json")
        identity = job["directory"]

        check(identity + " exit", job["exit_code"] == 0)
        check(identity + " completed", result["passed"] and result["source_unchanged"])
        check(identity + " commit", manifest["commit"] == campaign["commit"])
        check(identity + " backend identity", manifest["backend"] == job["backend"])
        check(identity + " batch size", manifest["batch"] == job["batch"])
        check(identity + " seed", manifest["seed"] == job["seed"])
        check(
            identity + " protocol",
            manifest["protocol_sha256_lf"] == campaign["protocol_sha256_lf"],
        )
        check(
            identity + " harness",
            manifest["harness_sha256_lf"] == campaign["harness_sha256_lf"]["campaign.py"],
        )

        sources[identity] = manifest["source_sha256_lf"]
        batch = job["batch"]
        npu = job["backend"] == "npu"

        execution = result["execution"]
        expected_execution = {
            "backend": "torch" if npu else "cpu",
            "device": "npu:0" if npu else "cpu",
            "state_dtype": "torch.float32" if npu else "float64",
            "parameter_dtype": "torch.float32" if npu else "float64",
        }
        check(identity + " actual execution", execution == expected_execution)

        with np.load(directory / "acquired.npz", allow_pickle=False) as saved:
            meta = json.loads(str(saved["meta"]))
            generic = json.loads(str(saved["generic"]))

        for name, effective in (("learning", meta["config"]), ("reward", generic["reward"])):
            recorded = result["graph"][name]
            founders = FOUNDERS[name]
            check(
                identity + " saved " + name,
                {k: effective.get(k) for k in recorded} == recorded
                and all(
                    k in founders and founders[k] == value
                    for k, value in effective.items()
                    if k not in recorded
                ),
            )
            check(
                identity + " requested " + name,
                all(effective.get(k) == value for k, value in protocol[name].items()),
            )

        check(
            identity + " acquisition",
            result["bootstrap_accuracy"] >= protocol["gate"]["bootstrap_held_out_accuracy"]
            and result["bootstrap_accuracy"] > result["founder_accuracy"],
        )
        check(
            identity + " retention",
            result["retained_accuracy"] >= protocol["gate"]["retained_held_out_accuracy"],
        )
        check(identity + " saved continuation", result["saved_actions_equal"])

        check(
            identity + " live inventory",
            [(r["stage"], r["tick"]) for r in result["live"]]
            == [(s["name"], tick) for s in protocol["stages"] for tick in range(s["calls"])],
        )

        check(
            identity + " free admission",
            all(
                r["settlement"]["qualified"]
                and r["settlement"]["row_qualified"] == [True] * batch
                and len(r["settlement"]["residual"]) == batch
                and all(
                    np.isfinite(value) and 0 <= value <= protocol["learning"]["tolerance"]
                    for value in r["settlement"]["residual"]
                )
                and np.isclose(
                    r["settlement"]["max_residual"], max(r["settlement"]["residual"]), atol=1e-7
                )
                and r["settlement"]["tolerance"] == protocol["learning"]["tolerance"]
                and r["settlement"]["budget"] == protocol["learning"]["free_steps"]
                for r in result["live"]
            ),
        )

        check(
            identity + " actual live scores",
            all(
                len(r["actions"]) == len(r["labels"]) == batch
                and all(0 <= a < protocol["actions"] for a in r["actions"] + r["labels"])
                and r["correct"]
                == sum(a == b for a, b in zip(r["actions"], r["labels"], strict=True))
                for r in result["live"]
            ),
        )

        check(
            identity + " no failed calls",
            all(r["failure"] is None for r in result["measurements"]),
        )

        check(
            identity + " latency stages",
            set(result["latencies"]) == {s["name"] for s in protocol["stages"]},
        )

        for stage in protocol["stages"]:
            name = stage["name"]
            group = result["latencies"].get(name)
            rows = [r for r in result["measurements"] if r["stage"] == name]
            durations = [r["seconds"] for r in rows]
            check(
                identity + " " + name + " complete timed calls",
                len(rows) == stage["calls"]
                and all(np.isfinite(d) and d > 0 for d in durations)
                and all(r["decisions"] == batch for r in rows),
            )
            if group is not None and durations:
                check(
                    identity + " " + name + " totals and deadline misses",
                    group["calls"] == stage["calls"]
                    and np.isclose(group["seconds"], sum(durations), rtol=1e-12, atol=1e-12)
                    and group["max_seconds"] == max(durations)
                    and group["deadline_misses"]
                    == sum(d > protocol["latency"]["deadline_seconds"] for d in durations),
                )
                for p in (50, 95, 99):
                    check(
                        identity + f" {name} p{p}",
                        np.isclose(
                            group[f"p{p}_seconds"],
                            np.percentile(durations, p),
                            rtol=1e-12,
                            atol=1e-12,
                        ),
                    )
                check(
                    identity + " " + name + " throughput",
                    np.isclose(
                        group["decisions_per_second"],
                        sum(r["decisions"] for r in rows) / sum(durations),
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                )

        record = {k: v for k, v in job.items() if k != "command"}
        record["result"] = result
        records.append(record)

    # Matched arm consistency
    for batch in protocol["batches"]:
        for seed in protocol["seeds"]:
            pair = [r for r in records if r["batch"] == batch and r["seed"] == seed]
            label = f"batch{batch}_seed{seed}"
            has_both = len(pair) == 2 and {r["backend"] for r in pair} == {"cpu", "npu"}
            check(f"{label} cpu and npu present", has_both)
            if has_both:
                cpu_rec = next(r for r in pair if r["backend"] == "cpu")
                npu_rec = next(r for r in pair if r["backend"] == "npu")
                gate = protocol["gate"]
                boot_gate = gate["bootstrap_held_out_accuracy"]
                ret_gate = gate["retained_held_out_accuracy"]
                check(
                    f"{label} both acquired",
                    cpu_rec["result"]["bootstrap_accuracy"] >= boot_gate
                    and npu_rec["result"]["bootstrap_accuracy"] >= boot_gate,
                )
                check(
                    f"{label} both retained",
                    cpu_rec["result"]["retained_accuracy"] >= ret_gate
                    and npu_rec["result"]["retained_accuracy"] >= ret_gate,
                )
                check(
                    f"{label} continuation valid",
                    cpu_rec["result"]["saved_actions_equal"]
                    and npu_rec["result"]["saved_actions_equal"],
                )

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "campaign": campaign,
        "source_sha256_lf": sources,
        "records": records,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Campaign directory to verify")
    parser.add_argument(
        "--out", type=Path, help="Destination for verified receipt (.json or .json.gz)"
    )
    args = parser.parse_args()

    receipt = verify(args.folder.resolve())
    passed_count = sum(1 for c in receipt["checks"] if c["passed"])
    total_count = len(receipt["checks"])
    print(
        f"Verified {total_count} checks: {passed_count} passed, {total_count - passed_count} failed"
    )

    if not receipt["passed"]:
        for c in receipt["checks"]:
            if not c["passed"]:
                print(f"  FAIL: {c['check']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.out.name.endswith(".gz"):
            with gzip.open(args.out, "wt", encoding="utf-8") as f:
                f.write(raw)
        else:
            args.out.write_text(raw, encoding="utf-8")
        print(f"Receipt written to {args.out}")

    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
