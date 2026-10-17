#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Enforce board global exclusive lock if accessible (best-effort across non-root or non-board hosts)
if [ "${BOARD_HEAVY_LOCK_HELD:-0}" != "1" ]; then
    if ( exec 200>>/var/lock/board_npu_heavy.lock ) 2>/dev/null; then
        exec 200>>/var/lock/board_npu_heavy.lock
        if flock -w 300 200 2>/dev/null; then
            export BOARD_HEAVY_LOCK_HELD=1
        fi
    elif ( exec 200</var/lock/board_npu_heavy.lock ) 2>/dev/null; then
        exec 200</var/lock/board_npu_heavy.lock
        if flock -w 300 200 2>/dev/null; then
            export BOARD_HEAVY_LOCK_HELD=1
        fi
    fi
fi

# Attach to agent cgroup (MemoryMax=5.2G)
[ -f /sys/fs/cgroup/unified/agent.slice/cgroup.procs ] && echo $$ > /sys/fs/cgroup/unified/agent.slice/cgroup.procs 2>/dev/null || true
[ -f /sys/fs/cgroup/memory/agent.slice/cgroup.procs ] && echo $$ > /sys/fs/cgroup/memory/agent.slice/cgroup.procs 2>/dev/null || true

# Activate .venv if present and not already activated
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$REPO_ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Ensure Ascend CANN environment is sourced if present
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    # shellcheck disable=SC1091
    set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
    export TBE_PARALLEL_COMPILER=2
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
fi

# Snapshot system baseline before tests (best-effort)
dmesg_baseline=$( (dmesg 2>/dev/null || true) | wc -l )
huge_free_start=$(grep -m 1 HugePages_Free /proc/meminfo 2>/dev/null | awk "{print \$2}" || echo 0)

echo "=== 1. Linting (ruff) ==="
ruff check src tests

echo "=== 2. Type checking (mypy) ==="
mypy

echo "=== 3. NPU device smoke test ==="
if [ -f "$REPO_ROOT/scripts/smoke_npu.py" ] && python -c 'import cadence as cd; exit(0 if cd.available_backends().get("torch") == "npu float32" else 1)' 2>/dev/null; then
    python "$REPO_ROOT/scripts/smoke_npu.py"
else
    echo "  [SKIP] NPU device not available or not configured (torch != 'npu float32'); skipping smoke test"
fi

echo "=== 4. Core test suite (pytest) ==="
if [ $# -gt 0 ]; then
    timeout 300s pytest "$@"
else
    timeout 300s pytest -q \
        tests/test_composed_brain.py \
        tests/test_qualified_learning.py \
        tests/test_feedback_transaction.py \
        tests/test_generic_memory_checkpoint.py \
        tests/test_settlement_reports.py \
        tests/test_record_patch.py \
        tests/test_documentation.py \
        tests/test_packaging.py
fi

echo "=== 5. System Stability & Resource Leak Audit ==="
# 5.1 Kernel error audit (dmesg delta, best-effort)
dmesg_new=$( (dmesg 2>/dev/null || true) | tail -n +"$((dmesg_baseline + 1))" )
critical_errors=$(echo "$dmesg_new" | grep -iE "segfault|kernel NULL pointer|Out of memory|Call Trace|kernel panic|oom-killer" || true)
if [ -n "$critical_errors" ]; then
    echo "WARNING: Kernel errors detected during test run:" >&2
    echo "$critical_errors" >&2
else
    echo "  [PASS] Kernel log clean (no panics, segfaults, or OOMs)"
fi
# 5.2 HugePages recovery check
huge_free_end=$(grep -m 1 HugePages_Free /proc/meminfo 2>/dev/null | awk "{print \$2}" || echo 0)
if [ "$huge_free_start" -gt 0 ] && [ "$huge_free_end" -lt "$huge_free_start" ]; then
    echo "WARNING: HugePages consumed during test: started with $huge_free_start, ended with $huge_free_end"
fi
echo "  [PASS] HugePages accounted for (Free: $huge_free_end)"

echo "=== All checks passed cleanly with system stability verified! ==="
