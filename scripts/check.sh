#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate .venv if present and not already activated
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$REPO_ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Ensure Ascend CANN environment is sourced if present
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

echo "=== 1. Linting (ruff) ==="
ruff check src tests

echo "=== 2. Type checking (mypy) ==="
mypy

echo "=== 3. Core test suite (pytest) ==="
if [ $# -gt 0 ]; then
    pytest "$@"
else
    pytest -q \
        tests/test_composed_brain.py \
        tests/test_qualified_learning.py \
        tests/test_feedback_transaction.py \
        tests/test_generic_memory_checkpoint.py \
        tests/test_settlement_reports.py \
        tests/test_record_patch.py \
        tests/test_documentation.py
fi

echo "=== All checks passed cleanly! ==="
