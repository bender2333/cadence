"""Keep source and distribution identities consistent before building a wheel."""

import ast
import os
import subprocess
import tomllib
from pathlib import Path

import cadence


def test_package_version_matches_distribution_metadata():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert cadence.__version__ == metadata["project"]["version"]


def test_documented_installation_matches_distribution_identity():
    root = Path(__file__).resolve().parents[1]
    assert f"cadence-net=={cadence.__version__}" in (root / "README.md").read_text()


def test_scripts_included_in_distribution_sdist():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    only_include = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert "scripts" in only_include


def test_scripts_gate_contract_and_executability():
    root = Path(__file__).resolve().parents[1]
    check_sh = root / "scripts" / "check.sh"
    smoke_npu = root / "scripts" / "smoke_npu.py"

    assert check_sh.is_file(), "scripts/check.sh must exist"
    assert os.access(check_sh, os.X_OK), "scripts/check.sh must be executable"
    subprocess.run(["bash", "-n", str(check_sh)], check=True)

    assert smoke_npu.is_file(), "scripts/smoke_npu.py must exist"
    compile(smoke_npu.read_text(), str(smoke_npu), "exec")
    tree = ast.parse(smoke_npu.read_text())
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in funcs, "scripts/smoke_npu.py must define main()"
