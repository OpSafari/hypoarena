#!/usr/bin/env python3
"""Smoke-test a built wheel in a throwaway virtual environment.

Usage: ``python scripts/check_package.py [path-to-wheel]``

Without an argument the newest ``dist/*.whl`` is used (run ``make build`` first).
The script creates a temporary venv, installs the wheel, and verifies that the
installed distribution imports, reports the version recorded in the source
tree, and ships the ``hypoarena`` console script.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = "import hypoarena; print(hypoarena.__version__)"


def fail(message: str) -> int:
    print(f"check_package: FAIL {message}")
    return 1


def find_wheel(explicit: str | None) -> Path | None:
    """Return the wheel to test: the explicit path, or the newest in dist/."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates = sorted((ROOT / "dist").glob("hypoarena-*.whl"))
    return candidates[-1] if candidates else None


def project_version() -> str:
    """Read the declared version from pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("check_package: cannot read version from pyproject.toml")
    return match.group(1)


def declares_console_script(wheel: Path) -> bool:
    """True when the wheel ships a `hypoarena` console entry point."""
    with zipfile.ZipFile(wheel) as archive:
        entries = [
            n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt")
        ]
        if not entries:
            return False
        return "hypoarena = hypoarena.cli:main" in archive.read(entries[0]).decode()


def venv_python(venv: Path) -> str:
    return str(venv / "bin" / "python")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def main(argv: list[str]) -> int:
    """Install the wheel in a fresh venv and probe it; return a process exit code."""
    wheel = find_wheel(argv[1] if len(argv) > 1 else None)
    if wheel is None:
        return fail("no wheel found; run `make build` first")
    print(f"check_package: testing {wheel.name}")
    if not declares_console_script(wheel):
        return fail("wheel does not declare the `hypoarena` console script")
    expected = project_version()
    with tempfile.TemporaryDirectory(prefix="hypoarena-check-") as temporary:
        venv = Path(temporary) / "venv"
        create = run([sys.executable, "-m", "venv", str(venv)])
        if create.returncode:
            return fail(f"cannot create venv: {create.stderr.strip()[-400:]}")
        python = venv_python(venv)
        install = run([python, "-m", "pip", "install", "--quiet", str(wheel)])
        if install.returncode:
            return fail(f"pip install failed: {install.stderr.strip()[-600:]}")
        probe = run([python, "-c", PROBE])
        if probe.returncode:
            return fail(f"import probe failed: {probe.stderr.strip()[-600:]}")
        reported = probe.stdout.strip()
        if reported != expected:
            return fail(f"installed version {reported!r} != source {expected!r}")
        console = run([str(venv / "bin" / "hypoarena"), "--version"])
        if console.returncode or expected not in console.stdout:
            return fail(f"console script failed: {console.stderr.strip()[-400:]}")
    print(f"check_package: OK hypoarena {expected} installs and imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
