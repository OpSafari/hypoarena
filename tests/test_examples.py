"""Every bundled example runs offline, exits 0 and stays honest.

Examples are loaded in-process (no subprocess, no network) and their printed
output is asserted to contain the measured markers each one promises plus the
standing honesty note. Running them must not write files or touch the network.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# The markers each example must print, keyed by its directory name. Pinning the
# set of examples makes adding or removing one a deliberate, reviewed change.
EXPECTED_MARKERS = {
    "planted_corpus": ("grounded_rate=", "fabricated"),
    "tournament": ("order_recovery=", "recovered order equals planted order: True"),
    "evolution": ("grew=", "accepted="),
}


def example_scripts() -> list[Path]:
    """Return every example script, sorted for stable parametrization."""
    return sorted(EXAMPLES.glob("*/*.py"))


def run_example(path: Path) -> str:
    """Import one example and run its ``main``; return what it printed."""
    name = f"_hypoarena_example_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = module.main()
    assert code == 0, f"{path} exited with {code}"
    return buffer.getvalue()


def test_the_bundled_examples_are_exactly_the_expected_set() -> None:
    assert {path.parent.name for path in example_scripts()} == set(EXPECTED_MARKERS)


@pytest.mark.parametrize("script", example_scripts(), ids=lambda p: p.parent.name)
def test_example_runs_offline_and_is_honest(script: Path) -> None:
    out = run_example(script)
    assert out.strip()
    assert "synthetic" in out.lower()
    assert "note:" in out.lower()
    for marker in EXPECTED_MARKERS[script.parent.name]:
        assert marker in out
