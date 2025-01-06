"""Smoke checks: the package imports and advertises its version."""

import re

import hypoarena
from hypoarena._version import __version__ as raw_version


def test_version_is_exposed() -> None:
    assert hypoarena.__version__ == raw_version


def test_version_follows_semver_patch_series() -> None:
    assert re.fullmatch(r"0\.\d+\.\d+", hypoarena.__version__)


def test_all_entries_are_importable_attributes() -> None:
    for name in hypoarena.__all__:
        assert hasattr(hypoarena, name), name
