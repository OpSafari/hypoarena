"""Golden snapshot of ``hypoarena.__all__``.

Adding a public name is a deliberate API change: update ``EXPECTED`` in the same
commit so reviewers see the surface grow.
"""

from __future__ import annotations

import hypoarena

EXPECTED: tuple[str, ...] = ("__version__",)


def test_all_matches_the_golden_snapshot() -> None:
    assert tuple(hypoarena.__all__) == EXPECTED


def test_all_is_sorted_and_duplicate_free() -> None:
    names = list(hypoarena.__all__)
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_every_exported_name_resolves() -> None:
    for name in hypoarena.__all__:
        assert hasattr(hypoarena, name), name


def test_no_private_helpers_are_exported() -> None:
    private = [
        n for n in hypoarena.__all__ if n.startswith("_") and not n.startswith("__")
    ]
    assert private == []
