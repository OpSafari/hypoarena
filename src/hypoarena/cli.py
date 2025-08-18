"""Command line entry point for hypoarena.

The parser grows with the toolkit; this initial version only reports the
installed version so that packaging smoke tests exercise a real console script.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hypoarena._version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="hypoarena",
        description=(
            "Offline workbench for hypothesis-discovery pipelines: grounded "
            "claim graphs, synthetic literature, debate loops and Elo tournaments."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"hypoarena {__version__}"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and return a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
