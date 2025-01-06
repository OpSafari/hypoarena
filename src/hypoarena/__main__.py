"""Allow ``python -m hypoarena`` to reach the CLI."""

from hypoarena.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
