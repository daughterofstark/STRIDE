"""Run the explicit workflow CLI with ``python -m mechanism.workflow``."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised by Python's module runner
    raise SystemExit(main())
