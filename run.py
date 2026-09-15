"""Explicit command-line entry point for completed research stages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.data_loader import main as stage2_main
from src.universe import main as stage3_main


def main(argv: Sequence[str] | None = None) -> None:
    """Run one completed stage only when the caller explicitly selects it."""
    parser = argparse.ArgumentParser(description="Run a completed research stage.")
    parser.add_argument("stage", nargs="?", choices=("stage2", "stage3"), help="stage to run")
    args = parser.parse_args(argv)

    if args.stage is None:
        parser.print_help()
        return
    if args.stage == "stage2":
        stage2_main([])
    else:
        stage3_main([])


if __name__ == "__main__":
    main()
