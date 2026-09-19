"""Explicit command-line entry point for completed research stages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.data_loader import main as stage2_main
from src.factor_test import main as stage5_main
from src.factors import main as stage4_main
from src.portfolio import main as stage6_main
from src.backtest import main as stage7_main
from src.costs import main as stage8_main
from src.performance import main as stage9_main
from src.robustness import main as stage10_main
from src.visualization import main as stage11_main
from src.universe import main as stage3_main


def main(argv: Sequence[str] | None = None) -> None:
    """Run one completed stage only when the caller explicitly selects it."""
    parser = argparse.ArgumentParser(description="Run a completed research stage.")
    parser.add_argument("stage", nargs="?", choices=("stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "stage10", "stage11"), help="stage to run")
    args = parser.parse_args(argv)

    if args.stage is None:
        parser.print_help()
        return
    if args.stage == "stage2":
        stage2_main([])
    elif args.stage == "stage3":
        stage3_main([])
    elif args.stage == "stage4":
        stage4_main([])
    elif args.stage == "stage5":
        stage5_main([])
    elif args.stage == "stage6":
        stage6_main([])
    elif args.stage == "stage7":
        stage7_main([])
    elif args.stage == "stage8":
        stage8_main([])
    elif args.stage == "stage9":
        stage9_main([])
    elif args.stage == "stage10":
        stage10_main([])
    else:
        stage11_main([])


if __name__ == "__main__":
    main()
