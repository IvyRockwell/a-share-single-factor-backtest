"""Command-line entry point for the research pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from src.backtest import main as backtest_main
from src.costs import main as costs_main
from src.data_loader import main as clean_main
from src.factor_test import main as factor_test_main
from src.factors import main as factors_main
from src.performance import main as performance_main
from src.portfolio import main as portfolio_main
from src.robustness import main as robustness_main
from src.universe import main as universe_main
from src.visualization import main as figures_main

STEPS: dict[str, Callable[[list[str]], None]] = {
    "clean": clean_main,
    "universe": universe_main,
    "factors": factors_main,
    "evaluate": factor_test_main,
    "portfolio": portfolio_main,
    "backtest": backtest_main,
    "costs": costs_main,
    "performance": performance_main,
    "robustness": robustness_main,
    "figures": figures_main,
}
ORDER = tuple(STEPS)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected steps in pipeline order."""
    parser = argparse.ArgumentParser(description="Run research pipeline steps.")
    parser.add_argument(
        "steps",
        nargs="*",
        choices=(*ORDER, "all"),
        help="steps to run; each one depends on the earlier ones having run",
    )
    args = parser.parse_args(argv)
    if not args.steps:
        parser.print_help()
        return
    selected = set(ORDER) if "all" in args.steps else set(args.steps)
    for step in ORDER:
        if step in selected:
            STEPS[step]([])


if __name__ == "__main__":
    main()
