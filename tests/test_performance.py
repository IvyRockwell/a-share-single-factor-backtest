"""Tests for Stage 9 risk-return metrics and source-lineage checks."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.performance import PerformanceConfig, build_performance_outputs, validate_outputs


FACTORS = ("pe", "momentum", "roe")
DATES = pd.date_range("2021-01-04", periods=3, freq="B")


def _config() -> PerformanceConfig:
    return PerformanceConfig(Path("data/processed"), 252, 0.0, DATES.min(), DATES.max(), len(DATES))


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage7_rows: list[dict[str, object]] = []
    stage8_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    gross_returns = np.array([-0.10, 0.20, 0.05])
    benchmark_returns = np.array([-0.20, 0.10, 0.02])
    gross_nav = np.cumprod(1 + gross_returns)
    benchmark_nav = np.cumprod(1 + benchmark_returns)
    for factor in FACTORS:
        for day, gross_return, nav, benchmark_return, benchmark_value in zip(
            DATES, gross_returns, gross_nav, benchmark_returns, benchmark_nav
        ):
            stage7_rows.append({
                "factor": factor,
                "trade_date": day,
                "gross_return": gross_return,
                "nav": nav,
                "benchmark_return": benchmark_return,
                "benchmark_nav": benchmark_value,
            })
        for scenario in (0, 10, 20):
            net_returns = gross_returns.copy()
            net_returns[0] -= (scenario + 3) / 10_000
            net_nav = np.cumprod(1 + net_returns)
            for index, (day, gross_return, nav, benchmark_return, benchmark_value, net_return, net_value) in enumerate(zip(
                DATES, gross_returns, gross_nav, benchmark_returns, benchmark_nav, net_returns, net_nav
            )):
                transaction_cost = (scenario + 3) / 10_000 if index == 0 else 0.0
                turnover = 1.0 if index == 0 else 0.0
                stage8_rows.append({
                    "factor": factor,
                    "trade_date": day,
                    "slippage_bp": scenario,
                    "gross_return": gross_return,
                    "nav": nav,
                    "benchmark_return": benchmark_return,
                    "benchmark_nav": benchmark_value,
                    "net_return": net_return,
                    "net_nav": net_value,
                    "turnover": turnover,
                    "total_transaction_cost": transaction_cost,
                })
            cost_rows.append({
                "factor": factor,
                "trade_date": DATES[0],
                "slippage_bp": scenario,
                "turnover": 1.0,
                "total_transaction_cost": (scenario + 3) / 10_000,
            })
    return pd.DataFrame(stage7_rows), pd.DataFrame(stage8_rows), pd.DataFrame(cost_rows)


def test_metrics_use_confirmed_daily_formulas_and_initial_nav_drawdown() -> None:
    outputs = build_performance_outputs(*_inputs(), _config())
    pe = outputs.metrics.loc[outputs.metrics["series_id"].eq("pe_gross")].iloc[0]
    returns = np.array([-0.10, 0.20, 0.05])
    cumulative = np.prod(1 + returns) - 1
    assert pe.cumulative_return == pytest.approx(cumulative)
    assert pe.annualized_return == pytest.approx((1 + cumulative) ** (252 / 3) - 1)
    assert pe.annualized_volatility == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252))
    assert pe.sharpe_ratio == pytest.approx(returns.mean() / np.std(returns, ddof=1) * np.sqrt(252))
    assert pe.max_drawdown == pytest.approx(-0.10)


def test_outputs_keep_gross_net_and_benchmark_cost_meanings_separate() -> None:
    outputs = build_performance_outputs(*_inputs(), _config())
    validate_outputs(outputs, _config())
    assert len(outputs.metrics) == 13
    gross = outputs.metrics.loc[outputs.metrics["series_type"].eq("gross")]
    net = outputs.metrics.loc[outputs.metrics["series_type"].eq("net")]
    benchmark = outputs.metrics.loc[outputs.metrics["series_type"].eq("benchmark")]
    assert gross["cumulative_turnover"].eq(1.0).all()
    assert gross["cumulative_transaction_cost"].isna().all()
    assert net["cumulative_transaction_cost"].gt(0).all()
    assert benchmark[["cumulative_turnover", "annualized_turnover", "cumulative_transaction_cost"]].isna().all().all()
    assert len(outputs.monthly) == 13
    assert outputs.manual_review["absolute_difference"].le(1e-12).all()


def test_mismatched_strategy_dates_are_rejected() -> None:
    stage7, stage8, costs = _inputs()
    stage7 = stage7.loc[~(stage7["factor"].eq("roe") & stage7["trade_date"].eq(DATES[-1]))]
    with pytest.raises(ValueError, match="date ranges are not identical"):
        build_performance_outputs(stage7, stage8, costs, _config())
