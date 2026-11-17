"""Tests for the Stage 10 robustness grid and independent audit rules."""

from pathlib import Path

import pandas as pd
import pytest

from src.performance import PerformanceOutputs
from src.robustness import (
    _compare_frames,
    _metrics_table,
    _portfolio_audit,
    filter_signal_frequency,
    load_robustness_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_contains_the_complete_confirmed_parameter_grid() -> None:
    config = load_robustness_config(ROOT / "configs" / "config.yaml")
    assert config.selection_quantiles == (0.1, 0.2, 0.3)
    assert config.rebalance_frequencies == ("monthly", "quarterly")
    assert config.quarterly_signal_months == (3, 6, 9, 12)
    assert config.slippage_scenarios_bp == (0, 10, 20)
    assert config.commission_rate == pytest.approx(0.0003)


def test_quarterly_uses_existing_calendar_quarter_end_signals() -> None:
    details = pd.DataFrame({
        "signal_date": pd.to_datetime([
            "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30",
            "2020-06-30", "2020-09-30", "2020-12-31",
        ]),
        "stock_code": "000001",
    })
    quarterly = filter_signal_frequency(details, "quarterly", (3, 6, 9, 12))
    assert quarterly["signal_date"].dt.month.tolist() == [3, 6, 9, 12]
    assert len(filter_signal_frequency(details, "monthly", (3, 6, 9, 12))) == len(details)


def test_portfolio_audit_keeps_boundary_ties_and_next_trading_day() -> None:
    details = pd.DataFrame({
        "signal_date": pd.Timestamp("2021-03-31"),
        "stock_code": ["000001", "000002", "000003", "000004", "000005"],
        "pe_is_rankable": [True] * 5,
        "momentum_is_rankable": [True] * 5,
        "roe_is_rankable": [True] * 5,
    })
    targets = pd.DataFrame([
        {"factor": factor, "signal_date": pd.Timestamp("2021-03-31"), "stock_code": code, "target_weight": 0.5}
        for factor in ("pe", "momentum", "roe") for code in ("000004", "000005")
    ])
    trading_dates = pd.Series(pd.to_datetime(["2021-03-31", "2021-04-01"]))
    audit = _portfolio_audit(details, targets, trading_dates, 0.2, "quarterly")
    assert audit["selected_stock_count"].eq(2).all()
    assert audit["actual_selection_ratio"].eq(0.4).all()
    assert audit["target_weight_sum"].eq(1.0).all()
    assert audit["execution_date"].eq(pd.Timestamp("2021-04-01")).all()


def _performance_outputs() -> PerformanceOutputs:
    rows = []
    for factor in ("pe", "momentum", "roe"):
        rows.append({
            "series_id": f"{factor}_gross", "series_type": "gross", "factor": factor,
            "slippage_bp": float("nan"), "start_date": pd.Timestamp("2021-01-04"),
            "end_date": pd.Timestamp("2021-01-06"), "observation_count": 3,
            "cumulative_return": 0.10, "annualized_return": 0.20,
            "annualized_volatility": 0.30, "max_drawdown": -0.10,
            "sharpe_ratio": 0.50, "cumulative_turnover": 1.0,
            "annualized_turnover": 84.0, "cumulative_transaction_cost": float("nan"),
        })
        for scenario in (0, 10, 20):
            rows.append({
                "series_id": f"{factor}_net_{scenario}bp", "series_type": "net", "factor": factor,
                "slippage_bp": scenario, "start_date": pd.Timestamp("2021-01-04"),
                "end_date": pd.Timestamp("2021-01-06"), "observation_count": 3,
                "cumulative_return": 0.09 - scenario / 10_000, "annualized_return": 0.19,
                "annualized_volatility": 0.30, "max_drawdown": -0.11,
                "sharpe_ratio": 0.45, "cumulative_turnover": 1.0,
                "annualized_turnover": 84.0, "cumulative_transaction_cost": 0.001 + scenario / 10_000,
            })
    return PerformanceOutputs(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_metrics_table_has_each_factor_cost_scenario_and_separate_gross_net_fields() -> None:
    metrics = _metrics_table(_performance_outputs(), 0.2, "monthly", (0, 10, 20))
    assert len(metrics) == 9
    assert not metrics.duplicated(["selection_quantile", "rebalance_frequency", "factor", "slippage_bp"]).any()
    assert {0, 10, 20} == set(metrics["slippage_bp"])
    assert metrics["gross_cumulative_return"].eq(0.10).all()
    assert metrics.loc[metrics["slippage_bp"].eq(0), "cumulative_transaction_cost"].gt(0).all()
    assert metrics["net_cumulative_return"].lt(metrics["gross_cumulative_return"]).all()


def test_baseline_comparison_uses_the_confirmed_floating_tolerance() -> None:
    expected = pd.DataFrame({"key": ["a", "b"], "value": [1.0, 2.0]})
    within = pd.DataFrame({"key": ["a", "b"], "value": [1.0 + 5e-13, 2.0]})
    result = _compare_frames("fixture", within, expected)
    assert result["status"] == "passed"
    assert result["max_absolute_difference"] <= 1e-12
    outside = pd.DataFrame({"key": ["a", "b"], "value": [1.0 + 1e-5, 2.0]})
    with pytest.raises(ValueError, match="Baseline reproduction failed"):
        _compare_frames("fixture", outside, expected)


def test_baseline_comparison_treats_pd_na_and_nan_as_the_same_missing_value() -> None:
    actual = pd.DataFrame({"key": ["a", "b"], "weight": pd.Series([1.0, pd.NA], dtype="object")})
    expected = pd.DataFrame({"key": ["a", "b"], "weight": [1.0, float("nan")]})
    assert _compare_frames("missing_fixture", actual, expected)["status"] == "passed"


def test_baseline_comparison_normalizes_csv_date_strings() -> None:
    actual = pd.DataFrame({"signal_date": [pd.Timestamp("2021-03-31")], "value": [1.0]})
    expected = pd.DataFrame({"signal_date": ["2021-03-31"], "value": [1.0]})
    assert _compare_frames("date_fixture", actual, expected)["status"] == "passed"
