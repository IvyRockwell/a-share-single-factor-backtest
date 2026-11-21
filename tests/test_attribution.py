"""Tests for the same-universe monthly benchmark and the attribution tables."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attribution import (
    BENCHMARK_FACTORS,
    AttributionConfig,
    build_benchmark_targets,
    build_bridge,
    build_long_short,
    build_pool_benchmark,
    build_alpha_table,
    load_attribution_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> AttributionConfig:
    return AttributionConfig(Path("data/processed"), 12, 24, 0.0, 0.8, ("raw", "neutral"), "1")


def test_benchmark_targets_use_equal_and_cap_weights_and_fall_back() -> None:
    execution = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-29"]),
        "execution_date": pd.to_datetime(["2021-02-01"]),
    })
    pool = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-29", "2021-01-29"]),
        "stock_code": ["A", "B"],
    })
    caps = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-29", "2021-01-29"]),
        "stock_code": ["A", "B"],
        "market_cap": [1.0, 3.0],
        "has_market_cap": [True, True],
    })
    targets, fallback = build_benchmark_targets(pool, caps, execution, min_cap_coverage=0.8)
    ew = targets.loc[targets["factor"].eq(BENCHMARK_FACTORS[0])].set_index("stock_code")["target_weight"]
    cw = targets.loc[targets["factor"].eq(BENCHMARK_FACTORS[1])].set_index("stock_code")["target_weight"]
    assert ew.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    assert cw.to_dict() == pytest.approx({"A": 0.25, "B": 0.75})
    assert fallback.empty

    partial = caps.copy()
    partial["has_market_cap"] = [True, False]
    targets, fallback = build_benchmark_targets(pool, partial, execution, min_cap_coverage=0.8)
    cw = targets.loc[targets["factor"].eq(BENCHMARK_FACTORS[1])].set_index("stock_code")["target_weight"]
    assert cw.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    assert len(fallback) == 1


def test_pool_benchmark_is_open_to_open_and_includes_rebalance_gap() -> None:
    days = pd.bdate_range("2021-02-01", "2021-03-31")
    rows = []
    for day in days:
        if day == pd.Timestamp("2021-02-01"):
            rows.append({"stock_code": "A", "trade_date": day, "open_price": 100.0, "close_price": 110.0,
                         "total_return": 0.10, "trading_status": "1", "quality_flags": ""})
        elif day == pd.Timestamp("2021-03-01"):
            rows.append({"stock_code": "A", "trade_date": day, "open_price": 121.0, "close_price": 121.0,
                         "total_return": 0.10, "trading_status": "1", "quality_flags": ""})
        else:
            price = 110.0 if day < pd.Timestamp("2021-03-01") else 121.0
            rows.append({"stock_code": "A", "trade_date": day, "open_price": price, "close_price": price,
                         "total_return": 0.0, "trading_status": "1", "quality_flags": ""})
    daily = pd.DataFrame(rows)
    targets = pd.DataFrame({
        "factor": ["pe", "pe"],
        "signal_date": pd.to_datetime(["2021-01-29", "2021-02-26"]),
        "execution_date": pd.to_datetime(["2021-02-01", "2021-03-01"]),
        "stock_code": ["A", "A"],
        "target_weight": [1.0, 1.0],
    })
    pool = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-29", "2021-02-26"]),
        "stock_code": ["A", "A"],
    })
    caps = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-29", "2021-02-26"]),
        "stock_code": ["A", "A"],
        "market_cap": [100.0, 100.0],
        "has_market_cap": [True, True],
    })
    monthly, fallback = build_pool_benchmark(pool, daily, caps, targets, _config())
    ew = monthly.loc[monthly["factor"].eq("ew_pool_monthly")].sort_values("month")
    # February's open-to-open return includes the 1 February intraday move and
    # the 1 March opening gap; March then starts from that open.
    assert ew["monthly_return"].tolist() == pytest.approx([0.21, 0.0])
    assert ew["month"].tolist() == ["2021-02", "2021-03"]
    assert ew["end_nav"].tolist() == pytest.approx([1.21, 1.21])
    assert fallback.empty


def _monthly_inputs(months: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_keys = pd.period_range("2019-01", periods=months, freq="M").astype(str)
    x = np.linspace(-0.02, 0.03, months)
    y = 0.005 + 0.5 * x
    strategy_rows = []
    for factor in ("pe", "momentum", "roe"):
        for return_type in ("gross", "net_0bp", "net_10bp", "net_20bp"):
            for month, value in zip(month_keys, y):
                strategy_rows.append({"factor": factor, "month_key": month, "strategy_return": value,
                                      "return_type": return_type})
    strategy = pd.DataFrame(strategy_rows)
    hs300 = pd.DataFrame({"month_key": month_keys, "benchmark_monthly_return": x})
    benchmark_rows = []
    for factor in BENCHMARK_FACTORS:
        for month, value in zip(month_keys, x):
            benchmark_rows.append({"factor": factor, "month_key": month, "monthly_return": value})
    return strategy, hs300, pd.DataFrame(benchmark_rows)


def test_alpha_regression_recovers_a_known_alpha_and_beta() -> None:
    strategy, hs300, benchmark = _monthly_inputs()
    alpha, style = build_alpha_table(strategy, hs300, benchmark, _config())
    row = alpha.loc[
        alpha["factor"].eq("pe")
        & alpha["return_type"].eq("gross")
        & alpha["benchmark"].eq("H00300")
    ].iloc[0]
    assert row["alpha_annualized"] == pytest.approx((1.005) ** 12 - 1, rel=1e-6)
    assert row["beta"] == pytest.approx(0.5, rel=1e-6)
    assert np.isfinite(row["alpha_t"])
    style_row = style.loc[style["factor"].eq("pe") & style["return_type"].eq("gross")].iloc[0]
    assert style_row["alpha_annualized"] == pytest.approx((1.005) ** 12 - 1, rel=1e-6)


def test_long_short_summary_and_bridge_use_monthly_means() -> None:
    rows: list[dict[str, object]] = []
    spreads = [0.10, -0.05, 0.02]
    q1 = [0.00, 0.01, 0.02]
    q5 = [0.10, -0.04, 0.04]
    for day, spread, low, high in zip(pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"]),
                                      spreads, q1, q5):
        for quantile in range(1, 6):
            value = low if quantile == 1 else high if quantile == 5 else 0.02
            rows.append({"factor": "pe", "score_type": "raw", "signal_date": day, "quantile": quantile,
                         "equal_weight_forward_return": value, "q_high_minus_q_low": spread})
    quantiles = pd.DataFrame(rows)
    summary, monthly = build_long_short(quantiles, None, _config())
    assert len(summary) == 1
    assert summary.iloc[0]["n_months"] == 3
    assert summary.iloc[0]["mean_monthly"] == pytest.approx(np.mean(spreads))
    bridge = build_bridge(quantiles)
    assert bridge.iloc[0]["q5_minus_q1"] == pytest.approx(np.mean(spreads))
    assert bridge.iloc[0]["q5_minus_pool"] == pytest.approx(np.mean(q5) - np.mean([np.mean(q1), 0.02, 0.02, 0.02, np.mean(q5)]))
    assert len(monthly) == 3


def test_attribution_config_carries_the_shared_delisting_assumption() -> None:
    config = load_attribution_config(ROOT / "configs" / "config.yaml")
    assert config.delisting_return == pytest.approx(-1.0)
