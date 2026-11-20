"""Regression tests for equal-weight execution and return rules."""

from pathlib import Path

import pandas as pd
import pytest

from src.backtest import BacktestConfig, build_backtest_outputs, validate_outputs

FACTORS = ("pe", "momentum", "roe")
CONFIG = BacktestConfig(Path("data/processed"), "1")


def _market(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    defaults = {
        "open_price": 100.0, "close_price": 100.0, "total_return": 0.0,
        "PreClosePrice": "100", "trading_status": "1", "quality_flags": "",
    }
    for key, value in defaults.items():
        if key not in frame:
            frame[key] = value
        else:
            frame[key] = frame[key].fillna(value)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _market_preserving_missing(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build market rows without replacing an explicitly missing value."""
    frame = pd.DataFrame(rows)
    defaults = {
        "open_price": 100.0, "close_price": 100.0, "total_return": 0.0,
        "PreClosePrice": "100", "trading_status": "1", "quality_flags": "",
    }
    for key, value in defaults.items():
        if key not in frame:
            frame[key] = value
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _inputs(periods: list[tuple[str, str, dict[str, float]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = []
    actions = []
    prior: dict[str, set[str]] = {factor: set() for factor in FACTORS}
    for signal, execution, weights in periods:
        for factor in FACTORS:
            current = set(weights)
            for code, weight in weights.items():
                targets.append({"factor": factor, "signal_date": pd.Timestamp(signal), "execution_date": pd.Timestamp(execution), "stock_code": code, "target_weight": weight})
                actions.append({"factor": factor, "signal_date": pd.Timestamp(signal), "execution_date": pd.Timestamp(execution), "stock_code": code, "target_action": "continue_holding" if code in prior[factor] else "new_buy", "target_weight": weight})
            for code in prior[factor] - current:
                actions.append({"factor": factor, "signal_date": pd.Timestamp(signal), "execution_date": pd.Timestamp(execution), "stock_code": code, "target_action": "planned_sell", "target_weight": None})
            prior[factor] = current
    return pd.DataFrame(targets), pd.DataFrame(actions)


def _benchmark(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": pd.to_datetime(days), "total_return": 0.0, "quality_flags": ""})


def test_open_rebalance_reduces_overweight_adds_underweight_and_restores_equal_weight() -> None:
    targets, actions = _inputs([
        ("2021-01-29", "2021-02-01", {"A": 0.5, "B": 0.5}),
        ("2021-02-26", "2021-03-01", {"A": 0.5, "B": 0.5}),
    ])
    daily = _market([
        {"stock_code": "A", "trade_date": "2021-02-01", "close_price": 120.0},
        {"stock_code": "B", "trade_date": "2021-02-01", "close_price": 80.0},
        {"stock_code": "A", "trade_date": "2021-03-01"}, {"stock_code": "B", "trade_date": "2021-03-01"},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-03-01"]), CONFIG)
    records = output.rebalance_records.loc[output.rebalance_records["trade_date"].eq(pd.Timestamp("2021-03-01"))]
    assert set(records["trade_reason"]) == {"rebalance_reduce", "rebalance_increase"}
    weights = output.holdings.loc[output.holdings["factor"].eq("pe") & output.holdings["trade_date"].eq(pd.Timestamp("2021-03-01")), "portfolio_weight"]
    assert weights.tolist() == pytest.approx([0.5, 0.5])
    validate_outputs(output)


def test_cash_short_buys_are_scaled_proportionally() -> None:
    targets, actions = _inputs([
        ("2021-01-29", "2021-02-01", {"X": 0.4, "Y": 0.6}),
        ("2021-02-26", "2021-03-01", {"A": 0.5, "B": 0.5}),
    ])
    daily = _market([
        {"stock_code": code, "trade_date": day, "trading_status": status}
        for day, values in [("2021-02-01", [("X", "1"), ("Y", "1")]), ("2021-03-01", [("X", "1"), ("Y", "2"), ("A", "1"), ("B", "1")])]
        for code, status in values
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-03-01"]), CONFIG)
    buys = output.rebalance_records.loc[output.rebalance_records["factor"].eq("pe") & output.rebalance_records["trade_date"].eq(pd.Timestamp("2021-03-01")) & output.rebalance_records["event_type"].eq("buy")]
    assert set(buys["stock_code"]) == {"A", "B"}
    assert buys["cash_scale"].tolist() == pytest.approx([0.4, 0.4])
    assert buys["executed_open_value"].tolist() == pytest.approx([0.2, 0.2])
    assert buys["status"].eq("partially_filled_cash_constrained").all()


def test_failed_new_buy_and_failed_continuing_adjustment_are_not_retried() -> None:
    targets, actions = _inputs([
        ("2021-01-29", "2021-02-01", {"A": 0.5, "B": 0.5}),
        ("2021-02-26", "2021-03-01", {"A": 0.5, "C": 0.5}),
    ])
    daily = _market([
        {"stock_code": "A", "trade_date": "2021-02-01", "close_price": 120.0}, {"stock_code": "B", "trade_date": "2021-02-01", "close_price": 80.0},
        {"stock_code": "A", "trade_date": "2021-03-01", "trading_status": "2"}, {"stock_code": "B", "trade_date": "2021-03-01"}, {"stock_code": "C", "trade_date": "2021-03-01", "trading_status": "2"},
        {"stock_code": "A", "trade_date": "2021-03-02"}, {"stock_code": "C", "trade_date": "2021-03-02"},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-03-01", "2021-03-02"]), CONFIG)
    failed = output.rebalance_records.loc[output.rebalance_records["trade_date"].eq(pd.Timestamp("2021-03-01")) & output.rebalance_records["stock_code"].isin(["A", "C"])]
    assert set(failed["trade_reason"]) == {"rebalance_reduce", "new_entry"}
    assert failed["executed_open_value"].eq(0).all()
    assert output.rebalance_records.loc[output.rebalance_records["trade_date"].eq(pd.Timestamp("2021-03-02")) & output.rebalance_records["stock_code"].isin(["A", "C"])].empty


def test_failed_complete_exit_remains_pending_until_recovery() -> None:
    targets, actions = _inputs([("2021-01-29", "2021-02-01", {"A": 1.0}), ("2021-02-26", "2021-03-01", {"B": 1.0})])
    daily = _market([
        {"stock_code": "A", "trade_date": "2021-02-01"}, {"stock_code": "A", "trade_date": "2021-03-01", "trading_status": "2"},
        {"stock_code": "B", "trade_date": "2021-03-01"}, {"stock_code": "A", "trade_date": "2021-03-02"},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-03-01", "2021-03-02"]), CONFIG)
    a = output.rebalance_records.loc[output.rebalance_records["stock_code"].eq("A")]
    assert {"pending_sale", "executed"}.issubset(set(a["status"]))
    recovered = a.loc[a["trade_date"].eq(pd.Timestamp("2021-03-02"))].iloc[0]
    assert recovered["trade_reason"] == "pending_exit_recovery"
    assert recovered["target_weight"] == recovered["target_open_value"] == 0


def test_pending_exit_is_cancelled_when_stock_reenters_target() -> None:
    targets, actions = _inputs([
        ("2021-01-29", "2021-02-01", {"A": 1.0}), ("2021-02-26", "2021-03-01", {"B": 1.0}), ("2021-03-31", "2021-04-01", {"A": 1.0}),
    ])
    daily = _market([
        {"stock_code": "A", "trade_date": "2021-02-01"}, {"stock_code": "A", "trade_date": "2021-03-01", "trading_status": "2"},
        {"stock_code": "B", "trade_date": "2021-03-01"}, {"stock_code": "A", "trade_date": "2021-04-01"}, {"stock_code": "B", "trade_date": "2021-04-01"},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-03-01", "2021-04-01"]), CONFIG)
    assert not output.rebalance_records.loc[output.rebalance_records["stock_code"].eq("A") & output.rebalance_records["trade_date"].eq(pd.Timestamp("2021-04-01")), "trade_reason"].eq("pending_exit_recovery").any()
    last = output.daily.loc[output.daily["factor"].eq("pe") & output.daily["trade_date"].eq(pd.Timestamp("2021-04-01"))].iloc[0]
    assert last["pending_sale_count"] == 0


def test_execution_day_uses_close_over_open_when_dretwd_differs_from_price_change() -> None:
    targets, actions = _inputs([("2021-01-29", "2021-02-01", {"A": 1.0})])
    daily = _market([
        {"stock_code": "A", "trade_date": "2021-02-01", "open_price": 100.0, "close_price": 110.0, "PreClosePrice": "100", "total_return": 0.20},
        {"stock_code": "A", "trade_date": "2021-02-02", "open_price": 110.0, "close_price": 121.0, "PreClosePrice": "110", "total_return": 0.10},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-02-02"]), CONFIG)
    pe = output.daily.loc[output.daily["factor"].eq("pe")].set_index("trade_date")
    assert pe.loc[pd.Timestamp("2021-02-01"), "nav"] == pytest.approx(1.10)
    assert pe.loc[pd.Timestamp("2021-02-02"), "nav"] == pytest.approx(1.21)


def test_valid_open_with_missing_close_freezes_post_trade_open_value() -> None:
    targets, actions = _inputs([
        ("2021-01-29", "2021-02-01", {"A": 1.0}),
        ("2021-02-26", "2021-03-01", {"A": 0.5, "B": 0.5}),
    ])
    daily = _market_preserving_missing([
        {"stock_code": "A", "trade_date": "2021-02-01"},
        {"stock_code": "A", "trade_date": "2021-03-01", "open_price": 110.0, "close_price": None, "total_return": 0.05},
        {"stock_code": "B", "trade_date": "2021-03-01"},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01", "2021-03-01"]), CONFIG)
    day = output.daily.loc[
        output.daily["factor"].eq("pe") & output.daily["trade_date"].eq(pd.Timestamp("2021-03-01"))
    ].iloc[0]
    holdings = output.holdings.loc[
        output.holdings["factor"].eq("pe") & output.holdings["trade_date"].eq(pd.Timestamp("2021-03-01"))
    ]
    assert day["nav"] == pytest.approx(1.0)
    assert day["cash_value"] >= 0
    assert holdings.set_index("stock_code").loc["A", "valuation_status"] == "frozen_at_post_trade_open_value"
    assert holdings["close_value"].notna().all()

def test_opening_limit_up_blocks_a_buy_and_leaves_cash() -> None:
    targets, actions = _inputs([("2021-01-29", "2021-02-01", {"A": 1.0})])
    daily = _market([
        {"stock_code": "A", "trade_date": "2021-02-01", "open_price": 110.0,
         "close_price": 110.0, "PreClosePrice": "100", "Markettype": "1"},
    ])
    output = build_backtest_outputs(targets, actions, daily, _benchmark(["2021-02-01"]), CONFIG)
    day = output.daily.loc[output.daily["factor"].eq("pe")].iloc[0]
    assert day["nav"] == pytest.approx(1.0)
    record = output.rebalance_records.loc[output.rebalance_records["factor"].eq("pe")].iloc[0]
    assert record["reason"] == "limit_up_open"


def test_backtest_engine_accepts_a_custom_factor_set_and_no_actions() -> None:
    targets = pd.DataFrame({
        "factor": ["benchmark"],
        "signal_date": [pd.Timestamp("2021-01-29")],
        "execution_date": [pd.Timestamp("2021-02-01")],
        "stock_code": ["A"],
        "target_weight": [1.0],
    })
    daily = _market([{"stock_code": "A", "trade_date": "2021-02-01"}])
    output = build_backtest_outputs(
        targets, None, daily, _benchmark(["2021-02-01"]), CONFIG, factors=("benchmark",)
    )
    assert set(output.daily["factor"]) == {"benchmark"}
    assert output.daily["nav"].iloc[-1] == pytest.approx(1.0)


def test_universe_benchmarks_are_equal_and_cap_weighted() -> None:
    from src.backtest import _universe_returns

    group = pd.DataFrame({
        "stock_code": ["A", "B", "C", "D"],
        "total_return": [0.10, 0.20, -0.10, 0.50],
        "quality_flags": ["", "", "", ""],
    })
    members = {pd.Timestamp("2021-01-29"): {"A", "B", "C"}}
    weights = {pd.Timestamp("2021-01-29"): pd.Series({"A": 1.0, "B": 3.0, "C": 6.0})}

    equal, capped = _universe_returns(group, pd.Timestamp("2021-01-29"), members, weights)

    assert equal == pytest.approx((0.10 + 0.20 - 0.10) / 3)
    assert capped == pytest.approx((1.0 * 0.10 + 3.0 * 0.20 + 6.0 * -0.10) / 10.0)


def test_universe_benchmark_skips_an_empty_membership() -> None:
    from src.backtest import _universe_returns

    group = pd.DataFrame({"stock_code": ["A"], "total_return": [0.10], "quality_flags": [""]})
    equal, capped = _universe_returns(group, pd.Timestamp("2021-01-29"), {}, {})
    assert pd.isna(equal) and pd.isna(capped)
