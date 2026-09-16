"""Tests for the confirmed Stage 7 execution and return rules."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BacktestConfig, build_backtest_outputs


FACTORS = ("pe", "momentum", "roe")


def _daily() -> pd.DataFrame:
    return pd.DataFrame([
        {"stock_code": "000001", "trade_date": "2021-02-01", "open_price": 100.0, "close_price": 100.0, "total_return": 0.10, "PreClosePrice": "100", "trading_status": "1", "quality_flags": ""},
        {"stock_code": "000001", "trade_date": "2021-02-02", "open_price": 100.0, "close_price": 100.0, "total_return": 0.10, "PreClosePrice": "100", "trading_status": "2", "quality_flags": ""},
        {"stock_code": "000001", "trade_date": "2021-02-03", "open_price": 100.0, "close_price": 100.0, "total_return": 0.10, "PreClosePrice": "100", "trading_status": "1", "quality_flags": ""},
        {"stock_code": "000001", "trade_date": "2021-03-01", "open_price": 100.0, "close_price": 100.0, "total_return": 0.0, "PreClosePrice": "100", "trading_status": "1", "quality_flags": ""},
    ]).assign(trade_date=lambda x: pd.to_datetime(x.trade_date))


def _inputs(unfilled: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets, actions = [], []
    for factor in FACTORS:
        targets.extend([
            {"factor": factor, "signal_date": pd.Timestamp("2021-01-29"), "execution_date": pd.Timestamp("2021-02-01"), "stock_code": "000001"},
            {"factor": factor, "signal_date": pd.Timestamp("2021-03-31"), "execution_date": pd.NaT, "stock_code": "000001"},
        ])
        actions.extend([
            {"factor": factor, "signal_date": pd.Timestamp("2021-01-29"), "execution_date": pd.Timestamp("2021-02-01"), "stock_code": "000001", "target_action": "new_buy", "target_weight": 1.0, "execution_status": "unfilled_cash" if unfilled else "executable", "execution_reason": "missing_execution_day_trading_record" if unfilled else "normal_trading_status_and_valid_open_price"},
            {"factor": factor, "signal_date": pd.Timestamp("2021-02-26"), "execution_date": pd.Timestamp("2021-03-01"), "stock_code": "000001", "target_action": "planned_sell", "target_weight": None, "execution_status": "executable", "execution_reason": "normal_trading_status_and_valid_open_price"},
        ])
    benchmark = pd.DataFrame({"trade_date": pd.date_range("2021-02-01", "2021-03-01", freq="B"), "total_return": 0.0, "quality_flags": ""})
    return pd.DataFrame(targets), pd.DataFrame(actions), benchmark


def test_entry_return_subsequent_return_and_final_signal_exclusion() -> None:
    targets, actions, benchmark = _inputs()
    output = build_backtest_outputs(targets, actions, _daily(), benchmark, BacktestConfig(Path("data/processed"), "1"))
    pe = output.daily.loc[output.daily.factor.eq("pe")].set_index("trade_date")
    assert pe.loc[pd.Timestamp("2021-02-01"), "nav"] == pytest.approx(1.1)
    assert pe.loc[pd.Timestamp("2021-02-02"), "nav"] == pytest.approx(1.21)
    assert output.rebalance_records["trade_date"].max() == pd.Timestamp("2021-03-01")
    assert output.rebalance_records.loc[output.rebalance_records.event_type.eq("sell"), "status"].eq("executed").all()


def test_unfilled_buy_remains_cash() -> None:
    targets, actions, benchmark = _inputs(unfilled=True)
    output = build_backtest_outputs(targets, actions, _daily(), benchmark, BacktestConfig(Path("data/processed"), "1"))
    assert output.daily["nav"].eq(1).all()
    buys = output.rebalance_records.loc[output.rebalance_records.event_type.eq("buy")]
    assert buys["executed_open_value"].eq(0).all()


def test_pending_sale_recovers_and_missing_return_freezes_value() -> None:
    targets, actions, benchmark = _inputs()
    actions.loc[actions.target_action.eq("planned_sell"), "execution_date"] = pd.Timestamp("2021-02-02")
    data = _daily()
    data.loc[data.trade_date.eq(pd.Timestamp("2021-02-02")), "total_return"] = None
    output = build_backtest_outputs(targets, actions, data, benchmark, BacktestConfig(Path("data/processed"), "1"))
    sells = output.rebalance_records.loc[output.rebalance_records.event_type.eq("sell")]
    assert "pending_sale" in set(sells.status)
    assert sells.loc[sells.trade_date.eq(pd.Timestamp("2021-02-03")), "status"].eq("executed").any()
    pe = output.daily.loc[output.daily.factor.eq("pe")].set_index("trade_date")
    assert pe.loc[pd.Timestamp("2021-02-02"), "nav"] == pytest.approx(1.1)


def test_cash_constrained_new_buy_is_not_financed_or_delayed() -> None:
    targets, actions, benchmark = _inputs()
    extra = []
    for factor in FACTORS:
        extra.append({"factor": factor, "signal_date": pd.Timestamp("2021-02-01"), "execution_date": pd.Timestamp("2021-02-02"), "stock_code": "000002", "target_action": "new_buy", "target_weight": 1.0, "execution_status": "executable", "execution_reason": "normal_trading_status_and_valid_open_price"})
    actions = pd.concat([actions, pd.DataFrame(extra)], ignore_index=True)
    data = pd.concat([_daily(), pd.DataFrame([{"stock_code": "000002", "trade_date": pd.Timestamp("2021-02-02"), "open_price": 100.0, "close_price": 100.0, "total_return": 0.0, "PreClosePrice": "100", "trading_status": "1", "quality_flags": ""}])], ignore_index=True)
    output = build_backtest_outputs(targets, actions, data, benchmark, BacktestConfig(Path("data/processed"), "1"))
    constrained = output.rebalance_records.loc[(output.rebalance_records.stock_code.eq("000002")) & (output.rebalance_records.event_type.eq("buy"))]
    assert constrained.status.eq("partially_filled_cash_constrained").all()
    assert constrained.executed_open_value.eq(0).all()
