"""Tests for Stage 8 costs derived from actual Stage 7 fills."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.costs import CostConfig, build_cost_outputs, validate_outputs


def _config() -> CostConfig:
    return CostConfig(Path("data/processed"), 0.0003, ((pd.Timestamp("2014-01-01"), 0.001), (pd.Timestamp("2023-08-28"), 0.0005)), (0, 10, 20))


def _daily() -> pd.DataFrame:
    return pd.DataFrame([
        {"factor": "pe", "trade_date": "2023-08-25", "gross_return": 0.10, "nav": 1.10, "benchmark_return": 0.0, "benchmark_nav": 1.0},
        {"factor": "pe", "trade_date": "2023-08-28", "gross_return": 0.00, "nav": 1.10, "benchmark_return": 0.0, "benchmark_nav": 1.0},
        {"factor": "pe", "trade_date": "2023-08-29", "gross_return": 0.00, "nav": 1.10, "benchmark_return": 0.0, "benchmark_nav": 1.0},
    ]).assign(trade_date=lambda frame: pd.to_datetime(frame.trade_date))


def _records() -> pd.DataFrame:
    return pd.DataFrame([
        {"factor": "pe", "trade_date": "2023-08-25", "event_type": "buy", "status": "executed", "executed_open_value": 0.5},
        {"factor": "pe", "trade_date": "2023-08-25", "event_type": "sell", "status": "executed", "executed_open_value": 0.2},
        {"factor": "pe", "trade_date": "2023-08-28", "event_type": "sell", "status": "executed", "executed_open_value": 0.5},
        {"factor": "pe", "trade_date": "2023-08-29", "event_type": "buy", "status": "unfilled_cash", "executed_open_value": 1.0},
        {"factor": "pe", "trade_date": "2023-08-29", "event_type": "sell", "status": "pending_sale", "executed_open_value": 1.0},
    ]).assign(trade_date=lambda frame: pd.to_datetime(frame.trade_date))


def test_costs_use_only_actual_fills_and_apply_each_cost_direction() -> None:
    outputs = build_cost_outputs(_daily(), _records(), _config())
    first = outputs.costs.loc[(outputs.costs.trade_date.eq(pd.Timestamp("2023-08-25"))) & outputs.costs.slippage_bp.eq(10)].iloc[0]
    assert first.actual_buy_amount == pytest.approx(0.5)
    assert first.actual_sell_amount == pytest.approx(0.2)
    assert first.turnover == pytest.approx(0.7)
    assert first.commission == pytest.approx(0.7 * 0.0003)
    assert first.stamp_duty == pytest.approx(0.2 * 0.001)
    assert first.slippage == pytest.approx(0.7 * 0.001)
    assert not outputs.costs.trade_date.eq(pd.Timestamp("2023-08-29")).any()


def test_historical_stamp_duty_change_and_net_results_are_below_gross() -> None:
    outputs = build_cost_outputs(_daily(), _records(), _config())
    post_change = outputs.costs.loc[(outputs.costs.trade_date.eq(pd.Timestamp("2023-08-28"))) & outputs.costs.slippage_bp.eq(0)].iloc[0]
    assert post_change.stamp_duty == pytest.approx(0.5 * 0.0005)
    assert set(outputs.daily.slippage_bp) == {0, 10, 20}
    assert outputs.daily.net_nav.le(outputs.daily.nav).all()
    assert outputs.daily.net_return.le(outputs.daily.gross_return).all()
    validate_outputs(outputs)
