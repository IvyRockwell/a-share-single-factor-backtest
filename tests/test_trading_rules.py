"""Tests for A-share limit-up/down and execution-block rules."""

from __future__ import annotations

import pandas as pd

from src.trading_rules import limit_prices, limit_rate, trade_block_reason


def test_limit_rates_follow_board_and_chinext_reform() -> None:
    assert limit_rate("2021-01-04", "1") == 0.10
    assert limit_rate("2021-01-04", "4") == 0.10
    assert limit_rate("2020-08-21", "16") == 0.10
    assert limit_rate("2020-08-24", "16") == 0.20
    assert limit_rate("2021-01-04", "32") == 0.20
    assert limit_rate("2021-01-04", "8") is None


def test_limit_prices_are_rounded_to_cents() -> None:
    assert limit_prices(10.0, "2021-01-04", "1") == (11.0, 9.0)
    assert limit_prices(10.0, "2021-01-04", "16") == (12.0, 8.0)
    assert limit_prices(None, "2021-01-04", "1") == (None, None)


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "trade_date": pd.Timestamp("2021-01-04"),
        "trading_status": "1",
        "open_price": 10.0,
        "pre_close": 10.0,
        "market_type": "1",
        "quality_flags": "",
    }
    row.update(overrides)
    return row


def test_opening_limit_blocks_the_correct_side() -> None:
    assert trade_block_reason(_row(open_price=11.0), "1", "buy") == "limit_up_open"
    assert trade_block_reason(_row(open_price=11.0), "1", "sell") is None
    assert trade_block_reason(_row(open_price=9.0), "1", "sell") == "limit_down_open"
    assert trade_block_reason(_row(open_price=9.0), "1", "buy") is None


def test_missing_limit_fields_do_not_block_and_status_precedes_limit() -> None:
    assert trade_block_reason(_row(pre_close=None), "1", "buy") is None
    assert trade_block_reason(_row(market_type=None), "1", "buy") is None
    assert trade_block_reason(_row(trading_status="2", open_price=11.0), "1", "buy") == "non_normal_trading_status"
    assert trade_block_reason(_row(open_price=0.0), "1", "buy") == "invalid_open_price"
