"""Tests for the point-in-time market-cap build."""

from pathlib import Path

import pandas as pd
import pytest

from src.market_cap import (
    MarketCapConfig,
    MarketCapOutputs,
    attach_shares,
    build_market_cap,
    validate_outputs,
)

POOL = pd.DataFrame({
    "signal_date": pd.to_datetime(["2021-01-29", "2021-01-29", "2021-02-26", "2021-02-26"]),
    "stock_code": ["000001", "000002", "000001", "000002"],
})
CAPITAL = pd.DataFrame({
    "stock_code": ["000001", "000001", "000002"],
    "effective_date": pd.to_datetime(["2020-01-02", "2021-02-01", "2020-06-01"]),
    "total_shares": [1.0e8, 2.0e8, 5.0e7],
    "float_shares": [0.8e8, 1.6e8, 4.0e7],
})
PRICES = pd.DataFrame({
    "stock_code": ["000001", "000002", "000001", "000002"],
    "signal_date": pd.to_datetime(["2021-01-29", "2021-01-29", "2021-02-26", "2021-02-26"]),
    "close_price": [10.0, 20.0, 11.0, 19.0],
})


def _config() -> MarketCapConfig:
    return MarketCapConfig(Path("data/processed"))


def test_attach_shares_takes_the_latest_record_on_or_before_the_signal_date() -> None:
    frame = attach_shares(POOL, CAPITAL).set_index(["stock_code", "signal_date"])
    assert frame.loc[("000001", pd.Timestamp("2021-01-29")), "total_shares"] == 1.0e8
    assert frame.loc[("000001", pd.Timestamp("2021-02-26")), "total_shares"] == 2.0e8
    assert frame.loc[("000002", pd.Timestamp("2021-02-26")), "shares_effective_date"] == pd.Timestamp("2020-06-01")


def test_market_cap_is_total_shares_times_close() -> None:
    frame = build_market_cap(POOL, CAPITAL, PRICES).set_index(["stock_code", "signal_date"])
    assert frame.loc[("000001", pd.Timestamp("2021-01-29")), "market_cap"] == pytest.approx(1.0e9)
    assert frame.loc[("000001", pd.Timestamp("2021-02-26")), "market_cap"] == pytest.approx(2.2e9)
    assert frame.loc[("000002", pd.Timestamp("2021-02-26")), "float_market_cap"] == pytest.approx(7.6e8)
    assert frame["has_market_cap"].all()


def test_missing_price_leaves_the_row_uncovered() -> None:
    prices = PRICES.loc[~((PRICES["stock_code"].eq("000002")) & (PRICES["signal_date"].eq(pd.Timestamp("2021-02-26"))))]
    frame = build_market_cap(POOL, CAPITAL, prices)
    uncovered = frame.loc[frame["stock_code"].eq("000002") & frame["signal_date"].eq(pd.Timestamp("2021-02-26"))].iloc[0]
    assert not uncovered["has_market_cap"]
    assert pd.isna(uncovered["log_market_cap"])


def test_low_coverage_is_rejected() -> None:
    outputs = MarketCapOutputs(build_market_cap(POOL, CAPITAL.iloc[:1], PRICES))
    with pytest.raises(ValueError, match="covers only"):
        validate_outputs(outputs, POOL)
