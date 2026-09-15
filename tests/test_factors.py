"""Tests for the confirmed Stage 4 factor rules."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.factors import FactorConfig, build_factor_details, build_quality_summary, validate_outputs, FactorOutputs


CONFIG = FactorConfig(Path("processed"), 6)


def _pool() -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-07-30"] * 4),
        "stock_code": ["000001", "000002", "000003", "000004"],
    })


def _valuation() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["000001", "000002", "000003", "000004"],
        "trading_date": pd.to_datetime(["2021-07-30"] * 4),
        "pe_ttm": [10.0, 20.0, -1.0, None],
        "quality_flags": ["", "", "nonpositive_pe_ttm", "missing_pe_ttm"],
    })


def _daily() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["000001", "000001", "000002", "000002", "000003", "000003", "000004", "000004"],
        "trade_date": pd.to_datetime(["2021-01-29", "2021-07-30", "2021-01-28", "2021-07-30", "2021-01-29", "2021-07-30", "2021-01-29", "2021-07-30"]),
        "adjusted_close": [10.0, 12.0, 10.0, 11.0, 0.0, 12.0, 10.0, None],
        "quality_flags": ["", "", "", "", "", "", "", "invalid_adjusted_close"],
    })


def _financials() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["000001", "000001", "000002", "000002", "000003", "000004"],
        "report_date": pd.to_datetime(["2019-12-31", "2020-12-31", "2019-12-31", "2020-12-31", "2020-12-31", "2020-12-31"]),
        "Typrep": ["A"] * 6,
        "actual_disclosure_date": pd.to_datetime(["2020-03-01", "2021-03-01", "2021-07-30", "2021-03-01", "2021-03-01", "2021-03-01"]),
        "net_profit": [10.0, 20.0, 30.0, 20.0, -10.0, 10.0],
        "shareholder_equity": [100.0, 100.0, 100.0, 100.0, 100.0, -1.0],
        "invalid_actual_disclosure_date": [False] * 6,
    })


def test_factor_rules_cover_ranking_momentum_and_roe_availability() -> None:
    details = build_factor_details(_pool(), _valuation(), _daily(), _financials(), CONFIG)
    rows = details.set_index("stock_code")

    assert rows.loc["000001", "pe_rank"] == 1
    assert rows.loc["000001", "pe_score"] == 1
    assert rows.loc["000002", "pe_rank"] == 2
    assert rows.loc["000003", "pe_unrankable_reason"] == "nonpositive_pe_ttm"
    assert rows.loc["000004", "pe_unrankable_reason"] == "missing_or_invalid_pe_ttm"

    assert rows.loc["000002", "momentum_start_date"] == pd.Timestamp("2021-01-28")
    assert rows.loc["000001", "momentum_6m_return"] == pytest.approx(0.2)
    assert rows.loc["000001", "momentum_rank"] == 1
    assert rows.loc["000003", "momentum_unrankable_reason"] == "no_valid_price_on_or_before_lookback_date"
    assert rows.loc["000004", "momentum_unrankable_reason"] == "missing_or_invalid_signal_adjusted_close"

    assert rows.loc["000001", "roe_report_date"] == pd.Timestamp("2020-12-31")
    assert rows.loc["000001", "roe"] == 0.2
    assert rows.loc["000002", "roe_report_date"] == pd.Timestamp("2020-12-31")
    assert rows.loc["000002", "roe_actual_disclosure_date"] == pd.Timestamp("2021-03-01")
    assert rows.loc["000003", "roe"] == -0.1
    assert rows.loc["000004", "roe_unrankable_reason"] == "missing_or_nonpositive_shareholder_equity"


def test_quality_summary_reconciles_every_factor_independently() -> None:
    details = build_factor_details(_pool(), _valuation(), _daily(), _financials(), CONFIG)
    quality = build_quality_summary(details)
    validate_outputs(FactorOutputs(details, quality), _pool())
    assert set(quality["factor"]) == {"pe", "momentum", "roe"}
    assert quality.loc[quality["factor"].eq("pe"), "positive_pe_max"].iloc[0] == 20
