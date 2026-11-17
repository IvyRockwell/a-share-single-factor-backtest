"""Tests for the confirmed Stage 3 point-in-time universe rules."""

from pathlib import Path

import pandas as pd

from src.universe import UniverseConfig, build_historical_universe


def config() -> UniverseConfig:
    return UniverseConfig(Path("processed"), pd.Timestamp("2021-03-01"), pd.Timestamp("2021-04-30"), frozenset({"1", "4", "16", "32"}), 6, "1")


def test_historical_universe_applies_confirmed_rules_without_future_lookahead() -> None:
    listings = pd.DataFrame({
        "stock_code": ["000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "600701"],
        "LISTINGDATE": ["2020-01-01", "2020-10-01", "2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"],
    })
    rows = []
    for date in pd.to_datetime(["2021-03-31", "2021-04-30"]):
        for code in ["000001", "000002", "000003", "000004", "000006", "000007", "000008", "000009", "600701"]:
            rows.append({"stock_code": code, "trade_date": date, "market_type": "1", "trading_status": "1", "quality_flags": ""})
    daily = pd.DataFrame(rows)
    daily.loc[(daily.stock_code == "000003") & (daily.trade_date == pd.Timestamp("2021-03-31")), "trading_status"] = "2"
    daily.loc[(daily.stock_code == "000004") & (daily.trade_date == pd.Timestamp("2021-03-31")), "market_type"] = "64"
    status = pd.DataFrame({"stock_code": ["000006", "000007", "000008", "000009"], "ChangedItem": ["上市状态"] * 4,
                           "event_date": pd.to_datetime(["2021-03-01", "2021-03-01", "2021-03-01", "2021-04-01"]),
                           "ValueAfter": ["终止上市", "暂停上市", "退市整理期", "终止上市"]})
    calendar = pd.DataFrame({"market_type": ["1", "1"], "State": ["O", "O"], "calendar_date": pd.to_datetime(["2021-03-31", "2021-04-30"])})

    outputs = build_historical_universe(daily, listings, status, calendar, config())
    march = set(outputs.pool.loc[outputs.pool.signal_date.eq(pd.Timestamp("2021-03-31")), "stock_code"])
    april = set(outputs.pool.loc[outputs.pool.signal_date.eq(pd.Timestamp("2021-04-30")), "stock_code"])
    assert "000001" in march
    assert "000009" in march  # Termination is only effective in the next month.
    assert "000009" not in april
    assert "600701" not in march
    assert march.isdisjoint({"000002", "000003", "000004", "000005", "000006", "000007", "000008"})

    reasons = outputs.exclusions.set_index(["signal_date", "stock_code"])["exclusion_reasons"]
    assert "not_listed_for_six_calendar_months" in reasons.loc[(pd.Timestamp("2021-03-31"), "000002")]
    assert "non_normal_trading_status" in reasons.loc[(pd.Timestamp("2021-03-31"), "000003")]
    assert "outside_candidate_market" in reasons.loc[(pd.Timestamp("2021-03-31"), "000004")]
    assert "missing_signal_day_trading_record" in reasons.loc[(pd.Timestamp("2021-03-31"), "000005")]
    assert "terminated_listing_status" in reasons.loc[(pd.Timestamp("2021-03-31"), "000006")]
    assert "suspended_listing_status" in reasons.loc[(pd.Timestamp("2021-03-31"), "000007")]
    assert "delisting_arrangement_status" in reasons.loc[(pd.Timestamp("2021-03-31"), "000008")]
    assert "status_conflict_interval" in reasons.loc[(pd.Timestamp("2021-03-31"), "600701")]
    assert not outputs.pool.duplicated(["signal_date", "stock_code"]).any()
