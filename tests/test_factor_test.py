"""Tests for the confirmed Stage 5 forward-return and factor-test rules."""

from pathlib import Path

import pandas as pd
import pytest

from src.factor_test import (
    assign_quantiles,
    build_factor_test_details,
    build_forward_returns,
    build_forward_returns_streaming,
    build_ic_series,
    build_quantile_returns,
)


def _factor_details() -> pd.DataFrame:
    rows = []
    for signal_date in pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"]):
        for stock_code, score in [("000001", 0.2), ("000002", 0.2), ("000003", 0.9)]:
            row = {"signal_date": signal_date, "stock_code": stock_code}
            for factor in ("pe", "momentum", "roe"):
                row[f"{factor}_score"] = score
                row[f"{factor}_is_rankable"] = True
                row[f"{factor}_unrankable_reason"] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


def _daily() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["000001", "000002", "000003", "000001", "000001"],
        "trade_date": pd.to_datetime(["2021-02-01", "2021-02-26", "2021-02-01", "2021-03-01", "2021-03-31"]),
        "total_return": [0.10, 0.20, 0.05, 0.03, 0.04],
        "quality_flags": ["", "", "", "extreme_total_return", ""],
    })


def test_forward_window_excludes_signal_date_and_records_missing_or_early_exit() -> None:
    forward = build_forward_returns(_factor_details(), _daily()).set_index(["signal_date", "stock_code"])
    first = forward.loc[(pd.Timestamp("2021-01-29"), "000001")]
    assert first["first_holding_date"] == pd.Timestamp("2021-02-01")
    assert first["holding_end_date"] == pd.Timestamp("2021-02-26")
    assert first["forward_return"] == pytest.approx(0.1)
    assert first["forward_eligible"]
    assert forward.loc[(pd.Timestamp("2021-01-29"), "000002"), "forward_unavailable_reason"] == "missing_valid_first_holding_day"
    assert forward.loc[(pd.Timestamp("2021-01-29"), "000003"), "early_exit_or_data_cutoff"]
    assert forward.loc[(pd.Timestamp("2021-03-31"), "000001"), "forward_unavailable_reason"] == "no_future_holding_period"


def test_streaming_forward_returns_matches_in_memory_result(tmp_path: Path) -> None:
    source = tmp_path / "daily.parquet"
    _daily().to_parquet(source, index=False)
    columns = ["signal_date", "stock_code", "forward_return", "forward_unavailable_reason", "early_exit_or_data_cutoff"]
    expected = build_forward_returns(_factor_details(), _daily())[columns]
    actual = build_forward_returns_streaming(_factor_details(), source)[columns]
    pd.testing.assert_frame_equal(actual, expected)


def test_quantiles_keep_ties_together_and_q5_is_high_score() -> None:
    scores = pd.Series([0.2, 0.2, 0.9, 0.9])
    assigned = assign_quantiles(scores, 5)
    assert assigned.iloc[0] == assigned.iloc[1]
    assert assigned.iloc[2] == assigned.iloc[3] == 5
    details = build_factor_test_details(_factor_details(), build_forward_returns(_factor_details(), _daily()), 5)
    members, grouped = build_quantile_returns(details, 5)
    assert members.loc[members["quantile"].eq(5), "factor_score"].min() >= members.loc[members["quantile"].lt(5), "factor_score"].max()
    assert grouped["empty_group"].any()


def test_repeated_q5_minus_q1_cumulative_value_is_identical_across_all_quantiles() -> None:
    details = build_factor_test_details(_factor_details(), build_forward_returns(_factor_details(), _daily()), 5)
    _, grouped = build_quantile_returns(details, 5)
    copies = grouped.groupby(["factor", "signal_date"])["cumulative_q_high_minus_q_low"].nunique(dropna=False)
    assert copies.eq(1).all()


def test_rank_ic_requires_minimum_pairs_and_reports_positive_win() -> None:
    small = pd.DataFrame({"factor": ["pe"] * 3, "signal_date": pd.to_datetime(["2021-01-29"] * 3), "test_eligible": [True] * 3, "factor_score": [0.1, 0.2, 0.3], "forward_return": [0.1, 0.2, 0.3]})
    result = build_ic_series(small, 4).iloc[0]
    assert result["ic_unavailable_reason"] == "insufficient_valid_pairs"
    enough = pd.concat([small, small.assign(signal_date=pd.Timestamp("2021-02-26"))], ignore_index=True)
    result = build_ic_series(enough, 3)
    assert result["rank_ic"].dropna().eq(1).all()
