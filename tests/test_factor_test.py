"""Tests for the forward-return and factor-test rules."""

from pathlib import Path

import numpy as np
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
    small = pd.DataFrame({"factor": ["pe"] * 3, "score_type": ["raw"] * 3, "signal_date": pd.to_datetime(["2021-01-29"] * 3), "test_eligible": [True] * 3, "factor_score": [0.1, 0.2, 0.3], "forward_return": [0.1, 0.2, 0.3]})
    result = build_ic_series(small, 4).iloc[0]
    assert result["ic_unavailable_reason"] == "insufficient_valid_pairs"
    enough = pd.concat([small, small.assign(signal_date=pd.Timestamp("2021-02-26"))], ignore_index=True)
    result = build_ic_series(enough, 3)
    assert result["rank_ic"].dropna().eq(1).all()

def test_newey_west_lags_follows_the_sample_size() -> None:
    from src.factor_test import newey_west_lags

    assert newey_west_lags(28) == 3
    assert newey_west_lags(135) == 4
    assert newey_west_lags(1000) == 6


def test_newey_west_matches_the_plain_t_on_white_noise() -> None:
    from src.factor_test import newey_west_t

    rng = np.random.default_rng(20260930)
    values = pd.Series(rng.normal(0.02, 0.1, 600))
    plain = values.mean() / values.std(ddof=1) * np.sqrt(len(values))
    assert newey_west_t(values) == pytest.approx(plain, rel=0.1)


def test_newey_west_shrinks_the_t_statistic_under_positive_autocorrelation() -> None:
    from src.factor_test import newey_west_t

    rng = np.random.default_rng(11)
    shocks = rng.normal(0.0, 0.1, 800)
    values = np.zeros_like(shocks)
    for index in range(1, shocks.size):
        values[index] = 0.6 * values[index - 1] + shocks[index]
    series = pd.Series(values + 0.01)
    plain = abs(series.mean() / series.std(ddof=1) * np.sqrt(len(series)))
    assert 0 < abs(newey_west_t(series)) < plain


def test_newey_west_returns_nan_when_the_variance_is_degenerate() -> None:
    from src.factor_test import newey_west_t

    assert pd.isna(newey_west_t(pd.Series([0.05, 0.05, 0.05, 0.05])))
    assert pd.isna(newey_west_t(pd.Series([0.01, 0.02])))


def _subperiod_frames() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    dates = pd.date_range("2021-01-31", periods=12, freq="ME")
    ic = pd.DataFrame({
        "factor": "pe", "score_type": "raw", "signal_date": dates,
        "rank_ic": np.linspace(-0.1, 0.1, 12),
    })
    quantiles = pd.DataFrame({
        "factor": "pe", "score_type": "raw", "signal_date": dates,
        "quantile": 1, "q_high_minus_q_low": np.linspace(-0.02, 0.02, 12),
    })
    bounds = [
        ("Q1", pd.Timestamp("2021-01-01"), pd.Timestamp("2021-04-30")),
        ("Q2", pd.Timestamp("2021-05-01"), pd.Timestamp("2021-08-31")),
        ("Q3", pd.Timestamp("2021-09-01"), pd.Timestamp("2021-12-31")),
    ]
    return ic, quantiles, bounds


def test_subperiods_partition_the_sample() -> None:
    from src.factor_test import build_subperiod_table

    ic, quantiles, bounds = _subperiod_frames()
    table = build_subperiod_table(quantiles, ic, tuple(bounds))
    assert table["periods"].sum() == 12
    assert table["periods"].tolist() == [4, 4, 4]


def test_subperiod_bounds_are_inclusive() -> None:
    from src.factor_test import build_subperiod_table

    ic, quantiles, _ = _subperiod_frames()
    bounds = (("only", pd.Timestamp("2021-01-31"), pd.Timestamp("2021-04-30")),)
    table = build_subperiod_table(quantiles, ic, bounds)
    assert table["periods"].tolist() == [4]


def test_summary_reports_newey_west_columns() -> None:
    from src.factor_test import build_summary

    details = pd.DataFrame({
        "factor": ["pe"] * 4, "score_type": ["raw"] * 4,
        "signal_date": pd.date_range("2021-01-31", periods=4, freq="ME"),
        "test_eligible": [True] * 4,
        "test_unavailable_reason": pd.Series([pd.NA] * 4, dtype="string"),
        "early_exit_or_data_cutoff": [False] * 4,
        "extreme_return_days": [0] * 4,
    })
    ic = pd.DataFrame({
        "factor": "pe", "score_type": "raw",
        "signal_date": details["signal_date"], "rank_ic": [0.05, 0.02, 0.08, 0.03],
    })
    quantiles = pd.DataFrame({
        "factor": "pe", "score_type": "raw", "signal_date": details["signal_date"],
        "quantile": 1, "q_high_minus_q_low": [0.01, 0.0, 0.02, 0.005], "empty_group": False,
    })
    summary = build_summary(details, quantiles, ic)
    row = summary.iloc[0]
    assert {"ic_t_nw", "spread_t_nw", "nw_lags"} <= set(summary.columns)
    assert pd.notna(row["ic_t_nw"]) and pd.notna(row["spread_t_nw"])
    assert row["nw_lags"] >= 1

def test_pe_source_comparison_negates_so_low_pe_is_positive() -> None:
    from src.factor_test import build_pe_source_comparison

    dates = pd.to_datetime(["2021-01-31", "2021-02-28"])
    codes = ["000001", "000002", "000003", "000004"]
    factor_details = pd.DataFrame({
        "signal_date": dates.repeat(4),
        "stock_code": codes * 2,
        "pe_pit": [1.0, 2.0, 3.0, 4.0] * 2,
        "pe_ttm": [1.0, 2.0, 3.0, 4.0] * 2,
    })
    forward_returns = pd.DataFrame({
        "signal_date": dates.repeat(4),
        "stock_code": codes * 2,
        "forward_return": [0.04, 0.03, 0.02, 0.01] * 2,
        "forward_eligible": True,
    })

    comparison = build_pe_source_comparison(factor_details, forward_returns, min_pairs=3)

    assert comparison["pit_ic"].tolist() == pytest.approx([1.0, 1.0])
    assert comparison["vendor_ic"].tolist() == pytest.approx([1.0, 1.0])
    assert comparison["ic_gap"].abs().max() < 1e-12


def test_pe_source_comparison_skips_thin_cross_sections() -> None:
    from src.factor_test import build_pe_source_comparison

    dates = pd.to_datetime(["2021-01-31"])
    factor_details = pd.DataFrame({
        "signal_date": dates, "stock_code": ["000001"], "pe_pit": [5.0], "pe_ttm": [5.0],
    })
    forward_returns = pd.DataFrame({
        "signal_date": dates, "stock_code": ["000001"], "forward_return": [0.02], "forward_eligible": True,
    })
    comparison = build_pe_source_comparison(factor_details, forward_returns, min_pairs=50)
    assert pd.isna(comparison["pit_ic"].iloc[0])


def test_attach_pit_check_labels_the_look_ahead_months() -> None:
    from src.factor_test import attach_pit_check

    comparison = pd.DataFrame({"signal_date": pd.to_datetime(["2021-01-31", "2021-06-30"]), "pit_ic": [0.01, 0.02]})
    check = pd.DataFrame({"signal_date": pd.to_datetime(["2021-01-31", "2021-06-30"]), "share_within_1pct": [0.05, 1.0]})
    merged = attach_pit_check(comparison, check)
    assert merged["vendor_ahead_of_disclosure"].tolist() == [True, False]
