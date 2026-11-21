"""Tests for the delisting event table and the forward-return adjustment."""

from pathlib import Path

import pandas as pd
import pytest

from src.delisting import (
    apply_forward_delisting,
    load_delisting_events,
    validate_delisting_return,
    validate_forward_delisting,
)


def _status_changes(tmp_path: Path) -> Path:
    rows = [
        ("000001", "2021-02-10", "上市状态", "正常上市", "终止上市"),
        ("000001", "2021-01-05", "上市状态", "正常上市", "终止上市"),
        ("000002", "2021-03-01", "上市状态", "正常上市", "暂停上市"),
        ("000003", "2021-04-01", "上市状态", "正常上市", "退市整理期"),
        ("000004", "2021-05-01", "证券简称", "A", "B"),
        ("000005", "2021-07-01", "上市状态", "正常上市", "终止上市"),
        ("000005", "2021-06-01", "上市状态", "正常上市", "终止上市"),
    ]
    frame = pd.DataFrame(rows, columns=["stock_code", "event_date", "ChangedItem", "ValueBefore", "ValueAfter"])
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    path = tmp_path / "status_changes.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_load_delisting_events_keeps_only_terminations_and_the_earliest_date(tmp_path: Path) -> None:
    events = load_delisting_events(_status_changes(tmp_path)).set_index("stock_code")
    assert set(events.index) == {"000001", "000005"}
    assert events.loc["000001", "delisting_date"] == pd.Timestamp("2021-01-05")
    assert events.loc["000005", "delisting_date"] == pd.Timestamp("2021-06-01")
    assert events["delisting_date"].dtype == "datetime64[ns]"


def test_load_delisting_events_returns_an_empty_table_without_events(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "stock_code": ["000001"],
        "event_date": pd.to_datetime(["2021-01-05"]),
        "ChangedItem": ["上市状态"],
        "ValueAfter": ["暂停上市"],
    })
    path = tmp_path / "status_changes.parquet"
    frame.to_parquet(path, index=False)
    events = load_delisting_events(path)
    assert events.empty
    assert list(events.columns) == ["stock_code", "delisting_date"]


def _forward() -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-29", "2021-01-29"]),
        "stock_code": ["A", "B"],
        "holding_end_date": pd.to_datetime(["2021-02-26", "2021-02-26"]),
        "first_holding_date": pd.to_datetime(["2021-02-01", "2021-02-01"]),
        "last_observed_date": pd.to_datetime(["2021-02-10", "2021-02-26"]),
        "forward_return": [0.10, 0.20],
        "forward_eligible": [True, True],
    })


def _events() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["A", "B"],
        "delisting_date": pd.to_datetime(["2021-02-15", "2021-03-05"]),
    })


def test_forward_delisting_adjusts_only_the_window_that_stopped_trading() -> None:
    adjusted = apply_forward_delisting(_forward(), _events(), assumption=-0.5).set_index("stock_code")
    assert bool(adjusted.loc["A", "delisting_event"])
    assert adjusted.loc["A", "delisting_date"] == pd.Timestamp("2021-02-15")
    assert adjusted.loc["A", "forward_return"] == pytest.approx((1.0 + 0.10) * (1.0 - 0.5) - 1.0)
    assert not bool(adjusted.loc["B", "delisting_event"])
    assert adjusted.loc["B", "forward_return"] == pytest.approx(0.20)


def test_forward_delisting_assumption_moves_between_the_two_boundaries() -> None:
    total_loss = apply_forward_delisting(_forward(), _events(), assumption=-1.0).set_index("stock_code")
    no_loss = apply_forward_delisting(_forward(), _events(), assumption=0.0).set_index("stock_code")
    assert total_loss.loc["A", "forward_return"] == pytest.approx(-1.0)
    assert no_loss.loc["A", "forward_return"] == pytest.approx(0.10)


def test_forward_delisting_without_events_only_adds_empty_columns() -> None:
    result = apply_forward_delisting(_forward(), None, assumption=-1.0)
    assert not result["delisting_event"].any()
    assert result["delisting_date"].isna().all()
    assert result.loc[result["stock_code"].eq("A"), "forward_return"].iloc[0] == pytest.approx(0.10)


def test_delisting_assumption_must_stay_between_minus_one_and_zero() -> None:
    assert validate_delisting_return(-1.0) == -1.0
    assert validate_delisting_return(0.0) == 0.0
    with pytest.raises(ValueError, match="between"):
        validate_delisting_return(-1.5)
    with pytest.raises(ValueError, match="between"):
        validate_delisting_return(0.5)
    with pytest.raises(ValueError, match="number"):
        validate_delisting_return("bad")


def test_validate_forward_delisting_requires_a_return_for_eligible_rows() -> None:
    marked = apply_forward_delisting(_forward(), _events(), assumption=-1.0)
    marked.loc[marked["stock_code"].eq("A"), "forward_return"] = float("nan")
    with pytest.raises(ValueError, match="adjusted forward return"):
        validate_forward_delisting(marked)
    validate_forward_delisting(apply_forward_delisting(_forward(), _events(), assumption=-1.0))
