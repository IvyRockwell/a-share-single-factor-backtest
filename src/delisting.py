"""Delisting and absorption-merger events and their return treatment.

CSMAR ``STK_ITEMCHANGE`` records a *status* change, not the reason behind it.
Two economically different situations therefore share the single value
``终止上市``: a genuine delisting that wipes out the equity, and an absorption
merger that usually pays a consideration to the holders.  The research pipeline
cannot separate them from the fields that are available, so every event is
written off with the same configurable assumption and the assumption is
reported alongside the result.  0% is the optimistic merger boundary and
-100% is the conservative delisting boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DELISTING_ITEM = "上市状态"
DELISTING_VALUE = "终止上市"
VALID_DELISTING_RETURNS = (-1.0, 0.0)


def load_delisting_events(status_changes_path: Path | str) -> pd.DataFrame:
    """Return one ``stock_code`` / ``delisting_date`` row per terminated listing.

    Only ``ChangedItem == "上市状态"`` and ``ValueAfter == "终止上市"`` are
    treated as write-off events: suspension, delisting arrangement, and
    relisting are status changes but not terminations.  Repeated termination
    events keep the earliest date.
    """
    frame = pd.read_parquet(
        Path(status_changes_path),
        columns=["stock_code", "event_date", "ChangedItem", "ValueAfter"],
    )
    events = frame.loc[
        frame["ChangedItem"].astype("string").eq(DELISTING_ITEM)
        & frame["ValueAfter"].astype("string").eq(DELISTING_VALUE),
        ["stock_code", "event_date"],
    ].copy()
    events["stock_code"] = events["stock_code"].astype("string")
    events["delisting_date"] = pd.to_datetime(events["event_date"]).astype("datetime64[ns]")
    events = events.dropna(subset=["delisting_date"])
    if events.empty:
        return pd.DataFrame({
            "stock_code": pd.Series(dtype="string"),
            "delisting_date": pd.Series(dtype="datetime64[ns]"),
        })
    events = events.groupby("stock_code", as_index=False)["delisting_date"].min()
    return events.sort_values(["stock_code"]).reset_index(drop=True)


def validate_delisting_return(value: float) -> float:
    """The write-off assumption has to lie between total loss and no loss."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("delisting_return must be a number.") from exc
    if not np.isfinite(converted) or not (
        VALID_DELISTING_RETURNS[0] <= converted <= VALID_DELISTING_RETURNS[1]
    ):
        raise ValueError("delisting_return must be between -1.0 and 0.0.")
    return converted


def apply_forward_delisting(
    forward: pd.DataFrame,
    events: pd.DataFrame | None,
    assumption: float = -1.0,
) -> pd.DataFrame:
    """Mark and adjust forward returns for a termination inside the holding window.

    A write-off only applies when the termination date falls in
    ``(signal_date, holding_end_date]`` *and* the last observed valid trading day
    is earlier than the holding end: the stock stopped trading inside the
    window.  A termination before the first holding day leaves the row
    ineligible instead of adding a return.
    """
    assumption = validate_delisting_return(assumption)
    result = forward.copy()
    result = result.drop(columns=[column for column in ("delisting_date",) if column in result.columns])
    if events is None or events.empty:
        result["delisting_date"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    else:
        lookup = events[["stock_code", "delisting_date"]].copy()
        lookup["stock_code"] = lookup["stock_code"].astype("string")
        lookup["delisting_date"] = pd.to_datetime(lookup["delisting_date"]).astype("datetime64[ns]")
        lookup = lookup.groupby("stock_code", as_index=False)["delisting_date"].min()
        result = result.merge(lookup, on="stock_code", how="left", validate="many_to_one")
    if "holding_end_date" not in result.columns:
        result["delisting_event"] = False
        return result
    in_window = (
        result["delisting_date"].notna()
        & result["holding_end_date"].notna()
        & result["delisting_date"].gt(result["signal_date"])
        & result["delisting_date"].le(result["holding_end_date"])
        & result["last_observed_date"].notna()
        & result["last_observed_date"].lt(result["holding_end_date"])
    )
    result["delisting_event"] = in_window.astype(bool)
    marked = result.loc[result["delisting_event"], "forward_return"]
    if marked.notna().any():
        result.loc[result["delisting_event"], "forward_return"] = (
            (1.0 + marked) * (1.0 + assumption) - 1.0
        )
    return result


def validate_forward_delisting(forward: pd.DataFrame) -> None:
    """Fail when an eligible write-off row was not given a return."""
    if "delisting_event" not in forward.columns:
        raise ValueError("Forward returns are missing the delisting_event column.")
    marked = forward.loc[forward["delisting_event"].fillna(False)]
    if marked.empty:
        return
    if marked["delisting_date"].isna().any():
        raise ValueError("A marked delisting event has no delisting date.")
    eligible = marked.loc[marked["forward_eligible"].fillna(False)]
    if eligible["forward_return"].isna().any():
        raise ValueError("An eligible delisting-window row has no adjusted forward return.")
