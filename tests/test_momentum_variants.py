"""Tests for the 12-1 / 6-1 / 1M momentum variant step."""

from pathlib import Path

import pandas as pd
import pytest

from src.factor_test import FactorTestConfig
from src.factors import FactorConfig, MomentumVariant
from src.momentum_variants import build_variant_outputs


def _details() -> pd.DataFrame:
    rows = []
    for signal_date in pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"]):
        for stock_code, score in [("000001", 0.2), ("000002", 0.8)]:
            row = {"signal_date": signal_date, "stock_code": stock_code}
            for factor in ("momentum", "momentum_6_1", "momentum_1m"):
                row[f"{factor}_score"] = score
                row[f"{factor}_is_rankable"] = True
                row[f"{factor}_unrankable_reason"] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


def _forward() -> pd.DataFrame:
    rows = []
    for signal_date in pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"]):
        for stock_code, value in [("000001", -0.05), ("000002", 0.10)]:
            rows.append({
                "signal_date": signal_date, "stock_code": stock_code,
                "forward_return": value, "forward_eligible": True,
                "forward_unavailable_reason": pd.NA, "early_exit_or_data_cutoff": False,
                "extreme_return_days": 0,
            })
    return pd.DataFrame(rows)


def _factor_config() -> FactorConfig:
    return FactorConfig(
        Path("processed"), 12, 1,
        (MomentumVariant("momentum_6_1", 6, 1), MomentumVariant("momentum_1m", 1, 0, False)),
    )


def test_variant_outputs_cover_primary_6_1_and_1m() -> None:
    outputs = build_variant_outputs(
        _details(), _forward(), _factor_config(),
        FactorTestConfig(Path("processed"), 5, 1, ()),
    )
    assert set(outputs.summary["factor"]) == {"momentum", "momentum_6_1", "momentum_1m"}
    assert len(outputs.ic) == 3 * 3
    assert outputs.summary["ic_valid_periods"].eq(3).all()
    assert outputs.summary.loc[outputs.summary["factor"].eq("momentum_1m"), "ic_mean"].iloc[0] > 0
