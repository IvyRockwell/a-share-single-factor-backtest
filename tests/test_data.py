"""Tests for Stage 2 standardization rules."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    PipelineConfig,
    clean_benchmark,
    clean_daily_trading,
    clean_valuation,
)


CONFIG = PipelineConfig(Path("raw"), Path("processed"), pd.Timestamp("2014-09-12"), pd.Timestamp("2025-12-31"))


def test_daily_trading_standardizes_codes_filters_dates_and_marks_quality() -> None:
    raw = pd.DataFrame({
        "Stkcd": ["1", "000001", "bad", "000002"],
        "Trddt": ["2014-09-12", "2014-09-12", "2014-09-13", "2026-01-01"],
        "Opnprc": ["10", "10", "x", "9"],
        "Clsprc": ["11", "11", "12", "10"],
        "Dretwd": ["1.2", "0.1", "0.2", "0.3"],
        "Adjprcwd": ["11", "11", "12", "10"],
        "Markettype": ["1", "1", "4", "4"],
        "Trdsta": ["1", "1", "1", "1"],
    })
    result, outside = clean_daily_trading(raw, CONFIG)
    assert outside == 1
    assert result.loc[0, "stock_code"] == "000001"
    assert "duplicate_key" in result.loc[0, "quality_flags"]
    assert "extreme_total_return" in result.loc[0, "quality_flags"]
    assert pd.isna(result.loc[2, "stock_code"])
    assert "invalid_open_price" in result.loc[2, "quality_flags"]


def test_valuation_preserves_invalid_pe_with_explicit_flags() -> None:
    raw = pd.DataFrame({
        "Symbol": ["000001", "000002", "000003"],
        "TradingDate": ["2014-09-12", "2014-09-12", "2013-01-01"],
        "PE1TTM": ["", "-3", "12"],
    })
    result, outside = clean_valuation(raw, CONFIG)
    assert outside == 1
    assert "missing_pe_ttm" in result.loc[0, "quality_flags"]
    assert "nonpositive_pe_ttm" in result.loc[1, "quality_flags"]
    assert result.loc[1, "pe_ttm"] == -3


def test_benchmark_converts_percent_return_to_decimal_and_marks_duplicates() -> None:
    raw = pd.DataFrame({
        "Indexcd": ["H00300", "H00300", "000300"],
        "Idxtrd01": ["2014-09-12", "2014-09-12", "2014-09-12"],
        "Idxtrd05": ["3000", "3000", "2000"],
        "Idxtrd08": ["1.25", "1.25", "2.0"],
    })
    result, outside = clean_benchmark(raw, CONFIG)
    assert outside == 0
    assert len(result) == 2
    assert result.loc[result.index[0], "total_return"] == 0.0125
    assert result["quality_flags"].str.contains("duplicate_key").all()
