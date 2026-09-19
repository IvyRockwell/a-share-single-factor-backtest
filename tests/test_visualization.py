"""Tests for Stage 11 research-figure inputs and fixed conventions."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization import (
    FACTOR_COLORS,
    FIGURE_FILES,
    SCENARIO_STYLES,
    build_figure_index,
    prepare_q_spread,
)


def test_q5_minus_q1_uses_the_verified_q1_copy_of_stage5_cumulative_field() -> None:
    rows = []
    spreads = [0.10, -0.04]
    for date, spread, cumulative in zip(pd.to_datetime(["2020-01-31", "2020-02-28"]), spreads, [0.10, 0.06]):
        for quantile in range(1, 6):
            rows.append({
                "factor": "pe", "signal_date": date, "quantile": quantile,
                "q_high_minus_q_low": spread,
                "cumulative_q_high_minus_q_low": cumulative if quantile == 1 else cumulative + quantile,
            })
    result = prepare_q_spread(pd.DataFrame(rows))
    assert result["cumulative_q_high_minus_q_low"].tolist() == pytest.approx([0.10, 0.06])


def test_q5_minus_q1_rejects_a_q1_copy_that_breaks_the_fixed_stage5_rule() -> None:
    frame = pd.DataFrame({
        "factor": ["pe", "pe"], "signal_date": pd.to_datetime(["2020-01-31", "2020-02-28"]),
        "quantile": [1, 1], "q_high_minus_q_low": [0.10, -0.04],
        "cumulative_q_high_minus_q_low": [0.10, 0.07],
    })
    with pytest.raises(ValueError, match="cumulative Q5-Q1"):
        prepare_q_spread(frame)


def test_figure_manifest_and_index_cover_every_requested_chart() -> None:
    assert len(FIGURE_FILES) == 17
    assert len(set(FIGURE_FILES)) == len(FIGURE_FILES)
    index = build_figure_index()
    for filename in FIGURE_FILES:
        assert filename in index
    assert "0 bp 仅表示无滑点" in index
    assert "不选择最佳参数" in index


def test_fixed_colors_and_cost_scenario_styles_are_stable() -> None:
    assert FACTOR_COLORS == {"pe": "#0072B2", "momentum": "#E69F00", "roe": "#009E73"}
    assert [SCENARIO_STYLES[bp]["color"] for bp in (0, 10, 20)] == ["#B8B8B8", "#777777", "#242424"]
    assert len({SCENARIO_STYLES[bp]["linestyle"] for bp in (0, 10, 20)}) == 3
