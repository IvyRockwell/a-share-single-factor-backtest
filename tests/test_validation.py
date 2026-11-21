"""Tests for the time-split and multiple-testing validation step."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.validation as validation
from src.validation import (
    ValidationConfig,
    assemble_monthly,
    benjamini_hochberg_adjust,
    build_holdout,
    build_multiple_testing,
    build_subperiod_portfolio,
    build_walk_forward,
    holm_adjust,
    stationary_bootstrap_indices,
    white_reality_check,
)

FACTORS = ("pe", "momentum", "roe")
SUBPERIODS = (
    ("2016", pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31")),
    ("2017", pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31")),
)


def _months(start: str, count: int) -> list[str]:
    return [str(period) for period in pd.period_range(start=start, periods=count, freq="M")]


def _monthly(months: list[str], values: dict[str, float], wiggle: bool = False) -> pd.DataFrame:
    """One row per factor/month with a constant or deterministic monthly pattern."""
    rows = []
    for factor in FACTORS:
        offset = values.get(factor, 0.0)
        for index, month in enumerate(months):
            pattern = 0.002 * ((index % 3) - 1) if wiggle else 0.0
            rows.append({
                "factor": factor,
                "month": month,
                "month_start": pd.Timestamp(month + "-01"),
                "gross_return": 0.01 + offset + pattern,
                "net20_return": 0.009 + offset + pattern,
                "ew_return": 0.005,
                "cw_return": 0.004,
                "h00300_return": 0.003,
            })
    return pd.DataFrame(rows)


def _ic_series(months: list[str], pe: float, momentum: float, roe: float) -> pd.DataFrame:
    rows = []
    for month in months:
        for factor, value in (("pe", pe), ("momentum", momentum), ("roe", roe)):
            rows.append({
                "factor": factor, "score_type": "raw",
                "signal_date": pd.Timestamp(month + "-01"), "ic_pair_count": 100,
                "rank_ic": value, "ic_unavailable_reason": pd.NA,
            })
    return pd.DataFrame(rows)


def _config(**overrides) -> ValidationConfig:
    base = {
        "processed_dir": Path("processed"),
        "subperiods": SUBPERIODS,
        "dev_start": pd.Timestamp("2016-01-01"),
        "holdout_start": pd.Timestamp("2017-01-01"),
        "walk_forward_start_year": 2017,
        "min_train_months": 12,
        "bootstrap_reps": 200,
        "bootstrap_block": 3,
        "seed": 20260701,
    }
    base.update(overrides)
    return ValidationConfig(**base)


def test_assemble_monthly_aligns_strategy_net_and_benchmarks() -> None:
    months = _months("2016-01", 12)
    gross = pd.DataFrame({
        "factor": "pe", "month": months,
        "monthly_return": 0.01, "benchmark_monthly_return": 0.002,
    })
    net = pd.DataFrame({
        "factor": "pe", "slippage_bp": 20, "month": months, "net_monthly_return": 0.008,
    })
    benchmark = pd.DataFrame({
        "factor": ["ew_pool_monthly"] * 12 + ["cw_pool_monthly"] * 12,
        "month_key": months * 2,
        "monthly_return": [0.005] * 12 + [0.004] * 12,
    })

    frame = assemble_monthly(gross, net, benchmark)

    assert len(frame) == 12
    assert frame["ew_return"].eq(0.005).all()
    assert frame["cw_return"].eq(0.004).all()
    assert frame["net20_return"].eq(0.008).all()
    assert frame["month_start"].is_monotonic_increasing


def test_assemble_monthly_ignores_other_slippage_scenarios() -> None:
    months = _months("2016-01", 2)
    gross = pd.DataFrame({"factor": "pe", "month": months, "monthly_return": 0.01,
                          "benchmark_monthly_return": 0.0})
    net = pd.DataFrame({
        "factor": "pe", "slippage_bp": [0, 0, 20, 20], "month": months * 2,
        "net_monthly_return": [0.01, 0.01, 0.008, 0.008],
    })
    benchmark = pd.DataFrame({
        "factor": ["ew_pool_monthly"] * 2 + ["cw_pool_monthly"] * 2,
        "month_key": months * 2,
        "monthly_return": [0.0] * 4,
    })

    frame = assemble_monthly(gross, net, benchmark)

    assert frame["net20_return"].eq(0.008).all()


def test_subperiod_table_covers_factors_periods_and_return_types() -> None:
    months = _months("2016-01", 24)
    monthly = _monthly(months, {"pe": 0.005})

    table = build_subperiod_portfolio(monthly, _config())

    assert len(table) == len(FACTORS) * len(SUBPERIODS) * 2
    assert set(table["return_type"]) == {"gross", "net20"}
    assert table["n_months"].eq(12).all()
    pe = table.loc[table["factor"].eq("pe") & table["subperiod"].eq("2016")]
    gross = pe.loc[pe["return_type"].eq("gross")].iloc[0]
    assert gross["strategy_annualized"] == pytest.approx(1.015 ** 12 - 1)
    assert gross["ew_annualized"] == pytest.approx(1.005 ** 12 - 1)
    assert gross["excess_mean_monthly"] == pytest.approx(0.010)


def test_subperiod_table_annualizes_windows_shorter_than_a_year() -> None:
    months = _months("2016-01", 12)
    monthly = _monthly(months, {})
    config = _config(subperiods=(
        ("H1", pd.Timestamp("2016-01-01"), pd.Timestamp("2016-06-30")),
        ("H2", pd.Timestamp("2016-07-01"), pd.Timestamp("2016-12-31")),
    ))

    table = build_subperiod_portfolio(monthly, config)

    # Six months of 1% compound to 1.01**6, annualised back to 1.01**12 - 1.
    assert table["n_months"].eq(6).all()
    assert table["strategy_annualized"].iloc[0] == pytest.approx(1.01 ** 12 - 1)


def test_holdout_selects_on_development_ic_only() -> None:
    months = _months("2016-01", 24)
    monthly = _monthly(months, {"pe": 0.004, "momentum": 0.02})
    # PE wins the development window; momentum has the better test-period IC.
    ic = pd.concat([
        _ic_series(months[:12], pe=0.10, momentum=0.01, roe=0.02),
        _ic_series(months[12:], pe=0.005, momentum=0.30, roe=0.02),
    ], ignore_index=True)

    table = build_holdout(monthly, ic, _config())

    selected = set(table.loc[table["selected"], "factor"])
    assert selected == {"pe"}
    assert len(table) == len(FACTORS) * 2
    assert set(table["sample"]) == {"dev", "test"}
    assert table.loc[table["selected"], "sample"].tolist() == ["dev", "test"]
    assert table["start"].isna().sum() == 0


def test_holdout_reports_insufficient_ic_without_raising() -> None:
    months = _months("2016-01", 24)
    monthly = _monthly(months, {})
    ic = _ic_series(months[:12], pe=0.1, momentum=0.05, roe=0.02)
    ic = ic.loc[ic["factor"].eq("pe")]

    table = build_holdout(monthly, ic, _config())

    momentum = table.loc[table["factor"].eq("momentum") & table["sample"].eq("dev")].iloc[0]
    assert momentum["n_ic_months"] == 0
    assert pd.isna(momentum["ic_t_nw"])
    assert table.loc[table["selected"], "factor"].tolist() == ["pe", "pe"]


def test_walk_forward_reselects_each_year_and_chains_the_oos_returns() -> None:
    months = _months("2016-01", 24)
    monthly = _monthly(months, {})
    # The 2017 selection uses only the 2016 IC, so PE is chosen and held for 2017.
    ic = pd.concat([
        _ic_series(months[:12], pe=0.10, momentum=0.0, roe=0.02),
        _ic_series(months[12:], pe=0.0, momentum=0.20, roe=0.02),
    ], ignore_index=True)

    table = build_walk_forward(monthly, ic, _config())

    yearly = table.loc[table["year"].astype(str).eq("2017")]
    assert len(yearly) == 1
    assert yearly["selected_factor"].iloc[0] == "pe"
    assert yearly["train_months"].iloc[0] == 12
    assert yearly["oos_return"].iloc[0] == pytest.approx(1.01 ** 12 - 1)
    chain = table.loc[table["year"].astype(str).eq("ALL-CHAIN")]
    assert len(chain) == 1
    assert chain["chained_nav"].iloc[0] == pytest.approx(1.01 ** 12)
    assert table["train_months"].loc[table["year"].astype(str).ne("ALL-CHAIN")].ge(12).all()


def test_stationary_bootstrap_is_seeded_and_in_range() -> None:
    first = stationary_bootstrap_indices(20, 30, 3, seed=7)
    second = stationary_bootstrap_indices(20, 30, 3, seed=7)

    assert first.shape == (30, 20)
    assert first.min() == 0
    assert first.max() == 19
    assert np.array_equal(first, second)


def test_stationary_bootstrap_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(0, 10, 3, seed=1)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(10, 0, 3, seed=1)


def test_white_reality_check_is_small_for_a_strong_signal() -> None:
    index = np.arange(60)
    # Deterministic columns: one with a strong positive mean, one with an exact zero mean.
    strong = np.column_stack([0.02 + 0.01 * np.sin(index / 3 + shift) for shift in range(3)])
    neutral = np.column_stack([0.01 * np.sin(index / 3 + shift) for shift in range(3)])
    neutral = neutral - neutral.mean(axis=0)

    assert white_reality_check(strong, 500, 6, 20260701) <= 0.02
    assert white_reality_check(neutral, 500, 6, 20260701) > 0.5


def test_holm_adjust_uses_a_monotone_step_down() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_benjamini_hochberg_adjust_uses_a_monotone_step_up() -> None:
    adjusted = benjamini_hochberg_adjust(np.array([0.01, 0.04, 0.03]))

    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


def test_adjustments_never_fall_below_the_raw_p_value() -> None:
    raw = np.array([0.9, 0.2, 0.05, 0.5])

    assert (holm_adjust(raw) >= raw).all()
    assert (benjamini_hochberg_adjust(raw) >= raw).all()


def _multiple_testing_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = _months("2016-01", 36)
    monthly = _monthly(months, {"pe": 0.004, "momentum": -0.002}, wiggle=True)
    specs = ("pe", "ep_all", "bm", "pe_12m")
    ic_rows, quantile_rows = [], []
    for spec, value in zip(specs, (0.05, 0.03, 0.06, 0.05)):
        for index, month in enumerate(months):
            ic_rows.append({
                "factor": spec, "score_type": "raw", "signal_date": pd.Timestamp(month + "-01"),
                "ic_pair_count": 200, "rank_ic": value + 0.01 * ((index % 3) - 1),
                "ic_unavailable_reason": pd.NA,
            })
            quantile_rows.append({
                "factor": spec, "score_type": "raw", "signal_date": pd.Timestamp(month + "-01"),
                "quantile": 1,
                "q_high_minus_q_low": (0.002 if spec != "momentum" else -0.001) + 0.001 * ((index % 3) - 1),
            })
    momentum_ic_rows, momentum_quantile_rows = [], []
    for variant, value in zip(("momentum", "momentum_6_1", "momentum_1m"), (0.02, -0.01, 0.04)):
        for index, month in enumerate(months):
            momentum_ic_rows.append({
                "factor": variant, "score_type": "raw", "signal_date": pd.Timestamp(month + "-01"),
                "ic_pair_count": 200, "rank_ic": value + 0.005 * ((index % 3) - 1),
                "ic_unavailable_reason": pd.NA,
            })
            momentum_quantile_rows.append({
                "factor": variant, "score_type": "raw", "signal_date": pd.Timestamp(month + "-01"),
                "quantile": 1,
                "q_high_minus_q_low": (0.001 if variant != "momentum_6_1" else -0.001) + 0.0005 * ((index % 3) - 1),
            })
    neutral_rows = []
    for factor, offset in (("pe", 0.004), ("momentum", -0.002), ("roe", 0.001)):
        for index, month in enumerate(months):
            neutral_rows.append({
                "factor": factor, "series_type": "net", "slippage_bp": 20,
                "month": month, "monthly_return": 0.005 + offset + 0.001 * ((index % 3) - 1),
            })
    return (
        monthly, pd.DataFrame(ic_rows), pd.DataFrame(quantile_rows),
        pd.DataFrame(momentum_ic_rows), pd.DataFrame(momentum_quantile_rows),
        pd.DataFrame(neutral_rows),
    )


def test_multiple_testing_covers_six_families_with_adjusted_values() -> None:
    monthly, ic, quantiles, momentum_ic, momentum_quantiles, neutral = _multiple_testing_inputs()

    table = build_multiple_testing(
        monthly, momentum_ic, momentum_quantiles, neutral, ic, quantiles, _config()
    )

    assert list(table.columns) == list(validation.MULTIPLE_TESTING_COLUMNS)
    assert set(table["family"]) == set(validation.FAMILY_ORDER)
    assert len(table) == 3 + 3 + 3 + 3 + 4 + 4
    for column in ("p_raw", "p_holm", "q_bh", "p_rc"):
        assert table[column].between(0, 1).all()
    assert (table["p_holm"] >= table["p_raw"] - 1e-12).all()
    assert (table["q_bh"] >= table["p_raw"] - 1e-12).all()
    assert table.loc[table["family"].eq("pe_specs_ic"), "p_rc"].nunique() == 1
    assert table.loc[table["family"].eq("neutral_long_only"), "metric"].eq(
        "mean_net20_excess_monthly"
    ).all()


def test_validate_outputs_accepts_the_built_tables() -> None:
    monthly, spec_ic, quantiles, momentum_ic, momentum_quantiles, neutral = _multiple_testing_inputs()
    config = _config(subperiods=())
    primary = _ic_series(_months("2016-01", 36), pe=0.04, momentum=-0.01, roe=0.01)
    combined = pd.concat([primary, spec_ic], ignore_index=True)
    outputs = validation.ValidationOutputs(
        subperiods=build_subperiod_portfolio(monthly, config),
        holdout=build_holdout(monthly, combined, config),
        walk_forward=build_walk_forward(monthly, combined, config),
        multiple_testing=build_multiple_testing(
            monthly, momentum_ic, momentum_quantiles, neutral, spec_ic, quantiles, config
        ),
    )

    validation.validate_outputs(outputs, config)


def test_validate_outputs_rejects_a_missing_family() -> None:
    monthly, spec_ic, quantiles, momentum_ic, momentum_quantiles, neutral = _multiple_testing_inputs()
    config = _config()
    primary = _ic_series(_months("2016-01", 36), pe=0.04, momentum=-0.01, roe=0.01)
    combined = pd.concat([primary, spec_ic], ignore_index=True)
    testing = build_multiple_testing(
        monthly, momentum_ic, momentum_quantiles, neutral, spec_ic, quantiles, config
    )
    outputs = validation.ValidationOutputs(
        subperiods=build_subperiod_portfolio(monthly, config),
        holdout=build_holdout(monthly, combined, config),
        walk_forward=build_walk_forward(monthly, combined, config),
        multiple_testing=testing.loc[testing["family"].ne("pe_specs_ic")],
    )

    with pytest.raises(ValueError, match="family"):
        validation.validate_outputs(outputs, config)
