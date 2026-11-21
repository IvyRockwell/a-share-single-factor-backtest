"""Tests for the PE specification robustness step."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pe_specs import (
    DEFAULT_SPECS,
    PORTFOLIO_SPECS,
    PeSpecConfig,
    _combine_portfolio_rows,
    add_pe_specs,
    build_spec_outputs,
    load_pe_spec_config,
    render_report,
    validate_portfolio_outputs,
    validate_spec_details,
)

SIGNAL_DATES = ("2021-01-29", "2021-02-26", "2021-03-31")


def _details() -> pd.DataFrame:
    """Six stocks: positive earner, loss maker, negative equity, expensive, empty, new listing."""
    universe = {
        # code: (ttm profit, market cap, equity, listing date, rankable PE, PE reason)
        "000001": (10.0, 100.0, 60.0, "2010-01-04", True, pd.NA),
        "000002": (-5.0, 100.0, 50.0, "2010-01-04", False, "nonpositive_or_missing_ttm"),
        "000003": (20.0, 100.0, -1.0, "2020-06-15", True, pd.NA),
        "000004": (5.0, 200.0, 100.0, "2015-03-02", True, pd.NA),
        "000005": (None, 100.0, None, "2015-03-02", False, "no_disclosed_report"),
        "000006": (2.0, 50.0, 5.0, "2019-12-31", True, pd.NA),
    }
    pe_scores = {"000001": 0.75, "000003": 1.0, "000004": 0.25, "000006": 0.5}
    pe_ranks = {"000001": 2.0, "000003": 1.0, "000004": 4.0, "000006": 3.0}
    rows = []
    for signal_date in pd.to_datetime(SIGNAL_DATES):
        for code, (profit, cap, equity, listing, rankable, reason) in universe.items():
            pe = cap / profit if rankable and profit else float("nan")
            rows.append({
                "signal_date": signal_date,
                "stock_code": code,
                "pe_pit": pe,
                "pe_is_rankable": rankable,
                "pe_unrankable_reason": reason,
                "pe_pit_ttm_net_profit": profit,
                "market_cap": cap,
                "shareholder_equity": equity,
                "roe_actual_disclosure_date": pd.Timestamp("2020-04-30") if equity is not None else pd.NaT,
                "pe_rank": pe_ranks.get(code),
                "pe_score": pe_scores.get(code),
            })
    return pd.DataFrame(rows)


def _listing() -> pd.DataFrame:
    rows = []
    dates = {
        "000001": "2010-01-04", "000002": "2010-01-04", "000003": "2020-06-15",
        "000004": "2015-03-02", "000005": "2015-03-02", "000006": "2019-12-31",
    }
    for signal_date in pd.to_datetime(SIGNAL_DATES):
        for code, listing in dates.items():
            rows.append({
                "signal_date": signal_date, "stock_code": code,
                "listing_date": pd.Timestamp(listing),
            })
    return pd.DataFrame(rows)


def _config(**overrides) -> PeSpecConfig:
    base = {"processed_dir": Path("processed"), "specs": DEFAULT_SPECS,
            "listed_months": 12, "quantile_groups": 5, "min_ic_pairs": 2}
    base.update(overrides)
    return PeSpecConfig(**base)


def _forward(details: pd.DataFrame) -> pd.DataFrame:
    frame = details[["signal_date", "stock_code", "pe_score"]].copy()
    # Higher score -> higher forward return, so every specification should show a positive IC.
    frame["forward_return"] = frame["pe_score"].fillna(0.0) * 0.02
    frame["forward_eligible"] = True
    frame["forward_unavailable_reason"] = pd.NA
    frame["early_exit_or_data_cutoff"] = False
    frame["extreme_return_days"] = 0
    return frame.drop(columns="pe_score")


def test_add_pe_specs_builds_all_five_columns_per_spec() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())

    for spec in DEFAULT_SPECS:
        for suffix in ("_raw", "_is_rankable", "_unrankable_reason", "_rank", "_score"):
            assert f"{spec}{suffix}" in specs.columns
    validate_spec_details(specs, DEFAULT_SPECS)


def test_ep_all_keeps_loss_makers_that_the_positive_pe_drops() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())
    first = specs.loc[specs["signal_date"].eq(pd.Timestamp(SIGNAL_DATES[0]))]

    loss = first.loc[first["stock_code"].eq("000002")].iloc[0]
    assert not loss["pe_is_rankable"]
    assert loss["ep_all_is_rankable"]
    assert loss["ep_all_raw"] == pytest.approx(-0.05)
    # The highest earnings yield still scores 1.0 and the loss maker sits at the bottom.
    assert first.loc[first["stock_code"].eq("000003"), "ep_all_score"].iloc[0] == pytest.approx(1.0)
    assert first.loc[first["stock_code"].eq("000002"), "ep_all_score"].iloc[0] == pytest.approx(0.2)


def test_bm_requires_positive_book_equity_and_map_market_cap() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())
    first = specs.loc[specs["signal_date"].eq(pd.Timestamp(SIGNAL_DATES[0]))]

    negative_equity = first.loc[first["stock_code"].eq("000003")].iloc[0]
    assert not negative_equity["bm_is_rankable"]
    assert negative_equity["bm_unrankable_reason"] == "nonpositive_shareholder_equity"
    assert first.loc[first["stock_code"].eq("000001"), "bm_raw"].iloc[0] == pytest.approx(0.6)
    assert first.loc[first["stock_code"].eq("000001"), "bm_score"].iloc[0] == pytest.approx(1.0)
    recent_reason = first.loc[first["stock_code"].eq("000005"), "bm_unrankable_reason"].iloc[0]
    assert recent_reason == "no_prior_annual_consolidated_disclosure"


def test_pe_12m_drops_recent_listings_and_reranks_the_filtered_pool() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())
    first = specs.loc[specs["signal_date"].eq(pd.Timestamp(SIGNAL_DATES[0]))]

    recent = first.loc[first["stock_code"].eq("000003")].iloc[0]
    assert not recent["pe_12m_is_rankable"]
    assert recent["pe_12m_unrankable_reason"] == "listed_less_than_12_months"
    # 000001 was rank 2 of 4 in the primary pool; after 000003 drops out it is 1 of 3.
    survivor = first.loc[first["stock_code"].eq("000001")].iloc[0]
    assert survivor["pe_rank"] == pytest.approx(2.0)
    assert survivor["pe_12m_rank"] == pytest.approx(1.0)
    assert survivor["pe_12m_score"] == pytest.approx(1.0)
    assert survivor["pe_12m_score"] != pytest.approx(float(survivor["pe_score"]))


def test_pe_12m_is_a_subset_of_the_positive_pe_sample() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())
    subset = specs["pe_12m_is_rankable"].fillna(False)
    assert not (subset & ~specs["pe_is_rankable"].fillna(False)).any()


def test_unrankable_rows_always_carry_a_reason() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())

    for spec in DEFAULT_SPECS:
        mask = ~specs[f"{spec}_is_rankable"].fillna(False)
        assert specs.loc[mask, f"{spec}_unrankable_reason"].notna().all()


def test_validate_spec_details_rejects_a_leaking_twelve_month_sample() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())
    broken = specs.copy()
    first_month = broken["signal_date"].eq(pd.Timestamp(SIGNAL_DATES[0]))
    loss_maker = broken.index[broken["stock_code"].eq("000002") & first_month]
    broken.loc[loss_maker, "pe_12m_is_rankable"] = True

    with pytest.raises(ValueError, match="subset"):
        validate_spec_details(broken, DEFAULT_SPECS)


def test_spec_factor_tests_cover_every_specification() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())

    summary, ic, quantiles = build_spec_outputs(specs, _forward(specs), _config())

    assert list(summary["spec"]) == list(DEFAULT_SPECS)
    assert summary["ic_valid_periods"].eq(len(SIGNAL_DATES)).all()
    assert set(ic["factor"]) == set(DEFAULT_SPECS)
    assert {"q_high_minus_q_low", "equal_weight_forward_return"} <= set(quantiles.columns)


def test_spec_report_states_the_twelve_month_benchmark_caveat() -> None:
    specs = add_pe_specs(_details(), _listing(), _config())
    summary, ic, quantiles = build_spec_outputs(specs, _forward(specs), _config())
    from src.pe_specs import PeSpecOutputs

    portfolio = pd.DataFrame([{
        "spec": "pe", "annualized_return": 0.1, "annualized_volatility": 0.2,
        "sharpe_ratio": 0.5, "max_drawdown": -0.3, "alpha_annualized": 0.02,
        "alpha_t": 1.0, "excess_mean_monthly": 0.001, "excess_t_nw": 0.8, "n_months": 12,
    }])
    text = render_report(PeSpecOutputs(specs, summary, ic, quantiles, portfolio), _config())

    assert "不做纯多头" in text
    assert "基准错配" in text
    assert "负盈利" in text
    assert "winsorize" in text


def test_combine_portfolio_rows_joins_metrics_and_equal_weight_alpha() -> None:
    metrics = pd.DataFrame([
        {"spec": "pe", "annualized_return": 0.10, "annualized_volatility": 0.20,
         "sharpe_ratio": 0.50, "max_drawdown": -0.30},
        {"spec": "ep_all", "annualized_return": 0.09, "annualized_volatility": 0.21,
         "sharpe_ratio": 0.43, "max_drawdown": -0.31},
        {"spec": "bm", "annualized_return": 0.08, "annualized_volatility": 0.22,
         "sharpe_ratio": 0.36, "max_drawdown": -0.32},
    ])
    alpha = pd.DataFrame([
        {"spec": spec, "alpha_annualized": 0.02, "alpha_t": 1.2,
         "excess_mean_monthly": 0.001, "excess_t_nw": 0.9, "n_months": 135}
        for spec in PORTFOLIO_SPECS
    ])

    combined = _combine_portfolio_rows(metrics, alpha, PORTFOLIO_SPECS)

    assert list(combined["spec"]) == list(PORTFOLIO_SPECS)
    assert combined.loc[combined["spec"].eq("bm"), "alpha_t"].iloc[0] == pytest.approx(1.2)
    validate_portfolio_outputs(combined, DEFAULT_SPECS)


def test_combine_portfolio_rows_drops_a_spec_without_alpha() -> None:
    metrics = pd.DataFrame([
        {"spec": "pe", "annualized_return": 0.10, "annualized_volatility": 0.20,
         "sharpe_ratio": 0.50, "max_drawdown": -0.30},
        {"spec": "ep_all", "annualized_return": 0.09, "annualized_volatility": 0.21,
         "sharpe_ratio": 0.43, "max_drawdown": -0.31},
    ])
    alpha = pd.DataFrame([{"spec": "pe", "alpha_annualized": 0.02, "alpha_t": 1.2,
                           "excess_mean_monthly": 0.001, "excess_t_nw": 0.9, "n_months": 135}])

    combined = _combine_portfolio_rows(metrics, alpha, PORTFOLIO_SPECS)

    assert list(combined["spec"]) == ["pe"]


def test_validate_portfolio_outputs_requires_two_primary_pool_specs() -> None:
    single = pd.DataFrame([{"spec": "pe"}])
    with pytest.raises(ValueError, match="two"):
        validate_portfolio_outputs(single, DEFAULT_SPECS)
    complete = pd.DataFrame([{"spec": "pe"}, {"spec": "ep_all"}, {"spec": "bm"}])
    validate_portfolio_outputs(complete, DEFAULT_SPECS)
    missing = complete.loc[complete["spec"].ne("bm")]
    with pytest.raises(ValueError, match="cover"):
        validate_portfolio_outputs(missing, DEFAULT_SPECS)


def test_load_pe_spec_config_uses_defaults_when_the_section_is_absent(tmp_path: Path) -> None:
    config = {
        "data": {"processed_dir": "data/processed"},
        "factors": {"quantile_groups": 5},
        "factor_test": {"min_ic_pairs": 50, "subperiods": []},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    loaded = load_pe_spec_config(path)

    assert loaded.specs == DEFAULT_SPECS
    assert loaded.listed_months == 12
    assert loaded.quantile_groups == 5
    assert loaded.min_ic_pairs == 50


def test_load_pe_spec_config_rejects_an_unknown_specification(tmp_path: Path) -> None:
    config = {
        "data": {"processed_dir": "data/processed"},
        "factors": {"quantile_groups": 5},
        "factor_test": {"min_ic_pairs": 50, "subperiods": []},
        "pe_specs": {"specs": ["pe", "not_a_spec"]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown PE specification"):
        load_pe_spec_config(path)
