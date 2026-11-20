"""Tests for the factor definitions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.factors import (
    FactorConfig,
    FactorOutputs,
    MomentumVariant,
    build_factor_details,
    build_quality_summary,
    validate_outputs,
)

CONFIG = FactorConfig(Path("processed"), 6)


def _pool() -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-07-30"] * 4),
        "stock_code": ["000001", "000002", "000003", "000004"],
    })


def _valuation() -> pd.DataFrame:
    # The vendor field deliberately ranks the stocks the opposite way from pe_pit
    # so that the test fails if PE is still ranked on the supplier column.
    return pd.DataFrame({
        "stock_code": ["000001", "000002", "000003", "000004"],
        "trading_date": pd.to_datetime(["2021-07-30"] * 4),
        "pe_ttm": [100.0, 10.0, -1.0, None],
    })


def _market_cap() -> pd.DataFrame:
    caps = [200.0, 400.0, 200.0, None]
    return pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-07-30"] * 4),
        "stock_code": ["000001", "000002", "000003", "000004"],
        "market_cap": caps,
        "log_market_cap": [np.log(cap) if cap and cap > 0 else np.nan for cap in caps],
        "has_market_cap": [True, True, True, False],
    })


def _daily() -> pd.DataFrame:
    rows = [
        ("000001", "2020-07-30", 8.0, ""),
        ("000001", "2021-01-29", 10.0, ""),
        ("000001", "2021-06-30", 12.0, ""),
        ("000001", "2021-07-30", 12.0, ""),
        ("000002", "2020-07-30", 10.0, ""),
        ("000002", "2021-01-28", 10.0, ""),
        ("000002", "2021-06-30", 10.0, ""),
        ("000002", "2021-07-30", 11.0, ""),
        ("000003", "2021-01-29", 0.0, ""),
        ("000003", "2021-06-30", 12.0, ""),
        ("000003", "2021-07-30", 12.0, ""),
        ("000004", "2020-07-30", 10.0, ""),
        ("000004", "2021-01-29", 10.0, ""),
        ("000004", "2021-06-30", None, "invalid_adjusted_close"),
        ("000004", "2021-07-30", None, "invalid_adjusted_close"),
    ]
    frame = pd.DataFrame(rows, columns=["stock_code", "trade_date", "adjusted_close", "quality_flags"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _financials() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["000001", "000001", "000002", "000002", "000003", "000004"],
        "report_date": pd.to_datetime(["2019-12-31", "2020-12-31", "2019-12-31", "2020-12-31", "2020-12-31", "2020-12-31"]),
        "Typrep": ["A"] * 6,
        "actual_disclosure_date": pd.to_datetime(["2020-03-01", "2021-03-01", "2021-07-30", "2021-03-01", "2021-03-01", "2021-03-01"]),
        "net_profit": [10.0, 20.0, 30.0, 20.0, -10.0, 10.0],
        "shareholder_equity": [100.0, 100.0, 100.0, 100.0, 100.0, -1.0],
        "invalid_actual_disclosure_date": [False] * 6,
    })


def test_factor_rules_cover_ranking_momentum_and_roe_availability() -> None:
    details = build_factor_details(_pool(), _valuation(), _daily(), _financials(), CONFIG, _market_cap())
    rows = details.set_index("stock_code")

    assert "pe_ttm" in details.columns
    assert rows.loc["000001", "pe_pit"] == 10
    assert rows.loc["000002", "pe_pit"] == 20
    assert rows.loc["000001", "pe_rank"] == 1
    assert rows.loc["000001", "pe_score"] == 1
    assert rows.loc["000002", "pe_rank"] == 2
    assert rows.loc["000003", "pe_unrankable_reason"] == "nonpositive_or_missing_ttm"
    assert rows.loc["000004", "pe_unrankable_reason"] == "missing_market_cap"

    assert rows.loc["000002", "momentum_start_date"] == pd.Timestamp("2021-01-28")
    assert rows.loc["000001", "momentum_return"] == pytest.approx(0.2)
    assert rows.loc["000001", "momentum_rank"] == 1
    assert rows.loc["000003", "momentum_unrankable_reason"] == "no_valid_price_on_or_before_start_date"
    assert rows.loc["000004", "momentum_unrankable_reason"] == "end_date_not_after_start_date"

    assert rows.loc["000001", "roe_report_date"] == pd.Timestamp("2020-12-31")
    assert rows.loc["000001", "roe"] == 0.2
    assert rows.loc["000002", "roe_report_date"] == pd.Timestamp("2020-12-31")
    assert rows.loc["000002", "roe_actual_disclosure_date"] == pd.Timestamp("2021-03-01")
    assert rows.loc["000003", "roe"] == -0.1
    assert rows.loc["000004", "roe_unrankable_reason"] == "missing_or_nonpositive_shareholder_equity"


def test_momentum_variants_use_12_1_6_1_and_1m_reversal() -> None:
    config = FactorConfig(
        Path("processed"), 12, 1,
        (MomentumVariant("momentum_6_1", 6, 1), MomentumVariant("momentum_1m", 1, 0, False)),
    )
    details = build_factor_details(_pool(), _valuation(), _daily(), _financials(), config, _market_cap())
    rows = details.set_index("stock_code")

    assert rows.loc["000001", "momentum_return"] == pytest.approx(0.5)
    assert rows.loc["000001", "momentum_6_1_return"] == pytest.approx(0.2)
    assert rows.loc["000001", "momentum_1m_return"] == pytest.approx(0.0)
    assert rows.loc["000001", "momentum_1m_start_date"] == pd.Timestamp("2021-06-30")
    assert rows.loc["000001", "momentum_1m_end_date"] == pd.Timestamp("2021-07-30")
    # 1M reversal gives the lower return the higher score.
    assert rows.loc["000001", "momentum_1m_score"] > rows.loc["000002", "momentum_1m_score"]


def test_quality_summary_reconciles_every_factor_independently() -> None:
    details = build_factor_details(_pool(), _valuation(), _daily(), _financials(), CONFIG, _market_cap())
    quality = build_quality_summary(details)
    validate_outputs(FactorOutputs(details, quality), _pool())
    assert set(quality["factor"]) == {"pe", "momentum", "roe"}
    assert quality.loc[quality["factor"].eq("pe"), "positive_pe_max"].iloc[0] == 20


def test_pe_unrankable_reasons_name_the_missing_input() -> None:
    financials = _financials().loc[lambda frame: frame["stock_code"].ne("000002")]
    details = build_factor_details(_pool(), _valuation(), _daily(), financials, CONFIG, _market_cap())
    rows = details.set_index("stock_code")

    assert rows.loc["000001", "pe_is_rankable"]
    assert rows.loc["000002", "pe_unrankable_reason"] == "no_disclosed_report"
    assert rows.loc["000003", "pe_unrankable_reason"] == "nonpositive_or_missing_ttm"
    assert rows.loc["000004", "pe_unrankable_reason"] == "missing_market_cap"


def test_pe_reason_distribution_follows_the_disclosure_calendar() -> None:
    pool = pd.DataFrame({
        "signal_date": pd.to_datetime(["2020-12-31", "2020-12-31", "2021-07-30", "2021-07-30"]),
        "stock_code": ["000001", "000002", "000001", "000002"],
    })
    dates = pool["signal_date"]
    market_cap = pd.DataFrame({
        "signal_date": dates,
        "stock_code": pool["stock_code"],
        "market_cap": [100.0, 100.0, 100.0, 100.0],
        "log_market_cap": np.log(100.0),
        "has_market_cap": True,
    })
    financials = pd.DataFrame({
        "stock_code": ["000001", "000002"],
        "report_date": pd.to_datetime(["2020-12-31", "2020-12-31"]),
        "Typrep": ["A", "A"],
        "actual_disclosure_date": pd.to_datetime(["2021-03-01", "2021-03-01"]),
        "net_profit": [10.0, 20.0],
        "shareholder_equity": [100.0, 100.0],
        "invalid_actual_disclosure_date": [False, False],
    })

    details = build_factor_details(pool, _valuation(), _daily(), financials, CONFIG, market_cap)
    quality = build_quality_summary(details).set_index(["factor", "signal_date"])
    early = quality.loc[("pe", pd.Timestamp("2020-12-31"))]
    late = quality.loc[("pe", pd.Timestamp("2021-07-30"))]

    assert early["rankable_count"] == 0
    assert early["unrankable_by_reason"] == '{"no_disclosed_report": 2}'
    assert late["rankable_count"] == 2
    assert late["unrankable_by_reason"] == "{}"

def test_neutral_scores_drop_size_and_industry_exposure() -> None:
    import numpy as np

    from src.factors import neutralize_scores

    rng = np.random.default_rng(7)
    count = 600
    industries = rng.choice(["A", "B", "C", "D"], count)
    size = rng.normal(10.0, 1.5, count)
    industry_effect = pd.Series(industries).map({"A": 0.2, "B": -0.1, "C": 0.0, "D": 0.3}).to_numpy()
    raw = size * 0.8 + industry_effect + rng.normal(0.0, 0.5, count)
    period = pd.DataFrame({
        "signal_date": pd.Timestamp("2021-01-29"),
        "stock_code": [f"{index:06d}" for index in range(count)],
        "pe_score": raw, "momentum_score": raw, "roe_score": raw,
        "log_market_cap": size, "industry_code": industries,
    })

    result = neutralize_scores(period, use_industry=True, min_group_size=5, control_set="industry+size")

    neutral = result["pe_score_neutral"]
    assert result["pe_neutral_eligible"].all()
    assert abs(neutral.corr(pd.Series(size), method="spearman")) < 0.05
    assert neutral.groupby(industries).mean().std() < 0.03


def _pit_financials() -> pd.DataFrame:
    rows = [
        ("000001", "2020-03-31", "2020-04-28", 20.0),
        ("000001", "2020-12-31", "2021-03-30", 100.0),
        ("000001", "2021-03-31", "2021-04-28", 30.0),
    ]
    frame = pd.DataFrame(rows, columns=["stock_code", "report_date", "actual_disclosure_date", "net_profit"])
    frame["Typrep"] = "A"
    frame["invalid_actual_disclosure_date"] = False
    frame["report_date"] = pd.to_datetime(frame["report_date"]).astype("datetime64[ns]")
    frame["actual_disclosure_date"] = pd.to_datetime(frame["actual_disclosure_date"]).astype("datetime64[ns]")
    return frame


def test_ttm_uses_the_annual_profit_for_annual_reports() -> None:
    from src.factors import build_ttm_net_profit

    ttm = build_ttm_net_profit(_pit_financials()).set_index("report_date")
    assert ttm.loc[pd.Timestamp("2020-12-31"), "ttm_net_profit"] == 100.0


def test_ttm_splices_prior_year_components_for_interim_reports() -> None:
    from src.factors import build_ttm_net_profit

    ttm = build_ttm_net_profit(_pit_financials()).set_index("report_date")
    assert ttm.loc[pd.Timestamp("2021-03-31"), "ttm_net_profit"] == 110.0


def test_ttm_is_missing_when_a_component_is_missing() -> None:
    from src.factors import build_ttm_net_profit

    partial = _pit_financials().loc[lambda frame: frame["report_date"].ne(pd.Timestamp("2020-03-31"))]
    ttm = build_ttm_net_profit(partial).set_index("report_date")
    assert pd.isna(ttm.loc[pd.Timestamp("2021-03-31"), "ttm_net_profit"])


def test_duplicate_report_periods_keep_the_latest_disclosure() -> None:
    from src.factors import build_ttm_net_profit

    frame = _pit_financials()
    restated = frame.loc[frame["report_date"].eq(pd.Timestamp("2020-12-31"))].copy()
    restated["actual_disclosure_date"] = pd.Timestamp("2021-04-20")
    restated["net_profit"] = 120.0
    ttm = build_ttm_net_profit(pd.concat([frame, restated], ignore_index=True)).set_index("report_date")
    assert ttm.loc[pd.Timestamp("2020-12-31"), "ttm_net_profit"] == 120.0


def test_pit_panel_ignores_reports_disclosed_on_the_signal_date() -> None:
    from src.factors import build_pit_panel, build_ttm_net_profit

    pool = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-04-28", "2021-04-29"]).astype("datetime64[ns]"),
        "stock_code": ["000001", "000001"],
    })
    panel = build_pit_panel(pool, build_ttm_net_profit(_pit_financials())).set_index("signal_date")
    assert panel.loc[pd.Timestamp("2021-04-28"), "pe_pit_report_period"] == pd.Timestamp("2020-12-31")
    assert panel.loc[pd.Timestamp("2021-04-29"), "pe_pit_report_period"] == pd.Timestamp("2021-03-31")


def _pe_check_frames(pit_values: list[float], vendor_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2021-01-29", "2021-02-26"]).astype("datetime64[ns]")
    details = pd.DataFrame({"signal_date": dates, "stock_code": ["000001", "000001"], "pe_pit": pit_values})
    valuation = pd.DataFrame({
        "stock_code": ["000001", "000001"], "trading_date": dates, "pe_ttm": vendor_values,
    })
    return details, valuation


def test_pe_pit_check_counts_the_matched_names() -> None:
    from src.factors import build_pe_pit_check

    details, valuation = _pe_check_frames([10.0, 12.0], [10.0, 12.0])
    check = build_pe_pit_check(details, valuation)
    assert check["overlap"].tolist() == [1, 1]
    assert check["share_within_1pct"].tolist() == [1.0, 1.0]


def test_pit_check_tolerates_disagreement_inside_a_disclosure_window() -> None:
    from src.factors import build_pe_pit_check, validate_pit_check

    details, valuation = _pe_check_frames([10.0, 15.0], [10.0, 12.0])
    check = build_pe_pit_check(details, valuation)
    validate_pit_check(check)


def test_pit_check_rejects_a_rebuild_that_misses_the_vendor_names() -> None:
    from src.factors import build_pe_pit_check, validate_pit_check

    details, valuation = _pe_check_frames([10.0, float("nan")], [10.0, 12.0])
    check = build_pe_pit_check(details, valuation)
    with pytest.raises(ValueError, match="can be rebuilt"):
        validate_pit_check(check)
