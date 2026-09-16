"""Tests for the confirmed Stage 6 target-portfolio and execution-audit rules."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import PortfolioConfig, build_portfolio_outputs, select_targets, validate_outputs


def _config() -> PortfolioConfig:
    return PortfolioConfig(
        processed_dir=Path("data/processed"), selection_quantile=0.2, weighting="equal",
        selection_tie_policy="preserve_boundary_ties", execution="next_trading_day_open",
        execution_availability="normal_trading_status_and_valid_open_price", normal_trading_status="1",
    )


def _factor_details() -> pd.DataFrame:
    rows = []
    scores_by_date = {
        pd.Timestamp("2021-01-29"): {"000001": 0.1, "000002": 0.2, "000003": 0.7, "000004": 0.9, "000005": 0.9},
        pd.Timestamp("2021-02-26"): {"000001": 0.9, "000002": 0.9, "000003": 0.7, "000004": 0.2, "000005": 0.1},
        pd.Timestamp("2021-03-31"): {"000001": 0.1, "000002": 0.2, "000003": 0.7, "000004": 0.9, "000005": 0.9},
    }
    for signal_date, scores in scores_by_date.items():
        for stock_code, score in scores.items():
            row = {"signal_date": signal_date, "stock_code": stock_code}
            for factor in ("pe", "momentum", "roe"):
                row[f"{factor}_score"] = score
                row[f"{factor}_is_rankable"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def _daily() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": ["000004", "000005", "000001", "000002", "000004"],
        "trade_date": pd.to_datetime(["2021-02-01", "2021-02-01", "2021-03-01", "2021-03-01", "2021-03-01"]),
        "trading_status": ["1", "2", "1", "1", "1"],
        "open_price": [10.0, 10.0, 0.0, 10.0, 10.0],
        "quality_flags": ["", "", "", "", ""],
    })


def test_boundary_ties_are_kept_and_target_weights_are_equal() -> None:
    targets = select_targets(_factor_details(), _config())
    first = targets.loc[(targets["factor"].eq("pe")) & (targets["signal_date"].eq(pd.Timestamp("2021-01-29")))]
    assert set(first["stock_code"]) == {"000004", "000005"}
    assert first["selection_rank_pct"].eq(0.9).all()
    assert first["target_weight"].eq(0.5).all()
    assert targets.groupby(["factor", "signal_date"])["target_weight"].sum().eq(1).all()


def test_actions_audit_buy_sell_and_execution_failures_without_replacement() -> None:
    outputs = build_portfolio_outputs(_factor_details(), _daily(), _config())
    actions = outputs.actions
    first = actions.loc[(actions["factor"].eq("pe")) & (actions["signal_date"].eq(pd.Timestamp("2021-01-29")))].set_index("stock_code")
    assert first.loc["000004", "target_action"] == "new_buy"
    assert first.loc["000004", "execution_status"] == "executable"
    assert first.loc["000005", "execution_status"] == "unfilled_cash"
    assert first.loc["000005", "execution_reason"] == "non_normal_trading_status"
    second = actions.loc[(actions["factor"].eq("pe")) & (actions["signal_date"].eq(pd.Timestamp("2021-02-26")))].set_index("stock_code")
    assert second.loc["000001", "execution_status"] == "unfilled_cash"
    assert second.loc["000001", "execution_reason"] == "invalid_open_price"
    assert second.loc["000002", "execution_status"] == "executable"
    assert second.loc["000004", "target_action"] == "planned_sell"
    assert second.loc["000004", "execution_status"] == "executable"
    assert second.loc["000005", "execution_status"] == "pending_sale"
    assert second.loc["000005", "execution_reason"] == "missing_execution_day_trading_record"
    second_targets = outputs.targets.loc[
        outputs.targets["factor"].eq("pe") & outputs.targets["signal_date"].eq(pd.Timestamp("2021-02-26")), "stock_code"
    ]
    assert second_targets.isin(["000004", "000005"]).sum() == 0
    first_summary = outputs.period_summary.loc[
        outputs.period_summary["factor"].eq("pe") & outputs.period_summary["signal_date"].eq(pd.Timestamp("2021-01-29")), "unfilled_buy_cash_weight"
    ]
    assert first_summary.iloc[0] == pytest.approx(0.5)


def test_last_signal_is_preserved_but_has_no_execution_date() -> None:
    outputs = build_portfolio_outputs(_factor_details(), _daily(), _config())
    last_targets = outputs.targets.loc[outputs.targets["signal_date"].eq(pd.Timestamp("2021-03-31"))]
    assert last_targets["execution_date"].isna().all()
    last_orders = outputs.actions.loc[(outputs.actions["signal_date"].eq(pd.Timestamp("2021-03-31"))) & (outputs.actions["target_action"].ne("continue_holding"))]
    assert last_orders["execution_status"].eq("execution_unavailable_due_to_data_end").all()
    validate_outputs(outputs, _config())
