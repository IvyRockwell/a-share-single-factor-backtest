"""Build single-factor target portfolios and execution audits."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

FACTORS = ("pe", "momentum", "roe")


@dataclass(frozen=True)
class PortfolioConfig:
    processed_dir: Path
    selection_quantile: float
    weighting: str
    selection_tie_policy: str
    execution: str
    execution_availability: str
    normal_trading_status: str


@dataclass(frozen=True)
class PortfolioOutputs:
    targets: pd.DataFrame
    actions: pd.DataFrame
    period_summary: pd.DataFrame


def load_portfolio_config(config_path: Path) -> PortfolioConfig:
    """Load the portfolio construction rules."""
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    portfolio = settings["portfolio"]
    return PortfolioConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        selection_quantile=float(portfolio["selection_quantile"]),
        weighting=str(portfolio["weighting"]),
        selection_tie_policy=str(portfolio["selection_tie_policy"]),
        execution=str(portfolio["execution"]),
        execution_availability=str(portfolio["execution_availability"]),
        normal_trading_status=str(settings["universe"]["normal_trading_status"]),
    )


def validate_config(config: PortfolioConfig) -> None:
    """Reject configuration values that would contradict the portfolio rules."""
    if not 0 < config.selection_quantile < 1:
        raise ValueError("selection_quantile must be strictly between zero and one.")
    if config.weighting != "equal":
        raise ValueError("Only equal weighting is supported.")
    if config.selection_tie_policy != "preserve_boundary_ties":
        raise ValueError("Ties at the selection boundary must be preserved.")
    if config.execution != "next_trading_day_open":
        raise ValueError("Execution must be at the next trading day's open.")
    if config.execution_availability != "normal_trading_status_and_valid_open_price":
        raise ValueError("Execution requires a normal trading status and a valid open price.")


def build_execution_dates(signal_dates: pd.Series, trading_dates: pd.Series) -> pd.DataFrame:
    """Map each signal date to the first later market trading date."""
    signals = pd.Series(pd.to_datetime(signal_dates).unique()).sort_values().reset_index(drop=True)
    calendar = pd.Series(pd.to_datetime(trading_dates).unique()).sort_values().reset_index(drop=True)
    positions = calendar.searchsorted(signals, side="right")
    return pd.DataFrame({
        "signal_date": signals,
        "execution_date": [calendar.iloc[position] if position < len(calendar) else pd.NaT for position in positions],
    })


def select_targets(factor_details: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Select each factor's high-score names while preserving ties at the cutoff."""
    frames = []
    for factor in FACTORS:
        frame = factor_details[["signal_date", "stock_code", f"{factor}_score", f"{factor}_is_rankable"]].copy()
        frame = frame.rename(columns={f"{factor}_score": "factor_score", f"{factor}_is_rankable": "factor_is_rankable"})
        frame = frame.loc[frame["factor_is_rankable"].fillna(False)].copy()
        frame["factor"] = factor
        frame["selection_rank_pct"] = frame.groupby("signal_date")["factor_score"].rank(method="average", pct=True)
        frame["rankable_stock_count"] = frame.groupby("signal_date")["stock_code"].transform("size")
        frame["nominal_selection_count"] = frame["rankable_stock_count"].map(
            lambda count: math.ceil(count * config.selection_quantile)
        )
        frame["selection_cutoff_score"] = frame.groupby("signal_date")["factor_score"].transform(
            lambda scores: scores.nlargest(math.ceil(len(scores) * config.selection_quantile)).iloc[-1]
        )
        frame = frame.loc[frame["factor_score"].ge(frame["selection_cutoff_score"])].copy()
        frame["target_stock_count"] = frame.groupby("signal_date")["stock_code"].transform("size")
        frame["target_weight"] = 1 / frame["target_stock_count"]
        frames.append(frame)
    columns = [
        "factor", "signal_date", "stock_code", "factor_score", "selection_rank_pct",
        "rankable_stock_count", "nominal_selection_count", "selection_cutoff_score",
        "target_stock_count", "target_weight",
    ]
    return pd.concat(frames, ignore_index=True)[columns].sort_values(["factor", "signal_date", "stock_code"]).reset_index(drop=True)


def _execution_status(frame: pd.DataFrame, config: PortfolioConfig, action: str) -> pd.Series:
    unavailable_prefix = "unfilled_cash" if action == "new_buy" else "pending_sale"
    result = pd.Series("executable", index=frame.index, dtype="string")
    result.loc[frame["execution_date"].isna()] = "execution_unavailable_due_to_data_end"
    missing_record = frame["execution_date"].notna() & frame["trading_status"].isna()
    result.loc[missing_record] = f"{unavailable_prefix}:missing_execution_day_trading_record"
    non_normal = frame["execution_date"].notna() & frame["trading_status"].notna() & ~frame["trading_status"].astype("string").eq(config.normal_trading_status)
    result.loc[non_normal] = f"{unavailable_prefix}:non_normal_trading_status"
    valid_open = frame["open_price"].notna() & frame["open_price"].gt(0) & ~frame["quality_flags"].fillna("").astype("string").str.contains("invalid_open_price", regex=False)
    invalid_open = frame["execution_date"].notna() & frame["trading_status"].astype("string").eq(config.normal_trading_status) & ~valid_open
    result.loc[invalid_open] = f"{unavailable_prefix}:invalid_open_price"
    return result


def _split_execution_status(status: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Separate an execution outcome from its auditable reason."""
    status = status.astype("string")
    outcome = status.str.split(":", n=1).str[0]
    reason = status.str.split(":", n=1).str[1]
    reason = reason.fillna(status.map({
        "executable": "normal_trading_status_and_valid_open_price",
        "rebalance_required": "target_continues_and_is_rebalanced_by_backtest",
        "execution_unavailable_due_to_data_end": "no_later_market_trading_date_in_data",
    }))
    return outcome, reason.astype("string")


def build_actions(targets: pd.DataFrame, execution_daily: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Compare adjacent target lists and audit their scheduled execution-day feasibility."""
    dates = targets[["signal_date", "execution_date"]].drop_duplicates().sort_values("signal_date")
    target_keys = {(factor, date): set(group["stock_code"]) for (factor, date), group in targets.groupby(["factor", "signal_date"])}
    target_weights = targets.set_index(["factor", "signal_date", "stock_code"])["target_weight"]
    trading = execution_daily[["stock_code", "trade_date", "trading_status", "open_price", "quality_flags"]].rename(columns={"trade_date": "execution_date"})
    rows = []
    prior_by_factor = {factor: set() for factor in FACTORS}
    for row in dates.itertuples(index=False):
        for factor in FACTORS:
            current = target_keys.get((factor, row.signal_date), set())
            prior = prior_by_factor[factor]
            for stock_code, action in ((code, "new_buy") for code in current - prior):
                rows.append({"factor": factor, "signal_date": row.signal_date, "execution_date": row.execution_date, "stock_code": stock_code, "target_action": action, "target_weight": target_weights[(factor, row.signal_date, stock_code)]})
            for stock_code, action in ((code, "continue_holding") for code in current & prior):
                rows.append({"factor": factor, "signal_date": row.signal_date, "execution_date": row.execution_date, "stock_code": stock_code, "target_action": action, "target_weight": target_weights[(factor, row.signal_date, stock_code)]})
            for stock_code, action in ((code, "planned_sell") for code in prior - current):
                rows.append({"factor": factor, "signal_date": row.signal_date, "execution_date": row.execution_date, "stock_code": stock_code, "target_action": action, "target_weight": pd.NA})
            prior_by_factor[factor] = current
    actions = pd.DataFrame(rows).merge(trading, on=["stock_code", "execution_date"], how="left", validate="many_to_one")
    actions["execution_status"] = "rebalance_required"
    for action in ("new_buy", "planned_sell"):
        mask = actions["target_action"].eq(action)
        actions.loc[mask, "execution_status"] = _execution_status(actions.loc[mask], config, action)
    actions["execution_status"], actions["execution_reason"] = _split_execution_status(actions["execution_status"])
    return actions.sort_values(["factor", "signal_date", "stock_code", "target_action"]).reset_index(drop=True)


def build_period_summary(targets: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Summarize target weights and scheduled order outcomes without calculating returns."""
    base = targets.groupby(["factor", "signal_date", "execution_date"], as_index=False, dropna=False).agg(
        target_stock_count=("stock_code", "size"), target_weight_sum=("target_weight", "sum"),
    )
    orders = actions.loc[actions["target_action"].isin(["new_buy", "planned_sell"])].copy()
    orders["unfilled_buy_cash_weight"] = orders["target_weight"].where(orders["execution_status"].eq("unfilled_cash"), 0).fillna(0)
    summary = orders.groupby(["factor", "signal_date", "execution_date"], as_index=False, dropna=False).agg(
        executable_buy_count=("execution_status", lambda value: int(value.eq("executable").sum())),
        unfilled_buy_cash_weight=("unfilled_buy_cash_weight", "sum"),
        planned_sell_count=("target_action", lambda value: int(value.eq("planned_sell").sum())),
        pending_sale_count=("execution_status", lambda value: int(value.eq("pending_sale").sum())),
        execution_unavailable_due_to_data_end_count=("execution_status", lambda value: int(value.eq("execution_unavailable_due_to_data_end").sum())),
    )
    return base.merge(summary, on=["factor", "signal_date", "execution_date"], how="left").fillna({
        "executable_buy_count": 0, "unfilled_buy_cash_weight": 0, "planned_sell_count": 0,
        "pending_sale_count": 0, "execution_unavailable_due_to_data_end_count": 0,
    }).sort_values(["factor", "signal_date"]).reset_index(drop=True)


def build_portfolio_outputs(factor_details: pd.DataFrame, daily: pd.DataFrame, config: PortfolioConfig) -> PortfolioOutputs:
    """Build all portfolio tables from the factor data and normalized daily prices."""
    validate_config(config)
    targets = select_targets(factor_details, config)
    dates = build_execution_dates(targets["signal_date"], daily["trade_date"])
    targets = targets.merge(dates, on="signal_date", how="left", validate="many_to_one")
    execution_dates = dates["execution_date"].dropna().unique().tolist()
    execution_daily = daily.loc[daily["trade_date"].isin(execution_dates), ["stock_code", "trade_date", "trading_status", "open_price", "quality_flags"]].copy()
    actions = build_actions(targets, execution_daily, config)
    target_execution = actions.loc[actions["target_action"].isin(["new_buy", "continue_holding"]), [
        "factor", "signal_date", "stock_code", "execution_status", "execution_reason",
    ]]
    targets = targets.merge(target_execution, on=["factor", "signal_date", "stock_code"], how="left", validate="one_to_one")
    summary = build_period_summary(targets, actions)
    return PortfolioOutputs(targets, actions, summary)


def validate_outputs(outputs: PortfolioOutputs, config: PortfolioConfig) -> None:
    """Validate the portfolio invariants before writing artifacts."""
    targets, actions, summary = outputs.targets, outputs.actions, outputs.period_summary
    if targets.duplicated(["factor", "signal_date", "stock_code"]).any():
        raise ValueError("Target holdings contain duplicate keys.")
    if targets["factor_score"].lt(targets["selection_cutoff_score"]).any():
        raise ValueError("A target falls below the score threshold at the nominal selection position.")
    if targets["target_stock_count"].lt(targets["nominal_selection_count"]).any():
        raise ValueError("Boundary-tie handling selected fewer stocks than the nominal ceiling count.")
    weight_sums = targets.groupby(["factor", "signal_date"])["target_weight"].sum()
    if not weight_sums.sub(1).abs().lt(1e-12).all():
        raise ValueError("Target weights do not sum to one.")
    active = targets["execution_date"].notna()
    if targets.loc[active, "execution_date"].le(targets.loc[active, "signal_date"]).any():
        raise ValueError("An execution date is not after its signal date.")
    if targets[["execution_status", "execution_reason"]].isna().any().any():
        raise ValueError("Every target holding must retain an execution status and reason.")
    if actions.duplicated(["factor", "signal_date", "stock_code", "target_action"]).any():
        raise ValueError("Rebalance actions contain duplicate keys.")
    if actions.loc[actions["target_action"].eq("continue_holding"), "execution_status"].ne("rebalance_required").any():
        raise ValueError("Continuing target holdings are rebalanced by the backtest.")
    if summary.duplicated(["factor", "signal_date"]).any():
        raise ValueError("Portfolio summary contains duplicate period keys.")


def render_report(outputs: PortfolioOutputs, config: PortfolioConfig) -> str:
    """Render the portfolio report."""
    summary = outputs.period_summary
    lines = [
        "# 单因子目标组合与执行审计", "", "## 固定规则", "",
        f"- 每个因子独立在每个信号日从可排名股票中选择 Top {config.selection_quantile:.0%}；边界同分全部保留，不按股票代码拆分。",
        "- 目标持仓等权，目标权重在每个因子和信号日严格加总为 1。",
        "- 执行日为信号日后的首个市场交易日，执行价格口径为开盘；可执行需有该日交易记录、`trading_status=1` 且开盘价有效。",
        "- 继续入选股票不视为无委托；回测在每个执行日按开盘前组合净值重新评估等权调整。",
        "- 不可买入不替代、不跨日追单；完全退出但不可卖出的持仓由回测逐日重试。",
        "- 2025-12-31 的目标候选保留，但因数据末端没有后续执行日而不标记为已执行。", "", "## 输出核对", "",
        f"- 目标持仓：{len(outputs.targets):,} 条；调仓动作：{len(outputs.actions):,} 条；逐期汇总：{len(summary):,} 条。",
        f"- 各期目标权重均为 1；不可买入对应的目标现金权重总和为 {summary['unfilled_buy_cash_weight'].sum():.6f}。", "",
    ]
    lines.extend(["", "## 限制", "", "- 本步骤只形成目标组合和执行审计，不计算实际市值权重、现金收益、组合收益或延迟卖出后的成交结果。", "- 现有可靠字段不足以识别涨跌停造成的拒单；该限制不被静默视为可成交。", ""])
    return "\n".join(lines)


def run(config: PortfolioConfig) -> PortfolioOutputs:
    """Read the factor data, write portfolio artifacts, and return them in memory."""
    directory = config.processed_dir
    factor_details = pd.read_parquet(directory / "factor_details.parquet")
    daily = pd.read_parquet(directory / "daily_trading.parquet", columns=["stock_code", "trade_date", "trading_status", "open_price", "quality_flags"])
    outputs = build_portfolio_outputs(factor_details, daily, config)
    validate_outputs(outputs, config)
    outputs.targets.to_parquet(directory / "portfolio_targets.parquet", index=False)
    outputs.actions.to_parquet(directory / "portfolio_rebalance_actions.parquet", index=False)
    outputs.period_summary.to_csv(directory / "portfolio_period_summary.csv", index=False, encoding="utf-8-sig")
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "portfolio.md").write_text(render_report(outputs, config), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the single-factor target portfolios.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_portfolio_config(args.config))
    print(f"Built {len(outputs.targets):,} target-holding rows and {len(outputs.actions):,} rebalance-action rows.")


if __name__ == "__main__":
    main()
