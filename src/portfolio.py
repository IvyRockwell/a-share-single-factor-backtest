"""Build single-factor target portfolios and execution audits."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.trading_rules import trade_block_reason

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
    initial_capital: float = 100_000_000.0
    adv_window: int = 20
    adv_fraction: float | None = None
    min_adv_periods: int = 10


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
    capacity = settings.get("capacity") or {}
    research = settings.get("research") or {}
    adv_fraction = capacity.get("adv_fraction")
    return PortfolioConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        selection_quantile=float(portfolio["selection_quantile"]),
        weighting=str(portfolio["weighting"]),
        selection_tie_policy=str(portfolio["selection_tie_policy"]),
        execution=str(portfolio["execution"]),
        execution_availability=str(portfolio["execution_availability"]),
        normal_trading_status=str(settings["universe"]["normal_trading_status"]),
        initial_capital=float(research.get("initial_capital", 100_000_000.0)),
        adv_window=int(capacity.get("adv_window", 20)),
        adv_fraction=float(adv_fraction) if adv_fraction is not None else None,
        min_adv_periods=int(capacity.get("min_adv_periods", 10)),
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
    if config.adv_fraction is not None and not 0 < config.adv_fraction <= 1:
        raise ValueError("adv_fraction must be in (0, 1] when enabled.")


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


def apply_adv_cap(targets: pd.DataFrame, daily: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Cap each target weight at a fraction of its signal-date 20-day ADV.

    The cap is applied at the initial-capital level and the excess weight is
    left as cash, not renormalised.  This is a capacity constraint, not a
    partial-fill model: the backtest still executes the (capped) target at the
    next open.
    """
    result = targets.copy()
    result["uncapped_target_weight"] = result["target_weight"]
    result["capacity_capped"] = False
    if config.adv_fraction is None:
        result["adv_20"] = np.nan
        result["adv_cap_weight"] = np.nan
        return result

    prices = daily[["stock_code", "trade_date", "trade_amount"]] if "trade_amount" in daily else daily
    if "trade_amount" not in prices:
        result["adv_20"] = np.nan
        result["adv_cap_weight"] = np.nan
        return result
    prices = prices[["stock_code", "trade_date", "trade_amount"]].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).astype("datetime64[ns]")
    prices["trade_amount"] = pd.to_numeric(prices["trade_amount"], errors="coerce")
    prices = prices.loc[prices["trade_amount"].gt(0)].sort_values(["stock_code", "trade_date"])
    prices["adv_20"] = prices.groupby("stock_code")["trade_amount"].transform(
        lambda values: values.rolling(config.adv_window, min_periods=config.min_adv_periods).mean()
    )
    adv = prices[["stock_code", "trade_date", "adv_20"]].rename(columns={"trade_date": "signal_date"})
    result = result.merge(adv, on=["signal_date", "stock_code"], how="left", validate="many_to_one")
    cap_weight = result["adv_20"] * float(config.adv_fraction) / float(config.initial_capital)
    result["adv_cap_weight"] = cap_weight
    result["capacity_capped"] = cap_weight.notna() & result["target_weight"].gt(cap_weight)
    result["target_weight"] = np.where(
        cap_weight.notna(), np.minimum(result["target_weight"], cap_weight), result["target_weight"]
    )
    return result


def build_capacity_summary(targets: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Estimate the AUM at which the ADV cap starts binding for each period.

    ``max_aum = min_i(adv_20_i * adv_fraction / target_weight_i)``.  The
    minimum across names is the binding capacity; the median across periods is
    the headline number.
    """
    if config.adv_fraction is None or "adv_20" not in targets:
        return pd.DataFrame()
    frame = targets.loc[
        targets["adv_20"].notna() & targets["uncapped_target_weight"].gt(0)
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["max_aum"] = frame["adv_20"] * float(config.adv_fraction) / frame["uncapped_target_weight"]
    periods = frame.groupby(["factor", "signal_date"], as_index=False).agg(
        max_aum=("max_aum", "min"),
        capped_count=("capacity_capped", "sum"),
    )
    summary = periods.groupby("factor", as_index=False).agg(
        periods=("signal_date", "size"),
        max_aum_min=("max_aum", "min"),
        max_aum_median=("max_aum", "median"),
        max_aum_max=("max_aum", "max"),
        capped_target_rows=("capped_count", "sum"),
    )
    summary["adv_fraction"] = float(config.adv_fraction)
    summary["initial_capital"] = float(config.initial_capital)
    return summary


def _execution_status(frame: pd.DataFrame, config: PortfolioConfig, action: str) -> pd.Series:
    unavailable_prefix = "unfilled_cash" if action == "new_buy" else "pending_sale"
    side = "buy" if action == "new_buy" else "sell"
    result = pd.Series("executable", index=frame.index, dtype="string")
    result.loc[frame["execution_date"].isna()] = "execution_unavailable_due_to_data_end"
    missing_record = frame["execution_date"].notna() & frame["trading_status"].isna()
    result.loc[missing_record] = f"{unavailable_prefix}:missing_execution_day_trading_record"
    checked = frame["execution_date"].notna() & frame["trading_status"].notna()
    for index in frame.index[checked]:
        reason = trade_block_reason(frame.loc[index], config.normal_trading_status, side)
        if reason is not None:
            result.loc[index] = f"{unavailable_prefix}:{reason}"
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
        target_stock_count=("stock_code", "size"),
        target_weight_sum=("target_weight", "sum"),
        capacity_capped_count=("capacity_capped", "sum"),
    )
    base["unallocated_cash_weight"] = (1.0 - base["target_weight_sum"]).clip(lower=0.0)
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
    daily = daily.copy()
    if "trade_amount" not in daily and "Dnvaltrd" in daily:
        daily["trade_amount"] = pd.to_numeric(daily["Dnvaltrd"], errors="coerce")
    if "pre_close" not in daily and "PreClosePrice" in daily:
        daily["pre_close"] = pd.to_numeric(daily["PreClosePrice"], errors="coerce")
    if "market_type" not in daily and "Markettype" in daily:
        daily["market_type"] = daily["Markettype"].astype("string")
    targets = select_targets(factor_details, config)
    if config.adv_fraction is not None:
        targets = apply_adv_cap(targets, daily, config)
    if "uncapped_target_weight" not in targets:
        targets["uncapped_target_weight"] = targets["target_weight"]
    if "capacity_capped" not in targets:
        targets["capacity_capped"] = False
    dates = build_execution_dates(targets["signal_date"], daily["trade_date"])
    targets = targets.merge(dates, on="signal_date", how="left", validate="many_to_one")
    execution_dates = dates["execution_date"].dropna().unique().tolist()
    execution_columns = ["stock_code", "trade_date", "trading_status", "open_price", "quality_flags"]
    execution_columns += [column for column in ("pre_close", "market_type") if column in daily.columns]
    execution_daily = daily.loc[daily["trade_date"].isin(execution_dates), execution_columns].copy()
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
    if weight_sums.lt(-1e-12).any() or weight_sums.gt(1 + 1e-12).any():
        raise ValueError("Target weights must be between zero and one.")
    if summary["unallocated_cash_weight"].lt(-1e-12).any():
        raise ValueError("Unallocated capacity cash cannot be negative.")
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
        "- 目标持仓等权；若启用 ADV 容量，单票权重再按信号日 20 日 ADV × 上限比例 ÷ 初始资金截断，目标权重之和可小于 1，差额为容量预留现金。",
        "- 执行日为信号日后的首个市场交易日，执行价格口径为开盘；可执行需有该日交易记录、`trading_status=1`、开盘价有效且开盘不处于涨停（买入）或跌停（卖出）。",
        "- 继续入选股票不视为无委托；回测在每个执行日按开盘前组合净值重新评估等权调整。",
        "- 不可买入不替代、不跨日追单；完全退出但不可卖出的持仓由回测逐日重试。",
        "- 2025-12-31 的目标候选保留，但因数据末端没有后续执行日而不标记为已执行。", "", "## 输出核对", "",
        f"- 目标持仓：{len(outputs.targets):,} 条；调仓动作：{len(outputs.actions):,} 条；逐期汇总：{len(summary):,} 条。",
        f"- 目标权重之和平均 {summary['target_weight_sum'].mean():.6f}；容量预留现金合计 {summary['unallocated_cash_weight'].sum():.6f}；不可买入对应的目标现金权重合计 {summary['unfilled_buy_cash_weight'].sum():.6f}。",
        f"- 触发 ADV 容量上限的目标记录数：{int(summary['capacity_capped_count'].sum()):,}。", "",
    ]
    capacity_path = config.processed_dir / "capacity_summary.csv"
    if capacity_path.exists():
        capacity = pd.read_csv(capacity_path)
        lines.extend([
            "## ADV 容量",
            "",
            f"按 `max_aum = min_i(ADV_i × {config.adv_fraction:.0%} ÷ 目标权重_i)`，初始资金 {config.initial_capital:,.0f} 元。",
            "",
            "| 因子 | 期数 | 最小 max AUM | 中位 max AUM | 最大 max AUM | 当前资金绑定行 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in capacity.itertuples(index=False):
            lines.append(
                f"| {row.factor} | {row.periods} | {row.max_aum_min:,.0f} | {row.max_aum_median:,.0f} | "
                f"{row.max_aum_max:,.0f} | {int(row.capped_target_rows):,} |"
            )
        lines.append("")
    lines.extend(["", "## 限制", "", "- 本步骤只形成目标组合和执行审计，不计算实际市值权重、现金收益、组合收益或延迟卖出后的成交结果。", "- 涨跌停按开盘价与交易所昨收重建的限价判断；开盘未封板但盘中被封、或一字板打开后的成交仍未建模。", "- ADV 是目标权重级容量上限，不是逐笔冲击/盘口深度模型。", ""])
    return "\n".join(lines)


def run(config: PortfolioConfig) -> PortfolioOutputs:
    """Read the factor data, write portfolio artifacts, and return them in memory."""
    directory = config.processed_dir
    factor_details = pd.read_parquet(directory / "factor_details.parquet")
    daily = pd.read_parquet(directory / "daily_trading.parquet", columns=[
        "stock_code", "trade_date", "trading_status", "open_price", "quality_flags",
        "Dnvaltrd", "PreClosePrice", "Markettype",
    ])
    outputs = build_portfolio_outputs(factor_details, daily, config)
    validate_outputs(outputs, config)
    outputs.targets.to_parquet(directory / "portfolio_targets.parquet", index=False)
    outputs.actions.to_parquet(directory / "portfolio_rebalance_actions.parquet", index=False)
    outputs.period_summary.to_csv(directory / "portfolio_period_summary.csv", index=False, encoding="utf-8-sig")
    capacity = build_capacity_summary(outputs.targets, config)
    if not capacity.empty:
        capacity.to_csv(directory / "capacity_summary.csv", index=False, encoding="utf-8-sig")
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
