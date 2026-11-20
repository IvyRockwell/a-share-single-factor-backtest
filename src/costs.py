"""Apply commission, stamp duty, and slippage to the actual fills."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ACTUAL_STATUSES = {"executed", "partially_filled_cash_constrained"}


@dataclass(frozen=True)
class CostConfig:
    processed_dir: Path
    commission_rate: float
    stamp_duty_rates: tuple[tuple[pd.Timestamp, float], ...]
    slippage_scenarios_bp: tuple[int, ...]
    minimum_commission: float | None = None
    initial_capital: float = 100_000_000.0


@dataclass(frozen=True)
class CostOutputs:
    costs: pd.DataFrame
    daily: pd.DataFrame
    monthly: pd.DataFrame
    breakeven: pd.DataFrame


def load_cost_config(config_path: Path) -> CostConfig:
    """Load the commission, stamp-duty, and slippage assumptions."""
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    costs = settings["costs"]
    commission = costs["commission_rate"]
    minimum = costs.get("minimum_commission")
    research = settings.get("research") or {}
    initial_capital = float(research.get("initial_capital", 100_000_000.0))
    if commission is None:
        raise ValueError("costs require a percentage commission rate.")
    if minimum is not None and float(minimum) < 0:
        raise ValueError("minimum_commission must be non-negative.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    stamp = tuple(sorted((pd.Timestamp(item["start_date"]), float(item["sell_rate"])) for item in costs["stamp_duty_rates"]))
    scenarios = tuple(int(value) for value in costs["slippage_scenarios_bp"])
    if not stamp or not scenarios or any(value < 0 for value in scenarios):
        raise ValueError("cost assumptions are incomplete.")
    return CostConfig(
        config_path.parent.parent / settings["data"]["processed_dir"],
        float(commission),
        stamp,
        scenarios,
        float(minimum) if minimum is not None else None,
        initial_capital,
    )


def _stamp_duty_rate(day: pd.Timestamp, rates: tuple[tuple[pd.Timestamp, float], ...]) -> float:
    eligible = [rate for start, rate in rates if start <= day]
    if not eligible:
        raise ValueError(f"No stamp-duty rule is configured for {day:%Y-%m-%d}.")
    return eligible[-1]


def build_cost_outputs(daily: pd.DataFrame, rebalance_records: pd.DataFrame, config: CostConfig) -> CostOutputs:
    """Apply costs to positive backtest fills without creating any new trades."""
    base = daily.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base = base.sort_values(["factor", "trade_date"])
    base["prior_gross_nav"] = base.groupby("factor")["nav"].shift().fillna(1.0)

    records = rebalance_records.copy()
    records["trade_date"] = pd.to_datetime(records["trade_date"])
    fills = records.loc[
        records["status"].isin(ACTUAL_STATUSES) & records["executed_open_value"].gt(0),
        ["factor", "trade_date", "event_type", "executed_open_value"],
    ].copy()
    fills["buy_amount"] = fills["executed_open_value"].where(fills["event_type"].eq("buy"), 0.0)
    fills["sell_amount"] = fills["executed_open_value"].where(fills["event_type"].eq("sell"), 0.0)
    fills = fills.merge(
        base[["factor", "trade_date", "prior_gross_nav"]],
        on=["factor", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    if fills["prior_gross_nav"].isna().any():
        raise ValueError("An actual fill has no matching backtest daily NAV.")
    minimum = float(config.minimum_commission) if config.minimum_commission is not None else 0.0
    notional = fills["executed_open_value"] * config.initial_capital * fills["prior_gross_nav"]
    commission_yuan = np.maximum(notional * config.commission_rate, minimum)
    fills["commission_nav"] = commission_yuan / (config.initial_capital * fills["prior_gross_nav"])
    totals = fills.groupby(["factor", "trade_date"], as_index=False).agg(
        actual_buy_amount=("buy_amount", "sum"),
        actual_sell_amount=("sell_amount", "sum"),
        actual_fill_count=("executed_open_value", "size"),
        commission=("commission_nav", "sum"),
    )
    totals["actual_trade_amount"] = totals["actual_buy_amount"] + totals["actual_sell_amount"]

    rows: list[dict[str, object]] = []
    for row in totals.itertuples(index=False):
        prior = base.loc[(base["factor"].eq(row.factor)) & (base["trade_date"].eq(row.trade_date)), "prior_gross_nav"]
        if prior.empty:
            raise ValueError("An actual fill has no matching backtest daily NAV.")
        prior_nav = float(prior.iloc[0])
        stamp_rate = _stamp_duty_rate(row.trade_date, config.stamp_duty_rates)
        commission = row.commission
        stamp_duty = row.actual_sell_amount * stamp_rate
        for scenario_bp in config.slippage_scenarios_bp:
            slippage = row.actual_trade_amount * scenario_bp / 10_000
            rows.append({
                "factor": row.factor,
                "trade_date": row.trade_date,
                "slippage_bp": scenario_bp,
                "actual_buy_amount": row.actual_buy_amount,
                "actual_sell_amount": row.actual_sell_amount,
                "actual_trade_amount": row.actual_trade_amount,
                "actual_fill_count": row.actual_fill_count,
                "prior_gross_nav": prior_nav,
                "turnover": row.actual_trade_amount / prior_nav,
                "commission": commission,
                "stamp_duty": stamp_duty,
                "slippage": slippage,
                "total_transaction_cost": commission + stamp_duty + slippage,
            })
    costs = pd.DataFrame(rows)
    if costs.empty:
        raise ValueError("No actual fills are available for cost calculation.")

    scenario_daily: list[pd.DataFrame] = []
    for scenario_bp in config.slippage_scenarios_bp:
        item = base.copy()
        item["slippage_bp"] = scenario_bp
        item = item.merge(costs.loc[costs["slippage_bp"].eq(scenario_bp)], on=["factor", "trade_date", "slippage_bp"], how="left")
        for column in ("actual_buy_amount", "actual_sell_amount", "actual_trade_amount", "actual_fill_count", "turnover", "commission", "stamp_duty", "slippage", "total_transaction_cost"):
            item[column] = item[column].fillna(0.0)
        net_navs, net_returns = [], []
        for _, factor_days in item.groupby("factor", sort=False):
            prior_net_nav = 1.0
            for row in factor_days.itertuples(index=False):
                net_nav = prior_net_nav * (1 + row.gross_return) - row.total_transaction_cost
                if net_nav <= 0:
                    raise ValueError("Transaction costs exhausted the portfolio value.")
                net_navs.append(net_nav)
                net_returns.append(net_nav / prior_net_nav - 1)
                prior_net_nav = net_nav
        item["net_return"] = net_returns
        item["net_nav"] = net_navs
        scenario_daily.append(item)
    daily_output = pd.concat(scenario_daily, ignore_index=True).drop(columns="prior_gross_nav_y").rename(columns={"prior_gross_nav_x": "prior_gross_nav"})
    daily_output["month"] = daily_output["trade_date"].dt.to_period("M").astype(str)
    monthly = daily_output.groupby(["factor", "slippage_bp", "month"], as_index=False).agg(
        gross_monthly_return=("gross_return", lambda values: (1 + values).prod() - 1),
        gross_end_nav=("nav", "last"),
        net_monthly_return=("net_return", lambda values: (1 + values).prod() - 1),
        net_end_nav=("net_nav", "last"),
        transaction_cost=("total_transaction_cost", "sum"),
        benchmark_monthly_return=("benchmark_return", lambda values: (1 + values).prod() - 1),
        benchmark_end_nav=("benchmark_nav", "last"),
    )
    breakeven = build_breakeven(costs, daily_output)
    return CostOutputs(
        costs.sort_values(["factor", "slippage_bp", "trade_date"]),
        daily_output.drop(columns="month"),
        monthly,
        breakeven,
    )


def build_breakeven(costs: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Approximate the one-way slippage that zeroes net NAV.

    The extra cost is linear in slippage, so the break-even point is the
    surplus net-of-commission NAV divided by cumulative traded notional.  This
    ignores compounding and is a ranking/scale diagnostic, not an exact root.
    """
    rows: list[dict[str, object]] = []
    for factor in sorted(costs["factor"].unique()):
        base = daily.loc[daily["factor"].eq(factor) & daily["slippage_bp"].eq(0)].sort_values("trade_date")
        cost0 = costs.loc[costs["factor"].eq(factor) & costs["slippage_bp"].eq(0)]
        if base.empty:
            continue
        net0 = float(base["net_nav"].iloc[-1])
        gross = float(base["nav"].iloc[-1])
        benchmark = float(base["benchmark_nav"].iloc[-1])
        trade_amount = float(cost0["actual_trade_amount"].sum())
        commission_stamp = float(cost0["commission"].sum() + cost0["stamp_duty"].sum())
        rows.append({
            "factor": factor,
            "gross_end_nav": gross,
            "net0_end_nav": net0,
            "benchmark_end_nav": benchmark,
            "commission_stamp_cost": commission_stamp,
            "cumulative_trade_amount": trade_amount,
            "breakeven_slippage_bp": (net0 - 1.0) / trade_amount * 10_000 if trade_amount > 0 else np.nan,
            "breakeven_vs_h00300_bp": (net0 - benchmark) / trade_amount * 10_000 if trade_amount > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def validate_outputs(outputs: CostOutputs) -> None:
    if outputs.costs["turnover"].lt(0).any() or outputs.costs["total_transaction_cost"].lt(0).any():
        raise ValueError("Costs and turnover must be non-negative.")
    if outputs.daily["net_nav"].le(0).any() or outputs.daily["net_nav"].gt(outputs.daily["nav"] + 1e-12).any():
        raise ValueError("Net NAV must be positive and no greater than gross NAV.")
    if outputs.daily.duplicated(["factor", "trade_date", "slippage_bp"]).any():
        raise ValueError("Duplicate daily cost rows.")


def _format_bp(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1f} bp"


def render_report(outputs: CostOutputs, config: CostConfig) -> str:
    lines = ["# 交易成本", "", "## 成本口径", "", "- 成本只对回测调仓记录中状态为 `executed` 或 `partially_filled_cash_constrained` 且实际开盘成交金额大于 0 的逐笔记录计提；不按目标权重重建成交。", f"- 佣金：买卖双边各 {config.commission_rate * 10_000:.0f} bp；每笔最低 {config.minimum_commission if config.minimum_commission is not None else 0:.2f} 元（初始资金 {config.initial_capital:,.0f} 元）。", "- 印花税：仅卖出，2014-01-01 至 2023-08-27 为 10 bp，2023-08-28 起为 5 bp。", "- 滑点：买卖双边各 0 bp、10 bp、20 bp；因此 0 bp 情景仍包含佣金与印花税。", "- 换手率 = 当日实际买入金额与实际卖出金额之和 ÷ 前一交易日回测毛净值；首日分母为 1。", "", "## 参数来源", "", "- 印花税历史规则：[财政部 2008 年单边征收公告](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm)；[财政部、税务总局 2023 年减半公告](https://shanxi.chinatax.gov.cn/web/detail/sx-11400-545-1780448)。", "- 佣金 3 bp 与滑点情景均为本研究确认的假设；非特定券商实际费率。", "", "## 情景结果", "", "| 因子 | 滑点情景 | 期末毛净值 | 期末净值 | 累计毛收益 | 累计净收益 | 成本合计 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    latest = outputs.daily.sort_values("trade_date").groupby(["factor", "slippage_bp"], as_index=False).tail(1)
    cost_totals = outputs.costs.groupby(["factor", "slippage_bp"], as_index=False)["total_transaction_cost"].sum().rename(columns={"total_transaction_cost": "cumulative_transaction_cost"})
    for row in latest.merge(cost_totals, on=["factor", "slippage_bp"]).sort_values(["factor", "slippage_bp"]).itertuples(index=False):
        lines.append(f"| {row.factor} | {row.slippage_bp} bp | {row.nav:.6f} | {row.net_nav:.6f} | {row.nav - 1:.6f} | {row.net_nav - 1:.6f} | {row.cumulative_transaction_cost:.6f} |")
    lines.extend([
        "",
        "## 盈亏平衡滑点",
        "",
        "近似口径：`(net0 期末净值 - 1) / 累计成交额 × 10000`；相对 H00300 用 `(net0 - 基准期末净值) / 累计成交额 × 10000`。忽略复利路径，只作量级和排序。",
        "",
        "| 因子 | 毛期末 | net0 期末 | 佣金+印花税 | 累计成交额 | 盈亏平衡滑点 | 相对 H00300 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.breakeven.sort_values("factor").itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.gross_end_nav:.6f} | {row.net0_end_nav:.6f} | "
            f"{row.commission_stamp_cost:.6f} | {row.cumulative_trade_amount:.6f} | "
            f"{_format_bp(row.breakeven_slippage_bp)} | {_format_bp(row.breakeven_vs_h00300_bp)} |"
        )
    lines.extend([
        "",
        "## 限制",
        "",
        "- 佣金 3 bp / 最低 5 元 / 初始资金 1 亿元是研究假设，不代表某券商实际费率或真实账户规模。",
        "- 涨跌停按开盘价与交易所昨收重建的限价拒单；开盘未封板但盘中被封、或一字板打开后的成交仍未建模。",
        "- 2025-12-31 无后续执行日，未进入实际回测；买入未成交留现金，待卖出恢复可交易后卖出的既有规则不变。",
        "- 日频数据无法用盘口或逐笔成交估计真实滑点；现金收益为 0，未加入融资或借款。",
        "",
    ])
    return "\n".join(lines)


def run(config: CostConfig) -> CostOutputs:
    p = config.processed_dir
    outputs = build_cost_outputs(pd.read_parquet(p / "backtest_daily_returns.parquet"), pd.read_parquet(p / "backtest_rebalance_records.parquet"), config)
    validate_outputs(outputs)
    outputs.costs.to_parquet(p / "backtest_costs.parquet", index=False)
    outputs.daily.to_parquet(p / "backtest_net_daily_returns.parquet", index=False)
    outputs.monthly.to_csv(p / "backtest_net_monthly_returns.csv", index=False, encoding="utf-8-sig")
    outputs.breakeven.to_csv(p / "backtest_breakeven.csv", index=False, encoding="utf-8-sig")
    report_dir = p.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "costs.md").write_text(render_report(outputs, config), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the transaction-cost analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_cost_config(args.config))
    print(f"Built {len(outputs.costs):,} cost rows and {len(outputs.daily):,} daily scenario rows.")


if __name__ == "__main__":
    main()
