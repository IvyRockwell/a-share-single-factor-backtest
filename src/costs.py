"""Run the Stage 8 transaction-cost analysis from Stage 7 actual fills."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


ACTUAL_STATUSES = {"executed", "partially_filled_cash_constrained"}


@dataclass(frozen=True)
class CostConfig:
    processed_dir: Path
    commission_rate: float
    stamp_duty_rates: tuple[tuple[pd.Timestamp, float], ...]
    slippage_scenarios_bp: tuple[int, ...]


@dataclass(frozen=True)
class CostOutputs:
    costs: pd.DataFrame
    daily: pd.DataFrame
    monthly: pd.DataFrame


def load_cost_config(config_path: Path) -> CostConfig:
    """Load the explicitly confirmed Stage 8 assumptions."""
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    costs = settings["costs"]
    commission = costs["commission_rate"]
    minimum = costs.get("minimum_commission")
    if commission is None or minimum is not None:
        raise ValueError("Stage 8 requires a percentage commission rate and no minimum commission.")
    stamp = tuple(sorted((pd.Timestamp(item["start_date"]), float(item["sell_rate"])) for item in costs["stamp_duty_rates"]))
    scenarios = tuple(int(value) for value in costs["slippage_scenarios_bp"])
    if not stamp or not scenarios or any(value < 0 for value in scenarios):
        raise ValueError("Stage 8 cost assumptions are incomplete.")
    return CostConfig(config_path.parent.parent / settings["data"]["processed_dir"], float(commission), stamp, scenarios)


def _stamp_duty_rate(day: pd.Timestamp, rates: tuple[tuple[pd.Timestamp, float], ...]) -> float:
    eligible = [rate for start, rate in rates if start <= day]
    if not eligible:
        raise ValueError(f"No stamp-duty rule is configured for {day:%Y-%m-%d}.")
    return eligible[-1]


def build_cost_outputs(daily: pd.DataFrame, rebalance_records: pd.DataFrame, config: CostConfig) -> CostOutputs:
    """Apply costs to positive Stage 7 fills without creating any new trades."""
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
    totals = fills.groupby(["factor", "trade_date"], as_index=False).agg(
        actual_buy_amount=("buy_amount", "sum"),
        actual_sell_amount=("sell_amount", "sum"),
        actual_fill_count=("executed_open_value", "size"),
    )
    totals["actual_trade_amount"] = totals["actual_buy_amount"] + totals["actual_sell_amount"]

    rows: list[dict[str, object]] = []
    for row in totals.itertuples(index=False):
        prior = base.loc[(base["factor"].eq(row.factor)) & (base["trade_date"].eq(row.trade_date)), "prior_gross_nav"]
        if prior.empty:
            raise ValueError("An actual fill has no matching Stage 7 daily NAV.")
        prior_nav = float(prior.iloc[0])
        stamp_rate = _stamp_duty_rate(row.trade_date, config.stamp_duty_rates)
        commission = row.actual_trade_amount * config.commission_rate
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
        raise ValueError("No actual Stage 7 fills are available for cost calculation.")

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
    return CostOutputs(costs.sort_values(["factor", "slippage_bp", "trade_date"]), daily_output.drop(columns="month"), monthly)


def validate_outputs(outputs: CostOutputs) -> None:
    if outputs.costs["turnover"].lt(0).any() or outputs.costs["total_transaction_cost"].lt(0).any():
        raise ValueError("Costs and turnover must be non-negative.")
    if outputs.daily["net_nav"].le(0).any() or outputs.daily["net_nav"].gt(outputs.daily["nav"] + 1e-12).any():
        raise ValueError("Net NAV must be positive and no greater than gross NAV.")
    if outputs.daily.duplicated(["factor", "trade_date", "slippage_bp"]).any():
        raise ValueError("Duplicate Stage 8 daily rows.")


def render_report(outputs: CostOutputs, config: CostConfig) -> str:
    lines = ["# 阶段 8：交易成本", "", "## 成本口径", "", "- 成本只对阶段 7 调仓记录中状态为 `executed` 或 `partially_filled_cash_constrained` 且实际开盘成交金额大于 0 的逐笔记录计提；不按目标权重重建成交。", f"- 佣金：买卖双边各 {config.commission_rate * 10_000:.0f} bp；不计最低佣金。", "- 印花税：仅卖出，2014-01-01 至 2023-08-27 为 10 bp，2023-08-28 起为 5 bp。", "- 滑点：买卖双边各 0 bp、10 bp、20 bp；因此 0 bp 情景仍包含佣金与印花税。", "- 换手率 = 当日实际买入金额与实际卖出金额之和 ÷ 前一交易日阶段 7 毛净值；首日分母为 1。", "", "## 参数来源", "", "- 印花税历史规则：[财政部 2008 年单边征收公告](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm)；[财政部、税务总局 2023 年减半公告](https://shanxi.chinatax.gov.cn/web/detail/sx-11400-545-1780448)。", "- 佣金 3 bp 与滑点情景均为本研究确认的假设；非特定券商实际费率。", "", "## 情景结果", "", "| 因子 | 滑点情景 | 期末毛净值 | 期末净值 | 累计毛收益 | 累计净收益 | 成本合计 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    latest = outputs.daily.sort_values("trade_date").groupby(["factor", "slippage_bp"], as_index=False).tail(1)
    cost_totals = outputs.costs.groupby(["factor", "slippage_bp"], as_index=False)["total_transaction_cost"].sum().rename(columns={"total_transaction_cost": "cumulative_transaction_cost"})
    for row in latest.merge(cost_totals, on=["factor", "slippage_bp"]).sort_values(["factor", "slippage_bp"]).itertuples(index=False):
        lines.append(f"| {row.factor} | {row.slippage_bp} bp | {row.nav:.6f} | {row.net_nav:.6f} | {row.nav - 1:.6f} | {row.net_nav - 1:.6f} | {row.cumulative_transaction_cost:.6f} |")
    lines.extend(["", "## 抽样复核", "", "| 因子 | 日期 | 滑点 | 买入额 | 卖出额 | 换手率 | 佣金 | 印花税 | 滑点成本 | 净成本 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    samples = outputs.costs.loc[outputs.costs["slippage_bp"].eq(10)].sample(n=min(8, len(outputs.costs.loc[outputs.costs["slippage_bp"].eq(10)])), random_state=20260916).sort_values(["factor", "trade_date"])
    for row in samples.itertuples(index=False):
        lines.append(f"| {row.factor} | {row.trade_date:%Y-%m-%d} | {row.slippage_bp} bp | {row.actual_buy_amount:.6f} | {row.actual_sell_amount:.6f} | {row.turnover:.6f} | {row.commission:.6f} | {row.stamp_duty:.6f} | {row.slippage:.6f} | {row.total_transaction_cost:.6f} |")
    lines.extend(["", "## 限制", "", "- 佣金 3 bp 和滑点情景是研究假设，不代表某券商实际费率。缺少真实初始资金、股数和订单金额，无法判断最低 5 元佣金，故不纳入。", "- 涨跌停无法可靠识别；正常状态和有效开盘价不等于必然成交。成本仅对阶段 7 已记录的实际成交计提。", "- 2025-12-31 无后续执行日，未进入实际回测；买入未成交留现金，待卖出恢复可交易后卖出的既有规则不变。", "- 日频数据无法用盘口或逐笔成交估计真实滑点；现金收益为 0，未加入融资或借款。", ""])
    return "\n".join(lines)


def run(config: CostConfig) -> CostOutputs:
    p = config.processed_dir
    outputs = build_cost_outputs(pd.read_parquet(p / "backtest_daily_returns.parquet"), pd.read_parquet(p / "backtest_rebalance_records.parquet"), config)
    validate_outputs(outputs)
    outputs.costs.to_parquet(p / "backtest_stage8_costs.parquet", index=False)
    outputs.daily.to_parquet(p / "backtest_stage8_daily_returns.parquet", index=False)
    outputs.monthly.to_csv(p / "backtest_stage8_monthly_returns.csv", index=False, encoding="utf-8-sig")
    (p.parent.parent / "reports" / "backtest_stage8.md").write_text(render_report(outputs, config), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Stage 8 transaction-cost analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_cost_config(args.config))
    print(f"Built {len(outputs.costs):,} cost rows and {len(outputs.daily):,} daily scenario rows.")


if __name__ == "__main__":
    main()
