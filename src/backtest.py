"""Run the Stage 7 gross-return historical backtest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FACTORS = ("pe", "momentum", "roe")


@dataclass(frozen=True)
class BacktestConfig:
    processed_dir: Path
    normal_trading_status: str


@dataclass(frozen=True)
class BacktestOutputs:
    daily: pd.DataFrame
    monthly: pd.DataFrame
    holdings: pd.DataFrame
    rebalance_records: pd.DataFrame


def load_backtest_config(config_path: Path) -> BacktestConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    return BacktestConfig(config_path.parent.parent / settings["data"]["processed_dir"], str(settings["universe"]["normal_trading_status"]))


def _valid_return(row: pd.Series | None) -> bool:
    return row is not None and pd.notna(row["total_return"]) and "invalid_total_return" not in str(row.get("quality_flags", ""))


def _can_trade(row: pd.Series | None, normal: str) -> bool:
    return _valid_return(row) and str(row["trading_status"]) == normal and all(pd.notna(row[x]) and float(row[x]) > 0 for x in ("open_price", "close_price", "PreClosePrice")) and "invalid_open_price" not in str(row.get("quality_flags", ""))


def _open_factor(row: pd.Series | None) -> float:
    if not _valid_return(row) or float(row["open_price"]) <= 0 or float(row["close_price"]) <= 0:
        return 1.0
    return (1 + float(row["total_return"])) * float(row["open_price"]) / float(row["close_price"])


def _entry_factor(row: pd.Series) -> float:
    return (1 + float(row["total_return"])) * float(row["PreClosePrice"]) / float(row["open_price"])


def build_backtest_outputs(targets: pd.DataFrame, actions: pd.DataFrame, daily: pd.DataFrame, benchmark: pd.DataFrame, config: BacktestConfig) -> BacktestOutputs:
    """Simulate Stage 6 orders with natural drift, cash, and pending sales."""
    executable_dates = targets.loc[targets["execution_date"].notna(), "execution_date"]
    if executable_dates.empty:
        raise ValueError("No execution dates are available.")
    start = executable_dates.min()
    daily = daily.loc[daily["trade_date"].ge(start)].copy()
    daily["PreClosePrice"] = pd.to_numeric(daily["PreClosePrice"], errors="coerce")
    daily = daily.sort_values(["trade_date", "stock_code"])
    actions = actions.loc[actions["execution_date"].notna()].copy()
    by_action_day = {(f, d): x for (f, d), x in actions.groupby(["factor", "execution_date"], sort=False)}
    bench = benchmark.loc[benchmark["trade_date"].ge(start)].copy().set_index("trade_date")
    bench_nav = 1.0
    bench_values: dict[pd.Timestamp, tuple[float, float]] = {}
    for day, row in bench.iterrows():
        ret = float(row["total_return"]) if _valid_return(row) else 0.0
        bench_nav *= 1 + ret
        bench_values[day] = (ret, bench_nav)
    states = {f: {"cash": 1.0, "positions": {}, "pending": set(), "prior_nav": 1.0} for f in FACTORS}
    daily_rows, holding_rows, record_rows = [], [], []
    for day, group in daily.groupby("trade_date", sort=False):
        day_frame = group.set_index("stock_code", drop=False)
        bench_ret, bench_value = bench_values.get(day, (0.0, np.nan))
        for factor in FACTORS:
            state = states[factor]
            positions: dict[str, float] = state["positions"]
            events = by_action_day.get((factor, day), pd.DataFrame())
            signal = events["signal_date"].iloc[0] if not events.empty else pd.NaT
            new_codes = set(events.loc[events["target_action"].eq("new_buy"), "stock_code"]) if not events.empty else set()
            needed_codes = set(positions) | set(state["pending"]) | new_codes
            rows = {code: day_frame.loc[code] for code in needed_codes if code in day_frame.index}
            opening = {code: value * _open_factor(rows.get(code)) for code, value in positions.items()}
            state["pending"].difference_update(new_codes)
            if not events.empty:
                state["pending"].update(code for code in events.loc[events["target_action"].eq("planned_sell"), "stock_code"] if code in positions and code not in new_codes)
            for code in sorted(list(state["pending"])):
                row = rows.get(code)
                if _can_trade(row, config.normal_trading_status):
                    value = opening.get(code, positions[code])
                    state["cash"] += value
                    positions.pop(code); opening.pop(code, None); state["pending"].remove(code)
                    record_rows.append({"factor": factor, "trade_date": day, "signal_date": signal, "stock_code": code, "event_type": "sell", "status": "executed", "reason": "normal_trading_status_and_valid_open_price", "target_weight": np.nan, "desired_open_value": np.nan, "executed_open_value": value, "cash_scale": np.nan})
                elif not events.empty:
                    record_rows.append({"factor": factor, "trade_date": day, "signal_date": signal, "stock_code": code, "event_type": "sell", "status": "pending_sale", "reason": "missing_or_non_normal_trading_or_invalid_open", "target_weight": np.nan, "desired_open_value": np.nan, "executed_open_value": 0.0, "cash_scale": np.nan})
            buy_events = events.loc[events["target_action"].eq("new_buy")] if not events.empty else pd.DataFrame()
            opening_nav = state["cash"] + sum(opening.values())
            candidates = []
            for _, action in buy_events.iterrows():
                code, desired, row = action["stock_code"], opening_nav * float(action["target_weight"]), rows.get(action["stock_code"])
                if code in positions:
                    record_rows.append({"factor": factor, "trade_date": day, "signal_date": signal, "stock_code": code, "event_type": "buy", "status": "no_order_already_held", "reason": "pending_sale_cancelled_by_new_target", "target_weight": action["target_weight"], "desired_open_value": desired, "executed_open_value": 0.0, "cash_scale": np.nan})
                elif str(action["execution_status"]) == "executable" and _can_trade(row, config.normal_trading_status):
                    candidates.append((action, desired, row))
                else:
                    record_rows.append({"factor": factor, "trade_date": day, "signal_date": signal, "stock_code": code, "event_type": "buy", "status": "unfilled_cash", "reason": str(action["execution_reason"]), "target_weight": action["target_weight"], "desired_open_value": desired, "executed_open_value": 0.0, "cash_scale": 0.0})
            desired_sum = sum(x[1] for x in candidates)
            scale = min(1.0, state["cash"] / desired_sum) if desired_sum else 0.0
            bought = set()
            for action, desired, row in candidates:
                code, value = action["stock_code"], desired * scale
                state["cash"] -= value
                positions[code] = value * _entry_factor(row); bought.add(code)
                record_rows.append({"factor": factor, "trade_date": day, "signal_date": signal, "stock_code": code, "event_type": "buy", "status": "executed" if scale == 1 else "partially_filled_cash_constrained", "reason": "normal_trading_status_and_valid_open_price" if scale == 1 else "cash_constrained_by_pending_or_existing_positions", "target_weight": action["target_weight"], "desired_open_value": desired, "executed_open_value": value, "cash_scale": scale})
            frozen = 0
            for code in list(positions):
                if code not in bought:
                    row = rows.get(code)
                    if _valid_return(row): positions[code] *= 1 + float(row["total_return"])
                    else: frozen += 1
            equity, nav = sum(positions.values()), state["cash"] + sum(positions.values())
            daily_rows.append({"factor": factor, "trade_date": day, "gross_return": nav / state["prior_nav"] - 1, "nav": nav, "cash_value": state["cash"], "equity_value": equity, "held_stock_count": len(positions), "pending_sale_count": len(state["pending"]), "frozen_valuation_count": frozen, "benchmark_return": bench_ret, "benchmark_nav": bench_value})
            state["prior_nav"] = nav
            for code, value in positions.items():
                holding_rows.append({"factor": factor, "trade_date": day, "stock_code": code, "close_value": value, "portfolio_weight": value / nav, "is_pending_sale": code in state["pending"], "valuation_status": "updated_with_total_return" if _valid_return(rows.get(code)) else "frozen_missing_or_invalid_return"})
    daily_output = pd.DataFrame(daily_rows)
    daily_output["month"] = daily_output["trade_date"].dt.to_period("M").astype(str)
    monthly = daily_output.groupby(["factor", "month"], as_index=False).agg(monthly_return=("gross_return", lambda x: (1 + x).prod() - 1), end_nav=("nav", "last"), benchmark_monthly_return=("benchmark_return", lambda x: (1 + x).prod() - 1), benchmark_end_nav=("benchmark_nav", "last"))
    return BacktestOutputs(daily_output.drop(columns="month"), monthly, pd.DataFrame(holding_rows), pd.DataFrame(record_rows))


def validate_outputs(outputs: BacktestOutputs) -> None:
    if set(outputs.daily["factor"].unique()) != set(FACTORS) or outputs.daily.duplicated(["factor", "trade_date"]).any(): raise ValueError("Invalid daily series.")
    if outputs.daily[["nav", "cash_value", "equity_value"]].isna().any().any() or outputs.daily["nav"].le(0).any(): raise ValueError("Invalid NAV.")
    if outputs.monthly.duplicated(["factor", "month"]).any(): raise ValueError("Duplicate monthly rows.")


def render_report(outputs: BacktestOutputs) -> str:
    daily, monthly, records = outputs.daily, outputs.monthly, outputs.rebalance_records
    lines = ["# 阶段 7：历史回测", "", "## 固定规则", "", "- 初始净值为 1，现金收益为 0；不计交易成本、融资、整手约束、Sharpe 或稳健性测试。", "- 信号后首个市场交易日开盘执行；继续持仓自然漂移，不可买入留现金，待卖出仅在恢复正常交易且开盘价有效时卖出。", "- 买入首日收益为 `(1 + Dretwd) × PreClosePrice / open_price - 1`；整日持有使用 `Dretwd`。", "", "## 结果汇总", "", "| 因子 | 起始日 | 结束日 | 期末净值 | 累计毛收益 | 月度记录数 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for factor, item in daily.groupby("factor", sort=True): lines.append(f"| {factor} | {item.trade_date.min():%Y-%m-%d} | {item.trade_date.max():%Y-%m-%d} | {item.nav.iloc[-1]:.6f} | {item.nav.iloc[-1] - 1:.6f} | {len(monthly.loc[monthly.factor.eq(factor)])} |")
    last = daily.loc[daily.factor.eq("pe")].iloc[-1]
    lines.extend(["", f"- H00300 同期净值：{last.benchmark_nav:.6f}；累计收益 {last.benchmark_nav - 1:.6f}。", f"- 冻结估值持仓日：{int(daily.frozen_valuation_count.sum()):,}；期末待卖出持仓：{int(daily.loc[daily.trade_date.eq(last.trade_date), 'pending_sale_count'].sum()):,}。", "", "## 固定抽样复核", "", "| 因子 | 日期 | 股票 | 事件 | 状态 | 目标开盘金额 | 实际开盘金额 |", "| --- | --- | --- | --- | --- | ---: | ---: |"])
    for row in records.sample(n=min(8, len(records)), random_state=20260916).sort_values(["factor", "trade_date", "stock_code"]).itertuples(index=False): lines.append(f"| {row.factor} | {row.trade_date:%Y-%m-%d} | {row.stock_code} | {row.event_type} | {row.status} | {0 if pd.isna(row.desired_open_value) else row.desired_open_value:.6f} | {row.executed_open_value:.6f} |")
    lines.extend(["", "## 限制", "", "- 2025-12-31 没有后续执行日，未进入实际回测。", "- H00300 缺少可靠开盘价：基准在执行日使用完整 `Idxtrd08` 日收益，策略使用开盘至收盘收益，首日并非严格同一盘中起点。", "- 涨跌停无法可靠识别；正常状态和有效开盘价不构成成交保证。", "- 缺失或无效收益记录冻结最后可得估值；期末未恢复可交易的待卖出持仓不被虚构为已卖出。", ""])
    return "\n".join(lines)


def run(config: BacktestConfig) -> BacktestOutputs:
    p = config.processed_dir
    outputs = build_backtest_outputs(pd.read_parquet(p / "portfolio_targets.parquet"), pd.read_parquet(p / "portfolio_rebalance_actions.parquet"), pd.read_parquet(p / "daily_trading.parquet", columns=["stock_code", "trade_date", "open_price", "close_price", "total_return", "PreClosePrice", "trading_status", "quality_flags"]), pd.read_parquet(p / "benchmark.parquet", columns=["trade_date", "total_return", "quality_flags"]), config)
    validate_outputs(outputs)
    outputs.daily.to_parquet(p / "backtest_daily_returns.parquet", index=False); outputs.monthly.to_csv(p / "backtest_monthly_returns.csv", index=False, encoding="utf-8-sig"); outputs.holdings.to_parquet(p / "backtest_holdings.parquet", index=False); outputs.rebalance_records.to_parquet(p / "backtest_rebalance_records.parquet", index=False)
    (p.parent.parent / "reports" / "backtest_stage7.md").write_text(render_report(outputs), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Stage 7 gross-return backtest."); parser.add_argument("--config", type=Path, default=Path("configs/config.yaml")); args = parser.parse_args(argv)
    outputs = run(load_backtest_config(args.config)); print(f"Built {len(outputs.daily):,} daily rows and {len(outputs.holdings):,} holding snapshots.")


if __name__ == "__main__": main()
