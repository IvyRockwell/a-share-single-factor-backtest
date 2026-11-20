"""Run the gross-return historical backtest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.trading_rules import trade_block_reason

FACTORS = ("pe", "momentum", "roe")
RELATIVE_VALUE_TOLERANCE = 1e-12


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


def build_universe_lookups(pool: pd.DataFrame, market_cap: pd.DataFrame | None
                           ) -> tuple[dict[pd.Timestamp, set[str]], dict[pd.Timestamp, pd.Series]]:
    """Map each signal date to its member set and, when available, their market caps."""
    members = {day: set(group["stock_code"]) for day, group in pool.groupby("signal_date")}
    weights: dict[pd.Timestamp, pd.Series] = {}
    if market_cap is not None:
        usable = market_cap.loc[market_cap["has_market_cap"]]
        weights = {day: group.set_index("stock_code")["market_cap"] for day, group in usable.groupby("signal_date")}
    return members, weights


def _universe_returns(group: pd.DataFrame, signal: pd.Timestamp,
                      members: dict[pd.Timestamp, set[str]],
                      weights: dict[pd.Timestamp, pd.Series]) -> tuple[float, float]:
    """Equal- and cap-weighted return of the pool on one trading day."""
    codes = members.get(signal)
    if not codes:
        return np.nan, np.nan
    frame = group.loc[group["stock_code"].isin(codes) & _valid_return_mask(group)]
    if frame.empty:
        return np.nan, np.nan
    equal = float(frame["total_return"].mean())
    cap = weights.get(signal)
    if cap is None:
        return equal, np.nan
    frame = frame.assign(weight=cap.reindex(frame["stock_code"]).to_numpy())
    frame = frame.loc[frame["weight"].gt(0)]
    if frame.empty:
        return equal, np.nan
    total = float(frame["weight"].sum())
    return equal, float((frame["weight"] * frame["total_return"]).sum() / total)


def _valid_return_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["total_return"].notna() & ~frame["quality_flags"].fillna("").astype("string").str.contains(
        "invalid_total_return", regex=False)


def _can_trade(row: pd.Series | None, normal: str, side: str = "buy") -> bool:
    return row is not None and trade_block_reason(row, normal, side) is None


def _block_reason(row: pd.Series | None, normal: str, side: str) -> str:
    reason = trade_block_reason(row, normal, side) if row is not None else None
    return reason or "missing_or_non_normal_trading_or_invalid_open"


def _open_factor(row: pd.Series | None) -> float:
    if (
        not _valid_return(row)
        or pd.isna(row["open_price"])
        or pd.isna(row["close_price"])
        or float(row["open_price"]) <= 0
        or float(row["close_price"]) <= 0
    ):
        return 1.0
    return (1 + float(row["total_return"])) * float(row["open_price"]) / float(row["close_price"])


def _close_open_factor(row: pd.Series | None) -> float:
    """Return the price-only open-to-close factor for an opening trade."""
    if row is None or pd.isna(row["open_price"]) or pd.isna(row["close_price"]):
        return 1.0
    if float(row["open_price"]) <= 0 or float(row["close_price"]) <= 0:
        return 1.0
    return float(row["close_price"]) / float(row["open_price"])


def build_backtest_outputs(targets: pd.DataFrame, actions: pd.DataFrame | None, daily: pd.DataFrame,
                           benchmark: pd.DataFrame, config: BacktestConfig,
                           pool: pd.DataFrame | None = None,
                           market_cap: pd.DataFrame | None = None,
                           factors: tuple[str, ...] = FACTORS) -> BacktestOutputs:
    """Restore target equal weights at each open and retain only failed full exits.

    ``factors`` lets the same execution and valuation engine price a synthetic
    benchmark portfolio (all pool members, fixed weights) without changing the
    strategy path.  ``actions`` is retained for API compatibility but the engine
    only needs ``targets``; pass ``None`` for a synthetic benchmark.
    """
    executable_dates = targets.loc[targets["execution_date"].notna(), "execution_date"]
    if executable_dates.empty:
        raise ValueError("No execution dates are available.")
    if not factors:
        raise ValueError("At least one factor is required.")
    start = executable_dates.min()
    daily = daily.loc[daily["trade_date"].ge(start)].copy()
    daily["PreClosePrice"] = pd.to_numeric(daily["PreClosePrice"], errors="coerce")
    daily = daily.sort_values(["trade_date", "stock_code"])
    if actions is not None:
        actions = actions.loc[actions["execution_date"].notna()].copy()
    targets = targets.loc[targets["execution_date"].notna()].copy()
    by_target_day = {(f, d): x for (f, d), x in targets.groupby(["factor", "execution_date"], sort=False)}
    bench = benchmark.loc[benchmark["trade_date"].ge(start)].copy().set_index("trade_date")
    bench_nav = 1.0
    bench_values: dict[pd.Timestamp, tuple[float, float]] = {}
    for day, row in bench.iterrows():
        ret = float(row["total_return"]) if _valid_return(row) else 0.0
        bench_nav *= 1 + ret
        bench_values[day] = (ret, bench_nav)
    members: dict[pd.Timestamp, set[str]] = {}
    cap_weights: dict[pd.Timestamp, pd.Series] = {}
    signal_for_day: dict[pd.Timestamp, pd.Timestamp] = {}
    if pool is not None:
        members, cap_weights = build_universe_lookups(pool, market_cap)
        signal_dates = np.array(sorted(members))
        day_values = np.array(sorted(daily["trade_date"].unique()))
        positions = np.searchsorted(signal_dates, day_values, side="right") - 1
        signal_for_day = {
            day: signal_dates[position] for day, position in zip(day_values, positions) if position >= 0
        }
    states = {f: {"cash": 1.0, "positions": {}, "pending": {}, "prior_nav": 1.0} for f in factors}
    daily_rows, holding_rows, record_rows = [], [], []

    def record(
        factor: str, day: pd.Timestamp, signal_date: pd.Timestamp | pd.NaT, stock_code: str,
        event_type: str, trade_reason: str, status: str, reason: str, target_weight: float,
        pre_trade_open_value: float, target_open_value: float, requested_open_value: float,
        executed_open_value: float, cash_scale: float,
    ) -> None:
        record_rows.append({
            "factor": factor, "trade_date": day, "signal_date": signal_date,
            "stock_code": stock_code, "event_type": event_type, "trade_reason": trade_reason,
            "status": status, "reason": reason, "target_weight": target_weight,
            "pre_trade_open_value": pre_trade_open_value, "target_open_value": target_open_value,
            "requested_open_value": requested_open_value,
            "executed_open_value": executed_open_value, "cash_scale": cash_scale,
        })

    for day, group in daily.groupby("trade_date", sort=False):
        day_frame = group.set_index("stock_code", drop=False)
        bench_ret, bench_value = bench_values.get(day, (0.0, np.nan))
        signal = signal_for_day.get(day)
        ew_ret, cw_ret = (
            _universe_returns(group, signal, members, cap_weights) if signal is not None else (np.nan, np.nan)
        )
        for factor in factors:
            state = states[factor]
            positions: dict[str, float] = state["positions"]
            target_frame = by_target_day.get((factor, day), pd.DataFrame())
            signal = target_frame["signal_date"].iloc[0] if not target_frame.empty else pd.NaT
            target_weights = (
                target_frame.set_index("stock_code")["target_weight"].astype(float).to_dict()
                if not target_frame.empty else {}
            )
            target_codes = set(target_weights)
            pending_before = set(state["pending"])
            if not target_frame.empty:
                for code in target_codes:
                    state["pending"].pop(code, None)
                for code in set(positions) - target_codes:
                    state["pending"].setdefault(code, signal)
            needed_codes = set(positions) | set(state["pending"]) | target_codes
            rows = {code: day_frame.loc[code] for code in needed_codes if code in day_frame.index}
            prior_close = positions.copy()
            opening = {code: value * _open_factor(rows.get(code)) for code, value in positions.items()}
            pre_trade_open = opening.copy()
            traded_at_open: set[str] = set()

            for code in sorted(list(state["pending"])):
                row = rows.get(code)
                if _can_trade(row, config.normal_trading_status, "sell"):
                    value = opening.get(code, positions[code])
                    state["cash"] += value
                    positions.pop(code, None)
                    opening.pop(code, None)
                    pending_signal = state["pending"].pop(code)
                    trade_reason = "pending_exit_recovery" if code in pending_before else "complete_exit"
                    record(factor, day, pending_signal, code, "sell", trade_reason, "executed",
                           "normal_trading_status_and_valid_open_price", 0.0,
                           pre_trade_open.get(code, value), 0.0, value, value, np.nan)
                elif code not in pending_before:
                    record(factor, day, state["pending"][code], code, "sell", "complete_exit", "pending_sale",
                           _block_reason(row, config.normal_trading_status, "sell"), 0.0,
                           pre_trade_open.get(code, positions.get(code, 0.0)), 0.0,
                           pre_trade_open.get(code, positions.get(code, 0.0)), 0.0, np.nan)

            if not target_frame.empty:
                opening_nav = state["cash"] + sum(opening.values())
                value_tolerance = max(1.0, abs(opening_nav)) * RELATIVE_VALUE_TOLERANCE
                buy_candidates: list[tuple[str, float, float, float, str]] = []
                for code in sorted(target_codes):
                    weight = target_weights[code]
                    current = opening.get(code, 0.0)
                    target_value = opening_nav * weight
                    difference = target_value - current
                    if abs(difference) <= value_tolerance:
                        continue
                    row = rows.get(code)
                    if difference < 0:
                        requested = -difference
                        if _can_trade(row, config.normal_trading_status, "sell"):
                            opening[code] = current - requested
                            state["cash"] += requested
                            traded_at_open.add(code)
                            record(factor, day, signal, code, "sell", "rebalance_reduce", "executed",
                                   "normal_trading_status_and_valid_open_price", weight, current,
                                   target_value, requested, requested, np.nan)
                        else:
                            record(factor, day, signal, code, "sell", "rebalance_reduce", "adjustment_not_executed",
                                   _block_reason(row, config.normal_trading_status, "sell"), weight, current,
                                   target_value, requested, 0.0, np.nan)
                    else:
                        trade_reason = "rebalance_increase" if code in positions else "new_entry"
                        if _can_trade(row, config.normal_trading_status, "buy"):
                            buy_candidates.append((code, difference, current, target_value, trade_reason))
                        else:
                            record(factor, day, signal, code, "buy", trade_reason, "unfilled_cash",
                                   _block_reason(row, config.normal_trading_status, "buy"), weight, current,
                                   target_value, difference, 0.0, 0.0)

                requested_sum = sum(item[1] for item in buy_candidates)
                available_cash = max(0.0, state["cash"])
                scale = min(1.0, available_cash / requested_sum) if requested_sum > value_tolerance else 0.0
                for code, requested, current, target_value, trade_reason in buy_candidates:
                    executed = requested * scale
                    if executed > 0:
                        state["cash"] -= executed
                        opening[code] = current + executed
                        positions.setdefault(code, 0.0)
                        traded_at_open.add(code)
                    status = "executed" if scale >= 1 - RELATIVE_VALUE_TOLERANCE else (
                        "partially_filled_cash_constrained" if executed > 0 else "unfilled_cash"
                    )
                    reason = (
                        "normal_trading_status_and_valid_open_price"
                        if status == "executed" else "cash_constrained_by_pending_or_failed_reductions"
                    )
                    record(factor, day, signal, code, "buy", trade_reason, status, reason,
                           target_weights[code], current, target_value, requested, executed, scale)

            frozen = 0
            close_positions: dict[str, float] = {}
            valuation_status: dict[str, str] = {}
            for code in list(positions):
                row = rows.get(code)
                if code in traded_at_open:
                    factor_value = _close_open_factor(row)
                    close_positions[code] = opening[code] * factor_value
                    valid_close = row is not None and pd.notna(row["close_price"]) and float(row["close_price"]) > 0
                    valuation_status[code] = "updated_with_close_open_price" if valid_close else "frozen_at_post_trade_open_value"
                    if not valid_close:
                        frozen += 1
                elif _valid_return(row):
                    close_positions[code] = prior_close[code] * (1 + float(row["total_return"]))
                    valuation_status[code] = "updated_with_total_return"
                else:
                    close_positions[code] = prior_close[code]
                    valuation_status[code] = "frozen_missing_or_invalid_return"
                    frozen += 1
            positions.clear()
            positions.update(close_positions)
            nav_scale = max(1.0, abs(state["prior_nav"]))
            if abs(state["cash"]) <= nav_scale * RELATIVE_VALUE_TOLERANCE:
                state["cash"] = 0.0
            equity, nav = sum(positions.values()), state["cash"] + sum(positions.values())
            daily_rows.append({"factor": factor, "trade_date": day, "gross_return": nav / state["prior_nav"] - 1, "nav": nav, "cash_value": state["cash"], "equity_value": equity, "held_stock_count": len(positions), "pending_sale_count": len(state["pending"]), "frozen_valuation_count": frozen, "benchmark_return": bench_ret, "benchmark_nav": bench_value, "ew_benchmark_return": ew_ret, "cw_benchmark_return": cw_ret})
            state["prior_nav"] = nav
            for code, value in positions.items():
                holding_rows.append({"factor": factor, "trade_date": day, "stock_code": code, "close_value": value, "portfolio_weight": value / nav, "is_pending_sale": code in state["pending"], "valuation_status": valuation_status[code]})
    daily_output = pd.DataFrame(daily_rows)
    daily_output["month"] = daily_output["trade_date"].dt.to_period("M").astype(str)
    monthly = daily_output.groupby(["factor", "month"], as_index=False).agg(monthly_return=("gross_return", lambda x: (1 + x).prod() - 1), end_nav=("nav", "last"), benchmark_monthly_return=("benchmark_return", lambda x: (1 + x).prod() - 1), benchmark_end_nav=("benchmark_nav", "last"), ew_benchmark_monthly_return=("ew_benchmark_return", lambda x: (1 + x.fillna(0)).prod() - 1), cw_benchmark_monthly_return=("cw_benchmark_return", lambda x: (1 + x.fillna(0)).prod() - 1))
    return BacktestOutputs(daily_output.drop(columns="month"), monthly, pd.DataFrame(holding_rows), pd.DataFrame(record_rows))


def validate_outputs(outputs: BacktestOutputs) -> None:
    if set(outputs.daily["factor"].unique()) != set(FACTORS) or outputs.daily.duplicated(["factor", "trade_date"]).any(): raise ValueError("Invalid daily series.")
    if outputs.daily[["nav", "cash_value", "equity_value"]].isna().any().any() or outputs.daily["nav"].le(0).any(): raise ValueError("Invalid NAV.")
    if outputs.daily["cash_value"].lt(-1e-12).any(): raise ValueError("Backtest borrowed cash.")
    if outputs.monthly.duplicated(["factor", "month"]).any(): raise ValueError("Duplicate monthly rows.")
    required = {
        "factor", "trade_date", "signal_date", "stock_code", "event_type", "trade_reason",
        "status", "reason", "target_weight", "pre_trade_open_value", "target_open_value",
        "requested_open_value", "executed_open_value", "cash_scale",
    }
    if not required.issubset(outputs.rebalance_records.columns):
        raise ValueError("Rebalance records are missing required audit fields.")
    failed = ~outputs.rebalance_records["status"].isin({"executed", "partially_filled_cash_constrained"})
    if outputs.rebalance_records.loc[failed, "executed_open_value"].ne(0).any():
        raise ValueError("Failed or unfilled records must have zero executed value.")
    exits = outputs.rebalance_records["trade_reason"].isin({"complete_exit", "pending_exit_recovery"})
    if outputs.rebalance_records.loc[exits, ["target_weight", "target_open_value"]].ne(0).any().any():
        raise ValueError("Complete exits must have zero target weight and target value.")


def render_report(outputs: BacktestOutputs) -> str:
    daily, monthly = outputs.daily, outputs.monthly
    lines = ["# 历史回测", "", "## 固定规则", "", "- 初始净值为 1，现金收益为 0；不计交易成本、融资、整手约束、Sharpe 或稳健性测试。", "- 每个执行日按开盘前组合净值把当期目标组合恢复为等权：先完全退出和超配减仓，再新建和低配加仓。", "- 可执行买单在现金不足时同比例缩放；新建和继续持仓调权失败不跨日追单，只有完全退出失败进入待卖并逐日重试。", "- 开盘交易后的剩余或新增持仓按 `close_price / open_price - 1` 计当日价格收益；未在当日开盘交易的整日持仓使用 `Dretwd`。", "", "## 结果汇总", "", "| 因子 | 起始日 | 结束日 | 期末净值 | 累计毛收益 | 月度记录数 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for factor, item in daily.groupby("factor", sort=True): lines.append(f"| {factor} | {item.trade_date.min():%Y-%m-%d} | {item.trade_date.max():%Y-%m-%d} | {item.nav.iloc[-1]:.6f} | {item.nav.iloc[-1] - 1:.6f} | {len(monthly.loc[monthly.factor.eq(factor)])} |")
    last = daily.loc[daily.factor.eq("pe")].iloc[-1]
    lines.extend(["", f"- H00300 同期净值：{last.benchmark_nav:.6f}；累计收益 {last.benchmark_nav - 1:.6f}。", f"- 冻结估值持仓日：{int(daily.frozen_valuation_count.sum()):,}；期末待卖出持仓：{int(daily.loc[daily.trade_date.eq(last.trade_date), 'pending_sale_count'].sum()):,}。", ""])
    lines.extend(["", "## 限制", "", "- 2025-12-31 没有后续执行日，未进入实际回测。", "- H00300 缺少可靠开盘价：基准在执行日使用完整 `Idxtrd08` 日收益，策略使用开盘至收盘收益，首日并非严格同一盘中起点。", "- 涨跌停无法可靠识别；正常状态和有效开盘价不构成成交保证。", "- 缺失或无效收益记录冻结最后可得估值；期末未恢复可交易的待卖出持仓不被虚构为已卖出。", ""])
    return "\n".join(lines)


def run(config: BacktestConfig) -> BacktestOutputs:
    p = config.processed_dir
    pool = pd.read_parquet(p / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    market_cap = pd.read_parquet(p / "market_cap.parquet", columns=["signal_date", "stock_code", "market_cap", "has_market_cap"])
    outputs = build_backtest_outputs(pd.read_parquet(p / "portfolio_targets.parquet"), pd.read_parquet(p / "portfolio_rebalance_actions.parquet"), pd.read_parquet(p / "daily_trading.parquet", columns=["stock_code", "trade_date", "open_price", "close_price", "total_return", "PreClosePrice", "trading_status", "market_type", "quality_flags"]), pd.read_parquet(p / "benchmark.parquet", columns=["trade_date", "total_return", "quality_flags"]), config, pool, market_cap)
    validate_outputs(outputs)
    outputs.daily.to_parquet(p / "backtest_daily_returns.parquet", index=False); outputs.monthly.to_csv(p / "backtest_monthly_returns.csv", index=False, encoding="utf-8-sig"); outputs.holdings.to_parquet(p / "backtest_holdings.parquet", index=False); outputs.rebalance_records.to_parquet(p / "backtest_rebalance_records.parquet", index=False)
    report_dir = p.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "backtest.md").write_text(render_report(outputs), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the gross-return backtest."); parser.add_argument("--config", type=Path, default=Path("configs/config.yaml")); args = parser.parse_args(argv)
    outputs = run(load_backtest_config(args.config)); print(f"Built {len(outputs.daily):,} daily rows and {len(outputs.holdings):,} holding snapshots.")


if __name__ == "__main__": main()
