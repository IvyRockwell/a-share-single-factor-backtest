"""Run the parameter and cost sensitivity grid."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import BacktestConfig, BacktestOutputs, build_backtest_outputs
from src.backtest import validate_outputs as validate_backtest
from src.costs import CostConfig, CostOutputs, build_cost_outputs
from src.costs import validate_outputs as validate_costs
from src.performance import PerformanceConfig, PerformanceOutputs, build_performance_outputs
from src.performance import validate_outputs as validate_performance
from src.portfolio import (
    FACTORS,
    PortfolioConfig,
    PortfolioOutputs,
    build_execution_dates,
    build_portfolio_outputs,
)
from src.portfolio import (
    validate_outputs as validate_portfolio,
)


@dataclass(frozen=True)
class RobustnessConfig:
    processed_dir: Path
    selection_quantiles: tuple[float, ...]
    rebalance_frequencies: tuple[str, ...]
    quarterly_signal_months: tuple[int, ...]
    weighting: str
    selection_tie_policy: str
    execution: str
    execution_availability: str
    normal_trading_status: str
    commission_rate: float
    stamp_duty_rates: tuple[tuple[pd.Timestamp, float], ...]
    slippage_scenarios_bp: tuple[int, ...]
    annualization_days: int
    risk_free_rate: float


@dataclass(frozen=True)
class RobustnessOutputs:
    metrics: pd.DataFrame
    daily: pd.DataFrame
    costs: pd.DataFrame
    portfolio_summary: pd.DataFrame
    baseline_review: pd.DataFrame
    sensitivity: pd.DataFrame


def load_robustness_config(config_path: Path) -> RobustnessConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    robust = settings["robustness"]
    portfolio = settings["portfolio"]
    costs = settings["costs"]
    research = settings["research"]
    config = RobustnessConfig(
        processed_dir=config_path.parent.parent / settings["data"]["processed_dir"],
        selection_quantiles=tuple(float(value) for value in robust["selection_quantiles"]),
        rebalance_frequencies=tuple(str(value) for value in robust["rebalance_frequencies"]),
        quarterly_signal_months=tuple(int(value) for value in robust["quarterly_signal_months"]),
        weighting=str(portfolio["weighting"]),
        selection_tie_policy=str(portfolio["selection_tie_policy"]),
        execution=str(portfolio["execution"]),
        execution_availability=str(portfolio["execution_availability"]),
        normal_trading_status=str(settings["universe"]["normal_trading_status"]),
        commission_rate=float(costs["commission_rate"]),
        stamp_duty_rates=tuple(sorted(
            (pd.Timestamp(item["start_date"]), float(item["sell_rate"]))
            for item in costs["stamp_duty_rates"]
        )),
        slippage_scenarios_bp=tuple(int(value) for value in costs["slippage_scenarios_bp"]),
        annualization_days=int(research["annualization_days"]),
        risk_free_rate=float(research["risk_free_rate"]),
    )
    validate_config(config)
    return config


def validate_config(config: RobustnessConfig) -> None:
    if config.selection_quantiles != (0.1, 0.2, 0.3):
        raise ValueError("The grid requires Top 10%, Top 20%, and Top 30%.")
    if config.rebalance_frequencies != ("monthly", "quarterly"):
        raise ValueError("The grid requires monthly and quarterly rebalancing.")
    if config.quarterly_signal_months != (3, 6, 9, 12):
        raise ValueError("Quarterly signals must use calendar quarter ends.")
    if config.slippage_scenarios_bp != (0, 10, 20):
        raise ValueError("The grid requires 0/10/20 bp slippage scenarios.")
    if config.weighting != "equal" or config.selection_tie_policy != "preserve_boundary_ties":
        raise ValueError("The grid must preserve the portfolio weighting and tie rules.")
    if config.execution != "next_trading_day_open":
        raise ValueError("The grid must preserve next-trading-day open execution.")


def filter_signal_frequency(factor_details: pd.DataFrame, frequency: str, quarter_months: tuple[int, ...]) -> pd.DataFrame:
    """Select existing monthly signals for one confirmed rebalance frequency."""
    if frequency == "monthly":
        return factor_details.copy()
    if frequency != "quarterly":
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    dates = pd.to_datetime(factor_details["signal_date"])
    return factor_details.loc[dates.dt.month.isin(quarter_months)].copy()


def _portfolio_config(config: RobustnessConfig, quantile: float) -> PortfolioConfig:
    return PortfolioConfig(
        processed_dir=config.processed_dir,
        selection_quantile=quantile,
        weighting=config.weighting,
        selection_tie_policy=config.selection_tie_policy,
        execution=config.execution,
        execution_availability=config.execution_availability,
        normal_trading_status=config.normal_trading_status,
    )


def _portfolio_audit(
    details: pd.DataFrame,
    targets: pd.DataFrame,
    trading_dates: pd.Series,
    quantile: float,
    frequency: str,
) -> pd.DataFrame:
    signal_dates = pd.Series(pd.to_datetime(details["signal_date"]).unique()).sort_values()
    execution = build_execution_dates(signal_dates, trading_dates)
    frames: list[pd.DataFrame] = []
    for factor in FACTORS:
        base = details[["signal_date", f"{factor}_is_rankable"]].copy()
        base["factor"] = factor
        base["is_rankable"] = base[f"{factor}_is_rankable"].fillna(False).astype(bool)
        counts = base.groupby(["factor", "signal_date"], as_index=False).agg(
            rankable_stock_count=("is_rankable", "sum")
        )
        frames.append(counts)
    audit = pd.concat(frames, ignore_index=True)
    selected = targets.groupby(["factor", "signal_date"], as_index=False).agg(
        selected_stock_count=("stock_code", "size"),
        target_weight_sum=("target_weight", "sum"),
    )
    audit = audit.merge(selected, on=["factor", "signal_date"], how="left")
    audit = audit.merge(execution, on="signal_date", how="left", validate="many_to_one")
    audit["selected_stock_count"] = audit["selected_stock_count"].fillna(0).astype(int)
    audit["target_weight_sum"] = audit["target_weight_sum"].fillna(0.0)
    audit["actual_selection_ratio"] = audit["selected_stock_count"].div(
        audit["rankable_stock_count"].replace(0, np.nan)
    )
    audit["selection_quantile"] = quantile
    audit["rebalance_frequency"] = frequency
    audit["sample_status"] = np.where(
        audit["rankable_stock_count"].eq(0), "insufficient_rankable_sample", "complete"
    )
    return audit[[
        "selection_quantile", "rebalance_frequency", "factor", "signal_date", "execution_date",
        "rankable_stock_count", "selected_stock_count", "actual_selection_ratio",
        "target_weight_sum", "sample_status",
    ]].sort_values(["selection_quantile", "rebalance_frequency", "factor", "signal_date"])


def _metrics_table(
    performance: PerformanceOutputs,
    quantile: float,
    frequency: str,
    scenarios: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in FACTORS:
        gross = performance.metrics.loc[
            performance.metrics["series_id"].eq(f"{factor}_gross")
        ].iloc[0]
        for scenario in scenarios:
            net = performance.metrics.loc[
                performance.metrics["series_id"].eq(f"{factor}_net_{scenario}bp")
            ].iloc[0]
            row: dict[str, object] = {
                "selection_quantile": quantile,
                "rebalance_frequency": frequency,
                "factor": factor,
                "slippage_bp": scenario,
                "start_date": gross["start_date"],
                "end_date": gross["end_date"],
                "observation_count": int(gross["observation_count"]),
            }
            for metric in (
                "cumulative_return", "annualized_return", "annualized_volatility",
                "max_drawdown", "sharpe_ratio",
            ):
                row[f"gross_{metric}"] = gross[metric]
                row[f"net_{metric}"] = net[metric]
            row["cumulative_turnover"] = net["cumulative_turnover"]
            row["annualized_turnover"] = net["annualized_turnover"]
            row["cumulative_transaction_cost"] = net["cumulative_transaction_cost"]
            rows.append(row)
    return pd.DataFrame(rows)


def _compare_frames(label: str, actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, object]:
    if list(actual.columns) != list(expected.columns):
        raise ValueError(f"Baseline reproduction failed for {label}: columns differ.")
    if len(actual) != len(expected):
        raise ValueError(f"Baseline reproduction failed for {label}: row counts differ.")
    left = actual.reset_index(drop=True).convert_dtypes()
    right = expected.reset_index(drop=True).convert_dtypes()
    for column in left.columns:
        if column.endswith("_date"):
            left[column] = pd.to_datetime(left[column], errors="coerce")
            right[column] = pd.to_datetime(right[column], errors="coerce")
    try:
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12
        )
    except AssertionError as exc:
        raise ValueError(f"Baseline reproduction failed for {label}: {exc}") from exc
    numeric = [column for column in left.columns if pd.api.types.is_numeric_dtype(left[column])]
    max_difference = 0.0
    for column in numeric:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            max_difference = max(max_difference, float(np.max(np.abs(a[finite] - b[finite]))))
    return {"artifact": label, "row_count": len(left), "max_absolute_difference": max_difference, "status": "passed"}


def _baseline_review(
    p: Path,
    portfolio: PortfolioOutputs,
    backtest: BacktestOutputs,
    costs: CostOutputs,
    performance: PerformanceOutputs,
) -> pd.DataFrame:
    comparisons = [
        ("portfolio_targets", portfolio.targets, pd.read_parquet(p / "portfolio_targets.parquet")),
        ("portfolio_actions", portfolio.actions, pd.read_parquet(p / "portfolio_rebalance_actions.parquet")),
        ("portfolio_periods", portfolio.period_summary, pd.read_csv(p / "portfolio_period_summary.csv")),
        ("backtest_daily", backtest.daily, pd.read_parquet(p / "backtest_daily_returns.parquet")),
        ("backtest_monthly", backtest.monthly, pd.read_csv(p / "backtest_monthly_returns.csv")),
        ("backtest_holdings", backtest.holdings, pd.read_parquet(p / "backtest_holdings.parquet")),
        ("backtest_trades", backtest.rebalance_records, pd.read_parquet(p / "backtest_rebalance_records.parquet")),
        ("costs", costs.costs, pd.read_parquet(p / "backtest_costs.parquet")),
        ("net_daily", costs.daily, pd.read_parquet(p / "backtest_net_daily_returns.parquet")),
        ("net_monthly", costs.monthly, pd.read_csv(p / "backtest_net_monthly_returns.csv")),
        ("metrics", performance.metrics, pd.read_csv(p / "performance_metrics.csv", parse_dates=["start_date", "end_date"])),
        ("monthly", performance.monthly, pd.read_csv(p / "performance_monthly_returns.csv")),
        ("drawdowns", performance.drawdowns, pd.read_parquet(p / "performance_drawdowns.parquet")),
    ]
    return pd.DataFrame([_compare_frames(label, actual, expected) for label, actual, expected in comparisons])


def _sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = (
        "net_cumulative_return", "net_annualized_volatility", "net_max_drawdown",
        "net_sharpe_ratio", "cumulative_turnover", "cumulative_transaction_cost",
    )
    for factor, frame in metrics.groupby("factor", sort=True):
        row: dict[str, object] = {
            "factor": factor,
            "net_return_direction_reversal": bool(
                frame["net_cumulative_return"].min() < 0 < frame["net_cumulative_return"].max()
            ),
        }
        for column in columns:
            row[f"{column}_min"] = frame[column].min()
            row[f"{column}_max"] = frame[column].max()
            row[f"{column}_range"] = frame[column].max() - frame[column].min()
        rows.append(row)
    return pd.DataFrame(rows)


def validate_outputs(outputs: RobustnessOutputs, config: RobustnessConfig) -> None:
    expected = len(config.selection_quantiles) * len(config.rebalance_frequencies) * len(FACTORS) * len(config.slippage_scenarios_bp)
    keys = ["selection_quantile", "rebalance_frequency", "factor", "slippage_bp"]
    if len(outputs.metrics) != expected or outputs.metrics.duplicated(keys).any():
        raise ValueError("The metrics do not contain the complete 54-scenario grid.")
    for column in ("start_date", "end_date", "observation_count"):
        if outputs.metrics[column].nunique() != 1:
            raise ValueError(f"Scenarios do not share one {column}.")
    if outputs.metrics["observation_count"].iloc[0] != 2735:
        raise ValueError("All scenarios must share one observation count.")
    if not outputs.baseline_review["status"].eq("passed").all():
        raise ValueError("Top 20% monthly did not reproduce the recorded baseline.")
    zero_bp = outputs.metrics.loc[outputs.metrics["slippage_bp"].eq(0)]
    if zero_bp["cumulative_transaction_cost"].le(0).any():
        raise ValueError("Zero-bp scenarios must still include commission and stamp duty.")


def render_report(outputs: RobustnessOutputs, config: RobustnessConfig) -> str:
    metrics = outputs.metrics.sort_values(["factor", "selection_quantile", "rebalance_frequency", "slippage_bp"])
    insufficient = int(outputs.portfolio_summary["sample_status"].ne("complete").sum())
    lines = [
        "# 稳健性测试", "", "## 参数与统一口径", "",
        "- 参数网格：Top 10% / 20% / 30% × Monthly / Quarterly × 0 / 10 / 20 bp，共 18 组参数；PE、Momentum、ROE 分别运行，共 54 个策略成本情景。",
        "- Quarterly 使用 3/6/9/12 月现有月末信号，下一交易日开盘执行，持有至下一季度执行；共有 46 个季度信号，2025-12-31 仅为无后续执行日的末端候选。",
        "- 选股边界同分全部保留并等权，因此实际选股比例可能略高于名义比例。无法买入留现金，待卖出恢复可交易后卖出。",
        f"- 统一评价区间：{metrics.start_date.iloc[0]:%Y-%m-%d} 至 {metrics.end_date.iloc[0]:%Y-%m-%d}，共 {int(metrics.observation_count.iloc[0]):,} 个日收益；年化交易日数 {config.annualization_days}，年化无风险利率 {config.risk_free_rate:.2%}。",
        "- 0 bp 仅表示无滑点，仍包含买卖双边 3 bp 佣金和卖出印花税；最低佣金不计。",
        "", "## 指标公式", "",
        "- 累计收益 = `∏(1 + 日收益) - 1`；年化收益 = `(1 + 累计收益)^(252 / 日收益数) - 1`。",
        "- 年化波动率 = 日收益样本标准差（`ddof=1`）× `√252`；Sharpe = 日超额收益均值 ÷ 日收益样本标准差 × `√252`。",
        "- 最大回撤 = `净值 ÷ 含初始净值 1 的历史最高净值 - 1` 的最小值。",
        "- 累计换手率为日度换手率之和；年化换手率 = 累计换手率 × `252 / 2735`；累计交易成本为归一化组合的成本 NAV 点数之和。",
        "", "## 全部参数组合", "",
        "| 因子 | Top | 频率 | 滑点 | 毛累计 | 净累计 | 净年化 | 净波动 | 净最大回撤 | 净 Sharpe | 累计换手 | 年化换手 | 累计成本 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.selection_quantile:.0%} | {row.rebalance_frequency} | {int(row.slippage_bp)} bp | "
            f"{row.gross_cumulative_return:.6f} | {row.net_cumulative_return:.6f} | {row.net_annualized_return:.6f} | "
            f"{row.net_annualized_volatility:.6f} | {row.net_max_drawdown:.6f} | {row.net_sharpe_ratio:.6f} | "
            f"{row.cumulative_turnover:.6f} | {row.annualized_turnover:.6f} | {row.cumulative_transaction_cost:.6f} |"
        )
    lines.extend(["", "## Top 20% + Monthly 基准复现", "", "| 产物 | 行数 | 最大绝对差异 | 结果 |", "| --- | ---: | ---: | --- |"]) 
    for row in outputs.baseline_review.itertuples(index=False):
        lines.append(f"| {row.artifact} | {row.row_count:,} | {row.max_absolute_difference:.3e} | {row.status} |")
    lines.extend(["", "## 敏感性事实", "", "不设置主观阈值，不选择最佳参数；下表只报告方向反转和全参数范围。", "", "| 因子 | 净累计收益反转 | 净累计收益范围 | 净波动范围 | 最大回撤范围 | Sharpe 范围 | 换手率范围 | 成本范围 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]) 
    for row in outputs.sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {'是' if row.net_return_direction_reversal else '否'} | "
            f"[{row.net_cumulative_return_min:.6f}, {row.net_cumulative_return_max:.6f}] | "
            f"[{row.net_annualized_volatility_min:.6f}, {row.net_annualized_volatility_max:.6f}] | "
            f"[{row.net_max_drawdown_min:.6f}, {row.net_max_drawdown_max:.6f}] | "
            f"[{row.net_sharpe_ratio_min:.6f}, {row.net_sharpe_ratio_max:.6f}] | "
            f"[{row.cumulative_turnover_min:.6f}, {row.cumulative_turnover_max:.6f}] | "
            f"[{row.cumulative_transaction_cost_min:.6f}, {row.cumulative_transaction_cost_max:.6f}] |"
        )
    lines.extend(["", f"- 无可排名股票的因子/信号期记录数：{insufficient}；这些期间保留为样本不足记录，不填造持仓。", ""])
    lines.extend([
        "", "## 输出文件", "",
        "- `robustness_metrics.csv`：54 个策略成本情景的毛、净绩效指标。",
        "- `robustness_daily_returns.parquet`：日度毛净收益、换手率和成本。",
        "- `robustness_costs.parquet`：实际成交成本明细。",
        "- `robustness_portfolio_summary.csv`：逐期选股比例、信号日和执行日。",
        "", "## 研究限制", "",
        "- 策略执行日使用开盘至收盘收益，H00300 使用完整日收益，盘中起点不完全一致。",
        "- 涨跌停无法可靠识别；正常交易状态和有效开盘价不构成一定能够成交的保证。",
        "- 缺少真实初始资金和订单金额，最低佣金未纳入。",
        "- 双边 3 bp 佣金和 0/10/20 bp 滑点是研究假设，不代表特定券商费率或真实冲击。",
        "- PE 使用按信号日前已披露财报重建的 TTM 净利润；供应商 `PE1TTM` 实测含前视，本网格沿用重建口径。",
        "- 本步骤只做参数敏感性事实总结，不依据全样本结果修改策略或选择最佳参数。", "",
    ])
    return "\n".join(lines)


def build_robustness_outputs(
    factor_details: pd.DataFrame,
    daily_trading: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: RobustnessConfig,
    pool: pd.DataFrame | None = None,
    market_cap: pd.DataFrame | None = None,
) -> RobustnessOutputs:
    metric_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    cost_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    baseline = pd.DataFrame()
    performance_config = PerformanceConfig(config.processed_dir, config.annualization_days, config.risk_free_rate)
    cost_config = CostConfig(config.processed_dir, config.commission_rate, config.stamp_duty_rates, config.slippage_scenarios_bp)
    backtest_config = BacktestConfig(config.processed_dir, config.normal_trading_status)

    quantile_order = (0.2,) + tuple(value for value in config.selection_quantiles if value != 0.2)
    for quantile in quantile_order:
        for frequency in config.rebalance_frequencies:
            details = filter_signal_frequency(factor_details, frequency, config.quarterly_signal_months)
            portfolio_config = _portfolio_config(config, quantile)
            portfolio = build_portfolio_outputs(details, daily_trading, portfolio_config)
            validate_portfolio(portfolio, portfolio_config)
            backtest = build_backtest_outputs(portfolio.targets, portfolio.actions, daily_trading, benchmark, backtest_config, pool, market_cap)
            validate_backtest(backtest)
            cost_output = build_cost_outputs(backtest.daily, backtest.rebalance_records, cost_config)
            validate_costs(cost_output)
            performance = build_performance_outputs(backtest.daily, cost_output.daily, cost_output.costs, performance_config)
            validate_performance(performance)

            metric_frames.append(_metrics_table(performance, quantile, frequency, config.slippage_scenarios_bp))
            daily_item = cost_output.daily.copy()
            daily_item.insert(0, "rebalance_frequency", frequency)
            daily_item.insert(0, "selection_quantile", quantile)
            daily_frames.append(daily_item)
            cost_item = cost_output.costs.copy()
            cost_item.insert(0, "rebalance_frequency", frequency)
            cost_item.insert(0, "selection_quantile", quantile)
            cost_frames.append(cost_item)
            summary_frames.append(_portfolio_audit(details, portfolio.targets, daily_trading["trade_date"], quantile, frequency))
            if quantile == 0.2 and frequency == "monthly":
                baseline = _baseline_review(config.processed_dir, portfolio, backtest, cost_output, performance)

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
        ["selection_quantile", "rebalance_frequency", "factor", "slippage_bp"]
    ).reset_index(drop=True)
    daily = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["selection_quantile", "rebalance_frequency", "factor", "slippage_bp", "trade_date"]
    ).reset_index(drop=True)
    costs = pd.concat(cost_frames, ignore_index=True).sort_values(
        ["selection_quantile", "rebalance_frequency", "factor", "slippage_bp", "trade_date"]
    ).reset_index(drop=True)
    summaries = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["selection_quantile", "rebalance_frequency", "factor", "signal_date"]
    ).reset_index(drop=True)
    outputs = RobustnessOutputs(metrics, daily, costs, summaries, baseline, _sensitivity(metrics))
    validate_outputs(outputs, config)
    return outputs


def run(config: RobustnessConfig) -> RobustnessOutputs:
    p = config.processed_dir
    factor_details = pd.read_parquet(p / "factor_details.parquet")
    daily = pd.read_parquet(p / "daily_trading.parquet", columns=[
        "stock_code", "trade_date", "open_price", "close_price", "total_return",
        "PreClosePrice", "trading_status", "quality_flags",
    ])
    benchmark = pd.read_parquet(p / "benchmark.parquet", columns=["trade_date", "total_return", "quality_flags"])
    pool = pd.read_parquet(p / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    market_cap = pd.read_parquet(p / "market_cap.parquet", columns=["signal_date", "stock_code", "market_cap", "has_market_cap"])
    outputs = build_robustness_outputs(factor_details, daily, benchmark, config, pool, market_cap)
    outputs.metrics.to_csv(p / "robustness_metrics.csv", index=False, encoding="utf-8-sig")
    outputs.daily.to_parquet(p / "robustness_daily_returns.parquet", index=False)
    outputs.costs.to_parquet(p / "robustness_costs.parquet", index=False)
    outputs.portfolio_summary.to_csv(p / "robustness_portfolio_summary.csv", index=False, encoding="utf-8-sig")
    report_dir = p.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "robustness.md").write_text(render_report(outputs, config), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the robustness tests.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_robustness_config(args.config))
    print(f"Built {len(outputs.metrics)} robustness scenarios across {outputs.daily['trade_date'].nunique():,} trading days.")


if __name__ == "__main__":
    main()
