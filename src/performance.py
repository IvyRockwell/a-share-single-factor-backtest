"""Risk-return evaluation over the gross and net daily series."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FACTORS = ("pe", "momentum", "roe")
SCENARIOS = (0, 10, 20)
BENCHMARKS = (
    ("h00300_benchmark", "H00300", "benchmark_return"),
    ("ew_all_a_benchmark", "EW_ALL_A", "ew_benchmark_return"),
    ("cw_all_a_benchmark", "CW_ALL_A", "cw_benchmark_return"),
)


@dataclass(frozen=True)
class PerformanceConfig:
    processed_dir: Path
    annualization_days: int
    risk_free_rate: float


@dataclass(frozen=True)
class PerformanceOutputs:
    metrics: pd.DataFrame
    monthly: pd.DataFrame
    drawdowns: pd.DataFrame


def load_performance_config(config_path: Path) -> PerformanceConfig:
    """Load the annualization assumptions."""
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    research = settings["research"]
    annualization_days = research.get("annualization_days")
    risk_free_rate = research.get("risk_free_rate")
    if not isinstance(annualization_days, int) or annualization_days <= 0:
        raise ValueError("annualization_days must be a positive integer.")
    if risk_free_rate is None or float(risk_free_rate) <= -1:
        raise ValueError("risk_free_rate must be set and greater than -100%.")
    return PerformanceConfig(
        processed_dir=config_path.parent.parent / settings["data"]["processed_dir"],
        annualization_days=annualization_days,
        risk_free_rate=float(risk_free_rate),
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    return np.allclose(
        left.to_numpy(dtype=float),
        right.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    )


def validate_source_inputs(
    gross_df: pd.DataFrame,
    net_df: pd.DataFrame,
    costs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate common dates and the gross/net lineage without rerunning either step."""
    gross_df = gross_df.copy()
    net_df = net_df.copy()
    costs = costs.copy()
    _require_columns(
        gross_df,
        {"factor", "trade_date", "gross_return", "nav", "benchmark_return", "benchmark_nav",
         "ew_benchmark_return", "cw_benchmark_return"},
        "backtest daily results",
    )
    _require_columns(
        net_df,
        {
            "factor", "trade_date", "slippage_bp", "gross_return", "nav", "benchmark_return",
            "benchmark_nav", "net_return", "net_nav", "turnover", "total_transaction_cost",
        },
        "cost daily results",
    )
    _require_columns(
        costs,
        {"factor", "trade_date", "slippage_bp", "turnover", "total_transaction_cost"},
        "cost details",
    )
    for frame in (gross_df, net_df, costs):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])

    if set(gross_df["factor"].unique()) != set(FACTORS):
        raise ValueError("the gross series must contain exactly the PE, Momentum, and ROE strategies.")
    if set(net_df["factor"].unique()) != set(FACTORS) or set(net_df["slippage_bp"].unique()) != set(SCENARIOS):
        raise ValueError("the net series must contain all three strategies and 0/10/20 bp scenarios.")
    if set(costs["factor"].unique()) != set(FACTORS) or set(costs["slippage_bp"].unique()) != set(SCENARIOS):
        raise ValueError("cost details must contain all strategies and scenarios.")
    if gross_df.duplicated(["factor", "trade_date"]).any():
        raise ValueError("the gross series contains duplicate factor/date rows.")
    if net_df.duplicated(["factor", "trade_date", "slippage_bp"]).any():
        raise ValueError("the net series contains duplicate factor/date/scenario rows.")
    if costs.duplicated(["factor", "trade_date", "slippage_bp"]).any():
        raise ValueError("cost details contain duplicate factor/date/scenario rows.")

    gross_df = gross_df.sort_values(["factor", "trade_date"]).reset_index(drop=True)
    net_df = net_df.sort_values(["factor", "slippage_bp", "trade_date"]).reset_index(drop=True)
    costs = costs.sort_values(["factor", "slippage_bp", "trade_date"]).reset_index(drop=True)
    common_dates = pd.Index(gross_df.loc[gross_df["factor"].eq(FACTORS[0]), "trade_date"])
    if common_dates.empty:
        raise ValueError("The gross-return series is empty.")
    for factor in FACTORS:
        dates = pd.Index(gross_df.loc[gross_df["factor"].eq(factor), "trade_date"])
        if not dates.equals(common_dates):
            raise ValueError("strategy date ranges are not identical.")
    for factor in FACTORS:
        for scenario in SCENARIOS:
            dates = pd.Index(
                net_df.loc[
                    net_df["factor"].eq(factor) & net_df["slippage_bp"].eq(scenario),
                    "trade_date",
                ]
            )
            if not dates.equals(common_dates):
                raise ValueError("scenario dates do not match the gross series.")

    benchmark = gross_df.pivot(index="trade_date", columns="factor", values="benchmark_return")
    if benchmark.isna().any().any() or benchmark.nunique(axis=1).gt(1).any():
        raise ValueError("The benchmark must cover every common date with one consistent return.")
    merged = net_df.merge(
        gross_df[["factor", "trade_date", "gross_return", "nav", "benchmark_return", "benchmark_nav"]],
        on=["factor", "trade_date"],
        suffixes=("_net", "_gross"),
        validate="many_to_one",
    )
    for column in ("gross_return", "nav", "benchmark_return", "benchmark_nav"):
        if not _same_values(merged[f"{column}_net"], merged[f"{column}_gross"]):
            raise ValueError(f"net {column} does not reproduce the gross series.")

    for column in ("gross_return", "benchmark_return", "net_return"):
        if net_df[column].isna().any() or net_df[column].le(-1).any():
            raise ValueError(f"net {column} contains missing or invalid returns.")
    if gross_df[["gross_return", "benchmark_return"]].isna().any().any():
        raise ValueError("the gross series contains missing strategy or benchmark returns.")
    if net_df.groupby(["factor", "trade_date"])["turnover"].nunique().gt(1).any():
        raise ValueError("Turnover must be identical across slippage scenarios.")
    if not set(costs["trade_date"]).issubset(set(common_dates)):
        raise ValueError("cost details contain dates outside the common evaluation range.")

    daily_totals = net_df.groupby(["factor", "slippage_bp"], as_index=False).agg(
        turnover=("turnover", "sum"),
        transaction_cost=("total_transaction_cost", "sum"),
    )
    cost_totals = costs.groupby(["factor", "slippage_bp"], as_index=False).agg(
        turnover=("turnover", "sum"),
        transaction_cost=("total_transaction_cost", "sum"),
    )
    totals = daily_totals.merge(cost_totals, on=["factor", "slippage_bp"], suffixes=("_daily", "_detail"), validate="one_to_one")
    for column in ("turnover", "transaction_cost"):
        if not _same_values(totals[f"{column}_daily"], totals[f"{column}_detail"]):
            raise ValueError(f"daily and detailed {column} totals do not agree.")
    return gross_df, net_df, costs


def _series_frame(
    source: pd.DataFrame,
    series_id: str,
    series_type: str,
    factor: str,
    slippage_bp: float,
    return_column: str,
) -> pd.DataFrame:
    result = source[["trade_date", return_column]].rename(columns={return_column: "daily_return"}).copy()
    result.insert(0, "series_id", series_id)
    result.insert(1, "series_type", series_type)
    result.insert(2, "factor", factor)
    result.insert(3, "slippage_bp", slippage_bp)
    result["nav"] = (1 + result["daily_return"]).cumprod()
    result["running_peak"] = result["nav"].cummax().clip(lower=1.0)
    result["drawdown"] = result["nav"] / result["running_peak"] - 1
    return result


def _metric_row(
    series: pd.DataFrame,
    config: PerformanceConfig,
    cumulative_turnover: float = np.nan,
    cumulative_transaction_cost: float = np.nan,
) -> dict[str, object]:
    returns = series["daily_return"].astype(float)
    observations = len(returns)
    cumulative_return = float(series["nav"].iloc[-1] - 1)
    annualized_return = float((1 + cumulative_return) ** (config.annualization_days / observations) - 1)
    daily_volatility = float(returns.std(ddof=1))
    annualized_volatility = daily_volatility * np.sqrt(config.annualization_days)
    daily_risk_free = (1 + config.risk_free_rate) ** (1 / config.annualization_days) - 1
    sharpe_ratio = np.nan if daily_volatility == 0 else float(
        (returns - daily_risk_free).mean() / daily_volatility * np.sqrt(config.annualization_days)
    )
    first = series.iloc[0]
    return {
        "series_id": first["series_id"],
        "series_type": first["series_type"],
        "factor": first["factor"],
        "slippage_bp": first["slippage_bp"],
        "start_date": series["trade_date"].min(),
        "end_date": series["trade_date"].max(),
        "observation_count": observations,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": float(series["drawdown"].min()),
        "sharpe_ratio": sharpe_ratio,
        "cumulative_turnover": cumulative_turnover,
        "annualized_turnover": cumulative_turnover * config.annualization_days / observations,
        "cumulative_transaction_cost": cumulative_transaction_cost,
    }


def build_performance_outputs(
    gross_df: pd.DataFrame,
    net_df: pd.DataFrame,
    costs: pd.DataFrame,
    config: PerformanceConfig,
) -> PerformanceOutputs:
    gross_df, net_df, costs = validate_source_inputs(gross_df, net_df, costs)
    series_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    cost_totals = costs.groupby(["factor", "slippage_bp"], as_index=False).agg(
        cumulative_turnover=("turnover", "sum"),
        cumulative_transaction_cost=("total_transaction_cost", "sum"),
    )
    totals_lookup = cost_totals.set_index(["factor", "slippage_bp"])

    for factor in FACTORS:
        gross_source = gross_df.loc[gross_df["factor"].eq(factor)].sort_values("trade_date")
        turnover = float(totals_lookup.loc[(factor, SCENARIOS[0]), "cumulative_turnover"])
        gross = _series_frame(gross_source, f"{factor}_gross", "gross", factor, np.nan, "gross_return")
        series_frames.append(gross)
        metric_rows.append(_metric_row(gross, config, cumulative_turnover=turnover))
        for scenario in SCENARIOS:
            net_source = net_df.loc[
                net_df["factor"].eq(factor) & net_df["slippage_bp"].eq(scenario)
            ].sort_values("trade_date")
            net = _series_frame(net_source, f"{factor}_net_{scenario}bp", "net", factor, float(scenario), "net_return")
            series_frames.append(net)
            total = totals_lookup.loc[(factor, scenario)]
            metric_rows.append(_metric_row(
                net,
                config,
                cumulative_turnover=float(total["cumulative_turnover"]),
                cumulative_transaction_cost=float(total["cumulative_transaction_cost"]),
            ))

    for series_id, label, column in BENCHMARKS:
        source = gross_df.loc[gross_df["factor"].eq(FACTORS[0]), ["trade_date", column]].sort_values("trade_date")
        frame = _series_frame(source, series_id, "benchmark", label, np.nan, column)
        series_frames.append(frame)
        metric_rows.append(_metric_row(frame, config))

    drawdowns = pd.concat(series_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    drawdowns["month"] = drawdowns["trade_date"].dt.to_period("M").astype(str)
    monthly = drawdowns.groupby(
        ["series_id", "series_type", "factor", "slippage_bp", "month"],
        dropna=False,
        as_index=False,
    ).agg(
        monthly_return=("daily_return", lambda values: (1 + values).prod() - 1),
        end_nav=("nav", "last"),
    )
    drawdowns = drawdowns.drop(columns="month")
    return PerformanceOutputs(metrics, monthly, drawdowns)


def validate_outputs(outputs: PerformanceOutputs) -> None:
    expected = len(FACTORS) * (1 + len(SCENARIOS)) + len(BENCHMARKS)
    if len(outputs.metrics) != expected or outputs.metrics["series_id"].nunique() != expected:
        raise ValueError(f"Performance evaluation must produce {expected} distinct series.")
    if outputs.drawdowns.duplicated(["series_id", "trade_date"]).any():
        raise ValueError("Drawdown output contains duplicate series/date rows.")
    if outputs.drawdowns.groupby("series_id")["trade_date"].size().nunique() != 1:
        raise ValueError("Performance series do not share one observation count.")
    benchmark = outputs.metrics.loc[outputs.metrics["series_type"].eq("benchmark")]
    if len(benchmark) != len(BENCHMARKS):
        raise ValueError("Every benchmark series must be present exactly once.")
    if benchmark[["cumulative_turnover", "annualized_turnover", "cumulative_transaction_cost"]].notna().any().any():
        raise ValueError("Benchmark turnover and transaction costs must remain not applicable.")
    gross = outputs.metrics.loc[outputs.metrics["series_type"].eq("gross")]
    if gross["cumulative_transaction_cost"].notna().any():
        raise ValueError("Gross strategy rows must not claim an applied transaction cost.")


def _format_value(value: float, digits: int = 6) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def render_report(outputs: PerformanceOutputs, config: PerformanceConfig) -> str:
    lines = [
        "# 风险收益评价",
        "",
        "## 评价口径",
        "",
        f"- 评价区间：{outputs.metrics['start_date'].min():%Y-%m-%d} 至 {outputs.metrics['end_date'].max():%Y-%m-%d}，共 {int(outputs.metrics['observation_count'].max()):,} 个日收益观测；所有策略和基准使用相同交易日。Momentum 在 2015-04-01 首次持仓前保持现金，期间日收益为 0。",
        f"- 年化交易日数：{config.annualization_days}；年化无风险利率：{config.risk_free_rate:.2%}。",
        "- 累计收益 = `∏(1 + 日收益) - 1`；年化收益 = `(1 + 累计收益)^(252 / 日收益数) - 1`。",
        "- 年化波动率 = 日收益样本标准差（`ddof=1`）× `√252`。",
        "- Sharpe Ratio = 日超额收益均值 ÷ 日收益样本标准差 × `√252`；日无风险收益由年化无风险利率复利换算。若日波动率为 0，则不虚构 Sharpe，输出为空。",
        "- 最大回撤 = `净值 ÷ 历史最高净值 - 1` 的最小值；历史最高净值包含回测前初始净值 1。",
        "- 累计换手率为成本步骤的日度换手率之和；年化换手率 = 累计换手率 × `252 / 2735`。累计交易成本为归一化组合上的 NAV 点数之和，并不等于期末毛净值减期末净净值。",
        "",
        "## 毛收益、净收益与基准",
        "",
        "- 毛收益来自回测，未扣除交易成本；净收益来自成本步骤。0 bp 表示滑点为 0，仍包含买卖双边 3 bp 佣金及卖出印花税。",
        "- H00300 使用 CSMAR `Idxtrd08` 收益字段，数据清洗时已由百分数除以 100 转为小数日收益；基准不虚构换手率或交易成本。",
        "- 等权/市值加权是**日频再平衡诊断**；同股票池、月频、同开盘执行的基准与回归 alpha 见 `reports/generated/attribution.md`，不要用本表的日频差值做因子归因。",
        "",
        "| 序列 | 类型 | 滑点 | 累计收益 | 年化收益 | 年化波动率 | 最大回撤 | Sharpe | 累计换手率 | 年化换手率 | 累计成本 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = {"gross": 0, "net": 1, "benchmark": 2}
    metrics = outputs.metrics.assign(_order=outputs.metrics["series_type"].map(order)).sort_values(["_order", "factor", "slippage_bp"], na_position="first")
    for row in metrics.itertuples(index=False):
        slip = "—" if pd.isna(row.slippage_bp) else f"{int(row.slippage_bp)} bp"
        label = str(row.factor)
        lines.append(
            f"| {label} | {row.series_type} | {slip} | {_format_value(row.cumulative_return)} | {_format_value(row.annualized_return)} | {_format_value(row.annualized_volatility)} | {_format_value(row.max_drawdown)} | {_format_value(row.sharpe_ratio)} | {_format_value(row.cumulative_turnover)} | {_format_value(row.annualized_turnover)} | {_format_value(row.cumulative_transaction_cost)} |"
        )
    lines.extend([
        "",
        "## 输出文件",
        "",
        "- `performance_metrics.csv`：13 条毛收益、净收益和基准绩效汇总。",
        "- `performance_monthly_returns.csv`：由同一批日收益直接复利得到的月度收益与月末净值。",
        "- `performance_drawdowns.parquet`：逐序列、逐日的日收益、净值、历史峰值和回撤。",
        "",
        "## 数据与方法限制",
        "",
        "- H00300 缺少可靠开盘价：执行日基准使用完整 `Idxtrd08` 日收益，而策略首次买入日使用开盘至收盘收益，盘中起点不完全一致。",
        "- 涨跌停无法可靠识别；正常交易状态和有效开盘价不构成一定能够成交的保证。",
        "- 缺少真实初始资金、股数和订单金额，最低佣金未纳入。",
        "- 双边 3 bp 佣金和 0/10/20 bp 滑点是研究假设，不代表特定券商费率或真实成交冲击；日频数据也不能估计盘口滑点。",
        "- PE 已改用信号日前已披露财报重建的 TTM 净利润，供应商 `PE1TTM` 实测含前视；本步骤不重新计算因子，只评价既有回测与成本结果。",
        "- 本步骤只评价回测与成本的既有结果，没有重新回测或改变成交、成本、持仓和现金规则。",
        "",
    ])
    return "\n".join(lines)


def run(config: PerformanceConfig) -> PerformanceOutputs:
    p = config.processed_dir
    outputs = build_performance_outputs(
        pd.read_parquet(p / "backtest_daily_returns.parquet"),
        pd.read_parquet(p / "backtest_net_daily_returns.parquet"),
        pd.read_parquet(p / "backtest_costs.parquet"),
        config,
    )
    validate_outputs(outputs)
    outputs.metrics.to_csv(p / "performance_metrics.csv", index=False, encoding="utf-8-sig")
    outputs.monthly.to_csv(p / "performance_monthly_returns.csv", index=False, encoding="utf-8-sig")
    outputs.drawdowns.to_parquet(p / "performance_drawdowns.parquet", index=False)
    report_dir = p.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance.md").write_text(render_report(outputs, config), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the risk-return evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_performance_config(args.config))
    print(f"Built {len(outputs.metrics)} performance rows and {len(outputs.drawdowns):,} daily drawdown rows.")


if __name__ == "__main__":
    main()
