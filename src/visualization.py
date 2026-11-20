"""Generate the research figures from the saved pipeline outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

FACTORS = ("pe", "momentum", "roe")
FACTOR_LABELS = {"pe": "PE", "momentum": "Momentum", "roe": "ROE"}
FACTOR_COLORS = {"pe": "#0072B2", "momentum": "#E69F00", "roe": "#009E73"}
BENCHMARK_COLOR = "#4D4D4D"
SCENARIO_STYLES = {
    0: {"color": "#B8B8B8", "linestyle": "--", "hatch": "//"},
    10: {"color": "#777777", "linestyle": "-.", "hatch": "xx"},
    20: {"color": "#242424", "linestyle": ":", "hatch": ".."},
}
FIGURE_FILES = (
    "01_sample_quality.png",
    "02_factor_pe.png",
    "03_factor_momentum.png",
    "04_factor_roe.png",
    "05_gross_nav_vs_benchmark.png",
    "06_gross_vs_net.png",
    "07_drawdowns.png",
    "08_monthly_returns.png",
    "09a_cumulative_return.png",
    "09b_annualized_return.png",
    "09c_annualized_volatility.png",
    "09d_max_drawdown.png",
    "09e_sharpe_ratio.png",
    "10_turnover_and_cost.png",
    "11_robustness_pe.png",
    "12_robustness_momentum.png",
    "13_robustness_roe.png",
    "14_neutralization_ic.png",
)


@dataclass(frozen=True)
class VisualizationConfig:
    processed_dir: Path
    figure_dir: Path
    index_path: Path


@dataclass(frozen=True)
class FigureInputs:
    quality: pd.DataFrame
    quantiles: pd.DataFrame
    ic: pd.DataFrame
    ic_all: pd.DataFrame
    factor_summary: pd.DataFrame
    gross_daily: pd.DataFrame
    cost_daily: pd.DataFrame
    metrics: pd.DataFrame
    monthly: pd.DataFrame
    drawdowns: pd.DataFrame
    robustness: pd.DataFrame


def load_visualization_config(config_path: Path) -> VisualizationConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    return VisualizationConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        figure_dir=root / "reports" / "figures",
        index_path=root / "reports" / "generated" / "figures.md",
    )


def load_inputs(processed_dir: Path) -> FigureInputs:
    parse_csv_dates = {
        "factor_period_quality.csv": ["signal_date"],
        "factor_quantile_returns.csv": ["signal_date"],
        "factor_ic_series.csv": ["signal_date"],
        "performance_metrics.csv": ["start_date", "end_date"],
    }

    def read_csv(name: str) -> pd.DataFrame:
        return pd.read_csv(processed_dir / name, parse_dates=parse_csv_dates.get(name))

    quantiles = read_csv("factor_quantile_returns.csv")
    ic = read_csv("factor_ic_series.csv")
    if "score_type" in quantiles.columns:
        quantiles = quantiles.loc[quantiles["score_type"].eq("raw")].reset_index(drop=True)
    if "score_type" in ic.columns:
        ic = ic.loc[ic["score_type"].eq("raw")].reset_index(drop=True)

    return FigureInputs(
        quality=read_csv("factor_period_quality.csv"),
        quantiles=quantiles,
        ic=ic,
        ic_all=read_csv("factor_ic_series.csv"),
        factor_summary=read_csv("factor_test_summary.csv"),
        gross_daily=pd.read_parquet(processed_dir / "backtest_daily_returns.parquet"),
        cost_daily=pd.read_parquet(processed_dir / "backtest_net_daily_returns.parquet"),
        metrics=read_csv("performance_metrics.csv"),
        monthly=pd.read_csv(processed_dir / "performance_monthly_returns.csv"),
        drawdowns=pd.read_parquet(processed_dir / "performance_drawdowns.parquet"),
        robustness=read_csv("robustness_metrics.csv"),
    )


def _require_columns(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def prepare_q_spread(quantiles: pd.DataFrame) -> pd.DataFrame:
    """Validate identical repeated spread fields and return one row per period."""
    columns = ["q_high_minus_q_low", "cumulative_q_high_minus_q_low"]
    copies = quantiles.groupby(["factor", "signal_date"])[columns].nunique(dropna=False)
    if copies.gt(1).any().any():
        raise ValueError("Q5-Q1 copies differ across quantile rows.")
    base = (
        quantiles[["factor", "signal_date", *columns]]
        .drop_duplicates(["factor", "signal_date"])
        .sort_values(["factor", "signal_date"])
        .copy()
    )
    direct = base.groupby("factor")["q_high_minus_q_low"].cumsum()
    if not np.allclose(
        base["cumulative_q_high_minus_q_low"].to_numpy(), direct.to_numpy(), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("The cumulative Q5-Q1 field does not match the stated cumulative rule.")
    return base


def validate_inputs(inputs: FigureInputs) -> None:
    _require_columns(inputs.quality, "factor_period_quality", {
        "signal_date", "factor", "pool_count", "rankable_count", "unrankable_count", "unrankable_by_reason"
    })
    _require_columns(inputs.quantiles, "factor_quantile_returns", {
        "factor", "signal_date", "quantile", "cumulative_group_return", "q_high_minus_q_low",
        "cumulative_q_high_minus_q_low", "empty_group"
    })
    _require_columns(inputs.ic, "factor_ic_series", {"factor", "signal_date", "ic_pair_count", "rank_ic"})
    _require_columns(inputs.gross_daily, "backtest_daily_returns", {"factor", "trade_date", "nav", "benchmark_nav"})
    _require_columns(inputs.cost_daily, "backtest_net_daily_returns", {
        "factor", "trade_date", "slippage_bp", "nav", "net_nav", "turnover", "total_transaction_cost"
    })
    _require_columns(inputs.metrics, "performance_metrics", {
        "series_id", "series_type", "factor", "slippage_bp", "start_date", "end_date", "cumulative_return",
        "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio", "annualized_turnover",
        "cumulative_transaction_cost"
    })
    _require_columns(inputs.monthly, "performance_monthly_returns", {
        "series_id", "series_type", "factor", "slippage_bp", "month", "monthly_return"
    })
    _require_columns(inputs.drawdowns, "performance_drawdowns", {
        "series_id", "series_type", "factor", "slippage_bp", "trade_date", "drawdown"
    })
    _require_columns(inputs.robustness, "robustness_metrics", {
        "selection_quantile", "rebalance_frequency", "factor", "slippage_bp", "start_date", "end_date",
        "net_cumulative_return"
    })
    for name, frame in (("quality", inputs.quality), ("quantiles", inputs.quantiles), ("ic", inputs.ic)):
        if not set(frame["factor"].unique()).issubset(set(FACTORS)):
            raise ValueError(f"{name} contains an unexpected factor.")
    if inputs.quality.duplicated(["factor", "signal_date"]).any():
        raise ValueError("Sample-quality rows contain duplicate factor/date keys.")
    if inputs.quantiles.duplicated(["factor", "signal_date", "quantile"]).any():
        raise ValueError("Quantile-return rows contain duplicate keys.")
    if set(inputs.quantiles["quantile"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("The quantile output does not contain exactly Q1-Q5.")
    if inputs.quantiles["empty_group"].any():
        raise ValueError("The quantile output contains empty groups; figures must not silently hide them.")
    prepare_q_spread(inputs.quantiles)
    metric_dates = inputs.metrics[["start_date", "end_date"]].drop_duplicates()
    if len(metric_dates) != 1 or tuple(inputs.metrics["series_type"].unique()) != ("gross", "net", "benchmark"):
        raise ValueError("Performance series do not share one evaluation range or expected types.")
    expected_grid = pd.MultiIndex.from_product(
        ([0.1, 0.2, 0.3], ["monthly", "quarterly"], FACTORS, [0, 10, 20]),
        names=["selection_quantile", "rebalance_frequency", "factor", "slippage_bp"],
    )
    actual_grid = pd.MultiIndex.from_frame(inputs.robustness[list(expected_grid.names)])
    if len(inputs.robustness) != 54 or set(actual_grid) != set(expected_grid):
        raise ValueError("The robustness grid is incomplete.")


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "savefig.facecolor": "white",
        "savefig.dpi": 180,
    })


def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)


def _percent_axis(ax: plt.Axes, xmax: float = 1.0) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax))


def _footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=8, color="#555555")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Research figure"})
    plt.close(fig)


def plot_sample_quality(inputs: FigureInputs, output: Path) -> None:
    quality = inputs.quality.copy()
    quality["unrankable_ratio"] = quality["unrankable_count"] / quality["pool_count"]
    pool = quality.groupby("signal_date", as_index=False)["pool_count"].first()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    axes[0].plot(pool["signal_date"], pool["pool_count"], color=BENCHMARK_COLOR, lw=2.0, label="股票池")
    for factor in FACTORS:
        item = quality.loc[quality["factor"].eq(factor)]
        axes[0].plot(item["signal_date"], item["rankable_count"], color=FACTOR_COLORS[factor], lw=1.6,
                     label=f"{FACTOR_LABELS[factor]} 可排名")
    axes[0].set_title("各月末股票池与三因子可排名样本数")
    axes[0].set_ylabel("股票数（只）")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(ncol=4, loc="upper left", frameon=False)
    for factor in FACTORS:
        item = quality.loc[quality["factor"].eq(factor)]
        axes[1].plot(item["signal_date"], item["unrankable_ratio"], color=FACTOR_COLORS[factor], lw=1.7,
                     label=FACTOR_LABELS[factor])
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_title("各因子不可排名比例（未插值）")
    axes[1].set_ylabel("不可排名比例")
    axes[1].set_xlabel("信号日")
    axes[1].set_ylim(0, 1.05)
    _percent_axis(axes[1])
    axes[1].legend(ncol=3, loc="upper right", frameon=False)
    _date_axis(axes[1])
    axes[1].annotate("Momentum 前 6 期无完整 6 个月历史价格",
                     xy=(pd.Timestamp("2014-12-31"), 1.0), xytext=(pd.Timestamp("2016-01-01"), 0.78),
                     arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=9)
    fig.suptitle("样本覆盖：股票池扩大，但因子可排名性存在差异", fontsize=15)
    _footer(fig, "来源：factor_period_quality.csv｜2014-09-30—2025-12-31｜单位：只、%｜gross/net/成本：不适用")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, output)


def plot_factor(inputs: FigureInputs, factor: str, output: Path) -> None:
    groups = inputs.quantiles.loc[inputs.quantiles["factor"].eq(factor)].copy()
    ic = inputs.ic.loc[inputs.ic["factor"].eq(factor)].sort_values("signal_date").copy()
    spread = prepare_q_spread(inputs.quantiles)
    spread = spread.loc[spread["factor"].eq(factor)]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for quantile in range(1, 6):
        item = groups.loc[groups["quantile"].eq(quantile)].sort_values("signal_date")
        if quantile == 5:
            style = {"color": FACTOR_COLORS[factor], "lw": 2.6, "alpha": 1.0, "zorder": 5}
        elif quantile == 1:
            style = {"color": FACTOR_COLORS[factor], "lw": 1.8, "alpha": 0.62, "linestyle": "--", "zorder": 4}
        else:
            style = {"color": "#9A9A9A", "lw": 1.0, "alpha": 0.45 + quantile * 0.08}
        axes[0].plot(item["signal_date"], item["cumulative_group_return"], label=f"Q{quantile}", **style)
    axes[0].set_title("Q1～Q5 下一持有期收益的累计表现")
    axes[0].set_ylabel("累计收益")
    _percent_axis(axes[0])
    axes[0].legend(ncol=5, loc="upper left", frameon=False)
    axes[1].plot(spread["signal_date"], spread["cumulative_q_high_minus_q_low"],
                 color=FACTOR_COLORS[factor], lw=2.0)
    axes[1].axhline(0, color="#333333", lw=1.0)
    axes[1].set_title("Q5-Q1 每期收益差的累计值（按期累加）")
    axes[1].set_ylabel("累计收益差")
    _percent_axis(axes[1])
    ic["rolling_12"] = ic["rank_ic"].rolling(12, min_periods=12).mean()
    axes[2].plot(ic["signal_date"], ic["rank_ic"], color=FACTOR_COLORS[factor], lw=0.9, alpha=0.45, label="Rank IC")
    axes[2].plot(ic["signal_date"], ic["rolling_12"], color=FACTOR_COLORS[factor], lw=2.2, label="12 期滚动均值（仅辅助阅读）")
    axes[2].axhline(0, color="#333333", lw=1.0)
    axes[2].set_title("Rank IC 与 12 期滚动均值")
    axes[2].set_ylabel("Rank IC")
    axes[2].set_xlabel("信号日")
    axes[2].legend(loc="upper left", frameon=False)
    _date_axis(axes[2])
    start = min(groups["signal_date"].min(), ic["signal_date"].min()).date()
    end = max(groups["signal_date"].max(), ic["signal_date"].max()).date()
    fig.suptitle(f"{FACTOR_LABELS[factor]} 因子：分组收益、Q5-Q1 与 Rank IC", fontsize=15)
    _footer(fig, f"来源：factor_quantile_returns.csv、factor_ic_series.csv｜{start}—{end}｜收益为下一持有期结果；非策略 gross/net，成本不适用")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    _save(fig, output)


def plot_gross_nav(inputs: FigureInputs, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for factor in FACTORS:
        item = inputs.gross_daily.loc[inputs.gross_daily["factor"].eq(factor)].sort_values("trade_date")
        ax.plot(item["trade_date"], item["nav"], color=FACTOR_COLORS[factor], lw=2.0, label=f"{FACTOR_LABELS[factor]} gross")
    benchmark = inputs.gross_daily.loc[inputs.gross_daily["factor"].eq("pe")].sort_values("trade_date")
    ax.plot(benchmark["trade_date"], benchmark["benchmark_nav"], color=BENCHMARK_COLOR, lw=2.1, label="H00300 全收益")
    if "ew_benchmark_return" in benchmark.columns:
        equal_weight = (1 + benchmark["ew_benchmark_return"].fillna(0.0)).cumprod()
        ax.plot(benchmark["trade_date"], equal_weight, color=BENCHMARK_COLOR, lw=1.6, linestyle="--",
                label="等权全 A")
    ax.set_title("三因子策略毛净值与基准")
    ax.set_ylabel("净值（初始值 = 1）")
    ax.set_xlabel("交易日")
    ax.set_ylim(bottom=0)
    _date_axis(ax)
    ax.legend(ncol=4, loc="upper left", frameon=False)
    _footer(fig, "来源：backtest_daily_returns.parquet｜2014-10-08—2025-12-31｜策略：gross、未扣成本；H00300 与等权全 A 为全收益口径｜执行日策略从开盘开始，基准为完整日收益")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, output)


def plot_gross_vs_net(inputs: FigureInputs, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, factor in zip(axes, FACTORS):
        gross = inputs.gross_daily.loc[inputs.gross_daily["factor"].eq(factor)].sort_values("trade_date")
        ax.plot(gross["trade_date"], gross["nav"], color=FACTOR_COLORS[factor], lw=2.2, linestyle="-", label="gross")
        for scenario in (0, 10, 20):
            item = inputs.cost_daily.loc[
                inputs.cost_daily["factor"].eq(factor) & inputs.cost_daily["slippage_bp"].eq(scenario)
            ].sort_values("trade_date")
            style = SCENARIO_STYLES[scenario]
            ax.plot(item["trade_date"], item["net_nav"], color=style["color"], linestyle=style["linestyle"],
                    lw=1.8, label=f"net {scenario} bp")
        ax.set_title(f"{FACTOR_LABELS[factor]}：gross 与 net 0/10/20 bp")
        ax.set_ylabel("净值")
        ax.set_ylim(bottom=0)
        ax.legend(ncol=4, loc="upper left", frameon=False)
    axes[-1].set_xlabel("交易日")
    _date_axis(axes[-1])
    fig.suptitle("交易成本情景对三因子策略净值的影响", fontsize=15)
    _footer(fig, "来源：backtest_daily_returns.parquet、backtest_net_daily_returns.parquet｜2014-10-08—2025-12-31｜gross 为实线；net 为虚线｜0 bp 仍含双边 3 bp 佣金和卖出印花税")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    _save(fig, output)


def plot_drawdowns(inputs: FigureInputs, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
    benchmark = inputs.drawdowns.loc[
        inputs.drawdowns["series_type"].eq("benchmark") & inputs.drawdowns["factor"].eq("H00300")
    ].sort_values("trade_date")
    for ax, factor in zip(axes, FACTORS):
        gross = inputs.drawdowns.loc[
            inputs.drawdowns["series_type"].eq("gross") & inputs.drawdowns["factor"].eq(factor)
        ].sort_values("trade_date")
        ax.plot(gross["trade_date"], gross["drawdown"], color=FACTOR_COLORS[factor], lw=1.7,
                label=f"{FACTOR_LABELS[factor]} gross")
        ax.plot(benchmark["trade_date"], benchmark["drawdown"], color=BENCHMARK_COLOR, lw=1.4, alpha=0.8,
                label="H00300 全收益")
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(f"{FACTOR_LABELS[factor]} 毛收益策略与 H00300 回撤")
        ax.set_ylabel("回撤")
        _percent_axis(ax)
        ax.legend(loc="lower left", frameon=False)
    axes[-1].set_xlabel("交易日")
    _date_axis(axes[-1])
    fig.suptitle("毛收益策略与 H00300 的历史回撤", fontsize=15)
    _footer(fig, "来源：performance_drawdowns.parquet｜2014-10-08—2025-12-31｜策略：gross、未扣成本；H00300：全收益｜回撤相对含初始净值 1 的历史峰值")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    _save(fig, output)


def _annotated_heatmap(ax: plt.Axes, values: np.ndarray, row_labels: list[str], col_labels: list[str],
                       norm: TwoSlopeNorm, title: str, percent: bool = True) -> object:
    image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "—" if np.isnan(value) else (f"{value:.1%}" if percent else f"{value:.2f}")
            color = "white" if not np.isnan(value) and abs(norm(value) - 0.5) > 0.28 else "#222222"
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color=color)
    return image


def plot_monthly_returns(inputs: FigureInputs, output: Path) -> None:
    gross = inputs.monthly.loc[inputs.monthly["series_type"].eq("gross")].copy()
    gross["month_date"] = pd.to_datetime(gross["month"] + "-01")
    gross["year"] = gross["month_date"].dt.year
    gross["month_number"] = gross["month_date"].dt.month
    limit = float(gross["monthly_return"].abs().max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    years = list(range(int(gross["year"].min()), int(gross["year"].max()) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    image = None
    for ax, factor in zip(axes, FACTORS):
        item = gross.loc[gross["factor"].eq(factor)]
        pivot = item.pivot(index="year", columns="month_number", values="monthly_return").reindex(index=years, columns=range(1, 13))
        image = _annotated_heatmap(ax, pivot.to_numpy(), [str(year) for year in years],
                                   [str(month) for month in range(1, 13)], norm, FACTOR_LABELS[factor])
        ax.set_xlabel("月份")
    axes[0].set_ylabel("年份")
    fig.suptitle("三因子毛收益策略的月度收益率", fontsize=15)
    _footer(fig, "来源：performance_monthly_returns.csv｜2014-10—2025-12｜单位：%｜仅 gross、未扣成本；缺失月份显示为“—”｜三个面板使用统一色阶")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.91, wspace=0.14)
    color_axis = fig.add_axes([0.92, 0.18, 0.014, 0.64])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("月收益率")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _save(fig, output)


def _metric_order(metrics: pd.DataFrame) -> pd.DataFrame:
    ordered_ids = [f"{factor}_{kind}" for factor in FACTORS for kind in ("gross", "net_0bp", "net_10bp", "net_20bp")]
    ordered_ids.extend(["h00300_benchmark", "ew_all_a_benchmark", "cw_all_a_benchmark"])
    present = [series_id for series_id in ordered_ids if series_id in set(metrics["series_id"])]
    remaining = [series_id for series_id in metrics["series_id"] if series_id not in set(present)]
    return metrics.set_index("series_id").loc[present + remaining].reset_index()


def plot_neutralization(inputs: FigureInputs, output: Path) -> None:
    """Mean Rank IC before and after neutralization, with the t statistic annotated."""
    order = (("raw", "原始分数"), ("raw_neutral_sample", "同子样本"), ("neutral", "行业+市值中性化"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    for axis, factor in zip(axes, FACTORS):
        labels, values, t_values = [], [], []
        for score_type, label in order:
            row = inputs.factor_summary.loc[
                inputs.factor_summary["factor"].eq(factor) & inputs.factor_summary["score_type"].eq(score_type)
            ]
            if row.empty:
                continue
            labels.append(label)
            values.append(float(row["ic_mean"].iloc[0]))
            t_value = row["ic_t"].iloc[0]
            t_values.append(float(t_value) if pd.notna(t_value) else np.nan)
        positions = np.arange(len(values))
        axis.bar(positions, values, color=FACTOR_COLORS[factor], alpha=0.85, width=0.6)
        axis.axhline(0, color=BENCHMARK_COLOR, linewidth=0.9)
        axis.set_xticks(positions, labels, fontsize=9)
        axis.set_title(f"{FACTOR_LABELS[factor]} 平均 Rank IC")
        axis.set_ylabel("IC 均值")
        axis.grid(axis="y", alpha=0.25)
        for position, value, t_value in zip(positions, values, t_values):
            text = f"{value:.4f}\nt={t_value:.1f}" if np.isfinite(t_value) else f"{value:.4f}"
            axis.annotate(text, (position, value), textcoords="offset points",
                          xytext=(0, 6 if value >= 0 else -18), ha="center", fontsize=8)
    fig.suptitle("中性化前后：平均 Rank IC 与 t 值", fontsize=15)
    _footer(fig, "来源：factor_test_summary.csv｜2014-09-30—2025-11-28｜同子样本 = 在与中性化相同的样本上重算原始 IC｜IC 为月度 Spearman 相关")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    _save(fig, output)


def plot_metric(inputs: FigureInputs, metric: str, title: str, ylabel: str, output: Path, percent: bool) -> None:
    data = _metric_order(inputs.metrics)
    colors = [FACTOR_COLORS.get(factor, BENCHMARK_COLOR) for factor in data["factor"]]
    hatches = []
    labels = []
    for row in data.itertuples(index=False):
        if row.series_type == "benchmark":
            hatches.append("")
            labels.append(str(row.factor))
        elif row.series_type == "gross":
            hatches.append("")
            labels.append(f"{FACTOR_LABELS[row.factor]}\ngross")
        else:
            scenario = int(row.slippage_bp)
            hatches.append(SCENARIO_STYLES[scenario]["hatch"])
            labels.append(f"{FACTOR_LABELS[row.factor]}\nnet {scenario}bp")
    fig, ax = plt.subplots(figsize=(15, 7))
    bars = ax.bar(range(len(data)), data[metric], color=colors, edgecolor="#333333", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(range(len(data)), labels, fontsize=8)
    ax.set_title(title, loc="left", pad=12)
    ax.set_ylabel(ylabel)
    if percent:
        _percent_axis(ax)
    ax.legend(handles=[
        Patch(facecolor="#FFFFFF", edgecolor="#333333", label="gross"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333", hatch="//", label="net 0 bp"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333", hatch="xx", label="net 10 bp"),
        Patch(facecolor="#FFFFFF", edgecolor="#333333", hatch="..", label="net 20 bp"),
        Patch(facecolor=BENCHMARK_COLOR, label="基准"),
    ], ncol=5, loc="lower right", bbox_to_anchor=(1.0, 1.005), frameon=False)
    _footer(fig, "来源：performance_metrics.csv｜2014-10-08—2025-12-31｜gross 未扣成本；net 0/10/20 bp 均含佣金与印花税，另按情景加入滑点｜H00300 为全收益指数，EW/CW 为股票池等权与市值加权")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, output)


def plot_turnover_and_cost(inputs: FigureInputs, output: Path) -> None:
    gross = inputs.metrics.loc[inputs.metrics["series_type"].eq("gross")].set_index("factor").loc[list(FACTORS)]
    net = inputs.metrics.loc[inputs.metrics["series_type"].eq("net")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    axes[0].bar([FACTOR_LABELS[f] for f in FACTORS], gross["annualized_turnover"],
                color=[FACTOR_COLORS[f] for f in FACTORS])
    axes[0].set_title("三因子策略年化换手率")
    axes[0].set_ylabel("年化换手率")
    _percent_axis(axes[0])
    x = np.arange(len(FACTORS))
    width = 0.24
    for index, scenario in enumerate((0, 10, 20)):
        item = net.loc[net["slippage_bp"].eq(scenario)].set_index("factor").loc[list(FACTORS)]
        style = SCENARIO_STYLES[scenario]
        axes[1].bar(x + (index - 1) * width, item["cumulative_transaction_cost"], width,
                    label=f"{scenario} bp", color=style["color"], edgecolor="#333333", hatch=style["hatch"])
    axes[1].set_xticks(x, [FACTOR_LABELS[f] for f in FACTORS])
    axes[1].set_title("0/10/20 bp 情景的累计交易成本")
    axes[1].set_ylabel("累计成本（净值点数）")
    _percent_axis(axes[1])
    axes[1].legend(ncol=3, loc="upper right", frameon=False)
    fig.suptitle("换手率与交易成本：高换手策略对成本更敏感", fontsize=15)
    _footer(fig, "来源：performance_metrics.csv｜2014-10-08—2025-12-31｜换手率在各滑点情景相同；0 bp 仍含双边 3 bp 佣金和卖出印花税｜成本为累计 NAV 点数")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, output)


def plot_robustness(inputs: FigureInputs, factor: str, output: Path) -> None:
    data = inputs.robustness.loc[inputs.robustness["factor"].eq(factor)].copy()
    limit = float(data["net_cumulative_return"].abs().max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    image = None
    for ax, scenario in zip(axes, (0, 10, 20)):
        item = data.loc[data["slippage_bp"].eq(scenario)].copy()
        item["top"] = item["selection_quantile"].map({0.1: "Top 10%", 0.2: "Top 20%", 0.3: "Top 30%"})
        item["frequency"] = item["rebalance_frequency"].map({"monthly": "Monthly", "quarterly": "Quarterly"})
        pivot = item.pivot(index="frequency", columns="top", values="net_cumulative_return").reindex(
            index=["Monthly", "Quarterly"], columns=["Top 10%", "Top 20%", "Top 30%"]
        )
        image = _annotated_heatmap(ax, pivot.to_numpy(), list(pivot.index), list(pivot.columns), norm,
                                   f"net {scenario} bp")
        ax.set_xlabel("选股比例")
    axes[0].set_ylabel("调仓频率")
    fig.suptitle(f"{FACTOR_LABELS[factor]} 稳健性：选股比例、调仓频率与成本情景", fontsize=15)
    _footer(fig, "来源：robustness_metrics.csv｜2014-10-08—2025-12-31｜单元格：net 累计收益（%）｜0 bp 仍含佣金和印花税｜同一因子三面板统一色阶，不标记最佳参数")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.18, top=0.82, wspace=0.18)
    color_axis = fig.add_axes([0.92, 0.23, 0.013, 0.50])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("净累计收益")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _save(fig, output)


def build_figure_index() -> str:
    rows = [
        ("01_sample_quality.png", "股票池规模与三因子可排名样本如何变化？", "factor_period_quality.csv: signal_date, factor, pool_count, rankable_count, unrankable_count, unrankable_by_reason", "不适用", "2014-09-30—2025-12-31", "Momentum 前 6 期无完整 6 个月历史价格；不插值。"),
        ("02_factor_pe.png", "PE 的五分位、Q5-Q1 与 Rank IC 如何变化？", "factor_quantile_returns.csv: quantile, cumulative_group_return, q_high_minus_q_low, cumulative_q_high_minus_q_low; factor_ic_series.csv: rank_ic", "分组未来收益，非策略 gross/net；成本不适用", "2014-09-30—2025-11-28", "每期 Q5-Q1 只计算一次，累计值在同期五个分位行中完全一致。"),
        ("03_factor_momentum.png", "Momentum 的五分位、Q5-Q1 与 Rank IC 如何变化？", "同 02，factor=momentum", "分组未来收益，非策略 gross/net；成本不适用", "2015-03-31—2025-11-28", "前 6 个信号期无完整回看历史，因此不填造结果。"),
        ("04_factor_roe.png", "ROE 的五分位、Q5-Q1 与 Rank IC 如何变化？", "同 02，factor=roe", "分组未来收益，非策略 gross/net；成本不适用", "2014-09-30—2025-11-28", "分组结果不代表开盘可成交策略。"),
        ("05_gross_nav_vs_benchmark.png", "三因子毛净值相对基准如何变化？", "backtest_daily_returns.parquet: trade_date, factor, nav, benchmark_nav, ew_benchmark_return", "策略 gross；H00300 与等权全 A 全收益；未扣策略成本", "2014-10-08—2025-12-31", "策略执行日从开盘开始，基准使用完整日收益，盘中起点不完全一致。"),
        ("06_gross_vs_net.png", "交易成本情景如何改变各因子净值？", "backtest_daily_returns.parquet: nav; backtest_net_daily_returns.parquet: net_nav, slippage_bp", "gross 与 net 0/10/20 bp；0 bp 仍含佣金和印花税", "2014-10-08—2025-12-31", "最低佣金与盘口冲击未纳入；滑点为研究假设。"),
        ("07_drawdowns.png", "各毛收益策略与 H00300 的历史回撤如何比较？", "performance_drawdowns.parquet: series_type, factor, trade_date, drawdown", "策略 gross；H00300 全收益；不展示 net", "2014-10-08—2025-12-31", "策略与基准执行日盘中起点不完全一致。"),
        ("08_monthly_returns.png", "三因子毛收益策略每月表现如何？", "performance_monthly_returns.csv: series_type, factor, month, monthly_return", "仅 gross；未扣成本", "2014-10—2025-12", "统一色阶保留极端月份；缺失格显示为破折号。"),
        ("09a_cumulative_return.png", "累计收益在 gross、net 成本情景和基准间如何比较？", "performance_metrics.csv: cumulative_return", "gross、net 0/10/20 bp、H00300", "2014-10-08—2025-12-31", "H00300 不适用策略成本。"),
        ("09b_annualized_return.png", "年化收益在各序列间如何比较？", "performance_metrics.csv: annualized_return", "同 09a", "2014-10-08—2025-12-31", "按 252 个交易日年化。"),
        ("09c_annualized_volatility.png", "年化波动率在各序列间如何比较？", "performance_metrics.csv: annualized_volatility", "同 09a", "2014-10-08—2025-12-31", "日收益样本标准差按 252 个交易日年化。"),
        ("09d_max_drawdown.png", "最大回撤在各序列间如何比较？", "performance_metrics.csv: max_drawdown", "同 09a", "2014-10-08—2025-12-31", "历史峰值包含初始净值 1。"),
        ("09e_sharpe_ratio.png", "Sharpe 在各序列间如何比较？", "performance_metrics.csv: sharpe_ratio", "同 09a", "2014-10-08—2025-12-31", "年化无风险利率为 0%，年化交易日为 252。"),
        ("10_turnover_and_cost.png", "因子换手率与累计交易成本有何差异？", "performance_metrics.csv: annualized_turnover, cumulative_transaction_cost", "换手率与滑点无关；成本为 net 0/10/20 bp", "2014-10-08—2025-12-31", "累计成本为归一化组合的 NAV 点数，不等于期末毛净值减净净值。"),
        ("11_robustness_pe.png", "PE 净累计收益对选股比例、调仓频率和成本是否敏感？", "robustness_metrics.csv: selection_quantile, rebalance_frequency, slippage_bp, net_cumulative_return", "net 0/10/20 bp", "2014-10-08—2025-12-31", "边界同分全部保留；不选择最佳参数。"),
        ("12_robustness_momentum.png", "Momentum 净累计收益对参数是否敏感？", "同 11，factor=momentum", "net 0/10/20 bp", "2014-10-08—2025-12-31", "部分参数下收益方向反转；不选择最佳参数。"),
        ("13_robustness_roe.png", "ROE 净累计收益对参数是否敏感？", "同 11，factor=roe", "net 0/10/20 bp", "2014-10-08—2025-12-31", "边界同分全部保留；不选择最佳参数。"),
        ("14_neutralization_ic.png", "中性化前后平均 Rank IC 与 t 值如何变化？", "factor_test_summary.csv: score_type, ic_mean, ic_t", "分组未来收益，非策略 gross/net；成本不适用", "2014-09-30—2025-11-28", "同子样本列用于排除样本变化的影响。"),
    ]
    lines = [
        "# 研究图表索引", "",
        "图表读取管线已固化的产物，不改写收益、成本、回撤或稳健性规则。", "",
        "## 统一视觉与口径", "",
        "- Matplotlib 白底；PE `#0072B2`、Momentum `#E69F00`、ROE `#009E73`、H00300 `#4D4D4D`。",
        "- gross 使用实线；net 0/10/20 bp 分别使用由浅到深颜色及 `--` / `-.` / `:` 线型。",
        "- 中文字体为 Microsoft YaHei（后备 SimHei）；所有收益、回撤、成本与换手率均以百分比或明确单位显示。", "",
        "## 逐图索引", "",
        "| 文件名 | 回答的问题 | 数据来源及字段 | gross/net 与成本口径 | 日期范围 | 已知限制 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(f"| `{name}` | {question} | `{source}` | {basis} | {dates} | {limit} |" for name, question, source, basis, dates, limit in rows)
    lines.extend(["", "## 固定研究限制", "",
                  "- H00300 为 CSMAR `Idxtrd08` 全收益；策略首次执行日从开盘开始，基准为完整日收益。",
                  "- 0 bp 仅表示无滑点，仍包含双边 3 bp 佣金和卖出印花税；最低佣金未纳入。",
                  "- PE1TTM 按交易日可用的既定假设处理，原始数据无法证明其财务分母已严格按披露时点消除前视偏差。",
                  "- 涨跌停无法可靠识别，正常交易状态和有效开盘价不保证实际成交。", ""])
    return "\n".join(lines)


def run(config: VisualizationConfig) -> tuple[Path, ...]:
    inputs = load_inputs(config.processed_dir)
    validate_inputs(inputs)
    _set_style()
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    for old in config.figure_dir.glob("*.png"):
        if old.name not in FIGURE_FILES:
            raise ValueError(f"Unexpected existing PNG is not part of the figure set: {old.name}")
    plot_sample_quality(inputs, config.figure_dir / FIGURE_FILES[0])
    plot_factor(inputs, "pe", config.figure_dir / FIGURE_FILES[1])
    plot_factor(inputs, "momentum", config.figure_dir / FIGURE_FILES[2])
    plot_factor(inputs, "roe", config.figure_dir / FIGURE_FILES[3])
    plot_gross_nav(inputs, config.figure_dir / FIGURE_FILES[4])
    plot_gross_vs_net(inputs, config.figure_dir / FIGURE_FILES[5])
    plot_drawdowns(inputs, config.figure_dir / FIGURE_FILES[6])
    plot_monthly_returns(inputs, config.figure_dir / FIGURE_FILES[7])
    metric_specs = (
        ("cumulative_return", "累计收益：gross、net 成本情景与基准", "累计收益", True),
        ("annualized_return", "年化收益：gross、net 成本情景与基准", "年化收益", True),
        ("annualized_volatility", "年化波动率：gross、net 成本情景与基准", "年化波动率", True),
        ("max_drawdown", "最大回撤：gross、net 成本情景与基准", "最大回撤", True),
        ("sharpe_ratio", "Sharpe Ratio：gross、net 成本情景与基准", "Sharpe Ratio", False),
    )
    for spec, filename in zip(metric_specs, FIGURE_FILES[8:13]):
        metric, title, ylabel, percent = spec
        plot_metric(inputs, metric, title, ylabel, config.figure_dir / filename, percent)
    plot_turnover_and_cost(inputs, config.figure_dir / FIGURE_FILES[13])
    plot_robustness(inputs, "pe", config.figure_dir / FIGURE_FILES[14])
    plot_robustness(inputs, "momentum", config.figure_dir / FIGURE_FILES[15])
    plot_robustness(inputs, "roe", config.figure_dir / FIGURE_FILES[16])
    plot_neutralization(inputs, config.figure_dir / FIGURE_FILES[17])
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    config.index_path.write_text(build_figure_index(), encoding="utf-8")
    outputs = tuple(config.figure_dir / name for name in FIGURE_FILES)
    missing = [str(path) for path in outputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"Figure generation produced no valid image files: {missing}")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the research figures.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_visualization_config(args.config))
    print(f"Generated {len(outputs)} figures and the figure index.")


if __name__ == "__main__":
    main()
