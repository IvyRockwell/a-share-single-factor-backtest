"""PE specification robustness: current PE, E/P with losses, B/M, and 12-month listing.

The three primary factors are fixed for the whole study, so the natural
robustness question for the value factor is not "which other metric scores best"
but "do the PE conclusions survive plausible alternative definitions of the same
idea".  Four specifications are built here:

``pe``
    The current factor: disclosure-timed TTM net profit / market cap, restricted
    to positive earnings, lower is better.
``ep_all``
    Earnings yield ``TTM profit / market cap`` with *negative* earnings kept.
    The rank transform makes the scale irrelevant, but the raw ratio is not
    truncated or winsorised, so the report states its extreme values.
``bm``
    Book-to-market using the latest *already disclosed* annual shareholder
    equity, the same point-in-time report the ROE factor uses.
``pe_12m``
    The current PE on a stock pool that additionally requires twelve calendar
    months of listing history, re-ranked inside the filtered cross-section.

The module also reuses the generic factor-test functions for the four
specifications and runs the long-only pipeline for the three specifications that
share the primary pool.  ``pe_12m`` deliberately stops at the factor test: its
pool is smaller, so reusing the primary equal-weight benchmark would compare
different universes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.attribution import (
    AttributionConfig,
    build_alpha_table,
    load_attribution_config,
)
from src.backtest import BacktestConfig, build_backtest_outputs, load_backtest_config
from src.costs import CostConfig, build_cost_outputs, load_cost_config
from src.delisting import load_delisting_events
from src.factor_test import (
    FactorTestConfig,
    build_factor_test_details,
    build_ic_series,
    build_quantile_returns,
    build_summary,
    load_factor_test_config,
)
from src.performance import PerformanceConfig, build_performance_outputs, load_performance_config
from src.portfolio import PortfolioConfig, build_portfolio_outputs, load_portfolio_config

DEFAULT_SPECS = ("pe", "ep_all", "bm", "pe_12m")
DEFAULT_LISTED_MONTHS = 12
#: The three specifications that keep the primary pool and can therefore reuse
#: the same equal-weight benchmark.  ``pe_12m`` is absent on purpose.
PORTFOLIO_SPECS = ("pe", "ep_all", "bm")
#: ``attribution.build_alpha_table`` labels the same-pool equal-weight benchmark
#: with this human-readable label; it is not the ``ew_pool_monthly`` factor id.
EW_POOL_LABEL = "EW_POOL_MONTHLY"
#: The tests are computed for ``pe`` / ``ep_all`` / ``bm`` / ``pe_12m``; the
#: portfolio stage reuses the engine's fixed three factor slots.
SPEC_SLOTS = {"pe": "pe", "ep_all": "momentum", "bm": "roe"}
SPEC_COLUMNS = (
    "signal_date", "stock_code", "pe_pit", "pe_is_rankable", "pe_unrankable_reason",
    "pe_pit_ttm_net_profit", "market_cap", "shareholder_equity", "roe_actual_disclosure_date",
    "pe_rank", "pe_score",
)
SUMMARY_COLUMNS = (
    "spec", "ic_valid_periods", "ic_mean", "ic_win_rate", "ic_t_nw",
    "spread_mean", "spread_t_nw", "mean_rankable_stocks",
)
PORTFOLIO_COLUMNS = (
    "spec", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown",
    "alpha_annualized", "alpha_t", "excess_mean_monthly", "excess_t_nw", "n_months",
)


@dataclass(frozen=True)
class PeSpecConfig:
    """Where the inputs live and which specifications to build."""

    processed_dir: Path
    specs: tuple[str, ...] = DEFAULT_SPECS
    listed_months: int = DEFAULT_LISTED_MONTHS
    quantile_groups: int = 5
    min_ic_pairs: int = 50


@dataclass(frozen=True)
class PeSpecOutputs:
    details: pd.DataFrame
    summary: pd.DataFrame
    ic: pd.DataFrame
    quantiles: pd.DataFrame
    portfolio: pd.DataFrame


def validate_specs(specs: tuple[str, ...]) -> tuple[str, ...]:
    """Reject an empty or unknown specification list."""
    chosen = tuple(str(value) for value in specs)
    if not chosen:
        raise ValueError("pe_specs.specs must not be empty.")
    unknown = [value for value in chosen if value not in DEFAULT_SPECS]
    if unknown:
        raise ValueError(f"Unknown PE specification(s): {sorted(unknown)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError("pe_specs.specs must not repeat a specification.")
    return chosen


def load_pe_spec_config(config_path: Path,
                        test_config: FactorTestConfig | None = None) -> PeSpecConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    section = settings.get("pe_specs") or {}
    test = test_config or load_factor_test_config(config_path)
    listed_months = int(section.get("listed_months", DEFAULT_LISTED_MONTHS))
    if listed_months < 0:
        raise ValueError("pe_specs.listed_months must not be negative.")
    return PeSpecConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        specs=validate_specs(tuple(section.get("specs") or DEFAULT_SPECS)),
        listed_months=listed_months,
        quantile_groups=test.quantile_groups,
        min_ic_pairs=test.min_ic_pairs,
    )


def _first_reason(frame: pd.DataFrame, conditions: list[tuple[str, pd.Series]]) -> pd.Series:
    """Return the first applicable unrankable reason per row."""
    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name, condition in conditions:
        reason = reason.mask(reason.isna() & condition.fillna(False), name)
    return reason


def _rank_cross_section(frame: pd.DataFrame, raw_column: str, rankable_column: str,
                        prefix: str, higher_is_better: bool) -> None:
    """Rank one specification inside each month and map it to a (0, 1] score.

    Ties share the average rank, and the score is ``(n - rank + 1) / n`` for a
    "lower is better" metric and the mirror image for "higher is better", so the
    best observation in every cross-section always scores 1.0.
    """
    frame[f"{prefix}_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame[f"{prefix}_score"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    valid = frame[rankable_column].fillna(False)
    if not valid.any():
        return
    ranked = frame.loc[valid].groupby("signal_date")[raw_column].rank(
        method="average", ascending=not higher_is_better
    )
    counts = frame.loc[valid].groupby("signal_date")[raw_column].transform("size")
    frame.loc[valid, f"{prefix}_rank"] = ranked.astype("Float64")
    frame.loc[valid, f"{prefix}_score"] = ((counts - ranked + 1) / counts).astype("Float64")


def _attach_listing(frame: pd.DataFrame, listing: pd.DataFrame, listed_months: int) -> pd.Series:
    """Flag the rows listed for at least ``listed_months`` calendar months."""
    keys = ["signal_date", "stock_code"]
    dates = listing[keys + ["listing_date"]].copy()
    dates["signal_date"] = pd.to_datetime(dates["signal_date"]).astype("datetime64[ns]")
    dates["listing_date"] = pd.to_datetime(dates["listing_date"]).astype("datetime64[ns]")
    dates["stock_code"] = dates["stock_code"].astype("string")
    merged = frame[keys].astype({"signal_date": "datetime64[ns]", "stock_code": "string"}).merge(
        dates, on=keys, how="left", validate="one_to_one"
    )
    # Same rule the point-in-time universe uses for its own listing filter:
    # "listed for N calendar months" means listing date + N months <= signal date.
    threshold = merged["listing_date"] + pd.DateOffset(months=listed_months)
    eligible = merged["listing_date"].notna() & merged["signal_date"].ge(threshold)
    return pd.Series(eligible.to_numpy(), index=frame.index, dtype=bool)


def add_pe_specs(factor_details: pd.DataFrame, listing: pd.DataFrame,
                 config: PeSpecConfig) -> pd.DataFrame:
    """Add ``raw``/``is_rankable``/``unrankable_reason``/``rank``/``score`` per spec."""
    missing = [column for column in SPEC_COLUMNS if column not in factor_details.columns]
    if missing:
        raise ValueError(f"factor_details is missing columns required by pe_specs: {missing}")
    frame = factor_details[list(SPEC_COLUMNS)].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).astype("datetime64[ns]")
    frame["stock_code"] = frame["stock_code"].astype("string")

    market_cap = frame["market_cap"].astype("float64")
    profit = frame["pe_pit_ttm_net_profit"].astype("float64")
    equity = frame["shareholder_equity"].astype("float64")
    cap_usable = market_cap.notna() & market_cap.gt(0)
    # A zero cap would divide to infinity; mask it out before the division.
    cap_safe = market_cap.where(cap_usable)

    # 1. The current PE: keep its published raw value, eligibility, and scores.
    frame["pe_raw"] = frame["pe_pit"]
    frame["pe_is_rankable"] = frame["pe_is_rankable"].fillna(False).astype(bool)
    frame["pe_unrankable_reason"] = frame["pe_unrankable_reason"]

    # 2. E/P including loss makers.  No truncation and no winsorising: the rank
    #    transform only needs an ordering, and the report states the extremes.
    frame["ep_all_raw"] = (profit / cap_safe).where(profit.notna())
    frame["ep_all_is_rankable"] = cap_usable & profit.notna()
    frame["ep_all_unrankable_reason"] = _first_reason(frame, [
        ("missing_ttm_net_profit", ~profit.notna()),
        ("missing_or_nonpositive_market_cap", profit.notna() & ~cap_usable),
    ])
    _rank_cross_section(frame, "ep_all_raw", "ep_all_is_rankable", "ep_all", higher_is_better=True)

    # 3. Book-to-market on the latest disclosed annual report (point in time).
    frame["bm_raw"] = (equity / cap_safe).where(equity.gt(0))
    frame["bm_is_rankable"] = equity.gt(0) & cap_usable
    frame["bm_unrankable_reason"] = _first_reason(frame, [
        ("no_prior_annual_consolidated_disclosure", frame["roe_actual_disclosure_date"].isna()),
        ("missing_shareholder_equity", frame["roe_actual_disclosure_date"].notna() & equity.isna()),
        ("nonpositive_shareholder_equity", equity.notna() & equity.le(0)),
        ("missing_or_nonpositive_market_cap", (equity.notna() & equity.gt(0)) & ~cap_usable),
    ])
    _rank_cross_section(frame, "bm_raw", "bm_is_rankable", "bm", higher_is_better=True)

    # 4. The current PE restricted to names listed for at least N calendar months.
    listed_ok = _attach_listing(frame, listing, config.listed_months)
    frame["pe_12m_raw"] = frame["pe_pit"]
    frame["pe_12m_is_rankable"] = listed_ok & frame["pe_is_rankable"].to_numpy()
    frame["pe_12m_unrankable_reason"] = _first_reason(frame, [
        ("listed_less_than_12_months", ~listed_ok),
    ])
    uncovered = frame["pe_12m_unrankable_reason"].isna() & ~frame["pe_12m_is_rankable"].to_numpy()
    frame.loc[uncovered, "pe_12m_unrankable_reason"] = frame.loc[uncovered, "pe_unrankable_reason"]
    _rank_cross_section(frame, "pe_12m_raw", "pe_12m_is_rankable", "pe_12m", higher_is_better=False)
    return frame.sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


def build_spec_columns(specs: tuple[str, ...]) -> list[str]:
    """The five columns every specification contributes."""
    return [f"{spec}{suffix}" for spec in specs
            for suffix in ("_raw", "_is_rankable", "_unrankable_reason", "_rank", "_score")]


def validate_spec_details(details: pd.DataFrame, specs: tuple[str, ...]) -> None:
    """Check the ranking invariants of every specification."""
    missing = [column for column in build_spec_columns(specs) if column not in details.columns]
    if missing:
        raise ValueError(f"The specification details are missing columns: {missing}")
    if details.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("PE specification details contain duplicate keys.")
    # Cross-specification invariants first: they are the ones that would silently
    # leak information if a specification filtered its own pool.
    pe_rankable = details["pe_is_rankable"].fillna(False).astype(bool)
    twelve = details["pe_12m_is_rankable"].fillna(False).astype(bool)
    if (twelve & ~pe_rankable).any():
        raise ValueError("pe_12m must be a subset of the positive-PE sample.")
    ep_all = details["ep_all_is_rankable"].fillna(False).astype(bool)
    if (ep_all & details["ep_all_raw"].isna()).any():
        raise ValueError("ep_all must have a value on every rankable row.")
    for spec in specs:
        rankable = details[f"{spec}_is_rankable"].fillna(False).astype(bool)
        raw = details[f"{spec}_raw"].astype("float64")
        score = details[f"{spec}_score"].astype("float64")
        if raw[rankable].isna().any():
            raise ValueError(f"Rankable {spec} rows are missing a raw value.")
        if score[rankable].isna().any() or details.loc[rankable, f"{spec}_rank"].isna().any():
            raise ValueError(f"Rankable {spec} rows are missing a rank or score.")
        if score[rankable].le(0).any() or score[rankable].gt(1).any():
            raise ValueError(f"{spec} scores must fall in (0, 1].")
        if score[~rankable].notna().any() or details.loc[~rankable, f"{spec}_rank"].notna().any():
            raise ValueError(f"Unrankable {spec} rows must not carry a rank or score.")
        reason = details.loc[~rankable, f"{spec}_unrankable_reason"]
        if reason.isna().any():
            raise ValueError(f"Every unrankable {spec} row needs a reason.")


def build_spec_outputs(details: pd.DataFrame, forward_returns: pd.DataFrame,
                       config: PeSpecConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the generic quantile/IC/summary functions for every specification."""
    long = build_factor_test_details(
        details, forward_returns, config.quantile_groups, score_type="raw",
        score_suffix="_score", eligibility_suffix="_is_rankable", factors=config.specs,
    )
    _, quantiles = build_quantile_returns(long, config.quantile_groups)
    ic = build_ic_series(long, config.min_ic_pairs)
    summary = build_summary(long, quantiles, ic)
    rankable = long.groupby(["factor", "signal_date"])["factor_is_rankable"].sum()
    mean_rankable = rankable.groupby(level=0).mean()
    rows = []
    for spec in config.specs:
        item = summary.loc[summary["factor"].eq(spec)]
        if item.empty:
            continue
        row = item.iloc[0]
        rows.append({
            "spec": spec,
            "ic_valid_periods": int(row["ic_valid_periods"]),
            "ic_mean": row["ic_mean"],
            "ic_win_rate": row["ic_win_rate"],
            "ic_t_nw": row["ic_t_nw"],
            "spread_mean": row["spread_mean"],
            "spread_t_nw": row["spread_t_nw"],
            "mean_rankable_stocks": float(mean_rankable.get(spec, np.nan)),
        })
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)), ic, quantiles


def build_spec_portfolio(details: pd.DataFrame, daily: pd.DataFrame, benchmark: pd.DataFrame,
                         pool: pd.DataFrame, market_cap: pd.DataFrame,
                         benchmark_monthly: pd.DataFrame,
                         portfolio_config: PortfolioConfig, backtest_config: BacktestConfig,
                         cost_config: CostConfig, performance_config: PerformanceConfig,
                         attribution_config: AttributionConfig,
                         delistings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the long-only pipeline for the specifications that share the primary pool.

    The engine is written around the three fixed factor slots, so the three
    specifications are mapped onto those slots and run in a single pass.  Factor
    labels are translated back into specification names afterwards.
    """
    specs = tuple(spec for spec in PORTFOLIO_SPECS if f"{spec}_score" in details.columns)
    if not specs:
        raise ValueError("No portfolio specification is available.")
    frame = details.copy()
    for spec in specs:
        slot = SPEC_SLOTS[spec]
        frame[f"{slot}_score"] = details[f"{spec}_score"].astype("float64")
        frame[f"{slot}_is_rankable"] = details[f"{spec}_is_rankable"].fillna(False).astype(bool)
    portfolio = build_portfolio_outputs(frame, daily, portfolio_config)
    backtest = build_backtest_outputs(
        portfolio.targets, portfolio.actions, daily, benchmark, backtest_config, pool, market_cap,
        delistings=delistings,
    )
    cost = build_cost_outputs(backtest.daily, backtest.rebalance_records, cost_config)
    performance = build_performance_outputs(backtest.daily, cost.daily, cost.costs, performance_config)

    gross_metrics = performance.metrics.loc[
        performance.metrics["series_type"].eq("gross")
        & performance.metrics["factor"].isin([SPEC_SLOTS[spec] for spec in specs])
    ].copy()
    slot_to_spec = {SPEC_SLOTS[spec]: spec for spec in specs}
    gross_metrics["spec"] = gross_metrics["factor"].map(slot_to_spec)

    monthly = performance.monthly
    strategy = monthly.loc[
        monthly["series_type"].eq("gross") & monthly["factor"].isin(list(slot_to_spec)),
        ["factor", "month", "monthly_return"],
    ].rename(columns={"monthly_return": "strategy_return"}).copy()
    strategy["return_type"] = "gross"
    strategy["month_key"] = strategy["month"].astype(str)
    hs300 = (
        backtest.monthly[["month", "benchmark_monthly_return"]]
        .drop_duplicates("month")
        .rename(columns={"month": "month_key"})
    )
    alpha, _ = build_alpha_table(
        strategy, hs300, benchmark_monthly, attribution_config,
        factors=tuple(SPEC_SLOTS[spec] for spec in specs),
    )
    alpha = alpha.loc[
        alpha["return_type"].eq("gross") & alpha["benchmark"].eq(EW_POOL_LABEL)
    ].copy()
    alpha["spec"] = alpha["factor"].map(slot_to_spec)
    return _combine_portfolio_rows(gross_metrics, alpha, specs)


def _combine_portfolio_rows(gross_metrics: pd.DataFrame, alpha: pd.DataFrame,
                            specs: tuple[str, ...]) -> pd.DataFrame:
    """Merge performance metrics with the equal-weight regression alpha per spec."""
    rows = []
    for spec in specs:
        metric = gross_metrics.loc[gross_metrics["spec"].eq(spec)]
        alpha_row = alpha.loc[alpha["spec"].eq(spec)]
        if metric.empty or alpha_row.empty:
            continue
        values = metric.iloc[0]
        stats = alpha_row.iloc[0]
        rows.append({
            "spec": spec,
            "annualized_return": values["annualized_return"],
            "annualized_volatility": values["annualized_volatility"],
            "sharpe_ratio": values["sharpe_ratio"],
            "max_drawdown": values["max_drawdown"],
            "alpha_annualized": stats["alpha_annualized"],
            "alpha_t": stats["alpha_t"],
            "excess_mean_monthly": stats["excess_mean_monthly"],
            "excess_t_nw": stats["excess_t_nw"],
            "n_months": int(stats["n_months"]),
        })
    return pd.DataFrame(rows, columns=list(PORTFOLIO_COLUMNS))


def validate_portfolio_outputs(portfolio: pd.DataFrame, specs: tuple[str, ...]) -> None:
    expected = [spec for spec in PORTFOLIO_SPECS if spec in specs]
    if len(portfolio) < 2:
        raise ValueError("At least two specifications are needed for a robustness comparison.")
    if list(portfolio["spec"]) != expected:
        raise ValueError("The specification portfolio table does not cover the primary-pool specs.")
    if portfolio.duplicated(["spec"]).any():
        raise ValueError("The specification portfolio table contains duplicate specs.")


def _format(value: float, digits: int = 4) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _format_percent(value: float, digits: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def render_report(outputs: PeSpecOutputs, config: PeSpecConfig) -> str:
    summary = outputs.summary
    lines = [
        "# PE specification robustness：正盈利 PE、含负盈利 E/P、B/M、12 个月上市门槛",
        "",
        "主因子集合（PE / Momentum / ROE）来自既有文献与全样本认知；本报告不新增因子，只问"
        "「价值因子换一种常见写法，结论是否还成立」。四个 spec 的样本不同，因此先看各自的有效期数与"
        "平均可排名股票数，再比较 IC 与多空价差。",
        "",
        "## 因子检验",
        "",
        "| spec | 有效 IC 期数 | 平均 Rank IC | IC 胜率 | IC t（NW） | Q5-Q1 月均 | Q5-Q1 t（NW） | 平均可排名股票数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.spec} | {int(row.ic_valid_periods)} | {_format(row.ic_mean)} | "
            f"{_format_percent(row.ic_win_rate, 1)} | {_format(row.ic_t_nw, 2)} | "
            f"{_format_percent(row.spread_mean)} | {_format(row.spread_t_nw, 2)} | "
            f"{_format(row.mean_rankable_stocks, 0)} |"
        )
    lines.extend([
        "",
        "## 同股票池纯多头 alpha",
        "",
        "`pe`、`ep_all`、`bm` 使用同一个主股票池，因此可以共用 `attribution_benchmark_monthly.csv` "
        "的同池月频等权基准；目标权重、ADV 容量（固定 1 亿元初始资金，不随 NAV 复利）、开盘成交、成本与退市核销规则与主回测完全一致。",
        "",
        "| spec | 年化 | 年化波动 | Sharpe | 最大回撤 | alpha 年化 | alpha t | 月均超额 | 超额 NW t |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    portfolio = outputs.portfolio
    for row in portfolio.itertuples(index=False):
        lines.append(
            f"| {row.spec} | {_format_percent(row.annualized_return)} | "
            f"{_format_percent(row.annualized_volatility)} | {_format(row.sharpe_ratio, 3)} | "
            f"{_format_percent(row.max_drawdown)} | {_format_percent(row.alpha_annualized)} | "
            f"{_format(row.alpha_t, 2)} | {_format_percent(row.excess_mean_monthly)} | "
            f"{_format(row.excess_t_nw, 2)} |"
        )
    extremes = {}
    for spec in ("ep_all", "bm"):
        rankable = outputs.details[f"{spec}_is_rankable"].fillna(False).astype(bool)
        values = outputs.details.loc[rankable, f"{spec}_raw"].astype("float64")
        extremes[spec] = (float(values.min()), float(values.max()), float(values.quantile(0.99)))
    lines.extend([
        "",
        "## 口径",
        "",
        "- `pe`：当前 PE，即按实际披露日重建的 TTM 净利润 ÷ 总市值，TTM 与市值都为正才排名，低 PE 高分。",
        "- `ep_all`：`E/P = TTM 净利润 ÷ 总市值`，包含负盈利，只要求 TTM 与市值非缺失且市值 > 0；E/P 越高分数越高。"
        "负盈利的 E/P 不截尾、不做 winsorize：rank 只使用了排序信息，但极端值仍会影响截面排序，"
        f"可排名样本的 E/P 最小值 {extremes['ep_all'][0]:.4g}、最大值 {extremes['ep_all'][1]:.4g}、99 分位 {extremes['ep_all'][2]:.4g}。",
        f"- `bm`：`B/M = 最新已披露年报股东权益 ÷ 总市值`，与 ROE 用同一份 PIT 年报；权益 > 0、市值 > 0 才排名。"
        f"可排名样本的 B/M 最小值 {extremes['bm'][0]:.4g}、最大值 {extremes['bm'][1]:.4g}、99 分位 {extremes['bm'][2]:.4g}。",
        f"- `pe_12m`：在当前 PE 的可排名样本上再要求上市满 {config.listed_months} 个自然月，"
        "并在**过滤后的横截面**内重新排名；不能沿用 6 个月门槛股票池的 `pe_score`。"
        "其不可排名原因会优先给出 `listed_less_than_12_months`。",
        "- `pe_12m` 的股票池与其他三个 spec 不同，本次**不做纯多头**：基准应同步换成 12 个月池的同池等权，"
        "直接套用主池基准会造成基准错配。",
        "- Q5-Q1 与纯多头都复用主流程的未来收益、分位定义与执行/成本规则；多空是统计组合，A 股融券受限。",
        "",
    ])
    return "\n".join(lines)


def run(config: PeSpecConfig, config_path: Path = Path("configs/config.yaml")) -> PeSpecOutputs:
    directory = config.processed_dir
    details = pd.read_parquet(directory / "factor_details.parquet")
    listing = pd.read_parquet(
        directory / "historical_universe.parquet",
        columns=["signal_date", "stock_code", "listing_date"],
    )
    forward = pd.read_parquet(directory / "factor_forward_returns.parquet")
    specs = add_pe_specs(details, listing, config)
    validate_spec_details(specs, config.specs)
    summary, ic, quantiles = build_spec_outputs(specs, forward, config)

    daily = pd.read_parquet(directory / "daily_trading.parquet", columns=[
        "stock_code", "trade_date", "trading_status", "open_price", "close_price", "total_return",
        "Dnvaltrd", "PreClosePrice", "Markettype", "market_type", "quality_flags",
    ])
    benchmark = pd.read_parquet(
        directory / "benchmark.parquet", columns=["trade_date", "total_return", "quality_flags"]
    )
    pool = pd.read_parquet(directory / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    market_cap = pd.read_parquet(directory / "market_cap.parquet", columns=[
        "signal_date", "stock_code", "market_cap", "has_market_cap",
    ])
    benchmark_monthly = pd.read_csv(directory / "attribution_benchmark_monthly.csv")
    delistings = load_delisting_events(directory / "status_changes.parquet")
    portfolio = build_spec_portfolio(
        specs, daily, benchmark, pool, market_cap, benchmark_monthly,
        load_portfolio_config(config_path),
        load_backtest_config(config_path),
        load_cost_config(config_path),
        load_performance_config(config_path),
        load_attribution_config(config_path),
        delistings=delistings,
    )
    validate_portfolio_outputs(portfolio, config.specs)
    outputs = PeSpecOutputs(specs, summary, ic, quantiles, portfolio)

    summary.to_csv(directory / "pe_specs_summary.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(directory / "pe_specs_ic.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(directory / "pe_specs_quantiles.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(directory / "pe_specs_portfolio.csv", index=False, encoding="utf-8-sig")
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "pe-specs.md").write_text(render_report(outputs, config), encoding="utf-8")
    print(f"Built {len(summary)} PE specifications and {len(portfolio)} long-only alpha rows.")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PE specification robustness checks.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    run(load_pe_spec_config(args.config), args.config)


if __name__ == "__main__":
    main()
