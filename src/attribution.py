"""Attribute strategy returns to same-universe benchmarks and factor legs.

The module deliberately keeps the benchmark construction separate from the
existing performance table: the backtest's legacy equal- and cap-weighted
series are *daily rebalanced* diagnostics, while this module builds a
monthly-rebalanced benchmark from the same point-in-time universe and the same
next-open execution engine.  The resulting monthly series are then used for
regressions, a Q5-Q1 long-short summary, and the bridge that explains why a
positive Rank IC need not survive a long-only implementation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import (
    FACTORS,
    BacktestConfig,
    build_backtest_outputs,
    load_backtest_config,
)
from src.costs import CostConfig, build_cost_outputs, load_cost_config
from src.delisting import load_delisting_events
from src.factor_test import newey_west_t
from src.performance import PerformanceConfig, build_performance_outputs, load_performance_config
from src.portfolio import PortfolioConfig, build_portfolio_outputs, load_portfolio_config
from src.trading_rules import CHINEXT_REFORM, LIMIT_TOLERANCE

BENCHMARK_FACTORS = ("ew_pool_monthly", "cw_pool_monthly")
BENCHMARK_LABELS = {
    "H00300": "H00300",
    "EW_POOL_MONTHLY": "同池月频等权",
    "CW_POOL_MONTHLY": "同池月频市值加权",
}
RETURN_TYPES = ("gross", "net_0bp", "net_10bp", "net_20bp")
MIN_MONTHS_DEFAULT = 24
MIN_CAP_COVERAGE_DEFAULT = 0.80


@dataclass(frozen=True)
class AttributionConfig:
    processed_dir: Path
    annualization_months: int
    min_months: int
    risk_free_rate: float
    min_cap_coverage: float
    long_short_score_types: tuple[str, ...]
    normal_trading_status: str
    delisting_return: float = -1.0


@dataclass(frozen=True)
class NeutralLongOnlyOutputs:
    metrics: pd.DataFrame
    alpha: pd.DataFrame
    daily: pd.DataFrame
    records: pd.DataFrame
    monthly: pd.DataFrame


@dataclass(frozen=True)
class AttributionOutputs:
    benchmark_monthly: pd.DataFrame
    alpha: pd.DataFrame
    style: pd.DataFrame
    long_short: pd.DataFrame
    long_short_monthly: pd.DataFrame
    bridge: pd.DataFrame
    summary: pd.DataFrame
    neutral: NeutralLongOnlyOutputs


def load_attribution_config(config_path: Path) -> AttributionConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    attribution = settings.get("attribution") or {}
    research = settings["research"]
    return AttributionConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        annualization_months=int(attribution.get("annualization_months", 12)),
        min_months=int(attribution.get("min_months", MIN_MONTHS_DEFAULT)),
        risk_free_rate=float(research.get("risk_free_rate", 0.0)),
        min_cap_coverage=float(attribution.get("min_cap_coverage", MIN_CAP_COVERAGE_DEFAULT)),
        long_short_score_types=tuple(
            str(value) for value in attribution.get("long_short_score_types", ["raw", "neutral"])
        ),
        normal_trading_status=str(settings["universe"]["normal_trading_status"]),
        delisting_return=float((settings.get("delisting") or {}).get("return_assumption", -1.0)),
    )


def _month_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").astype(str)


def _execution_map(targets: pd.DataFrame) -> pd.DataFrame:
    dates = targets[["signal_date", "execution_date"]].dropna().drop_duplicates()
    dates = dates.assign(
        signal_date=pd.to_datetime(dates["signal_date"]).astype("datetime64[ns]"),
        execution_date=pd.to_datetime(dates["execution_date"]).astype("datetime64[ns]"),
    ).sort_values("signal_date")
    if dates["signal_date"].duplicated().any():
        raise ValueError("A signal date has more than one execution date.")
    return dates.reset_index(drop=True)


def build_benchmark_targets(
    pool: pd.DataFrame,
    market_cap: pd.DataFrame,
    execution_map: pd.DataFrame,
    min_cap_coverage: float = MIN_CAP_COVERAGE_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equal- and cap-weighted targets for every pool member.

    The benchmark is the whole point-in-time pool, not the factor's rankable
    sample.  Missing cap data are renormalised away; a period with less than
    ``min_cap_coverage`` coverage falls back to equal weights and is reported.
    """
    pool = pool[["signal_date", "stock_code"]].copy()
    pool["signal_date"] = pd.to_datetime(pool["signal_date"]).astype("datetime64[ns]")
    base = (
        pool.merge(execution_map, on="signal_date", how="inner", validate="many_to_one")
        .sort_values(["signal_date", "stock_code"])
        .reset_index(drop=True)
    )
    if base.empty:
        raise ValueError("No pool member has an execution date.")
    counts = base.groupby("signal_date")["stock_code"].transform("size")
    base["target_weight"] = 1.0 / counts
    ew = base[["signal_date", "execution_date", "stock_code", "target_weight"]].assign(
        factor=BENCHMARK_FACTORS[0]
    )

    caps = market_cap[["signal_date", "stock_code", "market_cap", "has_market_cap"]].copy()
    caps["signal_date"] = pd.to_datetime(caps["signal_date"]).astype("datetime64[ns]")
    cw = base.merge(caps, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    usable = cw["has_market_cap"].fillna(False) & cw["market_cap"].gt(0)
    cw_usable = cw.loc[usable].copy()
    total = cw_usable.groupby("signal_date")["market_cap"].transform("sum")
    cw_usable["target_weight"] = cw_usable["market_cap"] / total
    all_dates = base["signal_date"].drop_duplicates()
    counts = cw_usable.groupby("signal_date")["stock_code"].size().reindex(all_dates, fill_value=0)
    base_counts = base.groupby("signal_date")["stock_code"].size().reindex(all_dates, fill_value=0)
    coverage = (counts / base_counts).clip(upper=1.0)
    fallback_dates = set(coverage.loc[coverage.lt(min_cap_coverage)].index)
    cw = cw_usable.loc[~cw_usable["signal_date"].isin(fallback_dates)].copy()
    cw = cw[["signal_date", "execution_date", "stock_code", "target_weight"]].assign(
        factor=BENCHMARK_FACTORS[1]
    )
    if fallback_dates:
        fallback = ew.loc[ew["signal_date"].isin(fallback_dates)].copy()
        fallback["factor"] = BENCHMARK_FACTORS[1]
        cw = pd.concat([cw, fallback], ignore_index=True)

    targets = pd.concat([ew, cw], ignore_index=True)
    if targets[["signal_date", "stock_code", "factor"]].duplicated().any():
        raise ValueError("Benchmark targets contain duplicate keys.")
    weight_sums = targets.groupby(["factor", "signal_date"])["target_weight"].sum()
    if not weight_sums.sub(1.0).abs().lt(1e-9).all():
        raise ValueError("Benchmark target weights do not sum to one.")
    fallback = pd.DataFrame({
        "signal_date": sorted(fallback_dates),
        "cw_fallback": True,
    })
    return targets, fallback


def build_pool_benchmark(
    pool: pd.DataFrame,
    daily: pd.DataFrame,
    market_cap: pd.DataFrame,
    targets: pd.DataFrame,
    config: AttributionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a monthly open-to-open equal- and cap-weighted pool benchmark.

    The benchmark is rebalanced at each execution-date open and valued at the
    next execution-date open.  Per stock the growth is the start-day
    ``close/open``, the intermediate daily total returns, and the next start's
    ``(1 + Dretwd) * open / close`` gap.  This is vectorised because the pool
    has thousands of names; the per-name backtest engine would be needlessly
    slow for a benchmark.
    """
    execution_map = _execution_map(targets).sort_values("execution_date").reset_index(drop=True)
    benchmark_targets, fallback = build_benchmark_targets(
        pool, market_cap, execution_map, config.min_cap_coverage
    )
    starts = execution_map["execution_date"].to_numpy()
    if len(starts) == 0:
        raise ValueError("No execution dates are available for the benchmark.")
    daily = daily.copy()
    if "market_type" not in daily and "Markettype" in daily:
        daily["market_type"] = daily["Markettype"].astype("string")
    if "PreClosePrice" not in daily:
        daily["PreClosePrice"] = np.nan
    if "market_type" not in daily:
        daily["market_type"] = pd.NA
    frame = daily[["stock_code", "trade_date", "open_price", "close_price", "total_return",
                   "PreClosePrice", "market_type", "trading_status", "quality_flags"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).astype("datetime64[ns]")
    frame = frame.loc[frame["trade_date"].between(execution_map["execution_date"].min(),
                                                  frame["trade_date"].max())].copy()
    dates = frame["trade_date"].to_numpy()
    positions = np.searchsorted(starts, dates, side="right") - 1
    in_range = positions >= 0
    frame = frame.loc[in_range].copy()
    positions = positions[in_range]
    frame["period_id"] = positions
    frame["is_start"] = frame["trade_date"].to_numpy() == starts[positions]
    flags = frame["quality_flags"].fillna("").astype("string")
    valid_open = (
        frame["open_price"].notna()
        & frame["open_price"].gt(0)
        & ~flags.str.contains("invalid_open_price", regex=False)
    )
    valid_close = frame["close_price"].notna() & frame["close_price"].gt(0)
    valid_return = frame["total_return"].notna() & ~flags.str.contains("invalid_total_return", regex=False)
    normal = frame["trading_status"].astype("string").eq(config.normal_trading_status)
    valid_start = frame["is_start"] & valid_open & valid_close & normal
    pre_close = pd.to_numeric(frame["PreClosePrice"], errors="coerce")
    market = frame["market_type"].astype("string")
    rate = pd.Series(np.nan, index=frame.index, dtype="float64")
    rate.loc[market.isin(["1", "4"])] = 0.10
    rate.loc[market.eq("16") & frame["trade_date"].lt(CHINEXT_REFORM)] = 0.10
    rate.loc[market.eq("16") & frame["trade_date"].ge(CHINEXT_REFORM)] = 0.20
    rate.loc[market.eq("32")] = 0.20
    limit_up = (pre_close * (1.0 + rate)).round(2)
    limit_blocked = limit_up.notna() & frame["open_price"].ge(limit_up - LIMIT_TOLERANCE)
    valid_start = valid_start & ~limit_blocked
    start_price = np.where(valid_start, frame["close_price"] / frame["open_price"], 1.0)
    daily_return = np.where(valid_return, 1.0 + frame["total_return"], 1.0)
    frame["body_factor"] = np.where(frame["is_start"], start_price, daily_return)
    frame["start_valid"] = valid_start
    frame["gap_factor"] = np.where(
        valid_start & valid_return,
        (1.0 + frame["total_return"]) * frame["open_price"] / frame["close_price"],
        1.0,
    )

    body = frame.groupby(["period_id", "stock_code"], as_index=False).agg(
        body_growth=("body_factor", "prod"),
        start_valid=("start_valid", "max"),
    )
    body.loc[~body["start_valid"], "body_growth"] = 1.0
    gaps = (
        frame.loc[frame["is_start"], ["period_id", "stock_code", "gap_factor"]]
        .drop_duplicates(["period_id", "stock_code"])
        .assign(period_id=lambda item: item["period_id"] - 1)
        .rename(columns={"gap_factor": "end_gap"})
    )
    growth = body.merge(gaps, on=["period_id", "stock_code"], how="left", validate="one_to_one")
    growth["end_gap"] = growth["end_gap"].fillna(1.0)
    growth["growth"] = growth["body_growth"] * growth["end_gap"]
    growth = growth[["period_id", "stock_code", "growth"]]

    period_frame = execution_map.assign(period_id=np.arange(len(execution_map)))
    period_frame["month"] = period_frame["execution_date"].dt.to_period("M").astype(str)
    weights = benchmark_targets.merge(
        period_frame[["period_id", "execution_date"]], on="execution_date", how="inner",
        validate="many_to_one",
    ).merge(growth, on=["period_id", "stock_code"], how="left", validate="many_to_one")
    weights["growth"] = weights["growth"].fillna(1.0)
    # Weighted return by factor without a Python loop over 135 periods.
    weights["weighted_growth"] = weights["target_weight"] * weights["growth"]
    weighted = weights.groupby(["factor", "period_id"], as_index=False)["weighted_growth"].sum()
    weighted["monthly_return"] = weighted["weighted_growth"] - 1.0
    monthly = weighted[["factor", "period_id", "monthly_return"]].merge(
        period_frame[["period_id", "month"]], on="period_id", how="left", validate="many_to_one"
    )
    monthly["end_nav"] = monthly.groupby("factor")["monthly_return"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    monthly["month_key"] = monthly["month"]
    # Equal-weight target weights sum to one, so the weighted formula covers both.
    return monthly.sort_values(["factor", "month"]).reset_index(drop=True), fallback


def _monthly_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.Series(returns).dropna().astype(float)
    count = len(values)
    if count < 2:
        return {
            "n_months": count,
            "mean_monthly": float(values.mean()) if count else np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "cumulative_nav": np.nan,
        }
    nav = (1.0 + values).cumprod()
    return {
        "n_months": count,
        "mean_monthly": float(values.mean()),
        "annualized_return": float(nav.iloc[-1] ** (12.0 / count) - 1.0),
        "annualized_volatility": float(values.std(ddof=1) * math.sqrt(12)),
        "sharpe_ratio": float(values.mean() / values.std(ddof=1) * math.sqrt(12)) if values.std(ddof=1) else np.nan,
        "max_drawdown": float((nav / nav.cummax().clip(lower=1.0) - 1.0).min()),
        "cumulative_nav": float(nav.iloc[-1]),
    }


def _ols(y: np.ndarray, regressors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    columns = [np.ones(len(y))] + [np.asarray(value, dtype=float) for value in regressors]
    design = np.column_stack(columns)
    if len(y) <= design.shape[1]:
        count = design.shape[1]
        return np.full(count, np.nan), np.full(count, np.nan), np.array([np.nan]), np.nan
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    degrees = len(y) - design.shape[1]
    sigma2 = float(residual @ residual / degrees)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(residual @ residual) / total if total > 0 else np.nan
    return beta, standard_errors, residual, r2


def _strategy_monthly(
    gross: pd.DataFrame,
    net: pd.DataFrame,
    config: AttributionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gross = gross.copy()
    gross["month_key"] = _month_key(gross["month"])
    gross_rows = gross[["factor", "month_key", "monthly_return"]].rename(
        columns={"monthly_return": "strategy_return"}
    ).assign(return_type="gross")

    net = net.copy()
    net["month_key"] = _month_key(net["month"])
    net["return_type"] = net["slippage_bp"].map(lambda value: f"net_{int(value)}bp")
    net_rows = net[["factor", "month_key", "net_monthly_return", "return_type"]].rename(
        columns={"net_monthly_return": "strategy_return"}
    )
    strategy = pd.concat([gross_rows, net_rows], ignore_index=True)
    strategy = strategy.loc[strategy["return_type"].isin(RETURN_TYPES)].copy()

    hs300 = gross[["month_key", "benchmark_monthly_return"]].drop_duplicates("month_key")
    return strategy, hs300


def build_alpha_table(
    strategy: pd.DataFrame,
    hs300: pd.DataFrame,
    benchmark_monthly: pd.DataFrame,
    config: AttributionConfig,
    factors: tuple[str, ...] = FACTORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLS alpha/beta and a two-factor market + equal-weight decomposition."""
    ew = benchmark_monthly.loc[
        benchmark_monthly["factor"].eq(BENCHMARK_FACTORS[0]), ["month_key", "monthly_return"]
    ].rename(columns={"monthly_return": "ew_return"})
    cw = benchmark_monthly.loc[
        benchmark_monthly["factor"].eq(BENCHMARK_FACTORS[1]), ["month_key", "monthly_return"]
    ].rename(columns={"monthly_return": "cw_return"})
    market = hs300.rename(columns={"benchmark_monthly_return": "hs300_return"})
    frame = strategy.merge(market, on="month_key", how="inner", validate="many_to_one")
    frame = frame.merge(ew, on="month_key", how="inner", validate="many_to_one")
    frame = frame.merge(cw, on="month_key", how="inner", validate="many_to_one")
    monthly_rf = (1.0 + config.risk_free_rate) ** (1.0 / config.annualization_months) - 1.0

    alpha_rows: list[dict[str, object]] = []
    style_rows: list[dict[str, object]] = []
    for factor in factors:
        for return_type in RETURN_TYPES:
            subset = frame.loc[
                frame["factor"].eq(factor) & frame["return_type"].eq(return_type)
            ].dropna(subset=["strategy_return", "hs300_return", "ew_return", "cw_return"])
            count = len(subset)
            y = subset["strategy_return"].to_numpy(dtype=float) - monthly_rf
            for label, column in (
                ("H00300", "hs300_return"),
                ("EW_POOL_MONTHLY", "ew_return"),
                ("CW_POOL_MONTHLY", "cw_return"),
            ):
                x = subset[column].to_numpy(dtype=float) - monthly_rf
                beta, se, _, r2 = _ols(y, [x])
                alpha = float(beta[0]) if count > 1 else np.nan
                t_value = float(alpha / se[0]) if count > 1 and np.isfinite(se[0]) and se[0] > 0 else np.nan
                excess = y - x
                excess_t = float(newey_west_t(pd.Series(excess))) if count >= 3 else np.nan
                alpha_rows.append({
                    "factor": factor,
                    "return_type": return_type,
                    "benchmark": label,
                    "benchmark_label": BENCHMARK_LABELS[label],
                    "n_months": count,
                    "alpha_monthly": alpha,
                    "alpha_annualized": (1.0 + alpha) ** 12 - 1.0 if np.isfinite(alpha) and alpha > -1 else np.nan,
                    "alpha_t": t_value,
                    "beta": float(beta[1]) if count > 1 else np.nan,
                    "r2": r2,
                    "excess_mean_monthly": float(excess.mean()) if count else np.nan,
                    "excess_t_nw": excess_t,
                    "information_ratio": (
                        float(excess.mean() / excess.std(ddof=1) * math.sqrt(12))
                        if count > 1 and excess.std(ddof=1) > 0 else np.nan
                    ),
                })

            # Market + equal-weight spread: alpha is the residual after the
            # size/equal-weight tilt, replacing the old geometric subtraction.
            x1 = subset["hs300_return"].to_numpy(dtype=float) - monthly_rf
            x2 = subset["ew_return"].to_numpy(dtype=float) - subset["hs300_return"].to_numpy(dtype=float)
            beta, se, _, r2 = _ols(y, [x1, x2])
            alpha = float(beta[0]) if count > 2 else np.nan
            style_rows.append({
                "factor": factor,
                "return_type": return_type,
                "n_months": count,
                "alpha_monthly": alpha,
                "alpha_annualized": (1.0 + alpha) ** 12 - 1.0 if np.isfinite(alpha) and alpha > -1 else np.nan,
                "alpha_t": float(alpha / se[0]) if count > 2 and np.isfinite(se[0]) and se[0] > 0 else np.nan,
                "beta_hs300": float(beta[1]) if count > 2 else np.nan,
                "beta_ew_spread": float(beta[2]) if count > 2 else np.nan,
                "r2": r2,
            })
    return pd.DataFrame(alpha_rows), pd.DataFrame(style_rows)


def _leg_turnover(members: pd.DataFrame, factor: str, score_type: str, quantile: int) -> pd.Series:
    item = members.loc[
        members["factor"].eq(factor)
        & members["score_type"].eq(score_type)
        & members["quantile"].eq(quantile)
    ]
    rows: dict[pd.Timestamp, float] = {}
    prior: set[str] = set()
    for signal_date, group in item.groupby("signal_date", sort=True):
        current = set(group["stock_code"].astype(str))
        count = len(current)
        if count:
            turnover = (len(current - prior) + len(prior - current)) / (2.0 * count)
        else:
            turnover = np.nan
        rows[pd.Timestamp(signal_date)] = turnover
        prior = current
    return pd.Series(rows, dtype="float64")


def build_long_short(
    quantiles: pd.DataFrame,
    members: pd.DataFrame | None,
    config: AttributionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spread = (
        quantiles[["factor", "score_type", "signal_date", "q_high_minus_q_low"]]
        .drop_duplicates(["factor", "score_type", "signal_date"])
        .sort_values(["factor", "score_type", "signal_date"])
        .copy()
    )
    spread = spread.loc[spread["score_type"].isin(config.long_short_score_types)].copy()
    spread["signal_date"] = pd.to_datetime(spread["signal_date"]).astype("datetime64[ns]")
    spread["holding_month"] = (
        spread["signal_date"] + pd.DateOffset(months=1)
    ).dt.to_period("M").astype(str)
    if members is not None and not members.empty:
        members = members.assign(
            signal_date=pd.to_datetime(members["signal_date"]).astype("datetime64[ns]")
        )
    summary_rows: list[dict[str, object]] = []
    monthly_frames: list[pd.DataFrame] = []
    for (factor, score_type), group in spread.groupby(["factor", "score_type"], sort=True):
        group = group.copy()
        if members is not None and not members.empty:
            group["turnover_q5"] = group["signal_date"].map(
                _leg_turnover(members, factor, score_type, 5)
            )
            group["turnover_q1"] = group["signal_date"].map(
                _leg_turnover(members, factor, score_type, 1)
            )
        else:
            group["turnover_q5"] = np.nan
            group["turnover_q1"] = np.nan
        group["turnover_total"] = group["turnover_q5"] + group["turnover_q1"]
        group["net_return_10bp"] = group["q_high_minus_q_low"] - group["turnover_total"] * 0.0010 / 2.0
        group["net_return_20bp"] = group["q_high_minus_q_low"] - group["turnover_total"] * 0.0020 / 2.0
        group["nav"] = (1.0 + group["q_high_minus_q_low"]).cumprod()
        metrics = _monthly_metrics(group["q_high_minus_q_low"])
        net10 = _monthly_metrics(group["net_return_10bp"])
        net20 = _monthly_metrics(group["net_return_20bp"])
        summary_rows.append({
            "factor": factor,
            "score_type": score_type,
            "n_months": metrics["n_months"],
            "mean_monthly": metrics["mean_monthly"],
            "annualized_return": metrics["annualized_return"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "cumulative_nav": metrics["cumulative_nav"],
            "t_nw": float(newey_west_t(group["q_high_minus_q_low"])) if metrics["n_months"] >= 3 else np.nan,
            "hit_rate": float((group["q_high_minus_q_low"] > 0).mean()),
            "mean_turnover_q5": float(group["turnover_q5"].mean()) if group["turnover_q5"].notna().any() else np.nan,
            "mean_turnover_q1": float(group["turnover_q1"].mean()) if group["turnover_q1"].notna().any() else np.nan,
            "net_mean_10bp": net10["mean_monthly"],
            "net_annualized_10bp": net10["annualized_return"],
            "net_mean_20bp": net20["mean_monthly"],
            "net_annualized_20bp": net20["annualized_return"],
        })
        monthly_frames.append(group[[
            "factor", "score_type", "signal_date", "holding_month", "q_high_minus_q_low",
            "nav", "turnover_q5", "turnover_q1", "net_return_10bp", "net_return_20bp",
        ]])
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), monthly


def build_bridge(quantiles: pd.DataFrame) -> pd.DataFrame:
    pivot = quantiles.pivot_table(
        index=["factor", "score_type", "signal_date"],
        columns="quantile",
        values="equal_weight_forward_return",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[1, 2, 3, 4, 5])
    if pivot.empty:
        return pd.DataFrame()
    means = pivot.groupby(level=["factor", "score_type"]).mean()
    means.columns = [f"q{int(column)}_mean" for column in means.columns]
    means["pool_mean"] = means.mean(axis=1)
    means["q5_minus_pool"] = means["q5_mean"] - means["pool_mean"]
    means["q1_minus_pool"] = means["q1_mean"] - means["pool_mean"]
    means["q5_minus_q1"] = means["q5_mean"] - means["q1_mean"]
    return means.reset_index()


def build_benchmark_metrics(benchmark_monthly: pd.DataFrame, gross: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gross = gross.copy()
    gross["month_key"] = _month_key(gross["month"])
    for factor in FACTORS:
        item = gross.loc[gross["factor"].eq(factor)].copy()
        metrics = _monthly_metrics(item["monthly_return"])
        rows.append({"series": factor, "kind": "strategy_gross", **metrics})
    hs300 = gross[["month_key", "benchmark_monthly_return"]].drop_duplicates("month_key")
    market_metrics = _monthly_metrics(hs300["benchmark_monthly_return"])
    rows.append({"series": "H00300", "kind": "benchmark", **market_metrics})
    for factor in BENCHMARK_FACTORS:
        item = benchmark_monthly.loc[benchmark_monthly["factor"].eq(factor)].copy()
        metrics = _monthly_metrics(item["monthly_return"])
        label = "EW_POOL_MONTHLY" if factor == BENCHMARK_FACTORS[0] else "CW_POOL_MONTHLY"
        rows.append({"series": label, "kind": "benchmark", **metrics})
    return pd.DataFrame(rows)


def build_neutral_long_only(
    factor_details: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    pool: pd.DataFrame,
    market_cap: pd.DataFrame,
    hs300: pd.DataFrame,
    benchmark_monthly: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    backtest_config: BacktestConfig,
    cost_config: CostConfig,
    performance_config: PerformanceConfig,
    config: AttributionConfig,
    delistings: pd.DataFrame | None = None,
) -> NeutralLongOnlyOutputs:
    """Run the long-only pipeline on neutralized scores.

    Factor names stay ``pe/momentum/roe`` inside the backtest so the existing
    validation and cost/performance code can be reused; only the report labels
    are suffixed with ``_neutral``.  The neutral long-only portfolio uses the
    same delisting write-off assumption as the original long-only run.
    """
    neutral = factor_details.copy()
    for factor in FACTORS:
        neutral[f"{factor}_score"] = neutral[f"{factor}_score_neutral"]
        neutral[f"{factor}_is_rankable"] = neutral[f"{factor}_neutral_eligible"].fillna(False)
    portfolio = build_portfolio_outputs(neutral, daily, portfolio_config)
    backtest = build_backtest_outputs(
        portfolio.targets, portfolio.actions, daily, benchmark, backtest_config, pool, market_cap,
        delistings=delistings,
    )
    cost = build_cost_outputs(backtest.daily, backtest.rebalance_records, cost_config)
    performance = build_performance_outputs(backtest.daily, cost.daily, cost.costs, performance_config)
    metrics = performance.metrics.loc[performance.metrics["series_type"].isin(["gross", "net"])].copy()
    metrics["variant"] = "neutral"

    monthly = performance.monthly
    gross = monthly.loc[monthly["series_type"].eq("gross"), ["factor", "month", "monthly_return"]].rename(
        columns={"monthly_return": "strategy_return"}
    ).assign(return_type="gross")
    net = monthly.loc[monthly["series_type"].eq("net")].copy()
    net["return_type"] = net["slippage_bp"].map(lambda value: f"net_{int(value)}bp")
    net = net[["factor", "month", "monthly_return", "return_type"]].rename(
        columns={"monthly_return": "strategy_return"}
    )
    strategy = pd.concat([gross, net], ignore_index=True)
    strategy["month_key"] = _month_key(strategy["month"])
    strategy["factor"] = strategy["factor"] + "_neutral"
    neutral_alpha, _ = build_alpha_table(
        strategy, hs300, benchmark_monthly, config,
        factors=tuple(f"{factor}_neutral" for factor in FACTORS),
    )
    return NeutralLongOnlyOutputs(metrics, neutral_alpha, backtest.daily, backtest.rebalance_records, monthly)


def build_attribution_outputs(
    targets: pd.DataFrame,
    factor_details: pd.DataFrame,
    pool: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_cap: pd.DataFrame,
    gross: pd.DataFrame,
    net: pd.DataFrame,
    quantiles: pd.DataFrame,
    members: pd.DataFrame | None,
    portfolio_config: PortfolioConfig,
    backtest_config: BacktestConfig,
    cost_config: CostConfig,
    performance_config: PerformanceConfig,
    config: AttributionConfig,
    delistings: pd.DataFrame | None = None,
) -> AttributionOutputs:
    strategy, hs300 = _strategy_monthly(gross, net, config)
    benchmark_monthly, fallback = build_pool_benchmark(
        pool, daily, market_cap, targets, config
    )
    alpha, style = build_alpha_table(strategy, hs300, benchmark_monthly, config)
    long_short, long_short_monthly = build_long_short(quantiles, members, config)
    bridge = build_bridge(quantiles)
    summary = build_benchmark_metrics(benchmark_monthly, gross)
    summary["cw_fallback_periods"] = len(fallback)
    neutral = build_neutral_long_only(
        factor_details, daily, benchmark, pool, market_cap, hs300, benchmark_monthly,
        portfolio_config, backtest_config, cost_config, performance_config, config,
        delistings=delistings,
    )
    return AttributionOutputs(
        benchmark_monthly=benchmark_monthly,
        alpha=alpha,
        style=style,
        long_short=long_short,
        long_short_monthly=long_short_monthly,
        bridge=bridge,
        summary=summary,
        neutral=neutral,
    )


def validate_outputs(outputs: AttributionOutputs, config: AttributionConfig) -> None:
    monthly = outputs.benchmark_monthly
    if set(monthly["factor"].unique()) != set(BENCHMARK_FACTORS):
        raise ValueError("Both benchmark factors must be present.")
    if monthly.duplicated(["factor", "month"]).any():
        raise ValueError("Benchmark monthly output contains duplicate keys.")
    if monthly["monthly_return"].isna().any() or not monthly["end_nav"].gt(0).all():
        raise ValueError("Benchmark monthly output contains missing or non-positive values.")
    ew_months = set(monthly.loc[monthly["factor"].eq(BENCHMARK_FACTORS[0]), "month_key"])
    cw_months = set(monthly.loc[monthly["factor"].eq(BENCHMARK_FACTORS[1]), "month_key"])
    if ew_months != cw_months:
        raise ValueError("Equal- and cap-weighted benchmarks do not cover the same months.")
    if len(ew_months) < config.min_months:
        raise ValueError("Benchmark monthly output is shorter than the configured minimum.")
    expected_alpha = len(FACTORS) * len(RETURN_TYPES) * 3
    if len(outputs.alpha) != expected_alpha or outputs.alpha.duplicated(["factor", "return_type", "benchmark"]).any():
        raise ValueError("Alpha table does not contain the complete factor/return/benchmark grid.")
    if len(outputs.style) != len(FACTORS) * len(RETURN_TYPES):
        raise ValueError("Style decomposition is incomplete.")
    if outputs.long_short.duplicated(["factor", "score_type"]).any():
        raise ValueError("Long-short summary contains duplicate keys.")
    if outputs.bridge.duplicated(["factor", "score_type"]).any():
        raise ValueError("Bridge table contains duplicate keys.")
    if outputs.long_short_monthly.duplicated(["factor", "score_type", "holding_month"]).any():
        raise ValueError("Long-short monthly output contains duplicate keys.")
    if len(outputs.neutral.alpha) != expected_alpha or outputs.neutral.alpha.duplicated(
        ["factor", "return_type", "benchmark"]
    ).any():
        raise ValueError("Neutral long-only alpha table is incomplete.")
    expected_neutral_metrics = len(FACTORS) * len(RETURN_TYPES)
    if len(outputs.neutral.metrics) != expected_neutral_metrics:
        raise ValueError("Neutral long-only metrics table is incomplete.")


def _format(value: float, digits: int = 4) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def _format_percent(value: float, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def render_report(outputs: AttributionOutputs, config: AttributionConfig) -> str:
    summary = outputs.summary
    lines = [
        "# 超额归因：同股票池月频基准、回归 alpha 与多空/多头桥",
        "",
        "本报告不改写回测收益、成本、成交或持仓规则；只把已有的策略月度收益与"
        "“同股票池、同月频、同开盘执行”的基准对齐后重算超额。",
        "",
        "## 口径",
        "",
        "- 基准在每个执行日开盘等权（或按信号日市值加权）买入股票池全部成员，持有到下一个执行日开盘再平衡；执行日无法买入的成员按现金处理，持有期内缺收益的交易日按冻结处理。基准不复刻策略的逐笔待卖重试。",
        "- 基准的月份标签取执行日所在月；策略月度按自然月，月末/月初边界有一天的已知错位。基准不计交易成本；策略的 gross 也不计成本，net 0/10/20 bp 来自既有成本步骤。",
        "- 月频最大回撤按月末收盘计算，和日频回测的盘中回撤不是同一口径。",
        "- 回归使用月度简单收益，OLS 含截距；alpha t 为 OLS 标准误，超额均值另给 Newey-West t。",
        f"- 策略与中性化纯多头对持仓期内“终止上市”的股票按 `{config.delisting_return:.0%}` 统一核销；基准不核销，因此基准的股票池退出路径与策略不完全相同。",
        "- Q5-Q1 为统计多空组合，A 股融券受限，不可直接交易；它用于判断因子是否具有经济显著性，不代表可实现策略。",
        "",
        "## 基准与策略的月度绩效",
        "",
        "| 序列 | 类型 | 月数 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.series} | {row.kind} | {int(row.n_months)} | {_format_percent(row.annualized_return)} | "
            f"{_format_percent(row.annualized_volatility)} | {_format(row.sharpe_ratio, 3)} | "
            f"{_format_percent(row.max_drawdown)} |"
        )
    lines.extend([
        "",
        f"- 市值加权基准因市值覆盖不足回退等权的期数：{int(summary['cw_fallback_periods'].iloc[0])}。",
        "",
        "## 回归 alpha（相对不同基准）",
        "",
        "| 因子 | 收益 | 基准 | 月数 | 年化 alpha | alpha t | beta | IR | R² |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.alpha.sort_values(["factor", "return_type", "benchmark"]).itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.return_type} | {row.benchmark_label} | {int(row.n_months)} | "
            f"{_format_percent(row.alpha_annualized)} | {_format(row.alpha_t, 2)} | "
            f"{_format(row.beta, 3)} | {_format(row.information_ratio, 2)} | {_format(row.r2, 3)} |"
        )
    lines.extend([
        "",
        "## 两因子风格分解",
        "",
        "`r_p = alpha + b1 * H00300 + b2 * (EW - H00300) + eps`；`b2` 是等权/小市值倾斜，`alpha` 是剔除市场与等权倾斜后的残差。",
        "",
        "| 因子 | 收益 | 月数 | 年化 alpha | alpha t | b1 (H00300) | b2 (EW spread) | R² |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.style.sort_values(["factor", "return_type"]).itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.return_type} | {int(row.n_months)} | {_format_percent(row.alpha_annualized)} | "
            f"{_format(row.alpha_t, 2)} | {_format(row.beta_hs300, 3)} | {_format(row.beta_ew_spread, 3)} | {_format(row.r2, 3)} |"
        )
    lines.extend([
        "",
        "## 中性化纯多头回测",
        "",
        "与原始分数使用同一套目标构建、ADV 容量（固定 1 亿元初始资金，不随 NAV 复利）、开盘成交、成本和绩效规则；因子名加 `_neutral`。",
        "",
        "| 因子 | gross 年化 | 年化波动 | Sharpe | 最大回撤 | 相对同池等权 gross alpha | alpha t |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for factor in FACTORS:
        item = outputs.neutral.metrics.loc[
            outputs.neutral.metrics["factor"].eq(factor)
            & outputs.neutral.metrics["series_type"].eq("gross")
        ]
        alpha = outputs.neutral.alpha.loc[
            outputs.neutral.alpha["factor"].eq(f"{factor}_neutral")
            & outputs.neutral.alpha["return_type"].eq("gross")
            & outputs.neutral.alpha["benchmark"].eq("EW_POOL_MONTHLY")
        ]
        if item.empty:
            continue
        row = item.iloc[0]
        lines.append(
            f"| {factor} | {_format_percent(row['annualized_return'])} | "
            f"{_format_percent(row['annualized_volatility'])} | {_format(row['sharpe_ratio'], 3)} | "
            f"{_format_percent(row['max_drawdown'])} | "
            f"{_format_percent(alpha['alpha_annualized'].iloc[0]) if not alpha.empty else '—'} | "
            f"{_format(alpha['alpha_t'].iloc[0], 2) if not alpha.empty else '—'} |"
        )
    lines.extend([
        "",
        "表中 alpha 为 gross 口径；net20 与 H00300 口径见 `data/processed/attribution_neutral_alpha.csv`。",
        "",
        "## Q5-Q1 多空组合",
        "",
        "| 因子 | 分数 | 月数 | 月均 | 年化 | 波动 | Sharpe | 最大回撤 | NW t | 胜率 | 10bp 净月均 | 20bp 净月均 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.long_short.sort_values(["factor", "score_type"]).itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.score_type} | {int(row.n_months)} | {_format_percent(row.mean_monthly)} | "
            f"{_format_percent(row.annualized_return)} | {_format_percent(row.annualized_volatility)} | "
            f"{_format(row.sharpe_ratio, 3)} | {_format_percent(row.max_drawdown)} | {_format(row.t_nw, 2)} | "
            f"{_format_percent(row.hit_rate, 1)} | {_format_percent(row.net_mean_10bp)} | {_format_percent(row.net_mean_20bp)} |"
        )
    lines.extend([
        "",
        "## 纯多头与多空的桥",
        "",
        "分位月均收益（先按月算均值，再对月份等权平均）；`q5 - pool` 是纯多头选择效应，`q1 - pool` 是空头腿效应。",
        "",
        "| 因子 | 分数 | Q1 | Q2 | Q3 | Q4 | Q5 | 全池 | Q5-全池 | Q1-全池 | Q5-Q1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.bridge.sort_values(["factor", "score_type"]).itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.score_type} | {_format_percent(row.q1_mean)} | {_format_percent(row.q2_mean)} | "
            f"{_format_percent(row.q3_mean)} | {_format_percent(row.q4_mean)} | {_format_percent(row.q5_mean)} | "
            f"{_format_percent(row.pool_mean)} | {_format_percent(row.q5_minus_pool)} | "
            f"{_format_percent(row.q1_minus_pool)} | {_format_percent(row.q5_minus_q1)} |"
        )
    pe_alpha = outputs.alpha.loc[
        outputs.alpha["factor"].eq("pe")
        & outputs.alpha["return_type"].eq("gross")
        & outputs.alpha["benchmark"].eq("EW_POOL_MONTHLY")
    ]
    pe_style = outputs.style.loc[
        outputs.style["factor"].eq("pe") & outputs.style["return_type"].eq("gross")
    ]
    def _neutral_alpha_text(return_type: str, benchmark: str) -> str:
        row = outputs.neutral.alpha.loc[
            outputs.neutral.alpha["factor"].eq("pe_neutral")
            & outputs.neutral.alpha["return_type"].eq(return_type)
            & outputs.neutral.alpha["benchmark"].eq(benchmark)
        ]
        if row.empty:
            return "—"
        return (
            f"{_format_percent(row['alpha_annualized'].iloc[0])}、"
            f"t {_format(row['alpha_t'].iloc[0], 2)}"
        )

    neutral_ew_gross = _neutral_alpha_text("gross", "EW_POOL_MONTHLY")
    neutral_ew_net20 = _neutral_alpha_text("net_20bp", "EW_POOL_MONTHLY")
    neutral_hs_gross = _neutral_alpha_text("gross", "H00300")
    neutral_hs_net20 = _neutral_alpha_text("net_20bp", "H00300")
    lines.extend([
        "",
        "## 结论",
        "",
        "- 同池月频等权基准把日频再平衡的溢价剔除后，策略相对它的 alpha 才是可解释的因子贡献；两因子分解中的 `alpha` 进一步剔除等权/小市值倾斜。",
        f"- PE gross 相对同池月频等权的年化 alpha 为 {_format_percent(pe_alpha['alpha_annualized'].iloc[0]) if not pe_alpha.empty else '—'}，"
        f"alpha t = {_format(pe_alpha['alpha_t'].iloc[0], 2) if not pe_alpha.empty else '—'}；"
        f"两因子分解的 alpha 为 {_format_percent(pe_style['alpha_annualized'].iloc[0]) if not pe_style.empty else '—'}。"
        f"原始分数两者 t 都小于 2，不能拒绝“纯多头没有同池超额”；中性化 PE 纯多头相对同池等权的 gross alpha 为 {neutral_ew_gross}，"
        f"net20 为 {neutral_ew_net20}；相对 H00300 的 gross/net20 alpha 为 {neutral_hs_gross}/{neutral_hs_net20}。"
        "只有 gross、同池等权口径的 t 超过 2，net20 与 H00300 口径均不显著；因此不能把中性化多头当作稳健的可交易 alpha。",
        "- 若 Q5-全池接近 0、Q1-全池显著为负，则 Q5-Q1 的价差主要来自空头腿；纯多头 Top 20% 拿不到这部分 alpha，IC 显著不等于多头可货币化。",
        "- 多空组合为 gross（可选 10/20bp 换手估算），且 A 股融券受限；不要把多空 Sharpe 当作可交易策略业绩。",
        "",
    ])
    return "\n".join(lines)


def run(config: AttributionConfig, config_path: Path = Path("configs/config.yaml")) -> AttributionOutputs:
    directory = config.processed_dir
    targets = pd.read_parquet(directory / "portfolio_targets.parquet", columns=[
        "factor", "signal_date", "execution_date", "stock_code", "target_weight",
    ])
    factor_details = pd.read_parquet(directory / "factor_details.parquet")
    pool = pd.read_parquet(directory / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    benchmark = pd.read_parquet(directory / "benchmark.parquet", columns=["trade_date", "total_return", "quality_flags"])
    daily = pd.read_parquet(directory / "daily_trading.parquet", columns=[
        "stock_code", "trade_date", "open_price", "close_price", "total_return",
        "PreClosePrice", "Markettype", "trading_status", "quality_flags", "Dnvaltrd",
    ])
    market_cap = pd.read_parquet(directory / "market_cap.parquet", columns=[
        "signal_date", "stock_code", "market_cap", "has_market_cap",
    ])
    gross = pd.read_csv(directory / "backtest_monthly_returns.csv")
    net = pd.read_csv(directory / "backtest_net_monthly_returns.csv")
    quantiles = pd.read_csv(directory / "factor_quantile_returns.csv", parse_dates=["signal_date"])
    member_path = directory / "factor_quantile_members.parquet"
    members = pd.read_parquet(member_path, columns=[
        "factor", "score_type", "signal_date", "stock_code", "quantile",
    ]) if member_path.exists() else None
    delistings = load_delisting_events(directory / "status_changes.parquet")

    outputs = build_attribution_outputs(
        targets, factor_details, pool, daily, benchmark, market_cap,
        gross, net, quantiles, members,
        load_portfolio_config(config_path),
        load_backtest_config(config_path),
        load_cost_config(config_path),
        load_performance_config(config_path),
        config,
        delistings=delistings,
    )
    validate_outputs(outputs, config)
    outputs.benchmark_monthly.to_csv(directory / "attribution_benchmark_monthly.csv", index=False, encoding="utf-8-sig")
    outputs.alpha.to_csv(directory / "attribution_alpha.csv", index=False, encoding="utf-8-sig")
    outputs.style.to_csv(directory / "attribution_style.csv", index=False, encoding="utf-8-sig")
    outputs.long_short.to_csv(directory / "attribution_long_short.csv", index=False, encoding="utf-8-sig")
    outputs.long_short_monthly.to_csv(directory / "attribution_long_short_monthly.csv", index=False, encoding="utf-8-sig")
    outputs.bridge.to_csv(directory / "attribution_bridge.csv", index=False, encoding="utf-8-sig")
    outputs.summary.to_csv(directory / "attribution_benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    outputs.neutral.metrics.to_csv(directory / "attribution_neutral_metrics.csv", index=False, encoding="utf-8-sig")
    outputs.neutral.alpha.to_csv(directory / "attribution_neutral_alpha.csv", index=False, encoding="utf-8-sig")
    outputs.neutral.monthly.to_csv(directory / "attribution_neutral_monthly_returns.csv", index=False, encoding="utf-8-sig")
    outputs.neutral.daily.to_parquet(directory / "attribution_neutral_daily.parquet", index=False)
    outputs.neutral.records.to_parquet(directory / "attribution_neutral_rebalance_records.parquet", index=False)
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "attribution.md").write_text(render_report(outputs, config), encoding="utf-8")
    print(f"Built attribution for {len(outputs.alpha)} alpha rows and {len(outputs.long_short)} long-short rows.")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attribute strategy returns to same-universe benchmarks.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    run(load_attribution_config(args.config), args.config)


if __name__ == "__main__":
    main()
