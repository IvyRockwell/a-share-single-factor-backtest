"""Validate the factor results with time splits and multiple-testing corrections.

Several different questions are answered here, and the report keeps them apart:

* **Sub-period portfolios (8A)** repeat the long-only excess statistics inside
  the four regime windows that the factor test already uses.  The windows were
  drawn *after* looking at the sample, so they describe how the result varies by
  regime; they are not out-of-sample evidence.
* **A fixed holdout (8B)** splits the sample into a development period and a
  2020-2025 test period and selects exactly one factor on the development IC t.
  This is out-of-sample evidence for *factor selection*, not for the full
  research question: the three candidate factors were already chosen from the
  literature and from knowledge of the whole sample.
* **An annual walk-forward (8C)** re-selects the best factor every year on the
  information available before that year and holds it for the next year.  It is
  out-of-sample evidence for *factor rotation*, but with only three candidates
  and no parameter optimisation.
* **Multiple testing (8D)** corrects the raw t statistics within separate
  families (Holm, Benjamini-Hochberg, and a White reality check using a
  stationary bootstrap on the max mean).  The reality check is a max-mean
  statistic, not a max-Sharpe one, and it does not replace the t tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.attribution import BENCHMARK_FACTORS, _monthly_metrics, _ols
from src.factor_test import newey_west_t, parse_subperiods

FACTORS = ("pe", "momentum", "roe")
MOMENTUM_VARIANTS = ("momentum", "momentum_6_1", "momentum_1m")
NEUTRAL_FACTORS = ("pe_neutral", "momentum_neutral", "roe_neutral")
RETURN_TYPES = ("gross", "net20")
FACTOR_ORDER = {factor: index for index, factor in enumerate(FACTORS)}
DEFAULT_SLIPPAGE_BP = 20
DEFAULT_DEV_START = pd.Timestamp("2014-10-01")
DEFAULT_HOLDOUT_START = pd.Timestamp("2020-01-01")
DEFAULT_WALK_FORWARD_START_YEAR = 2018
DEFAULT_MIN_TRAIN_MONTHS = 24
DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_BOOTSTRAP_BLOCK = 6
DEFAULT_SEED = 20260701
MIN_SERIES_MONTHS = 3
SUBPERIOD_COLUMNS = (
    "factor", "subperiod", "return_type", "start", "end", "n_months",
    "strategy_annualized", "ew_annualized", "excess_ew_annualized",
    "alpha_annualized", "alpha_t", "h00300_annualized", "excess_h00300_annualized",
    "sharpe", "max_drawdown", "excess_mean_monthly", "excess_t_nw",
)
HOLDOUT_COLUMNS = (
    "factor", "sample", "start", "end", "n_ic_months", "ic_mean", "ic_t_nw", "n_months",
    "gross_annualized", "net20_annualized", "alpha_gross_annualized", "alpha_gross_t",
    "alpha_net20_annualized", "alpha_net20_t", "selected",
)
WALK_FORWARD_COLUMNS = (
    "year", "selected_factor", "train_months", "train_ic_mean", "oos_return", "ew_return",
    "excess", "h00300_return", "excess_h00300", "oos_return_net20", "excess_net20",
    "chained_nav", "chained_cumulative_return", "chained_ew_nav", "chained_ew_cumulative_return",
    "annualized_return", "sharpe", "max_drawdown",
)
MULTIPLE_TESTING_COLUMNS = (
    "family", "strategy", "metric", "n", "estimate", "t", "p_raw", "p_holm", "q_bh", "p_rc",
)
FAMILY_ORDER = (
    "main_long_only",
    "momentum_variants_ic",
    "momentum_variants_long_short",
    "neutral_long_only",
    "pe_specs_ic",
    "pe_specs_long_short",
)


@dataclass(frozen=True)
class ValidationConfig:
    """The time split, the bootstrap, and the multiple-testing settings."""

    processed_dir: Path
    subperiods: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...] = ()
    dev_start: pd.Timestamp = DEFAULT_DEV_START
    holdout_start: pd.Timestamp = DEFAULT_HOLDOUT_START
    walk_forward_start_year: int = DEFAULT_WALK_FORWARD_START_YEAR
    min_train_months: int = DEFAULT_MIN_TRAIN_MONTHS
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
    bootstrap_block: int = DEFAULT_BOOTSTRAP_BLOCK
    seed: int = DEFAULT_SEED
    slippage_bp: int = DEFAULT_SLIPPAGE_BP


@dataclass(frozen=True)
class ValidationOutputs:
    subperiods: pd.DataFrame
    holdout: pd.DataFrame
    walk_forward: pd.DataFrame
    multiple_testing: pd.DataFrame


def load_validation_config(config_path: Path) -> ValidationConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    section = settings.get("validation") or {}
    dev_start = pd.Timestamp(section.get("dev_start", DEFAULT_DEV_START))
    holdout_start = pd.Timestamp(section.get("holdout_start", DEFAULT_HOLDOUT_START))
    if dev_start >= holdout_start:
        raise ValueError("validation.dev_start must be strictly before holdout_start.")
    reps = int(section.get("bootstrap_reps", DEFAULT_BOOTSTRAP_REPS))
    block = int(section.get("bootstrap_block", DEFAULT_BOOTSTRAP_BLOCK))
    if reps <= 0:
        raise ValueError("validation.bootstrap_reps must be positive.")
    if block <= 0:
        raise ValueError("validation.bootstrap_block must be positive.")
    return ValidationConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        subperiods=parse_subperiods((settings.get("factor_test") or {}).get("subperiods")),
        dev_start=dev_start,
        holdout_start=holdout_start,
        walk_forward_start_year=int(section.get("walk_forward_start_year", DEFAULT_WALK_FORWARD_START_YEAR)),
        min_train_months=int(section.get("min_train_months", DEFAULT_MIN_TRAIN_MONTHS)),
        bootstrap_reps=reps,
        bootstrap_block=block,
        seed=int(section.get("seed", DEFAULT_SEED)),
        slippage_bp=int(section.get("slippage_bp", DEFAULT_SLIPPAGE_BP)),
    )


def assemble_monthly(gross: pd.DataFrame, net: pd.DataFrame, benchmark: pd.DataFrame,
                     slippage_bp: int = DEFAULT_SLIPPAGE_BP) -> pd.DataFrame:
    """Line up strategy gross/net, the same-pool equal-weight benchmark, and H00300.

    ``month`` is the natural calendar month used by the strategy series and
    ``month_key`` on the attribution benchmark; they are the same ``YYYY-MM``
    string, so the join is exact.
    """
    gross = gross[["factor", "month", "monthly_return", "benchmark_monthly_return"]].copy()
    gross["month"] = gross["month"].astype(str)
    gross = gross.rename(columns={
        "monthly_return": "gross_return", "benchmark_monthly_return": "h00300_return",
    })
    net = net.loc[net["slippage_bp"].eq(slippage_bp), ["factor", "month", "net_monthly_return"]].copy()
    net["month"] = net["month"].astype(str)
    net = net.rename(columns={"net_monthly_return": "net20_return"})
    benchmark = benchmark.copy()
    benchmark["month_key"] = benchmark["month_key"].astype(str)
    ew = benchmark.loc[benchmark["factor"].eq(BENCHMARK_FACTORS[0]), ["month_key", "monthly_return"]]
    cw = benchmark.loc[benchmark["factor"].eq(BENCHMARK_FACTORS[1]), ["month_key", "monthly_return"]]
    ew = ew.rename(columns={"month_key": "month", "monthly_return": "ew_return"})
    cw = cw.rename(columns={"month_key": "month", "monthly_return": "cw_return"})
    frame = gross.merge(net, on=["factor", "month"], how="left", validate="one_to_one")
    frame = frame.merge(ew, on="month", how="left", validate="many_to_one")
    frame = frame.merge(cw, on="month", how="left", validate="many_to_one")
    if frame.duplicated(["factor", "month"]).any():
        raise ValueError("The monthly strategy frame contains duplicate factor/month keys.")
    if frame[["gross_return", "ew_return"]].isna().any().any():
        raise ValueError("The monthly strategy frame is missing a strategy or equal-weight return.")
    frame["month_start"] = pd.to_datetime(frame["month"] + "-01").astype("datetime64[ns]")
    return frame.sort_values(["factor", "month"]).reset_index(drop=True)


def _monthly_stats(values: pd.Series) -> dict[str, float]:
    metrics = _monthly_metrics(pd.Series(values).astype("float64"))
    return {
        "n_months": int(metrics["n_months"]),
        "annualized": metrics["annualized_return"],
        "sharpe": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
    }


def _alpha_vs_benchmark(strategy: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Monthly OLS alpha of the strategy on the benchmark, annualised."""
    frame = pd.DataFrame({"strategy": strategy, "benchmark": benchmark}).dropna()
    if len(frame) < 3:
        return np.nan, np.nan
    beta, standard_errors, _, _ = _ols(
        frame["strategy"].to_numpy(dtype=float), [frame["benchmark"].to_numpy(dtype=float)]
    )
    alpha = float(beta[0])
    error = float(standard_errors[0])
    annualized = (1.0 + alpha) ** 12 - 1.0 if np.isfinite(alpha) and alpha > -1 else np.nan
    t_value = alpha / error if np.isfinite(error) and error > 0 else np.nan
    return annualized, t_value


def build_subperiod_portfolio(monthly: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    """Repeat the long-only excess statistics inside each configured regime window."""
    if not config.subperiods:
        return pd.DataFrame(columns=list(SUBPERIOD_COLUMNS))
    rows: list[dict[str, object]] = []
    for factor in FACTORS:
        item = monthly.loc[monthly["factor"].eq(factor)].sort_values("month_start")
        for name, start, end in config.subperiods:
            window = item.loc[item["month_start"].between(start, end)]
            for return_type in RETURN_TYPES:
                column = f"{return_type}_return"
                sample = window.loc[window[column].notna()]
                stats = _monthly_stats(sample[column])
                ew_stats = _monthly_stats(sample["ew_return"])
                market_stats = _monthly_stats(sample["h00300_return"])
                alpha, alpha_t = _alpha_vs_benchmark(sample[column], sample["ew_return"])
                excess = sample[column].astype("float64") - sample["ew_return"].astype("float64")
                rows.append({
                    "factor": factor,
                    "subperiod": name,
                    "return_type": return_type,
                    "start": start,
                    "end": end,
                    "n_months": stats["n_months"],
                    "strategy_annualized": stats["annualized"],
                    "ew_annualized": ew_stats["annualized"],
                    "excess_ew_annualized": (
                        stats["annualized"] - ew_stats["annualized"]
                        if pd.notna(stats["annualized"]) and pd.notna(ew_stats["annualized"]) else np.nan
                    ),
                    "alpha_annualized": alpha,
                    "alpha_t": alpha_t,
                    "h00300_annualized": market_stats["annualized"],
                    "excess_h00300_annualized": (
                        stats["annualized"] - market_stats["annualized"]
                        if pd.notna(stats["annualized"]) and pd.notna(market_stats["annualized"]) else np.nan
                    ),
                    "sharpe": stats["sharpe"],
                    "max_drawdown": stats["max_drawdown"],
                    "excess_mean_monthly": float(excess.mean()) if len(excess) else np.nan,
                    "excess_t_nw": newey_west_t(excess) if len(excess) >= MIN_SERIES_MONTHS else np.nan,
                })
    frame = pd.DataFrame(rows, columns=list(SUBPERIOD_COLUMNS))
    period_order = {name: index for index, (name, _, _) in enumerate(config.subperiods)}
    frame["_period_order"] = frame["subperiod"].map(period_order)
    return (
        frame.sort_values(["factor", "_period_order", "return_type"])
        .drop(columns="_period_order")
        .reset_index(drop=True)
    )


def _split_mask(monthly: pd.DataFrame, config: ValidationConfig) -> dict[str, pd.Series]:
    return {
        "dev": monthly["month_start"].lt(config.holdout_start),
        "test": monthly["month_start"].ge(config.holdout_start),
    }


def _ic_stats(ic: pd.Series) -> tuple[int, float, float]:
    clean = ic.dropna()
    if clean.empty:
        return 0, np.nan, np.nan
    return len(clean), float(clean.mean()), newey_west_t(clean)


def _ic_window(raw_ic: pd.DataFrame, factor: str, start: pd.Timestamp,
               end: pd.Timestamp | None) -> pd.Series:
    item = raw_ic.loc[raw_ic["factor"].eq(factor)]
    mask = item["signal_date"].ge(start)
    if end is not None:
        mask = mask & item["signal_date"].lt(end)
    return item.loc[mask, "rank_ic"]


def _primary_ic(ic_series: pd.DataFrame) -> pd.DataFrame:
    """Keep only the three primary factors' raw Rank IC rows."""
    raw_ic = ic_series.loc[
        ic_series["score_type"].eq("raw") & ic_series["factor"].isin(FACTORS)
    ].copy()
    raw_ic["signal_date"] = pd.to_datetime(raw_ic["signal_date"]).astype("datetime64[ns]")
    return raw_ic


def build_holdout(monthly: pd.DataFrame, ic_series: pd.DataFrame,
                  config: ValidationConfig) -> pd.DataFrame:
    """Development/test split, selecting one factor on the development Rank IC t."""
    raw_ic = _primary_ic(ic_series)
    masks = _split_mask(monthly, config)
    rows: list[dict[str, object]] = []
    for factor in FACTORS:
        item = monthly.loc[monthly["factor"].eq(factor)]
        for sample, mask in (("dev", masks["dev"]), ("test", masks["test"])):
            window = item.loc[mask.loc[item.index]]
            if sample == "dev":
                ic_series_window = _ic_window(raw_ic, factor, config.dev_start, config.holdout_start)
            else:
                ic_series_window = _ic_window(raw_ic, factor, config.holdout_start, None)
            n_ic, ic_mean, ic_t = _ic_stats(ic_series_window)
            gross = _monthly_stats(window["gross_return"])
            net20 = _monthly_stats(window["net20_return"])
            alpha_gross, alpha_gross_t = _alpha_vs_benchmark(window["gross_return"], window["ew_return"])
            alpha_net20, alpha_net20_t = _alpha_vs_benchmark(window["net20_return"], window["ew_return"])
            rows.append({
                "factor": factor,
                "sample": sample,
                "start": config.dev_start if sample == "dev" else config.holdout_start,
                "end": (config.holdout_start - pd.Timedelta(days=1)) if sample == "dev"
                else window["month_start"].max() + pd.offsets.MonthEnd(0),
                "n_ic_months": n_ic,
                "ic_mean": ic_mean,
                "ic_t_nw": ic_t,
                "n_months": gross["n_months"],
                "gross_annualized": gross["annualized"],
                "net20_annualized": net20["annualized"],
                "alpha_gross_annualized": alpha_gross,
                "alpha_gross_t": alpha_gross_t,
                "alpha_net20_annualized": alpha_net20,
                "alpha_net20_t": alpha_net20_t,
                "selected": False,
            })
    frame = pd.DataFrame(rows, columns=list(HOLDOUT_COLUMNS))
    # The only selection rule: the highest development Rank-IC t, ties broken by
    # the configured factor order and never by any test-period statistic.
    dev = frame.loc[frame["sample"].eq("dev") & frame["ic_t_nw"].notna()]
    selected = (
        dev.assign(_order=dev["factor"].map(FACTOR_ORDER))
        .sort_values(["ic_t_nw", "_order"], ascending=[False, True])
        .head(1)["factor"]
        .iloc[0]
        if not dev.empty else None
    )
    if selected is not None:
        frame["selected"] = frame["factor"].eq(selected)
    return frame


def _yearly_returns(item: pd.DataFrame, column: str, year: int) -> pd.Series:
    window = item.loc[item["month_start"].dt.year.eq(year)]
    return window.set_index("month")[column].astype("float64").dropna()


def build_walk_forward(monthly: pd.DataFrame, ic_series: pd.DataFrame,
                       config: ValidationConfig) -> pd.DataFrame:
    """Re-select the factor every year using only earlier IC and hold it for one year."""
    raw_ic = _primary_ic(ic_series)
    years = range(config.walk_forward_start_year, int(monthly["month_start"].dt.year.max()) + 1)
    rows: list[dict[str, object]] = []
    chained: list[pd.Series] = []
    chained_ew: list[pd.Series] = []
    for year in years:
        cutoff = pd.Timestamp(year=int(year), month=1, day=1)
        train = raw_ic.loc[raw_ic["signal_date"].lt(cutoff)]
        means = train.groupby("factor")["rank_ic"].agg(["mean", "count"])
        eligible = means.loc[means["count"].ge(config.min_train_months)]
        if eligible.empty:
            continue
        selected = sorted(
            eligible.index, key=lambda factor: (-eligible.loc[factor, "mean"], FACTOR_ORDER[factor])
        )[0]
        train_months = int(eligible.loc[selected, "count"])
        train_mean = float(eligible.loc[selected, "mean"])
        item = monthly.loc[monthly["factor"].eq(selected)]
        gross = _yearly_returns(item, "gross_return", year)
        net20 = _yearly_returns(item, "net20_return", year)
        ew = _yearly_returns(item, "ew_return", year)
        market = _yearly_returns(item, "h00300_return", year)
        if gross.empty:
            continue
        oos = float((1.0 + gross).prod() - 1.0)
        ew_return = float((1.0 + ew.reindex(gross.index)).prod() - 1.0)
        market_return = float((1.0 + market.reindex(gross.index)).prod() - 1.0)
        net_return = float((1.0 + net20.reindex(gross.index).fillna(0.0)).prod() - 1.0)
        chained.append(gross)
        chained_ew.append(ew.reindex(gross.index).fillna(0.0))
        cumulative = float((1.0 + pd.concat(chained)).prod() - 1.0)
        ew_cumulative = float((1.0 + pd.concat(chained_ew)).prod() - 1.0)
        year_stats = _monthly_stats(gross)
        rows.append({
            "year": year,
            "selected_factor": selected,
            "train_months": train_months,
            "train_ic_mean": train_mean,
            "oos_return": oos,
            "ew_return": ew_return,
            "excess": oos - ew_return,
            "h00300_return": market_return,
            "excess_h00300": oos - market_return,
            "oos_return_net20": net_return,
            "excess_net20": net_return - ew_return,
            "chained_nav": 1.0 + cumulative,
            "chained_cumulative_return": cumulative,
            "chained_ew_nav": 1.0 + ew_cumulative,
            "chained_ew_cumulative_return": ew_cumulative,
            "annualized_return": year_stats["annualized"],
            "sharpe": year_stats["sharpe"],
            "max_drawdown": year_stats["max_drawdown"],
        })
    frame = pd.DataFrame(rows, columns=list(WALK_FORWARD_COLUMNS))
    if not chained:
        return frame
    series = pd.concat(chained).sort_index()
    ew_series = pd.concat(chained_ew).sort_index()
    chain_stats = _monthly_stats(series)
    cumulative = float((1.0 + series).prod() - 1.0)
    ew_cumulative = float((1.0 + ew_series).prod() - 1.0)
    summary = {
        "year": "ALL-CHAIN",
        "selected_factor": "chained",
        "train_months": int(frame["train_months"].sum()) if not frame.empty else 0,
        "train_ic_mean": np.nan,
        "oos_return": cumulative,
        "ew_return": ew_cumulative,
        "excess": cumulative - ew_cumulative,
        "h00300_return": np.nan,
        "excess_h00300": np.nan,
        "oos_return_net20": np.nan,
        "excess_net20": np.nan,
        "chained_nav": 1.0 + cumulative,
        "chained_cumulative_return": cumulative,
        "chained_ew_nav": 1.0 + ew_cumulative,
        "chained_ew_cumulative_return": ew_cumulative,
        "annualized_return": chain_stats["annualized"],
        "sharpe": chain_stats["sharpe"],
        "max_drawdown": chain_stats["max_drawdown"],
    }
    return pd.concat([frame, pd.DataFrame([summary])], ignore_index=True)


def stationary_bootstrap_indices(length: int, reps: int, block: int, seed: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix of shape (reps, length)."""
    if length <= 0 or reps <= 0 or block <= 0:
        raise ValueError("The bootstrap requires a positive length, repetition count, and block length.")
    probability = 1.0 / block
    rng = np.random.default_rng(seed)
    indices = np.empty((reps, length), dtype=np.int64)
    indices[:, 0] = rng.integers(0, length, size=reps)
    fresh = rng.random((reps, length)) < probability
    starts = rng.integers(0, length, size=(reps, length))
    for position in range(1, length):
        indices[:, position] = np.where(
            fresh[:, position], starts[:, position], (indices[:, position - 1] + 1) % length
        )
    return indices


def white_reality_check(matrix: np.ndarray, reps: int, block: int, seed: int) -> float:
    """p-value of the max-mean White reality check across the columns of ``matrix``."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < MIN_SERIES_MONTHS or values.shape[1] == 0:
        return np.nan
    if not np.isfinite(values).all():
        return np.nan
    observed = float(values.mean(axis=0).max())
    centered = values - values.mean(axis=0)
    indices = stationary_bootstrap_indices(len(values), reps, block, seed)
    boot_means = centered[indices].mean(axis=1)
    max_boot = boot_means.max(axis=1)
    return float((1 + int((max_boot >= observed).sum())) / (reps + 1))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment."""
    values = np.asarray(p_values, dtype=float)
    count = len(values)
    adjusted = np.ones(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(np.argsort(values)):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment."""
    values = np.asarray(p_values, dtype=float)
    count = len(values)
    adjusted = np.ones(count, dtype=float)
    running = 1.0
    for rank_desc, index in enumerate(np.argsort(values)[::-1]):
        rank_asc = count - rank_desc
        running = min(running, min(1.0, values[index] * count / rank_asc))
        adjusted[index] = running
    return adjusted


def _p_from_t(t_value: float, degrees: int) -> float:
    """Two-sided p-value for a t statistic; an unusable statistic returns 1."""
    if t_value is None or not np.isfinite(t_value) or degrees <= 0:
        return 1.0
    return float(2.0 * stats.t.sf(abs(float(t_value)), degrees))


def _family_statistics(matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy: n, mean estimate, Newey-West t, and two-sided raw p."""
    rows = []
    for strategy in matrix.columns:
        values = matrix[strategy].astype("float64").dropna()
        t_value = newey_west_t(values) if len(values) >= MIN_SERIES_MONTHS else np.nan
        rows.append({
            "strategy": strategy,
            "n": int(len(values)),
            "estimate": float(values.mean()) if len(values) else np.nan,
            "t": t_value,
            "p_raw": _p_from_t(t_value, len(values) - 1),
        })
    return pd.DataFrame(rows)


def build_multiple_testing(monthly: pd.DataFrame, momentum_ic: pd.DataFrame,
                           momentum_quantiles: pd.DataFrame, neutral_monthly: pd.DataFrame,
                           pe_specs_ic: pd.DataFrame, pe_specs_quantiles: pd.DataFrame,
                           config: ValidationConfig) -> pd.DataFrame:
    """Holm, Benjamini-Hochberg, and a max-mean reality check inside six families.

    Families are defined by *one metric and one research question*: the primary
    long-only strategy, the momentum variants' IC, the momentum variants' Q5-Q1,
    the neutral long-only net20 excess, and the two PE-specification families.
    IC and return-spread tests are deliberately never pooled.
    """
    # Family 1: the three primary factors' monthly long-only excess vs the pool EW.
    excess_series = {}
    for factor in FACTORS:
        item = monthly.loc[monthly["factor"].eq(factor)].sort_values("month")
        excess_series[factor] = (
            item.set_index("month")["gross_return"].astype("float64")
            - item.set_index("month")["ew_return"].astype("float64")
        )
    families: dict[str, tuple[pd.DataFrame, str]] = {
        "main_long_only": (pd.DataFrame(excess_series).dropna(), "mean_excess_monthly"),
    }

    # Family 2: the three momentum variants' monthly Rank IC.
    variant_ic = momentum_ic.loc[
        momentum_ic["score_type"].eq("raw") & momentum_ic["factor"].isin(MOMENTUM_VARIANTS)
    ].copy()
    variant_ic["signal_date"] = pd.to_datetime(variant_ic["signal_date"]).astype("datetime64[ns]")
    variant_ic_pivot = variant_ic.pivot_table(
        index="signal_date", columns="factor", values="rank_ic"
    ).dropna()
    families["momentum_variants_ic"] = (variant_ic_pivot, "mean_rank_ic")

    # Family 3: the same three momentum variants' Q5-Q1 monthly spread.
    variant_spread = (
        momentum_quantiles.loc[
            momentum_quantiles["score_type"].eq("raw")
            & momentum_quantiles["factor"].isin(MOMENTUM_VARIANTS),
            ["factor", "signal_date", "q_high_minus_q_low"],
        ]
        .drop_duplicates(["factor", "signal_date"])
        .copy()
    )
    variant_spread["signal_date"] = pd.to_datetime(variant_spread["signal_date"]).astype("datetime64[ns]")
    variant_spread_pivot = variant_spread.pivot_table(
        index="signal_date", columns="factor", values="q_high_minus_q_low"
    ).dropna()
    families["momentum_variants_long_short"] = (variant_spread_pivot, "mean_q5_q1_monthly")

    # Family 4: the three neutralized long-only strategies, net of 20bp slippage,
    # relative to the same-pool monthly equal-weight benchmark.
    neutral = neutral_monthly.copy()
    neutral["month"] = neutral["month"].astype(str)
    neutral_net = neutral.loc[
        neutral["series_type"].eq("net") & neutral["slippage_bp"].eq(config.slippage_bp),
        ["factor", "month", "monthly_return"],
    ].rename(columns={"monthly_return": "net20_return"})
    ew = monthly.loc[monthly["factor"].eq(FACTORS[0]), ["month", "ew_return"]].copy()
    ew["month"] = ew["month"].astype(str)
    neutral_frame = neutral_net.merge(ew, on="month", how="inner", validate="many_to_one")
    neutral_frame["factor"] = neutral_frame["factor"].map({
        "pe": "pe_neutral", "momentum": "momentum_neutral", "roe": "roe_neutral",
    })
    neutral_frame["excess"] = neutral_frame["net20_return"] - neutral_frame["ew_return"]
    neutral_pivot = neutral_frame.pivot_table(
        index="month", columns="factor", values="excess"
    ).dropna()
    families["neutral_long_only"] = (neutral_pivot, "mean_net20_excess_monthly")

    # Family 5: monthly Rank IC of the four PE specifications.
    raw_ic = pe_specs_ic.loc[pe_specs_ic["score_type"].eq("raw")].copy()
    raw_ic["signal_date"] = pd.to_datetime(raw_ic["signal_date"]).astype("datetime64[ns]")
    ic_pivot = raw_ic.pivot_table(index="signal_date", columns="factor", values="rank_ic").dropna()
    families["pe_specs_ic"] = (ic_pivot, "mean_rank_ic")

    # Family 6: the four PE specifications' Q5-Q1 monthly spread.
    spread = (
        pe_specs_quantiles[["factor", "signal_date", "q_high_minus_q_low"]]
        .drop_duplicates(["factor", "signal_date"])
        .copy()
    )
    spread["signal_date"] = pd.to_datetime(spread["signal_date"]).astype("datetime64[ns]")
    spread_pivot = spread.pivot_table(
        index="signal_date", columns="factor", values="q_high_minus_q_low"
    ).dropna()
    families["pe_specs_long_short"] = (spread_pivot, "mean_q5_q1_monthly")

    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        if family not in families:
            continue
        matrix, metric = families[family]
        statistics = _family_statistics(matrix)
        p_values = statistics["p_raw"].to_numpy(dtype=float)
        holm = holm_adjust(p_values)
        bh = benjamini_hochberg_adjust(p_values)
        rc = white_reality_check(
            matrix.to_numpy(dtype=float), config.bootstrap_reps, config.bootstrap_block, config.seed
        )
        for (_, item), p_holm, q_bh in zip(statistics.iterrows(), holm, bh):
            rows.append({
                "family": family,
                "strategy": item["strategy"],
                "metric": metric,
                "n": int(item["n"]),
                "estimate": item["estimate"],
                "t": item["t"],
                "p_raw": item["p_raw"],
                "p_holm": float(p_holm),
                "q_bh": float(q_bh),
                "p_rc": rc,
            })
    return pd.DataFrame(rows, columns=list(MULTIPLE_TESTING_COLUMNS))


def validate_outputs(outputs: ValidationOutputs, config: ValidationConfig) -> None:
    subperiods = outputs.subperiods
    expected = len(FACTORS) * len(config.subperiods) * len(RETURN_TYPES)
    if len(subperiods) != expected:
        raise ValueError("The sub-period portfolio table does not cover every factor/period/return type.")
    if subperiods.duplicated(["factor", "subperiod", "return_type"]).any():
        raise ValueError("The sub-period portfolio table contains duplicate rows.")
    if subperiods["n_months"].le(0).any():
        raise ValueError("Every sub-period row must contain at least one month.")
    holdout = outputs.holdout
    if set(holdout["sample"]) != {"dev", "test"}:
        raise ValueError("The holdout table must contain both the development and test samples.")
    if len(holdout) != len(FACTORS) * 2:
        raise ValueError("The holdout table must hold two rows per candidate factor.")
    if int(holdout["selected"].sum()) != 2:
        raise ValueError("Exactly one factor must be selected in both holdout samples.")
    walk = outputs.walk_forward
    if not walk.empty:
        yearly = walk.loc[~walk["year"].astype(str).eq("ALL-CHAIN")]
        if yearly.empty:
            raise ValueError("The walk-forward table has no yearly rows.")
        if yearly["selected_factor"].isna().any():
            raise ValueError("A walk-forward year has no selected factor.")
        if yearly["train_months"].lt(config.min_train_months).any():
            raise ValueError("A walk-forward training window is shorter than the configured minimum.")
        chain = walk.loc[walk["year"].astype(str).eq("ALL-CHAIN")]
        if len(chain) != 1:
            raise ValueError("The walk-forward table must end with exactly one chained summary row.")
    testing = outputs.multiple_testing
    if set(testing["family"]) != set(FAMILY_ORDER):
        raise ValueError("Every multiple-testing family must be present.")
    for column in ("p_raw", "p_holm", "q_bh", "p_rc"):
        if testing[column].isna().any():
            raise ValueError(f"Multiple-testing column {column} must be fully populated.")
        if testing[column].lt(0).any() or testing[column].gt(1).any():
            raise ValueError(f"Multiple-testing column {column} must fall in [0, 1].")
    for family, group in testing.groupby("family"):
        if group["p_holm"].lt(group["p_raw"] - 1e-12).any():
            raise ValueError(f"Holm-adjusted p-values in {family} are below the raw p-values.")
        if group["q_bh"].lt(group["p_raw"] - 1e-12).any():
            raise ValueError(f"Benjamini-Hochberg q-values in {family} are below the raw p-values.")


def _format(value: float, digits: int = 4) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _format_percent(value: float, digits: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def render_report(outputs: ValidationOutputs, config: ValidationConfig) -> str:
    lines = [
        "# 时间切分与多重检验",
        "",
        "本报告只做 8/9 两步中的验证部分：子区间组合超额、固定 holdout、年度 walk-forward，"
        "以及按检验族分开的多重检验。**三个层次的可信度不同**，下面逐节说明。",
        "",
        "## 1 子区间组合超额（描述性）",
        "",
        f"子区间沿用 `factor_test.subperiods`（{', '.join(name for name, _, _ in config.subperiods)}）。"
        "四个子区间是**事后按风格划分**的，只用于描述策略超额如何随风格变化，"
        "不是样本外证据；4.3 的 IC 子区间表同样不能当作稳定性证明。",
        "",
        "| 因子 | 区间 | 收益 | 月数 | 策略年化 | 同池等权年化 | 超额（年化） | alpha 年化 | alpha t | H00300 年化 | 超额H00300 | Sharpe | 最大回撤 | 月均超额 | 超额 NW t |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in outputs.subperiods.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.subperiod} | {row.return_type} | {int(row.n_months)} | "
            f"{_format_percent(row.strategy_annualized)} | {_format_percent(row.ew_annualized)} | "
            f"{_format_percent(row.excess_ew_annualized)} | {_format_percent(row.alpha_annualized)} | "
            f"{_format(row.alpha_t, 2)} | {_format_percent(row.h00300_annualized)} | "
            f"{_format_percent(row.excess_h00300_annualized)} | {_format(row.sharpe, 3)} | "
            f"{_format_percent(row.max_drawdown)} | {_format_percent(row.excess_mean_monthly)} | "
            f"{_format(row.excess_t_nw, 2)} |"
        )
    lines.extend([
        "",
        "`gross` 为不含成本、`net20` 为 20 bp 滑点后的净收益；同池等权与 H00300 都按同一批月份重算年化。"
        "Momentum 在 2014-10 至 2015-03 因没有 12 个月历史而持有现金，2014–2016 区间的策略年化包含这 6 个零收益月，"
        "不能与 PE 的 28 个月直接比较。",
        "子区间只有 16–36 个月，**不要把它当成样本外**，也不要在其中挑一个最有利的区间下结论。",
        "",
        "## 2 固定 holdout：因子选择的样本外",
        "",
        f"开发期 {config.dev_start:%Y-%m-%d} 至留出期起点前一日，测试期 {config.holdout_start:%Y-%m-%d} 之后。"
        "选择规则只有一条：**按开发期 Rank IC 的 NW t 选一个最高者**，测试期不再改选、不调参数。",
        "",
        "| 因子 | 样本 | 月数(IC) | IC 均值 | IC NW t | 收益月数 | gross 年化 | net20 年化 | alpha(gross) | alpha t | alpha(net20) | alpha t | 选中 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in outputs.holdout.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.sample} | {int(row.n_ic_months)} | {_format(row.ic_mean)} | "
            f"{_format(row.ic_t_nw, 2)} | {int(row.n_months)} | {_format_percent(row.gross_annualized)} | "
            f"{_format_percent(row.net20_annualized)} | {_format_percent(row.alpha_gross_annualized)} | "
            f"{_format(row.alpha_gross_t, 2)} | {_format_percent(row.alpha_net20_annualized)} | "
            f"{_format(row.alpha_net20_t, 2)} | {'是' if row.selected else '否'} |"
        )
    lines.extend([
        "",
        "**这是「因子选择」的样本外，不是完整研究问题的样本外。**三个候选因子本身来自既有文献与"
        "全样本认知，「只选一个」这一步在测试期没有再看数据，但候选集合、因子定义（TTM 口径、"
        "12-1 定义、年报 ROE）、股票池与执行规则都是全样本设计的。",
        "",
        "## 3 年度 walk-forward：因子轮动选择的样本外",
        "",
        f"从 {config.walk_forward_start_year} 年起，每年用该年 1 月 1 日之前、至少 {config.min_train_months} 个月的"
        " Rank IC 选择训练期平均 IC 最高的因子，再用该因子下一年的月度 gross/net20 作为 OOS。",
        "",
        "| 年份 | 选中因子 | 训练月数 | 训练期 IC 均值 | OOS 收益 | 同池等权 | 超额 | H00300 | 超额H00300 | OOS net20 | 净超额 | 链式净值 | 链式等权净值 | 年内 Sharpe | 年内最大回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.walk_forward.itertuples(index=False):
        label = str(row.year)
        if label == "ALL-CHAIN":
            lines.append(
                f"| **ALL-CHAIN** | 链式 | {int(row.train_months)} | — | {_format_percent(row.chained_cumulative_return)} | "
                f"{_format_percent(row.chained_ew_cumulative_return)} | {_format_percent(row.excess)} | — | — | — | — | "
                f"{_format(row.chained_nav, 4)} | {_format(row.chained_ew_nav, 4)} | "
                f"{_format(row.sharpe, 3)} | {_format_percent(row.max_drawdown)} |"
            )
            continue
        lines.append(
            f"| {label} | {row.selected_factor} | {int(row.train_months)} | {_format(row.train_ic_mean)} | "
            f"{_format_percent(row.oos_return)} | {_format_percent(row.ew_return)} | {_format_percent(row.excess)} | "
            f"{_format_percent(row.h00300_return)} | {_format_percent(row.excess_h00300)} | "
            f"{_format_percent(row.oos_return_net20)} | {_format_percent(row.excess_net20)} | "
            f"{_format(row.chained_nav, 4)} | {_format(row.chained_ew_nav, 4)} | "
            f"{_format(row.sharpe, 3)} | {_format_percent(row.max_drawdown)} |"
        )
    chain = outputs.walk_forward.loc[outputs.walk_forward["year"].astype(str).eq("ALL-CHAIN")]
    if not chain.empty:
        row = chain.iloc[0]
        lines.extend([
            "",
            f"链式结果：{_format_percent(row['chained_cumulative_return'])} 累计、"
            f"{_format_percent(row['annualized_return'])} 年化、Sharpe {_format(row['sharpe'], 3)}、"
            f"最大回撤 {_format_percent(row['max_drawdown'])}；同期同池等权链式净值 "
            f"{_format_percent(row['chained_ew_cumulative_return'])}，"
            f"因此这套轮动在 OOS 上相对等权为 {_format_percent(row['excess'])}。",
        ])
    lines.extend([
        "",
        "只有 3 个候选因子：walk-forward 验证的是**因子轮动选择**，不是参数优化；它没有调 Top 比例、"
        "滑点或权重，也不包含任何新的因子。它比子区间描述更强，但样本外只有 8 个年度决策，统计功效有限。",
        "",
        "## 4 多重检验（按检验族分开）",
        "",
        "每个族只检验一个指标，族内做 Holm/BH/RC，族间不合并排序。六个族分别是："
        "`main_long_only`（三主因子纯多头 gross 超额）、"
        "`momentum_variants_ic`（12-1/6-1/1M 的 Rank IC）、"
        "`momentum_variants_long_short`（三个变体的 Q5-Q1）、"
        "`neutral_long_only`（三个中性化纯多头的月均 net20 超额）、"
        "`pe_specs_ic`（四个 PE 写法的 Rank IC）、"
        "`pe_specs_long_short`（四个 PE 写法的 Q5-Q1）。"
        "IC 与收益价差、gross 与 net20 都不能放进同一个族。",
        "",
        "raw p 用普通 t 分布：`p = 2·sf(|t|, df)`，但 t 为 Newey-West 修正后的均值 t；"
        "均值类指标取 `df = n−1`，若表里加入回归 alpha 才用 `df = n−2`。"
        f"White Reality Check 为 **max-mean** 统计量（不是 max-Sharpe）：各列去均值后做 stationary bootstrap，"
        f"block length {config.bootstrap_block}、{config.bootstrap_reps} 次、固定 seed {config.seed}，"
        "`p_rc = (1 + #{max_boot ≥ observed_max}) / (reps + 1)`。",
        "",
        "| 检验族 | 策略 | 指标 | n | 估计 | t（NW） | p_raw | p_holm | q_bh | p_rc |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in outputs.multiple_testing.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.strategy} | {row.metric} | {int(row.n)} | {_format(row.estimate, 6)} | "
            f"{_format(row.t, 2)} | {_format(row.p_raw, 4)} | {_format(row.p_holm, 4)} | "
            f"{_format(row.q_bh, 4)} | {_format(row.p_rc, 4)} |"
        )
    for family, group in outputs.multiple_testing.groupby("family"):
        survivors = group.loc[group["p_holm"].le(0.05)]
        raw_survivors = group.loc[group["p_raw"].le(0.05)]
        rc_value = group["p_rc"].iloc[0]
        minimum_raw = group["p_raw"].min()
        note = ""
        if pd.notna(rc_value) and pd.notna(minimum_raw) and rc_value < minimum_raw:
            note = (
                f"注意 RC p 值（{rc_value:.4f}）低于该族最小的 raw p 值（{minimum_raw:.4f}）："
                "block bootstrap 保留了月度序列的相关结构，当这些序列呈负自相关时其长期方差小于 iid 方差，"
                "bootstrap 分布更窄，于是 p_rc 更小。两个 p 值对应的原假设分布不同、不是严格嵌套，"
                "不能把 p_rc 当成 p_raw 的更严格版本。"
            )
        lines.extend([
            "",
            f"- 族 `{family}`：raw p ≤ 0.05 的 {len(raw_survivors)} 个，Holm 后仍 ≤ 0.05 的 {len(survivors)} 个"
            f"（{', '.join(sorted(survivors['strategy'])) or '无'}）；RC p 值 {_format(rc_value, 4)}。{note}",
        ])
    lines.extend([
        "",
        "**读法：**若某个主因子的 raw t 显著、但 Holm/BH/RC 后不显著，必须写「多重检验后不显著」，"
        "不能只报 raw t。RC 是 max-mean 的联合检验，不替代 t 检验，也不替代 Holm/BH；"
        "它检验「整个族里没有任何一列优于基准」这一联合原假设，并以 bootstrap 方式保留序列相关。",
        "",
        "## 5 口径与限制",
        "",
        "- 子区间、holdout、walk-forward 都使用已算好的策略月度 gross/net20 与 IC 序列，不重新回测、不改执行规则。",
        "- holdout 的「选中」只看开发期 IC 的 NW t；测试期的 alpha、Sharpe、回撤都只作为结果报告，不参与任何选择。",
        "- walk-forward 只按训练期**平均 Rank IC** 选择，不看训练期的多空或纯多头收益；训练期与 OOS 之间没有重叠月份。",
        "- 月度年化按 12 个月折算；月频最大回撤按月末净值计算，与日频回测口径不同。",
        "- 多重检验的 t 值统一为 Newey-West 修正后的均值 t；raw p 仍按 t 分布近似，不是精确的 HAC 分布。",
        "- Holm/BH 只在同一检验族内校正；六个族分开做，不能把不同族的 p 值合并后统一排序。",
        "- `main_long_only` 是 gross 超额；`neutral_long_only` 是 net20 超额，两者分别回答不同问题。",
        "- RC 的 bootstrap 在原假设下对每列去均值，因此它对应「所有策略期望都不优于 0」的联合原假设；"
        "`main_long_only` 族的输入是相对同池等权的月度 gross 超额，`neutral_long_only` 是 net20 超额，"
        "IC 族与多空族的输入分别是月度 Rank IC 与 Q5-Q1 收益。",
        "",
    ])
    return "\n".join(lines)


def run(config: ValidationConfig) -> ValidationOutputs:
    directory = config.processed_dir
    monthly = assemble_monthly(
        pd.read_csv(directory / "backtest_monthly_returns.csv"),
        pd.read_csv(directory / "backtest_net_monthly_returns.csv"),
        pd.read_csv(directory / "attribution_benchmark_monthly.csv"),
        slippage_bp=config.slippage_bp,
    )
    ic_series = pd.read_csv(directory / "factor_ic_series.csv")
    momentum_ic = pd.read_csv(directory / "momentum_variants_ic.csv")
    momentum_quantiles = pd.read_csv(directory / "momentum_variants_quantiles.csv")
    neutral_monthly = pd.read_csv(directory / "attribution_neutral_monthly_returns.csv")
    pe_specs_ic = pd.read_csv(directory / "pe_specs_ic.csv")
    pe_specs_quantiles = pd.read_csv(directory / "pe_specs_quantiles.csv")
    subperiods = build_subperiod_portfolio(monthly, config)
    holdout = build_holdout(monthly, ic_series, config)
    walk_forward = build_walk_forward(monthly, ic_series, config)
    multiple_testing = build_multiple_testing(
        monthly, momentum_ic, momentum_quantiles, neutral_monthly,
        pe_specs_ic, pe_specs_quantiles, config,
    )
    outputs = ValidationOutputs(subperiods, holdout, walk_forward, multiple_testing)
    validate_outputs(outputs, config)
    subperiods.to_csv(directory / "validation_subperiod_portfolio.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(directory / "validation_holdout.csv", index=False, encoding="utf-8-sig")
    walk_forward.to_csv(directory / "validation_walkforward.csv", index=False, encoding="utf-8-sig")
    multiple_testing.to_csv(directory / "validation_multiple_testing.csv", index=False, encoding="utf-8-sig")
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation.md").write_text(render_report(outputs, config), encoding="utf-8")
    print(
        f"Built {len(subperiods)} sub-period rows, {len(holdout)} holdout rows, "
        f"{len(walk_forward)} walk-forward rows, and {len(multiple_testing)} multiple-testing rows."
    )
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the time-split and multiple-testing validation.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    run(load_validation_config(args.config))


if __name__ == "__main__":
    main()
