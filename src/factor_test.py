"""Test factors with forward returns, quantiles, and Rank IC."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

FACTORS = ("pe", "momentum", "roe")
MIN_T_STAT_ROWS = 3


@dataclass(frozen=True)
class FactorTestConfig:
    processed_dir: Path
    quantile_groups: int
    min_ic_pairs: int
    subperiods: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...] = ()


@dataclass(frozen=True)
class FactorTestOutputs:
    forward_returns: pd.DataFrame
    details: pd.DataFrame
    members: pd.DataFrame
    quantile_returns: pd.DataFrame
    ic_series: pd.DataFrame
    summary: pd.DataFrame
    subperiods: pd.DataFrame = field(default_factory=pd.DataFrame)


def parse_subperiods(raw: object) -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    """Read the configured sub-periods and reject overlaps or mixed-up order."""
    entries = tuple(
        (str(item["name"]), pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))
        for item in (raw or [])
    )
    for name, start, end in entries:
        if start > end:
            raise ValueError(f"Sub-period {name} starts after it ends.")
    for previous, following in zip(entries, entries[1:]):
        if following[1] <= previous[2]:
            raise ValueError(f"Sub-period {following[0]} overlaps {previous[0]}.")
    return entries


def load_factor_test_config(config_path: Path) -> FactorTestConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    return FactorTestConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        quantile_groups=int(settings["factors"]["quantile_groups"]),
        min_ic_pairs=int(settings["factor_test"]["min_ic_pairs"]),
        subperiods=parse_subperiods(settings["factor_test"].get("subperiods")),
    )


def build_periods(signal_dates: pd.Series, trading_dates: pd.Series) -> pd.DataFrame:
    """Pair every signal with its next signal and first later trading day."""
    signals = pd.Series(pd.to_datetime(signal_dates).unique()).sort_values().reset_index(drop=True)
    calendar = pd.Series(pd.to_datetime(trading_dates).unique()).sort_values().reset_index(drop=True)
    periods = pd.DataFrame({"signal_date": signals, "holding_end_date": signals.shift(-1)})
    positions = calendar.searchsorted(signals, side="right")
    periods["first_holding_date"] = [calendar.iloc[position] if position < len(calendar) else pd.NaT for position in positions]
    return periods


def _valid_return_rows(daily: pd.DataFrame) -> pd.Series:
    flags = daily["quality_flags"].fillna("").astype("string")
    return daily["total_return"].notna() & ~flags.str.contains("invalid_total_return", regex=False)


def build_forward_returns(factor_details: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Create one forward-return record for every factor row."""
    keys = factor_details[["signal_date", "stock_code"]].drop_duplicates().copy()
    periods = build_periods(keys["signal_date"], daily["trade_date"])
    mapped = pd.merge_asof(
        daily.sort_values("trade_date"), periods.sort_values("signal_date"),
        left_on="trade_date", right_on="signal_date", direction="backward", allow_exact_matches=False,
    )
    mapped = mapped.loc[
        mapped["signal_date"].notna() & mapped["holding_end_date"].notna()
        & mapped["trade_date"].le(mapped["holding_end_date"])
    ].copy()
    mapped["valid_return"] = _valid_return_rows(mapped)
    valid = mapped.loc[mapped["valid_return"]].copy()
    group_keys = ["signal_date", "stock_code"]
    aggregated = valid.groupby(group_keys, as_index=False).agg(
        last_observed_date=("trade_date", "max"),
        valid_return_days=("trade_date", "size"),
        extreme_return_days=("quality_flags", lambda values: int(values.fillna("").astype("string").str.contains("extreme_total_return", regex=False).sum())),
    )
    compounded = (1 + valid["total_return"]).groupby([valid["signal_date"], valid["stock_code"]]).prod().sub(1)
    aggregated = aggregated.merge(compounded.rename("forward_return").reset_index(), on=group_keys, how="left", validate="one_to_one")
    entry = valid.loc[valid["trade_date"].eq(valid["first_holding_date"]), group_keys].drop_duplicates()
    entry["has_valid_first_holding_day"] = True

    result = keys.merge(periods, on="signal_date", how="left", validate="many_to_one")
    result = result.merge(aggregated, on=group_keys, how="left", validate="one_to_one")
    result = result.merge(entry, on=group_keys, how="left", validate="one_to_one")
    result["has_valid_first_holding_day"] = result["has_valid_first_holding_day"].fillna(False).astype(bool)
    result["forward_eligible"] = result["holding_end_date"].notna() & result["has_valid_first_holding_day"]
    result["forward_unavailable_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[result["holding_end_date"].isna(), "forward_unavailable_reason"] = "no_future_holding_period"
    result.loc[result["holding_end_date"].notna() & ~result["has_valid_first_holding_day"], "forward_unavailable_reason"] = "missing_valid_first_holding_day"
    result["early_exit_or_data_cutoff"] = result["forward_eligible"] & result["last_observed_date"].lt(result["holding_end_date"])
    return result.sort_values(group_keys).reset_index(drop=True)


def build_forward_returns_streaming(factor_details: pd.DataFrame, daily_path: Path) -> pd.DataFrame:
    """Stream the large daily Parquet file while retaining the same forward-return rules."""
    keys = factor_details[["signal_date", "stock_code"]].drop_duplicates().copy()
    parquet_file = pq.ParquetFile(daily_path)
    trading_days: set[pd.Timestamp] = set()
    for batch in parquet_file.iter_batches(columns=["trade_date"], batch_size=250_000):
        trading_days.update(pd.to_datetime(batch.column(0).to_pandas()).unique())
    periods = build_periods(keys["signal_date"], pd.Series(list(trading_days)))
    signals = periods["signal_date"].to_numpy()
    ends = periods["holding_end_date"].to_numpy()
    firsts = periods["first_holding_date"].to_numpy()
    partials = []
    for batch in parquet_file.iter_batches(columns=["stock_code", "trade_date", "total_return", "quality_flags"], batch_size=250_000):
        chunk = batch.to_pandas()
        dates = chunk["trade_date"].to_numpy()
        positions = signals.searchsorted(dates, side="left") - 1
        in_window = (positions >= 0) & (positions < len(signals) - 1)
        chunk = chunk.loc[in_window].copy()
        positions = positions[in_window]
        chunk["signal_date"] = signals[positions]
        chunk["holding_end_date"] = ends[positions]
        chunk["first_holding_date"] = firsts[positions]
        chunk = chunk.loc[chunk["trade_date"].le(chunk["holding_end_date"])]
        chunk = chunk.merge(keys, on=["signal_date", "stock_code"], how="inner", validate="many_to_one")
        if chunk.empty:
            continue
        chunk = chunk.loc[_valid_return_rows(chunk)].copy()
        if chunk.empty:
            continue
        chunk["return_factor"] = 1 + chunk["total_return"]
        chunk["is_extreme_return"] = chunk["quality_flags"].fillna("").astype("string").str.contains("extreme_total_return", regex=False)
        chunk["is_first_holding_day"] = chunk["trade_date"].eq(chunk["first_holding_date"])
        partials.append(chunk.groupby(["signal_date", "stock_code"], as_index=False).agg(
            return_factor=("return_factor", "prod"),
            last_observed_date=("trade_date", "max"),
            valid_return_days=("trade_date", "size"),
            extreme_return_days=("is_extreme_return", "sum"),
            has_valid_first_holding_day=("is_first_holding_day", "max"),
        ))
    if not partials:
        raise ValueError("No valid daily returns were available for the factor test.")
    partial = pd.concat(partials, ignore_index=True)
    aggregated = partial.groupby(["signal_date", "stock_code"], as_index=False).agg(
        return_factor=("return_factor", "prod"),
        last_observed_date=("last_observed_date", "max"),
        valid_return_days=("valid_return_days", "sum"),
        extreme_return_days=("extreme_return_days", "sum"),
        has_valid_first_holding_day=("has_valid_first_holding_day", "max"),
    )
    aggregated["forward_return"] = aggregated.pop("return_factor") - 1
    result = keys.merge(periods, on="signal_date", how="left", validate="many_to_one")
    result = result.merge(aggregated, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    result["has_valid_first_holding_day"] = result["has_valid_first_holding_day"].fillna(False).astype(bool)
    result["forward_eligible"] = result["holding_end_date"].notna() & result["has_valid_first_holding_day"]
    result["forward_unavailable_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[result["holding_end_date"].isna(), "forward_unavailable_reason"] = "no_future_holding_period"
    result.loc[result["holding_end_date"].notna() & ~result["has_valid_first_holding_day"], "forward_unavailable_reason"] = "missing_valid_first_holding_day"
    result["early_exit_or_data_cutoff"] = result["forward_eligible"] & result["last_observed_date"].lt(result["holding_end_date"])
    return result.sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


def assign_quantiles(scores: pd.Series, quantile_groups: int) -> pd.Series:
    """Assign tied scores together; Q1 is lowest score and QN is highest score."""
    result = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    valid = scores.notna()
    if valid.any():
        percentile = scores.loc[valid].rank(method="average", pct=True)
        result.loc[valid] = (percentile * quantile_groups).apply(math.ceil).astype("Int64")
    return result


def available_score_types(factor_details: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return the (score_type, score suffix, eligibility suffix) triples that exist."""
    types = [("raw", "_score", "_is_rankable")]
    if "pe_neutral_eligible" in factor_details.columns:
        types.append(("raw_neutral_sample", "_score", "_neutral_eligible"))
        types.append(("neutral", "_score_neutral", "_neutral_eligible"))
    return types


def build_factor_test_details(factor_details: pd.DataFrame, forward_returns: pd.DataFrame,
                              quantile_groups: int, score_type: str = "raw",
                              score_suffix: str = "_score",
                              eligibility_suffix: str = "_is_rankable") -> pd.DataFrame:
    """Expand common forward returns into factor-specific eligibility and groups."""
    frames = []
    for factor in FACTORS:
        frame = factor_details[["signal_date", "stock_code"]].copy()
        frame["factor"] = factor
        frame["factor_score"] = factor_details[f"{factor}{score_suffix}"].astype("Float64").astype("float64").to_numpy()
        frame["factor_is_rankable"] = factor_details[f"{factor}_is_rankable"].fillna(False).astype(bool).to_numpy()
        frame["score_eligible"] = factor_details[f"{factor}{eligibility_suffix}"].fillna(False).astype(bool).to_numpy()
        frame["factor_unrankable_reason"] = factor_details[f"{factor}_unrankable_reason"].to_numpy()
        frame = frame.merge(forward_returns, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
        frame["score_type"] = score_type
        forward_eligible = frame["forward_eligible"].fillna(False).astype(bool)
        frame["test_eligible"] = frame["score_eligible"] & forward_eligible
        frame["test_unavailable_reason"] = frame["forward_unavailable_reason"].astype("string")
        missing_score = frame["test_unavailable_reason"].isna() & ~frame["score_eligible"]
        frame.loc[missing_score, "test_unavailable_reason"] = "factor_unrankable:" + frame.loc[missing_score, "factor_unrankable_reason"].fillna("unknown")
        frame["quantile"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        for _, indexes in frame.loc[frame["test_eligible"]].groupby("signal_date").groups.items():
            frame.loc[indexes, "quantile"] = assign_quantiles(frame.loc[indexes, "factor_score"], quantile_groups)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["score_type", "factor", "signal_date", "stock_code"]).reset_index(drop=True)


def build_quantile_returns(details: pd.DataFrame, quantile_groups: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = details.loc[details["quantile"].notna()].copy()
    keys = ["factor", "score_type", "signal_date", "quantile"]
    grouped = members.groupby(keys, as_index=False).agg(
        stock_count=("stock_code", "size"),
        equal_weight_forward_return=("forward_return", "mean"),
        extreme_return_stock_count=("extreme_return_days", lambda values: int(values.gt(0).sum())),
        early_exit_or_data_cutoff_count=("early_exit_or_data_cutoff", "sum"),
    )
    periods = details.loc[details["test_eligible"], ["factor", "score_type", "signal_date"]].drop_duplicates()
    grid = periods.merge(pd.DataFrame({"quantile": range(1, quantile_groups + 1)}), how="cross")
    output = grid.merge(grouped, on=keys, how="left", validate="one_to_one")
    output["stock_count"] = output["stock_count"].fillna(0).astype("int64")
    output["empty_group"] = output["stock_count"].eq(0)
    output = output.sort_values(["factor", "score_type", "quantile", "signal_date"]).reset_index(drop=True)
    output["cumulative_group_return"] = output.groupby(["factor", "score_type", "quantile"])[
        "equal_weight_forward_return"
    ].transform(lambda values: (1 + values.fillna(0)).cumprod() - 1)
    wide = output.pivot(index=["factor", "score_type", "signal_date"], columns="quantile",
                        values="equal_weight_forward_return")
    spread = wide.get(quantile_groups).sub(wide.get(1)).rename("q_high_minus_q_low").reset_index()
    spread = spread.sort_values(["factor", "score_type", "signal_date"]).reset_index(drop=True)
    spread["cumulative_q_high_minus_q_low"] = (
        spread.groupby(["factor", "score_type"])["q_high_minus_q_low"]
        .transform(lambda values: values.fillna(0).cumsum())
    )
    output = output.merge(spread, on=["factor", "score_type", "signal_date"], how="left")
    return members, output.sort_values(["factor", "score_type", "signal_date", "quantile"]).reset_index(drop=True)


def build_ic_series(details: pd.DataFrame, min_ic_pairs: int) -> pd.DataFrame:
    rows = []
    keys = ["factor", "score_type", "signal_date"]
    for (factor, score_type, signal_date), group in details.loc[details["test_eligible"]].groupby(keys, sort=True):
        pair_count = len(group)
        ic, reason = pd.NA, pd.NA
        if pair_count < min_ic_pairs:
            reason = "insufficient_valid_pairs"
        else:
            calculated = group["factor_score"].corr(group["forward_return"], method="spearman")
            if pd.isna(calculated):
                reason = "constant_factor_or_forward_return"
            else:
                ic = float(calculated)
        rows.append({"factor": factor, "score_type": score_type, "signal_date": signal_date,
                     "ic_pair_count": pair_count, "rank_ic": ic, "ic_unavailable_reason": reason})
    return pd.DataFrame(rows)


def newey_west_lags(count: int) -> int:
    """Newey-West (1994) automatic bandwidth: floor(4 * (T/100)^(2/9))."""
    return max(1, int(np.floor(4 * (count / 100) ** (2 / 9))))


def newey_west_t(values: pd.Series, lags: int | None = None) -> float:
    """t statistic of the mean with a Bartlett-weighted long-run variance."""
    clean = values.dropna().to_numpy(dtype=float)
    count = clean.size
    if count < MIN_T_STAT_ROWS:
        return np.nan
    bandwidth = min(newey_west_lags(count) if lags is None else int(lags), count - 1)
    centered = clean - clean.mean()
    long_run = float((centered ** 2).sum() / count)
    for lag in range(1, bandwidth + 1):
        covariance = float((centered[lag:] * centered[:-lag]).sum() / count)
        long_run += 2 * (1 - lag / (bandwidth + 1)) * covariance
    if long_run <= 0:
        return np.nan
    return float(clean.mean() / np.sqrt(long_run / count))


def _mean_ir_t(values: pd.Series) -> tuple[float | pd.NA, float | pd.NA, float | pd.NA, float | pd.NA]:
    """Mean, information ratio, plain t, and Newey-West t of a monthly series."""
    clean = values.dropna()
    if len(clean) < 2:
        return pd.NA, pd.NA, pd.NA, pd.NA
    mean = float(clean.mean())
    spread = float(clean.std(ddof=1))
    if spread == 0:
        return mean, pd.NA, pd.NA, pd.NA
    ratio = mean / spread
    return mean, ratio, ratio * np.sqrt(len(clean)), newey_west_t(clean)


def build_summary(details: pd.DataFrame, quantile_returns: pd.DataFrame, ic_series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, score_type), item in details.groupby(["factor", "score_type"], sort=True):
        ic = ic_series.loc[ic_series["factor"].eq(factor) & ic_series["score_type"].eq(score_type), "rank_ic"]
        groups = quantile_returns.loc[
            quantile_returns["factor"].eq(factor) & quantile_returns["score_type"].eq(score_type)
        ]
        spread = groups.loc[groups["quantile"].eq(1), ["signal_date", "q_high_minus_q_low"]].drop_duplicates()
        ic_mean, ic_ir, ic_t, ic_t_nw = _mean_ir_t(ic)
        spread_mean, _, spread_t, spread_t_nw = _mean_ir_t(spread["q_high_minus_q_low"])
        rows.append({
            "factor": factor, "score_type": score_type,
            "signal_periods": int(item["signal_date"].nunique()),
            "test_eligible_rows": int(item["test_eligible"].sum()),
            "no_future_holding_period_rows": int(item["test_unavailable_reason"].eq("no_future_holding_period").sum()),
            "missing_first_holding_day_rows": int(item["test_unavailable_reason"].eq("missing_valid_first_holding_day").sum()),
            "early_exit_or_data_cutoff_rows": int(item.loc[item["test_eligible"], "early_exit_or_data_cutoff"].sum()),
            "extreme_return_stock_rows": int(item.loc[item["test_eligible"], "extreme_return_days"].gt(0).sum()),
            "empty_group_rows": int(groups["empty_group"].sum()),
            "ic_valid_periods": int(ic.notna().sum()),
            "ic_mean": ic_mean, "ic_std": float(ic.std(ddof=1)) if ic.notna().sum() > 1 else pd.NA,
            "ic_ir": ic_ir, "ic_t": ic_t, "ic_t_nw": ic_t_nw,
            "nw_lags": newey_west_lags(int(ic.notna().sum())) if ic.notna().sum() else pd.NA,
            "ic_win_rate": ic.gt(0).mean() if ic.notna().any() else pd.NA,
            "spread_mean": spread_mean, "spread_t": spread_t, "spread_t_nw": spread_t_nw,
        })
    return pd.DataFrame(rows)


def build_subperiod_table(quantile_returns: pd.DataFrame, ic_series: pd.DataFrame,
                          subperiods: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]) -> pd.DataFrame:
    """Repeat the IC and spread statistics inside each configured regime."""
    if not subperiods:
        return pd.DataFrame()
    spread_all = (
        quantile_returns.loc[quantile_returns["quantile"].eq(1),
                            ["factor", "score_type", "signal_date", "q_high_minus_q_low"]]
        .drop_duplicates(["factor", "score_type", "signal_date"])
    )
    rows = []
    for (factor, score_type), ic_group in ic_series.groupby(["factor", "score_type"], sort=True):
        spread_group = spread_all.loc[
            spread_all["factor"].eq(factor) & spread_all["score_type"].eq(score_type)
        ]
        for name, start, end in subperiods:
            ic_slice = ic_group.loc[ic_group["signal_date"].between(start, end), "rank_ic"]
            spread_slice = spread_group.loc[
                spread_group["signal_date"].between(start, end), "q_high_minus_q_low"
            ]
            ic_mean, ic_ir, _, ic_t_nw = _mean_ir_t(ic_slice)
            spread_mean, _, _, spread_t_nw = _mean_ir_t(spread_slice)
            rows.append({
                "factor": factor, "score_type": score_type, "period": name,
                "start": start, "end": end,
                "periods": int(ic_slice.notna().sum()),
                "ic_mean": ic_mean, "ic_ir": ic_ir, "ic_t_nw": ic_t_nw,
                "ic_win_rate": ic_slice.gt(0).mean() if ic_slice.notna().any() else pd.NA,
                "spread_mean": spread_mean, "spread_t_nw": spread_t_nw,
            })
    return pd.DataFrame(rows)


def validate_outputs(outputs: FactorTestOutputs, quantile_groups: int, min_ic_pairs: int) -> None:
    forward, members, groups, ic = outputs.forward_returns, outputs.members, outputs.quantile_returns, outputs.ic_series
    if forward.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Forward returns contain duplicate keys.")
    active = forward["holding_end_date"].notna()
    if forward.loc[active, "first_holding_date"].le(forward.loc[active, "signal_date"]).any():
        raise ValueError("A forward-return window begins on or before its signal date.")
    if forward.loc[~active, "forward_unavailable_reason"].ne("no_future_holding_period").any():
        raise ValueError("The final signal period was not explicitly excluded.")
    if (~members["test_eligible"]).any() or members["quantile"].lt(1).any() or members["quantile"].gt(quantile_groups).any():
        raise ValueError("Quantile membership contains an ineligible or invalid row.")
    if groups.duplicated(["factor", "score_type", "signal_date", "quantile"]).any():
        raise ValueError("Quantile results have duplicate group keys.")
    if (ic["rank_ic"].notna() & ic["ic_pair_count"].lt(min_ic_pairs)).any():
        raise ValueError("Rank IC was calculated below the configured minimum sample size.")
    if set(ic["score_type"]) != set(members["score_type"]):
        raise ValueError("Every score type must appear in the IC series.")
    if not outputs.subperiods.empty:
        if outputs.subperiods.duplicated(["factor", "score_type", "period"]).any():
            raise ValueError("Sub-period table contains duplicate rows.")
        totals = outputs.subperiods.groupby(["factor", "score_type"])["periods"].sum()
        expected = outputs.summary.set_index(["factor", "score_type"])["ic_valid_periods"]
        if not totals.reindex(expected.index).fillna(0).eq(expected).all():
            raise ValueError("Sub-period counts do not add up to the overall IC periods.")


def render_report(outputs: FactorTestOutputs) -> str:
    lines = ["# 因子有效性检验", "", "本报告检验因子分数对下一月收益的区分能力，不代表实际开盘成交后的策略收益。", "",
             "## 汇总", "", "| 因子 | 分数 | 有效 IC 期数 | IC 均值 | IC 标准差 | ICIR | IC t 值 | IC t 值（NW） | IC 胜率 | Q5-Q1 均值 | Q5-Q1 t 值（NW） | NW 阶数 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    order = {"raw": 0, "raw_neutral_sample": 1, "neutral": 2}
    summary = outputs.summary.assign(_order=outputs.summary["score_type"].map(order)).sort_values(["_order", "factor"])
    for row in summary.itertuples(index=False):
        def fmt(value: float) -> str:
            return "—" if pd.isna(value) else f"{value:.4f}"
        lines.append(
            f"| {row.factor} | {row.score_type} | {row.ic_valid_periods} | "
            f"{fmt(row.ic_mean)} | {fmt(row.ic_std)} | {fmt(row.ic_ir)} | {fmt(row.ic_t)} | {fmt(row.ic_t_nw)} | "
            f"{fmt(row.ic_win_rate)} | {fmt(row.spread_mean)} | {fmt(row.spread_t_nw)} | {row.nw_lags} |"
        )
    if "neutral" in set(outputs.summary["score_type"]):
        lines.extend(["", "`raw` 为全样本原始分数，`raw_neutral_sample` 为同一批中性化可评估样本上的原始分数，"
                      "`neutral` 为行业与市值中性化后的分数。三者的差值才是中性化本身的效应。",
                      "`t 值（NW）` 用 Bartlett 核的 long-run 方差，带宽按 Newey-West 自动规则取 4×(T/100)^(2/9)，"
                      "与普通 t 值并列以便看出序列相关的影响。", ""])
    if not outputs.subperiods.empty:
        lines.extend(["", "## 子区间稳定性", "",
                      "| 因子 | 分数 | 区间 | 期数 | IC 均值 | IC t 值（NW） | ICIR | IC 胜率 | Q5-Q1 均值 | Q5-Q1 t 值（NW） |",
                      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in outputs.subperiods.itertuples(index=False):
            def fmt_period(value: float) -> str:
                return "—" if pd.isna(value) else f"{value:.4f}"
            lines.append(
                f"| {row.factor} | {row.score_type} | {row.period} | {row.periods} | "
                f"{fmt_period(row.ic_mean)} | {fmt_period(row.ic_t_nw)} | {fmt_period(row.ic_ir)} | "
                f"{fmt_period(row.ic_win_rate)} | {fmt_period(row.spread_mean)} | {fmt_period(row.spread_t_nw)} |"
            )
    lines.extend(["", "## 口径限制", "", "- 2025-12-31 没有后续持有期数据，未进入分组收益或 IC。",
                  "- Momentum 在 2014-09 至 2015-02 没有可用六个月历史价格，未进行分组或 IC。",
                  "- PE 按交易日 `PE1TTM` 可用的既定口径处理；该限制仅在最终报告中说明。", ""])
    return "\n".join(lines)


def run(config: FactorTestConfig) -> FactorTestOutputs:
    directory = config.processed_dir
    factor_details = pd.read_parquet(directory / "factor_details.parquet")
    forward = build_forward_returns_streaming(factor_details, directory / "daily_trading.parquet")
    frames = [
        build_factor_test_details(factor_details, forward, config.quantile_groups, score_type, score_suffix, eligibility_suffix)
        for score_type, score_suffix, eligibility_suffix in available_score_types(factor_details)
    ]
    details = pd.concat(frames, ignore_index=True)
    members, quantile_returns = build_quantile_returns(details, config.quantile_groups)
    ic_series = build_ic_series(details, config.min_ic_pairs)
    summary = build_summary(details, quantile_returns, ic_series)
    subperiods = build_subperiod_table(quantile_returns, ic_series, config.subperiods)
    outputs = FactorTestOutputs(forward, details, members, quantile_returns, ic_series, summary, subperiods)
    validate_outputs(outputs, config.quantile_groups, config.min_ic_pairs)
    forward.to_parquet(directory / "factor_forward_returns.parquet", index=False)
    details.to_parquet(directory / "factor_test_details.parquet", index=False)
    members.to_parquet(directory / "factor_quantile_members.parquet", index=False)
    quantile_returns.to_csv(directory / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")
    ic_series.to_csv(directory / "factor_ic_series.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(directory / "factor_test_summary.csv", index=False, encoding="utf-8-sig")
    if not subperiods.empty:
        subperiods.to_csv(directory / "factor_subperiods.csv", index=False, encoding="utf-8-sig")
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "factor-test.md").write_text(render_report(outputs), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the factor effectiveness tests.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_factor_test_config(args.config))
    print(f"Built {len(outputs.members):,} quantile-member rows and {len(outputs.ic_series):,} Rank IC observations.")


if __name__ == "__main__":
    main()
