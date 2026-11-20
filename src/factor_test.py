"""Test factors with forward returns, quantiles, and Rank IC."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

FACTORS = ("pe", "momentum", "roe")


@dataclass(frozen=True)
class FactorTestConfig:
    processed_dir: Path
    quantile_groups: int
    min_ic_pairs: int


@dataclass(frozen=True)
class FactorTestOutputs:
    forward_returns: pd.DataFrame
    details: pd.DataFrame
    members: pd.DataFrame
    quantile_returns: pd.DataFrame
    ic_series: pd.DataFrame
    summary: pd.DataFrame


def load_factor_test_config(config_path: Path) -> FactorTestConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    return FactorTestConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        quantile_groups=int(settings["factors"]["quantile_groups"]),
        min_ic_pairs=int(settings["factor_test"]["min_ic_pairs"]),
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
    """Create one auditable forward-return record for every Stage 4 factor row."""
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
        raise ValueError("No valid daily returns were available for Stage 5.")
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


def build_factor_test_details(factor_details: pd.DataFrame, forward_returns: pd.DataFrame,
                              quantile_groups: int) -> pd.DataFrame:
    """Expand common forward returns into factor-specific eligibility and groups."""
    frames = []
    for factor in FACTORS:
        frame = factor_details[["signal_date", "stock_code", f"{factor}_score", f"{factor}_is_rankable", f"{factor}_unrankable_reason"]].copy()
        frame = frame.rename(columns={f"{factor}_score": "factor_score", f"{factor}_is_rankable": "factor_is_rankable", f"{factor}_unrankable_reason": "factor_unrankable_reason"})
        frame["factor"] = factor
        frame = frame.merge(forward_returns, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
        factor_rankable = frame["factor_is_rankable"].fillna(False).astype(bool)
        forward_eligible = frame["forward_eligible"].fillna(False).astype(bool)
        frame["test_eligible"] = factor_rankable & forward_eligible
        frame["test_unavailable_reason"] = frame["forward_unavailable_reason"].astype("string")
        missing_factor = frame["test_unavailable_reason"].isna() & ~factor_rankable
        frame.loc[missing_factor, "test_unavailable_reason"] = "factor_unrankable:" + frame.loc[missing_factor, "factor_unrankable_reason"].fillna("unknown")
        frame["quantile"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        for _, indexes in frame.loc[frame["test_eligible"]].groupby("signal_date").groups.items():
            frame.loc[indexes, "quantile"] = assign_quantiles(frame.loc[indexes, "factor_score"], quantile_groups)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["factor", "signal_date", "stock_code"]).reset_index(drop=True)


def build_quantile_returns(details: pd.DataFrame, quantile_groups: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = details.loc[details["quantile"].notna()].copy()
    grouped = members.groupby(["factor", "signal_date", "quantile"], as_index=False).agg(
        stock_count=("stock_code", "size"),
        equal_weight_forward_return=("forward_return", "mean"),
        extreme_return_stock_count=("extreme_return_days", lambda values: int(values.gt(0).sum())),
        early_exit_or_data_cutoff_count=("early_exit_or_data_cutoff", "sum"),
    )
    periods = details.loc[details["test_eligible"], ["factor", "signal_date"]].drop_duplicates()
    grid = periods.merge(pd.DataFrame({"quantile": range(1, quantile_groups + 1)}), how="cross")
    output = grid.merge(grouped, on=["factor", "signal_date", "quantile"], how="left", validate="one_to_one")
    output["stock_count"] = output["stock_count"].fillna(0).astype("int64")
    output["empty_group"] = output["stock_count"].eq(0)
    output = output.sort_values(["factor", "quantile", "signal_date"]).reset_index(drop=True)
    output["cumulative_group_return"] = output.groupby(["factor", "quantile"])["equal_weight_forward_return"].transform(lambda values: (1 + values.fillna(0)).cumprod() - 1)
    wide = output.pivot(index=["factor", "signal_date"], columns="quantile", values="equal_weight_forward_return")
    spread = wide.get(quantile_groups).sub(wide.get(1)).rename("q_high_minus_q_low").reset_index()
    spread = spread.sort_values(["factor", "signal_date"]).reset_index(drop=True)
    spread["cumulative_q_high_minus_q_low"] = (
        spread.groupby("factor")["q_high_minus_q_low"]
        .transform(lambda values: values.fillna(0).cumsum())
    )
    output = output.merge(spread, on=["factor", "signal_date"], how="left")
    return members, output.sort_values(["factor", "signal_date", "quantile"]).reset_index(drop=True)


def build_ic_series(details: pd.DataFrame, min_ic_pairs: int) -> pd.DataFrame:
    rows = []
    for (factor, signal_date), group in details.loc[details["test_eligible"]].groupby(["factor", "signal_date"], sort=True):
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
        rows.append({"factor": factor, "signal_date": signal_date, "ic_pair_count": pair_count, "rank_ic": ic, "ic_unavailable_reason": reason})
    return pd.DataFrame(rows)


def build_summary(details: pd.DataFrame, quantile_returns: pd.DataFrame, ic_series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        item = details.loc[details["factor"].eq(factor)]
        ic = ic_series.loc[ic_series["factor"].eq(factor), "rank_ic"].dropna()
        groups = quantile_returns.loc[quantile_returns["factor"].eq(factor)]
        rows.append({"factor": factor, "signal_periods": int(item["signal_date"].nunique()), "test_eligible_rows": int(item["test_eligible"].sum()), "no_future_holding_period_rows": int(item["test_unavailable_reason"].eq("no_future_holding_period").sum()), "missing_first_holding_day_rows": int(item["test_unavailable_reason"].eq("missing_valid_first_holding_day").sum()), "early_exit_or_data_cutoff_rows": int(item.loc[item["test_eligible"], "early_exit_or_data_cutoff"].sum()), "extreme_return_stock_rows": int(item.loc[item["test_eligible"], "extreme_return_days"].gt(0).sum()), "empty_group_rows": int(groups["empty_group"].sum()), "ic_valid_periods": int(ic.size), "ic_mean": ic.mean() if not ic.empty else pd.NA, "ic_win_rate": ic.gt(0).mean() if not ic.empty else pd.NA})
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
    if groups.duplicated(["factor", "signal_date", "quantile"]).any():
        raise ValueError("Quantile results have duplicate group keys.")
    if (ic["rank_ic"].notna() & ic["ic_pair_count"].lt(min_ic_pairs)).any():
        raise ValueError("Rank IC was calculated below the configured minimum sample size.")


def render_report(outputs: FactorTestOutputs) -> str:
    lines = ["# 因子有效性检验", "", "本报告检验因子分数对下一月收益的区分能力，不代表实际开盘成交后的策略收益。", "", "## 汇总", "", "| 因子 | 可检验股票记录 | 有效 IC 期数 | IC 均值 | IC 胜率 | 首日缺失 | 提前退出/数据截断 | 极端收益股票 | 空组 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in outputs.summary.itertuples(index=False):
        mean = "—" if pd.isna(row.ic_mean) else f"{row.ic_mean:.4f}"
        win = "—" if pd.isna(row.ic_win_rate) else f"{row.ic_win_rate:.4f}"
        lines.append(f"| {row.factor} | {row.test_eligible_rows} | {row.ic_valid_periods} | {mean} | {win} | {row.missing_first_holding_day_rows} | {row.early_exit_or_data_cutoff_rows} | {row.extreme_return_stock_rows} | {row.empty_group_rows} |")
    lines.extend(["", "## 口径限制", "", "- 2025-12-31 没有后续持有期数据，未进入分组收益或 IC。", "- Momentum 在 2014-09 至 2015-02 没有可用六个月历史价格，未进行分组或 IC。", "- PE 按交易日 `PE1TTM` 可用的既定口径处理；该限制仅在最终报告中说明。", ""])
    return "\n".join(lines)


def run(config: FactorTestConfig) -> FactorTestOutputs:
    directory = config.processed_dir
    factor_details = pd.read_parquet(directory / "factor_details.parquet")
    forward = build_forward_returns_streaming(factor_details, directory / "daily_trading.parquet")
    details = build_factor_test_details(factor_details, forward, config.quantile_groups)
    members, quantile_returns = build_quantile_returns(details, config.quantile_groups)
    ic_series = build_ic_series(details, config.min_ic_pairs)
    summary = build_summary(details, quantile_returns, ic_series)
    outputs = FactorTestOutputs(forward, details, members, quantile_returns, ic_series, summary)
    validate_outputs(outputs, config.quantile_groups, config.min_ic_pairs)
    forward.to_parquet(directory / "factor_forward_returns.parquet", index=False)
    details.to_parquet(directory / "factor_test_details.parquet", index=False)
    members.to_parquet(directory / "factor_quantile_members.parquet", index=False)
    quantile_returns.to_csv(directory / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")
    ic_series.to_csv(directory / "factor_ic_series.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(directory / "factor_test_summary.csv", index=False, encoding="utf-8-sig")
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
