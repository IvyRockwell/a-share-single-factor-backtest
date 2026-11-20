"""Calculate the PE, Momentum, and ROE factor inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class FactorConfig:
    """Locations and the confirmed six-calendar-month Momentum window."""

    processed_dir: Path
    momentum_lookback_months: int


@dataclass(frozen=True)
class FactorOutputs:
    details: pd.DataFrame
    quality: pd.DataFrame


def load_factor_config(config_path: Path) -> FactorConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    return FactorConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        momentum_lookback_months=int(settings["factors"]["momentum_lookback_months"]),
    )


def _rank_factor(frame: pd.DataFrame, raw_column: str, rankable_column: str, score_column: str,
                 rank_column: str, higher_is_better: bool) -> None:
    """Give every valid cross-section a 1-based rank and a comparable (0, 1] score."""
    frame[rank_column] = pd.NA
    frame[score_column] = pd.NA
    valid = frame[rankable_column]
    ranked = frame.loc[valid].groupby("signal_date")[raw_column].rank(
        method="average", ascending=not higher_is_better
    )
    counts = frame.loc[valid].groupby("signal_date")[raw_column].transform("size")
    frame.loc[valid, rank_column] = ranked.astype("Float64")
    frame.loc[valid, score_column] = ((counts - ranked + 1) / counts).astype("Float64")


def _reason_from_conditions(frame: pd.DataFrame, conditions: list[tuple[str, pd.Series]]) -> pd.Series:
    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name, condition in conditions:
        reason = reason.mask(reason.isna() & condition.fillna(False), name)
    return reason


def add_pe_factor(pool: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    values = valuation[["stock_code", "trading_date", "pe_ttm", "quality_flags"]].copy()
    if values.duplicated(["stock_code", "trading_date"]).any():
        raise ValueError("Valuation data contains duplicate (stock_code, trading_date) keys.")
    values = values.rename(columns={"trading_date": "signal_date", "quality_flags": "pe_quality_flags"})
    result = pool.merge(values, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    invalid = result["pe_quality_flags"].fillna("").str.contains("invalid_pe_ttm")
    result["pe_is_rankable"] = result["pe_ttm"].gt(0) & ~invalid
    result["pe_unrankable_reason"] = _reason_from_conditions(result, [
        ("missing_or_invalid_pe_ttm", result["pe_ttm"].isna() | invalid),
        ("nonpositive_pe_ttm", result["pe_ttm"].le(0)),
    ])
    _rank_factor(result, "pe_ttm", "pe_is_rankable", "pe_score", "pe_rank", higher_is_better=False)
    return result.drop(columns=["pe_quality_flags"])


def add_momentum_factor(pool: pd.DataFrame, daily: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    prices = daily[["stock_code", "trade_date", "adjusted_close", "quality_flags"]].copy()
    if prices.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Daily trading data contains duplicate (stock_code, trade_date) keys.")
    prices["price_is_invalid"] = prices["quality_flags"].fillna("").str.contains("invalid_adjusted_close")
    signal_prices = prices.rename(columns={
        "trade_date": "signal_date", "adjusted_close": "momentum_signal_adjusted_close",
        "price_is_invalid": "momentum_signal_price_invalid",
    })[["stock_code", "signal_date", "momentum_signal_adjusted_close", "momentum_signal_price_invalid"]]
    result = pool.merge(signal_prices, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    result["momentum_target_date"] = result["signal_date"] - pd.DateOffset(months=lookback_months)

    valid_starts = prices.loc[
        prices["adjusted_close"].gt(0) & ~prices["price_is_invalid"],
        ["stock_code", "trade_date", "adjusted_close"],
    ].rename(columns={"trade_date": "momentum_start_date", "adjusted_close": "momentum_start_adjusted_close"})
    left = result[["stock_code", "momentum_target_date"]].sort_values("momentum_target_date")
    right = valid_starts.sort_values("momentum_start_date")
    starts = pd.merge_asof(
        left, right, left_on="momentum_target_date", right_on="momentum_start_date",
        by="stock_code", direction="backward", allow_exact_matches=True,
    )
    result = result.join(starts[["momentum_start_date", "momentum_start_adjusted_close"]])
    signal_invalid = result["momentum_signal_price_invalid"].fillna(False)
    result["momentum_6m_return"] = (
        result["momentum_signal_adjusted_close"] / result["momentum_start_adjusted_close"] - 1
    )
    result["momentum_is_rankable"] = (
        result["momentum_signal_adjusted_close"].gt(0)
        & ~signal_invalid
        & result["momentum_start_adjusted_close"].gt(0)
    )
    result["momentum_unrankable_reason"] = _reason_from_conditions(result, [
        ("missing_or_invalid_signal_adjusted_close", result["momentum_signal_adjusted_close"].isna() | signal_invalid),
        ("nonpositive_signal_adjusted_close", result["momentum_signal_adjusted_close"].le(0)),
        ("no_valid_price_on_or_before_lookback_date", result["momentum_start_adjusted_close"].isna()),
    ])
    _rank_factor(result, "momentum_6m_return", "momentum_is_rankable", "momentum_score", "momentum_rank", higher_is_better=True)
    return result.drop(columns=["momentum_signal_price_invalid"])


def add_roe_factor(pool: pd.DataFrame, financials: pd.DataFrame) -> pd.DataFrame:
    reports = financials.loc[
        financials["Typrep"].eq("A")
        & financials["report_date"].dt.month.eq(12)
        & financials["report_date"].dt.day.eq(31),
        ["stock_code", "report_date", "actual_disclosure_date", "net_profit", "shareholder_equity", "invalid_actual_disclosure_date"],
    ].copy()
    reports = reports.loc[
        reports["actual_disclosure_date"].notna() & ~reports["invalid_actual_disclosure_date"].fillna(False)
    ].drop(columns=["invalid_actual_disclosure_date"])
    left = pool[["signal_date", "stock_code"]].sort_values("signal_date")
    right = reports.sort_values(["actual_disclosure_date", "report_date"])
    selected = pd.merge_asof(
        left, right, left_on="signal_date", right_on="actual_disclosure_date", by="stock_code",
        direction="backward", allow_exact_matches=False,
    )
    result = pool.merge(
        selected, on=["signal_date", "stock_code"], how="left", validate="one_to_one"
    ).rename(columns={"report_date": "roe_report_date", "actual_disclosure_date": "roe_actual_disclosure_date"})
    result["roe"] = result["net_profit"] / result["shareholder_equity"]
    result["roe_is_rankable"] = (
        result["roe_actual_disclosure_date"].notna()
        & result["net_profit"].notna()
        & result["shareholder_equity"].gt(0)
    )
    result["roe_unrankable_reason"] = _reason_from_conditions(result, [
        ("no_prior_annual_consolidated_disclosure", result["roe_actual_disclosure_date"].isna()),
        ("missing_net_profit", result["roe_actual_disclosure_date"].notna() & result["net_profit"].isna()),
        ("missing_or_nonpositive_shareholder_equity", result["roe_actual_disclosure_date"].notna() & (result["shareholder_equity"].isna() | result["shareholder_equity"].le(0))),
    ])
    _rank_factor(result, "roe", "roe_is_rankable", "roe_score", "roe_rank", higher_is_better=True)
    return result


def build_factor_details(pool: pd.DataFrame, valuation: pd.DataFrame, daily: pd.DataFrame,
                         financials: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    """Compute each factor independently for every pool row."""
    if pool.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Historical universe contains duplicate (signal_date, stock_code) keys.")
    result = add_pe_factor(pool[["signal_date", "stock_code"]].copy(), valuation)
    result = add_momentum_factor(result, daily, config.momentum_lookback_months)
    return add_roe_factor(result, financials).sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


FACTOR_COLUMNS = {
    "pe": ("pe_ttm", "pe_is_rankable", "pe_unrankable_reason"),
    "momentum": ("momentum_6m_return", "momentum_is_rankable", "momentum_unrankable_reason"),
    "roe": ("roe", "roe_is_rankable", "roe_unrankable_reason"),
}


def build_quality_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_date, period in details.groupby("signal_date", sort=True):
        for factor, (raw, rankable, reason) in FACTOR_COLUMNS.items():
            reasons = period.loc[~period[rankable], reason].dropna().value_counts().sort_index().to_dict()
            valid = period.loc[period[rankable], raw]
            row = {
                "signal_date": signal_date, "factor": factor, "pool_count": len(period),
                "rankable_count": int(period[rankable].sum()), "unrankable_count": int((~period[rankable]).sum()),
                "unrankable_by_reason": json.dumps({str(key): int(value) for key, value in reasons.items()}, ensure_ascii=False, sort_keys=True),
                "raw_min": valid.min() if not valid.empty else pd.NA,
                "raw_max": valid.max() if not valid.empty else pd.NA,
            }
            if factor == "pe":
                positive = period.loc[period["pe_ttm"].gt(0), "pe_ttm"]
                row["positive_pe_p99"] = positive.quantile(0.99) if not positive.empty else pd.NA
                row["positive_pe_max"] = positive.max() if not positive.empty else pd.NA
            else:
                row["positive_pe_p99"] = pd.NA
                row["positive_pe_max"] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


def validate_outputs(outputs: FactorOutputs, pool: pd.DataFrame) -> None:
    details, quality = outputs.details, outputs.quality
    if len(details) != len(pool) or details.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Factor details do not have exactly one row per historical-universe key.")
    for factor, (raw, rankable, _) in FACTOR_COLUMNS.items():
        rank, score = f"{factor}_rank", f"{factor}_score"
        if details.loc[details[rankable], raw].isna().any() or details.loc[details[rankable], rank].isna().any() or details.loc[details[rankable], score].isna().any():
            raise ValueError(f"Rankable {factor} rows are missing a raw value, rank, or score.")
        if details.loc[~details[rankable], [rank, score]].notna().any().any():
            raise ValueError(f"Unrankable {factor} rows received a rank or score.")
    expected = len(pool["signal_date"].unique()) * len(FACTOR_COLUMNS)
    if len(quality) != expected or (quality["pool_count"] != quality["rankable_count"] + quality["unrankable_count"]).any():
        raise ValueError("Factor quality summary does not reconcile to the detail output.")
    if (details.loc[details["momentum_is_rankable"], "momentum_start_date"] > details.loc[details["momentum_is_rankable"], "momentum_target_date"]).any():
        raise ValueError("Momentum used a price after its six-month lookback target date.")
    if (details.loc[details["roe_is_rankable"], "roe_actual_disclosure_date"] >= details.loc[details["roe_is_rankable"], "signal_date"]).any():
        raise ValueError("ROE used a report disclosed on or after its signal date.")


def render_report(outputs: FactorOutputs) -> str:
    details, quality = outputs.details, outputs.quality
    lines = [
        "# PE / Momentum / ROE 因子", "", "## 已执行规则", "",
        "- 仅在月末历史股票池内计算；每个因子分别保留有效样本及不可排名原因。",
        "- PE 使用信号日 `pe_ttm` 的正值，低 PE 得到更高分数。",
        "- Momentum 使用信号日与不晚于六个自然月前目标日的最近有效 `adjusted_close`，高收益得到更高分数。",
        "- ROE 使用信号日前严格已披露的最新合并年度报告（同披露日取最新报告期），高 ROE 得到更高分数。",
        "- 分数为当期有效样本内的反向名次比例，最高看好为 `1.0`；同值采用平均名次。", "",
        "## 重要限制", "",
        "- 本项目按既定规则将交易日 `PE1TTM` 视作当日可用；原始数据无法复核财报期、披露时点、报表版本和历史股本生效时间。因此 PE 结果不能表述为已严格消除前视偏差。", "",
        "## 按期质量统计", "", "| 因子 | 信号日数 | 平均股票池 | 平均有效样本 | 平均不可排名 |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for factor, group in quality.groupby("factor", sort=False):
        lines.append(f"| {factor} | {len(group):,} | {group['pool_count'].mean():,.1f} | {group['rankable_count'].mean():,.1f} | {group['unrankable_count'].mean():,.1f} |")
    lines.extend(["", "### 不可排名原因汇总", "", "| 因子 | 原因 | 记录数 |", "| --- | --- | ---: |"])
    for factor, (_, rankable, reason) in FACTOR_COLUMNS.items():
        counts = details.loc[~details[rankable], reason].dropna().value_counts().sort_index()
        lines.extend(f"| {factor} | `{name}` | {int(count):,} |" for name, count in counts.items())
    pe = quality.loc[quality["factor"].eq("pe")]
    lines.extend([
        "", f"正值 PE 未截尾或删除；各期 99 分位数的最大值为 `{pe['positive_pe_p99'].max():.6g}`，各期最大值中的最大值为 `{pe['positive_pe_max'].max():.6g}`。",
        "完整的逐期极端值统计已写入 `data/processed/factor_period_quality.csv`。", "",
    ])
    return "\n".join(lines)


def run(config: FactorConfig) -> FactorOutputs:
    directory = config.processed_dir
    pool = pd.read_parquet(directory / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    valuation = pd.read_parquet(directory / "valuation.parquet", columns=["stock_code", "trading_date", "pe_ttm", "quality_flags"])
    earliest_target = pool["signal_date"].min() - pd.DateOffset(months=config.momentum_lookback_months)
    daily = pd.read_parquet(
        directory / "daily_trading.parquet", columns=["stock_code", "trade_date", "adjusted_close", "quality_flags"],
        filters=[("trade_date", ">=", earliest_target), ("trade_date", "<=", pool["signal_date"].max())],
    )
    financials = pd.read_parquet(directory / "financial_availability.parquet")
    details = build_factor_details(pool, valuation, daily, financials, config)
    quality = build_quality_summary(details)
    outputs = FactorOutputs(details, quality)
    validate_outputs(outputs, pool)
    details.to_parquet(directory / "factor_details.parquet", index=False)
    quality.to_csv(directory / "factor_period_quality.csv", index=False, encoding="utf-8-sig")
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "factors.md").write_text(render_report(outputs), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calculate the PE, Momentum, and ROE factor data.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_factor_config(args.config))
    print(f"Built {len(outputs.details):,} factor rows across {outputs.quality['signal_date'].nunique()} signal dates.")


if __name__ == "__main__":
    main()
