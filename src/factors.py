"""Calculate the PE, Momentum, and ROE factor inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FACTORS = ("pe", "momentum", "roe")
MIN_NEUTRAL_ROWS = 100


@dataclass(frozen=True)
class MomentumVariant:
    """One momentum definition tested beside the primary factor."""

    name: str
    lookback_months: int
    skip_months: int
    higher_is_better: bool = True


@dataclass(frozen=True)
class FactorConfig:
    """Locations, the Momentum windows, and the neutralization controls."""

    processed_dir: Path
    momentum_lookback_months: int
    momentum_skip_months: int = 1
    momentum_variants: tuple[MomentumVariant, ...] = ()
    neutralize: bool = True
    industry_standard: str = "P0207"
    min_group_size: int = 20
    min_industry_coverage: float = 0.8


@dataclass(frozen=True)
class FactorOutputs:
    details: pd.DataFrame
    quality: pd.DataFrame


def load_factor_config(config_path: Path) -> FactorConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    neutralization = settings.get("neutralization") or {}
    factor_settings = settings["factors"]
    variants = tuple(
        MomentumVariant(
            name=str(item["name"]),
            lookback_months=int(item["lookback_months"]),
            skip_months=int(item.get("skip_months", 1)),
            higher_is_better=bool(item.get("higher_is_better", True)),
        )
        for item in (factor_settings.get("momentum_variants") or [])
    )
    return FactorConfig(
        processed_dir=root / settings["data"]["processed_dir"],
        momentum_lookback_months=int(factor_settings["momentum_lookback_months"]),
        momentum_skip_months=int(factor_settings.get("momentum_skip_months", 1)),
        momentum_variants=variants,
        neutralize=bool(neutralization.get("enabled", True)),
        industry_standard=str(neutralization.get("industry_standard", "P0207")),
        min_group_size=int(neutralization.get("min_group_size", 20)),
        min_industry_coverage=float(neutralization.get("min_industry_coverage", 0.8)),
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


def add_pe_factor(pool: pd.DataFrame, valuation: pd.DataFrame, panel: pd.DataFrame,
                  market_cap: pd.DataFrame | None) -> pd.DataFrame:
    """Rank PE built from the reports that were public before each signal date.

    The vendor ``pe_ttm`` column is kept for comparison only; it prices the latest
    quarter end rather than the latest disclosure and therefore looks ahead.
    """
    values = valuation[["stock_code", "trading_date", "pe_ttm"]].copy()
    if values.duplicated(["stock_code", "trading_date"]).any():
        raise ValueError("Valuation data contains duplicate (stock_code, trading_date) keys.")
    values = values.rename(columns={"trading_date": "signal_date"})
    result = pool.merge(values, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    if market_cap is None:
        result["market_cap"] = np.nan
    else:
        result = result.merge(
            market_cap[["signal_date", "stock_code", "market_cap"]],
            on=["signal_date", "stock_code"], how="left", validate="one_to_one",
        )
    result = attach_pit_pe(result, panel)
    has_report = result["pe_pit_disclosure_date"].notna()
    positive_ttm = result["pe_pit_ttm_net_profit"].gt(0)
    result["pe_is_rankable"] = result["pe_pit"].gt(0)
    result["pe_unrankable_reason"] = _reason_from_conditions(result, [
        ("no_disclosed_report", ~has_report),
        ("nonpositive_or_missing_ttm", has_report & ~positive_ttm),
        ("missing_market_cap", has_report & positive_ttm & ~result["market_cap"].gt(0)),
    ])
    _rank_factor(result, "pe_pit", "pe_is_rankable", "pe_score", "pe_rank", higher_is_better=False)
    return result.drop(columns=["market_cap"])


def add_momentum_factor(pool: pd.DataFrame, daily: pd.DataFrame, lookback_months: int,
                        skip_months: int = 1, name: str = "momentum",
                        higher_is_better: bool = True) -> pd.DataFrame:
    """Add one point-in-time momentum definition to every pool row.

    ``lookback`` is measured from the signal date; ``skip`` excludes the most
    recent month.  The default 12-1 definition therefore uses prices from
    ``signal - 12m`` to ``signal - 1m`` and never touches the last month.
    """
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive.")
    if skip_months < 0 or skip_months >= lookback_months:
        raise ValueError("skip_months must be non-negative and smaller than lookback_months.")
    pool = pool.reset_index(drop=True)
    prices = daily[["stock_code", "trade_date", "adjusted_close", "quality_flags"]].copy()
    if prices.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Daily trading data contains duplicate (stock_code, trade_date) keys.")
    prices["price_is_invalid"] = prices["quality_flags"].fillna("").str.contains("invalid_adjusted_close")
    valid = prices.loc[
        prices["adjusted_close"].gt(0) & ~prices["price_is_invalid"],
        ["stock_code", "trade_date", "adjusted_close"],
    ].rename(columns={"trade_date": "price_date", "adjusted_close": "price"})
    valid = valid.sort_values("price_date")

    left = pool[["stock_code", "signal_date"]].copy()
    left["row_id"] = np.arange(len(left))
    left["start_target"] = left["signal_date"] - pd.DateOffset(months=lookback_months)
    left["end_target"] = left["signal_date"] - pd.DateOffset(months=skip_months) if skip_months else left["signal_date"]
    left = left.sort_values("start_target")
    starts = pd.merge_asof(
        left, valid.rename(columns={"price_date": "start_date", "price": "start_price"}),
        left_on="start_target", right_on="start_date", by="stock_code",
        direction="backward", allow_exact_matches=True,
    )
    starts = starts.sort_values("end_target")
    ends = pd.merge_asof(
        starts, valid.rename(columns={"price_date": "end_date", "price": "end_price"}),
        left_on="end_target", right_on="end_date", by="stock_code",
        direction="backward", allow_exact_matches=True,
    )
    result = pool.copy()
    result["row_id"] = np.arange(len(result))
    result = result.merge(
        ends[["row_id", "start_target", "end_target", "start_date", "start_price", "end_date", "end_price"]],
        on="row_id", how="left", validate="one_to_one",
    ).set_index("row_id", drop=True)
    result = result.sort_index()
    result[f"{name}_start_target"] = result["start_target"]
    result[f"{name}_end_target"] = result["end_target"]
    result[f"{name}_start_date"] = result["start_date"]
    result[f"{name}_end_date"] = result["end_date"]
    result[f"{name}_start_price"] = result["start_price"]
    result[f"{name}_end_price"] = result["end_price"]
    result[f"{name}_return"] = result["end_price"] / result["start_price"] - 1
    result[f"{name}_is_rankable"] = (
        result["start_price"].gt(0)
        & result["end_price"].gt(0)
        & result["start_date"].notna()
        & result["end_date"].notna()
        & result["start_date"].lt(result["end_date"])
    )
    result[f"{name}_unrankable_reason"] = _reason_from_conditions(result, [
        ("no_valid_price_on_or_before_start_date", result["start_price"].isna()),
        ("no_valid_price_on_or_before_end_date", result["end_price"].isna()),
        ("nonpositive_start_price", result["start_price"].le(0)),
        ("nonpositive_end_price", result["end_price"].le(0)),
        ("end_date_not_after_start_date", result["start_date"].notna() & result["end_date"].notna() & result["end_date"].le(result["start_date"])),
    ])
    _rank_factor(
        result, f"{name}_return", f"{name}_is_rankable",
        f"{name}_score", f"{name}_rank", higher_is_better=higher_is_better,
    )
    return result.drop(columns=[
        "start_target", "end_target", "start_date", "start_price", "end_date", "end_price",
    ])


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


def attach_exposures(result: pd.DataFrame, market_cap: pd.DataFrame, industry: pd.DataFrame | None,
                     industry_standard: str) -> pd.DataFrame:
    """Attach the size control and, when present, the industry effective on the signal date."""
    columns = ["signal_date", "stock_code", "market_cap", "log_market_cap", "has_market_cap"]
    frame = result.merge(market_cap[columns], on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    frame["industry_code"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    if industry is None or industry.empty:
        return frame
    selected = industry.loc[
        industry["industry_standard"].eq(industry_standard), ["stock_code", "effective_date", "industry_code"]
    ]
    if selected.empty:
        return frame
    right = selected.assign(
        effective_date=lambda item: pd.to_datetime(item["effective_date"]).astype("datetime64[ns]")
    ).sort_values("effective_date")
    left = frame.drop(columns="industry_code").assign(
        signal_date=lambda item: pd.to_datetime(item["signal_date"]).astype("datetime64[ns]")
    )
    merged = pd.merge_asof(
        left, right,
        left_on="signal_date", right_on="effective_date", by="stock_code", direction="backward",
    )
    return merged.drop(columns="effective_date")


def _residual_rank(period: pd.DataFrame, score_column: str, use_industry: bool,
                   min_group_size: int) -> tuple[pd.Series, pd.Series]:
    """Regress the score on log size and industry dummies, then rank the residual."""
    score = period[score_column].astype("Float64").astype("float64")
    size = period["log_market_cap"].astype("Float64").astype("float64")
    usable = score.notna() & size.notna()
    if use_industry:
        usable = usable & period["industry_code"].notna()
    ranked = pd.Series(np.nan, index=period.index, dtype="float64")
    if int(usable.sum()) < MIN_NEUTRAL_ROWS:
        return ranked, pd.Series(False, index=period.index)

    target = score.loc[usable]
    spread = float(target.std(ddof=0))
    target = (target - target.mean()) / spread if spread > 0 else target * 0.0
    blocks = [np.ones((usable.sum(), 1)), size.loc[usable].to_numpy().reshape(-1, 1)]
    if use_industry:
        codes = period.loc[usable, "industry_code"].astype(str)
        counts = codes.value_counts()
        codes = codes.where(~codes.isin(counts[counts < min_group_size].index), "OTHER")
        dummies = pd.get_dummies(codes, drop_first=True, dtype="float64")
        if dummies.shape[1]:
            blocks.append(dummies.to_numpy())
    design = np.hstack(blocks)
    beta, *_ = np.linalg.lstsq(design, target.to_numpy(), rcond=None)
    residual = target.to_numpy() - design @ beta
    values = pd.Series(residual).rank(method="average").to_numpy()
    ranked.loc[usable] = values / len(values)
    return ranked, pd.Series(usable, index=period.index)


def neutralize_scores(details: pd.DataFrame, use_industry: bool, min_group_size: int,
                      control_set: str) -> pd.DataFrame:
    """Add a neutralized score for each factor, one cross-section at a time."""
    result = details.copy()
    result["neutral_control"] = control_set
    for factor in FACTORS:
        neutral = pd.Series(np.nan, index=result.index, dtype="float64")
        eligible = pd.Series(False, index=result.index)
        for _, period in result.groupby("signal_date", sort=False):
            values, usable = _residual_rank(period, f"{factor}_score", use_industry, min_group_size)
            neutral.loc[period.index] = values
            eligible.loc[period.index] = usable
        result[f"{factor}_score_neutral"] = neutral
        result[f"{factor}_neutral_eligible"] = eligible
    return result


def build_neutralization_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_date, period in details.groupby("signal_date", sort=True):
        row = {
            "signal_date": signal_date,
            "control_set": period["neutral_control"].iloc[0],
            "pool_count": len(period),
            "size_coverage": float(period["log_market_cap"].notna().mean()),
            "industry_coverage": float(period["industry_code"].notna().mean()),
        }
        for factor in FACTORS:
            row[f"{factor}_neutral_count"] = int(period[f"{factor}_neutral_eligible"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_factor_details(pool: pd.DataFrame, valuation: pd.DataFrame, daily: pd.DataFrame,
                         financials: pd.DataFrame, config: FactorConfig,
                         market_cap: pd.DataFrame | None = None,
                         industry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute each factor independently for every pool row."""
    if pool.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Historical universe contains duplicate (signal_date, stock_code) keys.")
    # Parquet round-trips can hand back microsecond resolution, which merge_asof refuses to mix.
    pool = pool.assign(signal_date=pd.to_datetime(pool["signal_date"]).astype("datetime64[ns]"))
    daily = daily.assign(trade_date=pd.to_datetime(daily["trade_date"]).astype("datetime64[ns]"))
    financials = financials.assign(
        report_date=pd.to_datetime(financials["report_date"]).astype("datetime64[ns]"),
        actual_disclosure_date=pd.to_datetime(financials["actual_disclosure_date"]).astype("datetime64[ns]"),
    )
    if market_cap is not None:
        market_cap = market_cap.assign(signal_date=pd.to_datetime(market_cap["signal_date"]).astype("datetime64[ns]"))
    if industry is not None:
        industry = industry.assign(effective_date=pd.to_datetime(industry["effective_date"]).astype("datetime64[ns]"))
    ttm = build_ttm_net_profit(financials)
    panel = build_pit_panel(pool[["signal_date", "stock_code"]], ttm)
    result = add_pe_factor(pool[["signal_date", "stock_code"]].copy(), valuation, panel, market_cap)
    result = add_momentum_factor(
        result, daily, config.momentum_lookback_months, config.momentum_skip_months,
        "momentum", True,
    )
    for variant in config.momentum_variants:
        result = add_momentum_factor(
            result, daily, variant.lookback_months, variant.skip_months,
            variant.name, variant.higher_is_better,
        )
    result = add_roe_factor(result, financials)
    if market_cap is not None:
        result = attach_exposures(result, market_cap, industry, config.industry_standard)
        coverage = float(result["industry_code"].notna().mean())
        use_industry = config.neutralize and coverage >= config.min_industry_coverage
        if config.neutralize:
            control_set = f"industry({config.industry_standard})+size" if use_industry else "size"
            result = neutralize_scores(result, use_industry, config.min_group_size, control_set)
    return result.sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


FACTOR_COLUMNS = {
    "pe": ("pe_pit", "pe_is_rankable", "pe_unrankable_reason"),
    "momentum": ("momentum_return", "momentum_is_rankable", "momentum_unrankable_reason"),
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
                positive = period.loc[period[raw].gt(0), raw]
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
    if "neutral_control" in details:
        for factor in FACTORS:
            neutral, eligible = details[f"{factor}_score_neutral"], details[f"{factor}_neutral_eligible"]
            if neutral[eligible].isna().any() or neutral[~eligible].notna().any():
                raise ValueError(f"Neutral {factor} scores do not match their eligibility flag.")
            if neutral[eligible].le(0).any() or neutral[eligible].gt(1).any():
                raise ValueError(f"Neutral {factor} scores do not fall in (0, 1].")
    momentum = details.loc[details["momentum_is_rankable"]]
    if (momentum["momentum_start_date"] >= momentum["momentum_end_date"]).any():
        raise ValueError("Momentum start price is not before its end price.")
    if (momentum["momentum_start_date"] > momentum["momentum_start_target"]).any():
        raise ValueError("Momentum used a price after its start target date.")
    if (momentum["momentum_end_date"] > momentum["momentum_end_target"]).any():
        raise ValueError("Momentum used a price after its end target date.")
    if (details.loc[details["roe_is_rankable"], "roe_actual_disclosure_date"] >= details.loc[details["roe_is_rankable"], "signal_date"]).any():
        raise ValueError("ROE used a report disclosed on or after its signal date.")


def render_report(outputs: FactorOutputs) -> str:
    details, quality = outputs.details, outputs.quality
    lines = [
        "# PE / Momentum / ROE 因子", "", "## 已执行规则", "",
        "- 仅在月末历史股票池内计算；每个因子分别保留有效样本及不可排名原因。",
        "- PE 使用按实际披露日重建的 TTM 净利润（`pe_pit`）：只取信号日前严格已披露的最新报告，年报直接取净利润，季报用「本期 + 上年年报 − 上年同期」拼出 TTM；净利润与市值均为正时参与排名，低 PE 得到更高分数。",
        "- Momentum 主因子为 12-1（信号日前 12 个月到前 1 个月）；6-1 与 1M 反转作为变体并列检验，1M 反转是低收益得高分。",
        "- ROE 使用信号日前严格已披露的最新合并年度报告（同披露日取最新报告期），高 ROE 得到更高分数。",
        "- 分数为当期有效样本内的反向名次比例，最高看好为 `1.0`；同值采用平均名次。", "",
        "## 重要限制", "",
        "- 供应商 `PE1TTM` 实测按最近一个季度末而非最近一份已披露报告计算，在披露窗口内含有前视信息；本因子已改用 `pe_pit`，逐月对照写入 `data/processed/pe_pit_check.csv`。`pe_ttm` 列仅保留作对照，不参与排名。", "",
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


PIT_PE_MIN_OVERLAP = 0.90


def build_ttm_net_profit(financials: pd.DataFrame) -> pd.DataFrame:
    """Rebuild trailing-twelve-month profit from disclosure-timed reports."""
    reports = financials.loc[
        financials["Typrep"].eq("A")
        & financials["net_profit"].notna()
        & financials["actual_disclosure_date"].notna()
        & ~financials["invalid_actual_disclosure_date"].fillna(False),
        ["stock_code", "report_date", "actual_disclosure_date", "net_profit"],
    ].copy()
    reports["report_date"] = pd.to_datetime(reports["report_date"]).astype("datetime64[ns]")
    reports["actual_disclosure_date"] = pd.to_datetime(reports["actual_disclosure_date"]).astype("datetime64[ns]")
    reports = reports.sort_values("actual_disclosure_date").drop_duplicates(
        ["stock_code", "report_date"], keep="last"
    )
    reports["year"] = reports["report_date"].dt.year
    reports["month"] = reports["report_date"].dt.month

    annual = reports.loc[reports["month"].eq(12), ["stock_code", "year", "net_profit"]]
    prior_annual = annual.assign(year=annual["year"] + 1).rename(columns={"net_profit": "prior_annual_net_profit"})
    prior_same = (
        reports[["stock_code", "year", "month", "net_profit"]]
        .assign(year=lambda item: item["year"] + 1)
        .rename(columns={"net_profit": "prior_same_net_profit"})
    )
    frame = reports.merge(prior_same, on=["stock_code", "year", "month"], how="left")
    frame = frame.merge(prior_annual, on=["stock_code", "year"], how="left")
    quarterly = frame["month"].ne(12)
    frame["ttm_net_profit"] = np.where(
        quarterly,
        frame["net_profit"] + frame["prior_annual_net_profit"] - frame["prior_same_net_profit"],
        frame["net_profit"],
    )
    frame.loc[quarterly & (frame["prior_annual_net_profit"].isna() | frame["prior_same_net_profit"].isna()),
              "ttm_net_profit"] = np.nan
    return frame[["stock_code", "report_date", "actual_disclosure_date", "ttm_net_profit"]].sort_values(
        ["actual_disclosure_date", "stock_code"]).reset_index(drop=True)


def build_pit_panel(pool: pd.DataFrame, ttm: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest report period disclosed strictly before each signal date."""
    reports = ttm.sort_values(["stock_code", "report_date", "actual_disclosure_date"])
    frames = []
    for day in pd.Series(pd.to_datetime(pool["signal_date"]).unique()).sort_values():
        visible = reports.loc[reports["actual_disclosure_date"] < day]
        if visible.empty:
            continue
        latest = visible.drop_duplicates("stock_code", keep="last").assign(signal_date=day)
        frames.append(latest)
    if not frames:
        return pd.DataFrame(columns=["signal_date", "stock_code", "report_date", "actual_disclosure_date",
                                     "ttm_net_profit"])
    panel = pd.concat(frames, ignore_index=True)
    return panel.rename(columns={"report_date": "pe_pit_report_period",
                                 "actual_disclosure_date": "pe_pit_disclosure_date"})


def attach_pit_pe(details: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add the disclosure-timed PE and the inputs it was built from."""
    columns = ["signal_date", "stock_code", "ttm_net_profit", "pe_pit_report_period", "pe_pit_disclosure_date"]
    merged = details.merge(panel[columns], on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    merged["pe_pit_ttm_net_profit"] = merged["ttm_net_profit"]
    positive = merged["pe_pit_ttm_net_profit"].gt(0) & merged["market_cap"].gt(0)
    merged["pe_pit"] = np.where(positive, merged["market_cap"] / merged["pe_pit_ttm_net_profit"], np.nan)
    return merged.drop(columns=["ttm_net_profit"])


PIT_AHEAD_TOLERANCE = 0.01
STOCK_MONTH_COLUMNS = [
    "signal_date", "stock_code", "pe_pit", "pe_ttm", "pit_available", "vendor_available",
    "relative_difference", "vendor_ahead_of_disclosure", "flag_reason",
]
FLAG_REASONS = ("pit_missing", "valuation_mismatch_gt_1pct", "matched", "vendor_missing")


def flag_vendor_ahead(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark the stock-months where the vendor PE ran ahead of disclosure.

    ``pit_available`` means the rebuilt PE is positive and therefore rankable;
    ``vendor_available`` is the same test on the supplier field.  A stock-month
    is flagged when the supplier is usable *and* either the rebuild has no usable
    PE (so the supplier is pricing a report that was not public yet) or the two
    values differ by at least :data:`PIT_AHEAD_TOLERANCE`, the same 1% tolerance
    used by the monthly ``share_within_1pct`` diagnostic.
    """
    result = frame.copy()
    result["pit_available"] = result["pe_pit"].gt(0).fillna(False).astype(bool)
    result["vendor_available"] = result["pe_ttm"].gt(0).fillna(False).astype(bool)
    both = result["pit_available"] & result["vendor_available"]
    pit = result["pe_pit"].astype("float64")
    vendor = result["pe_ttm"].astype("float64")
    result["relative_difference"] = np.where(both, (pit - vendor).abs() / vendor, np.nan)
    pit_missing = result["vendor_available"] & ~result["pit_available"]
    mismatch = both & pd.Series(result["relative_difference"]).ge(PIT_AHEAD_TOLERANCE)
    result["vendor_ahead_of_disclosure"] = (pit_missing | mismatch).astype(bool)
    result["flag_reason"] = pd.Series(
        np.select(
            [~result["vendor_available"], pit_missing, mismatch],
            ["vendor_missing", "pit_missing", "valuation_mismatch_gt_1pct"],
            default="matched",
        ),
        index=result.index,
        dtype="string",
    )
    return result


def build_pe_pit_stock_month(details: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """Compare the rebuilt and supplier PE on every stock-month.

    This is the primary look-ahead diagnostic: the monthly table only aggregates
    it.  The result has one row per stock and signal date, with the two PE
    values, their availability flags, and the forward-looking label.
    """
    vendor = valuation[["stock_code", "trading_date", "pe_ttm"]].rename(columns={"trading_date": "signal_date"})
    vendor = vendor.assign(
        signal_date=pd.to_datetime(vendor["signal_date"]).astype("datetime64[ns]"),
        stock_code=vendor["stock_code"].astype("string"),
    )
    left = details[["signal_date", "stock_code", "pe_pit"]].copy()
    left = left.assign(
        signal_date=pd.to_datetime(left["signal_date"]).astype("datetime64[ns]"),
        stock_code=left["stock_code"].astype("string"),
    )
    frame = left.merge(vendor, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    frame = flag_vendor_ahead(frame)
    return frame[STOCK_MONTH_COLUMNS].sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


def build_pe_pit_check(details: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the stock-month comparison into the monthly diagnostic table."""
    frame = build_pe_pit_stock_month(details, valuation)
    frame["within_1pct"] = (
        frame["pit_available"] & frame["vendor_available"]
        & frame["relative_difference"].lt(PIT_AHEAD_TOLERANCE)
    )
    rows = []
    for signal_date, group in frame.groupby("signal_date", sort=True):
        both = group["pit_available"] & group["vendor_available"]
        overlap = int(both.sum())
        differences = group.loc[both, "relative_difference"]
        rows.append({
            "signal_date": signal_date,
            "pool_count": len(group),
            "pit_count": int(group["pit_available"].sum()),
            "csmar_count": int(group["vendor_available"].sum()),
            "overlap": overlap,
            "pit_only": int((group["pit_available"] & ~group["vendor_available"]).sum()),
            "csmar_only": int((~group["pit_available"] & group["vendor_available"]).sum()),
            "share_within_1pct": float(group.loc[both, "within_1pct"].mean()) if overlap else np.nan,
            "median_relative_difference": float(differences.median()) if overlap else np.nan,
            "vendor_ahead_count": int(group["vendor_ahead_of_disclosure"].sum()),
            "vendor_ahead_share": float(group["vendor_ahead_of_disclosure"].mean()) if len(group) else np.nan,
        })
    return pd.DataFrame(rows)


def validate_pe_pit_stock_month(frame: pd.DataFrame) -> None:
    """The stock-month labels must be internally consistent and exhaustive."""
    if frame.empty:
        return
    missing = set(STOCK_MONTH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"The stock-month PIT table is missing columns: {sorted(missing)}")
    if frame.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("The stock-month PIT table contains duplicate keys.")
    if not frame["flag_reason"].isin(FLAG_REASONS).all():
        raise ValueError("A stock-month PIT row has an unknown flag reason.")
    if frame["vendor_ahead_of_disclosure"].ne(
        frame["flag_reason"].isin(["pit_missing", "valuation_mismatch_gt_1pct"])
    ).any():
        raise ValueError("The stock-month look-ahead flag does not match its reason.")


def validate_pe_pit_check(check: pd.DataFrame) -> None:
    """Fail when the rebuilt PE cannot cover the vendor field.

    The two series are *expected* to differ: the rebuild only uses reports that were
    public on the signal date, which is the whole point of it. What must hold is that
    the rebuild reaches (almost) every name the vendor can price.
    """
    coverage = check["overlap"].sum() / max(check["csmar_count"].sum(), 1)
    if coverage < PIT_PE_MIN_OVERLAP:
        raise ValueError(f"Only {coverage:.1%} of the vendor PE can be rebuilt.")


validate_pit_check = validate_pe_pit_check


def run(config: FactorConfig) -> FactorOutputs:
    directory = config.processed_dir
    pool = pd.read_parquet(directory / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    valuation = pd.read_parquet(directory / "valuation.parquet", columns=["stock_code", "trading_date", "pe_ttm"])
    lookbacks = [config.momentum_lookback_months] + [variant.lookback_months for variant in config.momentum_variants]
    earliest_target = pool["signal_date"].min() - pd.DateOffset(months=max(lookbacks))
    daily = pd.read_parquet(
        directory / "daily_trading.parquet", columns=["stock_code", "trade_date", "adjusted_close", "quality_flags"],
        filters=[("trade_date", ">=", earliest_target), ("trade_date", "<=", pool["signal_date"].max())],
    )
    financials = pd.read_parquet(directory / "financial_availability.parquet")
    market_cap_path = directory / "market_cap.parquet"
    market_cap = pd.read_parquet(market_cap_path) if market_cap_path.exists() else None
    industry_path = directory / "industry.parquet"
    industry = pd.read_parquet(industry_path, columns=["stock_code", "industry_standard", "effective_date", "industry_code"]) if industry_path.exists() else None
    details = build_factor_details(pool, valuation, daily, financials, config, market_cap, industry)
    if market_cap is not None:
        stock_month = build_pe_pit_stock_month(details, valuation)
        validate_pe_pit_stock_month(stock_month)
        stock_month.to_csv(directory / "pe_pit_stock_month.csv", index=False, encoding="utf-8-sig")
        check = build_pe_pit_check(details, valuation)
        validate_pit_check(check)
        check.to_csv(directory / "pe_pit_check.csv", index=False, encoding="utf-8-sig")
        flagged = int(stock_month["vendor_ahead_of_disclosure"].sum())
        print(f"PIT PE check: median relative difference {check['median_relative_difference'].median():.2e}, "
              f"{check['share_within_1pct'].mean():.1%} of rows agree within 1%; "
              f"{flagged:,} of {len(stock_month):,} stock-months flagged as vendor look-ahead "
              f"({flagged / max(len(stock_month), 1):.2%}).")
    quality = build_quality_summary(details)
    outputs = FactorOutputs(details, quality)
    validate_outputs(outputs, pool)
    details.to_parquet(directory / "factor_details.parquet", index=False)
    quality.to_csv(directory / "factor_period_quality.csv", index=False, encoding="utf-8-sig")
    if "neutral_control" in details:
        summary = build_neutralization_summary(details)
        summary.to_csv(directory / "neutralization_summary.csv", index=False, encoding="utf-8-sig")
        print(f"Neutralization control set: {details['neutral_control'].iloc[0]}.")
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
