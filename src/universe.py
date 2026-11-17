"""Build the Stage 3 point-in-time historical A-share investment universe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATUS_CONFLICTS = (
    ("600200", pd.Timestamp("2025-12-10"), pd.Timestamp("2025-12-29")),
    ("600701", pd.Timestamp("2021-03-15"), pd.Timestamp("2021-04-26")),
)


@dataclass(frozen=True)
class UniverseConfig:
    processed_dir: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    markets: frozenset[str]
    min_listing_months: int
    normal_trading_status: str


@dataclass(frozen=True)
class UniverseOutputs:
    pool: pd.DataFrame
    period_summary: pd.DataFrame
    exclusions: pd.DataFrame
    signal_dates: pd.DatetimeIndex


def load_universe_config(config_path: Path) -> UniverseConfig:
    """Load Stage 3 parameters recorded in the project configuration."""
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    root = config_path.parent.parent
    data, universe = settings["data"], settings["universe"]
    return UniverseConfig(
        processed_dir=root / data["processed_dir"],
        start_date=pd.Timestamp(data["start_date"]),
        end_date=pd.Timestamp(data["end_date"]),
        markets=frozenset(str(value) for value in universe["markets"]),
        min_listing_months=int(universe["min_listing_months"]),
        normal_trading_status=str(universe["normal_trading_status"]),
    )


def monthly_signal_dates(calendar: pd.DataFrame, config: UniverseConfig) -> pd.DatetimeIndex:
    """Return the last open candidate-market calendar day in every research month."""
    open_days = calendar.loc[
        calendar["market_type"].astype("string").isin(config.markets)
        & calendar["State"].eq("O")
        & calendar["calendar_date"].between(config.start_date, config.end_date),
        "calendar_date",
    ].dropna()
    dates = open_days.groupby(open_days.dt.to_period("M")).max().sort_values()
    if dates.empty:
        raise ValueError("No open candidate-market calendar dates were found in the research period.")
    return pd.DatetimeIndex(dates.to_numpy())


def first_listing_dates(listings: pd.DataFrame) -> pd.DataFrame:
    """Return the verified first LISTINGDATE per code, not a current-status snapshot."""
    frame = listings[["stock_code", "LISTINGDATE"]].copy()
    frame["listing_date"] = pd.to_datetime(frame.pop("LISTINGDATE"), errors="coerce")
    frame = frame.dropna(subset=["stock_code", "listing_date"])
    return frame.groupby("stock_code", as_index=False)["listing_date"].min()


def _status_as_of(panel: pd.DataFrame, status_changes: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest listing-status event effective on or before each signal date."""
    events = status_changes.loc[
        status_changes["ChangedItem"].eq("上市状态"),
        ["stock_code", "event_date", "ValueAfter"],
    ].copy()
    events = events.dropna(subset=["stock_code", "event_date"]).sort_values(["event_date", "stock_code"])
    events = events.rename(columns={"event_date": "listing_status_effective_date", "ValueAfter": "listing_status"})
    left = panel.sort_values(["signal_date", "stock_code"])
    merged = pd.merge_asof(
        left, events, left_on="signal_date", right_on="listing_status_effective_date",
        by="stock_code", direction="backward", allow_exact_matches=True,
    )
    return merged.sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


def _conflict_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for stock_code, start, end in STATUS_CONFLICTS:
        mask |= frame["stock_code"].eq(stock_code) & frame["signal_date"].between(start, end)
    return mask


def _exclusion_reasons(frame: pd.DataFrame, config: UniverseConfig) -> pd.Series:
    has_trading_record = frame["has_signal_day_trading_record"].eq(True)
    conditions = (
        ("missing_signal_day_trading_record", ~has_trading_record),
        ("outside_candidate_market", frame["market_type"].notna() & ~frame["market_type"].isin(config.markets)),
        ("non_normal_trading_status", has_trading_record & ~frame["trading_status"].eq(config.normal_trading_status)),
        ("not_listed_for_six_calendar_months", frame["listing_date"].isna()
         | frame["listing_date"].add(pd.DateOffset(months=config.min_listing_months)).gt(frame["signal_date"])),
        ("status_conflict_interval", _conflict_mask(frame)),
        ("suspended_listing_status", frame["listing_status"].eq("暂停上市")),
        ("delisting_arrangement_status", frame["listing_status"].eq("退市整理期")),
        ("terminated_listing_status", frame["listing_status"].eq("终止上市")),
    )
    result = np.full(len(frame), "", dtype=object)
    for name, condition in conditions:
        active = condition.fillna(False).to_numpy(dtype=bool)
        result[active] = np.where(result[active] == "", name, result[active] + "|" + name)
    return pd.Series(result, index=frame.index, dtype="string")


def build_historical_universe(daily_trading: pd.DataFrame, listings: pd.DataFrame, status_changes: pd.DataFrame,
                              calendar: pd.DataFrame, config: UniverseConfig) -> UniverseOutputs:
    """Apply the confirmed Stage 3 rules to produce point-in-time pool snapshots."""
    signal_dates = monthly_signal_dates(calendar, config)
    panel = (
        pd.DataFrame({"signal_date": signal_dates}).merge(first_listing_dates(listings), how="cross")
        .sort_values(["signal_date", "stock_code"]).reset_index(drop=True)
    )
    signal_trading = daily_trading.loc[
        daily_trading["trade_date"].isin(signal_dates),
        ["stock_code", "trade_date", "market_type", "trading_status", "quality_flags"],
    ].rename(columns={"trade_date": "signal_date", "quality_flags": "trading_quality_flags"})
    signal_trading["has_signal_day_trading_record"] = True
    panel = panel.merge(signal_trading, on=["signal_date", "stock_code"], how="left", validate="one_to_one")
    panel = _status_as_of(panel, status_changes)
    panel["exclusion_reasons"] = _exclusion_reasons(panel, config)
    panel["is_eligible"] = panel["exclusion_reasons"].eq("")
    columns = ["signal_date", "stock_code", "listing_date", "market_type", "trading_status", "trading_quality_flags",
               "listing_status", "listing_status_effective_date", "exclusion_reasons"]
    pool = panel.loc[panel["is_eligible"], columns].sort_values(["signal_date", "stock_code"]).reset_index(drop=True)
    exclusions = panel.loc[~panel["is_eligible"], columns].sort_values(["signal_date", "stock_code"]).reset_index(drop=True)
    memberships = pool.groupby("signal_date")["stock_code"].agg(set).to_dict()
    prior: set[str] = set()
    summary_rows = []
    for signal_date in signal_dates:
        current = set(memberships.get(signal_date, set()))
        reasons = exclusions.loc[exclusions["signal_date"].eq(signal_date), "exclusion_reasons"]
        summary_rows.append({
            "signal_date": signal_date, "pool_count": len(current), "new_count": len(current - prior),
            "exit_count": len(prior - current), "excluded_count": len(reasons),
            "excluded_by_reason": reasons.str.split("\\|").explode().value_counts().sort_index().to_json(force_ascii=False),
        })
        prior = current
    return UniverseOutputs(pool, pd.DataFrame(summary_rows), exclusions, signal_dates)


def validate_outputs(outputs: UniverseOutputs) -> None:
    """Fail loudly if Stage 3 acceptance invariants do not hold."""
    if outputs.pool.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Historical universe contains duplicate (signal_date, stock_code) keys.")
    if outputs.period_summary["signal_date"].nunique() != len(outputs.signal_dates):
        raise ValueError("At least one signal date is missing from the period summary.")
    if (outputs.period_summary["pool_count"] <= 0).any():
        raise ValueError("At least one signal date has an empty investment universe.")
    counts = outputs.pool.groupby("signal_date").size().reindex(outputs.signal_dates, fill_value=0).to_numpy()
    if not (counts == outputs.period_summary["pool_count"].to_numpy()).all():
        raise ValueError("Period counts do not reconcile to the detailed universe.")


def _audit_sample(outputs: UniverseOutputs, daily_trading: pd.DataFrame, status_changes: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic post-hoc audit fields; never use them in pool construction."""
    sample = outputs.pool.sample(n=min(5, len(outputs.pool)), random_state=20260915).copy()
    last_trade = daily_trading.groupby("stock_code", as_index=False)["trade_date"].max().rename(columns={"trade_date": "post_hoc_last_observed_trade_date"})
    terminal = status_changes.loc[
        status_changes["ChangedItem"].eq("上市状态") & status_changes["ValueAfter"].eq("终止上市"), ["stock_code", "event_date"]
    ].groupby("stock_code", as_index=False)["event_date"].min().rename(columns={"event_date": "post_hoc_termination_effective_date"})
    return sample.merge(last_trade, on="stock_code", how="left").merge(terminal, on="stock_code", how="left")


def _terminal_audit_sample(outputs: UniverseOutputs, daily_trading: pd.DataFrame, status_changes: pd.DataFrame) -> pd.DataFrame:
    """Select terminated names with a pre-termination pool observation for audit evidence."""
    terminal = status_changes.loc[
        status_changes["ChangedItem"].eq("上市状态") & status_changes["ValueAfter"].eq("终止上市"), ["stock_code", "event_date"]
    ].groupby("stock_code", as_index=False)["event_date"].min().rename(columns={"event_date": "termination_effective_date"})
    prior_pool = outputs.pool.merge(terminal, on="stock_code", how="inner")
    prior_pool = prior_pool.loc[prior_pool["signal_date"].lt(prior_pool["termination_effective_date"])]
    prior_pool = prior_pool.sort_values("signal_date").groupby("stock_code", as_index=False).tail(1)
    last_trade = daily_trading.groupby("stock_code", as_index=False)["trade_date"].max().rename(columns={"trade_date": "last_observed_trade_date"})
    return prior_pool.merge(last_trade, on="stock_code", how="left").sample(n=min(3, len(prior_pool)), random_state=20260915)


def render_report(outputs: UniverseOutputs, daily_trading: pd.DataFrame, status_changes: pd.DataFrame, config: UniverseConfig) -> str:
    """Render the Stage 3 evidence report from the generated outputs."""
    sample = _audit_sample(outputs, daily_trading, status_changes)
    reason_counts = outputs.exclusions["exclusion_reasons"].str.split("\\|").explode().value_counts().sort_index()
    lines = [
        "# 阶段 3：历史股票池", "", "## 生成范围与口径", "",
        f"- 信号日：候选市场 `{', '.join(sorted(config.markets))}` 的交易日历 `State=O` 中每月最后一个交易日，共 {len(outputs.signal_dates)} 期（{outputs.signal_dates.min():%Y-%m-%d} 至 {outputs.signal_dates.max():%Y-%m-%d}）。",
        f"- 纳入条件：调仓日有日交易记录、`Trdsta={config.normal_trading_status}`、首次上市满 {config.min_listing_months} 个自然月，且未处于截至该日已生效的暂停上市、退市整理期或终止上市状态。",
        "- 状态只按 `ChangeDate`（标准字段 `event_date`）在信号日或之前生效；未来状态不倒推。`600200`（2025-12-10 至 2025-12-29）与 `600701`（2021-03-15 至 2021-04-26）的已审计冲突区间作为数据质量例外排除。",
        "- 本阶段只决定新建股票池；不定义后续持仓期间的停牌、ST 或退市收益和交易处理。", "",
        "## 输出核对", "",
        f"- 股票池明细：{len(outputs.pool):,} 条 `(signal_date, stock_code)` 记录，无重复键。",
        f"- 期间汇总：{len(outputs.period_summary):,} 期；每期股票池均非空。",
        f"- 排除审计：{len(outputs.exclusions):,} 条逐股票逐期记录；同一条记录可同时具有多个排除原因。", "",
        "### 排除原因计数", "", "| 排除原因 | 记录数 |", "| --- | ---: |",
    ]
    lines.extend(f"| `{reason}` | {int(count):,} |" for reason, count in reason_counts.items())
    lines.extend(["", "## 可复核抽样", "", "以下样本以固定随机种子 `20260915` 从已纳入明细中抽取。`post_hoc_*` 列仅用于事后核对上市后的交易/退出证据，未参与任何信号日筛选。", "", "| 信号日 | 股票 | 上市日 | 当日市场 | 当日状态 | 当期上市状态 | 状态生效日 | 资料中最后交易日 | 终止上市生效日 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    def text(value: object) -> str:
        if pd.isna(value):
            return "—"
        return pd.Timestamp(value).strftime("%Y-%m-%d") if isinstance(value, pd.Timestamp) else str(value)
    for row in sample.itertuples(index=False):
        lines.append("| " + " | ".join(text(getattr(row, name)) for name in [
            "signal_date", "stock_code", "listing_date", "market_type", "trading_status", "listing_status",
            "listing_status_effective_date", "post_hoc_last_observed_trade_date", "post_hoc_termination_effective_date",
        ]) + " |")
    terminal_sample = _terminal_audit_sample(outputs, daily_trading, status_changes)
    lines.extend(["", "### 退出状态核对", "", "以下为固定随机种子 `20260915` 从有终止上市事件、且终止前曾进入股票池的股票中抽取。终止上市生效日之后不进入新建股票池。", "", "| 终止前最后入池信号日 | 股票 | 上市日 | 最后观察交易日 | 终止上市生效日 |", "| --- | --- | --- | --- | --- |"])
    for row in terminal_sample.itertuples(index=False):
        lines.append("| " + " | ".join(text(getattr(row, name)) for name in [
            "signal_date", "stock_code", "listing_date", "last_observed_trade_date", "termination_effective_date",
        ]) + " |")
    lines.extend(["", "## 验收", "", "- 任意信号日均按当日可得信息建立股票池，不使用当前存活名单替代历史截面。", "- `universe_period_summary.csv` 的新增、退出和股票数量可由相邻期 `historical_universe.parquet` 名单复算。", "- ST/非正常交易、停牌（无当日记录）、次新股、退市相关状态和状态冲突区间均在 `universe_exclusions.parquet` 中保留原因。", "- 阶段 4 的 PE、Momentum、ROE 计算及其数据可得性判断不属于本阶段。", ""])
    return "\n".join(lines)


def run(config: UniverseConfig) -> UniverseOutputs:
    """Read Stage 2 data, write Stage 3 artifacts, and return their in-memory form."""
    directory = config.processed_dir
    listings = pd.read_parquet(directory / "listing_dates.parquet")
    status = pd.read_parquet(directory / "status_changes.parquet")
    calendar = pd.read_parquet(directory / "trading_calendar.parquet")
    signal_dates = monthly_signal_dates(calendar, config)
    daily = pd.read_parquet(
        directory / "daily_trading.parquet",
        columns=["stock_code", "trade_date", "market_type", "trading_status", "quality_flags"],
        filters=[("trade_date", "in", list(signal_dates))],
    )
    outputs = build_historical_universe(daily, listings, status, calendar, config)
    validate_outputs(outputs)
    outputs.pool.to_parquet(directory / "historical_universe.parquet", index=False)
    outputs.period_summary.to_csv(directory / "universe_period_summary.csv", index=False, encoding="utf-8-sig")
    outputs.exclusions.to_parquet(directory / "universe_exclusions.parquet", index=False)
    audit_daily = pd.read_parquet(directory / "daily_trading.parquet", columns=["stock_code", "trade_date"])
    report_path = directory.parent.parent / "reports" / "historical_universe_stage3.md"
    report_path.write_text(render_report(outputs, audit_daily, status, config), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 3 historical investment universe.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    outputs = run(load_universe_config(args.config))
    print(f"Built {len(outputs.pool):,} historical-universe records across {len(outputs.signal_dates)} signal dates.")


if __name__ == "__main__":
    main()
