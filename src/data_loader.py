"""Stage 2 CSMAR cleaning pipeline.

The pipeline treats ``项目初始数据`` as read-only.  It preserves source values,
adds normalized fields and explicit quality flags, and writes only to
``data/processed``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


RESEARCH_MARKETS = {"1", "4", "16", "32"}


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path
    processed_dir: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def load_config(config_path: Path) -> PipelineConfig:
    """Read the small project configuration used by the pipeline."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config["data"]
    project_root = config_path.parent.parent
    return PipelineConfig(
        raw_dir=project_root / data["raw_dir"],
        processed_dir=project_root / data["processed_dir"],
        start_date=pd.Timestamp(data["start_date"]),
        end_date=pd.Timestamp(data["end_date"]),
    )


def source_files(raw_dir: Path, table: str) -> list[Path]:
    """Return every data shard for a CSMAR table, never its description file."""
    files = sorted(path for path in raw_dir.rglob(f"{table}*.csv") if "[DES]" not in path.name)
    if not files:
        raise FileNotFoundError(f"No {table} CSV files under {raw_dir}")
    return files


def read_shards(paths: Iterable[Path]) -> pd.DataFrame:
    """Read UTF-8 CSMAR CSV shards while retaining source text for auditability."""
    return pd.concat((pd.read_csv(path, dtype="string", encoding="utf-8") for path in paths), ignore_index=True)


def normalize_code(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = values.astype("string").str.strip()
    normalized = raw.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    valid = normalized.str.fullmatch(r"\d{6}", na=False)
    return normalized.where(valid).astype("string"), (~valid).fillna(True)


def normalize_date(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    normalized = pd.to_datetime(values, errors="coerce")
    return normalized, normalized.isna() & values.notna()


def normalize_numeric(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    normalized = pd.to_numeric(values, errors="coerce")
    # A fixed floating schema is required when independent CSV chunks contain
    # different mixtures of integer-looking and decimal values.
    normalized = normalized.astype("float64")
    return normalized, normalized.isna() & values.notna() & values.astype("string").str.strip().ne("")


def add_flags(frame: pd.DataFrame, flags: dict[str, pd.Series]) -> pd.DataFrame:
    result = frame.copy()
    active = pd.DataFrame({name: flag.fillna(False) for name, flag in flags.items()})
    result["quality_flags"] = active.apply(
        lambda row: "|".join(name for name, value in row.items() if value), axis=1
    ).astype("string")
    result["has_quality_issue"] = result["quality_flags"].ne("")
    return result


def filter_research_period(frame: pd.DataFrame, date_column: str, config: PipelineConfig) -> tuple[pd.DataFrame, int]:
    in_period = frame[date_column].between(config.start_date, config.end_date, inclusive="both")
    return frame.loc[in_period].copy(), int((~in_period).sum())


def duplicate_flags(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    return frame.duplicated(keys, keep=False)


def clean_daily_trading(raw: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, int]:
    frame = raw.rename(columns={"Stkcd": "stock_code_source", "Trddt": "trade_date_source"}).copy()
    frame["stock_code"], bad_code = normalize_code(frame["stock_code_source"])
    frame["trade_date"], bad_date = normalize_date(frame["trade_date_source"])
    numeric_flags = {}
    for source, target in {"Opnprc": "open_price", "Clsprc": "close_price", "Dretwd": "total_return", "Adjprcwd": "adjusted_close"}.items():
        frame[target], numeric_flags[f"invalid_{target}"] = normalize_numeric(frame[source])
    frame["market_type"] = frame["Markettype"].astype("string")
    frame["trading_status"] = frame["Trdsta"].astype("string")
    frame = add_flags(frame, {
        "invalid_stock_code": bad_code,
        "invalid_trade_date": bad_date,
        "duplicate_key": duplicate_flags(frame, ["stock_code", "trade_date"]),
        "extreme_total_return": frame["total_return"].abs().gt(1),
        **numeric_flags,
    })
    filtered, outside = filter_research_period(frame, "trade_date", config)
    return filtered, outside


def clean_valuation(raw: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, int]:
    frame = raw.rename(columns={"Symbol": "stock_code_source", "TradingDate": "trading_date_source"}).copy()
    frame["stock_code"], bad_code = normalize_code(frame["stock_code_source"])
    frame["trading_date"], bad_date = normalize_date(frame["trading_date_source"])
    frame["pe_ttm"], bad_pe = normalize_numeric(frame["PE1TTM"])
    frame = add_flags(frame, {
        "invalid_stock_code": bad_code,
        "invalid_trading_date": bad_date,
        "invalid_pe_ttm": bad_pe,
        "missing_pe_ttm": frame["pe_ttm"].isna(),
        "nonpositive_pe_ttm": frame["pe_ttm"].le(0),
        "duplicate_key": duplicate_flags(frame, ["stock_code", "trading_date"]),
    })
    filtered, outside = filter_research_period(frame, "trading_date", config)
    return filtered, outside


def clean_benchmark(raw: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, int]:
    frame = raw.rename(columns={"Indexcd": "index_code", "Idxtrd01": "trade_date_source"}).copy()
    frame["trade_date"], bad_date = normalize_date(frame["trade_date_source"])
    frame["close"], bad_close = normalize_numeric(frame["Idxtrd05"])
    raw_return, bad_return = normalize_numeric(frame["Idxtrd08"])
    frame["total_return"] = raw_return / 100
    frame = frame.loc[frame["index_code"].astype("string").eq("H00300")].copy()
    frame = add_flags(frame, {
        "invalid_trade_date": bad_date.loc[frame.index],
        "invalid_close": bad_close.loc[frame.index],
        "invalid_total_return": bad_return.loc[frame.index],
        "duplicate_key": duplicate_flags(frame, ["index_code", "trade_date"]),
    })
    filtered, outside = filter_research_period(frame, "trade_date", config)
    return filtered, outside


def clean_financials(income: pd.DataFrame, balance: pd.DataFrame, disclosures: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    def prepare(frame: pd.DataFrame, value_column: str, target: str) -> pd.DataFrame:
        result = frame.rename(columns={"Stkcd": "stock_code_source", "Accper": "report_date_source"}).copy()
        result["stock_code"], _ = normalize_code(result["stock_code_source"])
        result["report_date"], _ = normalize_date(result["report_date_source"])
        result[target], _ = normalize_numeric(result[value_column])
        return result[["stock_code", "report_date", "Typrep", target]]

    profits = prepare(income, "B002000000", "net_profit")
    equity = prepare(balance, "A003000000", "shareholder_equity")
    disclosure = disclosures.rename(columns={"Stkcd": "stock_code_source", "Accper": "report_date_source"}).copy()
    disclosure["stock_code"], _ = normalize_code(disclosure["stock_code_source"])
    disclosure["report_date"], _ = normalize_date(disclosure["report_date_source"])
    disclosure["actual_disclosure_date"], bad_disclosure = normalize_date(disclosure["Actudt"])
    disclosure["invalid_actual_disclosure_date"] = bad_disclosure
    disclosure = disclosure[["stock_code", "report_date", "actual_disclosure_date", "invalid_actual_disclosure_date"]]
    frame = profits.merge(equity, on=["stock_code", "report_date", "Typrep"], how="outer", indicator="financial_merge")
    frame = frame.merge(disclosure, on=["stock_code", "report_date"], how="left")
    frame = frame.loc[frame["report_date"].le(config.end_date)].copy()
    annual = frame["report_date"].dt.month.eq(12) & frame["report_date"].dt.day.eq(31)
    consolidated = frame["Typrep"].eq("A")
    valid_disclosure = frame["actual_disclosure_date"].notna() & ~frame["invalid_actual_disclosure_date"].fillna(False)
    frame = add_flags(frame, {
        "missing_net_profit": frame["net_profit"].isna(),
        "missing_shareholder_equity": frame["shareholder_equity"].isna(),
        "nonpositive_shareholder_equity": frame["shareholder_equity"].le(0),
        "not_annual_report": ~annual,
        "not_consolidated_report": ~consolidated,
        "missing_or_invalid_actual_disclosure_date": ~valid_disclosure,
        "duplicate_key": duplicate_flags(frame, ["stock_code", "report_date", "Typrep"]),
    })
    frame["is_roe_candidate"] = annual & consolidated & valid_disclosure & frame["net_profit"].notna() & frame["shareholder_equity"].gt(0)
    return frame


def clean_simple_table(raw: pd.DataFrame, code_source: str, date_source: str, config: PipelineConfig) -> pd.DataFrame:
    frame = raw.rename(columns={code_source: "stock_code_source", date_source: "event_date_source"}).copy()
    frame["stock_code"], bad_code = normalize_code(frame["stock_code_source"])
    frame["event_date"], bad_date = normalize_date(frame["event_date_source"])
    frame = frame.loc[frame["event_date"].le(config.end_date)].copy()
    return add_flags(frame, {"invalid_stock_code": bad_code.loc[frame.index], "invalid_event_date": bad_date.loc[frame.index]})


def clean_calendar(raw: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Standardize calendar rows without treating market codes as stock codes."""
    frame = raw.rename(columns={"Markettype": "market_type_source", "Clddt": "calendar_date_source"}).copy()
    frame["market_type"] = frame["market_type_source"].astype("string")
    frame["calendar_date"], bad_date = normalize_date(frame["calendar_date_source"])
    frame = frame.loc[frame["calendar_date"].le(config.end_date)].copy()
    return add_flags(frame, {
        "invalid_calendar_date": bad_date.loc[frame.index],
        "duplicate_key": duplicate_flags(frame, ["market_type", "calendar_date"]),
    })


def build_quality_summary(outputs: dict[str, pd.DataFrame], outside_period: dict[str, int], source_paths: dict[str, list[Path]]) -> dict:
    datasets = {}
    for name, frame in outputs.items():
        flag_counts = frame["quality_flags"].replace("", pd.NA).dropna().str.split("\\|").explode().value_counts().to_dict()
        datasets[name] = {
            "output_records": int(len(frame)),
            "output_unique_stocks": int(frame["stock_code"].nunique()) if "stock_code" in frame else None,
            "rows_with_quality_issue": int(frame["has_quality_issue"].sum()),
            "quality_flag_counts": {str(key): int(value) for key, value in flag_counts.items()},
            "excluded_as_outside_research_period": outside_period.get(name, 0),
            "input_files": [str(path) for path in source_paths.get(name, [])],
        }
    return {"research_period": {"start": "2014-09-12", "end": "2025-12-31"}, "datasets": datasets}


def write_outputs(outputs: dict[str, pd.DataFrame], config: PipelineConfig, summary: dict) -> None:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_parquet(config.processed_dir / f"{name}.parquet", index=False)
    mapping = pd.DataFrame([
        ("daily_trading", "Stkcd", "stock_code"), ("daily_trading", "Trddt", "trade_date"),
        ("daily_trading", "Dretwd", "total_return"), ("daily_trading", "Adjprcwd", "adjusted_close"),
        ("valuation", "Symbol", "stock_code"), ("valuation", "TradingDate", "trading_date"),
        ("valuation", "PE1TTM", "pe_ttm"), ("benchmark", "Idxtrd05", "close"),
        ("benchmark", "Idxtrd08", "total_return (divided by 100)"),
        ("financial_availability", "Actudt", "actual_disclosure_date"),
    ], columns=["dataset", "source_field", "standard_field"])
    mapping.to_csv(config.processed_dir / "field_mapping.csv", index=False, encoding="utf-8-sig")
    with (config.processed_dir / "data_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def _issue_rows(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["stock_code", "trade_date", "trading_date", "report_date", "event_date", "quality_flags"] if column in frame]
    flagged = frame.loc[frame["has_quality_issue"], columns].copy()
    for column in ["stock_code", "trade_date", "trading_date", "report_date", "event_date", "quality_flags"]:
        if column not in flagged:
            flagged[column] = pd.NA
    flagged.insert(0, "dataset", name)
    flagged = flagged[["dataset", "stock_code", "trade_date", "trading_date", "report_date", "event_date", "quality_flags"]]
    # This shared audit table receives rows from differently dated datasets;
    # serialize identifiers and dates consistently across every append.
    return flagged.astype("string")


class ParquetAppender:
    """Append data frames without retaining the complete daily tables in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def stream_large_table(name: str, paths: list[Path], cleaner, config: PipelineConfig, issues: ParquetAppender) -> dict:
    """Clean large daily shards in bounded memory and append their audit rows."""
    output = ParquetAppender(config.processed_dir / f"{name}.parquet")
    records = stocks = outside = issue_rows = 0
    flags: dict[str, int] = {}
    for path in paths:
        for raw_chunk in pd.read_csv(path, dtype="string", encoding="utf-8", chunksize=250_000):
            cleaned, chunk_outside = cleaner(raw_chunk, config)
            output.append(cleaned)
            flagged = _issue_rows(name, cleaned)
            issues.append(flagged)
            records += len(cleaned)
            stocks += cleaned["stock_code"].nunique()
            outside += chunk_outside
            issue_rows += len(flagged)
            for flag, count in cleaned["quality_flags"].replace("", pd.NA).dropna().str.split("\\|").explode().value_counts().items():
                flags[str(flag)] = flags.get(str(flag), 0) + int(count)
    output.close()
    return {
        "output_records": records,
        "output_unique_stocks_chunk_sum": stocks,
        "rows_with_quality_issue": issue_rows,
        "quality_flag_counts": flags,
        "excluded_as_outside_research_period": outside,
        "input_files": [str(path) for path in paths],
    }


def run_pipeline(config: PipelineConfig) -> dict:
    """Create all standardized Stage 2 datasets and return their audit summary."""
    tables = {
        "stock_info": source_files(config.raw_dir, "TRD_Co"),
        "daily_trading": source_files(config.raw_dir, "TRD_Dalyr"),
        "valuation": source_files(config.raw_dir, "STK_MKT_ValuationMetrics"),
        "benchmark": source_files(config.raw_dir, "IDX_Idxtrd"),
        "trading_calendar": source_files(config.raw_dir, "TRD_Cale"),
        "income": source_files(config.raw_dir, "FS_Comins"),
        "balance": source_files(config.raw_dir, "FS_Combas"),
        "disclosures": source_files(config.raw_dir, "IAR_Forecdt"),
        "listing_dates": source_files(config.raw_dir, "STK_LISTEDCOINFOANL"),
        "status_changes": source_files(config.raw_dir, "STK_ITEMCHANGE"),
    }
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    issues = ParquetAppender(config.processed_dir / "unresolved_records.parquet")
    daily_summary = stream_large_table("daily_trading", tables["daily_trading"], clean_daily_trading, config, issues)
    valuation_summary = stream_large_table("valuation", tables["valuation"], clean_valuation, config, issues)
    stock_info = clean_simple_table(read_shards(tables["stock_info"]), "Stkcd", "Listdt", config)
    benchmark, benchmark_outside = clean_benchmark(read_shards(tables["benchmark"]), config)
    calendar = clean_calendar(read_shards(tables["trading_calendar"]), config)
    listings = clean_simple_table(read_shards(tables["listing_dates"]), "Symbol", "EndDate", config)
    status = clean_simple_table(read_shards(tables["status_changes"]), "Symbol", "ChangeDate", config)
    financials = clean_financials(read_shards(tables["income"]), read_shards(tables["balance"]), read_shards(tables["disclosures"]), config)
    outputs = {
        "stock_info": stock_info, "benchmark": benchmark,
        "trading_calendar": calendar, "listing_dates": listings, "status_changes": status,
        "financial_availability": financials,
    }
    source_paths = {name: paths for name, paths in tables.items()}
    source_paths["financial_availability"] = tables["income"] + tables["balance"] + tables["disclosures"]
    summary = build_quality_summary(outputs, {"benchmark": benchmark_outside}, source_paths)
    summary["datasets"]["daily_trading"] = daily_summary
    summary["datasets"]["valuation"] = valuation_summary
    write_outputs(outputs, config, summary)
    for name, frame in outputs.items():
        issues.append(_issue_rows(name, frame))
    issues.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized Stage 2 CSMAR datasets.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    summary = run_pipeline(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
