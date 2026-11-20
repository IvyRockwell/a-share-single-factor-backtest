"""Build point-in-time market capitalization for the historical pool."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MIN_MARKET_CAP = 1e6
MAX_MARKET_CAP = 1e13
COLUMNS = [
    "signal_date", "stock_code", "shares_effective_date", "total_shares", "float_shares",
    "close_price", "market_cap", "float_market_cap", "has_market_cap", "log_market_cap",
]


@dataclass(frozen=True)
class MarketCapConfig:
    processed_dir: Path


@dataclass(frozen=True)
class MarketCapOutputs:
    frame: pd.DataFrame


def load_market_cap_config(config_path: Path) -> MarketCapConfig:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    return MarketCapConfig(config_path.parent.parent / settings["data"]["processed_dir"])


def attach_shares(pool: pd.DataFrame, capital: pd.DataFrame) -> pd.DataFrame:
    """Attach the share count that was effective on or before each signal date."""
    left = pool.assign(signal_date=pd.to_datetime(pool["signal_date"]).astype("datetime64[ns]"))
    right = capital.assign(effective_date=pd.to_datetime(capital["effective_date"]).astype("datetime64[ns]"))
    left = left.sort_values("signal_date")
    right = right.sort_values("effective_date")
    merged = pd.merge_asof(
        left, right,
        left_on="signal_date", right_on="effective_date",
        by="stock_code", direction="backward",
    )
    return merged.rename(columns={"effective_date": "shares_effective_date"})


def build_market_cap(pool: pd.DataFrame, capital: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Multiply the effective share count by the raw close price of the signal day."""
    frame = attach_shares(pool, capital)
    frame = frame.merge(prices, on=["stock_code", "signal_date"], how="left", validate="one_to_one")
    frame["market_cap"] = frame["total_shares"] * frame["close_price"]
    frame["float_market_cap"] = frame["float_shares"] * frame["close_price"]
    frame["has_market_cap"] = frame["market_cap"].between(MIN_MARKET_CAP, MAX_MARKET_CAP)
    frame["log_market_cap"] = np.log(frame["market_cap"].where(frame["has_market_cap"]))
    return frame[COLUMNS].sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


def validate_outputs(outputs: MarketCapOutputs, pool: pd.DataFrame) -> None:
    frame = outputs.frame
    if len(frame) != len(pool) or frame.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Market cap does not have exactly one row per pool key.")
    coverage = float(frame["has_market_cap"].mean())
    if coverage < 0.90:
        raise ValueError(f"Market cap covers only {coverage:.1%} of the pool; check the capital table.")
    if frame.loc[frame["has_market_cap"], "total_shares"].le(0).any():
        raise ValueError("A covered row has non-positive share count.")


def coverage_by_year(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(frame["signal_date"].dt.year)["has_market_cap"].mean()


def run(config: MarketCapConfig) -> MarketCapOutputs:
    directory = config.processed_dir
    pool = pd.read_parquet(directory / "historical_universe.parquet", columns=["signal_date", "stock_code"])
    pool["signal_date"] = pd.to_datetime(pool["signal_date"])
    capital = pd.read_parquet(directory / "capital.parquet", columns=["stock_code", "effective_date", "total_shares", "float_shares"])
    signal_dates = pool["signal_date"].drop_duplicates()
    daily = pd.read_parquet(
        directory / "daily_trading.parquet",
        columns=["stock_code", "trade_date", "close_price"],
        filters=[("trade_date", "in", list(signal_dates))],
    ).rename(columns={"trade_date": "signal_date"})
    outputs = MarketCapOutputs(build_market_cap(pool, capital, daily))
    validate_outputs(outputs, pool)
    outputs.frame.to_parquet(directory / "market_cap.parquet", index=False)
    print(f"Market cap coverage {outputs.frame['has_market_cap'].mean():.3%} over {len(outputs.frame):,} pool rows.")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build market capitalization for the pool.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    run(load_market_cap_config(args.config))


if __name__ == "__main__":
    main()
