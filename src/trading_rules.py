"""A-share limit and execution-availability rules.

The daily data carries the exchange pre-close (``PreClosePrice``) and market
type, so limit prices can be reconstructed without look-ahead.  The rules are
deliberately conservative: an opening print at the limit is treated as
unfillable on the side that would need the other side of the book.
"""

from __future__ import annotations

import pandas as pd

CHINEXT_REFORM = pd.Timestamp("2020-08-24")
LIMIT_TOLERANCE = 0.005


def _market_type(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def limit_rate(trade_date: object, market_type: object) -> float | None:
    """Return the daily price-limit rate, or ``None`` when it is unknown.

    Main board (Shanghai/Shenzhen A) is 10%, ChiNext is 10% before the
    2020-08-24 reform and 20% after, STAR is 20%.  B shares, BSE and missing
    market types are outside the research universe and return ``None`` so the
    caller does not silently apply the wrong rule.
    """
    market = _market_type(market_type)
    if market in {"1", "4"}:
        return 0.10
    if market == "16":
        date = pd.Timestamp(trade_date)
        return 0.20 if date >= CHINEXT_REFORM else 0.10
    if market == "32":
        return 0.20
    return None


def limit_prices(pre_close: object, trade_date: object, market_type: object) -> tuple[float | None, float | None]:
    """Return the rounded (limit-up, limit-down) prices, or ``(None, None)``."""
    rate = limit_rate(trade_date, market_type)
    if rate is None or pre_close is None or pd.isna(pre_close):
        return None, None
    price = float(pre_close)
    if price <= 0:
        return None, None
    return round(price * (1.0 + rate), 2), round(price * (1.0 - rate), 2)


def _row_value(row: object, key: str) -> object:
    if isinstance(row, dict):
        return row.get(key)
    try:
        if key not in row:
            return None
        return row[key]
    except (TypeError, KeyError, IndexError):
        return None


def trade_block_reason(row: object, normal_status: str, side: str = "buy") -> str | None:
    """Return why an open trade cannot happen, or ``None`` if it can.

    The function first enforces normal trading status and a valid open price;
    limit checks are skipped when the pre-close or market type is unavailable
    so legacy fixtures without those fields keep their old behaviour.
    """
    status = _row_value(row, "trading_status")
    if status is None or pd.isna(status) or str(status) != str(normal_status):
        return "non_normal_trading_status"
    open_price = _row_value(row, "open_price")
    if open_price is None or pd.isna(open_price) or float(open_price) <= 0:
        return "invalid_open_price"
    flags = _row_value(row, "quality_flags")
    if flags is not None and "invalid_open_price" in str(flags):
        return "invalid_open_price"

    pre_close = _row_value(row, "pre_close")
    if pre_close is None:
        pre_close = _row_value(row, "PreClosePrice")
    market_type = _row_value(row, "market_type")
    if market_type is None:
        market_type = _row_value(row, "Markettype")
    if pre_close is None or market_type is None or pd.isna(pre_close):
        return None
    trade_date = _row_value(row, "trade_date")
    if trade_date is None:
        return None
    limit_up, limit_down = limit_prices(pre_close, trade_date, market_type)
    if limit_up is None or limit_down is None:
        return None
    price = float(open_price)
    if side == "buy" and price >= limit_up - LIMIT_TOLERANCE:
        return "limit_up_open"
    if side == "sell" and price <= limit_down + LIMIT_TOLERANCE:
        return "limit_down_open"
    return None
