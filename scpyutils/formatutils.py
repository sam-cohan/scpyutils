"""
Utilities useful for for displaying numbers, datetime, timedelta, etc.

Author: Sam Cohan
"""

import datetime
import math
import re
from typing import Any, Callable, Optional, Union

import pandas as pd
import pandas.api.types as types

DT_FMT_NICE = "%Y-%m-%d %H:%M:%S"
DT_FMT_COMP = "%Y%m%d%H%M%S"


def guess_timestamp_unit(ts: int) -> str:
    return (
        "s"
        if ts < 9223372036.854775
        else (
            "ms"
            if ts < 9223372036854.775
            else ("us" if ts < 9223372036854775 else "ns")
        )
    )


def try_get_datetime(
    x: Any,
    unit: Optional[str] = None,
    err_handler: Optional[Callable] = None,
) -> Optional[pd.Timestamp]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isnull(x):
            return None
        if isinstance(x, datetime.datetime) or types.is_datetime64_any_dtype(x):
            return pd.to_datetime(x)
        if isinstance(x, str) and re.match("^[0-9,.]+$", x):
            x = float(x.replace(",", ""))
        if types.is_number(x):
            x = float(x)
            if unit is None:
                unit = guess_timestamp_unit(x)
            return pd.to_datetime(float(x), unit=unit)
        return pd.to_datetime(x)
    except Exception as e:
        if err_handler:
            return err_handler(x, e)
        return None


def try_fmt_datetime(
    x: Any,
    unit: Optional[str] = None,
    err_handler: Optional[Callable] = None,
) -> str:
    dtm = try_get_datetime(x, unit=unit, err_handler=err_handler)
    if dtm is None:
        return "NaT"
    else:
        return dtm.strftime(DT_FMT_NICE)


def try_get_timedelta(
    x: Any,
    unit: str = "ms",
    err_handler: Optional[Callable] = None,
) -> Optional[pd.Timedelta]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isnull(x):
            return None
        if isinstance(x, datetime.datetime) or types.is_timedelta64_dtype(x):
            return pd.to_timedelta(x)
        if isinstance(x, str) and re.match("^[0-9,.]+$", x):
            x = float(x.replace(",", ""))
        if types.is_number(x):
            x = float(x)
            return pd.to_timedelta(float(x), unit=unit)
        return pd.to_timedelta(x)
    except Exception as e:
        if err_handler:
            return err_handler(x, e)
        return None


def try_fmt_timedelta(
    x: Any,
    unit: Optional[str] = None,
    max_unit: str = "d",
    full_precision: bool = True,
    round_td: Optional[Union[str, pd.Timedelta]] = None,
    err_handler: Optional[Callable] = None,
) -> str:
    try:
        x = try_get_timedelta(x, unit=unit)
        if x is None:
            return "NaT"
        if round_td:
            if isinstance(round_td, str):
                round_td = pd.to_timedelta(round_td)
            x = x.round(round_td)
        float_x = x.total_seconds()
        secs = int(float_x)
        ns_part = (float_x - secs) * 1e9
        sign = "-" if secs < 0 else ""
        secs = abs(secs)
        periods = [
            ("Y", 31536000),  # 60 * 60 * 24 * 365
            ("M", 2592000),  # 60 * 60 * 24 * 30,
            ("d", 86400),  # 60 * 60 * 24
            ("h", 3600),  # 60 * 60
            ("m", 60),
            ("s", 1),
        ]
        period_idx = {x[0]: i for i, x in enumerate(periods)}
        periods = periods[period_idx[max_unit] :]
        strs = []
        for prd_name, prd_secs in periods:
            if secs >= prd_secs:
                val, secs = divmod(secs, prd_secs)
                if val:
                    strs.append(f"{val:,.0f} {prd_name}")
                    continue
            if strs:
                strs.append("")
        if ns_part:
            ms_part = int(ns_part) / 1e6
            if ms_part:
                strs.append(f"{ms_part:.0f} ms")
        if not full_precision:
            strs = strs[:3]
        return (sign + " ".join([x for x in strs if x])) or "0"
    except Exception as e:
        if err_handler:
            return err_handler(x, e)
        return str(x)


def try_fmt_num(
    x: Any,
    full_precision: bool = True,
    err_handler: Optional[Callable] = None,
) -> str:
    """Format a numeric value by rounding it appropriately based on its magnitude."""
    try:
        if isinstance(x, str) and re.match("^[0-9,.]+$", x):
            x = float(x.replace(",", ""))
        if types.is_number(x):
            if pd.isnull(x):
                return ""
        abs_x = abs(float(x))
        dps = (
            0
            if abs(x - int(x)) < 1e-20
            else (
                4
                if abs_x < 1
                else 3
                if abs_x < 10
                else 2
                if abs_x < 100
                else 1
                if abs_x < 1000
                else 0
            )
        )
        if full_precision or abs_x < 1000:
            return ("{:,.%sf}" % dps).format(x)
        else:
            if abs_x > 0.9995e15:
                return "{:,.3f}P".format(x / 1e15)
            if abs_x > 0.9995e12:
                return "{:,.3f}T".format(x / 1e12)
            elif abs_x > 0.9995e9:
                return "{:,.3f}G".format(x / 1e9)
            elif abs_x > 0.9995e6:
                return "{:,.3f}M".format(x / 1e6)
            else:
                return "{:,.0f}".format(x)
    except Exception as e:
        if err_handler:
            return err_handler(x, e)
        return str(x)


def try_fmt_ccy(
    x: Any,
    ccy_sign: str = "$",
    err_handler: Optional[Callable] = None,
    full_precision: bool = True,
) -> str:
    """Format a currency value by rounding it appropriately based on its magnitude."""
    try:
        if types.is_number(x):
            if pd.isnull(x):
                return ""
        abs_x = abs(float(x))
        if abs_x == 0:
            return "{}0".format(ccy_sign)
        if abs_x < 1000:
            return "{}{:,.2f}".format(ccy_sign, x)
        elif full_precision or abs_x < 100000:
            return "{}{:,.0f}".format(ccy_sign, x)
        else:
            num_digits = math.ceil(math.log10(x))
            return "{}{:,.0f}".format(ccy_sign, round(x, -(num_digits - 5)))
    except Exception as e:
        if err_handler:
            return err_handler(x, e)
        return str(x)


def try_fmt_interval_from_start_end_timestamps(
    start_timestamp: Any,
    end_timestamp: Any,
    unit: Optional[str] = None,
    full_precision: bool = False,
) -> str:
    start_datetime = try_get_datetime(start_timestamp, unit=unit)
    end_datetime = try_get_datetime(end_timestamp, unit=unit)
    if start_datetime is not None and end_datetime is not None:
        timedelta = end_datetime - start_datetime
    else:
        timedelta = None
    return (
        f"[{try_fmt_datetime(start_timestamp)}"
        f", {try_fmt_datetime(end_timestamp)}]"
        f"({try_fmt_timedelta(timedelta, full_precision=full_precision)})"
    )


def try_fmt_timedeltas_from_interval_str(bin_name):
    match = re.match("([([])([0-9]+|[+-]*inf) *, *([0-9]+|[+]*inf)([])])", bin_name)
    if match:
        groups = match.groups()
        return "".join(
            [
                groups[0],
                try_fmt_timedelta(groups[1], full_precision=False),
                ", ",
                try_fmt_timedelta(groups[2], full_precision=False),
                groups[3],
            ]
        )
    return bin_name
