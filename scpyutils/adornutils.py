from datetime import datetime
from typing import Optional, Union

import mmh3
import numpy as np
import pandas as pd


def adorn_with_hash(
    df: pd.DataFrame, in_col: str, out_col: Optional[str] = None
) -> pd.DataFrame:
    """Adorns DataFrame with a hash column based on the contents of an input column.

    Args:
        df: The DataFrame to modify.
        in_col: The name of the input column containing the strings to be hashed.
        out_col: Optional; name of the output column. Defaults to '{in_col}_hash'.

    Returns:
        DataFrame: The original DataFrame with an additional column for the hash.
    """
    if not out_col:
        out_col = f"{in_col}_hash"

    def hash_to_hex(s: str) -> str:
        if s is None:
            s = ""
        return mmh3.hash_bytes(s.encode("utf-8")).hex()[:16]

    df[out_col] = df[in_col].map(hash_to_hex)

    return df


def adorn_with_timestamp(
    df: pd.DataFrame,
    ts_fld: str,
    start_ts: datetime,
    end_ts: datetime,
    id_fld: str,
    dt_fld: Optional[str] = None,
    scale: float = 1,
    jitter_days: float = 1,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Adorns DataFrame with a timestamp field based on a non-uniform distribution

    Args:
        df: The DataFrame to modify.
        ts_fld: Name of the timestamp field.
        start_ts: The start timestamp for the distribution range.
        end_ts: The end timestamp for the distribution range.
        id_fld: Rows with same values for this field will get the same timestamp.
        dt_fld: Optional name of date field to extract from timestamp.
        scale: Controls the shape of the distribution. Higher values make the
            distribution more uniform.
        jitter_days: The maximum number of days by which to jitter the assigned
            timestamp.
        seed: Optional seed for deterministic behavior.

    Returns:
        The original DataFrame with added `ts_fld` column with Timestamp type
        and optionally a `dt_fld` with Date type.
    """
    np.random.seed(seed)
    unique_ids = df[id_fld].unique()
    num_ids = len(unique_ids)

    num_dates = (end_ts - start_ts).days + 1
    date_range = pd.date_range(start=start_ts, end=end_ts, freq="D")
    probabilities = np.random.dirichlet(np.ones(num_dates) * scale, size=1)[0]
    id_dates = np.random.choice(date_range, size=num_ids, p=probabilities)
    jitter_deltas = np.random.uniform(-jitter_days, jitter_days, size=num_ids)
    id_dates_jittered = id_dates + pd.to_timedelta(jitter_deltas, unit="D")
    id_date_mapping = pd.Series(id_dates_jittered, index=unique_ids)
    df[ts_fld] = df[id_fld].map(id_date_mapping)
    if dt_fld:
        df[dt_fld] = pd.to_datetime(df[ts_fld].dt.date)
    return df


def adorn_with_groups(
    df: pd.DataFrame,
    group_fld: str,
    group_val_base: str,
    num_groups: int,
    id_fld: str,
    scale: float = 0.7,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Adorns DataFrame with a new group field with random values from range

    Args:
        df: The DataFrame to modify.
        group_fld: Name of the group field.
        group_val_base: Base name of values for the group field.
        num_groups: The number of unique groups to assign.
        id_fld: Rows with same values for this field will get the same
            timestamp.
        scale: Controls the shape of the distribution. Higher values make the
            distribution more uniform.
        seed: Optional seed for deterministic behavior.

    Returns:
        The original DataFrame with an added 'org' column.
    """
    np.random.seed(seed)
    unique_ids = df[id_fld].unique()
    num_ids = len(unique_ids)
    group_vals = [f"{group_val_base}{i:03d}" for i in range(1, num_groups + 1)]
    probabilities = np.random.dirichlet(np.ones(num_groups) * scale, size=1)[0]
    id_vals = np.random.choice(group_vals, size=num_ids, p=probabilities)
    id_val_mapping = pd.Series(id_vals, index=unique_ids)
    df[group_fld] = df[id_fld].map(id_val_mapping)
    return df


def adorn_with_is_test(
    df: pd.DataFrame, test_size: Union[int, float], seed: Optional[int] = None
):
    """Adorns a DataFrame with an `is_test` column

    Args:
        df: The DataFrame to process.
        test_size: If an integer greater than 1, will be treated as absolute number
            of rows to mark as test. If a float between 0 and 1, will be treated as
            proportion of rows to mark as test.
        seed: Optional seed for deterministic behavior.

    Returns:
        The DataFrame with an added `is_test` column.
    """
    if 0 < test_size < 1:
        test_size = int(test_size * len(df))
    df["is_test"] = False
    test_indices = df.sample(n=test_size, random_state=seed).index
    df.loc[test_indices, "is_test"] = True
    return df
