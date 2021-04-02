"""
This module has DataFrame utilities which are sometimes specific
to dealing with trading data.
"""
import datetime
import re
import traceback
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook

from .logutils import setup_logger

LOGGER = setup_logger(__name__)


def extract_sampled_df(
    df: str,
    sample_freq_str: str,
    col_pattern: str = ".*",
):
    subdf = df[[c for c in df if re.search(col_pattern, c)]]
    return subdf.resample(sample_freq_str).ffill().bfill()


class FieldGuesser:
    """Class for guessing common field names from a list of fields."""

    @staticmethod
    def guess_fld(flds: List, pattern: str) -> Optional[str]:
        matches = [c for c in flds if re.search(f"{pattern}", str(c), re.IGNORECASE)]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Highest priority match is if it starts with three underscores
        # as that is our convention for the original field backup.
        top_matches = [c for c in matches if c.startswith("___")]
        if top_matches:
            return sorted(top_matches, key=lambda x: len(str(x)))[0]
        return sorted(matches, key=lambda x: len(str(x)))[0]

    @staticmethod
    def guess_side_fld(flds: List) -> Optional[str]:
        return FieldGuesser.guess_fld(flds, pattern="side")

    @staticmethod
    def guess_symbol_fld(flds: List) -> Optional[str]:
        return FieldGuesser.guess_fld(flds, pattern="symbol")

    @staticmethod
    def guess_amount_fld(flds: List) -> Optional[str]:
        return FieldGuesser.guess_fld(flds, pattern="amount")

    @staticmethod
    def guess_price_fld(flds: List) -> Optional[str]:
        return FieldGuesser.guess_fld(flds, pattern="price")


class SideConverter:
    """Converter to be applied to rows of a trade or order DataFrame and
    change the side depending on whether the away asset is flipped or not.

    Exampled usage:
    orders_df["unified_side"] = orders_df.apply(SideConverter("usdt"), axis=1)
    """

    def __init__(
        self,
        away_asset: str,
        symbol_fld: Optional[str] = None,
        side_fld: Optional[str] = None,
        logger: Optional[Callable] = None,
    ):
        self.away_asset = away_asset.upper()
        self.side_fld = side_fld
        self.symbol_fld = symbol_fld
        self.logger = logger

    def log(self, msg):
        if self.logger:
            self.logger(msg)

    def __call__(self, x: pd.Series) -> str:
        if not self.symbol_fld:
            self.symbol_fld = FieldGuesser.guess_side_fld(list(x.index))
            self.log(f"symbol_fld={self.symbol_fld}")
        if not self.side_fld:
            self.side_fld = FieldGuesser.guess_symbol_fld(list(x.index))
            self.log(f"side_fld={self.side_fld}")
        away_asset = self.away_asset
        symbol = x[self.symbol_fld]
        side = x[self.side_fld]
        if not re.search(f"^{away_asset}", symbol, re.IGNORECASE):
            if re.search(f"{away_asset}$", symbol, re.IGNORECASE):
                # flip the side
                if side == "sell":
                    side = "buy"
                elif side == "buy":
                    side = "sell"
                else:
                    raise Exception("unknown side={side} for record={x}")
            else:
                self.log(
                    f"away_asset={self.away_asset} not part of instrument {symbol}"
                )
        return side

    def __str__(self):
        return (
            f"GetRealSide(away_asset={self.away_asset}"
            f", symbol_fld={self.symbol_fld}"
            f", side_fld={self.side_fld}"
            ")"
        )


def unify_side(
    df: pd.DataFrame,
    away_asset: str,
    symbol_fld: Optional[str] = None,
    side_fld: Optional[str] = None,
    amount_fld: Optional[str] = None,
    price_fld: str = "price",
    logger: Optional[Callable] = None,
) -> pd.DataFrame:
    """Given a DataFrame, and a risky asset unify the side, price and amount fields.

    Args:
        away_asset (str): the desired asset to have amounts in (i.e. base asset).
        symbol_fld (str): symbol field name. (defaults to None and guesses).
        side_fld (str): symbol field name. (defaults to None and guesses).
        amount_fld (str): amount field name. (defaults to None nd guesses).
        price_fld (str): price field name. (defaults to None and guesses).

    returns:
        (pd.DataFrame) converted DataFrame (not a copy).
    """
    if symbol_fld is None:
        symbol_fld = FieldGuesser.guess_symbol_fld(list(df.columns))
    if side_fld is None:
        side_fld = FieldGuesser.guess_side_fld(list(df.columns))
    if amount_fld is None:
        amount_fld = FieldGuesser.guess_amount_fld(list(df.columns))
    if price_fld is None:
        price_fld = FieldGuesser.guess_price_fld(list(df.columns))
    if side_fld not in df or symbol_fld not in df:
        print(f"WARNING: side_fld={side_fld} or symbol_fld={symbol_fld} missing.")
        return df
    backup_side_fld = f"___{side_fld}"
    if backup_side_fld not in df:
        df[backup_side_fld] = df[side_fld]
    else:
        df[side_fld] = df[backup_side_fld]
    df[side_fld] = df.apply(
        SideConverter(
            away_asset=away_asset,
            symbol_fld=symbol_fld,
            side_fld=side_fld,
            logger=logger,
        ),
        axis=1,
    )
    unequal_sides_idx = df[side_fld] != df[backup_side_fld]
    backup_symbol_fld = f"___{symbol_fld}"
    if backup_symbol_fld not in df:
        df[backup_symbol_fld] = df[symbol_fld]
    else:
        df[symbol_fld] = df[backup_symbol_fld]
    df.loc[unequal_sides_idx, symbol_fld] = (
        df.loc[unequal_sides_idx, symbol_fld]
        .astype(str)
        .str.upper()
        .str.replace("/", "")
        .str.replace("([^/]+)_([^/]+)", r"\2_\1")
    )
    if amount_fld in df and price_fld in df:
        backup_amount_fld = f"___{amount_fld}"
        backup_price_fld = f"___{price_fld}"
        if backup_amount_fld not in df:
            df[backup_amount_fld] = df[amount_fld]
        else:
            df[amount_fld] = df[backup_amount_fld]
        if backup_price_fld not in df:
            df[backup_price_fld] = df[price_fld]
        else:
            df[price_fld] = df[backup_price_fld]
        df.loc[unequal_sides_idx, amount_fld] = (
            df.loc[unequal_sides_idx, amount_fld] * df.loc[unequal_sides_idx, price_fld]
        )
        df.loc[unequal_sides_idx, price_fld] = 1 / df.loc[unequal_sides_idx, price_fld]
    return df


class DfConverter:
    def __init__(
        self,
    ):
        self.complex_cols = defaultdict(bool)
        self.valid_keys = defaultdict(bool)
        self.valid_keys_lists = []
        self.compile_filter = None

    def _get_simple_pd_type_str(self, x):
        if x is None:
            return ""
        if isinstance(x, int):
            return "int"
        if isinstance(x, float):
            return "float"
        if isinstance(x, str):
            return "str"
        if isinstance(x, datetime.datetime):
            return "datetime"
        if isinstance(x, datetime.timedelta):
            return "timedelta"
        if isinstance(x, list):
            if len(x) <= 2:
                return "str"
        return "complex"

    def is_valid_key(
        self,
        key: str,
    ) -> bool:
        if self.compiled_filter is None:
            return True
        try:
            is_valid_key = self.valid_keys[key]
        except Exception:
            is_valid_key = None
        if is_valid_key is None:
            is_valid_key = bool(self.compiled_filter.search(key))
            if is_valid_key:
                self.add_valid_key(key)
            self.valid_keys[key] = is_valid_key
        return is_valid_key

    def add_valid_key(
        self,
        key: str,
    ):
        key_split = key.split("__")
        for i, elt in enumerate(key_split):
            if i < len(self.valid_keys_lists):
                self.valid_keys_lists[i].add(elt)
            else:
                self.valid_keys_lists.append(set([elt]))

    def flatten_reduced_dict(
        self,
        d: dict,
        counter: int = 0,
        pre_key: str = "",
    ):
        if isinstance(d, dict):
            this_d = {}
            try:
                valid_keys_list = self.valid_keys_lists[counter]
            except:
                valid_keys_list = None
            for k, v in d.items():
                this_key = f"{pre_key}__{k}" if pre_key else k
                if valid_keys_list and k not in valid_keys_list:
                    continue
                if not isinstance(v, dict):
                    this_d[this_key] = v
                else:
                    this_d.update(
                        self.flatten_reduced_dict(
                            v, counter=counter + 1, pre_key=this_key
                        )
                    )
            return this_d
        return d

    def flatten_into_df(
        self,
        list_of_records: List[Dict],
        drop_complex_flds: bool = False,
        prevent_int_overflow: bool = False,
        match_regex_list=None,
        reset_valid_keys: bool = True,
        sim_iteration_frq: int = None,
        activate_reduce_dict: bool = False,
    ) -> pd.DataFrame:
        """Convert typical list of dicts into a flattened DataFrame.

        Arguments:
            list_of_records (List[Dict]): list of dict records which may have
                nested values.
            drop_complex_flds (bool): whether do drop fields which are of complex type
                simple types are defined as isinstance(x, (float, int, str)).
                (defaults to False).
            prevent_int_overflow (bool): whether to convert all large ints to float
                to prevent overflow exception. (defaults to True).
            match_regex (str): regular expression filter for flattened column names to
                keep in the Dataframe. (defaults to '.*' and will keep all fields)

        Returns:
            (pd.DataFrame) flattened results.
        """
        if reset_valid_keys:
            self.valid_keys = {}
            self.valid_keys_lists = []

        def get_regex(l):
            if not l:
                return l
            return "(" + "|".join(l) + ")"

        match_regex = get_regex(match_regex_list) or ".*"
        self.compiled_filter = re.compile(match_regex) if match_regex != ".*" else None
        list_of_records = [
            {
                fk: fv
                for fk, fv in self.flatten_reduced_dict(o).items()
                if self.is_valid_key(fk)
                and not self.is_complex_col(fk, fv, drop_complex_flds)
            }
            for i, o in enumerate(
                tqdm_notebook(list_of_records, desc="Flattening sim_res into df")
            )
            if self.is_valid_period(i, sim_iteration_frq)
        ]
        if prevent_int_overflow:
            min_int = -(2 ** 63)
            max_int = 2 ** 64 - 1
            list_of_records = [
                {
                    k: (str(v) if not (min_int <= v <= max_int) else v)
                    if pd.api.types.is_number(v)
                    else v
                    for k, v in rec.items()
                }
                for rec in list_of_records
                if rec
            ]
        df = pd.DataFrame(list_of_records)
        return df

    def is_valid_period(
        self,
        i,
        sim_iteration_frq: int = None,
    ) -> bool:
        if sim_iteration_frq is None:
            return True
        return i % sim_iteration_frq == 0

    def is_complex_col(
        self,
        key,
        value,
        drop_complex_flds: bool = False,
    ) -> bool:
        if not drop_complex_flds:
            return False
        complex_col = self.complex_cols.get(key)
        if complex_col is None:
            col_type = self._get_simple_pd_type_str(value)
            if "complex" in col_type:
                complex_col = True
                self.complex_cols[key] = complex_col
        return complex_col

    def get_df_converted(
        self,
        list_of_records: List[Dict],
        away_asset: Optional[str] = None,
        drop_complex_flds: bool = False,
        match_regex_list=None,
        reset_valid_keys: bool = True,
        sim_iteration_frq: int = None,
        activate_reduce_dict: bool = False,
    ) -> pd.DataFrame:
        """Convert a list of dicts into a flattened DataFrame with converted values.

        Arguments:
            list_of_records (List[Dict]): list of dict records which may have
                nested values.
            away_asset (str): away asset (i.e. base asset with amounts). If provided,
                then will attempt to unify the sides (defaults to None).
            drop_complex_flds (bool): whether do drop fields which are of complex type
                simple types are defined as isinstance(x, (float, int, str)).
                (defaults to False).
            match_regex (str): regular expression filter for flattened column names to
                keep in the Dataframe. (defaults to '.*' and will keep all fields)

        Returns:
            (pd.DataFrame) with converted values.
        """
        df = self.flatten_into_df(
            list_of_records,
            drop_complex_flds=drop_complex_flds,
            match_regex_list=match_regex_list,
            reset_valid_keys=reset_valid_keys,
            sim_iteration_frq=sim_iteration_frq,
            activate_reduce_dict=activate_reduce_dict,
        )
        return self.convert_df_vals(df, away_asset=away_asset)

    def convert_df_vals(
        self,
        df: pd.DataFrame,
        away_asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """Use heuristics to convert dataframe values into useful units.

        Arguments:
            df (pd.DataFrame): DataFrame to convert values for.
            away_asset (str): away asset (i.e. base asset with amounts). If provided,
                then will attempt to unify the sides (defaults to None).
        Return:
            (pd.DataFrame) with converted values (not a copy).
        """
        # convert amount fields from big integer to float
        amount_flds = [
            c
            for c in df
            if re.search(
                "nav|pnl|price[s]?|spread|(im)?balance|best_bid|best_ask|best_quotes|volume|amount|fee__cost|funding_rate",
                c,
                re.IGNORECASE,
            )
            and not re.search(
                "(timestamp|coeff|weight|drift"
                "|drift_spread|order_book_imbalance(_spread)?"
                ")$",
                c,
                re.IGNORECASE,
            )
        ]
        for fld in amount_flds:
            if pd.api.types.is_numeric_dtype(df[fld]):
                df[fld] /= 10 ** 12
        # convert all timestamp fields from ms from epoch to datetime
        ts_flds = sorted([c for c in df if re.search("timestamp$", c, re.IGNORECASE)])
        for fld in ts_flds:
            try:
                df[fld].replace(0, np.NaN, inplace=True)
                df[fld] = pd.to_datetime(df[fld], unit="ms")
            except:
                pass
        if "executed_timestamp" in df:
            df = df.set_index("executed_timestamp")
        elif "timestamp" in df:
            df = df.set_index("timestamp")
        elif ts_flds:
            df = df.rename(columns={ts_flds[0]: "timestamp"}).set_index("timestamp")

        if away_asset:
            df = unify_side(df, away_asset=away_asset)

        # Add some useful fields if they exist
        if "side" in df:
            if "amount" in df:
                df["sign"] = 1
                df.loc[df["side"] == "sell", "sign"] = -1
                df["signed_amount"] = df["amount"] * df["sign"]
        return df


def get_df_with_col_types(
    list_of_lists: List[List],
    col_types: List[Tuple[str, str]],
) -> pd.DataFrame:
    """

    Arguments:
        list_of_lists (List[List]]): List of row values.
        col_types (List[Tuple[str, str]]): List of tuples where first element
            is the column name and the second element is the column type.

    Returns:
        (pd.DataFrame) with timeseries of results.
    """
    df = pd.DataFrame(
        [x for x in list_of_lists if x], columns=[x[0] for x in col_types]
    )
    for col_name, col_type in col_types:
        if col_type == "bigint":
            df[col_name] /= 10 ** 12
        else:
            df[col_name] = df[col_name].astype(col_type)
    if "timestamp" in df:
        df = df.set_index("timestamp")
    return df


def get_mdp_member_as_df(
    mdp,
    member: str,
    col_types: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Get MarketDataProfiler member time-series as a DataFrame.

    Arguments:
        mdp (MarketDataProvider): instance of `::class::MarketDataProfiler`
        member (str): name of MDP member.
        col_types (List[Tuple[str, str]]): List of tuples where first element
            is the column name and the second element is the column type. If
            not provided, will try to get this information from the
            MarketDataProfiler's COLUMN_TYPES member.

    Returns:
        (pd.DataFrame) with timeseries of results.
    """
    col_types = col_types or mdp.COLUMN_TYPES[member]
    dfs = []
    for xch, inst_data in getattr(mdp, member).items():
        for inst, data in inst_data.items():
            df = get_df_with_col_types(data, col_types)
            df["exchange"] = xch
            df["instrument"] = inst
        dfs.append(df)
    df = pd.concat(dfs).sort_values(by=["timestamp", "exchange", "instrument"])
    return df.set_index(["timestamp", "exchange", "instrument"])


def is_df_col_plottable(
    df: Optional[pd.DataFrame],
    col: str,
    warn_logger: Optional[Callable] = None,
) -> bool:
    """Checks whether the given column of the given DataFrame can safely be plotted.

    Arguments:
        df (pd.DataFrame): pandas DataFrame object.
        col (str): column name to check for plotting.
        warn_logger (Optional[Callable]): a callable which can take a string.

    Returns:
        (bool) whether the column if plottable or not.
    """
    info = ""
    try:
        is_plottable = bool(
            isinstance(df, pd.DataFrame) and (col in df) and df[col].notnull().sum()
        )
    except Exception as e:
        is_plottable = False
        info = f"{traceback.format_exc()} -- {e}"
    if warn_logger and not is_plottable:
        occurence = None
        if col in df:
            occurence = df[col].notnull().sum()
        warn_logger(
            f"column={col} is not plottable: -- {info} -- {isinstance(df, pd.DataFrame)} -- {col in df} -- {occurence} -- {len(df)}"
        )
    return is_plottable


def is_non_empty_df(df: Optional[pd.DataFrame]) -> bool:
    """Checked whehter the given DataFrame is non-empty.

    Arguments:
        df (pd.DataFrame): pandas DataFrame.

    Returns:
        (bool) whether the input is a non-empty DataFrame.
    """
    return bool(isinstance(df, pd.DataFrame) and len(df))


def try_apply_query(query_str: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tries to Apply a query to a DataFrame and returns null if there are any errors.

    Arguments:
        query_str (str): query string to be applied for filtering.
        df (pd.DataFrame): pandas DataFrame.

    Returns:
        (Optional[pd.DataFrame]) filtered DataFrame.
    """
    try:
        if not query_str:
            return df
        return df.query(query_str)
    except Exception:
        LOGGER.warning("failed to apply query_str='%s'", query_str)
    return None


class RecordListMerger:
    """Object for iterating over a list of records to merge columns which are list of lists."""

    def __init__(self, match_regex=".*", column_names=None):
        self._data = defaultdict(list)
        self._match_regex = match_regex
        self._col_names = column_names

    def consume(self, rec):
        if rec:
            _ = [
                self._data[k].append(v)
                for k, v in rec.items()
                if isinstance(v, list) and re.match(self._match_regex, k, re.IGNORECASE)
            ]
        return self

    def get_merged_columns(self, as_df=True):
        res = {k: [x for l in v for x in l] for k, v in self._data.items()}
        if not as_df:
            return res
        res_dfs = {}
        for col_name, data in res.items():
            kwargs = dict(columns=self._col_names) if self._col_names else dict()
            res_dfs[col_name] = pd.DataFrame(data, **kwargs)
        return res_dfs
