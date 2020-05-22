from collections import Counter
import datetime
import math
from multiprocessing import cpu_count, Pool
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas.api.types as types
from pandas import Interval

from .cacheutils import get_hash
from .formatutils import try_fmt_num
from .logutils import setup_logger


LOGGER = setup_logger(__name__)


def get_first_row_as_header(df):
    new_header = df.iloc[0]
    df1 = df[1:]
    df1.columns = new_header
    return df1


# compiled interval regex (start and end will be available via \1 and \2)
INTERVAL_RE = re.compile(r"[\(\[](-inf|[0-9.]+), *([0-9.]+|inf)[\)\]]")


def get_sorted_intervals(interval_str_list):
    """Given a list of interval strings, return the sorted list.
    This is often useful after a merge of two categorical interval fields
    loses its order.
    """
    return sorted(interval_str_list,
                  key=lambda x: float(
                      INTERVAL_RE.sub(r"\1", x)
                      if INTERVAL_RE.match(str(x)) else -np.inf))


class DataPartitioner:
    """Abstract base class for partitioning data and iteratin over the partitions."""

    def __call__(
        self,
        df: Union[List, pd.DataFrame, pd.Series, np.ndarray],
        **meta
    ):
        raise NotImplementedError()


class DataMerger:
    """Abstract base class for merging a list of data."""

    def __call__(
        self,
        res_list: Union[List[List], List[pd.DataFrame], List[pd.Series], List[np.ndarray]],
    ) -> Union[List, pd.DataFrame, pd.Series, np.ndarray]:
        raise NotImplementedError()


class GenericDataPartitioner(DataPartitioner):
    """Partition a list, DataFrame, Series, or numpy array"""

    def __init__(
        self,
        n_parts: Optional[int] = None,
        partition_fld: Optional[str] = None,
        partition_fld_needs_hashing: Optional[bool] = None,
        yield_partition_val: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Create an instance of GenericDataPartitioner.

        Arguments:
            n_parts (int): number of parts to partition the Dataframe to.
                Defaults to None and in that case you should provide a
                partition_fld. If not None and partition field is also
                provided will hash the partition_fld (unless suppressed by
                the partition_fld_needs_hashing param) and attempt to take
                mod with n_parts.
            partition_fld (str): field name to be used as the partition_fld.
                Make sure there are no more than 32 unique values. Defaults to
                None and in that case you should provide n_parts to do sequential
                partition. If not None and n_parts field is also provided, will
                hash the partition_fld (unless suppressed by the
                partition_fld_needs_hashing param) and attempt to take mod with
                n_parts.
            partition_fld_needs_hashing (bool): whether to skip costly step
                of hashing the partition_fld. (default values depend on null-ness
                of n_parts and partition_fld: defaults to False when only
                partition_fld is provided because it assumes you are picking a
                partition field whose values are already appropriate to create
                less than 32 approximately evenly-sized groups. defaults to True
                when when both n_parts and partition_fld are provide as the mod
                of the partition_fld is generally assumed to not be a good
                partitioner, but if you know otherwise feed free to force this to
                be True.)
            yield_partition_val (bool): whether the partition val should be
                sent as the second argument in the yield (default to False)

        Yields:
            (pd.DataFrame | pd.Series) of partitions
        """
        self.partition_fld = partition_fld
        self.n_parts = n_parts
        self.partition_fld_needs_hashing = partition_fld_needs_hashing
        self.yield_partition_val = yield_partition_val
        if partition_fld_needs_hashing is None:
            if self.n_parts and self.partition_fld:
                self.partition_fld_needs_hashing = True
            elif self.partition_fld:
                self.partition_fld_needs_hashing = False
        assert self.partition_fld or self.n_parts, \
            "Please provide n_parts or partition_fld to GenericDataPartitioner"
        if self.partition_fld_needs_hashing and partition_fld_needs_hashing is None:
            warn_msg = (
                "To get rid of this warning pass in the partition_fld_needs_hashing"
                f" for partition_fld='{self.partition_fld}' explicitly! Guessing you want it to be True.")
            LOGGER.warning(warn_msg)

    def __call__(self, df: Union[List, pd.DataFrame, pd.Series, np.ndarray], **meta):
        """Calculate partition values and iteratively yield them."""
        if self.partition_fld:
            if self.partition_fld == "__index__":
                if len(df.index.names) > 1:
                    partition_vals = np.array(
                        list(zip(*[df.index.get_level_values(i) for i in range(len(df.index.names))])))
                else:
                    partition_vals = df.index.values
            if self.partition_fld == "__vals__":
                partition_vals = df.values
            else:
                partition_vals = df[self.partition_fld].values
        elif self.n_parts:
            len_df = len(df)
            partition_vals = np.zeros(len_df)
            len_part = math.ceil(len_df / self.n_parts)
            mark_idx = len_part
            if isinstance(df, (list, np.ndarray)):
                partition_vals = list(
                    range(len_part, len_df, len_part)) + [len_df - 1]
                print(partition_vals, self.n_parts, len_df, len_part)
            else:
                while mark_idx < len_df:
                    partition_vals[mark_idx] = 1
                    mark_idx += len_part
                partition_vals = np.cumsum(partition_vals)

        if self.partition_fld_needs_hashing:
            import hashlib
            partition_vals = np.vectorize(
                lambda x: int(hashlib.sha1(
                    str(x).encode()).hexdigest()[-2:], 16)
            )(partition_vals)
        if self.n_parts and self.partition_fld:
            partition_vals = np.remainder(partition_vals, self.n_parts)
        assert not np.any(pd.isnull(partition_vals)
                          ), "ERROR: partition values cannot be null"
        unique_partition_vals = sorted(set(partition_vals))
        assert len(unique_partition_vals) <= 64, "ERROR: {:,.0f} is too many partitions to handle".format(
            len(unique_partition_vals))
        for i, partition_val in enumerate(unique_partition_vals):
            if isinstance(df, (list, np.ndarray)):
                partition = df[(i or unique_partition_vals[i]): partition_val]
            else:
                partition = df.loc[partition_vals == partition_val]
                # _is_copy is deprecated as of 0.23.0, may have to live with warning
                partition._is_copy = False  # pylint: disable=protected-access
            debug_msg = f"yielding result of length {len(partition):,.0f} records for partition_val={partition_val}"
            LOGGER.debug(debug_msg)
            if self.yield_partition_val:
                yield partition, partition_val
            else:
                yield partition


class GenericDataMerger(DataMerger):
    def __call__(
        self,
        res_list: Union[List[List], List[pd.DataFrame], List[pd.Series], List[np.ndarray]],
    ) -> Union[List, pd.DataFrame, pd.Series, np.ndarray]:
        """Concatenates a list of lists, DataFrames, Series, or numpy arrays.

        This either does list comprehension, delegates to np.concatenate,
        or is a wrapper around pd.concat which makes an effort to change any
        altered dtypes back to their original dtype.
        """
        if not len(res_list):  # pylint: disable=len-as-condition
            return res_list
        first = res_list[0]
        if isinstance(first, list):
            return [x for lst in res_list for x in lst]
        if isinstance(first, np.ndarray):
            return np.concatenate(res_list)
        dtypes = res_list[0].dtypes
        df = pd.concat(res_list)
        if not isinstance(dtypes, pd.Series):
            return df
        # concat sometimes messed up the types (e.g. categories may turn into string). Force them back
        for fld, dtype in dtypes.iteritems():
            new_dtype = str(df[fld].dtype)
            old_dtype = str(dtype)
            if new_dtype != old_dtype:
                df[fld] = df[fld].astype(old_dtype)
            if old_dtype == "category":
                # This is a crude workaround which does not guarantee original sort order
                # force the sort order to be string-based with fields which are already string at the top
                df[fld].cat.reorder_categories(
                    sorted(df[fld].cat.categories,
                           key=lambda x: str(x) if not isinstance(x, str) else " {}".format(x)),
                    ordered=True,
                    inplace=True,)
        return df


def apply_parallel(
        func: Callable,
        df: Union[List, pd.DataFrame, pd.Series, np.ndarray],
        partitioner: DataPartitioner = None,
        merger: DataMerger = None,
        pool_size: Optional[int] = None
) -> Union[pd.DataFrame, pd.Series]:
    """Apply a function to a DataFrame with n_parts parallel processes.

    Arguments:
        func (Callable): function which takes a slice data and outputs the
            translated data.
        df (List, pd.DataFrame | pd.Series | np.ndarray): the data.
        partitioner (DataPartitioner): an iterator which iterates over
            data partitions. (defaults to `GenericDataPartitioner`)
        merger (DataMerger): function that takes a list of all processed
            data and merges them. (defaults to `GenericDataMerger`)
        pool_size (int): size of process pool. (defaults to None and will
            use as many processes as many processes and CPUs.)

    Returns:
        (List | pd.DataFrame | pd.Series | np.ndarray) data with the
        `func` applied to them.
    """
    pool_size = pool_size or cpu_count()
    pool = Pool(pool_size)
    if partitioner is None:
        partitioner = GenericDataPartitioner(n_parts=pool_size)
    if merger is None:
        merger = GenericDataMerger()
    try:
        res_list = pool.map(func, partitioner(df))
    except Exception as e:
        LOGGER.exception(e)
        raise e
    finally:
        pool.close()
        pool.join()
    return merger(res_list)


def get_pd_friendly_col_name(col: Any) -> str:
    return re.sub(
        "[^A-Za-z0-9_]", "_",
        re.sub("^([^_A-Za-z])", "_\g<1>", str(col)))  # pylint: disable=anomalous-backslash-in-string  # noqa=W605


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join([str(x) for x in col]) for col in df]
    return df


def get_date_from_epoch(
    sr: Union[pd.Series, Union[List[float], float]],
    unit: str = "ms",
) -> datetime.date:
    """Get a date object from a series or list of epochs."""
    return datetime.date(pd.to_datetime(sr, unit=unit).dt.floor("1D"))


def get_datetime_int_from_epoch(
    sr: Union[pd.Series, List[float], float],
    unit: str = "ms",
) -> int:
    """Get a an integer representation of the date.
    This is useful if you want a readable date which takes much less memory,
    however, be careful as they are not scaled correctly for plotting. It is
    more useful for grouping.

    Returns:
        (int) something like 20190101.
    """
    return pd.to_datetime(sr, unit=unit).dt.floor("1D").astype("str").str.replace("-", "").astype(int)


def is_null_or_zero(x: Any) -> bool:
    if not x:
        return True
    if pd.isnull(x):
        return True
    if isinstance(x, str):
        return x in ["None", "nan"]
    return False


def isfinite(x: Any) -> bool:
    """Check if input is a numeric and finite.
    This is useful as np.finite throws if you pass non-float to it.
    """
    if isinstance(x, (float, int)):
        return np.isfinite(x)
    else:
        return False


def unique_count(sr: Union[pd.Series, List]) -> int:
    """Count unique not-null values in the passed series"""
    return len(set([x for x in sr if pd.notnull(x)]))


def reduce_unique_sorted(
        sr: Union[pd.Series, List],
        old_delim: str = ", ",
        new_delim: str = ", ") -> List[str]:
    """Given a series, force it to be string and reduce it into a delim-separated sorted set.
    Useful for aggregating groupby results.
    """
    return new_delim.join(
        sorted(set([val for vals in sr for val in str(vals).split(old_delim)])))


def reduce_value_counts(sr: Union[pd.Series, List]) -> List[Any]:
    """Given a series, return the value_counts() as a sorted list of tuples."""
    ctr = Counter()
    for x in sr:
        # null values which comes from pd.Series will not equate and will mess up
        # the counter, pd.NaN works fine.
        if pd.isnull(x):
            x = np.NaN
        ctr[x] += 1
    return sorted(ctr.items(), key=lambda x: x[1], reverse=True)


def reduce_value_counts_as_str(sr: Union[pd.Series, List]) -> str:
    """Given a series, return the value_counts() in sorted string representation."""
    return " | ".join(["{}: {:.0f}".format(val, cnt) for val, cnt in reduce_value_counts(sr)])


def try_strip(x: Any) -> Any:
    """Try to strip strings."""
    return x.strip() if isinstance(x, str) else x


def force_strip_str(x: Any) -> str:
    "Force stripping regardless of whether it is string or not (force everything to be string)."
    return x.strip() if isinstance(x, str) else "" if pd.isnull(x) else str(x).strip()


def get_unique_sorted(
    sr: pd.Series,
    old_delim: str = ", ",
    new_delim: str = ", ",
    fillna: str = ''
):
    """Given a Series which points to old_delim-separated string data, returns a new Series
    with new_delim-separated string data which is unique and sorted
    (Useful for cleaning up SQL returned data as this operation is hard to do in SQL).
    By default fills na values with empty string.
    """
    if not old_delim or not isinstance(old_delim, str):
        raise Exception("old_delim must be non-empty string")

    def try_unique_sorted(st):
        try:
            return new_delim.join(sorted(set(st.split(old_delim))))
        except AttributeError as e:
            if not e.args[0].startswith("'NoneType' object has no attribute 'split'"):
                raise e
        return st

    sr = sr.apply(try_unique_sorted)
    if fillna is not None:
        return sr.astype(str).fillna(fillna)
    return sr


def dummy_agg(func_name: str) -> Callable:
    """If you are aggregating a groupby where some fields are a function of others,
    it is much faster to reserve room for them and then fill them in later."""
    def agg_func(_):
        return np.nan
    agg_func.__name__ = func_name
    return agg_func


def apply_dim_filters(
    df: pd.DataFrame,
    dim_filters: Dict,
    verbose: bool = True
):
    """Given a DataFrame with MultiIndex, and a dictionary mapping index field
    names to list of values, return a view of the df with only matching records.
    """
    if set(dim_filters) - set(df.index.names):
        raise Exception(
            f"dim_filter={list(dim_filters)} not all available in index={df.index.names}")
    indexer = tuple(dim_filters.get(fld, slice(None))
                    for fld in df.index.names)
    if verbose:
        info_msg = f"Applying dim_filters: {dim_filters} (size={len(df):,.0f})... "
        print(info_msg, end="", flush=True)
    df = df.loc[indexer, :]
    if verbose:
        info_msg = f"done (size={len(df):,.0f})."
        print(info_msg, flush=True)
    return df


def safe_fillna(
    df: pd.DataFrame,
    col: str,
    fill_val: Any = "",
) -> pd.DataFrame:
    """fillna which does not fail on category fields, instead adds the category.

    Arguments:
        df (pd.DataFrame): Dataframe to fill na values for.
        col (str): column name.
        fill_val (Any): value to fill with.

    Returns:
        (pd.DataFrame) with NA values filled safely. (not a copy)
    """
    if df is None or not len(df):  # pylint: disable=len-as-condition
        return df
    vals = df[col]
    if str(vals.dtype) == "category":
        if np.any(pd.isnull(vals)):
            if fill_val not in vals.cat.categories:
                vals = vals.cat.add_categories(fill_val)
            vals.fillna(fill_val, inplace=True)
    else:
        vals.fillna(fill_val, inplace=True)
    return vals


def safe_sort(
        df: pd.DataFrame,
        sort_fld: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        inplace: bool = False
) -> pd.DataFrame:
    """Sort a DataFrame regardless of whether the field is in index or columns.
    Note that if a field is not sortable because of TypeError, this will
    still throw an exception.

    Arguments:
        df (pd.DataFrame): DataFrame to sort.
        sort_fld (str, List[str]): field or list of fields to sort by. If you
            want to sort by index pass "__index__".
        ascending (bool, List[bool]): whether sort order should be ascending.
        inplace (bool): whether sort should be done on the same DataFrame (to
            save memory).

    Returns:
        (pd.DataFrame) sorted safely.
    """
    if sort_fld == "__index__":
        if inplace:
            df.sort_index(ascending=ascending, inplace=True, kind="mergesort")
        else:
            return df.sort_index(ascending=ascending, kind="mergesort")
    sort_func = None
    if isinstance(sort_fld, list):
        if not set(sort_fld) - set(df.columns):
            sort_func = "vals"
        elif not set(sort_fld) - set(df.index.names):
            sort_func = "index"
    else:
        if sort_fld in df.columns:
            sort_func = "vals"
        elif sort_fld in df.index.names:
            sort_func = "index"
        else:
            warn_msg = (
                f"sort_fld={sort_fld} not in index={list(df.index.names)}"
                f" or columns={list(df.columns.values)}")
            LOGGER.warning(warn_msg)
            return df

    if sort_func == "vals":
        if inplace:
            df.sort_values(by=sort_fld, ascending=ascending,
                           inplace=True, kind="mergesort")
        else:
            return df.sort_values(by=sort_fld, ascending=ascending, kind="mergesort")
    elif sort_func == "index":
        if inplace:
            df.sort_index(level=sort_fld, ascending=ascending,
                          inplace=True, kind="mergesort")
        else:
            return df.sort_index(level=sort_fld, ascending=ascending, kind="mergesort")


def get_sorted_groups(
        df: pd.DataFrame,
        sort_fld: str,
        groupby_flds: List[str] = None,
        ascending: bool = False,
        drop_group_totals: bool = True
) -> pd.DataFrame:
    """Sort a DataFrame by its group totals. (uses stable sorting algorithm).

    Arguments:
        df (pd.DataFrame): DataFrame object
        sort_fld (str): name of field to be summed for sorting
        groupby_flds (List[Str]): subset of fields in the index which were
            used to define the sort groups. (defaults to None, which results
            in taking all but the last index field).
        ascending (bool): whether groups should be sorted by increasing
            order. (default to False).
        drop_group_totals (bool): whether the intermediate group totals which
            are used for sorting should be dropped or not. (defaults to True
            and drops them).

    Returns:
        (pd.DataFrame) sorted by group totals.
    """
    if groupby_flds is None:
        groupby_flds = df.index.names[:-1]
    totals_fld = "{}_{}_totals".format(
        "_".join([str(c) for c in groupby_flds]), sort_fld)
    df[totals_fld] = df.groupby(groupby_flds)[sort_fld].transform(np.sum)
    # df = df.sort_values(by=totals_fld, ascending=ascending, kind="mergesort")
    df = safe_sort(df, sort_fld=totals_fld, ascending=ascending)
    return df.drop(totals_fld, axis=1) if drop_group_totals else df


def adorn_with_pcnt(
    df: pd.DataFrame,
    fld: Union[str, List[str]],
    cum: bool = True,
    sort_fld: Union[str, List[str]] = None,
    ascending: Union[bool, List[bool]] = False,
    groupby_flds: Union[str, List[str]] = None,
) -> pd.DataFrame:
    """Given a DataFrame and list of field names, for each field add the percent and
    cumulative percent fields.

    Arguments:
        df (pd.DataFrame): Dataframe with possible multi-index and at least a numerical
            column to sum.
        fld (str | List[str]): name of field(s) to calculate percentages for.
        cum (bool): whether the cumulative sum field should be added.
        sort_fld (str | List[str]): field (or list of field names) to sort final results by
            before calculating cumulative counts (default to None meaning
            data is assumed to be sorted).
        ascending (bool | List[bool]): boolean (or list of booleans) indicating initial sort
            order (default to False meaning sort order is largest to
            smallest), Note that groupby_sort_order overrides this
            sort order.
        groupby_flds (str | List[str]): field (or list of field names) to
            group the percentages by (default to None meaning everything is in one group).

    Returns:
        (pd.DataFrame) with percent columns added.
    """
    if not isinstance(fld, list):
        fld = [fld]
    if sort_fld is not None:
        df = safe_sort(df, sort_fld=sort_fld, ascending=ascending)
    else:
        sort_fld = fld[0]
    if groupby_flds:
        if not isinstance(groupby_flds, list):
            groupby_flds = [groupby_flds]
        if set(groupby_flds) - set(df.columns.values):
            groupby_args = {"by": groupby_flds}
        elif set(groupby_flds) - set(df.index.names):
            groupby_args = {"level": groupby_flds}
        else:
            raise Exception("groupby_flds={} must be in the index or columns")
    for f in fld:
        pcnt_fld = "{}_pcnt".format(f)
        if groupby_flds:
            group_pcnt_fld = "{}_{}_pcnt".format(
                "_".join([str(x) for x in groupby_flds]), f)
            df[group_pcnt_fld] = df[f].groupby(
                **groupby_args).transform(np.sum)
            df[group_pcnt_fld] = df[f] / df[group_pcnt_fld]
            if sort_fld in df and types.is_numeric_dtype(df[sort_fld]):
                df = get_sorted_groups(
                    df, sort_fld=sort_fld, groupby_flds=groupby_flds, ascending=ascending)
            else:
                df = safe_sort(df, sort_fld=groupby_flds, ascending=ascending)
            if cum:
                df["{}_cum".format(group_pcnt_fld)] = df[group_pcnt_fld].groupby(
                    **groupby_args).transform(np.cumsum)
        df[pcnt_fld] = df[f] / df[f].sum()
        if cum:
            df["{}_cum".format(pcnt_fld)] = df[pcnt_fld].cumsum()
    return df


def get_filtered_groups(
        df: pd.DataFrame,
        filter_fld: str,
        filter_func: Callable,
        groupby_flds: List[str] = None,
) -> pd.DataFrame:
    """Filter DataFrame based on a generic function applied to a field in each group.

    Arguments:
        df (pd.DataFrame): DataFrame object.
        filter_fld (str): name of field to use for filtering.
        filter_func (Callable): function which takes a series of filter_fld
            values and return either a single boolean or a boolean array of
            same dimension indicating which values to keep.
        groupby_flds (List[str]): subset of fields in the index which were
            used to define the groups. (defaults to None, which results in
            taking all but the last index field).

    Returns:
        (pd.DataFrame) filtered DataFrame.
    """
    if groupby_flds is None:
        groupby_flds = df.index.names[:-1]
    df["keep"] = df.groupby(groupby_flds)[filter_fld].transform(filter_func)
    return df[df["keep"] > 0].drop(columns="keep")


def get_binned(
        sr: pd.Series,
        bins: [List[float], np.ndarray] = None,
        bin_pctls: [List[float], np.ndarray] = None,
        fillna_val: float = None,
        na_bucket_label: str = "-",
        right: bool = True,
        include_lowest: bool = True
) -> pd.Series:
    """Given a Series and optional bins, return the Series of binned values.

    Arguments:
        sr (pd.Series): Series with numeric (possibly NaN) values.
        bins : array of binning boundaries (to avoid dropping make sure first
            and last are -np.inf and np.inf). (defaults to None and will try to
            figure out nice boundaries based on percentiles).
        bin_pctls: array of percentiles to be used as boundaries for binning.
            (defaults to None and will use the quantile values resulting from:
            [0, 0.01, 0.5, 0.1, ..., 0.9, 0.95, 0.99])
        fillna_val: if you think na values should be set to some valid value
            (e.g. 0), then set them here. (defaults to None and will not fill
            the na values).
        na_bucket_label (str): label to use for binning null values. Since pandas
            drops na values but this is often not the desired behavior,
            provide a label to the na bucket. (defaults to string "-").
        right (bool): whether the bins should be right-inclusive
        include_lowest (bool): whether the first interval should be
            left-inclusive or not. (default to True).

    Returns:
        (pd.Series) of binned values.
    """
    if fillna_val:
        sr = sr.fillna(fillna_val)
    else:
        sr = sr
    if not bins:
        unique_groupby_vals = sr.value_counts()
        if len(unique_groupby_vals) < 10:
            binned_vals = sr.astype("category", ordered=True).values
        else:
            bin_pctls = bin_pctls or (
                [0.01, 0.5] + list(np.linspace(.1, 0.9, 9)) + [0.95, 0.99])
            pctl_vals = sorted(set(sr.quantile(bin_pctls)))
            if sr.max() - sr.min() > 1:
                pctl_vals = sorted(set([int(x) for x in pctl_vals]))
            bins = [-np.inf] + pctl_vals + [np.inf]
            binned_vals = pd.cut(
                sr.values, bins=bins, right=right, include_lowest=include_lowest)
    else:
        binned_vals = pd.cut(
            sr.values, bins=bins, right=right, include_lowest=include_lowest)

    if na_bucket_label:
        if binned_vals.isnull().sum():
            # Add the label for the null to categories
            binned_vals = binned_vals.add_categories([na_bucket_label])
            # Make sure the null bucket is the first one
            binned_vals = binned_vals.reorder_categories(
                list(binned_vals.categories[-1:]) + list(binned_vals.categories[:-1]))
            # Actually rename the nulls to have the value
            binned_vals[binned_vals.isnull()] = na_bucket_label
    return binned_vals


def apply_filter_by_agg_query(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    agg_ts_query: Dict[str, Callable],
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given a DataFrame with a MultiIndex and another which has the same MultiIndex
    minus the last level, but instead has a MultiIndex column which represents the
    metrics as the first level and their stats as the second level, use queries
    on latter to filter the former.

    Arguemnts:
        df (pd.DataFrame): DataFrame with MultiIndex and singel level columns
        df_agg (pd.DataFrame) : DataFrame with same MultiIndex as df minus
            the last level, and MultiIndex columns where the first level is
            same as df and second level is aggregate stats from squashing the
            the last index level.
        agg_ts_query: map from metric name to query string to be applied to the
            stats.

    Returns:
        (Tuple[pd.DataFrame, pd.DataFrame]) filtered version of both
            Dataframes passed to it.
    """
    if verbose:
        LOGGER.info("Applying agg_ts_query: {} (size={:,.0f})... ".format(
            agg_ts_query, len(df)), end="", flush=True)
    # first apply the filters to the agg_df
    for col_name, query in agg_ts_query.items():
        qdf = agg_df[col_name].query(query)
        if len(qdf):  # pylint: disable=len-as-condition
            if isinstance(qdf.index, pd.MultiIndex):
                qdf_index = tuple(
                    [qdf.index.get_level_values(x).tolist()
                     for x in qdf.index.names])
            else:
                qdf_index = qdf.index
            agg_df = agg_df.loc[qdf_index, :]
        df = df.loc[tuple([agg_df[col_name].query(query).index.get_level_values(fld).tolist()
                           for fld in agg_df.index.names] + [slice(None)]), :]
    if verbose:
        LOGGER.info("done (size={:,.0f}).".format(len(df)), flush=True)
    return df, agg_df


def get_merged_interval_mappings(
    intervals: pd.Interval,
    boundaries: List[float]
) -> Dict[pd.Interval, List[Interval]]:
    """Given an iterable of intervals and some boundaries, returns a dictionary which maps the
    old intervals to a new set of merged intervals.

    Arguments:
        intervals (List[pd.Interval]): list of pd.Interval objects (typically
            from a pd.cut operation).
        boundaries (List[Boundary]): list of boundaries (must be a subset of
            the boundaries in the intervals list) (hint: to remap everything
            to same bucket, pass empty list or [-np.inf, np.inf]).

    Returns:
        (Dict[pd.Interval, List[Interval]]) mapping from old interval to list of new intervals.
    """
    intervals = sorted(set(intervals))
    allowed_boundaries = set([b for interval in intervals for b in [
                             interval.left, interval.right]])
    finite_boundaries = [b for b in boundaries if b not in [-np.inf, np.inf]]
    boundaries = sorted([-np.inf] + list(finite_boundaries) + [np.inf])
    assert set(finite_boundaries).issubset(allowed_boundaries), \
        "boundaries needs to be a subset of {}".format(
            sorted(allowed_boundaries))
    if not finite_boundaries:
        new_intervals = [Interval(-np.inf, np.inf, closed='both')]
        interval_remap_index = [0 for _ in intervals]
    else:
        new_intervals = [
            Interval(left, boundaries[i + 1], closed='right')
            for i, left in enumerate(boundaries[:-1])]
        interval_remap_index = np.digitize(
            [interval.right for interval in intervals], finite_boundaries, right=True)
    remaps = {b: new_intervals[interval_remap_index[i]]
              for i, b in enumerate(intervals)}
    return remaps


def get_anomalies(
        df: pd.DataFrame,
        field: str,
        lower_pctl: int = 5,
        upper_pctl: int = 95,
        print_stats: bool = True
):
    pctls = []
    lower_pctl_val = np.NaN
    upper_pctl_val = np.NaN
    if pd.isnull(lower_pctl):
        lower_pctl = np.NaN
    else:
        pctls.append(lower_pctl)
        lower_pctl_val = 0
    if pd.isnull(upper_pctl):
        upper_pctl = np.NaN
    else:
        pctls.append(upper_pctl)
        upper_pctl_val = len(pctls) - 1
    pctl_vals = np.nanpercentile(df[field].replace(np.inf, np.nan), pctls)
    if not pd.isnull(lower_pctl_val):
        lower_pctl_val = pctl_vals[lower_pctl_val]
    if not pd.isnull(upper_pctl_val):
        upper_pctl_val = pctl_vals[upper_pctl_val]
    if print_stats:
        print("{} stats: mean={}, median={}, min={}, max={}".format(
            *[try_fmt_num(x, lambda x: x) for x in [field,
                                                    np.mean(df[field]),
                                                    np.median(df[field]),
                                                    np.min(df[field]),
                                                    np.max(df[field])]]))
    print("Anomalies defined as {} value being outside percentile"
          " ({:.0f}%, {:.0f}%) = ({:,.4f}, {:,.4f})".format(
              field, lower_pctl, upper_pctl, lower_pctl_val, upper_pctl_val))
    dff = df[(
        (lower_pctl_val is not None) & (df[field] < lower_pctl_val)
    ) | (
        (upper_pctl_val is not None) & (df[field] > upper_pctl_val)
    )]
    print("%s of %s records were flagged as anomalous" % (len(dff), len(df)))
    return dff


class DataFrameFilterByQuery(object):

    def __init__(self, query, copy=True):
        self.query = query
        self.copy = copy
        self.__name__ = "DataFrameFilter_{}".format(query)
        self.__hash_override__ = get_hash(self.__name__)

    def __call__(self, df, **meta):
        df = df.query(self.query)
        if self.copy:
            df = df.copy()
        return df

    def __str__(self):
        return self.__name__

    __repr__ = __str__


class DataFrameFilterInVals(object):

    def __init__(self, fld, vals, negate=False, copy=True):
        self.fld = fld
        self.vals = set(vals)
        self.negate = negate
        self.copy = copy
        try:
            sorted_vals = sorted(vals)
        except TypeError:  # not everything is sortable!
            sorted_vals = vals
        str_vals = str(sorted_vals) if len(
            sorted_vals) < 100 else get_hash(sorted_vals)
        self.__name__ = "DataFrameFilter_{}_{}in_{}".format(
            fld, "not_" if negate else "", str_vals)
        self.__hash_override__ = get_hash(self.__name__)

    def __call__(self, df, **meta):
        selection = df[self.fld].isin(self.vals)
        if self.negate:
            selection = ~selection
        df = df[selection]
        if self.copy:
            df = df.copy()
        return df

    def __str__(self):
        return self.__name__

    __repr__ = __str__


class DataFrameFilterRegexp(object):

    def __init__(self, fld, regexp, flags=0, negate=False, copy=True):
        self.fld = fld
        self.regexp = regexp
        self.flags = flags
        self.negate = negate
        self.copy = copy
        self.__name__ = "DataFrameFilter_{}_{}matches_{}_flags={}".format(
            fld, "not_" if negate else "", regexp, flags)
        self.__hash_override__ = get_hash(self.__name__)

    def __call__(self, df, **meta):
        selection = df[self.fld].str.match(self.regexp, flags=self.flags)
        if self.negate:
            selection = ~selection
        df = df[selection]
        if self.copy:
            df = df.copy()
        return df

    def __str__(self):
        return self.__name__

    __repr__ = __str__


class SeriesReturnExtractor:
    def __init__(
        self,
        interval_td,
        min_interval_ratio=0.9,
        max_interval_ratio=1.1,
        diff_only=False
    ):
        self._interval_td = pd.to_timedelta(interval_td)
        self._interval_ms = int(self._interval_td.to_timedelta64()) / 1e6
        self._min_interval_td = pd.to_timedelta(
            self._interval_ms * min_interval_ratio, unit="ms")
        self._max_interval_td = pd.to_timedelta(
            self._interval_ms * max_interval_ratio, unit="ms")
        self._diff_only = diff_only

    def __call__(self, series):
        if series is None or len(series) < 2:
            return np. NaN
        start_ts, end_ts = series.index[[0, -1]]
        ts_diff = end_ts - start_ts
        if ts_diff < self._min_interval_td or ts_diff > self._max_interval_td:
            return np.NaN
        ts_diff_ms = int(ts_diff.to_timedelta64()) / 1e6
        scale_factor = self._interval_ms / ts_diff_ms
        start_val, end_val = series.iloc[[0, -1]]
        diff = (end_val - start_val) * scale_factor
        if self._diff_only:
            return diff
        if start_val:
            return diff / start_val
        return np. NaN

    def __str__(self):
        return (
            "SeriesReturn("
            f"diff_only={self._diff_only}"
            f", interval_td={self._interval_td}"
            f", min_interval_td={self._min_interval_td}",
            f", max_interval_td={self._max_interval_td}",
            ")"
        )

    __repr__ = __str__
