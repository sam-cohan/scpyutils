"""
Utilities related to Pandas I/O operations.

Author: Sam Cohan
"""
import re
from typing import Callable, Iterator, Optional

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def read_parquet_in_chunks(
    file_path: str,
    read_chunk_size: int,
    yield_chunk_size: Optional[int] = None,
    cols_pattern: Optional[str] = None,
    ignore_cols_pattern: Optional[str] = None,
    query: Optional[str] = None,
    trans_func: Optional[Callable] = None,
) -> Iterator[pd.DataFrame]:
    """Reads a Parquet file in chunks and yields the filtered DataFrames.

    Args:
        file_path: The path to the Parquet file.
        read_chunk_size: The number of rows to read in each chunk.
        yield_chunk_size: The number of rows to yield in each chunk (default:
            None, which defaults to read_chunk_size).  If set to -1, all
            non-empty reads are yielded straight away.
        cols_pattern: Regular expression pattern to match column names to
            include (default: None, which includes all columns).
        ignore_cols_pattern: Regular expression pattern to match column names to
            ignore (default: None).
        query: A string representing the query to filter the data (default: None).
        trans_func: A function that takes the loaded dataframe and performs
            arbitrary transformations on it.

    Yields:
        A DataFrame chunk with the specified yield_chunk_size and applied query
        filter.
    """
    # pd.options.mode.copy_on_write = True
    parquet_file = pq.ParquetFile(file_path)
    num_rows = parquet_file.metadata.num_rows
    arrow_schema = parquet_file.metadata.schema.to_arrow_schema()
    all_columns = [field.name for field in arrow_schema]

    if cols_pattern is None and ignore_cols_pattern is None:
        columns = all_columns
    else:
        if cols_pattern is not None:
            columns = [col for col in all_columns if re.match(cols_pattern, col)]
        else:
            columns = all_columns

        if ignore_cols_pattern is not None:
            columns = [col for col in columns if not re.match(ignore_cols_pattern, col)]

    if yield_chunk_size is None:
        yield_chunk_size = read_chunk_size

    total_passed = 0
    chunk_df = pd.DataFrame()

    with tqdm(
        total=num_rows, unit="rows", desc="Reading Parquet", dynamic_ncols=True
    ) as pbar:
        for batch in parquet_file.iter_batches(
            batch_size=read_chunk_size, columns=columns
        ):
            df = batch.to_pandas()
            pbar.update(len(df))
            if trans_func:
                df = trans_func(df)

            if query is not None:
                df = df.query(query).copy()

            total_passed += len(df)
            pbar.set_postfix(passed=total_passed)

            if yield_chunk_size == -1:
                if not df.empty:
                    yield df
            else:
                chunk_df = pd.concat([chunk_df, df], ignore_index=True)
                while len(chunk_df) >= yield_chunk_size:
                    yield chunk_df.iloc[:yield_chunk_size].copy()
                    chunk_df = chunk_df.iloc[yield_chunk_size:].copy()

        if yield_chunk_size != -1 and not chunk_df.empty:
            yield chunk_df
