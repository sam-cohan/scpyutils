"""
Utilities related to caching function results.

Author: Sam Cohan
"""

import hashlib
import inspect
import logging
import os
import re
import time
from functools import wraps
from typing import Any, Callable, Hashable, Optional, Union

import dill
import joblib
import pandas as pd

import scpyutils.persistutils as pstu


class HashSafeWrapper:
    """Class for wrapping objects to make them safe for use with memorize.

    The main use of this class is to wrap functions and classes which need to be
    passed as arguments to `memorize`. By default, the class will take the source of
    the function or class and keep its hash on the `__hash_override__` attribute as
    required by `memorize`. To actually decorate your functions, make use of
    `wrap_for_memorize` function defined in this module.

    """

    KNOWN_TYPES = ["class", "function"]

    def __init__(  # noqa: C901
        self,
        obj: Any,
        strict: bool = True,
        hash_salt: Optional[Any] = None,
        hash_override: Optional[str] = None,
    ):
        """Get a HashSafeWrapper instance.

        Args:
            obj: Any object (typically a function or class).
            strict: Whether to get the source code for functions and classes. If False,
                will try to use `__name__` or simply get the hash of the object string.
                (defaults to True)
            hash_salt: An optional salt used for hashing. This can be useful for
                debugging. (defaults to None and no salt will be used).
            hash_override: Provide a hash to override all other behavior. This can be
                useful for debugging. (defaults to None)
        """
        self._obj = obj
        self._hash_salt = hash_salt
        self._strict = strict
        self._hash_override = hash_override
        if hash_override:
            if strict:
                raise ValueError("hash_override not compatible with strict!")
            if hash_salt:
                raise ValueError("hash_override not compatible with hash_salt!")
        self._obj_type = (
            "function"
            if inspect.isfunction(obj)
            else "class" if inspect.isclass(obj) else "?"
        )
        if self._obj_type == "function":
            self.__doc__ = obj.__doc__
            self._name = get_func_name(obj, include_module=not strict)
        elif self._obj_type == "class":
            self.__doc__ = obj.__doc__
            self._name = obj.__name__ if strict else f"{obj.__module__}.{obj.__name__}"
        else:
            self._name = getattr(obj, "__name__", str(obj))
        if hash_override is not None:
            self.__hash_override__ = hash_override
        else:
            if hasattr(obj, "__hash_override__"):
                self.__hash_override__ = obj.__hash_override__
            else:
                if self._obj_type in self.KNOWN_TYPES:
                    if strict:
                        obj_src = inspect.getsource(obj)
                    else:
                        obj_src = f"<{self._obj_type} {self._name}>"
                else:
                    obj_src = str(obj)
                    if "object at 0x" in obj_src:
                        raise Exception(f"ERROR: obj={obj_src} cannot be made HashSafe")
                self.__hash_override__ = get_hash(
                    obj_src if hash_salt is None else (obj_src, hash_salt)
                )
        if self._obj_type in self.KNOWN_TYPES:
            hash_in_name = f" src_hash:{self.__hash_override__[:16]}" if strict else ""
            self.__name__ = f"<{self._obj_type} {self._name}{hash_in_name}>"
        else:
            self.__name__ = self._name

    def __hash__(self) -> str:  # type: ignore
        return self.__hash_override__

    def override_hash(self, hash_: str) -> "HashSafeWrapper":
        self.__hash_override__ = hash_
        if self._obj_type in self.KNOWN_TYPES:
            hash_in_name = f" src_hash:{hash_[:16]}" if self._strict else ""
            self.__name__ = f"<{self._obj_type} {self._name}{hash_in_name}>"
        return self

    @property
    def obj(self) -> Any:
        return self._obj

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._obj(*args, **kwargs)

    def __str__(self) -> str:
        return self.__name__

    __repr__ = __str__


def get_hash(x: Any) -> str:
    hash_override = getattr(x, "__hash_override__", "")
    if hash_override:
        return hash_override
    if isinstance(x, dict):
        return get_hash(str(sorted([(k, get_hash(v)) for k, v in x.items()])))
    if isinstance(x, set):
        return get_hash(str(sorted([get_hash(xx) for xx in x])))
    if isinstance(x, (list, tuple)):
        return get_hash(str([get_hash(xx) for xx in x]))
    if inspect.isfunction(x):
        func_version = getattr(x, "__version__", "")
        if func_version:
            return get_hash((get_func_name(x), func_version))
        # Use the name in the hash so function generators can use this effectively.
        return get_hash((get_func_name(x), inspect.getsource(x)))
    return hashlib.sha1(str(x).encode()).hexdigest()


def wrap_for_memorize(
    strict: bool = True,
    hash_salt: Optional[Any] = None,
    hash_override: Optional[str] = None,
) -> Callable[[Any], HashSafeWrapper]:
    """Decorator generator for making an object hashable for use with memorize.

    Args:
        strict: Whether to get the source code for functions and classes. If False,
            will try to use `__name__` or simply get the hash of the object string.
            (defaults to True)
        hash_salt: An optional salt used for hashing. This can be useful for
            debugging. (defaults to None and no salt will be used).
        hash_override: Provide a hash to override all other behavior. This can be
            useful for debugging. (defaults to None)

    Returns:
       Decorator generator function that is safe to be used as argument to `memorize`d
       function.
    """

    def wrap_for_memorize(func: Callable) -> HashSafeWrapper:
        return HashSafeWrapper(
            obj=func,
            strict=strict,
            hash_salt=hash_salt,
            hash_override=hash_override,
        )

    return wrap_for_memorize


def memorize(  # noqa: C901
    local_dir: str,
    s3_dir: Optional[str] = None,
    save_metadata: bool = True,
    kwargs_formatters: Optional[list[tuple[str, Callable[[Any], str]]]] = None,
    func_name_override: Optional[str] = None,
    create_local_dir: bool = True,
    strict: bool = False,
    max_filename_len: int = 255,
    hash_len: int = 16,
    file_ext: Optional[str] = None,
    dump_format: str = "joblib",
    save_func: Optional[Callable[[Any, str], bool]] = None,
    load_func: Optional[Callable[[str], Any]] = None,
    logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
) -> Callable[[Callable], Callable]:
    """Decorator for persisting results of generic functions. Note that the decorated
    function will accept the following special arguments starting with two underscores,
    which will be handled internally and not passed to the underlying function, whereas
    arguments starting with three underscores will be passed on to the function.

    __ignore_cache (bool): Whether to ignore the caching mechanism completely.
    __force_refresh (bool): Whether the cache should be refreshed even if it exists.
    __raise_on_error (bool): Whether error in saving cache file should raise an
        exception. (Defaults to True)
    __raise_on_cache_miss (bool): Whether cache miss should raise an exception. This
        is often useful for debugging cache misses (Defaults to False).
    __cache_key_prepend (str): Optional string to prepend to the cache file name.
    __cache_key_append (str): Optional string to append to the cache file name.
    __out_dict (Optional[dict]): A dictionary which can be passed in to be populated
        with the cache file paths.
    __local_dir (str): Override for `local_dir`.
    __s3_dir (Optional[str]): Override for `s3_dir`.
    __save_metadata (bool): Override for `save_metadata`.
    __kwargs_formatters (list[tuple[str, Callable]]): Override for `kwargs_formatters`.
    __func_name_override (Optional[str]): Override for `func_name_override`.
    __strict (bool): Override for `strict`.
    __max_filename_len (int): Override for `max_filename_len`.
    __file_ext (Optional[str]): Override for `file_ext`.
    __dump_format (str): Override for `dump_format`.
    __save_func (Optional[Callable[[Any, str], bool]]): Override for `save_func`.
    __load_func (Optional[Callable[[str], Any]]): Override for `load_func`.
    __logger (logging.Logger | Callable[[str], None]): Override for `logger`.

    Args:
        local_dir: Local cache directory
        s3_dir: Path to s3 in format "s3://<bucket>/<object_prefix>"
        save_metadata: Whether to save metadata about the function call.
        kwargs_formatters: A list of keyword args and their value_formatter functions.
            A value_formatter function is a function that takes the arg and returns a
            suitable representation of it to be included in the cache file name.
            Provide None or map to a constant to exclude arg from the cache key.
        create_local_dir: Whether the cache directory should be created if it does not
            exist. (Defaults to True).
        strict: Whether the cache should be invalidated when the function implementation
            is changed. (Defaults to False).
        func_name_override (Optional[str]): Override for function name in case you want
            a different name than the actual function name. (Defaults to None and will
            use the actual function name).
        max_filename_len: Maximum length of the cache file name (OSX seems to not
            like filenames that are more than 255 characters long, so that is the
            default). If file name is longer, the long part will be replaced with
            a hash.
        hash_len: Length of hexadecimal hash string.
        file_ext: File extension. (default: None and will fall back to value of
            `dump_format`)
        dump_format: Format of result if it is a DataFrame. Must be one of
            {'dill', 'joblib', 'parquet', 'csv'} (default: 'joblib')
        save_func: Function that takes the result and the path and saves it.
        load_func: Function that takes the path to a result and loads it.
        logger: Logging.Logger object, print, or any other logging function.
    """

    def memorize_(func: Callable) -> Callable:
        func_src_hash = getattr(func, "__hash_override__", None)
        if func_src_hash is None:
            func_src_hash = get_hash(func)[:hash_len]
        _metadata: dict[str, Any] = {}
        if save_metadata:
            _metadata["source"] = inspect.getsource(func)
            _metadata["hash_len"] = hash_len

        _special_kwargs_set = {
            "__ignore_cache",
            "__force_refresh",
            "__raise_on_error",
            "__raise_on_cache_miss",
            "__cache_key_append",
            "__cache_key_prepend",
            "__out_dict",
            "__local_dir",
            "__s3_dir",
            "__save_metadata",
            "__kwargs_formatters",
            "__func_name_override",
            "__strict",
            "__max_filename_len",
            "__dump_format",
            "__file_ext",
            "__save_func",
            "__load_func",
            "__logger",
        }

        @wraps(func)
        def memorized(*args: Any, **kwargs: Any) -> Any:
            # Extract special arguments (double underscores, not passed to the function)
            _ignore_cache = kwargs.pop(
                "__ignore_cache", kwargs.get("___ignore_cache", False)
            )
            _force_refresh = kwargs.pop(
                "__force_refresh", kwargs.get("___force_refresh", False)
            )
            _raise_on_error = kwargs.pop(
                "__raise_on_error", kwargs.get("___raise_on_error", False)
            )
            _raise_on_cache_miss = kwargs.pop(
                "__raise_on_cache_miss", kwargs.get("___raise_on_cache_miss", False)
            )
            _cache_key_prepend = kwargs.pop(
                "__cache_key_prepend", kwargs.get("___cache_key_prepend", "")
            )
            _cache_key_append = kwargs.pop(
                "__cache_key_append", kwargs.get("___cache_key_append", "")
            )
            _out_dict = kwargs.pop("__out_dict", kwargs.get("___out_dict", {}))
            _local_dir = kwargs.pop(
                "__local_dir", kwargs.get("___local_dir", local_dir)
            )
            _s3_dir = kwargs.pop("__s3_dir", kwargs.get("___s3_dir", s3_dir))
            _save_metadata = kwargs.pop(
                "__save_metadata", kwargs.get("___save_metadata", save_metadata)
            )
            _kwargs_formatters = (
                kwargs.pop(
                    "__kwargs_formatters",
                    kwargs.get("___kwargs_formatters", kwargs_formatters),
                )
                or []
            )
            _func_name_override = kwargs.pop(
                "__func_name_override",
                kwargs.get("___func_name_override", func_name_override),
            )
            _strict = kwargs.pop("__strict", kwargs.get("___strict", strict))
            _max_filename_len = kwargs.pop(
                "__max_filename_len",
                kwargs.get("___max_filename_len", max_filename_len),
            )
            _file_ext = kwargs.pop("__file_ext", kwargs.get("___file_ext", file_ext))
            _dump_format = kwargs.pop(
                "__dump_format", kwargs.get("___dump_format", dump_format)
            )
            assert _dump_format in {
                "dill",
                "joblib",
                "parquet",
                "csv",
            }, f"dump_format={_dump_format} not supported!"
            _save_func = kwargs.pop(
                "__save_func", kwargs.get("___save_func", save_func)
            )
            _load_func = kwargs.pop(
                "__load_func", kwargs.get("___load_func", load_func)
            )
            _logger = kwargs.pop("__logger", kwargs.get("___logger", logger))
            if not _logger:
                _logger = lambda x: None  # noqa: E731

            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            if _ignore_cache:
                return func(*bound_args.args, **bound_args.kwargs)

            all_kwargs = bound_args.arguments
            hash_kwargs = {
                k: v
                for k, v in all_kwargs.items()
                if (k not in _special_kwargs_set) and (k[1:] not in _special_kwargs_set)
            }
            if "kwargs" in hash_kwargs:
                hash_kwargs.update(hash_kwargs.pop("kwargs"))

            # Apply formatters to keyword arguments
            formatted_parts = []
            for key, formatter in _kwargs_formatters:
                if key in hash_kwargs:
                    if formatter is not None:
                        formatted_parts.append(formatter(all_kwargs[key]))
                    # Remove formatted args from hash_kwargs to exclude from hash
                    del hash_kwargs[key]

            # Generate cache key
            if _strict:
                hash_kwargs["src_hash"] = func_src_hash

            _cache_key_parts = (
                [
                    part
                    for part in [
                        _cache_key_prepend,
                        (_func_name_override or func.__name__),
                    ]
                    if part
                ]
                + formatted_parts
                + [get_hash(hash_kwargs)]
                + [part for part in [_cache_key_append] if part]
            )
            _cache_key = "__".join(_cache_key_parts)
            if len(_cache_key) > _max_filename_len:
                _cache_key = (
                    f"{_cache_key[:_max_filename_len-46]}__{get_hash(_cache_key)}"
                )

            cache_file_path = os.path.join(
                _local_dir, f"{_cache_key}.{_file_ext or _dump_format}"
            )
            metadata_file_path = (
                f"{cache_file_path}.meta.json" if _save_metadata else None
            )
            s3_file_path = (
                os.path.join(_s3_dir, f"{_cache_key}.{_file_ext or _dump_format}")
                if _s3_dir
                else None
            )
            s3_metadata_file_path = (
                f"{s3_file_path}.meta.json" if s3_file_path and _save_metadata else None
            )

            local_cache_exists = os.path.exists(cache_file_path)
            s3_cache_exists = False
            try:
                s3_cache_exists = s3_file_path and pstu.exists_in_s3(s3_file_path)
            except Exception as e:
                _logger(f"Failed to check existence of cache in s3: {e}")
            if not local_cache_exists and s3_cache_exists:
                if pstu.download_file_from_s3(
                    s3_file_path,
                    cache_file_path,
                    silent=True,
                    raise_on_error=False,
                ):
                    local_cache_exists = True
                    _logger(
                        f"Downloaded cache file from S3: {s3_file_path}"
                        f" to {cache_file_path}"
                    )
                else:
                    _logger(f"Failed to download cache file from S3: {s3_file_path}")
                if s3_metadata_file_path and pstu.download_file_from_s3(
                    s3_metadata_file_path,
                    metadata_file_path,
                    silent=True,
                    raise_on_error=False,
                ):
                    _logger(
                        f"Downloaded metadata file from S3: {s3_metadata_file_path}"
                        f" to {metadata_file_path}"
                    )
                else:
                    _logger(
                        "Failed to download metadata file from S3:"
                        f" {s3_metadata_file_path}"
                    )

            cache_loaded = False

            if local_cache_exists and not _force_refresh:
                _logger(f"Loading from cache file: {cache_file_path}")
                try:
                    if _load_func:
                        result = _load_func(cache_file_path)
                    elif _dump_format == "dill":
                        with open(cache_file_path, "rb") as f:
                            result = dill.load(f)
                    elif _dump_format == "joblib":
                        result = joblib.load(cache_file_path)
                    elif _dump_format in {"parquet", "csv"}:
                        result = (
                            pd.read_parquet(cache_file_path)
                            if _dump_format == "parquet"
                            else pd.read_csv(cache_file_path)
                        )
                    cache_loaded = True
                except Exception as e:
                    _logger(f"Cache load failed: {e}")
                    if _raise_on_cache_miss:
                        raise e

            if not local_cache_exists and _raise_on_cache_miss:
                raise Exception(f"Cache file {cache_file_path} not found.")

            if not cache_loaded:
                start_time = time.time()
                result = func(*bound_args.args, **bound_args.kwargs)
                end_time = time.time()
                duration = end_time - start_time

                if create_local_dir and not os.path.exists(_local_dir):
                    try:
                        os.makedirs(_local_dir)
                    except Exception as e:
                        _logger(f"Failed to create cache directory: {e}")
                        if _raise_on_error:
                            raise e
                        return result

                _logger(f"Saving to cache file: {cache_file_path}")
                try:
                    if _save_func:
                        _save_func(result, cache_file_path)
                    else:
                        if _dump_format == "dill":
                            with open(cache_file_path, "wb") as f:
                                dill.dump(result, f)
                        elif _dump_format == "joblib":
                            joblib.dump(result, cache_file_path)
                        elif _dump_format in {"parquet", "csv"}:
                            if _dump_format == "parquet":
                                result.to_parquet(cache_file_path)
                            else:
                                result.to_csv(cache_file_path)
                except Exception as e:
                    _logger(f"Failed to save cache: {e}")
                    if _raise_on_error:
                        raise e
                    return result

                if metadata_file_path:
                    _metadata["kwargs"] = {
                        k: get_short_str(v, max_len=256) for k, v in all_kwargs.items()
                    }
                    _metadata["hash_kwargs"] = {
                        k: get_short_str(v, max_len=256) for k, v in hash_kwargs.items()
                    }
                    _metadata["start_time"] = start_time
                    _metadata["end_time"] = end_time
                    _metadata["duration"] = duration
                    pstu.dump_local(_metadata, metadata_file_path, "json")
                    _logger(f"Saved metadata to: {metadata_file_path}")

                if s3_file_path:
                    _logger(f"Uploading to S3: {s3_file_path}")
                    try:
                        pstu.upload_file_to_s3(
                            cache_file_path, s3_file_path, silent=True
                        )
                        _logger(
                            f"Uploaded cache file from {cache_file_path}"
                            f" to {s3_file_path}"
                        )
                        if s3_metadata_file_path:
                            pstu.upload_file_to_s3(
                                metadata_file_path, s3_metadata_file_path, silent=True
                            )
                            _logger(
                                f"Uploaded metadata file from {metadata_file_path}"
                                f" to {s3_metadata_file_path}"
                            )
                    except Exception as e:
                        if _raise_on_error:
                            raise e
                        if _logger:
                            _logger(f"Failed to upload to S3: {e}")

            _out_dict["local_path"] = cache_file_path
            _out_dict["local_metadata_path"] = metadata_file_path
            _out_dict["s3_path"] = s3_file_path
            _out_dict["s3_metadata_path"] = s3_metadata_file_path

            return result

        return memorized

    return memorize_


def memoize(args_are_hashable: bool = True) -> Callable[[Callable], Callable]:
    """Decorator generator for caching results of generic function in memory.

    Args:
        args_are_hashable: Whether the arguments are hashable.

    Returns:
        The memoize decorator.
    """

    def memoize_decorator(func: Callable) -> Callable:
        """Decorator for caching results of generic function in memory.

        Args:
            func: The function to be memoized.

        Returns:
            The memoized function.
        """
        _cached_results_: dict[Hashable, Any] = {}
        hash_override = getattr(func, "__hash_override__", None)
        if hash_override is None:
            hash_override = get_hash(func)

        @wraps(func)
        def memoized(*args: Any, **kwargs: Any) -> Any:
            _force_refresh = kwargs.pop("__force_refresh", False)
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            _cache_key: Hashable
            if args_are_hashable:
                _cache_key = tuple(bound_args.arguments.items())
            else:
                _cache_key = get_hash(bound_args.arguments)

            if _cache_key in _cached_results_ and not _force_refresh:
                return _cached_results_[_cache_key]
            res = func(*bound_args.args, **bound_args.kwargs)
            _cached_results_[_cache_key] = res
            return res

        setattr(memoized, "_cached_results_", _cached_results_)
        setattr(memoized, "__hash_override__", hash_override)

        return memoized

    return memoize_decorator


def get_func_name(func: Callable, include_module: bool = True) -> str:
    module_name = ""
    module = inspect.getmodule(func)
    if include_module and module is not None:
        module_name = f"{module.__name__}."
    func_name = ""
    try:
        func_name = func.__name__
    except:  # noqa=E772
        try:
            func_name = re.search(  # type: ignore
                "function ([^ .]+)(at 0x[0-f]+)?", str(func)
            ).groups()[0]
        except:  # noqa=E772
            func_name = str(func)
    if func_name == "<lambda>":
        raise ValueError("lambda function does not have a name")
    return f"{module_name}{func_name}"


def get_short_str(x: Any, max_len: int = 100) -> str:
    """Get a shortened string representation of any object.

    If the length of `x` is less than `max_len`, it is returned as is.
    If it is more, it is truncated to fit the format:
    "<first_part> ... <second_part>" where the length is equal to `max_len`.

    Args:
        x: The input object to be shortened.
        max_len: The maximum allowed length for the output string. Defaults to 100.

    Returns:
        A string representation of `x` shortened to `max_len` characters if necessary.
    """
    str_x = str(x)
    if len(str_x) <= max_len:
        return str_x

    part_len = (
        max_len - 5
    ) // 2  # Length of each part, 5 characters are reserved for " ... "
    first_part = str_x[:part_len]
    second_part = str_x[-part_len:]

    return f"{first_part} ... {second_part}"
