"""
Utilities related to caching function results.

Author: Sam Cohan
"""
import datetime
import hashlib
import inspect
import logging
import os
import re
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Union

import dill
import joblib
import pandas as pd

import scpyutils.persistutils as pstu


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


class HashSafeWrapper:
    """Class for wrapping objects to make them safe for use with memorize.

    The main use of this class is to wrap functions and classes which need to be
    passed as arguments to `memorize`. By default, the class will take the source of
    the function or class and keep its hash on the `__hash_override__` attribute as
    required by `memorize`. To actually decorate your functions, make use of
    `wrap_for_memorize` function defined in this module.

    """

    KNOWN_TYPES = ["class", "function"]

    def __init__(
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
            else "class"
            if inspect.isclass(obj)
            else "?"
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

    def __hash__(self):
        return self.__hash_override__

    def override_hash(self, hash_: str):
        self.__hash_override__ = hash_
        if self._obj_type in self.KNOWN_TYPES:
            hash_in_name = f" src_hash:{hash_[:16]}" if self._strict else ""
            self.__name__ = f"<{self._obj_type} {self._name}{hash_in_name}>"
        return self

    @property
    def obj(self):
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __str__(self):
        return self.__name__

    __repr__ = __str__


def wrap_for_memorize(
    strict: bool = True,
    hash_salt: Optional[Any] = None,
    hash_override: Optional[str] = None,
):
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

    def wrap_for_memorize(func):
        return HashSafeWrapper(
            obj=func,
            strict=strict,
            hash_salt=hash_salt,
            hash_override=hash_override,
        )

    return wrap_for_memorize


def memorize(  # noqa=C901
    local_dir: str,
    s3_dir: Optional[str] = None,
    save_metadata: bool = True,
    kwargs_formatters: List[Tuple[str, Callable[[Any], str]]] = None,
    func_name_override: Optional[str] = None,
    num_args_to_ignore: int = 0,
    create_local_dir: bool = True,
    strict: bool = False,
    max_filename_len: int = 255,
    hash_len: int = 16,
    file_ext: Optional[str] = None,
    dump_format: str = "joblib",
    save_func: Optional[Callable[[Any, str], bool]] = None,
    load_func: Optional[Callable[[str], Any]] = None,
    logger: Union[logging.Logger, Callable[[str], None]] = None,
):
    """Decorator for persisting results of generic functions. Note that the decorated
    function will accept the following special arguments starting with two and three
    underscores, the ones with three underscores will be not be passed to the
    underlying function whereas the ones with two underscores will be passed on.

    __ignore_cache (bool): Whether to ignore the caching mechanism completely.
    __force_refresh (bool): Whether the cache should be refreshed even if it exists.
    __raise_on_error (bool): whether error in saving cache file should raise an
        exception. (Defaults to True)
    __raise_on_cache_miss (bool): whether cache miss should raise an exception. This
        is often useful for debugging cache misses (Defaults to False).
    __cache_key_prepend (str): optional string to append to the cache file name.
    __cache_key_append (str): optional string to append to cache file name.
    __out_dict (Optional[Dict]): A dictionary which can be passed in to be populated
        with the cache file paths.
    __local_dir (str): override for `local_dir`.
    __s3_dir (Optional[str]): override for `s3_dir`.
    __save_metadata (bool): override for `save_metadata`.
    __kwargs_formatters (List[Tuple[str, Callable]]): override for
        `kwargs_formatters`.
    __num_args_to_ignore (int): override for `num_args_to_ignore`.
    __func_name_override (Optional[str]): override for `func_name_override`.
    __strict (bool): override for `strict`.
    __max_filename_len (int): override for `max_filename_len`.
    __file_ext (Optional[str]): override for `file_ext`.
    __dump_format (str): override for `dump_format`.
    __save_func (Optional[Callable[[Any, str], bool]]): override for `save_func`.
    __load_func (Optional[Callable[[str], Any]]): override for `load_func`.
    __logger (logging.Logger|Callable): override for `logger`.

    Args:
        local_dir: local cache directory
        s3_dir: path to s3 in format "s3://<bucket>/<object_prefix>"
        save_metadata: Whether to save metadata about the function call.
        kwargs_formatters: A list of keyword args and their value_formatter
            functions. A value_formatter function is a function that takes the arg
            and returns a suitable representation of it to be included in the cache
            file name. Provide None or map to a constant to exclude arg from the
            cache key.
        num_args_to_ignore: number of args which will not be taken into
            account in the creation of the cache_key. This can be useful for
            functions where the first arguments are non-hashable accessor like a
            session or shell, etc. (Defaults to 0).
        create_local_dir: whether the cache directory should be created if
            it does not exist. (Defaults to True).
        strict: whether the cache should be invalidated when the function
            implementation is changed. (Defaults to False).
        func_name_override (Optional[str]): override for function name in case you
            want a different name than the actual function name. (Defaults to None
            and will use the actual function name).
        max_filename_len: maximum length of the cache file name (OSX seems to not
            like filenames that are more than 255 characters long, so tha is the
            default). In file name is longer, the the long part will be replaced with
            a hash.
        hash_len: length of hexadecimal hash string.
        file_ext: File extension. (default: None and will fall back to value of
            `dump_format`)
        dump_format: format of result if it is a DataFrame. Must be one of
            {'dill', 'joblib', 'parquet', 'csv'} (default: 'joblib')
        save_func: function that takes the result and the path and saves it.
        load_func: function that takes the path to a result and loads it.
        logger: logging.Logger object, print, or any other logging function.
    """

    def memorize_(func):
        func_src_hash = getattr(func, "__hash_override__", None)
        if func_src_hash is None:
            func_src_hash = get_hash(func)[:hash_len]
        _metadata = {}
        if save_metadata:
            _metadata["source"] = inspect.getsource(func)
            _metadata["hash_len"] = hash_len

        _special_kwargs_set = set(
            [
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
                "__num_args_to_ignore",
                "__func_name_override",
                "__strict",
                "__max_filename_len",
                "__dump_format",
                "__file_ext",
                "__save_func",
                "__load_func",
                "__logger",
            ]
        )

        @wraps(func)
        def memorize__(*args, **kwargs):
            _ignore_cache = kwargs.pop(
                "___ignore_cache", kwargs.get("__ignore_cache", False)
            )
            if _ignore_cache:
                return func(*args, **kwargs)
            _save_metadata = kwargs.pop(
                "___save_metadata", kwargs.get("__save_metadata", save_metadata)
            )
            if _save_metadata:
                # Opted to put this here even though it can slightly slow down cache
                # reads because we can still gain back perf by passing override for
                # __save_metadata=False
                _metadata["args"] = [str(x) for x in args]
                _metadata["kwargs"] = {k: str(v) for k, v in kwargs.items()}

            _force_refresh = kwargs.pop(
                "___force_refresh", kwargs.get("__force_refresh", False)
            )
            _raise_on_error = kwargs.pop(
                "___raise_on_error", kwargs.get("__raise_on_error", False)
            )
            _raise_on_cache_miss = kwargs.pop(
                "___raise_on_cache_miss", kwargs.get("__raise_on_cache_miss", False)
            )
            _cache_key_prepend = kwargs.pop(
                "___cache_key_prepend", kwargs.get("__cache_key_prepend", "")
            )
            _cache_key_append = kwargs.pop(
                "___cache_key_append", kwargs.get("__cache_key_append", "")
            )
            _out_dict = kwargs.pop("___out_dict", kwargs.get("__out_dict", {}))
            _local_dir = kwargs.pop(
                "___local_dir", kwargs.get("__local_dir", local_dir)
            )
            _s3_dir = kwargs.pop("___s3_dir", kwargs.get("__s3_dir", s3_dir))
            _kwargs_formatters = (
                kwargs.pop(
                    "___kwargs_formatters",
                    kwargs.get("__kwargs_formatters", kwargs_formatters),
                )
                or {}
            )
            _num_args_to_ignore = kwargs.pop(
                "___num_args_to_ignore",
                kwargs.get("__num_args_to_ignore", num_args_to_ignore),
            )
            _func_name_override = kwargs.pop(
                "___func_name_override",
                kwargs.get("__func_name_override", func_name_override),
            )
            _strict = kwargs.pop("___strict", kwargs.get("__strict", strict))
            _max_filename_len = kwargs.pop(
                "___max_filename_len",
                kwargs.get("__max_filename_len", max_filename_len),
            )
            _file_ext = kwargs.pop("___file_ext", kwargs.get("__file_ext", file_ext))
            _dump_format = kwargs.pop(
                "___dump_format", kwargs.get("__dump_format", dump_format)
            )
            assert _dump_format in {
                "dill",
                "joblib",
                "parquet",
                "csv",
            }, f"dump_format={dump_format} not supported!"
            _save_func = kwargs.pop(
                "___save_func", kwargs.get("__save_func", save_func)
            )
            _load_func = kwargs.pop(
                "___load_func", kwargs.get("__load_func", load_func)
            )
            _logger = kwargs.pop("___logger", kwargs.get("__logger", logger))
            if not _logger:
                _logger = lambda x: None  # noqa: E731
            # Make sure we don't grab a cache file if the function does not support
            # arbitrary keyword args and the provided arguments are not supported
            # (could happen if you change function signature).
            argspec = inspect.getfullargspec(func)
            extra_kwargs = set(kwargs) - set(argspec.args)
            if extra_kwargs and not argspec.varkw:
                unacceptable_kwargs = extra_kwargs - _special_kwargs_set
                if unacceptable_kwargs:
                    raise Exception(
                        f"Following args are not supported: {unacceptable_kwargs}"
                    )
                for kwarg in extra_kwargs:
                    kwargs.pop(kwarg)

            _args_str = "__".join(
                [
                    x.strftime("%Y%m%d_%H%M%S")
                    if isinstance(x, datetime.datetime)
                    else (
                        "_".join([str(xx) for xx in x])
                        if isinstance(x, (list, tuple))
                        else str(x)
                    )
                    for x in args[_num_args_to_ignore:]
                ]
            )
            _explicit_kwargs_str = "__".join(
                [
                    val_formatter(kwargs[kwarg])
                    for kwarg, val_formatter in _kwargs_formatters
                    if val_formatter and kwarg in kwargs
                ]
            )
            remaining_kwargs = {
                k: v for k, v in kwargs.items() if k not in dict(_kwargs_formatters)
            }
            _kwargs_in_hash = {
                k: v
                for k, v in remaining_kwargs.items()
                if k not in _special_kwargs_set
            }
            if _strict:
                _kwargs_in_hash["src_hash"] = func_src_hash
            _remaining_kwargs_hash = (
                get_hash(_kwargs_in_hash)[:hash_len] if _kwargs_in_hash else ""
            )
            if _func_name_override is None:
                _func_name = func.__name__
            else:
                _func_name = func_name_override
            if _file_ext is None:
                _file_ext = _dump_format
            _file_ext = f".{_file_ext}"
            _filename = (
                "__".join(
                    [
                        str(x)
                        for x in (
                            [
                                _cache_key_prepend,
                                _func_name,
                                _args_str,
                                _explicit_kwargs_str,
                                _remaining_kwargs_hash,
                                _cache_key_append,
                            ]
                        )
                        if x
                    ]
                )
                + _file_ext
            )
            if _max_filename_len and len(_filename) > _max_filename_len:
                getattr(_logger, "warning", _logger)(
                    f"Cache filename longer than {_max_filename_len} will be"
                    " truncated."
                )
                _filename = (
                    f"{_filename[:_max_filename_len - 46]}"
                    f"__{get_hash(_filename[_max_filename_len - 46:])[:hash_len]}"
                    f"{_file_ext}"
                )
            _local_path = os.path.join(_local_dir, _filename)
            _s3_path = os.path.join(_s3_dir, _filename) if _s3_dir else None
            _local_metadata_path = (
                (_local_path.rsplit(".", 1)[0] + ".meta.json")
                if _save_metadata
                else None
            )
            _s3_metadata_path = (
                (_s3_path.rsplit(".", 1)[0] + ".meta.json")
                if _s3_path and _save_metadata
                else None
            )
            local_cache_exists = os.path.isfile(_local_path)
            if not local_cache_exists and not _force_refresh:
                getattr(_logger, "info", _logger)(
                    f"No local cache file found at '{_local_path}'"
                )
                if _s3_path:
                    s3_path_exists = pstu.exists_in_s3(_s3_path)
                    if not s3_path_exists:
                        getattr(_logger, "info", _logger)(
                            f"No s3 cache file found at '{_s3_path}'"
                        )
                    else:
                        getattr(_logger, "info", _logger)(
                            "Downloading cache file from:"
                            f" '{_s3_path}' to '{_local_path}'"
                        )
                        pstu.download_file_from_s3(
                            s3_path=_s3_path,
                            local_path=_local_path,
                            silent=True,
                            raise_on_error=_raise_on_error,
                        )
                        if _save_metadata:
                            getattr(_logger, "info", _logger)(
                                "Downloading meta file from:"
                                f" '{_s3_metadata_path}' to '{_local_metadata_path}'"
                            )
                            try:
                                pstu.download_file_from_s3(
                                    s3_path=_s3_metadata_path,
                                    local_path=_local_metadata_path,
                                    silent=True,
                                    raise_on_error=True,
                                )
                            except Exception as e:
                                getattr(_logger, "error", _logger)(
                                    "ERROR: Failed to download meta file from:"
                                    f" '{_s3_metadata_path}' {e}"
                                )
                local_cache_exists = os.path.isfile(_local_path)
            if local_cache_exists and not _force_refresh:
                getattr(_logger, "info", _logger)(
                    f"Loading from cache file: '{_local_path}' ..."
                )
                if _load_func:
                    res = load_func(_local_path)
                elif _dump_format in {"parquet", "csv"}:
                    res = getattr(pd, f"read_{_dump_format}")(_local_path)
                elif _dump_format == "dill":
                    res = dill.load(open(_local_path, "rb"))
                else:
                    res = joblib.load(_local_path)
            else:
                if not local_cache_exists:
                    if _raise_on_cache_miss:
                        raise Exception(f"Missing cache file: '{_local_path}'")
                else:
                    getattr(_logger, "info", _logger)(
                        f"Refreshing cache file at '{_local_path}'"
                        " by calling function ..."
                    )
                start_time = time.time()
                # Actually call the function!
                res = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                getattr(_logger, "info", _logger)(
                    f"Function call took {duration} seconds. "
                    f"Saving cache file '{_local_path}' ..."
                )
                if create_local_dir:
                    pstu.ensure_dirs(_local_dir, raise_on_error=False)
                try:
                    if _save_func:
                        _save_func(res, _local_path)
                    elif _dump_format in {"parquet", "csv"}:
                        getattr(res, f"to_{_dump_format}")(
                            _local_path, index=all(res.index.names)
                        )
                    else:
                        pstu.dump_local(
                            obj=res,
                            path=_local_path,
                            dump_format=_dump_format,
                            create_dirs=False,
                            silent=True,
                            raise_on_error=True,
                        )
                except Exception as e:
                    getattr(_logger, "exception", _logger)(
                        f"ERROR: failed to save cache file '{_local_path}' : {e}"
                    )
                    _local_path = None

                if _local_path and _local_metadata_path:
                    local_vars = locals()
                    for kwarg in _special_kwargs_set:
                        if kwarg not in {"__logger", "__out_dict"}:
                            _metadata[kwarg[2:]] = local_vars.get(kwarg[1:])
                    _metadata["start_time"] = start_time
                    _metadata["end_time"] = end_time
                    _metadata["duration"] = duration
                    _metadata["filename"] = _filename
                    getattr(_logger, "info", _logger)(
                        f"Saving metadata file to '{_local_metadata_path}' ..."
                    )
                    metadata_dumped = pstu.dump_local(
                        obj=_metadata,
                        path=_local_metadata_path,
                        dump_format="json",
                        create_dirs=False,
                        silent=True,
                        raise_on_error=_raise_on_error,
                    )
                    if not metadata_dumped:
                        getattr(_logger, "error", _logger)(
                            "ERROR: failed to save metadata file"
                            f" '{_local_metadata_path}'"
                        )
                        _local_metadata_path = None

                if _local_path and _s3_path:
                    getattr(_logger, "info", _logger)(
                        f"Uploading cache file from '{_local_path}'"
                        f"to '{_s3_path}' ..."
                    )
                    cache_uploaded = pstu.upload_file_to_s3(
                        local_path=_local_path,
                        s3_path=_s3_path,
                        silent=True,
                        raise_on_error=_raise_on_error,
                    )
                    if not cache_uploaded:
                        getattr(_logger, "error", _logger)(
                            f"ERROR: failed to upload cache file to '{_s3_path}'"
                        )
                        _s3_path = None
                if _s3_path and _local_metadata_path:
                    getattr(_logger, "info", _logger)(
                        f"Uploading metadata file from '{_local_metadata_path}'"
                        f" to '{_s3_metadata_path}' ..."
                    )
                    metadata_uploaded = pstu.upload_file_to_s3(
                        local_path=_local_metadata_path,
                        s3_path=_s3_metadata_path,
                        silent=True,
                        raise_on_error=_raise_on_error,
                    )
                    if not metadata_uploaded:
                        getattr(_logger, "error", _logger)(
                            "ERROR: failed to upload metadata file to"
                            f" '{_s3_metadata_path}'"
                        )
                        _s3_metadata_path = None
            _out_dict["local_path"] = _local_path
            _out_dict["local_metadata_path"] = _local_metadata_path
            _out_dict["s3_path"] = _s3_path
            _out_dict["s3_metadata_path"] = _s3_metadata_path

            return res

        memorize__.__hash_override__ = func_src_hash
        return memorize__

    return memorize_


def memoize(func):
    """Decorator for caching results of generic function in memory."""
    _cached_results_ = {}
    hash_override = getattr(func, "__hash_override__", None)
    if hash_override is None:
        hash_override = get_hash(func)

    @wraps(func)
    def memoized(*args, **kwargs):
        _cache_key = get_hash((args, kwargs))
        try:
            res = _cached_results_[_cache_key]
        except KeyError:
            res = func(*args, **kwargs)
            _cached_results_[_cache_key] = res
        return res

    memoized._cached_results_ = _cached_results_  # pylint: disable=protected-access
    memoized.__hash_override__ = hash_override

    return memoized


def memoize_with_hashable_args(func):
    """Decorator for fast caching of functions which have hashable args.
    Note that it will convert np.NaN to None for caching to avoid this common
    case causing a cache miss.
    """
    _cached_results_ = {}
    hash_override = getattr(func, "__hash_override__", None)
    if hash_override is None:
        hash_override = get_hash(func)

    @wraps(func)
    def memoized(*args):
        try:
            lookup_args = tuple(x if pd.notnull(x) else None for x in args)
            res = _cached_results_[lookup_args]
        except KeyError:
            res = func(*args)
            _cached_results_[lookup_args] = res
        return res

    memoized._cached_results_ = _cached_results_  # pylint: disable=protected-access
    memoized.__hash_override__ = hash_override

    return memoized
