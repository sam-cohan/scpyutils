import base64
import datetime
from functools import wraps
import hashlib
import inspect
import logging
import os
import pickle
import re
from tempfile import NamedTemporaryFile
import zlib

import boto3
import botocore
import pandas as pd
import simplejson as json
import requests

from .jsonutils import json_default
from .logutils import setup_logger

LOGGER = setup_logger(__name__, log_level=logging.INFO)


def get_func_name(func, include_module=True):
    module_name = inspect.getmodule(func).__name__ if include_module else ""
    func_name = ""
    try:
        func_name = func_name.__name__
    except:
        try:
            func_name = re.search(
                "function ([^ .]+)(at 0x[0-f]+)?", str(func)).groups()[0]
        except:
            func_name = str(func)
    if func_name == "<lambda>":
        raise ValueError("lambda function does not have a name")
    return f"{module_name}.{func_name}"


def get_hash(x):
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

    def __init__(self, obj, hash_salt="", forced_hash_override=None):
        self._hash_salt = hash_salt
        self._obj = obj
        if inspect.isfunction(obj):
            self._obj_str = get_func_name(obj)
        else:
            self._obj_str = str(obj)
        if forced_hash_override:
            self._forced_hash_override = forced_hash_override
            self.__hash_override__ = forced_hash_override
        else:
            self._forced_hash_override = None
            if hasattr(self._obj, "__hash_override__"):
                self.__hash_override__ = self._obj.__hash_override__
            else:
                self.__hash_override__ = get_hash(
                    (self._hash_salt, self._obj_str))

    def __hash__(self):
        return self.__hash_override__

    @property
    def obj(self):
        return self._obj

    def __str__(self):
        return (
            f"HashSafeWrapper("
            f"{self._obj_str}"
            f", hash_salt={self._hash_salt}"
            f", forced_hash_override={self._forced_hash_override}"
            ")")

    __repr__ = __str__


class FileObj(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # LOGGER.debug("reading total_bytes={:,.0f} KB... ".format(n >> 10))
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # LOGGER.debug("reading bytes [{:,.0f}, {:,.0f})...".format(idx, idx + batch_size))
                buffer[idx: idx + batch_size] = self.f.read(batch_size)
                # LOGGER.debug("done.")
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        LOGGER.debug("writing total_bytes={:,.0f} KB... ".format(n >> 10))
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            LOGGER.debug(
                "writing bytes [{:,.0f}, {:,.0f})... ".format(
                    idx, idx + batch_size)
            )
            self.f.write(buffer[idx: idx + batch_size])
            LOGGER.debug("done.")
            idx += batch_size


def dump_pickle(obj, file_path, create_dirs=True):
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        LOGGER.warning("Creating dirs: %s", dir_name)
        os.makedirs(dir_name)
    with open(file_path, "wb") as f:
        return pickle.dump(obj, FileObj(f), protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(FileObj(f))


class S3Helper:

    s3_client = None
    BUCKET_NAME = os.environ.get("SCPY_S3_BUCKET_NAME")
    AWS_ACCESS_KEY_ID = os.environ.get("SCPY_S3_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("SCPY_S3_SECRET_ACCESS_KEY")

    @classmethod
    def get_s3_client(cls):
        if cls.s3_client is None:
            cls.s3_client = boto3.client(
                "s3",
                aws_access_key_id=cls.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=cls.AWS_SECRET_ACCESS_KEY)
        return cls.s3_client

    @classmethod
    def key_exists_in_s3(cls, key, bucket_name=None):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        s3_client = cls.get_s3_client()
        try:
            s3_client.head_object(
                Bucket=cls.BUCKET_NAME,
                Key=key,
            )
        except botocore.exceptions.ClientError:
            return False
        except Exception as e:
            LOGGER.error(
                f"unexpected exception when checking existence of s3 key=%s: %s", key, e
            )
            return False
        return True

    @classmethod
    def get_key_content(cls, key, bucket_name=None):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        s3_client = cls.get_s3_client()
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return obj['Body'].read().decode('utf-8')

    @classmethod
    def delete_key(cls, key, bucket_name=None):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        s3_client = cls.get_s3_client()
        return s3_client.delete_object(Bucket=bucket_name, Key=key)

    @classmethod
    def get_bucket_key_list(cls, bucket_name=None, prefix=""):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        s3_client = cls.get_s3_client()
        return [key['Key']
                for key in s3_client.list_objects(Bucket=bucket_name, Prefix=prefix)['Contents']]

    @classmethod
    def save_json_to_s3(cls, json_obj, key, bucket_name=None, expires_secs=None, clear=None, compress=True):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        expires_secs = expires_secs or (30 * 24 * 3600)
        s3_client = cls.get_s3_client()
        with NamedTemporaryFile('w') as fp:
            if compress:
                info = base64.b64encode(
                    zlib.compress(
                        json.dumps(json_obj,
                                   default=json_default).encode('utf-8')
                    )
                ).decode('ascii')
                fp.write(info)
            else:
                json.dump(json_obj, fp, default=json_default)
            fp.flush()
            if clear is not None:
                clear(json_obj)
            s3_client.upload_file(
                fp.name,
                bucket_name,
                key)
            url = cls.generate_presigned_url_for_key(
                key=key, bucket_name=bucket_name, expires_secs=expires_secs)
        return url

    @classmethod
    def get_json_from_s3_link(cls, s3_link):
        info = requests.get(s3_link).content
        try:
            return json.loads(zlib.decompress(base64.b64decode(info)))
        except Exception as e:
            try:
                return info.json()
            except:
                raise e

    @classmethod
    def upload_file_to_s3(cls, filename, key, bucket_name=None, expires_secs=None):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        expires_secs = expires_secs or (30 * 24 * 3600)
        s3_client = cls.get_s3_client()
        s3_client.upload_file(
            filename,
            bucket_name,
            key)
        url = cls.generate_presigned_url_for_key(key=key,
                                                 bucket_name=bucket_name,
                                                 expires_secs=expires_secs)
        return url

    @classmethod
    def generate_presigned_url_for_key(cls, key, bucket_name=None, expires_secs=None):
        bucket_name = cls.BUCKET_NAME if bucket_name is None else bucket_name
        expires_secs = expires_secs or (30 * 24 * 3600)
        s3_client = S3Helper.get_s3_client()
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket_name,
                    "Key": key},
            ExpiresIn=expires_secs,
        )


def memorize(
    cache_dir,
    strict=False,
    create_cache_dir_if_not_exists=True,
    kwargs_explicitely_in_cache_key=None,
    num_args_to_ignore_in_cache_key=0,
    max_filename_len=255,
):
    """Decorator for caching results of generic functions to disk.

    Note that the decorated function will accept the following special arguments starting
    with two and three underscores, the ones with three underscores will be not be passed
    to the underlying function.
    __from_cache (bool): Whether to ignore the caching mechanism completely.
    __cache_dir (str): override for the cache_dir of the decorator.
    __cache_key_prepend (str): optional string to append to the file_name if no _cache_file_
        is provided.
    __cache_key_append (str): optional string to append to the file_name if no cache_file_
        is provided.
    __logger (logging.Logger): optional logger (defaults to the module's logger)
    __raise_if_cache_miss (bool): whehter cache miss should raise an exception. This is often
        useful for debugging cache misses (defaults to False).
    __kwargs_excplicitely_in_cache_key (dict): override for args_explicitely_in_cache_key
    __max_filename_len (int): override for max_filename_len

    Arguments:
        cache_dir (str): cache directory
        strict (bool): whether the cache should be invalidated when the function implementation
            is changed. (defaults to False).
        create_cache_dir_if_exists (bool): whether the cache directory should be created if
            it does not exist. (defaults to True).
        args_explicitely_in_cache_key (Dict[Str, ValueFormatter]): a dictionary of keyword args
            and their value_formatter functions.
        num_args_to_ignore_in_cache_key (int): number of args which will not be taken into account
            in the creation of the cache_key. This can be useful for functions where the first
            arguments are non-hashable accessor like a session or shell, etc. (defaults to 0).
        max_filename_len (int): maximum length of the cache file name (OSX seems to not like filenames
            that are more than 255 characters long, so tha is the default). In file name is longer,
            the the long part will be replaced with a hash.
    """

    def memorize_(func):
        hash_override = getattr(func, "__hash_override__", None)
        if strict and hash_override is None:
            hash_override = get_hash(func)

        _special_kwargs_set = set([
            "__ignore_cache",
            "__refresh_cache",
            "__cache_dir",
            "__cache_key_append",
            "__cache_key_prepend",
            "__logger",
            "__kwargs_explicitely_in_cache_key",
            "__max_filename_len",
        ])

        @wraps(func)
        def memorized(*args, **kwargs):
            _ignore_cache = kwargs.pop(
                "___ignore_cache", kwargs.get("__ignore_cache", False))
            _refresh_cache = kwargs.pop(
                "___refresh_cache", kwargs.get("__refresh_cache", False))
            _cache_dir = kwargs.pop(
                "___cache_dir", kwargs.get("__cache_dir", cache_dir))
            _cache_key_prepend = kwargs.pop(
                "___cache_key_prepend", kwargs.get("__cache_key_prepend", ""))
            _cache_key_append = kwargs.pop(
                "___cache_key_append", kwargs.get("__cache_key_append", ""))
            _logger = kwargs.pop("___logger", kwargs.get("__logger", LOGGER))
            _raise_if_cache_miss = kwargs.pop(
                "___raise_if_cache_miss", kwargs.get("__raise_if_cache_miss", False))
            _kwargs_excplicitely_in_cache_key = kwargs.pop(
                "___kwargs_explicitely_in_cache_key",
                kwargs.get("__kwargs_explicitely_in_cache_key",
                           kwargs_explicitely_in_cache_key)
            ) or {}
            _max_filename_len = kwargs.pop("___max_filename_len", kwargs.get(
                "__max_filename_len", max_filename_len))
            # Make sure we don't grab a cache file if the function does not support arbitrary keyword args
            # and the provided arguments are not supported (could happen if you change function signature).
            argspec = inspect.getfullargspec(func)
            extra_kwargs = set(kwargs) - set(argspec.args)
            if extra_kwargs and not argspec.varkw:
                unacceptable_kwargs = extra_kwargs - _special_kwargs_set
                if unacceptable_kwargs:
                    raise Exception(
                        f"Following args are not supported: {unacceptable_kwargs}")
                for kwarg in extra_kwargs:
                    kwargs.pop(kwarg)
            if _ignore_cache:
                res = func(*args, **kwargs)
            else:
                _args_str = "__".join(
                    [
                        x.strftime("%Y%m%d_%H%M%S")
                        if isinstance(x, datetime.datetime)
                        else (
                            "_".join([str(xx) for xx in x])
                            if isinstance(x, (list, tuple))
                            else str(x)
                        )
                        for x in args[num_args_to_ignore_in_cache_key:]
                    ]
                )
                _explicit_kwarg_vals = [
                    val_formatter(kwargs.get(kwarg))
                    for kwarg, val_formatter in _kwargs_excplicitely_in_cache_key.items()
                ]
                _explicit_kwargs_str = "__".join(
                    [str(x) for x in _explicit_kwarg_vals
                     if x != "__ignore__"])
                _remaining_kwargs_hash = get_hash(
                    {
                        k: v
                        for k, v in list(kwargs.items())
                        if k not in _kwargs_excplicitely_in_cache_key
                    }
                )
                _cache_file = (
                    "__".join(
                        [
                            str(x)
                            for x in ([
                                _cache_key_prepend,
                                ("{}_{}".format(func.__name__, hash_override[:16])
                                    if strict
                                    else func.__name__),
                                _args_str
                            ] + [
                                _explicit_kwargs_str,
                                _remaining_kwargs_hash,
                                _cache_key_append,
                            ])
                            if x
                        ]
                    ) + ".pkl"
                )
                if _max_filename_len and len(_cache_file) > _max_filename_len:
                    getattr(_logger, "warning", _logger)(
                        f"Cache filename longer than {_max_filename_len} will be truncated.")
                    _cache_file = (
                        f"{_cache_file[:_max_filename_len - 46]}"
                        f"__{get_hash(_cache_file[_max_filename_len - 46:])}.pkl")
                _file_path = os.path.join(_cache_dir, _cache_file)
                cache_exists = os.path.isfile(_file_path)
                if cache_exists and not _refresh_cache:
                    getattr(_logger, "info", _logger)(
                        f"Loading from cache file: {_file_path} ...")
                    res = load_pickle(_file_path)
                else:
                    if not cache_exists:
                        getattr(_logger, "info", _logger)(
                            f"No cache file found at {_file_path}. Calling function ...")
                        if _raise_if_cache_miss:
                            raise Exception(
                                f"Missing cache file: {_file_path}")
                    else:
                        getattr(_logger, "warning", _logger)(
                            f"Refreshing cache file at {_file_path} by calling function ...")
                    res = func(*args, **kwargs)
                    getattr(_logger, "info", _logger)(
                        f"Saving cache file {_file_path} ...")
                    try:
                        if create_cache_dir_if_not_exists:
                            if not os.path.exists(_cache_dir):
                                os.makedirs(_cache_dir)
                        dump_pickle(res, _file_path)
                    except Exception:
                        import traceback
                        traceback.print_exc()
            return res

        memorized.__hash_override__ = hash_override
        return memorized

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
