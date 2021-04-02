"""
Utilities related to persisting data to file storage.
"""
import datetime
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import botocore
import joblib

import dill


def ensure_dirs(
    dir_name: str, silent: bool = False, raise_on_error: bool = True
) -> bool:
    """Create directories if they do not exist.

    Args:
        dir_name: Directory name on local machine.
        silent: Whether to suppress printing info message. (Defaults to False)
        raise_on_error: Whether to raise exception on any errors. (Defaults to True)

    Returns:
        Boolean of whether the dir_name exists.
    """
    if not dir_name or dir_name == ".":
        return True
    try:
        if not os.path.exists(dir_name):
            if not silent:
                print(f"Creating new directory {dir_name}")
            os.makedirs(dir_name)
    except Exception as e:
        print(f"ERROR: failed to create directory {dir_name}. {e}")
        if raise_on_error:
            raise e
        return False
    return True


def dump_local(
    obj: Any,
    path: str,
    dump_format: str = "joblib",
    create_dirs=True,
    silent: bool = False,
    raise_on_error: bool = True,
) -> bool:
    """Save a python object to local drive in joblib or json format.

    Args:
        obj: Arbitrary serializable python object.
        path: Path on local machine to store the object.
        dump_format: format of the saved file. Currently support {'dill', 'joblib',
            'json'}. (Defaults to 'joblib')
        create_dirs: Whether the path directory should be created. (Defaults to True)
        silent: Whether to suppress printing info message. (Defaults to False)
        raise_on_error: Whether to raise exception on any errors. (Defaults to True)

    Returns:
        Boolean of whether the file was successfully saved.
    """
    if not silent:
        print(f"Writing to path={path}")
    dir_name = os.path.dirname(path) or "."
    try:
        if create_dirs:
            ensure_dirs(dir_name, silent=silent)
        write_mode = "wb" if dump_format in ["dill", "joblib"] else "w"
        with tempfile.NamedTemporaryFile(write_mode, dir=dir_name, delete=False) as tf:
            if dump_format == "joblib":
                joblib.dump(obj, tf)
            elif dump_format == "dill":
                dill.dump(obj, tf)
            elif dump_format == "json":
                json.dump(obj, tf, default=str)
            temp_path = tf.name
        os.rename(temp_path, path)
    except Exception as e:
        print(f"ERROR: failed to save file {path}: {e}")
        if raise_on_error:
            raise e
        return False
    return True


def decompose_s3_path(s3_path: str) -> Tuple[str, str]:
    """Get bucket and key from full s3 path.

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".

    Returns:
        Tuple of bucket and key.
    """
    assert s3_path.startswith(
        "s3://"
    ), f"s3_path must start with s3:// invalid s3_path='{s3_path}'"
    match = re.match("s3://([^/]+)/(.+)", s3_path)
    if not match:
        raise ValueError(f"ERROR: not a valid s3_path: {s3_path}")
    bucket, key = match.groups()
    return (bucket, key)


def upload_file_to_s3(
    local_path: str,
    s3_path: str,
    silent: bool = False,
    raise_on_error: bool = True,
    boto3_kwargs: Optional[Dict[str, Union[str, float]]] = None,
) -> bool:
    """Upload a file from local machine to s3.

    Args:
        local_path: Path on local machine.
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".
        silent: Whether to suppress printing info message. (Defaults to False)
        raise_on_error: Whether to raise exception on any errors. (Defaults to True)
        boto3_kwargs: The parameters for s3.meta.client.upload_file() function.

    Returns:
        Boolean of whether the file was successfully uploaded.
    """
    if boto3_kwargs is None:
        boto3_kwargs = {}
    bucket, key = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    try:
        if not silent:
            print(f"Uploading file from '{local_path}' to '{s3_path}'")
        s3_client.upload_file(local_path, bucket, key, **boto3_kwargs)
    except Exception as e:
        print(f"ERROR: failed to upload from '{local_path}' to '{s3_path}': {e}")
        if raise_on_error:
            raise e
        return False
    return True


def download_file_from_s3(
    s3_path: str,
    local_path: str,
    create_dirs: bool = True,
    silent: bool = False,
    raise_on_error: bool = True,
    boto3_kwargs: Optional[Dict[str, Union[str, float]]] = None,
) -> bool:
    """Download a file from s3 to local machine.

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".
        local_path: Path on local machine.
        create_dirs: Whether the path directory should be created. (Defaults to True)
        silent: Whether to print debug information.
        raise_on_error: Whether to raise exception on any errors. (Defaults to True)
        boto3_kwargs: The parameters for s3.meta.client.download_fileobj() function.

    Returns:
        Boolean of whether the file was successfully downloaded.
    """
    if boto3_kwargs is None:
        boto3_kwargs = {}
    bucket, key = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    try:
        if not silent:
            print(f"Downloading file from '{s3_path}' to '{local_path}'")
        dir_name = os.path.dirname(local_path)
        if create_dirs:
            ensure_dirs(dir_name, silent=silent)
        with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
            s3_client.download_fileobj(bucket, key, tf, **boto3_kwargs)
            temp_path = tf.name
        os.rename(temp_path, local_path)
    except Exception as e:
        print(f"ERROR: failed to download from {s3_path} to {local_path}: {e}")
        if raise_on_error:
            raise e
        return False
    return True


def exists_in_s3(s3_path: str) -> Optional[bool]:
    """Check whether a fully specified s3 path exists.

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".

    Returns:
        Boolean of whether the file exists on s3 (None if there was an error.)
    """
    bucket, key = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError:
        return False
    except Exception as e:
        print(
            f"ERROR: unexpected exception checking existence of s3_path={s3_path}"
            f": {e}"
        )
        return None
    return True


def delete_from_s3(s3_path: str) -> bool:
    """Delete a path from s3

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".

    Returns:
        Boolean of whether the delete was successful.
    """
    bucket, key = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception as e:
        print(f"ERROR: unexpected exception deletion of s3_path={s3_path}: {e}")
        return False
    return True


def list_s3_keys(
    s3_path: str,
    inc_regex: Optional[str] = None,
    exc_regex: Optional[str] = None,
    fld_regex: Optional[str] = None,
) -> Union[List[Dict[str, Any]], List[Tuple[Any]]]:
    """List s3 object under a particular path.

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".
        inc_regex: regular expression to apply as include filter to fetched keys.
        exc_regex: regular expression to apply as exclude filter to fetched keys.
        fld_regex: regular expression for matching fields to show for each object.
            (Defaults to None and would only list the keys)

    Returns:
        List of s3 object information tuples or dictionaries.
    """
    bucket, prefix = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    if inc_regex:
        inc_re = re.compile(inc_regex)
    if exc_regex:
        exc_re = re.compile(exc_regex)
    return [
        info["Key"]
        if not fld_regex
        else [
            (
                f"{v/1048576:,.3f} MB"
                if isinstance(v, int)
                else v.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(v, datetime.datetime)
                else v
            )
            for k, v in info.items()
            if re.search(fld_regex, k, re.IGNORECASE)
        ]
        for info in s3_client.list_objects(Bucket=bucket, Prefix=prefix)["Contents"]
        if (not inc_regex or inc_re.search(info["Key"]))
        and (not exc_regex or not exc_re.search(info["Key"]))
    ]
