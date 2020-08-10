"""
Utilities related to persisting data to file storage.
"""
import json
import os
import re
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import boto3
import botocore
import joblib


def ensure_dirs(
    dir_name: str, silent: bool = False, raise_on_error: bool = True
) -> bool:
    """Create directories if they do not exist.

    Args:
        dir_name: Directory name on local machine.
        silent: Whether to suppress printing info message. (Defaults to False)
        raise_on_error: Whether to raise exception on any erros. (Defaults to True)

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
        dump_format: format of the saved file. Currently support {'json', 'joblib'}.
            (Defaults to 'joblib')
        create_dirs: Whether the path directory should be created. (Defaults to True)
        silent: Whether to suppress printing info message. (Defaults to False)
        raise_on_error: Whether to raise exception on any erros. (Defaults to True)

    Returns:
        Boolean of whether the file was successfully saved.
    """
    if not silent:
        print(f"Writing to path={path}")
    dir_name = os.path.dirname(path) or "."
    try:
        if create_dirs:
            ensure_dirs(dir_name, silent=silent)
        write_mode = "wb" if dump_format == "joblib" else "w"
        with tempfile.NamedTemporaryFile(write_mode, dir=dir_name, delete=False) as tf:
            if dump_format == "joblib":
                joblib.dump(obj, tf)
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
    local_path: str, s3_path: str, silent: bool = False, raise_on_error: bool = True,
) -> bool:
    """Upload a file from local machine to s3.

    Args:
        local_path: Path on local machine.
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".
        silent: Whether to suppress printing info message. (Defaults to False)
        raise_on_error: Whether to raise exception on any erros. (Defaults to True)

    Returns:
        Boolean of whether the file was successfully uploaded.
    """
    bucket, key = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    try:
        if not silent:
            print(f"Uploading file from '{local_path}' to '{s3_path}'")
        s3_client.upload_file(local_path, bucket, key)
    except Exception as e:
        print(f"ERROR: failed to upload from '{local_path}' to '{s3_path}': {e}")
        if raise_on_error:
            raise e
        return False
    return True


def download_file_from_s3(
    s3_path: str, local_path: str, silent: bool = False, raise_on_error: bool = True,
) -> bool:
    """Download a file from s3 to local machine.

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".
        local_path: Path on local machine.
        silent: Whether to print debug information.
        raise_on_error: Whether to raise exception on any erros. (Defaults to True)

    Returns:
        Boolean of whether the file was successfully downloaded.
    """
    bucket, key = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    try:
        if not silent:
            print(f"Downloading file from '{s3_path}' to '{local_path}'")
        dir_name = os.path.dirname(local_path)
        with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
            s3_client.download_fileobj(bucket, key, tf)
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
        Boolean of whether the delete was sucessful.
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
    flds: Optional[Iterable[str]] = ("Key",),
    include_filter: Optional[str] = None,
    exclude_filter: Optional[str] = None,
) -> Union[List[Dict[str, Any]], List[Tuple[Any]]]:
    """List s3 object under a particular path.

    Args:
        s3_path: Full path on s3 in format "s3://<bucket_name>/<obj_path>".
        flds: Iterable of fields from the object info to include in result.
            (Defaults to ('Key',))
        include_filter: regular expression to apply as include filter to fetched keys.
        exclude_filter: regular expression to apply as exclude filter to fetched keys.

    Returns:
        List of s3 object information tuples or dictionaries.
    """
    bucket, prefix = decompose_s3_path(s3_path)
    s3_client = boto3.client("s3")
    if include_filter:
        include_re = re.compile(include_filter)
    if exclude_filter:
        exclude_re = re.compile(exclude_filter)
    return [
        info if not flds else [v for k, v in info.items() if k in flds]
        for info in s3_client.list_objects(Bucket=bucket, Prefix=prefix)["Contents"]
        if (not include_filter or include_re.search(info["Key"]))
        and (not exclude_filter or not exclude_re.search(info["Key"]))
    ]

