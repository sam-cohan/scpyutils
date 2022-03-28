"""
This module makes use of exiftool wrapper library pyexiftool.
Make sure the exiftool is installed and available in your PATH.
brew install exiftool && pip install pyexiftool

You also need to install pyheif which requires to first install libheif:
brew install libheif

To install requirements:
pip install exifread pandas Pillow piexif pyheif reverse_geocode tqdm

Author: Sam Cohan
"""
import datetime
import hashlib
import json
import multiprocessing as mproc
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import exifread
import exiftool
import pandas as pd
import reverse_geocode
from tqdm.notebook import tqdm_notebook

from scpyutils.cacheutils import memorize

HASH_KEYS = {
    "File:FileSize",
    "File:MIMEType",
    "File:ImageWidth",
    "File:ImageHeight",
    "Composite:ImageSize",
    "Composite:Megapixels",
    # JPG specific tags:
    "EXIF:CreateDate",
    "EXIF:DatetimeOriginal",
    "EXIF:ModifyDate",
    "EXIF:XResolution",
    "EXIF:YResolution",
    # PNG specific tags:
    "PNG:ImageWidth",
    "PNG:ImageHeight",
    "PNG:BitDepth",
    # MOV/MP4 specific tags:
    "QuickTime:CreateDate",
    "QuickTime:ModifyDate",
    "QuickTime:MediaCreateDate",
    "QuickTime:MediaModifyDate",
    "QuickTime:TrackCreateDate",
    "QuickTime:TrackModifyDate",
    "QuickTime:MediaDataSize",
    "QuickTime:MediaDuration",
    "QuickTime:AudioFormat",
    "QuickTime:AudioBitsPerSample",
    "QuickTime:AudioChannels",
    # AVI specific tags:
    "RIFF:DateTimeOriginal",
}

MEDIA_EXT_RE = "(arw|avi|cr2|dat|divx|gif|heic|jpe?g|mkv|mp4|mov|mpg|png|tiff?)$"

MIN_DT = pd.to_datetime("2000-01-01")


def get_hash_from_metadata(metadata: dict):
    hash_content = str(sorted([(k, v) for k, v in metadata.items() if k in HASH_KEYS]))
    return hashlib.sha1(hash_content.encode()).hexdigest()[:16]


def get_all_file_paths(
    root_dir,
    match_re=None,
    match_case_sensitive=False,
    not_match_re=None,
    not_match_case_sensitive=False,
):
    match_re_compile = None
    not_match_re_compile = None
    if match_re:
        match_re_flags = tuple() if match_case_sensitive else (re.IGNORECASE,)
        match_re_compile = re.compile(match_re, *match_re_flags)
    if not_match_re:
        not_match_re_flags = tuple() if not_match_case_sensitive else (re.IGNORECASE,)
        not_match_re_compile = re.compile(match_re, *not_match_re_flags)
    all_file_paths = []

    for subdir, _dirs, files in tqdm_notebook(os.walk(root_dir)):
        for file in files:
            file_path = os.path.join(subdir, file)
            if not_match_re_compile and not_match_re_compile.search(file_path):
                continue
            if match_re and match_re_compile.search(file_path):
                all_file_paths.append(file_path)
    return all_file_paths


def get_all_media_file_paths(root_dir):
    return get_all_file_paths(root_dir, match_re=MEDIA_EXT_RE)


def get_metadata(file_path: str) -> dict:
    with exiftool.ExifTool() as et:
        return et.get_metadata(file_path)


def get_metadatas(file_paths: List[str]) -> List[dict]:
    with exiftool.ExifTool() as et:
        return et.get_metadata_batch(file_paths)


def get_exif(file_path):
    """Deprecated function for getting exif from jpg and heic files.

    Make use of `get_metadata` which wraps the command line exiftool.
    """
    if file_path[-4:].lower() == ".jpg":
        import exifread

        return exifread.process_file(open(file_path, "rb"))
    if file_path[-5:].lower() == ".heic":
        # Install pyheif with:
        # brew install libheif
        # pip install git+https://github.com/david-poirier-csn/pyheif.git
        import piexif
        import pyheif

        return {
            f"{k} {piexif.TAGS[k][kk]['name']}": vv
            for k, v in piexif.load(
                pyheif.read_heif(file_path).metadata[0]["data"]
            ).items()
            for kk, vv in (v.items() if v and isinstance(v, dict) else [])
        }
    raise Exception("File not supported!")


def get_file_create_date(file_path):
    stat = os.stat(file_path)
    try:
        dt = stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        dt = stat.st_mtime
    return datetime.datetime.fromtimestamp(dt)


def _get_if_exist(data, key):
    if key in data:
        return data[key]
    return None


def _convert_to_degress(ratios: List[exifread.utils.Ratio]):
    """
    Helper function to convert the GPS coordinates stored in the EXIF
    to degress in float format.
    """
    d = float(ratios[0].num) / float(ratios[0].den)
    m = float(ratios[1].num) / float(ratios[1].den)
    s = float(ratios[2].num) / float(ratios[2].den)

    return d + (m / 60.0) + (s / 3600.0)


def get_lat_lng_from_exif(exif) -> Optional[Tuple[float, float]]:
    """Get latitude and longitude from exif."""
    lat = None
    lng = None

    gps_latitude = _get_if_exist(exif, "GPS GPSLatitude")
    gps_latitude_ref = _get_if_exist(exif, "GPS GPSLatitudeRef")
    gps_longitude = _get_if_exist(exif, "GPS GPSLongitude")
    gps_longitude_ref = _get_if_exist(exif, "GPS GPSLongitudeRef")

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = _convert_to_degress(gps_latitude.values)
        if gps_latitude_ref.values[0] != "N":
            lat = 0 - lat

        lng = _convert_to_degress(gps_longitude.values)
        if gps_longitude_ref.values[0] != "E":
            lng = 0 - lng

    return (lat, lng) if (lat and lng) else None


def get_lat_lng_from_metadata(metadata: dict) -> Optional[Tuple[float, float]]:
    lat = metadata.get("Composite:GPSLatitude")
    lng = metadata.get("Composite:GPSLongitude")
    return (lat, lng) if (lat and lng) else None


def get_location_from_metadata(metadata: dict) -> str:
    lat_lng = get_lat_lng_from_metadata(metadata)
    if lat_lng:
        res = reverse_geocode.get(lat_lng)
        return f"{res['country_code']}_{res['city'].replace(' ', '_')}"
    return ""


def get_location(file_path: str) -> Optional[Tuple[str, str]]:
    metadata = get_metadata(file_path)
    return get_location_from_metadata(metadata)


def try_get_dt(dt_str: Optional[str]) -> Optional[pd.Timestamp]:
    if not dt_str:
        return None
    try:
        return pd.to_datetime(dt_str.replace(":", "-", 2)).tz_localize(None)
    except:  # noqa: E722
        pass


def get_create_dt_from_metadata(metadata: dict) -> Optional[pd.Timestamp]:
    dts = [
        try_get_dt(metadata.get(fld))
        for fld in [
            "EXIF:DateTimeOriginal",
            "EXIF:ModifyDate",
            "QuickTime:CreateDate",
            "RIFF:DateTimeOriginal",
            "File:FileModifyDate",
        ]
    ]
    dts = [x for x in dts if x is not None]
    if not dts:
        print(f"ERROR: Failed to extract create_dt from metadata={metadata}")
        return None
    dt = min(dts)
    if dt < MIN_DT:
        print(f"ERROR: create_dt before {MIN_DT} is surely impossible")
        return None
    return dt


def get_create_dt(file_path: str) -> Optional[pd.Timestamp]:
    metadata = get_metadata(file_path)
    return get_create_dt_from_metadata(metadata)


def get_batches(lst: List, batch_size: int) -> List[List]:
    batch_size = max(1, batch_size)
    return (lst[i : i + batch_size] for i in range(0, len(lst), batch_size))


class GetMetaDatasAugmented:
    def __init__(
        self,
        include_metadata_hash_in_dest: bool = True,
        include_orig_name_in_dest: bool = False,
    ):
        self.include_metadata_hash_in_dest = include_metadata_hash_in_dest
        self.include_orig_name_in_dest = include_orig_name_in_dest

    def __call__(self, file_paths: List[str]) -> List[dict]:
        assert isinstance(file_paths, list)
        with exiftool.ExifTool() as et:
            metadatas = et.get_metadata_batch(file_paths)
        for metadata in metadatas:
            create_dt = get_create_dt_from_metadata(metadata)
            file_path = metadata["SourceFile"]
            base_name, ext = os.path.basename(file_path).rsplit(".", 1)
            loc = get_location_from_metadata(metadata)
            metadata["Location"] = loc
            metadata_hash = get_hash_from_metadata(metadata)
            metadata["MetadataHash"] = metadata_hash
            new_file_name = (
                f"{create_dt.strftime('%Y%m%d_%H%M%S')}"
                f"{f'__{loc}' if loc else ''}"
                f"{f'__{base_name}' if self.include_orig_name_in_dest else ''}"
                f"{f'__{metadata_hash}' if self.include_metadata_hash_in_dest else ''}"
                f".{ext.lower()}"
            )
            metadata["DestFileBase"] = new_file_name

        return metadatas


@memorize(
    local_dir="./.cache",
)
def get_metadatas_mproc(
    file_paths: List[str],
    batch_size: int = 64,
    include_orig_name_in_dest: bool = False,
) -> List[Dict]:
    batch_size = 32
    file_path_batches = list(get_batches(file_paths, batch_size))

    with mproc.Pool(mproc.cpu_count()) as pool:
        matadata_batches = list(
            tqdm_notebook(
                pool.imap(
                    GetMetaDatasAugmented(
                        include_orig_name_in_dest=include_orig_name_in_dest,
                    ),
                    file_path_batches,
                ),
                total=len(file_path_batches),
            )
        )
        metadatas = [m for mb in matadata_batches for m in mb]
    return metadatas


def cleaup_media_files(
    src_root_dir: str,
    dest_root_dir: str,
    dry_run: bool = True,
    refresh_metacache: bool = False,
    move_or_copy: str = "move",
    log_file_path: str = None,
):
    assert move_or_copy in [
        "move",
        "copy",
    ], f"move_or_copy='{move_or_copy}' must be 'move' or 'copy"
    move_or_copy = shutil.move if move_or_copy == "move" else shutil.copy2
    print("Getting all source media files...")
    file_paths = sorted(get_all_media_file_paths(src_root_dir))
    print(f"Retrieved {len(file_paths):,.0f} media files from {src_root_dir}")

    print(f"Getting metadatas for all media files...")
    metadatas = get_metadatas_mproc(  # pylint: disable=unexpected-keyword-arg
        file_paths=file_paths, __force_refresh=refresh_metacache
    )
    print(f"Retrieved {len(metadatas):,.0f} metadatas.")

    dest_file_src_files = defaultdict(list)
    for metadata in metadatas:
        dest_file_src_files[metadata["DestFileBase"]].append(metadata["SourceFile"])
    duplicates = [
        (dest, srcs) for dest, srcs in dest_file_src_files.items() if len(srcs) > 1
    ]
    if duplicates:
        print(f"WARNING: found {len(duplicates):,.0f} duplicate source media files.")

    # Create yearly folders for destination
    dest_years = sorted(
        set([dest_file_base[:4] for dest_file_base in dest_file_src_files])
    )
    for dest_year in dest_years:
        dest_dir = os.path.join(dest_root_dir, dest_year)
        if not os.path.exists(dest_dir):
            print(f"WARNING: will create destination directory {dest_dir}")
            if not dry_run:
                os.makedirs(dest_dir)

    if log_file_path is None:
        if dry_run:
            log_file_path = "/dev/null"
        else:
            log_file_path = os.path.join(dest_root_dir, "log.txt")
    print("Moving files. Will log output to {log_file_path} ...")
    rows = []
    dest_exists_cnt = 0
    unexpected_err_cnt = 0
    success_cnt = 0
    with open(log_file_path, "a") as log_file:
        for dest_file_base, src_file_paths in tqdm_notebook(
            sorted(dest_file_src_files.items())
        ):
            # Make sure the dest_file_path includes the year of the file.
            dest_year = dest_file_base[:4]
            dest_file_path = os.path.join(dest_root_dir, dest_year, dest_file_base)

            # Move the first available source file.
            src_file_path = src_file_paths[0]
            row = dict(
                dest=dest_file_path,
                src=src_file_path,
                other_srcs=src_file_paths[1:],
            )
            if os.path.exists(dest_file_path):
                dest_exists_cnt += 1
                row["src"] = ""
                row["other_srcs"] = src_file_paths
                row["error"] = "DESTINATION_EXISTS"
            else:
                error = ""
                if not dry_run:
                    try:
                        move_or_copy(src_file_path, dest_file_path)
                        success_cnt += 1
                    except Exception as e:
                        unexpected_err_cnt += 1
                        error = str(e)
                        print(f"unable to move src={src_file_path}: {e}")
                log_file.write(f"{json.dumps(row)}\n")
                row["error"] = error

            rows.append(row)

        if success_cnt:
            print(f"Successfully moved {success_cnt:,.0f} files!")
        else:
            print("NO FILES WERE MOVED!")
        if dest_exists_cnt:
            print(f"WARNING: {dest_exists_cnt:,.0f} destinations already existed!")
        if unexpected_err_cnt:
            print(f"ERROR: Encountered {unexpected_err_cnt:,.0f} unexpected errors!")
    return rows
