"""Utilities for organizing photos and videos using ExifTool metadata.

This module scans media files, extracts capture time and location, and builds
canonical destination filenames for deduplication and library cleanup.

Dependencies:
    ExifTool must be installed and on PATH (``brew install exiftool``).
    Python: ``pyexiftool``, ``pandas``, ``exifread``, ``reverse_geocode``, etc.
    Optional HEIC support: ``libheif``, ``pyheif``, ``piexif``.

Capture-date extraction uses a **tiered minimum** strategy (see
``get_create_dt_from_metadata``): among tags that mean "when was this recorded",
the earliest valid timestamp usually reflects the true capture time because
re-exports and re-wraps tend to push container dates *forward*, not backward.
``ModifyDate`` and similar edit timestamps are never considered.

**Timezone policy:** All capture times are normalized to **naive UTC** (including
destination filenames). Tag offsets are converted to UTC for consistent ordering;
we do not infer timezone from GPS or the local machine.

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
    "EXIF:DateTimeOriginal",
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

# Capture-time tags: earliest valid value wins within a tier (re-exports are usually later).
_CAPTURE_DT_FIELDS = (
    "EXIF:DateTimeOriginal",
    "QuickTime:CreationDate",  # Apple Photos original capture (not QuickTime:CreateDate)
    "XMP:DateTimeOriginal",
    "RIFF:DateTimeOriginal",
    "IPTC:DateCreated",
    "Composite:SubSecDateTimeOriginal",
)
_CAPTURE_DT_SECONDARY = (
    "EXIF:CreateDate",
    "XMP:CreateDate",
    "PNG:CreationTime",
)
# Container timestamps — often rewritten on export; only used if no capture tier matches.
_CONTAINER_DT_FIELDS = (
    "QuickTime:MediaCreateDate",
    "QuickTime:TrackCreateDate",
    "QuickTime:CreateDate",
)
_FILE_DT_FIELDS = (
    "File:FileModifyDate",
    "File:FileCreateDate",
)
_CREATE_DT_TIERS = (
    _CAPTURE_DT_FIELDS,
    _CAPTURE_DT_SECONDARY,
    _CONTAINER_DT_FIELDS,
    _FILE_DT_FIELDS,
)
# Suffixes for extra capture tags ExifTool may return under unfamiliar group names.
_CAPTURE_DT_KEY_SUFFIXES = ("DateTimeOriginal", "CreationDate")

_INVALID_DT_PREFIXES = ("0000:", "0001:", "1970:01:01")


def get_hash_from_metadata(metadata: dict) -> str:
    """Build a short content fingerprint from stable ExifTool tags.

    Uses only ``HASH_KEYS`` (dimensions, MIME type, select date/resolution fields)
    so visually identical or duplicate exports hash the same even if filenames
    differ. The hash is appended to destination names to disambiguate collisions
    at the same capture second and location.

    Args:
        metadata: ExifTool metadata dict (``SourceFile`` plus group-prefixed tags).

    Returns:
        First 16 hex chars of a SHA-1 digest of sorted key/value pairs.
    """
    hash_content = str(sorted([(k, v) for k, v in metadata.items() if k in HASH_KEYS]))
    return hashlib.sha1(hash_content.encode()).hexdigest()[:16]


def get_all_file_paths(
    root_dir,
    match_re=None,
    match_case_sensitive=False,
    not_match_re=None,
    not_match_case_sensitive=False,
) -> List[str]:
    """Walk ``root_dir`` and collect file paths matching optional regex filters.

    Args:
        root_dir: Directory tree to walk recursively.
        match_re: If set, only paths matching this regex are included.
        match_case_sensitive: Whether ``match_re`` is case-sensitive.
        not_match_re: If set, paths matching this regex are excluded.
        not_match_case_sensitive: Whether ``not_match_re`` is case-sensitive.

    Returns:
        List of absolute or relative file paths (same form as ``os.walk`` joins).
    """
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


def get_all_media_file_paths(root_dir) -> List[str]:
    """Collect paths under ``root_dir`` for common photo/video extensions.

    Args:
        root_dir: Root directory to scan.

    Returns:
        File paths whose names match ``MEDIA_EXT_RE`` (jpg, mov, heic, etc.).
    """
    return get_all_file_paths(root_dir, match_re=MEDIA_EXT_RE)


def get_metadata(file_path: str) -> dict:
    """Read all ExifTool metadata for a single file.

    Prefer this over ``get_exif``; ExifTool supports many more formats (MOV,
    HEIC, PNG, etc.) with consistent group-prefixed tag names.

    Args:
        file_path: Path to the media file.

    Returns:
        Metadata dict including ``SourceFile`` and tags like ``EXIF:DateTimeOriginal``.
    """
    with exiftool.ExifTool() as et:
        return et.get_metadata(file_path)


def get_metadatas(file_paths: List[str]) -> List[dict]:
    """Read ExifTool metadata for many files in one ExifTool process.

    Args:
        file_paths: Paths to media files.

    Returns:
        List of metadata dicts in the same order as ``file_paths``.
    """
    with exiftool.ExifTool() as et:
        return et.get_metadata_batch(file_paths)


def get_exif(file_path) -> dict:
    """Read EXIF from JPEG or HEIC using pure-Python libraries (deprecated).

    Prefer ``get_metadata``, which uses ExifTool and supports the same tag
    naming as the rest of this module.

    Args:
        file_path: Path ending in ``.jpg`` or ``.heic``.

    Returns:
        EXIF tag dict (exifread format for JPEG; piexif-derived for HEIC).

    Raises:
        Exception: If the extension is not supported.
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


def get_file_create_date(file_path) -> datetime.datetime:
    """Return filesystem birth time, or mtime on platforms without birth time.

    Used as a last-resort capture time when ExifTool returns no usable tags
    (e.g. screenshots or stripped metadata).

    Args:
        file_path: Path to the file.

    Returns:
        Naive UTC ``datetime`` from ``st_birthtime`` (macOS/BSD) or
        ``st_mtime`` (Linux fallback).
    """
    stat = os.stat(file_path)
    try:
        dt = stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        dt = stat.st_mtime
    return datetime.datetime.fromtimestamp(dt, tz=datetime.timezone.utc).replace(
        tzinfo=None
    )


def _get_if_exist(data, key):
    """Return ``data[key]`` if present, else ``None``.

    Args:
        data: Mapping (typically exifread output).
        key: Key to look up.

    Returns:
        Value at ``key``, or ``None``.
    """
    if key in data:
        return data[key]
    return None


def _convert_to_degress(ratios: List[exifread.utils.Ratio]) -> float:
    """Convert EXIF GPS DMS rationals to decimal degrees.

    Args:
        ratios: Three ``exifread.utils.Ratio`` values (degrees, minutes, seconds).

    Returns:
        Signed decimal degrees (caller applies N/S/E/W ref).
    """
    d = float(ratios[0].num) / float(ratios[0].den)
    m = float(ratios[1].num) / float(ratios[1].den)
    s = float(ratios[2].num) / float(ratios[2].den)

    return d + (m / 60.0) + (s / 3600.0)


def get_lat_lng_from_exif(exif) -> Optional[Tuple[float, float]]:
    """Parse latitude and longitude from exifread EXIF dict.

    Args:
        exif: Output of ``exifread.process_file``.

    Returns:
        ``(lat, lng)`` in decimal degrees, or ``None`` if GPS tags are missing.
    """
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
    """Read GPS coordinates from ExifTool composite tags.

    ExifTool pre-computes decimal lat/lng; prefer this over ``get_lat_lng_from_exif``
    when metadata comes from ``get_metadata``.

    Args:
        metadata: ExifTool metadata dict.

    Returns:
        ``(lat, lng)`` or ``None`` if composite GPS tags are absent.
    """
    lat = metadata.get("Composite:GPSLatitude")
    lng = metadata.get("Composite:GPSLongitude")
    return (lat, lng) if (lat and lng) else None


def get_location_from_metadata(metadata: dict) -> str:
    """Reverse-geocode GPS into a short location token for filenames.

    Format: ``{country_code}_{city_with_underscores}`` (e.g. ``CA_Toronto``).
    Empty string if no GPS data.

    Args:
        metadata: ExifTool metadata dict.

    Returns:
        Location token, or ``""``.
    """
    lat_lng = get_lat_lng_from_metadata(metadata)
    if lat_lng:
        res = reverse_geocode.get(lat_lng)
        return f"{res['country_code']}_{res['city'].replace(' ', '_')}"
    return ""


def get_location(file_path: str) -> str:
    """Reverse-geocode GPS for a single file.

    Args:
        file_path: Path to the media file.

    Returns:
        Location token from ``get_location_from_metadata``, or ``""``.
    """
    metadata = get_metadata(file_path)
    return get_location_from_metadata(metadata)


def _parse_metadata_datetime(value) -> Optional[pd.Timestamp]:
    """Parse one ExifTool datetime value into a naive UTC-normalized timestamp.

    Handles ExifTool's ``YYYY:MM:DD HH:MM:SS`` form (colons in the date part),
    timezone offsets, subseconds, and numeric Unix epochs. Rejects sentinel values
    (``0000:00:00``, pre-2000) via ``_INVALID_DT_PREFIXES`` and ``MIN_DT``.

    Timezone-aware values are converted to UTC then stored naive. This is the
    module's intentional policy: one consistent clock for comparisons, tiered
    ``min()``, and ``DestFileBase`` filenames (not local wall time at capture).

    Args:
        value: Tag value from ExifTool (str, ``Timestamp``, number, etc.).

    Returns:
        Parsed timestamp, or ``None`` if missing/invalid.
    """
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        dt = value
    elif isinstance(value, datetime.datetime):
        dt = pd.Timestamp(value)
    elif isinstance(value, (int, float)):
        if value <= 0:
            return None
        dt = pd.to_datetime(value, unit="s", utc=True)
    else:
        s = str(value).strip()
        if not s or s.startswith(_INVALID_DT_PREFIXES):
            return None
        # ExifTool default format uses colons in the date portion: YYYY:MM:DD ...
        if len(s) > 4 and s[4] == ":":
            s = s.replace(":", "-", 2)
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            return None

    if dt.tzinfo is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    if dt < MIN_DT:
        return None
    return dt


def _collect_valid_dts(
    metadata: dict, fields: Tuple[str, ...]
) -> List[pd.Timestamp]:
    """Parse and collect all valid datetimes for a fixed list of tag names.

    Args:
        metadata: ExifTool metadata dict.
        fields: Tag names to read (e.g. ``_CAPTURE_DT_FIELDS``).

    Returns:
        List of valid timestamps (may be empty).
    """
    return [
        dt
        for fld in fields
        if (dt := _parse_metadata_datetime(metadata.get(fld))) is not None
    ]


def _collect_dynamic_capture_dts(metadata: dict) -> List[pd.Timestamp]:
    """Discover extra capture-date tags not listed in ``_CREATE_DT_TIERS``.

    Scans metadata keys ending in ``DateTimeOriginal`` or ``CreationDate``.
    Skips keys containing ``Modify`` (edit times, not capture) and keys already
    handled explicitly. This catches vendor-specific groups without maintaining
    an exhaustive ExifTool tag list.

    Args:
        metadata: ExifTool metadata dict.

    Returns:
        List of valid timestamps from matching keys.
    """
    known = set(
        f
        for tier in _CREATE_DT_TIERS
        for f in tier
    )
    dts = []
    for key, value in metadata.items():
        if key in known or "Modify" in key:
            continue
        if not any(key.endswith(suffix) for suffix in _CAPTURE_DT_KEY_SUFFIXES):
            continue
        if (dt := _parse_metadata_datetime(value)) is not None:
            dts.append(dt)
    return dts


def get_create_dt_from_metadata(metadata: dict) -> Optional[pd.Timestamp]:
    """Estimate when media was captured or originally recorded.

    **Why tiered minimum (not first-match or global min)?**

    * **First-match** breaks when an important tag is missing from the list
      (e.g. Apple ``QuickTime:CreationDate`` vs ``QuickTime:CreateDate``).
    * **Global min** over all date tags lets corrupt ``ModifyDate`` or stale
      ``FileModifyDate`` beat the real capture time.
    * **Tiered min** limits ``min()`` to tags with similar meaning per tier, and
      only falls back when higher tiers have no valid values.

    Re-exports (Photos export, re-wrap) usually set container ``CreateDate`` /
    ``MediaCreateDate`` *later* than the true capture; among capture-semantics
    tags, the earliest valid date is typically correct.

    Tiers (see module constants):

    1. Capture: ``DateTimeOriginal``, ``QuickTime:CreationDate`` (Apple original),
       XMP/RIFF/IPTC, plus dynamic ``*DateTimeOriginal`` / ``*CreationDate`` scan.
       ``min()`` within tier 1.
    2. Secondary capture: ``EXIF:CreateDate``, etc. Only if tier 1 is empty
       (avoids a bad ``CreateDate`` overriding a good ``DateTimeOriginal``).
    3. Container: QuickTime ``MediaCreateDate`` / ``CreateDate`` (re-export fallback).
    4. Filesystem: ``File:FileModifyDate`` / ``FileCreateDate``.

    ``*ModifyDate`` tags are never used.

    Args:
        metadata: ExifTool metadata dict.

    Returns:
        Best-effort capture timestamp (naive UTC), or ``None`` if no valid tag.
    """
    for tier_idx, fields in enumerate(_CREATE_DT_TIERS):
        dts = _collect_valid_dts(metadata, fields)
        if tier_idx == 0:
            dts.extend(_collect_dynamic_capture_dts(metadata))
        if dts:
            return min(dts)

    print(
        "ERROR: Failed to extract create_dt from metadata; "
        f"tried tiers={_CREATE_DT_TIERS}, SourceFile={metadata.get('SourceFile')}"
    )
    return None


def get_create_dt(file_path: str) -> Optional[pd.Timestamp]:
    """Read metadata for one file and return estimated capture time.

    Args:
        file_path: Path to the media file.

    Returns:
        Result of ``get_create_dt_from_metadata``, or ``None``.
    """
    metadata = get_metadata(file_path)
    return get_create_dt_from_metadata(metadata)


def get_batches(lst: List, batch_size: int):
    """Yield consecutive slices of ``lst`` for batch processing.

    Args:
        lst: Sequence to chunk.
        batch_size: Maximum items per batch (clamped to at least 1).

    Yields:
        Sublists of ``lst`` of length up to ``batch_size``.
    """
    batch_size = max(1, batch_size)
    return (lst[i : i + batch_size] for i in range(0, len(lst), batch_size))


class GetMetaDatasAugmented:
    """Batch-load ExifTool metadata and attach destination filename fields.

    Callable intended for ``multiprocessing.Pool.imap``: one ExifTool process
    per batch, then per-file enrichment (capture time, location, content hash,
    ``DestFileBase``).

    Destination pattern (timestamp is naive UTC)::
        {YYYYMMDD_HHMMSS}[__{location}][__{orig_base}][__{hash}].{ext}
    """

    def __init__(
        self,
        include_metadata_hash_in_dest: bool = True,
        include_orig_name_in_dest: bool = False,
    ):
        """Configure destination filename components.

        Args:
            include_metadata_hash_in_dest: Append ``MetadataHash`` before extension
                to separate collisions at the same second/location.
            include_orig_name_in_dest: Include original basename (without extension)
                in the destination name for traceability.
        """
        self.include_metadata_hash_in_dest = include_metadata_hash_in_dest
        self.include_orig_name_in_dest = include_orig_name_in_dest

    def __call__(self, file_paths: List[str]) -> List[dict]:
        """Load metadata for a batch and set ``DestFileBase`` on each dict.

        If ``get_create_dt_from_metadata`` returns ``None``, falls back to
        ``get_file_create_date`` so renaming never crashes on missing EXIF.

        Args:
            file_paths: List of media file paths (one batch).

        Returns:
            Metadata dicts with ``Location``, ``MetadataHash``, ``DestFileBase``.
        """
        assert isinstance(file_paths, list)
        with exiftool.ExifTool() as et:
            metadatas = et.get_metadata_batch(file_paths)
        for metadata in metadatas:
            file_path = metadata["SourceFile"]
            create_dt = get_create_dt_from_metadata(metadata)
            if create_dt is None:
                create_dt = pd.Timestamp(get_file_create_date(file_path))
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
    """Load augmented metadata for many files using a process pool.

    Results are cached via ``@memorize`` (see ``cacheutils``). Each worker runs
    ``GetMetaDatasAugmented`` on a batch so ExifTool amortizes startup cost.

    Note:
        ``batch_size`` argument is overridden to 32 in the function body.

    Args:
        file_paths: All media paths to process.
        batch_size: Ignored; kept for API compatibility (actual batch size is 32).
        include_orig_name_in_dest: Passed to ``GetMetaDatasAugmented``.

    Returns:
        Flat list of metadata dicts with ``DestFileBase`` set.
    """
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
) -> List[dict]:
    """Organize media into year folders with canonical capture-based filenames.

    Scans ``src_root_dir``, computes destination names from metadata (capture time,
    optional location/hash), groups by ``DestFileBase``, and moves or copies one
    file per destination. Duplicate sources mapping to the same dest are reported.

    Layout: ``{dest_root_dir}/{YYYY}/{YYYYMMDD_HHMMSS}__...ext``

    Args:
        src_root_dir: Tree of source photos/videos.
        dest_root_dir: Library root; year subdirs are created as needed.
        dry_run: If True, do not move/copy or create dirs (still writes log rows).
        refresh_metacache: If True, bypass ``get_metadatas_mproc`` cache.
        move_or_copy: ``"move"`` (``shutil.move``) or ``"copy"`` (``shutil.copy2``).
        log_file_path: Log file path; defaults to ``/dev/null`` (dry run) or
            ``{dest_root_dir}/log.txt``.

    Returns:
        List of per-file result dicts (``dest``, ``src``, ``other_srcs``, ``error``).

    Raises:
        AssertionError: If ``move_or_copy`` is not ``"move"`` or ``"copy"``.
    """
    assert move_or_copy in [
        "move",
        "copy",
    ], f"move_or_copy='{move_or_copy}' must be 'move' or 'copy"
    move_or_copy = shutil.move if move_or_copy == "move" else shutil.copy2
    print("Getting all source media files...")
    file_paths = sorted(get_all_media_file_paths(src_root_dir))
    print(f"Retrieved {len(file_paths):,.0f} media files from {src_root_dir}")

    print("Getting metadatas for all media files...")
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
    print(f"Moving files. Will log output to {log_file_path} ...")
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
