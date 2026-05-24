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

**Timezone policy:** Capture times are stored as **naive UTC** for comparisons
and ``YYYYMMDD_HHMMSS`` destination filenames. We do not infer timezone from GPS
or the local machine.

Author: Sam Cohan
"""

import datetime
import filecmp
import hashlib
import json
import multiprocessing as mproc
import os
import re
import shutil
import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import exifread
import exiftool
import pandas as pd
import reverse_geocode

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

# Bump when raw-cache shape or ExifTool batch behavior changes materially.
DEST_LOGIC_VERSION = 1

_LIVE_PHOTO_EXTS = {".heic", ".mov"}


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string for log rows."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _in_notebook() -> bool:
    """Return True when running inside a Jupyter notebook kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def _iter_progress(
    iterable: Iterable,
    *,
    progress: bool = True,
    desc: Optional[str] = None,
    total: Optional[int] = None,
):
    """Wrap an iterable with tqdm (notebook or CLI)."""
    if not progress:
        return iterable
    if _in_notebook():
        from tqdm.notebook import tqdm_notebook

        return tqdm_notebook(iterable, desc=desc, total=total)
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total)


def _log_event(log_file, row: dict, session: Optional[dict] = None) -> dict:
    """Append a timestamped JSON event to the cleanup log file."""
    row = {**row, "logged_at": _utc_now_iso()}
    if session:
        row = {**session, **row}
    log_file.write(f"{json.dumps(row)}\n")
    return row


def _print_destination_exists(row: dict) -> None:
    """Print a human-readable summary when the destination path already exists."""
    label = row.get("error", "DESTINATION_EXISTS")
    print(f"{label} (logged_at={row.get('logged_at', 'n/a')}):")
    print(f"  dest: {row['dest']}")
    print(f"  src:  {row['src']}")
    if row.get("duplicate_reloc"):
        print(f"  duplicate_reloc: {row['duplicate_reloc']}")
    if row.get("quarantine_reloc"):
        print(f"  quarantine_reloc: {row['quarantine_reloc']}")
    for key in ("src_size", "dest_size", "src_sha256", "dest_sha256"):
        if key in row:
            print(f"  {key}: {row[key]}")
    for other_src in row.get("other_srcs") or []:
        print(f"  other_src: {other_src}")


def _normalize_exclude_roots(exclude_dirs: Optional[List[str]]) -> List[str]:
    """Return absolute, existing exclude roots (deduplicated)."""
    if not exclude_dirs:
        return []
    roots: List[str] = []
    seen: Set[str] = set()
    for path in exclude_dirs:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            roots.append(abs_path)
    return roots


def _is_excluded_path(file_path: str, exclude_roots: List[str]) -> bool:
    """Return True if ``file_path`` is under any exclude root."""
    abs_path = os.path.abspath(file_path)
    for root in exclude_roots:
        if abs_path == root or abs_path.startswith(root + os.sep):
            return True
    return False


def _file_size(path: str) -> Optional[int]:
    """Return file size in bytes, or None if unreadable."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Return hex SHA-256 of file contents, or None if unreadable."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _files_are_identical(path_a: str, path_b: str) -> bool:
    """Return True if two files have the same size and byte content."""
    try:
        if os.path.samefile(path_a, path_b):
            return True
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        return filecmp.cmp(path_a, path_b, shallow=False)
    except OSError:
        return False


def _identical_duplicate_dest_path(
    identical_duplicates_dir: str,
    dest_year: str,
    dest_file_base: str,
    src_file_path: str,
) -> str:
    """Build a unique path under ``identical_duplicates_dir`` for a duplicate source."""
    dup_dir = os.path.join(identical_duplicates_dir, dest_year)
    src_base = os.path.basename(src_file_path)
    candidate = os.path.join(dup_dir, f"{dest_file_base}__{src_base}")
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(candidate)
    n = 2
    while True:
        numbered = f"{stem}__{n}{ext}"
        if not os.path.exists(numbered):
            return numbered
        n += 1


def _quarantine_dest_path(
    collisions_dir: str,
    dest_year: str,
    dest_file_base: str,
    src_file_path: str,
) -> str:
    """Build a unique quarantine path for a colliding source file."""
    q_dir = os.path.join(collisions_dir, dest_year)
    src_base = os.path.basename(src_file_path)
    candidate = os.path.join(q_dir, f"{dest_file_base}__{src_base}")
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(candidate)
    n = 2
    while True:
        numbered = f"{stem}__{n}{ext}"
        if not os.path.exists(numbered):
            return numbered
        n += 1


def _collision_details(
    src_path: str,
    dest_path: str,
    src_metadata: Optional[dict] = None,
    dest_metadata: Optional[dict] = None,
    *,
    include_sha256: bool = True,
) -> dict:
    """Build diagnostic fields for collision log rows."""
    details: dict = {}
    src_size = _file_size(src_path)
    dest_size = _file_size(dest_path)
    if src_size is not None:
        details["src_size"] = src_size
    if dest_size is not None:
        details["dest_size"] = dest_size
    if include_sha256:
        src_sha = _file_sha256(src_path)
        dest_sha = _file_sha256(dest_path)
        if src_sha:
            details["src_sha256"] = src_sha
        if dest_sha:
            details["dest_sha256"] = dest_sha
    if src_metadata:
        details["src_metadata_hash"] = src_metadata.get("MetadataHash")
        details["src_capture_dt_utc"] = src_metadata.get("CaptureDtUtc")
        details["src_capture_date_source"] = src_metadata.get("CaptureDateSource")
    if dest_metadata:
        details["dest_metadata_hash"] = dest_metadata.get("MetadataHash")
        details["dest_capture_dt_utc"] = dest_metadata.get("CaptureDtUtc")
    return details


def _safe_transfer(
    src_path: str,
    dest_path: str,
    *,
    is_move: bool,
    verify: bool = True,
) -> None:
    """Copy or move ``src_path`` to ``dest_path`` with optional verify and atomic replace."""
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_path) and _files_are_identical(src_path, dest_path):
        if is_move and not os.path.samefile(src_path, dest_path):
            os.remove(src_path)
        return

    tmp_dest = dest_path + ".picutils.part"
    if os.path.exists(tmp_dest):
        os.remove(tmp_dest)
    shutil.copy2(src_path, tmp_dest)
    if verify and not _files_are_identical(src_path, tmp_dest):
        os.remove(tmp_dest)
        raise OSError(f"transfer verification failed: src={src_path} dest={dest_path}")
    os.replace(tmp_dest, dest_path)
    if is_move:
        os.remove(src_path)


def _live_photo_companion_path(path: str) -> Optional[str]:
    """Return the HEIC/MOV companion path for Apple Live Photos, if present."""
    dirname = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    ext = ext.lower()
    if ext == ".heic":
        for alt in (".mov", ".MOV"):
            candidate = os.path.join(dirname, stem + alt)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    elif ext == ".mov":
        for alt in (".heic", ".HEIC"):
            candidate = os.path.join(dirname, stem + alt)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None


def _transfer_live_companion(
    src_path: str,
    *,
    metadata_by_path: Dict[str, dict],
    dest_root_dir: str,
    is_move: bool,
    verify: bool,
    processed_srcs: Set[str],
    dry_run: bool,
) -> Optional[str]:
    """Transfer a Live Photo companion that shares the same directory stem."""
    companion = _live_photo_companion_path(src_path)
    if not companion or companion in processed_srcs:
        return None
    if companion not in metadata_by_path:
        return None
    companion_meta = metadata_by_path[companion]
    dest_file_base = companion_meta["DestFileBase"]
    dest_year = dest_file_base[:4]
    dest_path = os.path.join(dest_root_dir, dest_year, dest_file_base)
    if os.path.exists(dest_path):
        return None
    if not dry_run:
        _safe_transfer(companion, dest_path, is_move=is_move, verify=verify)
    processed_srcs.add(companion)
    return companion


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
    progress: bool = True,
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
        not_match_re_compile = re.compile(not_match_re, *not_match_re_flags)
    all_file_paths = []

    for subdir, _dirs, files in _iter_progress(
        os.walk(root_dir), desc="scanning", progress=progress
    ):
        for file in files:
            file_path = os.path.join(subdir, file)
            if not_match_re_compile and not_match_re_compile.search(file_path):
                continue
            if match_re and match_re_compile.search(file_path):
                all_file_paths.append(file_path)
    return all_file_paths


def get_all_media_file_paths(
    root_dir: str,
    *,
    exclude_dirs: Optional[List[str]] = None,
    progress: bool = True,
) -> List[str]:
    """Collect paths under ``root_dir`` for common photo/video extensions.

    Args:
        root_dir: Root directory to scan.
        exclude_dirs: Directory trees to skip (e.g. destination library paths).
        progress: Whether to show a progress bar while walking.

    Returns:
        File paths whose names match ``MEDIA_EXT_RE`` (jpg, mov, heic, etc.).
    """
    exclude_roots = _normalize_exclude_roots(exclude_dirs)
    paths = get_all_file_paths(root_dir, match_re=MEDIA_EXT_RE, progress=progress)
    if not exclude_roots:
        return paths
    return [p for p in paths if not _is_excluded_path(p, exclude_roots)]


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

    Timezone-aware values are converted to UTC then stored naive for consistent
    comparisons (tiered ``min()``) and ``YYYYMMDD_HHMMSS`` destination filenames.

    Args:
        value: Tag value from ExifTool (str, ``Timestamp``, number, etc.).

    Returns:
        Parsed timestamp, or ``None`` if missing/invalid.
    """
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        dt = pd.to_datetime(value, unit="s", utc=True)
    elif isinstance(value, datetime.datetime):
        dt = pd.Timestamp(value)
    elif isinstance(value, pd.Timestamp):
        dt = value
    else:
        s = str(value).strip()
        if not s or s.startswith(_INVALID_DT_PREFIXES):
            return None
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
    known = set(f for tier in _CREATE_DT_TIERS for f in tier)
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


def augment_metadata_for_dest(
    metadata: dict,
    *,
    include_metadata_hash_in_dest: bool = True,
    include_orig_name_in_dest: bool = False,
) -> dict:
    """Derive destination fields from ExifTool metadata using current logic.

    Sets ``Location``, ``MetadataHash``, ``CaptureDtUtc``, and ``DestFileBase``.
    Called on every ``cleaup_media_files`` run so date/naming code changes apply
    without re-running ExifTool (cached tags are enough).

    Destination pattern (timestamp is naive UTC)::
        {YYYYMMDD_HHMMSS}[__{location}][__{orig_base}][__{hash}].{ext}

    Args:
        metadata: ExifTool metadata dict (must include ``SourceFile``).
        include_metadata_hash_in_dest: Append content hash before extension.
        include_orig_name_in_dest: Include original basename in the dest name.

    Returns:
        The same ``metadata`` dict, updated in place.
    """
    file_path = metadata["SourceFile"]
    create_dt = get_create_dt_from_metadata(metadata)
    capture_date_source = "metadata"
    if create_dt is None:
        create_dt = pd.Timestamp(get_file_create_date(file_path))
        capture_date_source = "file_mtime"
    base_name, ext = os.path.basename(file_path).rsplit(".", 1)
    loc = get_location_from_metadata(metadata)
    metadata["Location"] = loc
    metadata_hash = get_hash_from_metadata(metadata)
    metadata["MetadataHash"] = metadata_hash
    metadata["CaptureDtUtc"] = str(create_dt)
    metadata["CaptureDateSource"] = capture_date_source
    metadata["ContentSha256"] = _file_sha256(file_path)
    metadata["DestFileBase"] = (
        f"{create_dt.strftime('%Y%m%d_%H%M%S')}"
        f"{f'__{loc}' if loc else ''}"
        f"{f'__{base_name}' if include_orig_name_in_dest else ''}"
        f"{f'__{metadata_hash}' if include_metadata_hash_in_dest else ''}"
        f".{ext.lower()}"
    )
    return metadata


def augment_metadatas_for_dest(
    metadatas: List[dict],
    *,
    include_metadata_hash_in_dest: bool = True,
    include_orig_name_in_dest: bool = False,
) -> List[dict]:
    """Apply ``augment_metadata_for_dest`` to each metadata dict."""
    for metadata in metadatas:
        augment_metadata_for_dest(
            metadata,
            include_metadata_hash_in_dest=include_metadata_hash_in_dest,
            include_orig_name_in_dest=include_orig_name_in_dest,
        )
    return metadatas


class GetMetaDatasAugmented:
    """Batch-fetch ExifTool metadata (used by ``get_metadatas_mproc`` workers)."""

    def __call__(self, file_paths: List[str]) -> List[dict]:
        """Return raw ExifTool metadata for a batch of files."""
        assert isinstance(file_paths, list)
        with exiftool.ExifTool() as et:
            return et.get_metadata_batch(file_paths)


@memorize(
    local_dir="./.cache",
)
def get_metadatas_mproc(
    file_paths: List[str],
    batch_size: int = 64,
    include_orig_name_in_dest: bool = False,
    dest_logic_version: int = DEST_LOGIC_VERSION,
) -> List[Dict]:
    """Load ExifTool metadata for many files using a process pool.

    Caches **raw** ExifTool tags only. ``cleaup_media_files`` (and
    ``augment_metadatas_for_dest``) apply capture-date and ``DestFileBase`` logic
    on each run so code changes do not require ``refresh_metacache``.

    ``dest_logic_version`` is part of the cache key; bump ``DEST_LOGIC_VERSION``
    when raw-cache semantics change.

    Note:
        ``batch_size`` argument is overridden to 32 in the function body.

    Args:
        file_paths: All media paths to process.
        batch_size: Ignored; kept for API compatibility (actual batch size is 32).
        include_orig_name_in_dest: Unused here; kept for API compatibility.
        dest_logic_version: Cache-key version for invalidation.

    Returns:
        Flat list of raw ExifTool metadata dicts.
    """
    _ = (include_orig_name_in_dest, dest_logic_version)
    batch_size = 32
    file_path_batches = list(get_batches(file_paths, batch_size))

    with mproc.Pool(mproc.cpu_count()) as pool:
        matadata_batches = list(
            _iter_progress(
                pool.imap(GetMetaDatasAugmented(), file_path_batches),
                total=len(file_path_batches),
                desc="metadata batches",
            )
        )
        metadatas = [m for mb in matadata_batches for m in mb]
    return metadatas


def _make_session(*, dry_run: bool, move_or_copy: str) -> dict:
    """Build session fields attached to every cleanup log row."""
    return {
        "run_id": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + f"-{uuid.uuid4().hex[:8]}",
        "dry_run": dry_run,
        "move_or_copy": move_or_copy,
    }


def _compute_preflight_counts(
    dest_file_src_files: dict,
    dest_root_dir: str,
    metadata_by_path: Dict[str, dict],
) -> dict:
    """Simulate sequential processing to estimate outcome counts."""
    counts = defaultdict(int)
    for dest_file_base, src_file_paths in sorted(dest_file_src_files.items()):
        dest_year = dest_file_base[:4]
        dest_file_path = os.path.join(dest_root_dir, dest_year, dest_file_base)
        dest_exists = os.path.exists(dest_file_path)
        for src_path in sorted(src_file_paths):
            if not os.path.exists(src_path):
                counts["src_missing"] += 1
                continue
            src_meta = metadata_by_path.get(src_path, {})
            if src_meta.get("CaptureDateSource") == "file_mtime":
                counts["mtime_fallback"] += 1
            if dest_exists:
                if _files_are_identical(src_path, dest_file_path):
                    counts["identical_at_dest"] += 1
                else:
                    counts["collision"] += 1
            else:
                counts["transfer"] += 1
                dest_exists = True
    return dict(counts)


def _print_preflight_summary(counts: dict, *, dry_run: bool) -> None:
    """Print pre-flight outcome estimates."""
    print("Pre-flight summary (sequential simulation):")
    for key in (
        "transfer",
        "identical_at_dest",
        "collision",
        "src_missing",
        "mtime_fallback",
    ):
        if counts.get(key):
            print(f"  {key}: {counts[key]:,.0f}")
    if dry_run:
        print("  (dry_run=True — no files will be moved or copied)")


def _process_one_source(
    *,
    src_file_path: str,
    dest_file_path: str,
    dest_year: str,
    dest_file_base: str,
    sibling_srcs: List[str],
    src_metadata: dict,
    dest_metadata: Optional[dict],
    metadata_by_path: Dict[str, dict],
    dest_root_dir: str,
    is_move: bool,
    dry_run: bool,
    verify_transfers: bool,
    identical_duplicates_dir: Optional[str],
    collisions_dir: Optional[str],
    log_file,
    session: dict,
    verbose_collisions: bool,
    processed_srcs: Set[str],
    counters: dict,
) -> dict:
    """Handle one source file against its destination path."""
    row = dict(
        dest=dest_file_path,
        src=src_file_path,
        other_srcs=sibling_srcs,
        dest_file_base=dest_file_base,
        src_metadata_hash=src_metadata.get("MetadataHash"),
        src_capture_dt_utc=src_metadata.get("CaptureDtUtc"),
        src_capture_date_source=src_metadata.get("CaptureDateSource"),
        src_content_sha256=src_metadata.get("ContentSha256"),
    )
    if src_metadata.get("CaptureDateSource") == "file_mtime":
        row["warning"] = "NO_CAPTURE_DATE_USED_MTIME"

    if not os.path.exists(src_file_path):
        row["error"] = "SRC_MISSING"
        counters["src_missing"] += 1
        return _log_event(log_file, row, session)

    if os.path.exists(dest_file_path):
        row.update(
            _collision_details(
                src_file_path,
                dest_file_path,
                src_metadata,
                dest_metadata,
            )
        )
        if _files_are_identical(src_file_path, dest_file_path):
            if is_move and identical_duplicates_dir:
                dup_path = _identical_duplicate_dest_path(
                    identical_duplicates_dir,
                    dest_year,
                    dest_file_base,
                    src_file_path,
                )
                row["duplicate_reloc"] = dup_path
                if not dry_run:
                    try:
                        os.makedirs(os.path.dirname(dup_path), exist_ok=True)
                        _safe_transfer(
                            src_file_path,
                            dup_path,
                            is_move=True,
                            verify=verify_transfers,
                        )
                        counters["identical_reloc"] += 1
                        row["error"] = "IDENTICAL_MOVED_TO_DUPLICATES"
                        processed_srcs.add(src_file_path)
                    except Exception as e:
                        counters["unexpected"] += 1
                        row["error"] = f"IDENTICAL_RELOC_FAILED: {e}"
                        print(
                            f"unable to relocate identical duplicate "
                            f"src={src_file_path}: {e}"
                        )
                else:
                    row["error"] = "IDENTICAL_WOULD_MOVE_TO_DUPLICATES"
            else:
                counters["identical_dest"] += 1
                row["error"] = "IDENTICAL_DESTINATION_EXISTS"
        else:
            if is_move and collisions_dir and not dry_run:
                q_path = _quarantine_dest_path(
                    collisions_dir,
                    dest_year,
                    dest_file_base,
                    src_file_path,
                )
                row["quarantine_reloc"] = q_path
                try:
                    os.makedirs(os.path.dirname(q_path), exist_ok=True)
                    _safe_transfer(
                        src_file_path,
                        q_path,
                        is_move=True,
                        verify=verify_transfers,
                    )
                    counters["quarantine"] += 1
                    row["error"] = "COLLISION_MOVED_TO_QUARANTINE"
                    processed_srcs.add(src_file_path)
                except Exception as e:
                    counters["unexpected"] += 1
                    counters["dest_exists"] += 1
                    row["error"] = f"QUARANTINE_FAILED: {e}"
                    print(f"unable to quarantine src={src_file_path}: {e}")
            elif is_move and collisions_dir and dry_run:
                counters["dest_exists"] += 1
                row["error"] = "COLLISION_WOULD_MOVE_TO_QUARANTINE"
            else:
                counters["dest_exists"] += 1
                row["error"] = "DESTINATION_EXISTS"
        row = _log_event(log_file, row, session)
        if verbose_collisions:
            _print_destination_exists(row)
        return row

    error = ""
    if not dry_run:
        try:
            _safe_transfer(
                src_file_path,
                dest_file_path,
                is_move=is_move,
                verify=verify_transfers,
            )
            counters["success"] += 1
            processed_srcs.add(src_file_path)
            companion = _transfer_live_companion(
                src_file_path,
                metadata_by_path=metadata_by_path,
                dest_root_dir=dest_root_dir,
                is_move=is_move,
                verify=verify_transfers,
                processed_srcs=processed_srcs,
                dry_run=False,
            )
            if companion:
                row["linked_companion"] = companion
        except Exception as e:
            counters["unexpected"] += 1
            error = str(e)
            print(f"unable to transfer src={src_file_path}: {e}")
    else:
        companion = _live_photo_companion_path(src_file_path)
        if (
            companion
            and companion in metadata_by_path
            and companion not in processed_srcs
            and not os.path.exists(
                os.path.join(
                    dest_root_dir,
                    metadata_by_path[companion]["DestFileBase"][:4],
                    metadata_by_path[companion]["DestFileBase"],
                )
            )
        ):
            row["linked_companion"] = companion
    row["error"] = error
    return _log_event(log_file, row, session)


def cleaup_media_files(
    src_root_dir: str,
    dest_root_dir: str,
    dry_run: bool = True,
    refresh_metacache: bool = False,
    move_or_copy: str = "move",
    log_file_path: str = None,
    verbose_collisions: bool = False,
    include_metadata_hash_in_dest: bool = True,
    include_orig_name_in_dest: bool = False,
    identical_duplicates_dir: Optional[str] = None,
    collisions_dir: Optional[str] = None,
    exclude_dirs: Optional[List[str]] = None,
    progress: bool = True,
    verify_transfers: bool = True,
) -> List[dict]:
    """Organize media into year folders with canonical capture-based filenames.

    Scans ``src_root_dir``, computes destination names from metadata (capture time,
    optional location/hash), groups by ``DestFileBase``, and moves or copies files
    to their destinations. Every source in a duplicate group is processed in one run.

    Layout: ``{dest_root_dir}/{YYYY}/{YYYYMMDD_HHMMSS}__...ext``

    Args:
        src_root_dir: Tree of source photos/videos.
        dest_root_dir: Library root; year subdirs are created as needed.
        dry_run: If True, do not move/copy or create dirs (still writes log rows).
        refresh_metacache: If True, re-run ExifTool and replace the on-disk cache
            (use when files are new/updated; not needed for date/naming logic changes).
        move_or_copy: ``"move"`` (``shutil.move``) or ``"copy"`` (``shutil.copy2``).
        include_metadata_hash_in_dest: Append content hash to destination filenames.
        include_orig_name_in_dest: Include original basename in destination filenames.
        identical_duplicates_dir: If set and ``move_or_copy`` is ``"move"``, when the
            destination already exists and is byte-identical to the source, move the
            source under this directory (year subfolders) for later deletion. In
            ``copy`` mode (or move without this path), byte-identical collisions are
            still detected and logged as ``IDENTICAL_DESTINATION_EXISTS`` without
            copying or relocating.
        collisions_dir: If set and ``move_or_copy`` is ``"move"``, when the destination
            exists with different content, move the source here for manual review.
        exclude_dirs: Extra directory trees to exclude from the source scan. ``dest_root_dir``,
            ``identical_duplicates_dir``, and ``collisions_dir`` are always excluded.
        progress: Show progress bars during scan and transfer.
        verify_transfers: After copy/move, verify byte identity before removing sources.
        log_file_path: Log file path; defaults to ``{dest_root_dir}/log.txt``.
            All events (moves, collisions, dry-run skips) are appended with
            a UTC ``logged_at`` timestamp and session ``run_id``.
        verbose_collisions: If True, print each collision src/dest to stdout.

    Returns:
        List of per-file result dicts (one row per source file). Includes ``run_id``,
        ``dry_run``, ``move_or_copy``, ``error``, and optional collision diagnostics.
        Every row is appended to the cleanup log file (including dry runs).

    Raises:
        AssertionError: If ``move_or_copy`` is not ``"move"`` or ``"copy"``.
    """
    assert move_or_copy in [
        "move",
        "copy",
    ], f"move_or_copy='{move_or_copy}' must be 'move' or 'copy"
    is_move = move_or_copy == "move"
    session = _make_session(dry_run=dry_run, move_or_copy=move_or_copy)
    if identical_duplicates_dir and not is_move:
        print(
            "NOTE: identical_duplicates_dir only relocates sources in move mode; "
            "copy mode will log IDENTICAL_DESTINATION_EXISTS without copying."
        )
    if collisions_dir and not is_move:
        print(
            "NOTE: collisions_dir only quarantines sources in move mode; "
            "copy mode will log DESTINATION_EXISTS without relocating."
        )

    auto_exclude = [
        dest_root_dir,
        identical_duplicates_dir,
        collisions_dir,
    ]
    exclude_roots = _normalize_exclude_roots(
        (exclude_dirs or []) + [p for p in auto_exclude if p]
    )

    print("Getting all source media files...")
    file_paths = sorted(
        get_all_media_file_paths(
            src_root_dir,
            exclude_dirs=exclude_roots,
            progress=progress,
        )
    )
    print(f"Retrieved {len(file_paths):,.0f} media files from {src_root_dir}")
    if exclude_roots:
        print(f"Excluded paths under: {', '.join(exclude_roots)}")

    print("Getting metadatas for all media files...")
    metadatas = get_metadatas_mproc(  # pylint: disable=unexpected-keyword-arg
        file_paths=file_paths,
        __force_refresh=refresh_metacache,
        dest_logic_version=DEST_LOGIC_VERSION,
    )
    print(f"Retrieved {len(metadatas):,.0f} metadatas.")
    print("Computing destination names from metadata (current date/location logic)...")
    augment_metadatas_for_dest(
        metadatas,
        include_metadata_hash_in_dest=include_metadata_hash_in_dest,
        include_orig_name_in_dest=include_orig_name_in_dest,
    )
    metadata_by_path = {m["SourceFile"]: m for m in metadatas}

    dest_file_src_files = defaultdict(list)
    for metadata in metadatas:
        dest_file_src_files[metadata["DestFileBase"]].append(metadata["SourceFile"])
    duplicate_groups = [
        (dest, srcs) for dest, srcs in dest_file_src_files.items() if len(srcs) > 1
    ]
    if duplicate_groups:
        print(
            f"WARNING: found {len(duplicate_groups):,.0f} duplicate source groups "
            f"({sum(len(s) for _, s in duplicate_groups):,.0f} files)."
        )

    preflight = _compute_preflight_counts(
        dest_file_src_files, dest_root_dir, metadata_by_path
    )
    _print_preflight_summary(preflight, dry_run=dry_run)

    dest_years = sorted(
        set(dest_file_base[:4] for dest_file_base in dest_file_src_files)
    )
    for dest_year in dest_years:
        dest_dir = os.path.join(dest_root_dir, dest_year)
        if not os.path.exists(dest_dir):
            print(f"WARNING: will create destination directory {dest_dir}")
            if not dry_run:
                os.makedirs(dest_dir)

    if log_file_path is None:
        log_file_path = os.path.join(dest_root_dir, "log.txt")
    os.makedirs(dest_root_dir, exist_ok=True)
    if identical_duplicates_dir and not dry_run:
        os.makedirs(identical_duplicates_dir, exist_ok=True)
    if collisions_dir and not dry_run:
        os.makedirs(collisions_dir, exist_ok=True)

    print(f"Transferring files. Logging all events to {log_file_path} ...")
    rows: List[dict] = []
    counters = defaultdict(int)
    processed_srcs: Set[str] = set()

    with open(log_file_path, "a") as log_file:
        for dest_file_base, src_file_paths in _iter_progress(
            sorted(dest_file_src_files.items()),
            progress=progress,
            desc="transfer",
        ):
            dest_year = dest_file_base[:4]
            dest_file_path = os.path.join(dest_root_dir, dest_year, dest_file_base)
            dest_metadata = metadata_by_path.get(dest_file_path)
            sorted_srcs = sorted(src_file_paths)

            for idx, src_file_path in enumerate(sorted_srcs):
                if src_file_path in processed_srcs:
                    continue
                sibling_srcs = sorted_srcs[:idx] + sorted_srcs[idx + 1 :]
                row = _process_one_source(
                    src_file_path=src_file_path,
                    dest_file_path=dest_file_path,
                    dest_year=dest_year,
                    dest_file_base=dest_file_base,
                    sibling_srcs=sibling_srcs,
                    src_metadata=metadata_by_path[src_file_path],
                    dest_metadata=dest_metadata,
                    metadata_by_path=metadata_by_path,
                    dest_root_dir=dest_root_dir,
                    is_move=is_move,
                    dry_run=dry_run,
                    verify_transfers=verify_transfers,
                    identical_duplicates_dir=identical_duplicates_dir,
                    collisions_dir=collisions_dir if is_move else None,
                    log_file=log_file,
                    session=session,
                    verbose_collisions=verbose_collisions,
                    processed_srcs=processed_srcs,
                    counters=counters,
                )
                rows.append(row)

        action = "moved/copied" if not dry_run else "would transfer"
        if counters["success"]:
            print(f"Successfully {action} {counters['success']:,.0f} files!")
        elif not counters["identical_reloc"] and not counters["quarantine"]:
            print(f"NO FILES WERE {action.upper()}!")
        if counters["identical_reloc"]:
            print(
                f"Moved {counters['identical_reloc']:,.0f} byte-identical duplicates to "
                f"{identical_duplicates_dir}"
            )
        if counters["identical_dest"]:
            print(
                f"Skipped {counters['identical_dest']:,.0f} byte-identical files already at "
                f"destination (error=IDENTICAL_DESTINATION_EXISTS)."
            )
        if counters["quarantine"]:
            print(
                f"Quarantined {counters['quarantine']:,.0f} colliding sources to "
                f"{collisions_dir}"
            )
        if counters["dest_exists"]:
            print(
                f"WARNING: {counters['dest_exists']:,.0f} destinations already existed with "
                f"different content! See {log_file_path} (error=DESTINATION_EXISTS)."
            )
        if counters["mtime_fallback"]:
            print(
                f"NOTE: {counters['mtime_fallback']:,.0f} files used file mtime for "
                f"capture date (warning=NO_CAPTURE_DATE_USED_MTIME)."
            )
        if counters["unexpected"]:
            print(f"ERROR: Encountered {counters['unexpected']:,.0f} unexpected errors!")
    return rows
