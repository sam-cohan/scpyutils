"""Utilities for organizing photos and videos using ExifTool metadata.

This module scans media files, extracts capture time and location, and builds
canonical destination filenames for deduplication and library cleanup.

Dependencies:
    ExifTool must be installed and on PATH (``brew install exiftool``).
    Python: ``pyexiftool``, ``pandas``, ``exifread``, ``reverse_geocode``, ``send2trash``, etc.
    Optional HEIC support: ``libheif``, ``pyheif``, ``piexif``.
    Optional GPS timezone: ``timezonefinder`` (infer IANA zone from coordinates).

Capture-date extraction uses a **tiered minimum** strategy (see
``get_create_dt_from_metadata``): among tags that mean "when was this recorded",
the earliest valid timestamp usually reflects the true capture time because
re-exports and re-wraps tend to push container dates *forward*, not backward.
``ModifyDate`` and similar edit timestamps are never considered.

**Timezone policy:** Tags with an explicit offset (e.g. Apple ``QuickTime:CreationDate``,
``EXIF:OffsetTimeOriginal``) drive **local wall time** for ``{YYYY}/{MM}`` folders and
ISO filenames. Naive EXIF times without offset are treated as local at the GPS
location when coordinates exist (requires ``timezonefinder``). If the photo
timezone still cannot be determined, capture folders and ISO filenames fall
back to **UTC** (``+0000`` suffix). ``CaptureDtUtc`` is stored for sorting and logs.

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
import xxhash

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
_CAPTURE_DT_OFFSET_TAGS = {
    "EXIF:DateTimeOriginal": "EXIF:OffsetTimeOriginal",
    "EXIF:CreateDate": "EXIF:OffsetTime",
    "EXIF:ModifyDate": "EXIF:OffsetTime",
}
_CAPTURE_DT_KEY_SUFFIXES = ("DateTimeOriginal", "CreationDate")

_INVALID_DT_PREFIXES = ("0000:", "0001:", "1970:01:01")

# Bump when raw-cache shape or ExifTool batch behavior changes materially.
DEST_LOGIC_VERSION = 1

# Content fingerprinting: files larger than this use head+tail sampling (see below).
DEFAULT_CONTENT_HASH_FULL_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_CONTENT_HASH_SAMPLE_BYTES = 4 * 1024 * 1024
CONTENT_HASH_MODE_AUTO = "auto"
CONTENT_HASH_MODE_FULL = "full"
CONTENT_HASH_MODE_SAMPLE = "sample"
CONTENT_HASH_ALGORITHM = "xxh64"
_CONTENT_FP_CACHE_VERSION = 1
_CONTENT_FP_CACHE_DIR = "./.cache"
_CONTENT_FP_CACHE_FILENAME = "picutils_content_fp.json"

# How ``DestFileBase`` is built from capture metadata and content hash.
# Path layout: ``{dest_root}/{YYYY}/{MM}/{DestFileBase}``.
# Default ``iso_geo_hash``: ``{YYYYMMDD}_{HHMMSS}{±HHMM}__{geo}__{hash16}.{ext}`` (offset always present)
# ``content_only``: ``{hash_16}.{ext}`` (hash-only filename; same bytes → one name)
# ``capture_content``: legacy ``{YYYYMMDD}_{HHMMSS}[__{loc}][__{orig}][__{hash_16}].{ext}``
DEST_NAME_MODE_ISO_GEO_HASH = "iso_geo_hash"
DEST_NAME_MODE_CONTENT_ONLY = "content_only"
DEST_NAME_MODE_CAPTURE_CONTENT = "capture_content"
# Separator between capture / geo / hash segments in destination basenames.
DEST_FILENAME_FIELD_SEP = "__"

_LIVE_PHOTO_EXTS = {".heic", ".mov"}


def _logged_at_iso() -> str:
    """Return current local time (machine timezone) as an ISO-8601 string for log rows."""
    return datetime.datetime.now().astimezone().isoformat()


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
    """Wrap an iterable with tqdm (auto-detects notebook vs CLI vs plain)."""
    if not progress:
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc, total=total)


def _log_event(log_file, row: dict, session: Optional[dict] = None) -> dict:
    """Append a timestamped JSON event to the cleanup log file."""
    row = {**row, "logged_at": _logged_at_iso()}
    if session:
        row = {**session, **row}
    log_file.write(f"{json.dumps(row)}\n")
    return row


def _short_path_label(path: str) -> str:
    """Return a compact path label (last two components, or basename)."""
    if not path:
        return ""
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _print_destination_exists(row: dict) -> None:
    """Print a compact collision summary (at most two lines)."""
    label = row.get("error", "DESTINATION_EXISTS")
    src = _short_path_label(row.get("src", ""))
    dest = _short_path_label(row.get("dest", ""))
    tags: List[str] = []
    if row.get("quarantine_reloc"):
        tags.append(f"q→{_short_path_label(row['quarantine_reloc'])}")
    n_other = len(row.get("other_srcs") or [])
    if n_other:
        tags.append(f"+{n_other} other")
    line1 = f"{label}: {src} → {dest}"
    if tags:
        line1 += f" [{', '.join(tags)}]"
    print(line1)

    details: List[str] = []
    if "src_size" in row and "dest_size" in row:
        details.append(f"size {row['src_size']}/{row['dest_size']}")
    src_hash = row.get("src_content_hash")
    dest_hash = row.get("dest_content_hash")
    if src_hash and dest_hash:
        details.append(f"hash {src_hash}/{dest_hash}")
    elif src_hash:
        details.append(f"hash {src_hash}")
    if details:
        print("  " + " | ".join(details))


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


def _capture_iso_for_filename(
    local_dt: pd.Timestamp, tz_suffix: str = ""
) -> str:
    """Return filename-safe capture time: ``YYYYMMDD_HHMMSS±HHMM`` (offset always present)."""
    return local_dt.strftime("%Y%m%d_%H%M%S") + (tz_suffix or "+0000")


def _aware_to_local_naive(dt: pd.Timestamp) -> pd.Timestamp:
    """Drop timezone from an aware timestamp, keeping local wall-clock components."""
    return pd.Timestamp(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        microsecond=dt.microsecond,
    )


def _format_tz_suffix_from_aware(dt: pd.Timestamp) -> str:
    """Return ``-0400`` / ``+0530`` suffix for filenames, or ``""`` if naive."""
    if dt.tzinfo is None:
        return ""
    return dt.strftime("%z")


def _combine_naive_dt_with_offset_tag(
    naive_value, offset_value
) -> Optional[str]:
    """Join naive ExifTool date with a separate ``OffsetTime*`` tag if present."""
    if naive_value is None or offset_value is None:
        return None
    naive_s = str(naive_value).strip()
    offset_s = str(offset_value).strip()
    if not naive_s or not offset_s or naive_s.startswith(_INVALID_DT_PREFIXES):
        return None
    if len(naive_s) > 4 and naive_s[4] == ":":
        naive_s = naive_s.replace(":", "-", 2)
    naive_s = naive_s.replace(" ", "T", 1)
    if offset_s[0] in "+-" and ":" in offset_s:
        return f"{naive_s}{offset_s}"
    if offset_s[0] in "+-" and len(offset_s) >= 5:
        return f"{naive_s}{offset_s[:3]}:{offset_s[3:5]}"
    return None


_timezone_finder_instance = None


def _get_timezone_finder():
    """Return a cached TimezoneFinder singleton (expensive to construct)."""
    global _timezone_finder_instance
    if _timezone_finder_instance is None:
        try:
            from timezonefinder import TimezoneFinder
            _timezone_finder_instance = TimezoneFinder()
        except ImportError:
            return None
    return _timezone_finder_instance


def _apply_gps_timezone_to_parts(
    parts: dict, metadata: dict
) -> dict:
    """If no offset in metadata, infer IANA zone from GPS for naive local times."""
    if parts.get("tz_suffix"):
        return parts
    lat_lng = get_lat_lng_from_metadata(metadata)
    if not lat_lng:
        return parts
    tf = _get_timezone_finder()
    if tf is None:
        return parts
    try:
        import zoneinfo
    except ImportError:
        return parts
    lat, lng = lat_lng
    tz_name = tf.timezone_at(lng=float(lng), lat=float(lat))
    if not tz_name:
        return parts
    local = parts["local"]
    try:
        aware = local.tz_localize(zoneinfo.ZoneInfo(tz_name))
    except Exception:
        return parts
    utc = aware.tz_convert("UTC").tz_localize(None)
    return {
        "utc": utc,
        "local": _aware_to_local_naive(aware),
        "tz_suffix": _format_tz_suffix_from_aware(aware),
        "tz_source": "gps",
    }


def _apply_utc_fallback_when_tz_unknown(parts: dict) -> dict:
    """Use UTC wall time for dest naming when no tag offset or GPS zone applies."""
    if parts.get("tz_suffix") or parts.get("tz_source"):
        return parts
    utc = parts["utc"]
    return {
        "utc": utc,
        "local": utc,
        "tz_suffix": "+0000",
        "tz_source": "utc",
    }


def _parse_metadata_datetime_parts(
    value, metadata: Optional[dict] = None, field: Optional[str] = None
) -> Optional[dict]:
    """Parse ExifTool datetime into UTC, local wall time, and filename offset suffix.

    Returns:
        ``{"utc", "local", "tz_suffix", "tz_source"}`` or ``None``.
        ``tz_source`` is ``"tag"``, ``"gps"``, ``"utc"`` (fallback), or ``""``.
    """
    if value is None or value == "":
        return None

    tz_source = ""
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        aware = pd.to_datetime(value, unit="s", utc=True)
        utc = aware.tz_localize(None)
        parts = {"utc": utc, "local": utc, "tz_suffix": "", "tz_source": ""}
        return _apply_utc_fallback_when_tz_unknown(parts)

    if isinstance(value, datetime.datetime):
        dt = pd.Timestamp(value)
    elif isinstance(value, pd.Timestamp):
        dt = value
    else:
        combined = None
        if metadata and field and field in _CAPTURE_DT_OFFSET_TAGS:
            combined = _combine_naive_dt_with_offset_tag(
                value, metadata.get(_CAPTURE_DT_OFFSET_TAGS[field])
            )
        s = (combined or str(value)).strip()
        if not s or s.startswith(_INVALID_DT_PREFIXES):
            return None
        if len(s) > 4 and s[4] == ":":
            s = s.replace(":", "-", 2)
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None

    if dt.tzinfo is not None:
        utc = dt.tz_convert("UTC").tz_localize(None)
        local = _aware_to_local_naive(dt)
        tz_suffix = _format_tz_suffix_from_aware(dt)
        tz_source = "tag"
    else:
        local = pd.Timestamp(dt)
        utc = local
        tz_suffix = ""
    if utc < MIN_DT or local < MIN_DT:
        return None
    parts = {
        "utc": utc,
        "local": local,
        "tz_suffix": tz_suffix,
        "tz_source": tz_source,
    }
    if metadata is not None:
        parts = _apply_gps_timezone_to_parts(parts, metadata)
    return _apply_utc_fallback_when_tz_unknown(parts)


def _dest_year_month(metadata: dict) -> Tuple[str, str]:
    """Return ``(YYYY, MM)`` capture folder names from augmented metadata."""
    year = metadata.get("DestYear")
    month = metadata.get("DestMonth")
    if year and month:
        return str(year), str(month).zfill(2)
    local = metadata.get("CaptureDtLocal")
    if local:
        ts = pd.Timestamp(local)
        return ts.strftime("%Y"), ts.strftime("%m")
    capture = metadata.get("CaptureDtUtc")
    if capture:
        ts = pd.Timestamp(capture)
        return ts.strftime("%Y"), ts.strftime("%m")
    return "unknown", "00"


def _dest_group_key(metadata: dict) -> Tuple[str, str, str]:
    """Return ``(year, month, DestFileBase)`` for grouping sources."""
    year, month = _dest_year_month(metadata)
    return year, month, metadata["DestFileBase"]


def _dest_file_path(dest_root_dir: str, metadata: dict) -> str:
    """Return full destination path ``{root}/{YYYY}/{MM}/{DestFileBase}``."""
    year, month = _dest_year_month(metadata)
    return os.path.join(dest_root_dir, year, month, metadata["DestFileBase"])


def _file_size(path: str) -> Optional[int]:
    """Return file size in bytes, or None if unreadable."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _file_full_hash(path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Return hex xxHash64 of the raw file on disk, or None if unreadable."""
    try:
        digest = xxhash.xxh64()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _file_sample_fingerprint(
    path: str,
    *,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> Optional[str]:
    """Return xxHash64 of file size plus head/tail bytes (at most ``2 * sample_bytes``)."""
    try:
        size = os.path.getsize(path)
        digest = xxhash.xxh64()
        digest.update(str(size).encode("ascii"))
        with open(path, "rb") as handle:
            digest.update(handle.read(sample_bytes))
            if size > sample_bytes:
                handle.seek(max(sample_bytes, size - sample_bytes))
                digest.update(handle.read(sample_bytes))
        return digest.hexdigest()
    except OSError:
        return None


def _file_content_fingerprint(
    path: str,
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> Tuple[Optional[str], str]:
    """Return ``(hex_digest, method)`` where ``method`` is ``full`` or ``sample``."""
    if content_hash_mode not in (
        CONTENT_HASH_MODE_AUTO,
        CONTENT_HASH_MODE_FULL,
        CONTENT_HASH_MODE_SAMPLE,
    ):
        raise ValueError(
            f"content_hash_mode={content_hash_mode!r} must be "
            f"{CONTENT_HASH_MODE_AUTO!r}, {CONTENT_HASH_MODE_FULL!r}, or "
            f"{CONTENT_HASH_MODE_SAMPLE!r}"
        )
    if content_hash_mode == CONTENT_HASH_MODE_FULL:
        return _file_full_hash(path), CONTENT_HASH_MODE_FULL
    if content_hash_mode == CONTENT_HASH_MODE_SAMPLE:
        return _file_sample_fingerprint(path, sample_bytes=sample_bytes), CONTENT_HASH_MODE_SAMPLE
    size = _file_size(path)
    if size is None:
        return None, CONTENT_HASH_MODE_FULL
    if size <= full_max_bytes:
        return _file_full_hash(path), CONTENT_HASH_MODE_FULL
    return (
        _file_sample_fingerprint(path, sample_bytes=sample_bytes),
        CONTENT_HASH_MODE_SAMPLE,
    )


_content_fp_cache: Optional[Dict[str, dict]] = None
_content_fp_cache_dirty = False


def _content_fp_cache_path() -> str:
    return os.path.join(_CONTENT_FP_CACHE_DIR, _CONTENT_FP_CACHE_FILENAME)


def _load_content_fp_cache() -> Dict[str, dict]:
    global _content_fp_cache
    if _content_fp_cache is not None:
        return _content_fp_cache
    path = _content_fp_cache_path()
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("version") == _CONTENT_FP_CACHE_VERSION:
                _content_fp_cache = payload.get("entries", {})
                return _content_fp_cache
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    _content_fp_cache = {}
    return _content_fp_cache


def _persist_content_fp_cache() -> None:
    global _content_fp_cache_dirty
    if not _content_fp_cache_dirty or _content_fp_cache is None:
        return
    os.makedirs(_CONTENT_FP_CACHE_DIR, exist_ok=True)
    payload = {
        "version": _CONTENT_FP_CACHE_VERSION,
        "entries": _content_fp_cache,
    }
    with open(_content_fp_cache_path(), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    _content_fp_cache_dirty = False


def _get_content_fingerprint(
    path: str,
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
    use_cache: bool = True,
) -> Tuple[Optional[str], str]:
    """Return cached or freshly computed ``(digest, method)`` for ``path``."""
    abs_path = os.path.abspath(path)
    if use_cache:
        try:
            stat = os.stat(abs_path)
        except OSError:
            return _file_content_fingerprint(
                abs_path,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            )
        cache = _load_content_fp_cache()
        cached = cache.get(abs_path)
        if cached and cached.get("size") == stat.st_size and cached.get(
            "mtime_ns"
        ) == stat.st_mtime_ns and cached.get("algorithm") == CONTENT_HASH_ALGORITHM and cached.get(
            "mode"
        ) == content_hash_mode and cached.get(
            "full_max_bytes"
        ) == full_max_bytes and cached.get("sample_bytes") == sample_bytes:
            return cached["digest"], cached["method"]

    digest, method = _file_content_fingerprint(
        abs_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    )
    if use_cache and digest is not None:
        stat = os.stat(abs_path)
        cache = _load_content_fp_cache()
        cache[abs_path] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "digest": digest,
            "method": method,
            "algorithm": CONTENT_HASH_ALGORITHM,
            "mode": content_hash_mode,
            "full_max_bytes": full_max_bytes,
            "sample_bytes": sample_bytes,
        }
        global _content_fp_cache_dirty
        _content_fp_cache_dirty = True
    return digest, method


def _files_are_identical(
    path_a: str,
    path_b: str,
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> bool:
    """Return True if two files have the same size and content fingerprint."""
    try:
        if os.path.samefile(path_a, path_b):
            return True
        size_a = os.path.getsize(path_a)
        size_b = os.path.getsize(path_b)
        if size_a != size_b:
            return False
        if content_hash_mode == CONTENT_HASH_MODE_FULL or (
            content_hash_mode == CONTENT_HASH_MODE_AUTO and size_a <= full_max_bytes
        ):
            return filecmp.cmp(path_a, path_b, shallow=False)
        fp_a, _ = _get_content_fingerprint(
            path_a,
            content_hash_mode=CONTENT_HASH_MODE_SAMPLE,
            sample_bytes=sample_bytes,
        )
        fp_b, _ = _get_content_fingerprint(
            path_b,
            content_hash_mode=CONTENT_HASH_MODE_SAMPLE,
            sample_bytes=sample_bytes,
        )
        return fp_a is not None and fp_a == fp_b
    except OSError:
        return False


def _verified_dest_has_src_content(
    src_path: str,
    dest_path: str,
    src_metadata: dict,
    dest_content_hash: Optional[str],
    *,
    verify_transfers: bool,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> bool:
    """Return True only when destination is confirmed to hold the source bytes."""
    if not os.path.exists(dest_path):
        return False
    hash_match = _files_content_identical(
        src_path,
        dest_path,
        src_metadata,
        dest_content_hash,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    )
    if hash_match is True:
        return True
    if hash_match is False:
        return False
    if verify_transfers:
        return _files_are_identical(
            src_path,
            dest_path,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
    return False


def _trash_file(path: str) -> None:
    """Move a file to the system Trash (recoverable). Never permanently delete sources."""
    try:
        from send2trash import send2trash
    except ImportError as e:
        raise ImportError(
            "send2trash is required to remove sources in move mode "
            "(pip install send2trash)"
        ) from e
    send2trash(os.path.abspath(path))


def _handle_source_after_verify(
    src_path: str,
    *,
    is_move: bool,
    dry_run: bool,
    counters: dict,
    processed_srcs: Set[str],
    reason: str = "identical",
) -> str:
    """Remove source after verify in move mode; keep on disk in copy mode."""
    if is_move:
        return _delete_verified_source(
            src_path,
            dry_run=dry_run,
            counters=counters,
            processed_srcs=processed_srcs,
            reason=reason,
        )
    if dry_run:
        return "SOURCE_WOULD_KEEP_AFTER_COPY"
    processed_srcs.add(src_path)
    counters["source_kept"] += 1
    if reason == "transfer":
        return "TRANSFERRED_SOURCE_KEPT"
    return "IDENTICAL_SOURCE_KEPT"


def _delete_verified_source(
    src_path: str,
    *,
    dry_run: bool,
    counters: dict,
    processed_srcs: Set[str],
    reason: str = "identical",
) -> str:
    """Send source to Trash after the library destination is verified."""
    if dry_run:
        return "SOURCE_WOULD_TRASH_AFTER_VERIFY"
    try:
        _trash_file(src_path)
        processed_srcs.add(src_path)
        counters["source_deleted"] += 1
        if reason == "transfer":
            return "TRANSFERRED_SOURCE_DELETED"
        return "IDENTICAL_SOURCE_DELETED"
    except (OSError, ImportError) as e:
        counters["unexpected"] += 1
        return f"SOURCE_TRASH_FAILED: {e}"


def _quarantine_dest_path(
    collisions_dir: str,
    dest_year: str,
    dest_month: str,
    dest_file_base: str,
    src_file_path: str,
) -> str:
    """Build a unique quarantine path for a colliding source file."""
    q_dir = os.path.join(collisions_dir, dest_year, dest_month)
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


def _metadata_content_hash(metadata: dict) -> Optional[str]:
    """Return stored content hash from metadata (``ContentHash`` or legacy key)."""
    return metadata.get("ContentHash") or metadata.get("ContentSha256")


def _src_content_hash(
    src_metadata: dict,
    src_path: str,
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> Optional[str]:
    """Return cached content hash from metadata, computing it if missing."""
    digest = _metadata_content_hash(src_metadata)
    if digest:
        return digest
    digest, method = _get_content_fingerprint(
        src_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    )
    if digest:
        src_metadata["ContentHash"] = digest
        src_metadata["ContentHashMethod"] = method
    return digest


def _dest_content_fingerprint(
    dest_path: str,
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> Optional[str]:
    """Return content fingerprint for an on-disk destination file."""
    digest, _method = _get_content_fingerprint(
        dest_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    )
    return digest


def _files_content_identical(
    src_path: str,
    dest_path: str,
    src_metadata: dict,
    dest_content_hash: Optional[str],
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> Optional[bool]:
    """Compare content by fingerprint when possible.

    Returns:
        True or False when the fingerprint is decisive, or None to fall back.
    """
    src_size = _file_size(src_path)
    dest_size = _file_size(dest_path)
    if src_size is not None and dest_size is not None and src_size != dest_size:
        return False
    src_hash = _src_content_hash(
        src_metadata,
        src_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    )
    if not src_hash:
        return None
    if dest_content_hash is None and os.path.exists(dest_path):
        dest_content_hash = _dest_content_fingerprint(
            dest_path,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
    if dest_content_hash:
        return src_hash == dest_content_hash
    return None


def _collision_details(
    src_path: str,
    dest_path: str,
    src_metadata: Optional[dict] = None,
    dest_metadata: Optional[dict] = None,
    *,
    include_content_hash: bool = True,
    src_content_hash: Optional[str] = None,
    dest_content_hash: Optional[str] = None,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> dict:
    """Build diagnostic fields for collision log rows."""
    details: dict = {}
    src_size = _file_size(src_path)
    dest_size = _file_size(dest_path)
    if src_size is not None:
        details["src_size"] = src_size
    if dest_size is not None:
        details["dest_size"] = dest_size
    if include_content_hash:
        src_hash = src_content_hash or (
            _metadata_content_hash(src_metadata) if src_metadata else None
        )
        dest_hash = dest_content_hash
        if not src_hash:
            src_hash = _src_content_hash(
                src_metadata or {},
                src_path,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            )
        if not dest_hash:
            dest_hash = _dest_content_fingerprint(
                dest_path,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            )
        if src_hash:
            details["src_content_hash"] = src_hash
        if dest_hash:
            details["dest_content_hash"] = dest_hash
    if src_metadata:
        details["src_metadata_hash"] = src_metadata.get("MetadataHash")
        details["src_capture_dt_utc"] = src_metadata.get("CaptureDtUtc")
        details["src_capture_date_source"] = src_metadata.get("CaptureDateSource")
    if dest_metadata:
        details["dest_metadata_hash"] = dest_metadata.get("MetadataHash")
        details["dest_capture_dt_utc"] = dest_metadata.get("CaptureDtUtc")
    return details


def _same_filesystem(path_a: str, path_b: str) -> bool:
    """Return True when both paths reside on the same mounted filesystem."""
    try:
        return os.stat(path_a).st_dev == os.stat(path_b).st_dev
    except OSError:
        return False


def _safe_transfer(
    src_path: str,
    dest_path: str,
    *,
    verify: bool = True,
    is_move: bool = False,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> str:
    """Transfer ``src_path`` to ``dest_path``.

    When ``is_move`` is True and both paths share the same filesystem,
    ``os.rename`` is used for an instant atomic move (no copy, no verify
    needed, source disappears automatically).

    Otherwise falls back to copy → hash-verify → atomic replace (source
    stays on disk for the caller to handle).

    Returns:
        ``"renamed"`` if an instant rename was used, ``"copied"`` otherwise.
    """
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_path) and _files_are_identical(
        src_path,
        dest_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    ):
        return "copied"

    if is_move and _same_filesystem(src_path, dest_dir):
        os.rename(src_path, dest_path)
        return "renamed"

    tmp_dest = dest_path + ".picutils.part"
    if os.path.exists(tmp_dest):
        os.remove(tmp_dest)
    shutil.copy2(src_path, tmp_dest)
    if verify:
        src_hash, _ = _get_content_fingerprint(
            src_path,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
        dest_hash, _ = _get_content_fingerprint(
            tmp_dest,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
            use_cache=False,
        )
        if src_hash is None or src_hash != dest_hash:
            os.remove(tmp_dest)
            raise OSError(f"transfer verification failed: src={src_path} dest={dest_path}")
    os.replace(tmp_dest, dest_path)
    return "copied"


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
    verify: bool,
    processed_srcs: Set[str],
    counters: dict,
    dry_run: bool,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
    is_move: bool = True,
) -> Optional[str]:
    """Transfer a Live Photo companion that shares the same directory stem."""
    companion = _live_photo_companion_path(src_path)
    if not companion or companion in processed_srcs:
        return None
    if companion not in metadata_by_path:
        return None
    companion_meta = metadata_by_path[companion]
    dest_path = _dest_file_path(dest_root_dir, companion_meta)
    if os.path.exists(dest_path):
        return None
    if not dry_run:
        method = _safe_transfer(
            companion,
            dest_path,
            verify=verify,
            is_move=is_move,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
        if method == "renamed":
            counters["source_deleted"] += 1
        elif verify:
            _handle_source_after_verify(
                companion,
                is_move=is_move,
                dry_run=False,
                counters=counters,
                processed_srcs=processed_srcs,
                reason="transfer",
            )
    processed_srcs.add(companion)
    return companion


def get_hash_from_metadata(metadata: dict) -> str:
    """Build a short fingerprint from stable ExifTool tags (diagnostics only).

    Uses ``HASH_KEYS`` (dimensions, MIME type, select date/resolution fields).
    **Not used for destination filenames** — see ``get_content_hash_for_dest``.

    Args:
        metadata: ExifTool metadata dict (``SourceFile`` plus group-prefixed tags).

    Returns:
        First 16 hex chars of a SHA-1 digest of sorted key/value pairs.
    """
    hash_content = str(sorted([(k, v) for k, v in metadata.items() if k in HASH_KEYS]))
    return hashlib.sha1(hash_content.encode()).hexdigest()[:16]


def get_content_hash_for_dest(
    file_path: str,
    content_hash: Optional[str] = None,
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> Optional[str]:
    """Return the 16-char xxHash64 digest used in destination filenames."""
    if content_hash:
        digest = content_hash
    else:
        digest, _method = _get_content_fingerprint(
            file_path,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
    return digest[:16] if digest else None


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

    Format: ``{country_code}-{city-with-hyphens}`` (e.g. ``CA-Toronto``).
    Empty string if no GPS data.

    Args:
        metadata: ExifTool metadata dict.

    Returns:
        Location token, or ``""``.
    """
    lat_lng = get_lat_lng_from_metadata(metadata)
    if lat_lng:
        res = reverse_geocode.search([lat_lng])[0]
        return f"{res['country_code']}-{res['city'].replace(' ', '-')}"
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


def _parse_metadata_datetime(
    value,
    metadata: Optional[dict] = None,
    field: Optional[str] = None,
) -> Optional[pd.Timestamp]:
    """Parse one ExifTool datetime into naive UTC (see ``_parse_metadata_datetime_parts``)."""
    parts = _parse_metadata_datetime_parts(value, metadata=metadata, field=field)
    return parts["utc"] if parts else None


def _collect_valid_dt_parts(
    metadata: dict, fields: Tuple[str, ...]
) -> List[dict]:
    """Parse capture datetimes with timezone/local parts for fixed tag names."""
    out = []
    for fld in fields:
        parts = _parse_metadata_datetime_parts(
            metadata.get(fld), metadata=metadata, field=fld
        )
        if parts is not None:
            out.append(parts)
    return out


def _collect_dynamic_capture_dt_parts(metadata: dict) -> List[dict]:
    """Like ``_collect_dynamic_capture_dts`` but returns full datetime parts dicts."""
    known = set(f for tier in _CREATE_DT_TIERS for f in tier)
    out = []
    for key, value in metadata.items():
        if key in known or "Modify" in key:
            continue
        if not any(key.endswith(suffix) for suffix in _CAPTURE_DT_KEY_SUFFIXES):
            continue
        parts = _parse_metadata_datetime_parts(
            value, metadata=metadata, field=key
        )
        if parts is not None:
            out.append(parts)
    return out


def get_capture_dt_parts_from_metadata(metadata: dict) -> Optional[dict]:
    """Return capture datetime parts for the tiered-min winning tag.

    Keys: ``utc``, ``local``, ``tz_suffix``, ``tz_source`` (``tag``, ``gps``, ``utc``, or ``""``).
    """
    for tier_idx, fields in enumerate(_CREATE_DT_TIERS):
        parts_list = _collect_valid_dt_parts(metadata, fields)
        if tier_idx == 0:
            parts_list.extend(_collect_dynamic_capture_dt_parts(metadata))
        if parts_list:
            return min(parts_list, key=lambda p: p["utc"])
    print(
        "ERROR: Failed to extract create_dt from metadata; "
        f"tried tiers={_CREATE_DT_TIERS}, SourceFile={metadata.get('SourceFile')}"
    )
    return None


def _collect_valid_dts(
    metadata: dict, fields: Tuple[str, ...]
) -> List[pd.Timestamp]:
    """Parse and collect all valid UTC datetimes for a fixed list of tag names."""
    return [
        p["utc"]
        for p in _collect_valid_dt_parts(metadata, fields)
    ]


def _collect_dynamic_capture_dts(metadata: dict) -> List[pd.Timestamp]:
    """Discover extra capture-date tags; return UTC timestamps only."""
    return [p["utc"] for p in _collect_dynamic_capture_dt_parts(metadata)]


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
        Use ``get_capture_dt_parts_from_metadata`` for local wall time and offset.
    """
    parts = get_capture_dt_parts_from_metadata(metadata)
    return parts["utc"] if parts else None


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


def _build_dest_file_base(
    *,
    local_dt: pd.Timestamp,
    tz_suffix: str = "",
    ext_lower: str,
    loc: str,
    content_hash_for_dest: Optional[str],
    dest_name_mode: str,
    base_name: str,
    include_content_hash_in_dest: bool,
    include_orig_name_in_dest: bool,
) -> str:
    """Build ``DestFileBase`` from capture time, geo slot, and content hash."""
    iso = _capture_iso_for_filename(local_dt, tz_suffix)
    if dest_name_mode == DEST_NAME_MODE_ISO_GEO_HASH:
        parts = [iso, loc]
        if include_content_hash_in_dest and content_hash_for_dest:
            parts.append(content_hash_for_dest)
        return f"{DEST_FILENAME_FIELD_SEP.join(parts)}.{ext_lower}"
    if dest_name_mode == DEST_NAME_MODE_CONTENT_ONLY:
        if content_hash_for_dest:
            return f"{content_hash_for_dest}.{ext_lower}"
        return f"{iso}.{ext_lower}"
    if dest_name_mode == DEST_NAME_MODE_CAPTURE_CONTENT:
        parts = [_capture_iso_for_filename(local_dt, tz_suffix)]
        if loc:
            parts.append(loc)
        if include_orig_name_in_dest:
            parts.append(base_name)
        if include_content_hash_in_dest and content_hash_for_dest:
            parts.append(content_hash_for_dest)
        return f"{DEST_FILENAME_FIELD_SEP.join(parts)}.{ext_lower}"
    raise ValueError(
        f"dest_name_mode={dest_name_mode!r} must be "
        f"{DEST_NAME_MODE_ISO_GEO_HASH!r}, {DEST_NAME_MODE_CONTENT_ONLY!r}, or "
        f"{DEST_NAME_MODE_CAPTURE_CONTENT!r}"
    )


def augment_metadata_for_dest(
    metadata: dict,
    *,
    include_content_hash_in_dest: bool = True,
    include_orig_name_in_dest: bool = False,
    include_metadata_hash_in_dest: Optional[bool] = None,
    dest_name_mode: str = DEST_NAME_MODE_ISO_GEO_HASH,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    content_hash_full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    content_hash_sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
    _geo_lookup: Optional[Dict[Tuple[float, float], str]] = None,
) -> dict:
    """Derive destination fields from ExifTool metadata using current logic.

    Sets ``Location``, ``ContentHash``, ``ContentHashMethod``, ``MetadataHash``
    (diagnostic), ``CaptureDtUtc``, ``CaptureDtLocal``, ``CaptureTzSuffix``,
    ``CaptureTzSource``, and ``DestFileBase``. Called on every
    ``cleaup_media_files`` run so date/naming code changes apply without
    re-running ExifTool (cached tags are enough).

    Default ``dest_name_mode`` is ``iso_geo_hash``: **local** capture time at the
    photo location (tag offset or GPS-inferred zone), optional reverse-geocode
    token, then content hash — under ``{YYYY}/{MM}/`` from that local date.
    Use ``content_only`` for hash-only filenames (same bytes always share one name).

    Files above ``content_hash_full_max_bytes`` (default 64 MiB) use a fast
    head/tail sample (xxHash64) instead of reading every byte. Hashes are
    cached by path, size, and mtime under ``./.cache/``.

    Args:
        metadata: ExifTool metadata dict (must include ``SourceFile``).
        include_content_hash_in_dest: Append content hash suffix (all modes except
            when ``content_only`` omits other parts — hash is always the main name there).
        include_orig_name_in_dest: Include original basename (``capture_content`` only).
        include_metadata_hash_in_dest: Deprecated alias for ``include_content_hash_in_dest``.
        dest_name_mode: ``iso_geo_hash`` (default), ``content_only``, or ``capture_content``.
        content_hash_mode: ``auto`` (default), ``full``, or ``sample``.
        content_hash_full_max_bytes: In ``auto`` mode, full-file xxHash64 up to this size.
        content_hash_sample_bytes: Head/tail bytes read per large file (each end).
        _geo_lookup: Pre-computed ``{(lat, lng): location_token}`` from
            ``_batch_reverse_geocode``.  When provided, skips per-file reverse-geocode.

    Returns:
        The same ``metadata`` dict, updated in place.
    """
    if include_metadata_hash_in_dest is not None:
        print(
            "WARNING: include_metadata_hash_in_dest is deprecated; "
            "destination names now use byte content hash (xxHash64) via include_content_hash_in_dest."
        )
        include_content_hash_in_dest = include_metadata_hash_in_dest
    file_path = metadata["SourceFile"]
    capture_parts = get_capture_dt_parts_from_metadata(metadata)
    capture_date_source = "metadata"
    tz_suffix = ""
    tz_source = ""
    if capture_parts is not None:
        create_dt = capture_parts["utc"]
        local_dt = capture_parts["local"]
        tz_suffix = capture_parts.get("tz_suffix") or ""
        tz_source = capture_parts.get("tz_source") or ""
        if not tz_suffix:
            tz_suffix = "+0000"
            if not tz_source:
                tz_source = "utc"
    else:
        create_dt = pd.Timestamp(get_file_create_date(file_path))
        local_dt = create_dt
        capture_date_source = "file_mtime"
        tz_suffix = "+0000"
        tz_source = "file_mtime"
    base_name, ext = os.path.basename(file_path).rsplit(".", 1)
    if _geo_lookup is not None:
        lat_lng = get_lat_lng_from_metadata(metadata)
        loc = _geo_lookup.get(lat_lng, "") if lat_lng else ""
    else:
        loc = get_location_from_metadata(metadata)
    metadata["Location"] = loc
    metadata["MetadataHash"] = get_hash_from_metadata(metadata)
    metadata["CaptureDtUtc"] = str(create_dt)
    metadata["CaptureDtLocal"] = str(local_dt)
    metadata["CaptureTzSuffix"] = tz_suffix
    metadata["CaptureTzSource"] = tz_source
    metadata["CaptureDateSource"] = capture_date_source
    metadata["DestYear"] = local_dt.strftime("%Y")
    metadata["DestMonth"] = local_dt.strftime("%m")
    metadata["CaptureIsoForDest"] = _capture_iso_for_filename(local_dt, tz_suffix)
    content_hash, content_hash_method = _get_content_fingerprint(
        file_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=content_hash_full_max_bytes,
        sample_bytes=content_hash_sample_bytes,
    )
    metadata["ContentHash"] = content_hash
    metadata["ContentHashMethod"] = content_hash_method
    content_hash_for_dest = get_content_hash_for_dest(
        file_path,
        content_hash,
        content_hash_mode=content_hash_mode,
        full_max_bytes=content_hash_full_max_bytes,
        sample_bytes=content_hash_sample_bytes,
    )
    metadata["ContentHashForDest"] = content_hash_for_dest
    metadata["DestFileBase"] = _build_dest_file_base(
        local_dt=local_dt,
        tz_suffix=tz_suffix,
        ext_lower=ext.lower(),
        loc=loc,
        content_hash_for_dest=content_hash_for_dest,
        dest_name_mode=dest_name_mode,
        base_name=base_name,
        include_content_hash_in_dest=include_content_hash_in_dest,
        include_orig_name_in_dest=include_orig_name_in_dest,
    )
    return metadata


def _batch_reverse_geocode(metadatas: List[dict]) -> Dict[Tuple[float, float], str]:
    """Pre-compute location tokens for all GPS coordinates in one batch call."""
    coords = []
    seen = set()
    for m in metadatas:
        lat_lng = get_lat_lng_from_metadata(m)
        if lat_lng and lat_lng not in seen:
            coords.append(lat_lng)
            seen.add(lat_lng)
    if not coords:
        return {}
    results = reverse_geocode.search(coords)
    lookup = {}
    for coord, res in zip(coords, results):
        lookup[coord] = f"{res['country_code']}-{res['city'].replace(' ', '-')}"
    return lookup


def augment_metadatas_for_dest(
    metadatas: List[dict],
    *,
    include_content_hash_in_dest: bool = True,
    include_orig_name_in_dest: bool = False,
    include_metadata_hash_in_dest: Optional[bool] = None,
    dest_name_mode: str = DEST_NAME_MODE_ISO_GEO_HASH,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    content_hash_full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    content_hash_sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
    progress: bool = True,
) -> List[dict]:
    """Apply ``augment_metadata_for_dest`` to each metadata dict."""
    geo_lookup = _batch_reverse_geocode(metadatas)
    for metadata in _iter_progress(
        metadatas, progress=progress, desc="augment", total=len(metadatas)
    ):
        augment_metadata_for_dest(
            metadata,
            include_content_hash_in_dest=include_content_hash_in_dest,
            include_orig_name_in_dest=include_orig_name_in_dest,
            include_metadata_hash_in_dest=include_metadata_hash_in_dest,
            dest_name_mode=dest_name_mode,
            content_hash_mode=content_hash_mode,
            content_hash_full_max_bytes=content_hash_full_max_bytes,
            content_hash_sample_bytes=content_hash_sample_bytes,
            _geo_lookup=geo_lookup,
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
    print(f"Processing {len(file_paths):,} files in {len(file_path_batches):,} batches...")

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
    *,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
    progress: bool = True,
) -> dict:
    """Simulate sequential processing to estimate outcome counts."""
    counts = defaultdict(int)
    for group_key, src_file_paths in _iter_progress(
        sorted(dest_file_src_files.items()),
        progress=progress,
        desc="preflight",
        total=len(dest_file_src_files),
    ):
        dest_year, dest_month, dest_file_base = group_key
        dest_file_path = os.path.join(
            dest_root_dir, dest_year, dest_month, dest_file_base
        )
        dest_exists = os.path.exists(dest_file_path)
        for src_path in sorted(src_file_paths):
            if not os.path.exists(src_path):
                counts["src_missing"] += 1
                continue
            src_meta = metadata_by_path.get(src_path, {})
            if src_meta.get("CaptureDateSource") == "file_mtime":
                counts["mtime_fallback"] += 1
            if dest_exists:
                if _files_are_identical(
                    src_path,
                    dest_file_path,
                    content_hash_mode=content_hash_mode,
                    full_max_bytes=full_max_bytes,
                    sample_bytes=sample_bytes,
                ):
                    counts["identical_at_dest"] += 1
                else:
                    counts["collision"] += 1
            else:
                counts["transfer"] += 1
                dest_exists = True
    return dict(counts)


def _mode_action_label(move_or_copy: str, *, past: bool = False) -> str:
    """Return ``move``/``moved`` or ``copy``/``copied`` for user-facing messages."""
    if move_or_copy == "move":
        return "moved" if past else "move"
    return "copied" if past else "copy"


def _print_mode_banner(
    move_or_copy: str, *, dry_run: bool, verify_transfers: bool
) -> None:
    """Print how this run treats sources vs the library."""
    mode = move_or_copy.upper()
    if move_or_copy == "move":
        behavior = (
            "rename (same FS) or copy+verify+trash (cross FS) each source into the library"
        )
    else:
        behavior = "copy into the library and leave source files on disk"
    if not verify_transfers:
        behavior += " (verify_transfers=False — sources are never sent to Trash automatically)"
    if dry_run:
        print(f"DRY RUN — mode={mode}: would {behavior}.")
    else:
        print(f"Mode={mode}: {behavior}.")


def _print_preflight_summary(
    counts: dict, *, dry_run: bool, move_or_copy: str
) -> None:
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
        print(
            f"  (dry_run=True — no files will be "
            f"{_mode_action_label(move_or_copy, past=True)})"
        )


_NEW_LIBRARY_ERRORS = frozenset(
    {
        "TRANSFERRED_SOURCE_DELETED",
        "TRANSFERRED_SOURCE_RENAMED",
        "TRANSFERRED_SOURCE_KEPT",
        "TRANSFERRED_SOURCE_KEPT_UNVERIFIED",
    }
)
_DUPLICATE_TRASHED_ERRORS = frozenset({"IDENTICAL_SOURCE_DELETED"})
_DUPLICATE_KEPT_ERRORS = frozenset(
    {
        "IDENTICAL_SOURCE_KEPT",
        "IDENTICAL_DESTINATION_EXISTS",
        "IDENTICAL_SOURCE_ALREADY_HANDLED",
    }
)
_COLLISION_QUARANTINE_ERRORS = frozenset(
    {"COLLISION_MOVED_TO_QUARANTINE", "COLLISION_WOULD_MOVE_TO_QUARANTINE"}
)
_COLLISION_LOGGED_ERRORS = frozenset({"DESTINATION_EXISTS"})
_DRY_RUN_TRASH_ERRORS = frozenset({"SOURCE_WOULD_TRASH_AFTER_VERIFY"})
_DRY_RUN_KEEP_ERRORS = frozenset({"SOURCE_WOULD_KEEP_AFTER_COPY"})
_FAILURE_PREFIXES = ("SOURCE_TRASH_FAILED:", "SOURCE_DELETE_FAILED:", "QUARANTINE_FAILED:")


def _summarize_run_rows(rows: List[dict], *, dry_run: bool = False) -> dict:
    """Bucket per-file log rows into counts for the run summary."""
    summary = {
        "processed": len(rows),
        "new_to_library": 0,
        "duplicate_trashed": 0,
        "duplicate_kept": 0,
        "collision_quarantined": 0,
        "collision_logged": 0,
        "dry_run_would_trash": 0,
        "dry_run_would_keep": 0,
        "dry_run_would_copy": 0,
        "src_missing": 0,
        "mtime_fallback": 0,
        "errors": 0,
        "other": 0,
    }
    for row in rows:
        if row.get("warning") == "NO_CAPTURE_DATE_USED_MTIME":
            summary["mtime_fallback"] += 1
        err = row.get("error") or ""
        if err in _NEW_LIBRARY_ERRORS:
            summary["new_to_library"] += 1
        elif err in _DUPLICATE_TRASHED_ERRORS:
            summary["duplicate_trashed"] += 1
        elif err in _DUPLICATE_KEPT_ERRORS:
            summary["duplicate_kept"] += 1
        elif err in _COLLISION_QUARANTINE_ERRORS:
            summary["collision_quarantined"] += 1
        elif err in _COLLISION_LOGGED_ERRORS:
            summary["collision_logged"] += 1
        elif err in _DRY_RUN_TRASH_ERRORS:
            summary["dry_run_would_trash"] += 1
        elif err in _DRY_RUN_KEEP_ERRORS:
            summary["dry_run_would_keep"] += 1
        elif err == "SRC_MISSING":
            summary["src_missing"] += 1
        elif err.startswith(_FAILURE_PREFIXES):
            summary["errors"] += 1
        elif err == "" and dry_run:
            summary["dry_run_would_copy"] += 1
        elif err:
            summary["other"] += 1
    return summary


def _print_run_summary(
    rows: List[dict],
    *,
    move_or_copy: str,
    dry_run: bool,
    dest_root_dir: str,
    log_file_path: str,
    collisions_dir: Optional[str] = None,
) -> None:
    """Print a single end-of-run summary (copies, duplicates, Trash, collisions)."""
    s = _summarize_run_rows(rows, dry_run=dry_run)
    mode = move_or_copy.upper()
    run_kind = "DRY RUN" if dry_run else "LIVE"
    duplicates = s["duplicate_trashed"] + s["duplicate_kept"]
    collisions = s["collision_quarantined"] + s["collision_logged"]
    new_trashed = sum(
        1 for r in rows if (r.get("error") or "") == "TRANSFERRED_SOURCE_DELETED"
    )
    sent_to_trash = s["duplicate_trashed"] + new_trashed
    sources_left = sum(
        1
        for r in rows
        if (r.get("error") or "")
        in (
            "TRANSFERRED_SOURCE_KEPT",
            "IDENTICAL_SOURCE_KEPT",
            "IDENTICAL_DESTINATION_EXISTS",
            "IDENTICAL_SOURCE_ALREADY_HANDLED",
            "TRANSFERRED_SOURCE_KEPT_UNVERIFIED",
        )
    )

    def line(label: str, n: int, indent: int = 1) -> None:
        if n:
            print(f"{'  ' * indent}{label:<26} {n:>8,}")

    print("")
    print("=" * 56)
    print(f"Run summary ({run_kind}, mode={mode})")
    print("=" * 56)
    print(f"  Destination: {dest_root_dir}")
    print(f"  {'Sources processed:':<26} {s['processed']:>8,}")

    print("\n  Library")
    if dry_run:
        line("Would add (new)", s["dry_run_would_copy"])
        line("Would send to Trash", s["dry_run_would_trash"])
        line("Would keep at source", s["dry_run_would_keep"] + s["duplicate_kept"])
    else:
        print(f"    {'Added (new copies):':<26} {s['new_to_library']:>8,}")
        if move_or_copy == "move":
            print(f"    {'Sent to Trash (new):':<26} {new_trashed:>8,}")
        print(f"    {'Duplicates (same bytes):':<26} {duplicates:>8,}")
        if duplicates:
            if move_or_copy == "move":
                line("  → Trash", s["duplicate_trashed"], indent=2)
                line("  → kept at source", s["duplicate_kept"], indent=2)
            else:
                line("  → kept at source", duplicates, indent=2)

    if collisions or (dry_run and s["collision_quarantined"]):
        line("Collisions (different bytes)", collisions)
        if collisions:
            q_label = "Would quarantine" if dry_run else "Quarantined"
            line(f"  → {q_label.lower()}", s["collision_quarantined"], indent=2)
            line("  → logged only", s["collision_logged"], indent=2)

    if not dry_run and move_or_copy == "move" and sent_to_trash:
        print(f"\n  Total sent to Trash:      {sent_to_trash:>8,}")
    if not dry_run and move_or_copy == "copy" and sources_left:
        print(f"\n  Sources left on disk:     {sources_left:>8,}")

    problems = s["src_missing"] + s["errors"] + s["other"]
    if problems:
        print("\n  Problems")
        line("Missing source", s["src_missing"], indent=1)
        line("Failed", s["errors"], indent=1)
        line("Other", s["other"], indent=1)

    if s["mtime_fallback"]:
        print(f"\n  Note: {s['mtime_fallback']:,} files used file mtime for capture date.")

    print(f"\n  Log: {log_file_path}")
    if collisions_dir and s["collision_quarantined"]:
        print(f"  Quarantine: {collisions_dir}")
    print("=" * 56)
    print("")


def _process_one_source(
    *,
    src_file_path: str,
    dest_file_path: str,
    dest_year: str,
    dest_month: str,
    dest_file_base: str,
    sibling_srcs: List[str],
    src_metadata: dict,
    dest_metadata: Optional[dict],
    metadata_by_path: Dict[str, dict],
    dest_root_dir: str,
    is_move: bool,
    dry_run: bool,
    verify_transfers: bool,
    collisions_dir: Optional[str],
    log_file,
    session: dict,
    verbose_collisions: bool,
    processed_srcs: Set[str],
    counters: dict,
    dest_content_hash: Optional[str],
    group_content_hashes: Set[str],
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
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
        src_content_hash=_metadata_content_hash(src_metadata),
    )
    if src_metadata.get("CaptureDateSource") == "file_mtime":
        row["warning"] = "NO_CAPTURE_DATE_USED_MTIME"

    if not os.path.exists(src_file_path):
        row["error"] = "SRC_MISSING"
        counters["src_missing"] += 1
        return _log_event(log_file, row, session)

    src_hash = _src_content_hash(
        src_metadata,
        src_file_path,
        content_hash_mode=content_hash_mode,
        full_max_bytes=full_max_bytes,
        sample_bytes=sample_bytes,
    )

    # 1) Destination exists → compare content; identical → delete verified source.
    if os.path.exists(dest_file_path):
        hash_match = _files_content_identical(
            src_file_path,
            dest_file_path,
            src_metadata,
            dest_content_hash,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
        if hash_match is False:
            row.update(
                _collision_details(
                    src_file_path,
                    dest_file_path,
                    src_metadata,
                    dest_metadata,
                    src_content_hash=src_hash,
                    dest_content_hash=dest_content_hash,
                    content_hash_mode=content_hash_mode,
                    full_max_bytes=full_max_bytes,
                    sample_bytes=sample_bytes,
                )
            )
            is_identical = False
        elif hash_match is True:
            is_identical = True
        else:
            row.update(
                _collision_details(
                    src_file_path,
                    dest_file_path,
                    src_metadata,
                    dest_metadata,
                    src_content_hash=src_hash,
                    dest_content_hash=dest_content_hash,
                    content_hash_mode=content_hash_mode,
                    full_max_bytes=full_max_bytes,
                    sample_bytes=sample_bytes,
                )
            )
            is_identical = _files_are_identical(
                src_file_path,
                dest_file_path,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            )
        if is_identical:
            if _verified_dest_has_src_content(
                src_file_path,
                dest_file_path,
                src_metadata,
                dest_content_hash,
                verify_transfers=verify_transfers,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            ):
                row["error"] = _handle_source_after_verify(
                    src_file_path,
                    is_move=is_move,
                    dry_run=dry_run,
                    counters=counters,
                    processed_srcs=processed_srcs,
                    reason="identical",
                )
                if row["error"].startswith(("SOURCE_TRASH_FAILED", "SOURCE_DELETE_FAILED")):
                    print(
                        f"unable to trash verified duplicate "
                        f"src={src_file_path}: {row['error']}"
                    )
            else:
                counters["identical_kept"] += 1
                row["error"] = "IDENTICAL_DESTINATION_EXISTS"
            if src_hash:
                group_content_hashes.add(src_hash)
        else:
            if is_move and collisions_dir and not dry_run:
                q_path = _quarantine_dest_path(
                    collisions_dir,
                    dest_year,
                    dest_month,
                    dest_file_base,
                    src_file_path,
                )
                row["quarantine_reloc"] = q_path
                try:
                    os.makedirs(os.path.dirname(q_path), exist_ok=True)
                    q_method = _safe_transfer(
                        src_file_path,
                        q_path,
                        verify=verify_transfers,
                        is_move=True,
                        content_hash_mode=content_hash_mode,
                        full_max_bytes=full_max_bytes,
                        sample_bytes=sample_bytes,
                    )
                    if q_method == "renamed":
                        counters["source_deleted"] += 1
                    elif _verified_dest_has_src_content(
                        src_file_path,
                        q_path,
                        src_metadata,
                        None,
                        verify_transfers=verify_transfers,
                        content_hash_mode=content_hash_mode,
                        full_max_bytes=full_max_bytes,
                        sample_bytes=sample_bytes,
                    ):
                        _handle_source_after_verify(
                            src_file_path,
                            is_move=True,
                            dry_run=False,
                            counters=counters,
                            processed_srcs=processed_srcs,
                            reason="transfer",
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

    # 2) Same content already handled in this dest group (hash seen, dest on disk).
    if src_hash and src_hash in group_content_hashes and os.path.exists(dest_file_path):
        prior_match = _files_content_identical(
            src_file_path,
            dest_file_path,
            src_metadata,
            dest_content_hash,
            content_hash_mode=content_hash_mode,
            full_max_bytes=full_max_bytes,
            sample_bytes=sample_bytes,
        )
        if prior_match is not False:
            if _verified_dest_has_src_content(
                src_file_path,
                dest_file_path,
                src_metadata,
                dest_content_hash,
                verify_transfers=verify_transfers,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            ):
                row["error"] = _handle_source_after_verify(
                    src_file_path,
                    is_move=is_move,
                    dry_run=dry_run,
                    counters=counters,
                    processed_srcs=processed_srcs,
                    reason="identical",
                )
                if row["error"].startswith(("SOURCE_TRASH_FAILED", "SOURCE_DELETE_FAILED")):
                    print(
                        f"unable to trash duplicate source "
                        f"src={src_file_path}: {row['error']}"
                    )
            else:
                counters["identical_kept"] += 1
                row["error"] = "IDENTICAL_SOURCE_ALREADY_HANDLED"
            row = _log_event(log_file, row, session)
            if verbose_collisions:
                _print_destination_exists(row)
            return row

    # 3) Destination missing → move/copy, verify, handle source.
    error = ""
    if not dry_run:
        try:
            method = _safe_transfer(
                src_file_path,
                dest_file_path,
                verify=verify_transfers,
                is_move=is_move,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
            )
            counters["success"] += 1
            if src_hash:
                group_content_hashes.add(src_hash)
            if method == "renamed":
                processed_srcs.add(src_file_path)
                counters["source_deleted"] += 1
                error = "TRANSFERRED_SOURCE_RENAMED"
            elif verify_transfers:
                error = _handle_source_after_verify(
                    src_file_path,
                    is_move=is_move,
                    dry_run=False,
                    counters=counters,
                    processed_srcs=processed_srcs,
                    reason="transfer",
                )
                if error.startswith(("SOURCE_TRASH_FAILED", "SOURCE_DELETE_FAILED")):
                    print(
                        f"unable to trash source after "
                        f"{_mode_action_label('move' if is_move else 'copy', past=True)} "
                        f"src={src_file_path}: {error}"
                    )
            else:
                counters["source_kept_unverified"] += 1
                error = "TRANSFERRED_SOURCE_KEPT_UNVERIFIED"
            if src_file_path not in processed_srcs:
                processed_srcs.add(src_file_path)
            companion = _transfer_live_companion(
                src_file_path,
                metadata_by_path=metadata_by_path,
                dest_root_dir=dest_root_dir,
                verify=verify_transfers,
                processed_srcs=processed_srcs,
                counters=counters,
                dry_run=False,
                content_hash_mode=content_hash_mode,
                full_max_bytes=full_max_bytes,
                sample_bytes=sample_bytes,
                is_move=is_move,
            )
            if companion:
                row["linked_companion"] = companion
        except Exception as e:
            counters["unexpected"] += 1
            error = f"{type(e).__name__}: {e}"
            print(
                f"ERROR: skipping {_mode_action_label('move' if is_move else 'copy')} "
                f"src={src_file_path}: {error}"
            )
    else:
        companion = _live_photo_companion_path(src_file_path)
        if (
            companion
            and companion in metadata_by_path
            and companion not in processed_srcs
            and not os.path.exists(_dest_file_path(dest_root_dir, metadata_by_path[companion]))
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
    include_content_hash_in_dest: bool = True,
    include_orig_name_in_dest: bool = False,
    include_metadata_hash_in_dest: Optional[bool] = None,
    dest_name_mode: str = DEST_NAME_MODE_ISO_GEO_HASH,
    collisions_dir: Optional[str] = None,
    exclude_dirs: Optional[List[str]] = None,
    progress: bool = True,
    verify_transfers: bool = True,
    content_hash_mode: str = CONTENT_HASH_MODE_AUTO,
    content_hash_full_max_bytes: int = DEFAULT_CONTENT_HASH_FULL_MAX_BYTES,
    content_hash_sample_bytes: int = DEFAULT_CONTENT_HASH_SAMPLE_BYTES,
) -> List[dict]:
    """Organize media into year folders with canonical capture-based filenames.

    Uses a **streaming pipeline**: each file is augmented (datetime, location, hash),
    moved/copied, and logged before the next file starts. If interrupted, all
    previously processed files are done — restart picks up where it left off
    (moved sources are gone from the source scan, copied sources will match the
    existing destination and be logged as duplicates).

    On the **same filesystem**, ``move`` mode uses ``os.rename()`` for instant
    atomic moves (no copy, no verify, no trash needed). Cross-filesystem moves
    fall back to copy + hash-verify + send2trash.

    Layout: ``{dest_root_dir}/{YYYY}/{MM}/{DestFileBase}`` (default ``iso_geo_hash`` filenames:
    ``{YYYYMMDD}_{HHMMSS}{±HHMM}__{geo}__{hash16}.{ext}`` (empty geo → ``____`` in the name).

    Args:
        src_root_dir: Tree of source photos/videos.
        dest_root_dir: Library root; ``{YYYY}/{MM}`` subdirs are created as needed.
        dry_run: If True, do not move/copy or create dirs (still writes log rows).
        refresh_metacache: If True, re-run ExifTool and replace the on-disk cache
            (use when files are new/updated; not needed for date/naming logic changes).
        move_or_copy: ``"move"`` or ``"copy"``. On the same filesystem, **move** uses
            ``os.rename()`` (instant, atomic). Cross-filesystem moves copy bytes with
            hash verification, then send the source to **Trash** (``send2trash``).
            **Copy** always copies bytes and leaves sources on disk.
        include_content_hash_in_dest: When ``dest_name_mode="capture_content"``, append
            content hash (16 hex chars of xxHash64) to the capture-based filename.
        include_orig_name_in_dest: Include original basename (``capture_content`` only).
        include_metadata_hash_in_dest: Deprecated alias for ``include_content_hash_in_dest``.
        dest_name_mode: ``iso_geo_hash`` (default): ISO-like capture time, optional geo,
            then hash in the filename. ``content_only``: ``{hash_16}.{ext}`` only.
            ``capture_content``: legacy capture/location/hash pattern.
        collisions_dir: If set and ``move_or_copy`` is ``"move"``, when the destination
            exists with different content, move the source here for manual review.
        exclude_dirs: Extra directory trees to exclude from the source scan. ``dest_root_dir``
            and ``collisions_dir`` are always excluded.
        progress: Show progress bars during scan and transfer.
        verify_transfers: After copy to destination, verify bytes match before deleting
            the source. When False, sources are never sent to Trash automatically.
        content_hash_mode: ``auto`` (default): full-file xxHash64 up to
            ``content_hash_full_max_bytes``, then head/tail sampling for larger files.
            ``full`` always reads every byte; ``sample`` always uses head/tail sampling.
        content_hash_full_max_bytes: Size threshold for ``auto`` mode (default 64 MiB).
        content_hash_sample_bytes: Bytes read from each end of large files (default 4 MiB).
        log_file_path: Log file path; defaults to ``{dest_root_dir}/log.txt``.
            All events (moves, collisions, dry-run skips) are appended with
            a local-time ``logged_at`` timestamp (machine timezone) and session ``run_id``.
        verbose_collisions: If True, print a two-line summary per collision to stdout.

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
    if collisions_dir and not is_move:
        print(
            "NOTE: collisions_dir only quarantines sources in move mode; "
            "copy mode will log DESTINATION_EXISTS without relocating."
        )

    auto_exclude = [
        dest_root_dir,
        collisions_dir,
    ]
    exclude_roots = _normalize_exclude_roots(
        (exclude_dirs or []) + [p for p in auto_exclude if p]
    )

    _print_mode_banner(move_or_copy, dry_run=dry_run, verify_transfers=verify_transfers)

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

    print("Getting metadatas for all media files (cached if available)...")
    metadatas = get_metadatas_mproc(  # pylint: disable=unexpected-keyword-arg
        file_paths=file_paths,
        __force_refresh=refresh_metacache,
        dest_logic_version=DEST_LOGIC_VERSION,
    )
    print(f"Retrieved {len(metadatas):,.0f} metadatas.")

    if log_file_path is None:
        log_file_path = os.path.join(dest_root_dir, "log.txt")
    os.makedirs(dest_root_dir, exist_ok=True)
    if collisions_dir and not dry_run:
        os.makedirs(collisions_dir, exist_ok=True)

    geo_lookup = _batch_reverse_geocode(metadatas)
    metadata_by_path: Dict[str, dict] = {m["SourceFile"]: m for m in metadatas}

    action = _mode_action_label(move_or_copy, past=True)
    print(
        f"Processing files — augment + {action} "
        f"(logging to {log_file_path})..."
    )
    rows: List[dict] = []
    counters = defaultdict(int)
    processed_srcs: Set[str] = set()
    dest_file_src_files: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    dest_content_hashes: Dict[str, Optional[str]] = {}
    group_content_hashes_map: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    with open(log_file_path, "a") as log_file:
        for metadata in _iter_progress(
            metadatas, progress=progress, desc=move_or_copy, total=len(metadatas)
        ):
            src_file_path = metadata["SourceFile"]
            if src_file_path in processed_srcs:
                continue

            try:
                augment_metadata_for_dest(
                    metadata,
                    include_content_hash_in_dest=include_content_hash_in_dest,
                    include_orig_name_in_dest=include_orig_name_in_dest,
                    include_metadata_hash_in_dest=include_metadata_hash_in_dest,
                    dest_name_mode=dest_name_mode,
                    content_hash_mode=content_hash_mode,
                    content_hash_full_max_bytes=content_hash_full_max_bytes,
                    content_hash_sample_bytes=content_hash_sample_bytes,
                    _geo_lookup=geo_lookup,
                )
            except Exception as e:
                counters["unexpected"] += 1
                row = _log_event(
                    log_file,
                    {"src": src_file_path, "error": f"AUGMENT_FAILED: {e}"},
                    session,
                )
                rows.append(row)
                print(f"ERROR: augment failed for {src_file_path}: {e}")
                continue

            group_key = _dest_group_key(metadata)
            dest_file_src_files[group_key].append(src_file_path)
            sibling_srcs = [
                p for p in dest_file_src_files[group_key] if p != src_file_path
            ]

            dest_year, dest_month, dest_file_base = group_key
            dest_file_path = os.path.join(
                dest_root_dir, dest_year, dest_month, dest_file_base
            )
            if not dry_run:
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

            dest_metadata = metadata_by_path.get(dest_file_path)
            dest_content_hash = dest_content_hashes.get(dest_file_path)
            if dest_content_hash is None and os.path.exists(dest_file_path):
                dest_content_hash = _dest_content_fingerprint(
                    dest_file_path,
                    content_hash_mode=content_hash_mode,
                    full_max_bytes=content_hash_full_max_bytes,
                    sample_bytes=content_hash_sample_bytes,
                )
            group_content_hashes = group_content_hashes_map[group_key]

            row = _process_one_source(
                src_file_path=src_file_path,
                dest_file_path=dest_file_path,
                dest_year=dest_year,
                dest_month=dest_month,
                dest_file_base=dest_file_base,
                sibling_srcs=sibling_srcs,
                src_metadata=metadata,
                dest_metadata=dest_metadata,
                metadata_by_path=metadata_by_path,
                dest_root_dir=dest_root_dir,
                is_move=is_move,
                dry_run=dry_run,
                verify_transfers=verify_transfers,
                collisions_dir=collisions_dir if is_move else None,
                log_file=log_file,
                session=session,
                verbose_collisions=verbose_collisions,
                processed_srcs=processed_srcs,
                counters=counters,
                dest_content_hash=dest_content_hash,
                group_content_hashes=group_content_hashes,
                content_hash_mode=content_hash_mode,
                full_max_bytes=content_hash_full_max_bytes,
                sample_bytes=content_hash_sample_bytes,
            )
            rows.append(row)
            row_hash = _metadata_content_hash(metadata)
            if row_hash and row.get("error") in (
                "",
                "TRANSFERRED_SOURCE_DELETED",
                "TRANSFERRED_SOURCE_RENAMED",
                "TRANSFERRED_SOURCE_KEPT",
                "IDENTICAL_SOURCE_DELETED",
                "IDENTICAL_SOURCE_KEPT",
                "IDENTICAL_DESTINATION_EXISTS",
                "IDENTICAL_SOURCE_ALREADY_HANDLED",
            ):
                dest_content_hashes[dest_file_path] = row_hash
            elif os.path.exists(dest_file_path) and dest_file_path not in dest_content_hashes:
                dest_content_hashes[dest_file_path] = _dest_content_fingerprint(
                    dest_file_path,
                    content_hash_mode=content_hash_mode,
                    full_max_bytes=content_hash_full_max_bytes,
                    sample_bytes=content_hash_sample_bytes,
                )

        _print_run_summary(
            rows,
            move_or_copy=move_or_copy,
            dry_run=dry_run,
            dest_root_dir=dest_root_dir,
            log_file_path=log_file_path,
            collisions_dir=collisions_dir,
        )
    _persist_content_fp_cache()
    return rows
