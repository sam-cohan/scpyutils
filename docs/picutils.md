# picutils — media library cleanup

`scpyutils.picutils` scans photos and videos, reads capture metadata with ExifTool, and organizes files into a deduplicated library. Files are always **copied** into the library with verification. In **`move`** mode, verified sources are sent to the **Trash** (recoverable); in **`copy`** mode, sources stay on disk.

## Installation

`picutils` depends on the **ExifTool** CLI (system) and several Python packages that are **not** all pulled in by a minimal `pip install scpyutils`. Install everything below before running `cleaup_media_files`.

### 1. ExifTool (system)

ExifTool must be on your `PATH`. On macOS with Homebrew:

```bash
brew install exiftool
exiftool -ver   # sanity check — should print a version number
```

The Python wrapper `pyexiftool` shells out to this binary; it does not replace it.

### 2. Python packages for picutils

Install the picutils-specific packages into your virtualenv (versions that are known to work):

```bash
pip install pyexiftool==0.4.13 exifread reverse-geocode xxhash send2trash
```

One-liner matching a typical notebook setup:

```bash
brew install exiftool && pip install pyexiftool==0.4.13 exifread reverse-geocode xxhash send2trash send2trash
```

| Package | Purpose |
|---------|---------|
| `pyexiftool==0.4.13` | Batch-read EXIF/XMP/QuickTime tags via the ExifTool CLI |
| `exifread` | GPS / EXIF parsing fallback for some JPEG paths |
| `reverse-geocode` | Human-readable place names from GPS coordinates |
| `xxhash` | Fast content fingerprints for dedup and filenames |
| `send2trash` | Move mode: send verified sources to the system Trash (not permanent delete) |

Optional for **GPS → timezone** when EXIF has no offset tag:

```bash
pip install timezonefinder
```

`pandas`, `tqdm`, and other shared `scpyutils` deps are installed when you install the package (next step).

### 3. Install scpyutils

From the repo root:

```bash
pip install -e .
```

### 4. Optional — HEIC / Live Photos

Only needed if you process `.heic` files:

```bash
brew install libheif
pip install pyheif piexif
```

### Verify

```bash
python -c "import exiftool, exifread, reverse_geocode, xxhash; from scpyutils.picutils import cleaup_media_files; print('ok')"
```

## Destination layout

Default mode (`dest_name_mode="iso_geo_hash"`):

```
{dest_root}/{YYYY}/{MM}/{capture_iso}__{geo}__{hash16}.{ext}
```

Always **three** segments joined with **`__`**: capture time (with offset when known), geo, hash. Unknown GPS leaves geo as an empty string, which appears as **`____`** between the delimiters (e.g. `20191224_091530+0000____a1b2c3d4e5f67890.mov`). With a place name: `20200315_143022-0700__US-San-Francisco__hash`.

| Part | Meaning |
|------|---------|
| `YYYY` / `MM` | Capture year and month (local at photo, or UTC if zone unknown) |
| `capture_iso` | Always includes offset: `20200315_143022-0700` (tag/GPS), `20200101_120000+0000` (UTC / unknown) |
| `geo` | `CC-City` from GPS (e.g. `US-San-Francisco`), or empty → `____` in the filename |
| `hash16` | First 16 hex chars of **xxHash64** over file bytes |
| `ext` | Lowercase extension |

**Examples**

```
~/PhotosLibrary/2020/03/20200315_143022-0700__US-San-Francisco__a3f29b1c4e8d7012.jpg
~/PhotosLibrary/2019/12/20191224_091530+0000____a1b2c3d4e5f67890.mov   # no GPS
```

Byte-identical files with the **same** capture time and location share one filename. Different capture dates or GPS produce different names even when bytes match.

**Hash-only filenames** (old behavior): `dest_name_mode="content_only"` → `{YYYY}/{MM}/{hash16}.{ext}`.

**Legacy pattern**: `dest_name_mode="capture_content"` → `YYYYMMDD_HHMMSS` with optional location/original name (`__` separators).

Supported extensions include: `jpg`, `jpeg`, `png`, `gif`, `heic`, `mov`, `mp4`, `mkv`, `avi`, `cr2`, `arw`, `tiff`, and others (see `MEDIA_EXT_RE` in `picutils.py`).

## Quick start

### 1. Dry run (recommended first)

No files are written to the library (dry run). A log is still written under the destination root. The run prints `Mode=MOVE` or `Mode=COPY` and what it would do to sources.

```python
from scpyutils.picutils import cleaup_media_files

rows = cleaup_media_files(
    src_root_dir="/Volumes/CameraRoll/import",
    dest_root_dir="/Volumes/PhotosLibrary",
    dry_run=True,
    refresh_metacache=True,   # first run: build ExifTool cache
    move_or_copy="move",
    verify_transfers=True,
    progress=True,
)

print(f"Processed {len(rows)} files")
print(f"Sample result: {rows[0] if rows else 'no files'}")
```

Review the pre-flight summary, the **run summary** at the end, and `{dest_root}/log.txt`.

### 2. Live run

Use an empty or new destination directory.

```python
from scpyutils.picutils import cleaup_media_files

rows = cleaup_media_files(
    src_root_dir="/Volumes/CameraRoll/import",
    dest_root_dir="/Volumes/PhotosLibrary",
    dry_run=False,
    move_or_copy="move",
    verify_transfers=True,
    progress=True,
)
```

Flow per file (`move_or_copy="move"`):

1. Copy to `{dest}.picutils.part` → verify bytes → atomic rename.
2. **Move:** send source to Trash after verification (`send2trash`). **Copy:** leave source in place.
3. If destination already exists with **identical** content → Trash source in move mode; keep source in copy mode.
4. If destination exists with **different** content → log collision (optional quarantine in move mode).

### 3. Quarantine collisions

When the destination path is taken by different bytes, move the source aside for manual review:

```python
cleaup_media_files(
    src_root_dir="/data/inbox",
    dest_root_dir="/data/library",
    dry_run=False,
    move_or_copy="move",
    collisions_dir="/data/library/_collisions",
    verbose_collisions=True,
)
```

Quarantine layout mirrors the library: `{collisions_dir}/{YYYY}/{MM}/{DD}/{dest_base}__{original_name}`.

### 4. CLI one-liner

```bash
python -c "
from scpyutils.picutils import cleaup_media_files
cleaup_media_files('/path/src', '/path/dest', dry_run=False, refresh_metacache=True)
"
```

## Common options

```python
cleaup_media_files(
    src_root_dir="...",
    dest_root_dir="...",

    # Safety
    dry_run=True,              # default True — set False to actually move
    verify_transfers=True,     # verify copy before deleting source

    # Scan
    refresh_metacache=False,   # True when source files added/changed
    exclude_dirs=["/path/skip"],
    progress=True,

    # Naming (default: iso_geo_hash)
    dest_name_mode="iso_geo_hash",

    # Large files (videos)
    content_hash_mode="auto",       # "auto" | "full" | "sample"
    content_hash_full_max_bytes=64 * 1024 * 1024,
    content_hash_sample_bytes=4 * 1024 * 1024,

    # Logging
    log_file_path=None,        # default: {dest_root}/log.txt
    verbose_collisions=False,
)
```

### Legacy / alternate filename modes

```python
# Hash-only basename (same bytes → one name everywhere in the library)
cleaup_media_files("...", "...", dest_name_mode="content_only")

# Old YYYYMMDD_HHMMSS + optional location/original name
cleaup_media_files(
    "...",
    "...",
    dest_name_mode="capture_content",
    include_orig_name_in_dest=True,
)
```

Default `iso_geo_hash` does **not** use `capture_content`; order is always **date → geo → hash**.

## Content hashing

| Mode | Behavior |
|------|----------|
| `auto` (default) | Full-file xxHash64 if size ≤ 64 MiB; otherwise hash of size + first 4 MiB + last 4 MiB |
| `full` | Always read entire file |
| `sample` | Always use head/tail sampling |

Fingerprints are cached in `./.cache/picutils_content_fp.json` (keyed by path, size, mtime) so re-runs skip unchanged files.

## Capture date

Capture time comes from a tiered ExifTool tag search (earliest valid value wins within each tier). Apple `QuickTime:CreationDate` is preferred over container `CreateDate`. `ModifyDate` tags are ignored. If nothing valid is found, file modification time is used (`CaptureDateSource="file_mtime"` in the log).

**Folders and ISO filenames** use **local wall time** at the photo: explicit offsets on the tag (e.g. `EXIF:OffsetTimeOriginal`, Apple `QuickTime:CreationDate`), or GPS + `timezonefinder` when EXIF is naive. Capture time in filenames is always ``YYYYMMDD_HHMMSS±HHMM`` (no underscore before the offset). Offsets come from EXIF/tag, GPS-inferred zone, or ``+0000`` when unknown (including file-mtime fallback). `CaptureDtUtc` is still stored for sorting and logs.

## Run summary

At the end of each run, a boxed **run summary** is printed to stdout (counts come from `log.txt` rows):

```
========================================================
Run summary (LIVE, mode=MOVE)
========================================================
  Destination: /Volumes/PhotosLibrary
  Sources processed:             100

  Library
    Added (new copies):             60
    Sent to Trash (new):            60
    Duplicates (same bytes):        35
      → Trash                        33
      → kept at source                2
    Collisions (different bytes):      3
      → quarantined                     2
      → logged only                     1

  Total sent to Trash:              93

  Log: /Volumes/PhotosLibrary/log.txt
========================================================
```

In **copy** mode, “Sent to Trash” lines are omitted and **Sources left on disk** is shown instead. **Dry run** uses “Would …” labels and does not change files.

## Log file

Each run appends JSON lines to `{dest_root}/log.txt` (or `log_file_path`). Rows include:

- `run_id`, `dry_run`, `move_or_copy`, `logged_at` (local machine timezone, ISO-8601)
- `src`, `dest`, `error`
- Optional: `src_content_hash`, sizes, quarantine path, collision details

### Common `error` values

| Value | Meaning |
|-------|---------|
| `""` | Transferred successfully (source kept if `verify_transfers=False`) |
| `TRANSFERRED_SOURCE_DELETED` | Move mode: copied and verified; source sent to Trash |
| `TRANSFERRED_SOURCE_KEPT` | Copy mode: copied and verified; source left on disk |
| `IDENTICAL_SOURCE_DELETED` | Move mode: byte-identical to dest; source sent to Trash |
| `IDENTICAL_SOURCE_KEPT` | Copy mode: byte-identical to dest; source left on disk |
| `IDENTICAL_DESTINATION_EXISTS` | Same as dest; source kept (not verified enough to trash) |
| `SOURCE_TRASH_FAILED` | Move mode: could not send source to Trash |
| `SOURCE_WOULD_TRASH_AFTER_VERIFY` | Dry run: would send source to Trash after verify |
| `DESTINATION_EXISTS` | Dest path taken by different content |
| `COLLISION_MOVED_TO_QUARANTINE` | Different content; source moved to `collisions_dir` |
| `NO_CAPTURE_DATE_USED_MTIME` | Warning in `warning` field; mtime used for folders |

## Caches

| Path | Purpose |
|------|---------|
| `./.cache/` | ExifTool metadata cache (via `get_metadatas_mproc`) |
| `./.cache/picutils_content_fp.json` | Content hash cache |

Use `refresh_metacache=True` when files under `src_root_dir` change. Content hash cache invalidates automatically when size or mtime changes.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_picutils.py -v
```

## Example session

```python
from scpyutils.picutils import cleaup_media_files

SRC = "/Users/me/Pictures/Inbox"
DEST = "/Users/me/Pictures/Library"
QUARANTINE = "/Users/me/Pictures/Library/_review"

# Preview
cleaup_media_files(
    SRC, DEST,
    dry_run=True,
    refresh_metacache=True,
    collisions_dir=QUARANTINE,
)

# Go live
cleaup_media_files(
    SRC, DEST,
    dry_run=False,
    move_or_copy="move",
    verify_transfers=True,
    collisions_dir=QUARANTINE,
    verbose_collisions=True,
)
```

After a successful run, `Inbox` should be empty except for files that collided or failed verification; the library holds copies under `{YYYY}/{MM}/`.
