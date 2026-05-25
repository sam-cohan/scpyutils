"""Tests for scpyutils.picutils robustness helpers and cleanup flow."""

import json
import os
import sys
import types
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest

import scpyutils.picutils as picu


def test_parse_metadata_datetime_exif_and_offset():
    dt = picu._parse_metadata_datetime("2018:10:03 12:19:21-04:00")
    assert dt == pd.Timestamp("2018-10-03 16:19:21")
    assert dt.tzinfo is None


def test_parse_metadata_datetime_rejects_invalid():
    assert picu._parse_metadata_datetime("0000:00:00 00:00:00") is None
    assert picu._parse_metadata_datetime("") is None


def test_get_create_dt_prefers_apple_creation_date():
    metadata = {
        "SourceFile": "/fake.mov",
        "QuickTime:CreationDate": "2018:10:03 12:19:21-04:00",
        "QuickTime:CreateDate": "2018:12:21 14:52:52",
    }
    dt = picu.get_create_dt_from_metadata(metadata)
    assert dt == pd.Timestamp("2018-10-03 16:19:21")


def test_get_all_file_paths_not_match_excludes(tmp_path):
    included = tmp_path / "keep.jpg"
    excluded = tmp_path / "skip.jpg"
    included.write_bytes(b"a")
    excluded.write_bytes(b"b")
    paths = picu.get_all_file_paths(
        str(tmp_path),
        match_re=r"\.jpg$",
        not_match_re=r"skip",
        progress=False,
    )
    assert str(included) in paths
    assert str(excluded) not in paths


def test_exclude_dirs_skips_library_tree(tmp_path):
    src = tmp_path / "src"
    library = tmp_path / "library"
    (src / "inbox").mkdir(parents=True)
    (library / "2020").mkdir(parents=True)
    inbox_file = src / "inbox" / "photo.jpg"
    library_file = library / "2020" / "photo.jpg"
    inbox_file.write_bytes(b"inbox")
    library_file.write_bytes(b"library")

    paths = picu.get_all_media_file_paths(
        str(src),
        exclude_dirs=[str(library)],
        progress=False,
    )
    assert str(inbox_file) in paths
    assert str(library_file) not in paths


def test_files_are_identical_and_safe_transfer(tmp_path):
    src = tmp_path / "a.jpg"
    dest = tmp_path / "b.jpg"
    src.write_bytes(b"same-bytes")
    assert not picu._files_are_identical(str(src), str(dest))
    method = picu._safe_transfer(str(src), str(dest), verify=True)
    assert method == "copied"
    assert dest.read_bytes() == b"same-bytes"
    assert picu._files_are_identical(str(src), str(dest))
    assert src.exists()


def test_safe_transfer_rename_on_same_fs(tmp_path):
    src = tmp_path / "src" / "photo.jpg"
    dest = tmp_path / "dest" / "photo.jpg"
    src.parent.mkdir()
    dest.parent.mkdir()
    src.write_bytes(b"rename-me")
    method = picu._safe_transfer(str(src), str(dest), is_move=True)
    assert method == "renamed"
    assert not src.exists()
    assert dest.read_bytes() == b"rename-me"


def test_safe_transfer_copy_mode_never_renames(tmp_path):
    src = tmp_path / "src" / "photo.jpg"
    dest = tmp_path / "dest" / "photo.jpg"
    src.parent.mkdir()
    dest.parent.mkdir()
    src.write_bytes(b"keep-me")
    method = picu._safe_transfer(str(src), str(dest), is_move=False, verify=True)
    assert method == "copied"
    assert src.exists()
    assert dest.read_bytes() == b"keep-me"


def test_delete_after_verified_transfer(tmp_path, monkeypatch):
    trashed: list[str] = []

    def _record_trash(path: str) -> None:
        trashed.append(path)
        os.remove(path)

    monkeypatch.setattr(picu, "_trash_file", _record_trash)
    src = tmp_path / "a.jpg"
    dest = tmp_path / "out" / "a.jpg"
    src.write_bytes(b"move-me")
    counters = defaultdict(int)
    picu._safe_transfer(str(src), str(dest), verify=True)
    assert picu._verified_dest_has_src_content(
        str(src), str(dest), {}, None, verify_transfers=True
    )
    err = picu._delete_verified_source(
        str(src), dry_run=False, counters=counters, processed_srcs=set(), reason="transfer"
    )
    assert err == "TRANSFERRED_SOURCE_DELETED"
    assert trashed == [str(src)]
    assert dest.read_bytes() == b"move-me"
    assert not src.exists()


def test_live_photo_companion_path(tmp_path):
    heic = tmp_path / "IMG_1234.heic"
    mov = tmp_path / "IMG_1234.MOV"
    heic.write_bytes(b"h")
    mov.write_bytes(b"m")
    assert os.path.samefile(picu._live_photo_companion_path(str(heic)), str(mov))
    assert os.path.samefile(picu._live_photo_companion_path(str(mov)), str(heic))


def test_summarize_run_rows_move_and_duplicates():
    rows = [
        {"error": "TRANSFERRED_SOURCE_DELETED"},
        {"error": "IDENTICAL_SOURCE_DELETED"},
        {"error": "IDENTICAL_DESTINATION_EXISTS"},
        {"error": "DESTINATION_EXISTS"},
        {"error": "COLLISION_MOVED_TO_QUARANTINE"},
        {"warning": "NO_CAPTURE_DATE_USED_MTIME"},
    ]
    s = picu._summarize_run_rows(rows)
    assert s["processed"] == 6
    assert s["new_to_library"] == 1
    assert s["duplicate_trashed"] == 1
    assert s["duplicate_kept"] == 1
    assert s["collision_logged"] == 1
    assert s["collision_quarantined"] == 1
    assert s["mtime_fallback"] == 1


def test_batch_reverse_geocode_real():
    """Verify _batch_reverse_geocode uses the real reverse_geocode library correctly."""
    metadatas = [
        {"Composite:GPSLatitude": 37.7749, "Composite:GPSLongitude": -122.4194},
        {"Composite:GPSLatitude": 43.6532, "Composite:GPSLongitude": -79.3832},
        {},
    ]
    lookup = picu._batch_reverse_geocode(metadatas)
    assert len(lookup) == 2
    assert "US-San-Francisco" in lookup[(37.7749, -122.4194)]
    assert lookup[(43.6532, -79.3832)].startswith("CA-")


def test_get_location_from_metadata_real():
    """Verify get_location_from_metadata calls reverse_geocode without error."""
    meta = {"Composite:GPSLatitude": 37.7749, "Composite:GPSLongitude": -122.4194}
    loc = picu.get_location_from_metadata(meta)
    assert "US-San-Francisco" in loc

    assert picu.get_location_from_metadata({}) == ""


def test_dest_filename_iso_geo_hash_default(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"unique-bytes-for-hash")
    content_hash = picu.get_content_hash_for_dest(str(path))
    metadata = {
        "SourceFile": str(path),
        "EXIF:DateTimeOriginal": "2020:01:01 12:00:00",
    }
    picu.augment_metadata_for_dest(metadata)
    assert metadata["DestFileBase"] == f"20200101_120000+0000____{content_hash}.jpg"
    assert metadata["DestYear"] == "2020"
    assert metadata["DestMonth"] == "01"
    assert metadata["CaptureIsoForDest"] == "20200101_120000+0000"
    assert metadata["CaptureTzSource"] == "utc"
    assert metadata["Location"] == ""


def test_dest_filename_uses_exif_offset_local_time(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"offset-tz-bytes")
    metadata = {
        "SourceFile": str(path),
        "EXIF:DateTimeOriginal": "2020:03:15 14:30:22",
        "EXIF:OffsetTimeOriginal": "-04:00",
    }
    picu.augment_metadata_for_dest(
        metadata, include_content_hash_in_dest=False
    )
    assert metadata["CaptureIsoForDest"] == "20200315_143022-0400"
    assert metadata["DestYear"] == "2020"
    assert metadata["DestMonth"] == "03"
    assert metadata["DestFileBase"] == "20200315_143022-0400__.jpg"  # no hash → geo empty → iso__
    assert "18:30:22" in metadata["CaptureDtUtc"]


def test_dest_filename_gps_inferred_timezone(tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"gps-tz-bytes")

    class FakeTZF:
        def timezone_at(self, *, lng, lat):
            return "America/New_York"

    fake_mod = types.ModuleType("timezonefinder")
    fake_mod.TimezoneFinder = FakeTZF
    monkeypatch.setitem(sys.modules, "timezonefinder", fake_mod)
    metadata = {
        "SourceFile": str(path),
        "EXIF:DateTimeOriginal": "2020:07:04 12:00:00",
        "Composite:GPSLatitude": 40.7,
        "Composite:GPSLongitude": -74.0,
    }
    picu.augment_metadata_for_dest(
        metadata, include_content_hash_in_dest=False
    )
    assert metadata["CaptureTzSource"] == "gps"
    assert metadata["CaptureIsoForDest"].endswith("-0400") or metadata[
        "CaptureIsoForDest"
    ].endswith("-0500")
    assert metadata["DestMonth"] == "07"


def test_dest_filename_utc_fallback_without_tz(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"no-tz-bytes")
    metadata = {
        "SourceFile": str(path),
        "EXIF:DateTimeOriginal": "2020:06:10 09:15:30",
    }
    picu.augment_metadata_for_dest(
        metadata, include_content_hash_in_dest=False
    )
    assert metadata["CaptureTzSource"] == "utc"
    assert metadata["CaptureIsoForDest"] == "20200610_091530+0000"
    assert metadata["Location"] == ""


def test_dest_filename_content_only_mode(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"unique-bytes-for-hash")
    content_hash = picu.get_content_hash_for_dest(str(path))
    metadata = {
        "SourceFile": str(path),
        "EXIF:DateTimeOriginal": "2020:01:01 12:00:00",
    }
    picu.augment_metadata_for_dest(
        metadata, dest_name_mode=picu.DEST_NAME_MODE_CONTENT_ONLY
    )
    assert metadata["DestFileBase"] == f"{content_hash}.jpg"


def test_same_bytes_same_dest_name_different_capture_tags(tmp_path):
    """Byte-identical files share one dest name even if ExifTool dates differ."""
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"same-payload")
    b.write_bytes(b"same-payload")
    meta_a = {"SourceFile": str(a), "EXIF:DateTimeOriginal": "2020:01:01 12:00:00"}
    meta_b = {"SourceFile": str(b), "EXIF:DateTimeOriginal": "2021:06:15 08:30:00"}
    picu.augment_metadata_for_dest(meta_a, dest_name_mode=picu.DEST_NAME_MODE_CONTENT_ONLY)
    picu.augment_metadata_for_dest(meta_b, dest_name_mode=picu.DEST_NAME_MODE_CONTENT_ONLY)
    assert meta_a["DestFileBase"] == meta_b["DestFileBase"]


def test_augment_metadata_mtime_fallback(tmp_path):
    path = tmp_path / "no_meta.jpg"
    path.write_bytes(b"x")
    metadata = {"SourceFile": str(path)}
    with patch.object(picu, "get_create_dt_from_metadata", return_value=None):
        picu.augment_metadata_for_dest(
            metadata,
            include_content_hash_in_dest=False,
            include_orig_name_in_dest=False,
        )
    assert metadata["CaptureDateSource"] == "file_mtime"
    assert metadata["CaptureTzSuffix"] == "+0000"
    assert "+0000" in metadata["DestFileBase"]


def _reset_content_fp_cache():
    picu._content_fp_cache = None
    picu._content_fp_cache_dirty = False


@pytest.fixture(autouse=True)
def _isolated_content_fp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(picu, "_CONTENT_FP_CACHE_DIR", str(tmp_path / "cache"))
    _reset_content_fp_cache()
    yield
    _reset_content_fp_cache()


def test_large_file_uses_sample_fingerprint(tmp_path):
    """Files above the threshold should not require a full-file read."""
    path = tmp_path / "big.mov"
    head = b"A" * 64
    tail = b"B" * 64
    path.write_bytes(head + b"middle" * 1000 + tail)

    digest, method = picu._file_content_fingerprint(
        str(path),
        content_hash_mode=picu.CONTENT_HASH_MODE_AUTO,
        full_max_bytes=256,
        sample_bytes=64,
    )
    assert method == picu.CONTENT_HASH_MODE_SAMPLE
    assert digest == picu._file_sample_fingerprint(str(path), sample_bytes=64)
    assert digest != picu._file_full_hash(str(path))


def test_sample_fingerprint_matches_identical_large_files(tmp_path):
    a = tmp_path / "a.mov"
    b = tmp_path / "b.mov"
    payload = b"X" * 128 + b"video-bytes" * 500 + b"Y" * 128
    a.write_bytes(payload)
    b.write_bytes(payload)
    assert picu._files_are_identical(
        str(a),
        str(b),
        content_hash_mode=picu.CONTENT_HASH_MODE_AUTO,
        full_max_bytes=64,
        sample_bytes=32,
    )


def test_content_fingerprint_cache_skips_recompute(tmp_path, monkeypatch):
    path = tmp_path / "cached.jpg"
    path.write_bytes(b"cache-me")
    calls = {"n": 0}
    real = picu._file_content_fingerprint

    def counting_fingerprint(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(picu, "_file_content_fingerprint", counting_fingerprint)
    picu._get_content_fingerprint(str(path))
    picu._get_content_fingerprint(str(path))
    assert calls["n"] == 1


def test_compute_preflight_counts(tmp_path):
    dest_root = tmp_path / "dest"
    month_dir = dest_root / "2020" / "01"
    month_dir.mkdir(parents=True)
    dest_file = month_dir / "20200101_120000.jpg"
    dest_file.write_bytes(b"dest")

    src_identical = tmp_path / "same.jpg"
    src_collision = tmp_path / "diff.jpg"
    src_identical.write_bytes(b"dest")
    src_collision.write_bytes(b"other")

    dest_base = "20200101_120000.jpg"
    group_key = ("2020", "01", dest_base)
    groups = {group_key: [str(src_identical), str(src_collision)]}
    meta = {
        str(src_identical): {"CaptureDateSource": "metadata"},
        str(src_collision): {"CaptureDateSource": "metadata"},
    }
    counts = picu._compute_preflight_counts(groups, str(dest_root), meta)
    assert counts["identical_at_dest"] == 1
    assert counts["collision"] == 1

    src_new = tmp_path / "new.jpg"
    src_new.write_bytes(b"fresh")
    transfer_base = "20200101_130000.jpg"
    counts2 = picu._compute_preflight_counts(
        {("2020", "01", transfer_base): [str(src_new)]},
        str(dest_root),
        {str(src_new): {"CaptureDateSource": "file_mtime"}},
    )
    assert counts2["transfer"] == 1
    assert counts2["mtime_fallback"] == 1


def _fake_metadata(path: str, dest_base: str, *, capture_source: str = "metadata"):
    return {
        "SourceFile": path,
        "DestFileBase": dest_base,
        "MetadataHash": "abc123",
        "CaptureDtUtc": "2020-01-01 12:00:00",
        "CaptureDateSource": capture_source,
        "DestYear": "2020",
        "DestMonth": "01",
        "ContentHash": picu._file_full_hash(path),
    }


def _apply_fake_augment_single(metadata, **kwargs):
    """Mock for augment_metadata_for_dest (per-file): fill in fields if missing."""
    if "MetadataHash" not in metadata:
        metadata["MetadataHash"] = "abc123"
    if "CaptureDtUtc" not in metadata:
        metadata["CaptureDtUtc"] = "2020-01-01 12:00:00"
    if "CaptureDateSource" not in metadata:
        metadata["CaptureDateSource"] = "metadata"
    if "DestYear" not in metadata:
        metadata["DestYear"] = "2020"
    if "DestMonth" not in metadata:
        metadata["DestMonth"] = "01"


@patch.object(picu, "augment_metadata_for_dest", side_effect=_apply_fake_augment_single)
@patch.object(picu, "get_all_media_file_paths")
@patch.object(picu, "get_metadatas_mproc")
def test_cleaup_processes_all_sources_in_group(
    mock_mproc, mock_scan, _mock_augment, tmp_path
):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    src_root.mkdir()
    dest_root.mkdir(parents=True)

    dest_base = "20200101_120000__hash1.jpg"
    dest_path = dest_root / "2020" / "01" / dest_base
    dest_path.parent.mkdir(parents=True)
    dest_path.write_bytes(b"library")

    src_a = src_root / "a.jpg"
    src_b = src_root / "b.jpg"
    src_a.write_bytes(b"library")
    src_b.write_bytes(b"library")

    mock_scan.return_value = [str(src_a), str(src_b)]
    mock_mproc.return_value = [
        _fake_metadata(str(src_a), dest_base),
        _fake_metadata(str(src_b), dest_base),
    ]

    trashed = []
    with patch.object(picu, "_trash_file", side_effect=lambda p: (trashed.append(p), os.remove(p))):
        rows = picu.cleaup_media_files(
            str(src_root),
            str(dest_root),
            dry_run=False,
            move_or_copy="move",
            progress=False,
        )
    assert len(rows) == 2
    assert all(r["error"] == "IDENTICAL_SOURCE_DELETED" for r in rows)
    assert not src_a.exists()
    assert not src_b.exists()


@patch.object(picu, "augment_metadata_for_dest", side_effect=_apply_fake_augment_single)
@patch.object(picu, "get_all_media_file_paths")
@patch.object(picu, "get_metadatas_mproc")
def test_cleaup_copy_mode_identical_error(mock_mproc, mock_scan, _mock_augment, tmp_path):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    src_root.mkdir()
    dest_root.mkdir(parents=True)

    dest_base = "20200101_120000.jpg"
    (dest_root / "2020" / "01").mkdir(parents=True)
    (dest_root / "2020" / "01" / dest_base).write_bytes(b"x")

    src = src_root / "x.jpg"
    src.write_bytes(b"x")
    mock_scan.return_value = [str(src)]
    mock_mproc.return_value = [_fake_metadata(str(src), dest_base)]

    rows = picu.cleaup_media_files(
        str(src_root),
        str(dest_root),
        dry_run=False,
        move_or_copy="copy",
        progress=False,
    )
    assert rows[0]["error"] == "IDENTICAL_SOURCE_KEPT"
    assert src.exists()


@patch.object(picu, "augment_metadata_for_dest", side_effect=_apply_fake_augment_single)
@patch.object(picu, "get_all_media_file_paths")
@patch.object(picu, "get_metadatas_mproc")
def test_cleaup_quarantine_collision(mock_mproc, mock_scan, _mock_augment, tmp_path):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    quarantine = tmp_path / "quarantine"
    src_root.mkdir()
    dest_root.mkdir(parents=True)
    quarantine.mkdir()

    dest_base = "20200101_120000.jpg"
    (dest_root / "2020" / "01").mkdir(parents=True)
    (dest_root / "2020" / "01" / dest_base).write_bytes(b"dest")

    src = src_root / "other.jpg"
    src.write_bytes(b"other")
    mock_scan.return_value = [str(src)]
    mock_mproc.return_value = [_fake_metadata(str(src), dest_base)]

    rows = picu.cleaup_media_files(
        str(src_root),
        str(dest_root),
        dry_run=False,
        move_or_copy="move",
        collisions_dir=str(quarantine),
        progress=False,
    )
    assert rows[0]["error"] == "COLLISION_MOVED_TO_QUARANTINE"
    assert not src.exists()
    assert "quarantine_reloc" in rows[0]
    assert os.path.exists(rows[0]["quarantine_reloc"])


@patch.object(picu, "augment_metadata_for_dest", side_effect=_apply_fake_augment_single)
@patch.object(picu, "get_all_media_file_paths")
@patch.object(picu, "get_metadatas_mproc")
def test_cleaup_log_includes_session_fields(mock_mproc, mock_scan, _mock_augment, tmp_path):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    src_root.mkdir()
    dest_root.mkdir()

    dest_base = "20200101_120000.jpg"
    src = src_root / "new.jpg"
    src.write_bytes(b"new")
    mock_scan.return_value = [str(src)]
    mock_mproc.return_value = [_fake_metadata(str(src), dest_base)]

    picu.cleaup_media_files(
        str(src_root),
        str(dest_root),
        dry_run=True,
        progress=False,
    )
    log_path = dest_root / "log.txt"
    row = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert "run_id" in row
    assert row["dry_run"] is True
    assert row["move_or_copy"] == "move"
