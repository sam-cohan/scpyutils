"""Tests for scpyutils.picutils robustness helpers and cleanup flow."""

import json
import os
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
    picu._safe_transfer(str(src), str(dest), is_move=False, verify=True)
    assert dest.read_bytes() == b"same-bytes"
    assert picu._files_are_identical(str(src), str(dest))


def test_safe_transfer_move_removes_source(tmp_path):
    src = tmp_path / "a.jpg"
    dest = tmp_path / "out" / "a.jpg"
    src.write_bytes(b"move-me")
    picu._safe_transfer(str(src), str(dest), is_move=True, verify=True)
    assert dest.read_bytes() == b"move-me"
    assert not src.exists()


def test_live_photo_companion_path(tmp_path):
    heic = tmp_path / "IMG_1234.heic"
    mov = tmp_path / "IMG_1234.MOV"
    heic.write_bytes(b"h")
    mov.write_bytes(b"m")
    assert os.path.samefile(picu._live_photo_companion_path(str(heic)), str(mov))
    assert os.path.samefile(picu._live_photo_companion_path(str(mov)), str(heic))


def test_augment_metadata_mtime_fallback(tmp_path):
    path = tmp_path / "no_meta.jpg"
    path.write_bytes(b"x")
    metadata = {"SourceFile": str(path)}
    with patch.object(picu, "get_create_dt_from_metadata", return_value=None):
        picu.augment_metadata_for_dest(
            metadata,
            include_metadata_hash_in_dest=False,
            include_orig_name_in_dest=False,
        )
    assert metadata["CaptureDateSource"] == "file_mtime"
    assert metadata["DestFileBase"].endswith(".jpg")


def test_compute_preflight_counts(tmp_path):
    dest_root = tmp_path / "dest"
    year_dir = dest_root / "2020"
    year_dir.mkdir(parents=True)
    dest_file = year_dir / "20200101_120000.jpg"
    dest_file.write_bytes(b"dest")

    src_identical = tmp_path / "same.jpg"
    src_collision = tmp_path / "diff.jpg"
    src_identical.write_bytes(b"dest")
    src_collision.write_bytes(b"other")

    dest_base = "20200101_120000.jpg"
    groups = {dest_base: [str(src_identical), str(src_collision)]}
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
        {transfer_base: [str(src_new)]},
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
        "ContentSha256": picu._file_sha256(path),
    }


def _apply_fake_dest_metadatas(metadatas, **kwargs):
    """Keep DestFileBase from mocks; only add fields augment would set."""
    for metadata in metadatas:
        if "MetadataHash" not in metadata:
            metadata["MetadataHash"] = "abc123"
        if "CaptureDtUtc" not in metadata:
            metadata["CaptureDtUtc"] = "2020-01-01 12:00:00"
        if "CaptureDateSource" not in metadata:
            metadata["CaptureDateSource"] = "metadata"
    return metadatas


@patch.object(picu, "augment_metadatas_for_dest", side_effect=_apply_fake_dest_metadatas)
@patch.object(picu, "get_all_media_file_paths")
@patch.object(picu, "get_metadatas_mproc")
def test_cleaup_processes_all_sources_in_group(
    mock_mproc, mock_scan, _mock_augment, tmp_path
):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    dup_dir = tmp_path / "dups"
    src_root.mkdir()
    dest_root.mkdir(parents=True)
    dup_dir.mkdir()

    dest_base = "20200101_120000__hash1.jpg"
    year_dir = dest_root / "2020"
    year_dir.mkdir()
    dest_path = year_dir / dest_base
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

    rows = picu.cleaup_media_files(
        str(src_root),
        str(dest_root),
        dry_run=False,
        move_or_copy="move",
        identical_duplicates_dir=str(dup_dir),
        progress=False,
    )
    assert len(rows) == 2
    assert all(r["error"] == "IDENTICAL_MOVED_TO_DUPLICATES" for r in rows)
    assert not src_a.exists()
    assert not src_b.exists()


@patch.object(picu, "augment_metadatas_for_dest", side_effect=_apply_fake_dest_metadatas)
@patch.object(picu, "get_all_media_file_paths")
@patch.object(picu, "get_metadatas_mproc")
def test_cleaup_copy_mode_identical_error(mock_mproc, mock_scan, _mock_augment, tmp_path):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    src_root.mkdir()
    dest_root.mkdir(parents=True)

    dest_base = "20200101_120000.jpg"
    (dest_root / "2020").mkdir()
    (dest_root / "2020" / dest_base).write_bytes(b"x")

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
    assert rows[0]["error"] == "IDENTICAL_DESTINATION_EXISTS"
    assert src.exists()


@patch.object(picu, "augment_metadatas_for_dest", side_effect=_apply_fake_dest_metadatas)
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
    (dest_root / "2020").mkdir()
    (dest_root / "2020" / dest_base).write_bytes(b"dest")

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


@patch.object(picu, "augment_metadatas_for_dest", side_effect=_apply_fake_dest_metadatas)
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
