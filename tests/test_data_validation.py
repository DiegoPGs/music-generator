"""Tests for src/data_validation.py."""

import os

import numpy as np
import pandas as pd
import pretty_midi
import pytest

from src.data_validation import (
    build_stats_dataframe,
    extract_midi_stats,
    flag_outliers,
    load_metadata,
    validate_splits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_midi(path: str, notes: list) -> None:
    """Write a PrettyMIDI file from a list of (pitch, velocity, start, end) tuples."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for pitch, velocity, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(path)


def make_metadata_csv(path: str, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


_BASE_ROW = {
    "canonical_composer": "Bach",
    "canonical_title": "Fugue",
    "split": "train",
    "year": 2004,
    "audio_filename": "2004/test.wav",
    "duration": 120.0,
}


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_valid_csv(self, tmp_path):
        csv = str(tmp_path / "meta.csv")
        make_metadata_csv(csv, [{**_BASE_ROW, "midi_filename": "2004/test.midi"}])
        df = load_metadata(csv)
        assert len(df) == 1
        assert set(["split", "midi_filename", "duration"]).issubset(df.columns)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_metadata("/nonexistent/path.csv")

    def test_missing_column_raises(self, tmp_path):
        csv = str(tmp_path / "bad.csv")
        pd.DataFrame([{"foo": "bar"}]).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_metadata(csv)

    def test_multiple_splits(self, tmp_path):
        csv = str(tmp_path / "multi.csv")
        rows = [
            {**_BASE_ROW, "midi_filename": f"{i}.midi", "split": s}
            for i, s in enumerate(["train", "train", "validation", "test"])
        ]
        make_metadata_csv(csv, rows)
        df = load_metadata(csv)
        assert df["split"].nunique() == 3


# ---------------------------------------------------------------------------
# extract_midi_stats
# ---------------------------------------------------------------------------


class TestExtractMidiStats:
    def test_basic_stats(self, tmp_path):
        path = str(tmp_path / "basic.midi")
        make_midi(path, [(60 + i, 64, i * 0.5, i * 0.5 + 0.4) for i in range(10)])
        stats = extract_midi_stats(path)
        assert stats["parse_error"] is False
        assert stats["note_count"] == 10
        assert stats["duration"] > 0
        assert stats["notes_per_second"] > 0
        assert 0 <= stats["min_pitch"] <= 127
        assert stats["min_pitch"] <= stats["max_pitch"]

    def test_pitch_range(self, tmp_path):
        path = str(tmp_path / "range.midi")
        make_midi(path, [(40, 64, 0.0, 0.5), (80, 64, 0.5, 1.0)])
        stats = extract_midi_stats(path)
        assert stats["min_pitch"] == 40
        assert stats["max_pitch"] == 80
        assert stats["pitch_range"] == 40

    def test_velocity_stats(self, tmp_path):
        path = str(tmp_path / "vel.midi")
        make_midi(path, [(60, 40, 0.0, 0.5), (62, 80, 0.5, 1.0)])
        stats = extract_midi_stats(path)
        assert stats["mean_velocity"] == pytest.approx(60.0)
        assert stats["std_velocity"] > 0

    def test_nonexistent_file_returns_error(self):
        stats = extract_midi_stats("/nonexistent/file.midi")
        assert stats["parse_error"] is True
        assert stats["note_count"] is None

    def test_empty_midi_flagged(self, tmp_path):
        path = str(tmp_path / "empty.midi")
        pm = pretty_midi.PrettyMIDI()
        pm.write(path)
        stats = extract_midi_stats(path)
        assert stats["parse_error"] is False
        assert stats["note_count"] == 0

    def test_notes_per_second(self, tmp_path):
        path = str(tmp_path / "nps.midi")
        # 10 notes over ~5 seconds → ~2 nps
        make_midi(path, [(60, 64, i * 0.5, i * 0.5 + 0.4) for i in range(10)])
        stats = extract_midi_stats(path)
        assert stats["notes_per_second"] == pytest.approx(stats["note_count"] / stats["duration"])


# ---------------------------------------------------------------------------
# build_stats_dataframe
# ---------------------------------------------------------------------------


class TestBuildStatsDataframe:
    def test_shape_and_columns(self, tmp_path):
        midi_path = str(tmp_path / "a.midi")
        make_midi(midi_path, [(60, 64, 0.0, 0.5)])
        csv = str(tmp_path / "meta.csv")
        rows = [{**_BASE_ROW, "midi_filename": "a.midi"}]
        make_metadata_csv(csv, rows)
        df = load_metadata(csv)
        stats_df = build_stats_dataframe(df, str(tmp_path))
        assert len(stats_df) == 1
        assert "duration_computed" in stats_df.columns
        assert "note_count" in stats_df.columns
        assert "parse_error" in stats_df.columns

    def test_missing_file_sets_none(self, tmp_path):
        csv = str(tmp_path / "meta.csv")
        rows = [{**_BASE_ROW, "midi_filename": "does_not_exist.midi"}]
        make_metadata_csv(csv, rows)
        df = load_metadata(csv)
        stats_df = build_stats_dataframe(df, str(tmp_path))
        assert stats_df.iloc[0]["note_count"] is None


# ---------------------------------------------------------------------------
# flag_outliers
# ---------------------------------------------------------------------------


def _make_stats_df(n: int = 20, **overrides) -> pd.DataFrame:
    """Create a uniform stats DataFrame for outlier tests."""
    base = {
        "duration_computed": [120.0] * n,
        "notes_per_second": [5.0] * n,
        "pitch_range": [60] * n,
        "note_count": [600] * n,
        "parse_error": [False] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestFlagOutliers:
    def test_uniform_data_no_outliers(self):
        df = _make_stats_df(20)
        result = flag_outliers(df)
        assert result["is_outlier"].sum() == 0

    def test_extreme_duration_flagged(self):
        durations = [120.0] * 19 + [0.001]
        df = _make_stats_df(20, duration_computed=durations)
        result = flag_outliers(df)
        assert result.iloc[-1]["is_outlier"]
        assert "duration" in result.iloc[-1]["outlier_reasons"]

    def test_parse_error_always_flagged(self):
        df = pd.DataFrame({
            "duration_computed": [None],
            "notes_per_second": [None],
            "pitch_range": [None],
            "note_count": [None],
            "parse_error": [True],
        })
        result = flag_outliers(df)
        assert result.iloc[0]["is_outlier"]
        assert "parse_error" in result.iloc[0]["outlier_reasons"]

    def test_empty_file_flagged(self):
        df = _make_stats_df(5, note_count=[100, 100, 100, 100, 0])
        result = flag_outliers(df)
        assert result.iloc[-1]["is_outlier"]
        assert "no_notes" in result.iloc[-1]["outlier_reasons"]

    def test_outlier_reasons_nonempty_for_flagged(self):
        durations = [120.0] * 19 + [0.001]
        df = _make_stats_df(20, duration_computed=durations)
        result = flag_outliers(df)
        flagged = result[result["is_outlier"]]
        assert all(len(r) > 0 for r in flagged["outlier_reasons"])


# ---------------------------------------------------------------------------
# validate_splits
# ---------------------------------------------------------------------------


class TestValidateSplits:
    def test_counts_and_proportions(self):
        df = pd.DataFrame({"split": ["train"] * 75 + ["validation"] * 10 + ["test"] * 15})
        result = validate_splits(df)
        assert result["total"] == 100
        assert result["counts"]["train"] == 75
        assert result["proportions"]["train"] == pytest.approx(0.75)

    def test_returns_all_expected_keys(self):
        df = pd.DataFrame({"split": ["train", "test", "validation"]})
        result = validate_splits(df)
        assert {"counts", "proportions", "total"} == set(result.keys())
