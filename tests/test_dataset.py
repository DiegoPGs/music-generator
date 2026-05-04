"""Tests for src/dataset.py."""

import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import config
from src.dataset import build_dataset, get_sequence_paths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ROW = {
    "canonical_composer": "A",
    "canonical_title": "T",
    "year": 2004,
    "audio_filename": "2004/x.wav",
    "duration": 1.0,
}


def make_npy(path: str, length: int, vocab_size: int = config.VOCAB_SIZE) -> np.ndarray:
    """Write a .npy file with random token IDs of given length."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.randint(0, vocab_size, size=length, dtype=np.int32)
    np.save(path, arr)
    return arr


def make_npy_sequence(path: str, sequence: list) -> None:
    """Write a .npy file from an explicit token list."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(sequence, dtype=np.int32))


def make_csv(path: str, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# get_sequence_paths
# ---------------------------------------------------------------------------


class TestGetSequencePaths:
    def test_returns_existing_paths(self, tmp_path):
        seq_dir = tmp_path / "sequences" / "2004"
        seq_dir.mkdir(parents=True)
        (seq_dir / "a.npy").touch()
        (seq_dir / "b.npy").touch()

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": "2004/a.midi", "split": "train"},
            {**_BASE_ROW, "midi_filename": "2004/b.midi", "split": "train"},
            {**_BASE_ROW, "midi_filename": "2004/c.midi", "split": "validation"},
        ])

        paths = get_sequence_paths("train", csv_path=csv_path, sequences_dir=str(tmp_path / "sequences"))
        assert len(paths) == 2
        assert all(p.endswith(".npy") for p in paths)

    def test_filters_by_split(self, tmp_path):
        seq_dir = tmp_path / "sequences" / "2004"
        seq_dir.mkdir(parents=True)
        (seq_dir / "train.npy").touch()
        (seq_dir / "val.npy").touch()

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": "2004/train.midi", "split": "train"},
            {**_BASE_ROW, "midi_filename": "2004/val.midi", "split": "validation"},
        ])

        train_paths = get_sequence_paths("train", csv_path=csv_path, sequences_dir=str(tmp_path / "sequences"))
        val_paths = get_sequence_paths("validation", csv_path=csv_path, sequences_dir=str(tmp_path / "sequences"))
        assert len(train_paths) == 1
        assert len(val_paths) == 1

    def test_missing_npy_excluded_with_warning(self, tmp_path, caplog):
        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": "2004/missing.midi", "split": "train"},
        ])

        paths = get_sequence_paths("train", csv_path=csv_path, sequences_dir=str(tmp_path / "sequences"))
        assert len(paths) == 0

    def test_returns_sorted_paths(self, tmp_path):
        seq_dir = tmp_path / "sequences" / "2004"
        seq_dir.mkdir(parents=True)
        for name in ["z.npy", "a.npy", "m.npy"]:
            (seq_dir / name).touch()

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": f"2004/{n}.midi", "split": "train"}
            for n in ["z", "a", "m"]
        ])

        paths = get_sequence_paths("train", csv_path=csv_path, sequences_dir=str(tmp_path / "sequences"))
        assert paths == sorted(paths)

    def test_max_files_truncates_result(self, tmp_path):
        seq_dir = tmp_path / "sequences" / "2004"
        seq_dir.mkdir(parents=True)
        for name in ["a.npy", "b.npy", "c.npy", "d.npy", "e.npy"]:
            (seq_dir / name).touch()

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": f"2004/{n}.midi", "split": "train"}
            for n in ["a", "b", "c", "d", "e"]
        ])

        paths = get_sequence_paths(
            "train",
            csv_path=csv_path,
            sequences_dir=str(tmp_path / "sequences"),
            max_files=2,
        )
        assert len(paths) == 2
        # Must remain sorted: 'a' and 'b' come before 'c', 'd', 'e'
        assert paths == sorted(paths)

    def test_max_files_none_returns_all(self, tmp_path):
        seq_dir = tmp_path / "sequences" / "2004"
        seq_dir.mkdir(parents=True)
        for name in ["a.npy", "b.npy", "c.npy"]:
            (seq_dir / name).touch()

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": f"2004/{n}.midi", "split": "train"}
            for n in ["a", "b", "c"]
        ])

        paths = get_sequence_paths(
            "train",
            csv_path=csv_path,
            sequences_dir=str(tmp_path / "sequences"),
            max_files=None,
        )
        assert len(paths) == 3

    def test_max_files_larger_than_available(self, tmp_path):
        seq_dir = tmp_path / "sequences" / "2004"
        seq_dir.mkdir(parents=True)
        for name in ["a.npy", "b.npy"]:
            (seq_dir / name).touch()

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": f"2004/{n}.midi", "split": "train"}
            for n in ["a", "b"]
        ])

        paths = get_sequence_paths(
            "train",
            csv_path=csv_path,
            sequences_dir=str(tmp_path / "sequences"),
            max_files=10,
        )
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


class TestBuildDataset:
    SEQ_LEN = 8
    BATCH = 4

    def _make_paths(self, tmp_path, n_files: int = 3, seq_length: int = 200) -> list:
        paths = []
        for i in range(n_files):
            p = str(tmp_path / f"seq_{i}.npy")
            make_npy(p, seq_length)
            paths.append(p)
        return paths

    def test_raises_on_empty_paths(self):
        with pytest.raises(ValueError, match="empty"):
            build_dataset([], seq_len=self.SEQ_LEN, batch_size=self.BATCH)

    def test_output_shapes(self, tmp_path):
        paths = self._make_paths(tmp_path)
        ds = build_dataset(paths, seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        inputs, targets = next(iter(ds))
        assert inputs.shape == (self.BATCH, self.SEQ_LEN)
        assert targets.shape == (self.BATCH, self.SEQ_LEN)

    def test_output_dtype(self, tmp_path):
        paths = self._make_paths(tmp_path)
        ds = build_dataset(paths, seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        inputs, targets = next(iter(ds))
        assert inputs.dtype == tf.int32
        assert targets.dtype == tf.int32

    def test_target_is_input_shifted_by_one(self, tmp_path):
        # Use a known strictly increasing sequence so we can verify the shift
        path = str(tmp_path / "known.npy")
        make_npy_sequence(path, list(range(50)))
        ds = build_dataset([path], seq_len=self.SEQ_LEN, batch_size=1, shuffle=False, window_shift=self.SEQ_LEN)
        for inputs, targets in ds:
            inp = inputs.numpy()[0]
            tgt = targets.numpy()[0]
            # Every target token should be the next token in the original sequence
            assert np.all(tgt == inp + 1)

    def test_token_ids_within_vocab(self, tmp_path):
        paths = self._make_paths(tmp_path)
        ds = build_dataset(paths, seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        for inputs, targets in ds.take(5):
            assert tf.reduce_all(inputs >= 0)
            assert tf.reduce_all(inputs < config.VOCAB_SIZE)
            assert tf.reduce_all(targets >= 0)
            assert tf.reduce_all(targets < config.VOCAB_SIZE)

    def test_short_file_skipped_gracefully(self, tmp_path):
        # A file shorter than seq_len+1 should contribute 0 windows without crashing
        short_path = str(tmp_path / "short.npy")
        long_path = str(tmp_path / "long.npy")
        make_npy_sequence(short_path, list(range(self.SEQ_LEN - 1)))  # too short
        make_npy(long_path, 200)
        ds = build_dataset([short_path, long_path], seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        batches = list(ds.take(3))
        assert len(batches) > 0  # long file still contributes samples

    def test_window_shift_affects_sample_count(self, tmp_path):
        path = str(tmp_path / "seq.npy")
        make_npy(path, 500)
        ds_shift1 = build_dataset([path], seq_len=self.SEQ_LEN, batch_size=1, shuffle=False, window_shift=1)
        ds_shift8 = build_dataset([path], seq_len=self.SEQ_LEN, batch_size=1, shuffle=False, window_shift=self.SEQ_LEN)
        count_shift1 = sum(1 for _ in ds_shift1)
        count_shift8 = sum(1 for _ in ds_shift8)
        assert count_shift1 > count_shift8

    def test_shuffle_false_is_deterministic(self, tmp_path):
        paths = self._make_paths(tmp_path, n_files=2)
        ds1 = build_dataset(paths, seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        ds2 = build_dataset(paths, seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        for (i1, _), (i2, _) in zip(ds1.take(5), ds2.take(5)):
            assert np.array_equal(i1.numpy(), i2.numpy())

    def test_batch_drop_remainder(self, tmp_path):
        # With drop_remainder=True, all batches must be exactly batch_size
        paths = self._make_paths(tmp_path, n_files=2, seq_length=300)
        ds = build_dataset(paths, seq_len=self.SEQ_LEN, batch_size=self.BATCH, shuffle=False)
        for inputs, _ in ds:
            assert inputs.shape[0] == self.BATCH
