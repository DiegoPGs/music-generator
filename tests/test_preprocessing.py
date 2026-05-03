"""Tests for src/preprocessing.py."""

import os

import numpy as np
import pandas as pd
import pretty_midi
import pytest

import config
from src.preprocessing import (build_vocabulary, dequantize_velocity,
                               events_to_midi, events_to_tokens,
                               load_vocabulary, midi_to_events,
                               process_dataset, quantize_velocity,
                               save_vocabulary, tokens_to_events)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_midi(path: str, notes: list, tempo: float = 120.0) -> None:
    """Write a PrettyMIDI file from (pitch, velocity, start, end) tuples."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    for pitch, velocity, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(path)


def make_csv(path: str, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


_BASE_ROW = {
    "canonical_composer": "A",
    "canonical_title": "T",
    "year": 2004,
    "audio_filename": "2004/x.wav",
    "duration": 1.0,
}


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TestBuildVocabulary:
    def test_size(self):
        assert len(build_vocabulary()) == config.VOCAB_SIZE

    def test_ids_contiguous_from_zero(self):
        ids = sorted(build_vocabulary().values())
        assert ids == list(range(config.VOCAB_SIZE))

    def test_all_token_types_present(self):
        vocab = build_vocabulary()
        assert "NOTE_ON_0" in vocab and "NOTE_ON_127" in vocab
        assert "NOTE_OFF_0" in vocab and "NOTE_OFF_127" in vocab
        assert "TIME_SHIFT_1" in vocab and f"TIME_SHIFT_{config.MAX_TIME_SHIFT_BINS}" in vocab
        assert "SET_VELOCITY_0" in vocab and f"SET_VELOCITY_{config.VELOCITY_BINS - 1}" in vocab

    def test_layout_order(self):
        vocab = build_vocabulary()
        assert vocab["NOTE_ON_0"] == 0
        assert vocab["NOTE_OFF_0"] == 128
        assert vocab["TIME_SHIFT_1"] == 256
        assert vocab["SET_VELOCITY_0"] == 356


class TestSaveLoadVocabulary:
    def test_roundtrip(self, tmp_path):
        vocab = build_vocabulary()
        path = str(tmp_path / "vocab.json")
        save_vocabulary(vocab, path)
        loaded, inv = load_vocabulary(path)
        assert loaded == vocab
        assert len(inv) == config.VOCAB_SIZE
        assert inv[0] == "NOTE_ON_0"

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_vocabulary("/nonexistent/vocab.json")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


class TestQuantizeVelocity:
    def test_zero_maps_to_bin_zero(self):
        assert quantize_velocity(0) == 0

    def test_max_maps_to_last_bin(self):
        assert quantize_velocity(127) == config.VELOCITY_BINS - 1

    def test_output_always_in_range(self):
        for v in range(128):
            assert 0 <= quantize_velocity(v) < config.VELOCITY_BINS


class TestDequantizeVelocity:
    def test_output_valid_midi_velocity(self):
        for b in range(config.VELOCITY_BINS):
            assert 1 <= dequantize_velocity(b) <= 127

    def test_monotonically_increasing(self):
        vals = [dequantize_velocity(b) for b in range(config.VELOCITY_BINS)]
        assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# midi_to_events
# ---------------------------------------------------------------------------


class TestMidiToEvents:
    def test_returns_strings(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 64, 0.0, 0.5)])
        events = midi_to_events(path)
        assert isinstance(events, list)
        assert all(isinstance(e, str) for e in events)

    def test_single_note_has_expected_tokens(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 64, 0.0, 0.5)])
        events = midi_to_events(path)
        assert "NOTE_ON_60" in events
        assert "NOTE_OFF_60" in events
        assert any(e.startswith("SET_VELOCITY_") for e in events)

    def test_all_tokens_in_vocabulary(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 80, 0.0, 0.5), (64, 80, 0.1, 0.6)])
        vocab = build_vocabulary()
        for token in midi_to_events(path):
            assert token in vocab, f"Unknown token: {token}"

    def test_time_shift_between_notes(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 64, 0.0, 0.1), (62, 64, 0.6, 0.8)])
        events = midi_to_events(path)
        off_idx = next(i for i, e in enumerate(events) if e == "NOTE_OFF_60")
        on_idx = next(i for i, e in enumerate(events) if e == "NOTE_ON_62")
        between = events[off_idx + 1:on_idx]
        assert all(e.startswith("TIME_SHIFT_") or e.startswith("SET_VELOCITY_") for e in between)
        assert any(e.startswith("TIME_SHIFT_") for e in between)

    def test_long_gap_chained_time_shifts(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 64, 0.0, 0.1), (62, 64, 2.6, 2.8)])
        events = midi_to_events(path)
        off_idx = next(i for i, e in enumerate(events) if e == "NOTE_OFF_60")
        on_idx = next(i for i, e in enumerate(events) if e == "NOTE_ON_62")
        ts_tokens = [e for e in events[off_idx + 1:on_idx] if e.startswith("TIME_SHIFT_")]
        total_bins = sum(int(e.split("_")[2]) for e in ts_tokens)
        assert total_bins >= 240  # ~2.5 s at 10 ms/bin
        # No single TIME_SHIFT exceeds the max bin value
        assert all(int(e.split("_")[2]) <= config.MAX_TIME_SHIFT_BINS for e in ts_tokens)

    def test_polyphonic_three_notes(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 64, 0.0, 0.5), (64, 64, 0.0, 0.5), (67, 64, 0.0, 0.5)])
        events = midi_to_events(path)
        assert len([e for e in events if e.startswith("NOTE_ON_")]) == 3

    def test_velocity_change_emits_set_velocity(self, tmp_path):
        path = str(tmp_path / "t.midi")
        # velocities 10 and 120 land in very different bins
        make_midi(path, [(60, 10, 0.0, 0.4), (62, 120, 0.5, 0.9)])
        events = midi_to_events(path)
        assert len([e for e in events if e.startswith("SET_VELOCITY_")]) >= 2

    def test_same_velocity_single_set_velocity(self, tmp_path):
        path = str(tmp_path / "t.midi")
        make_midi(path, [(60, 64, 0.0, 0.4), (62, 64, 0.5, 0.9), (64, 64, 1.0, 1.4)])
        events = midi_to_events(path)
        assert len([e for e in events if e.startswith("SET_VELOCITY_")]) == 1

    def test_sustain_pedal_extends_note(self, tmp_path):
        path = str(tmp_path / "pedal.midi")
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=64, pitch=60, start=0.0, end=0.5))
        inst.control_changes.append(pretty_midi.ControlChange(64, 100, 0.2))  # pedal down
        inst.control_changes.append(pretty_midi.ControlChange(64, 0, 1.0))    # pedal up
        pm.instruments.append(inst)
        pm.write(path)

        events = midi_to_events(path)
        # Reconstruct NOTE_OFF time from the token stream
        t = 0.0
        note_off_time = None
        for ev in events:
            if ev.startswith("TIME_SHIFT_"):
                t += int(ev.split("_")[2]) * config.TIME_SHIFT_MS_PER_BIN / 1000.0
            elif ev == "NOTE_OFF_60":
                note_off_time = t
                break
        assert note_off_time is not None
        assert note_off_time > 0.7  # extended past the original 0.5 s end


# ---------------------------------------------------------------------------
# events_to_tokens / tokens_to_events
# ---------------------------------------------------------------------------


class TestTokenRoundtrip:
    def test_roundtrip(self):
        vocab = build_vocabulary()
        inv = {v: k for k, v in vocab.items()}
        events = ["SET_VELOCITY_10", "NOTE_ON_60", "TIME_SHIFT_25", "NOTE_OFF_60"]
        assert tokens_to_events(events_to_tokens(events, vocab), inv) == events

    def test_dtype_is_int32(self):
        vocab = build_vocabulary()
        assert events_to_tokens(["NOTE_ON_60"], vocab).dtype == np.int32

    def test_unknown_token_raises(self):
        with pytest.raises(KeyError):
            events_to_tokens(["UNKNOWN_TOKEN"], build_vocabulary())


# ---------------------------------------------------------------------------
# events_to_midi
# ---------------------------------------------------------------------------


class TestEventsToMidi:
    def test_single_note_roundtrip(self, tmp_path):
        src, dst = str(tmp_path / "src.midi"), str(tmp_path / "dst.midi")
        make_midi(src, [(60, 64, 0.0, 0.5)])
        events_to_midi(midi_to_events(src), dst)
        notes = [n for inst in pretty_midi.PrettyMIDI(dst).instruments for n in inst.notes]
        assert len(notes) == 1
        assert notes[0].pitch == 60

    def test_polyphonic_note_count_preserved(self, tmp_path):
        src, dst = str(tmp_path / "src.midi"), str(tmp_path / "dst.midi")
        make_midi(src, [(60, 64, 0.0, 0.5), (64, 64, 0.0, 0.5), (67, 64, 0.0, 0.5)])
        events_to_midi(midi_to_events(src), dst)
        notes = [n for inst in pretty_midi.PrettyMIDI(dst).instruments for n in inst.notes]
        assert len(notes) == 3

    def test_pitches_preserved(self, tmp_path):
        src, dst = str(tmp_path / "src.midi"), str(tmp_path / "dst.midi")
        pitches_in = [40, 50, 60, 70, 80]
        make_midi(src, [(p, 64, i * 0.5, i * 0.5 + 0.4) for i, p in enumerate(pitches_in)])
        events_to_midi(midi_to_events(src), dst)
        pitches_out = sorted(n.pitch for inst in pretty_midi.PrettyMIDI(dst).instruments for n in inst.notes)
        assert pitches_out == sorted(pitches_in)

    def test_timing_approximately_preserved(self, tmp_path):
        src, dst = str(tmp_path / "src.midi"), str(tmp_path / "dst.midi")
        make_midi(src, [(60, 64, 0.0, 0.5), (62, 64, 1.0, 1.5)])
        events_to_midi(midi_to_events(src), dst)
        notes = sorted(
            [n for inst in pretty_midi.PrettyMIDI(dst).instruments for n in inst.notes],
            key=lambda n: n.start,
        )
        assert notes[1].start == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# process_dataset
# ---------------------------------------------------------------------------


class TestProcessDataset:
    def _setup(self, tmp_path, filenames: list) -> tuple:
        """Create synthetic MIDI files and a matching CSV."""
        midi_root = tmp_path / "maestro"
        year_dir = midi_root / "2004"
        year_dir.mkdir(parents=True)
        for fname in filenames:
            make_midi(str(year_dir / fname), [(60, 64, 0.0, 0.5)])

        csv_path = str(tmp_path / "meta.csv")
        rows = [
            {**_BASE_ROW, "midi_filename": f"2004/{f}", "split": "train"}
            for f in filenames
        ]
        make_csv(csv_path, rows)
        return str(midi_root), csv_path

    def test_creates_vocab_and_npy_files(self, tmp_path):
        midi_root, csv_path = self._setup(tmp_path, ["a.midi", "b.midi"])
        vocab_path = str(tmp_path / "vocab.json")
        out_dir = str(tmp_path / "sequences")

        process_dataset(
            csv_path=csv_path, base_dir=midi_root,
            vocab_path=vocab_path, output_dir=out_dir, splits=["train"],
        )

        assert os.path.exists(vocab_path)
        assert os.path.exists(os.path.join(out_dir, "2004", "a.npy"))
        assert os.path.exists(os.path.join(out_dir, "2004", "b.npy"))

    def test_output_is_valid_int32_within_vocab(self, tmp_path):
        midi_root, csv_path = self._setup(tmp_path, ["c.midi"])
        out_dir = str(tmp_path / "sequences")

        process_dataset(
            csv_path=csv_path, base_dir=midi_root,
            vocab_path=str(tmp_path / "vocab.json"),
            output_dir=out_dir, splits=["train"],
        )

        arr = np.load(os.path.join(out_dir, "2004", "c.npy"))
        assert arr.dtype == np.int32
        assert len(arr) > 0
        assert np.all((arr >= 0) & (arr < config.VOCAB_SIZE))

    def test_split_filtering(self, tmp_path):
        midi_root = tmp_path / "maestro" / "2004"
        midi_root.mkdir(parents=True)
        make_midi(str(midi_root / "train.midi"), [(60, 64, 0.0, 0.5)])
        make_midi(str(midi_root / "test.midi"), [(62, 64, 0.0, 0.5)])

        csv_path = str(tmp_path / "meta.csv")
        make_csv(csv_path, [
            {**_BASE_ROW, "midi_filename": "2004/train.midi", "split": "train"},
            {**_BASE_ROW, "midi_filename": "2004/test.midi", "split": "test"},
        ])
        out_dir = str(tmp_path / "sequences")

        process_dataset(
            csv_path=csv_path, base_dir=str(tmp_path / "maestro"),
            vocab_path=str(tmp_path / "vocab.json"),
            output_dir=out_dir, splits=["train"],
        )

        assert os.path.exists(os.path.join(out_dir, "2004", "train.npy"))
        assert not os.path.exists(os.path.join(out_dir, "2004", "test.npy"))
