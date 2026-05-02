"""Tests for src/generate.py."""

import os

import numpy as np
import pretty_midi
import pytest
import tensorflow as tf

import config
from src.generate import (generate, generate_midi, load_model_and_vocab,
                          sample_token, seed_from_midi, seed_from_random)
from src.model import build_model
from src.preprocessing import build_vocabulary, save_vocabulary
from src.train import train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB = 20
_SEQ = 8


def _small_model() -> tf.keras.Model:
    return build_model(vocab_size=_VOCAB, embed_dim=8, lstm_units=16, num_lstm_layers=1, dropout_rate=0.0)


def _inv_vocab(size: int = _VOCAB) -> dict:
    return {i: f"NOTE_ON_{i}" for i in range(size)}


def _save_model(model: tf.keras.Model, path: str) -> str:
    """Compile, do one dummy pass to build weights, then save."""
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model(tf.constant([[0, 1, 2, 3]], dtype=tf.int32), training=False)
    model.save(path)
    return path


def _make_midi(path: str) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i in range(5):
        inst.notes.append(pretty_midi.Note(velocity=64, pitch=i, start=i * 0.5, end=i * 0.5 + 0.4))
    pm.instruments.append(inst)
    pm.write(path)


# ---------------------------------------------------------------------------
# sample_token
# ---------------------------------------------------------------------------


class TestSampleToken:
    def test_returns_valid_index(self):
        probs = np.array([0.1, 0.5, 0.2, 0.2])
        idx = sample_token(probs, temperature=1.0)
        assert 0 <= idx < len(probs)

    def test_temperature_zero_is_greedy(self):
        probs = np.array([0.1, 0.05, 0.8, 0.05])
        assert sample_token(probs, temperature=0.0) == 2

    def test_temperature_zero_always_same(self):
        probs = np.array([0.0, 0.0, 1.0, 0.0])
        results = {sample_token(probs, 0.0) for _ in range(20)}
        assert results == {2}

    def test_high_temperature_samples_broadly(self):
        """At very high temperature all tokens should be sampled eventually."""
        probs = np.full(8, 1 / 8)
        seen = set()
        for _ in range(200):
            seen.add(sample_token(probs, temperature=10.0))
        assert len(seen) > 4

    def test_low_temperature_concentrates_on_top_token(self):
        """At low temperature the highest-probability token should dominate."""
        probs = np.array([0.01, 0.01, 0.96, 0.01, 0.01])
        counts = np.zeros(5, dtype=int)
        for _ in range(100):
            counts[sample_token(probs, temperature=0.1)] += 1
        assert counts[2] > 90

    def test_uniform_probs_samples_all_indices(self):
        probs = np.ones(10) / 10
        seen = {sample_token(probs, 1.0) for _ in range(500)}
        assert len(seen) == 10


# ---------------------------------------------------------------------------
# seed_from_random
# ---------------------------------------------------------------------------


class TestSeedFromRandom:
    def test_length(self):
        assert len(seed_from_random(100, seq_len=16)) == 16

    def test_ids_in_range(self):
        ids = seed_from_random(50, seq_len=32)
        assert all(0 <= i < 50 for i in ids)

    def test_reproducible_with_seed(self):
        a = seed_from_random(100, seq_len=10, rng_seed=42)
        b = seed_from_random(100, seq_len=10, rng_seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = seed_from_random(100, seq_len=10, rng_seed=1)
        b = seed_from_random(100, seq_len=10, rng_seed=2)
        assert a != b


# ---------------------------------------------------------------------------
# seed_from_midi
# ---------------------------------------------------------------------------


class TestSeedFromMidi:
    def test_returns_list_of_ints(self, tmp_path):
        path = str(tmp_path / "seed.midi")
        _make_midi(path)
        vocab = build_vocabulary()
        result = seed_from_midi(path, vocab, seq_len=8)
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_length_capped_at_seq_len(self, tmp_path):
        path = str(tmp_path / "seed.midi")
        _make_midi(path)
        vocab = build_vocabulary()
        result = seed_from_midi(path, vocab, seq_len=4)
        assert len(result) <= 4

    def test_ids_within_vocab(self, tmp_path):
        path = str(tmp_path / "seed.midi")
        _make_midi(path)
        vocab = build_vocabulary()
        result = seed_from_midi(path, vocab, seq_len=32)
        assert all(0 <= i < config.VOCAB_SIZE for i in result)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_output_length_equals_seed_plus_generated(self):
        model = _small_model()
        seed = [0, 1, 2, 3]
        inv = _inv_vocab()
        result = generate(model, seed, inv, length=10, temperature=1.0, seq_len=_SEQ)
        assert len(result) == len(seed) + 10

    def test_output_are_strings(self):
        model = _small_model()
        result = generate(model, [0, 1], _inv_vocab(), length=5, temperature=1.0, seq_len=_SEQ)
        assert all(isinstance(t, str) for t in result)

    def test_output_tokens_in_inv_vocab(self):
        model = _small_model()
        inv = _inv_vocab()
        result = generate(model, [0, 1, 2], inv, length=8, temperature=1.0, seq_len=_SEQ)
        assert all(t in inv.values() for t in result)

    def test_temperature_zero_is_deterministic(self):
        model = _small_model()
        seed = [0, 1, 2, 3]
        inv = _inv_vocab()
        r1 = generate(model, seed, inv, length=10, temperature=0.0, seq_len=_SEQ)
        r2 = generate(model, seed, inv, length=10, temperature=0.0, seq_len=_SEQ)
        assert r1 == r2

    def test_empty_seed_raises(self):
        model = _small_model()
        with pytest.raises(ValueError, match="seed_tokens"):
            generate(model, [], _inv_vocab(), length=5)

    def test_seed_tokens_appear_at_start_of_output(self):
        model = _small_model()
        inv = _inv_vocab()
        seed = [3, 5, 7]
        result = generate(model, seed, inv, length=4, temperature=0.0, seq_len=_SEQ)
        # First len(seed) entries come directly from inv_vocab[seed[i]]
        for i, sid in enumerate(seed):
            assert result[i] == inv[sid]

    def test_single_token_seed(self):
        model = _small_model()
        result = generate(model, [0], _inv_vocab(), length=5, temperature=1.0, seq_len=_SEQ)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# load_model_and_vocab
# ---------------------------------------------------------------------------


class TestLoadModelAndVocab:
    def test_loads_successfully(self, tmp_path):
        model = _small_model()
        path = str(tmp_path / "m.keras")
        _save_model(model, path)
        vocab_path = str(tmp_path / "vocab.json")
        save_vocabulary(build_vocabulary(), vocab_path)
        loaded, vocab, inv = load_model_and_vocab(path, vocab_path)
        assert isinstance(loaded, tf.keras.Model)
        assert len(vocab) == config.VOCAB_SIZE
        assert len(inv) == config.VOCAB_SIZE

    def test_missing_model_raises(self, tmp_path):
        vocab_path = str(tmp_path / "vocab.json")
        save_vocabulary(build_vocabulary(), vocab_path)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model_and_vocab("/nonexistent/model.keras", vocab_path)

    def test_missing_vocab_raises(self, tmp_path):
        model = _small_model()
        path = str(tmp_path / "m.keras")
        _save_model(model, path)
        with pytest.raises(FileNotFoundError):
            load_model_and_vocab(path, "/nonexistent/vocab.json")


# ---------------------------------------------------------------------------
# generate_midi (end-to-end)
# ---------------------------------------------------------------------------


class TestGenerateMidi:
    def _setup(self, tmp_path) -> tuple:
        """Train a tiny model, save it, and return (model_path, vocab_path)."""
        model = _small_model()
        inputs = tf.constant([[i % _VOCAB for i in range(_SEQ)]] * 4, dtype=tf.int32)
        targets = tf.constant([[(i + 1) % _VOCAB for i in range(_SEQ)]] * 4, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensors((inputs, targets)).repeat()
        train(
            model, ds, ds,
            epochs=1, learning_rate=0.01,
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            patience=5,
            steps_per_epoch=2,
            validation_steps=1,
        )
        model_path = str(tmp_path / "models" / "best_model.keras")

        small_vocab = {f"NOTE_ON_{i}": i for i in range(_VOCAB)}
        vocab_path = str(tmp_path / "vocab.json")
        save_vocabulary(small_vocab, vocab_path)
        return model_path, vocab_path

    def test_creates_midi_file(self, tmp_path):
        model_path, vocab_path = self._setup(tmp_path)
        out = str(tmp_path / "output.midi")
        generate_midi(model_path, out, vocab_path=vocab_path, length=50, rng_seed=0)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_output_is_valid_midi(self, tmp_path):
        model_path, vocab_path = self._setup(tmp_path)
        out = str(tmp_path / "output.midi")
        generate_midi(model_path, out, vocab_path=vocab_path, length=50, rng_seed=0)
        pm = pretty_midi.PrettyMIDI(out)
        assert len(pm.instruments) > 0

    def test_seed_midi_accepted(self, tmp_path):
        model_path, vocab_path = self._setup(tmp_path)
        seed_path = str(tmp_path / "seed.midi")
        _make_midi(seed_path)
        out = str(tmp_path / "output.midi")
        generate_midi(model_path, out, vocab_path=vocab_path, length=50, seed_midi_path=seed_path)
        assert os.path.exists(out)

    def test_rng_seed_is_reproducible(self, tmp_path):
        model_path, vocab_path = self._setup(tmp_path)
        out1 = str(tmp_path / "out1.midi")
        out2 = str(tmp_path / "out2.midi")
        generate_midi(model_path, out1, vocab_path=vocab_path, length=30, temperature=0.0, rng_seed=7)
        generate_midi(model_path, out2, vocab_path=vocab_path, length=30, temperature=0.0, rng_seed=7)
        assert open(out1, "rb").read() == open(out2, "rb").read()
