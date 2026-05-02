"""Tests for src/model.py."""

import numpy as np
import pytest
import tensorflow as tf

import config
from src.model import build_model

# Small hyperparameters so tests run fast
_VOCAB = 20
_EMBED = 8
_UNITS = 16
_BATCH = 4
_SEQ = 12


def _small_model(**kwargs) -> tf.keras.Model:
    defaults = dict(vocab_size=_VOCAB, embed_dim=_EMBED, lstm_units=_UNITS, num_lstm_layers=2, dropout_rate=0.0)
    defaults.update(kwargs)
    return build_model(**defaults)


def _random_input(batch: int = _BATCH, seq: int = _SEQ) -> tf.Tensor:
    return tf.random.uniform((batch, seq), minval=0, maxval=_VOCAB, dtype=tf.int32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_returns_keras_model(self):
        assert isinstance(_small_model(), tf.keras.Model)

    def test_model_name(self):
        assert _small_model().name == "lstm_music_generator"

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_lstm_layers"):
            build_model(num_lstm_layers=0)

    def test_single_lstm_layer(self):
        model = _small_model(num_lstm_layers=1)
        lstm_layers = [layer for layer in model.layers if "lstm" in layer.name]
        assert len(lstm_layers) == 1

    def test_two_lstm_layers(self):
        model = _small_model(num_lstm_layers=2)
        lstm_layers = [layer for layer in model.layers if "lstm" in layer.name]
        assert len(lstm_layers) == 2

    def test_three_lstm_layers(self):
        model = _small_model(num_lstm_layers=3)
        lstm_layers = [layer for layer in model.layers if "lstm" in layer.name]
        assert len(lstm_layers) == 3

    def test_intermediate_lstms_return_sequences(self):
        model = _small_model(num_lstm_layers=3)
        # All LSTM layers except the last must have return_sequences=True
        lstm_layers = [layer for layer in model.layers if "lstm" in layer.name]
        for layer in lstm_layers[:-1]:
            assert layer.return_sequences is True

    def test_final_lstm_no_return_sequences(self):
        model = _small_model(num_lstm_layers=2)
        lstm_layers = [layer for layer in model.layers if "lstm" in layer.name]
        assert lstm_layers[-1].return_sequences is False

    def test_embedding_dim(self):
        model = _small_model(embed_dim=16)
        emb = model.get_layer("embedding")
        assert emb.output_dim == 16

    def test_embedding_vocab_size(self):
        model = _small_model(vocab_size=50)
        emb = model.get_layer("embedding")
        assert emb.input_dim == 50

    def test_lstm_units(self):
        model = _small_model(lstm_units=32)
        lstm = model.get_layer("lstm_1")
        assert lstm.units == 32

    def test_output_layer_vocab_size(self):
        model = _small_model(vocab_size=50)
        dense = model.get_layer("output")
        assert dense.units == 50

    def test_dropout_rate(self):
        model = _small_model(dropout_rate=0.5)
        dropout = model.get_layer("dropout")
        assert dropout.rate == pytest.approx(0.5)

    def test_uses_config_defaults(self):
        model = build_model()
        assert model.get_layer("embedding").input_dim == config.VOCAB_SIZE
        assert model.get_layer("embedding").output_dim == config.EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_output_shape(self):
        model = _small_model()
        out = model(_random_input(), training=False)
        assert out.shape == (_BATCH, _VOCAB)

    def test_output_shape_different_seq_len(self):
        model = _small_model()
        for seq_len in [1, 8, 32]:
            out = model(_random_input(seq=seq_len), training=False)
            assert out.shape == (_BATCH, _VOCAB)

    def test_output_shape_different_batch_size(self):
        model = _small_model()
        for batch in [1, 8, 16]:
            out = model(_random_input(batch=batch), training=False)
            assert out.shape == (batch, _VOCAB)

    def test_output_dtype_float32(self):
        model = _small_model()
        out = model(_random_input(), training=False)
        assert out.dtype == tf.float32

    def test_output_is_probability_distribution(self):
        model = _small_model()
        out = model(_random_input(), training=False).numpy()
        assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(out >= 0.0)

    def test_training_flag_accepted(self):
        model = _small_model(dropout_rate=0.5)
        # Should not raise in either mode
        model(_random_input(), training=True)
        model(_random_input(), training=False)

    def test_single_token_input(self):
        model = _small_model()
        out = model(tf.constant([[5]], dtype=tf.int32), training=False)
        assert out.shape == (1, _VOCAB)

    def test_output_values_differ_across_inputs(self):
        model = _small_model()
        inp_a = tf.constant([[0, 1, 2, 3]], dtype=tf.int32)
        inp_b = tf.constant([[4, 5, 6, 7]], dtype=tf.int32)
        out_a = model(inp_a, training=False).numpy()
        out_b = model(inp_b, training=False).numpy()
        assert not np.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Parameter count sanity check
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_has_trainable_params(self):
        assert _small_model().count_params() > 0

    def test_param_count_scales_with_vocab(self):
        small = build_model(vocab_size=10, embed_dim=_EMBED, lstm_units=_UNITS, num_lstm_layers=1)
        large = build_model(vocab_size=100, embed_dim=_EMBED, lstm_units=_UNITS, num_lstm_layers=1)
        assert large.count_params() > small.count_params()

    def test_param_count_scales_with_lstm_layers(self):
        one = _small_model(num_lstm_layers=1)
        two = _small_model(num_lstm_layers=2)
        assert two.count_params() > one.count_params()
