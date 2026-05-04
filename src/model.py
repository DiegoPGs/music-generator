"""LSTM music generation model definition."""

import io
import logging

import tensorflow as tf

import config

logger = logging.getLogger(__name__)


def build_model(
    vocab_size: int = config.VOCAB_SIZE,
    embed_dim: int = config.EMBEDDING_DIM,
    lstm_units: int = config.LSTM_UNITS,
    num_lstm_layers: int = config.NUM_LSTM_LAYERS,
    dropout_rate: float = config.DROPOUT_RATE,
) -> tf.keras.Model:
    """Build the LSTM music generation model.

    Architecture:
        Embedding → (num_lstm_layers - 1) × LSTM(return_sequences=True)
                  → LSTM(return_sequences=False) → Dropout → Dense(softmax)

    The model maps a sequence of token IDs to a softmax probability distribution
    over the next token (one prediction per input sequence).

    Args:
        vocab_size: Vocabulary size — both input range and output dimension.
        embed_dim: Dimensionality of the token embedding vectors.
        lstm_units: Number of hidden units in each LSTM layer.
        num_lstm_layers: Total number of stacked LSTM layers (must be >= 1).
        dropout_rate: Fraction of units to drop after the final LSTM.

    Returns:
        Uncompiled tf.keras.Model.
        Input shape:  (batch_size, seq_len)  — int32 token IDs.
        Output shape: (batch_size, vocab_size) — float32 next-token probabilities.

    Raises:
        ValueError: If num_lstm_layers < 1.
    """
    if num_lstm_layers < 1:
        raise ValueError(f"num_lstm_layers must be >= 1, got {num_lstm_layers}")

    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="token_ids")
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)

    for i in range(num_lstm_layers - 1):
        x = tf.keras.layers.LSTM(lstm_units, return_sequences=True, name=f"lstm_{i + 1}")(x)

    x = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name=f"lstm_{num_lstm_layers}")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    # dtype='float32' keeps logits/softmax in full precision under mixed_float16 policy.
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", dtype="float32", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_music_generator")
    _log_summary(model)
    return model


def _log_summary(model: tf.keras.Model) -> None:
    """Write model.summary() to the logger at INFO level."""
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))
    for line in buf.getvalue().splitlines():
        logger.info(line)
