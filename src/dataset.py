"""tf.data pipeline: sliding-window (input, target) sequence pairs for LSTM training."""

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

import config

logger = logging.getLogger(__name__)


def get_sequence_paths(
    split: str,
    csv_path: str = config.MAESTRO_CSV,
    sequences_dir: str = config.SEQUENCES_DIR,
) -> List[str]:
    """Return .npy file paths for the requested MAESTRO split.

    Args:
        split: One of 'train', 'validation', or 'test'.
        csv_path: Path to MAESTRO metadata CSV.
        sequences_dir: Root directory of preprocessed .npy sequence files.

    Returns:
        Sorted list of .npy paths that exist on disk for the given split.
        Logs a warning for any CSV entry whose .npy file is missing.
    """
    df = pd.read_csv(csv_path)
    split_df = df[df["split"] == split]

    paths, missing = [], 0
    for midi_filename in split_df["midi_filename"]:
        npy_path = os.path.join(sequences_dir, os.path.splitext(midi_filename)[0] + ".npy")
        if os.path.exists(npy_path):
            paths.append(npy_path)
        else:
            logger.warning("Sequence file not found (run preprocessing): %s", npy_path)
            missing += 1

    if missing:
        logger.warning("%d / %d files missing for split '%s'", missing, len(split_df), split)
    logger.info("Found %d sequence files for split '%s'", len(paths), split)
    return sorted(paths)


def build_dataset(
    sequence_paths: List[str],
    seq_len: int = config.SEQUENCE_LENGTH,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    shuffle_buffer_size: int = config.DATASET_SHUFFLE_BUFFER,
    window_shift: int = config.DATASET_WINDOW_SHIFT,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (input_tokens, target_tokens) pairs.

    Each .npy file is loaded lazily and split into overlapping windows of length
    seq_len + 1. Every window becomes one (input=window[:-1], target=window[1:])
    sample for next-token prediction. Files shorter than seq_len + 1 tokens are
    silently skipped (contribute zero windows).

    Args:
        sequence_paths: .npy file paths from get_sequence_paths.
        seq_len: Length of each input/target sequence in tokens.
        batch_size: Number of sequences per batch.
        shuffle: If True, shuffle file order and windows (use for training only).
        shuffle_buffer_size: Number of windows held in the shuffle buffer.
        window_shift: Stride between consecutive windows.
            1 = fully overlapping (maximum data, high correlation between windows).
            seq_len = non-overlapping (minimum correlation, fewer total samples).
        seed: Random seed for reproducibility.

    Returns:
        Batched tf.data.Dataset yielding (input, target) int32 tensor pairs of
        shape (batch_size, seq_len) each.

    Raises:
        ValueError: If sequence_paths is empty.
    """
    if not sequence_paths:
        raise ValueError("sequence_paths is empty — run src/preprocessing.py first.")

    def load_windows(path: tf.Tensor) -> tf.data.Dataset:
        """Load one .npy file and return a Dataset of fixed-length windows."""
        tokens = tf.numpy_function(
            lambda p: np.load(p.decode()).astype(np.int32),
            inp=[path],
            Tout=tf.int32,
        )
        tokens.set_shape([None])
        return (
            tf.data.Dataset.from_tensor_slices(tokens)
            .window(seq_len + 1, shift=window_shift, drop_remainder=True)
            .flat_map(lambda w: w.batch(seq_len + 1, drop_remainder=True))
        )

    path_ds = tf.data.Dataset.from_tensor_slices(sequence_paths)
    if shuffle:
        path_ds = path_ds.shuffle(len(sequence_paths), seed=seed)

    # interleave for parallel file loading; deterministic=False is fine for shuffled training
    dataset = path_ds.interleave(
        load_windows,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )

    dataset = dataset.map(
        lambda window: (window[:-1], window[1:]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
