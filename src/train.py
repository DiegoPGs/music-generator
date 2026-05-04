"""Training loop with checkpointing, early stopping, and TensorBoard logging."""

import argparse
import datetime
import logging
import os
import random
from typing import List, Optional

import numpy as np
import tensorflow as tf

import config
from src.dataset import build_dataset, get_sequence_paths
from src.model import build_model

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Random seeds set to %d", seed)


def prepare_for_training(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Extract the last target token from each window for next-token prediction.

    The dataset pipeline yields (input, target_seq) pairs where both tensors
    have shape (batch, seq_len). The model outputs a single next-token
    distribution, so the label for each window is target_seq[:, -1] — the
    token immediately following the input sequence.

    Args:
        dataset: tf.data.Dataset yielding (input, target_seq) int32 pairs.

    Returns:
        Mapped dataset yielding (input, next_token) where next_token has
        shape (batch,).
    """
    return dataset.map(lambda x, y: (x, y[:, -1]), num_parallel_calls=tf.data.AUTOTUNE)


def build_callbacks(
    model_dir: str = config.MODEL_DIR,
    log_dir: str = config.LOG_DIR,
    patience: int = config.EARLY_STOPPING_PATIENCE,
) -> List[tf.keras.callbacks.Callback]:
    """Build the standard set of training callbacks.

    Args:
        model_dir: Directory where the best model checkpoint is saved.
        log_dir: Root directory for TensorBoard logs.
        patience: Number of epochs with no val_loss improvement before stopping.

    Returns:
        List of [ModelCheckpoint, EarlyStopping, TensorBoard] callbacks.
    """
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, "fit", run_id),
        histogram_freq=1,
        update_freq="epoch",
    )

    return [checkpoint, early_stop, tensorboard]


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    model_dir: str = config.MODEL_DIR,
    log_dir: str = config.LOG_DIR,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    use_mixed_precision: bool = False,
) -> tf.keras.callbacks.History:
    """Compile the model and run the training loop.

    The datasets are mapped to (input, next_token) pairs before training.
    Optimizer is Adam; loss is SparseCategoricalCrossentropy from integer labels.

    Args:
        model: Uncompiled or previously compiled tf.keras.Model from build_model().
        train_ds: Training dataset from build_dataset() (shuffle=True).
        val_ds: Validation dataset from build_dataset() (shuffle=False).
        epochs: Maximum number of training epochs.
        learning_rate: Adam learning rate.
        model_dir: Directory for ModelCheckpoint output.
        log_dir: Root directory for TensorBoard logs.
        patience: Early stopping patience (epochs without val_loss improvement).
        steps_per_epoch: Cap on training batches per epoch (None = full epoch).
        validation_steps: Cap on validation batches per epoch (None = full val set).
        use_mixed_precision: Logged-only flag indicating mixed_float16 was enabled.
            The global policy must be set BEFORE build_model() is called, so this
            parameter only controls a GPU-availability warning.

    Returns:
        tf.keras.callbacks.History object with per-epoch metrics.
    """
    if not tf.config.list_physical_devices("GPU"):
        logger.warning("No GPU detected — training on CPU will be slow.")
        if use_mixed_precision:
            logger.warning(
                "Mixed precision enabled but no GPU detected — float16 will be slower than float32 on CPU."
            )

    train_ready = prepare_for_training(train_ds)
    val_ready = prepare_for_training(val_ds)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    logger.info("Model compiled. lr=%.4e  epochs=%d  patience=%d", learning_rate, epochs, patience)

    callbacks = build_callbacks(model_dir=model_dir, log_dir=log_dir, patience=patience)

    history = model.fit(
        train_ready,
        validation_data=val_ready,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    logger.info("Training complete. Best val_loss: %.4f", min(history.history["val_loss"]))
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM music generation model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, dest="learning_rate")
    parser.add_argument("--seq-len", type=int, default=config.SEQUENCE_LENGTH, dest="seq_len")
    parser.add_argument("--lstm-units", type=int, default=config.LSTM_UNITS, dest="lstm_units")
    parser.add_argument("--num-layers", type=int, default=config.NUM_LSTM_LAYERS, dest="num_lstm_layers")
    parser.add_argument("--dropout", type=float, default=config.DROPOUT_RATE, dest="dropout_rate")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None, dest="max_files",
                        help="Limit number of sequence files per split (default: all).")
    parser.add_argument("--steps-per-epoch", type=int, default=None, dest="steps_per_epoch",
                        help="Cap on training batches per epoch (default: full epoch).")
    parser.add_argument("--validation-steps", type=int, default=None, dest="validation_steps",
                        help="Cap on validation batches per epoch (default: full val set).")
    parser.add_argument("--mixed-precision", action="store_true", dest="mixed_precision",
                        help="Enable mixed_float16 global policy (recommended on Ampere+ GPUs).")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    if args.seed is not None:
        set_seeds(args.seed)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision policy set to 'mixed_float16'.")

    train_paths = get_sequence_paths("train", max_files=args.max_files)
    val_paths = get_sequence_paths("validation", max_files=args.max_files)

    train_ds = build_dataset(
        train_paths, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_ds = build_dataset(
        val_paths, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )

    model = build_model(
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
        dropout_rate=args.dropout_rate,
    )

    train(
        model, train_ds, val_ds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        use_mixed_precision=args.mixed_precision,
    )
