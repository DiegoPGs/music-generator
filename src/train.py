"""Training loop with academic progress walkthrough, diagnostics, checkpointing, and TensorBoard logging."""

import argparse
import datetime
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
from src.dataset import build_dataset, get_sequence_paths
from src.model import build_model
from src.preprocessing import build_vocabulary, dequantize_velocity

logger = logging.getLogger(__name__)

_NOTE_NAMES: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_TOKEN_CATEGORIES: Tuple[str, ...] = ("NOTE_ON", "NOTE_OFF", "TIME_SHIFT", "SET_VELOCITY")
_DEFAULT_INV_VOCAB: Dict[int, str] = {idx: tok for tok, idx in build_vocabulary().items()}
_EPSILON = 1e-12
_BYTES_PER_FLOAT32 = 4
_BYTES_PER_MB = 1024 * 1024
_MS_PER_SECOND = 1000.0
_PERCENT_SCALE = 100.0
_BANNER_WIDTH = 92


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


# ── Musical Token Formatting & Categorization ─────────────────────────────────


def _pitch_to_note_name(pitch: int) -> str:
    """Convert a MIDI pitch (0–127) to scientific pitch notation (e.g. 60 -> C4)."""
    octave = (pitch // config.MIDI_NOTES_PER_OCTAVE) - config.MIDI_OCTAVE_OFFSET
    note = _NOTE_NAMES[pitch % config.MIDI_NOTES_PER_OCTAVE]
    return f"{note}{octave}"


def describe_token(token_id: int, inv_vocab: Optional[Dict[int, str]] = None) -> str:
    """Format a token ID with human-readable musical annotations for academic output.

    Translates raw event tokens into musical domain descriptions:
    - NOTE_ON_60   -> 'NOTE_ON_60 (C4)'
    - NOTE_OFF_60  -> 'NOTE_OFF_60 (C4)'
    - TIME_SHIFT_5 -> 'TIME_SHIFT_5 (50ms)'
    - SET_VELOCITY_16 -> 'SET_VELOCITY_16 (vel~66)'

    Args:
        token_id: Integer token index.
        inv_vocab: Optional ID-to-token mapping. Defaults to the 388-token vocabulary.

    Returns:
        Formatted token string with musical domain annotation.
    """
    lookup = inv_vocab if inv_vocab is not None else _DEFAULT_INV_VOCAB
    token_str = lookup.get(int(token_id), f"TOKEN_{int(token_id)}")

    if token_str.startswith("NOTE_ON_"):
        pitch = int(token_str[len("NOTE_ON_"):])
        return f"{token_str} ({_pitch_to_note_name(pitch)})"
    if token_str.startswith("NOTE_OFF_"):
        pitch = int(token_str[len("NOTE_OFF_"):])
        return f"{token_str} ({_pitch_to_note_name(pitch)})"
    if token_str.startswith("TIME_SHIFT_"):
        steps = int(token_str[len("TIME_SHIFT_"):])
        ms = steps * config.TIME_SHIFT_MS_PER_BIN
        return f"{token_str} ({ms}ms)"
    if token_str.startswith("SET_VELOCITY_"):
        vel_bin = int(token_str[len("SET_VELOCITY_"):])
        vel = dequantize_velocity(vel_bin)
        return f"{token_str} (vel~{vel})"
    return token_str


def categorize_token_id(token_id: int, inv_vocab: Optional[Dict[int, str]] = None) -> str:
    """Return the musical event family for a token ID.

    Args:
        token_id: Integer token index.
        inv_vocab: Optional ID-to-token mapping. Defaults to the 388-token vocabulary.

    Returns:
        One of 'NOTE_ON', 'NOTE_OFF', 'TIME_SHIFT', 'SET_VELOCITY', or 'OTHER'.
    """
    lookup = inv_vocab if inv_vocab is not None else _DEFAULT_INV_VOCAB
    token_str = lookup.get(int(token_id), "")
    for category in _TOKEN_CATEGORIES:
        if token_str.startswith(f"{category}_"):
            return category
    return "OTHER"


def _format_ascii_bar(fraction: float, width: int = config.TRAINING_BAR_WIDTH) -> str:
    """Build a fixed-width ASCII progress/distribution bar for a fraction in [0, 1]."""
    clamped = max(0.0, min(1.0, float(fraction)))
    filled = int(round(clamped * width))
    return "█" * filled + "░" * (width - filled)


# ── Academic Diagnostic Helpers ───────────────────────────────────────────────


def inspect_pipeline_sample(
    raw_ds: tf.data.Dataset,
    prepared_ds: tf.data.Dataset,
    inv_vocab: Optional[Dict[int, str]] = None,
    context_tokens: int = config.TRAINING_DEMO_CONTEXT_TOKENS,
) -> Optional[Dict[str, Any]]:
    """Inspect a sample batch to demonstrate the data and tensor pipeline before training.

    Extracts one batch from both the raw sliding-window dataset and the prepared
    next-token classification dataset, logs a step-by-step walkthrough of how
    sequences are windowed and sliced, and returns the inspected metadata.

    Args:
        raw_ds: Dataset from build_dataset() yielding (input_seq, target_seq).
        prepared_ds: Dataset from prepare_for_training() yielding (input_seq, next_token).
        inv_vocab: Optional ID-to-token dictionary for decoding sample tokens.
        context_tokens: Number of trailing context tokens from Sample #0 to display.

    Returns:
        Dictionary with pipeline shapes, dtypes, sample token walkthrough, and
        'sample_batch' (prep_x, prep_y), or None if the dataset is empty.
    """
    raw_sample = next(iter(raw_ds), None)
    prep_sample = next(iter(prepared_ds), None)
    if raw_sample is None or prep_sample is None:
        return None

    raw_x, raw_y = raw_sample
    prep_x, prep_y = prep_sample

    batch_size = int(prep_x.shape[0])
    seq_len = int(prep_x.shape[1])
    tokens_per_batch = batch_size * seq_len

    sample_seq = prep_x[0].numpy().tolist()
    tail_ids = [int(t) for t in sample_seq[-context_tokens:]]
    tail_descriptions = [describe_token(t, inv_vocab) for t in tail_ids]
    target_id = int(prep_y[0].numpy())
    target_description = describe_token(target_id, inv_vocab)

    info: Dict[str, Any] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_batch": tokens_per_batch,
        "input_dtype": prep_x.dtype.name,
        "target_dtype": prep_y.dtype.name,
        "raw_input_shape": tuple(int(d) for d in raw_x.shape),
        "raw_target_shape": tuple(int(d) for d in raw_y.shape),
        "prepared_input_shape": tuple(int(d) for d in prep_x.shape),
        "prepared_target_shape": tuple(int(d) for d in prep_y.shape),
        "sample_context_ids": tail_ids,
        "sample_context_descriptions": tail_descriptions,
        "sample_target_id": target_id,
        "sample_target_description": target_description,
        "sample_batch": (prep_x, prep_y),
    }

    sep = "═" * _BANNER_WIDTH
    sub = "─" * _BANNER_WIDTH
    logger.info(sep)
    logger.info("STAGE 1/4 — DATA & TENSOR PIPELINE WALKTHROUGH")
    logger.info(sep)
    logger.info(
        "  1. Sliding Window Extraction : Each MIDI sequence is sliced into windows of length T+1 = %d tokens",
        seq_len + 1,
    )
    logger.info(
        "     • Raw Window Pair         : input_seq %s (%s) -> target_seq %s (%s)",
        info["raw_input_shape"],
        info["input_dtype"],
        info["raw_target_shape"],
        info["target_dtype"],
    )
    logger.info(
        "  2. Next-Token Target Slice   : prepare_for_training() selects target_seq[:, -1] as label y"
    )
    logger.info(
        "     • Prepared Training Batch : X %s -> y %s  (%d tokens processed per batch)",
        info["prepared_input_shape"],
        info["prepared_target_shape"],
        tokens_per_batch,
    )
    logger.info(sub)
    logger.info(
        "  Concrete Example (Batch Sample #0 — showing last %d of %d context tokens -> target):",
        len(tail_ids),
        seq_len,
    )
    start_pos = max(0, seq_len - len(tail_ids))
    for offset, (tid, desc) in enumerate(zip(tail_ids, tail_descriptions)):
        logger.info("    x[t=%2d] : ID %3d  │  %s", start_pos + offset, tid, desc)
    logger.info("    ──────────────────┼──────────────────────────────────────────────")
    logger.info("    y[next] : ID %3d  │  %s  <-- Ground-truth class label", target_id, target_description)
    logger.info(sep)

    return info


def summarize_model_layers(model: tf.keras.Model) -> List[Dict[str, Any]]:
    """Extract per-layer parameter counts, output shapes, and academic roles.

    Args:
        model: Compiled or uncompiled tf.keras.Model.

    Returns:
        List of dictionaries describing each layer in the model.
    """
    total_params = max(int(model.count_params()), 1)
    summaries: List[Dict[str, Any]] = []

    for layer in model.layers:
        layer_type = type(layer).__name__
        params = int(layer.count_params())
        pct = (params / total_params) * _PERCENT_SCALE
        try:
            out_shape = tuple(layer.output.shape)
        except AttributeError:
            out_shape = getattr(layer, "output_shape", ())

        if isinstance(layer, tf.keras.layers.InputLayer):
            role = "Receives batch of discrete token ID sequences (batch, seq_len)"
        elif isinstance(layer, tf.keras.layers.Embedding):
            role = f"Maps token IDs [0..{layer.input_dim - 1}] -> dense R^{layer.output_dim} vectors"
        elif isinstance(layer, tf.keras.layers.LSTM):
            if layer.return_sequences:
                role = f"Recurrent sequence encoder (emits hidden state h_t in R^{layer.units} at all steps)"
            else:
                role = f"Sequence-to-vector encoder (summarizes context into final state h_T in R^{layer.units})"
        elif isinstance(layer, tf.keras.layers.Dropout):
            role = f"Regularization: zeroes {layer.rate * _PERCENT_SCALE:.0f}% of activations during training"
        elif isinstance(layer, tf.keras.layers.Dense):
            role = f"Linear projection to {layer.units} logits + Softmax next-token probabilities"
        else:
            role = "Intermediate neural transformation"

        summaries.append(
            {
                "name": layer.name,
                "type": layer_type,
                "output_shape": out_shape,
                "params": params,
                "pct_params": pct,
                "role": role,
            }
        )

    return summaries


def compute_layer_weight_stats(
    model: tf.keras.Model,
    prev_weights: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[np.ndarray]]]:
    """Compute per-layer weight distribution statistics and parameter update norms.

    For each layer with trainable variables, measures the L2 norm ||W||_2, mean,
    standard deviation, max absolute weight, and the L2 displacement ||ΔW||_2
    relative to the previous epoch's weights snapshot.

    Args:
        model: tf.keras.Model being trained.
        prev_weights: Optional dictionary mapping layer names to lists of numpy
            weight arrays from the previous epoch.

    Returns:
        Tuple of (list of per-layer diagnostic dicts, current weight snapshot dict).
    """
    stats: List[Dict[str, Any]] = []
    snapshot: Dict[str, List[np.ndarray]] = {}

    for layer in model.layers:
        if not layer.trainable_weights:
            continue

        current_arrays = [np.array(w.numpy(), dtype=np.float32, copy=True) for w in layer.trainable_weights]
        snapshot[layer.name] = current_arrays

        flat = np.concatenate([arr.ravel() for arr in current_arrays])
        weight_l2 = float(np.linalg.norm(flat))
        weight_mean = float(np.mean(flat))
        weight_std = float(np.std(flat))
        weight_max_abs = float(np.max(np.abs(flat)))

        update_l2 = 0.0
        if prev_weights is not None and layer.name in prev_weights:
            prev_arrays = prev_weights[layer.name]
            if len(prev_arrays) == len(current_arrays):
                deltas = [curr - prev for curr, prev in zip(current_arrays, prev_arrays)]
                flat_delta = np.concatenate([d.ravel() for d in deltas])
                update_l2 = float(np.linalg.norm(flat_delta))

        relative_update = update_l2 / (weight_l2 + _EPSILON)
        stats.append(
            {
                "layer": layer.name,
                "type": type(layer).__name__,
                "weight_l2": weight_l2,
                "weight_mean": weight_mean,
                "weight_std": weight_std,
                "weight_max_abs": weight_max_abs,
                "update_l2": update_l2,
                "relative_update": relative_update,
            }
        )

    return stats, snapshot


def compute_sample_gradient_norms(
    model: tf.keras.Model,
    sample_batch: Tuple[tf.Tensor, tf.Tensor],
) -> Dict[str, float]:
    """Compute per-layer L2 gradient norms ||∇_W L||_2 on a reference batch via BPTT.

    Illustrates how error signals propagate backward from the Softmax output layer
    through stacked LSTM layers down to the Embedding lookup table.

    Args:
        model: tf.keras.Model being trained.
        sample_batch: Tuple of (sample_x, sample_y) tensors.

    Returns:
        Dictionary mapping each trainable layer name (and 'global') to its L2
        gradient norm.
    """
    sample_x, sample_y = sample_batch
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        preds = model(sample_x, training=False)
        loss_value = loss_fn(sample_y, preds)

    grads = tape.gradient(loss_value, model.trainable_variables)
    var_grad_map = {id(var): grad for var, grad in zip(model.trainable_variables, grads)}

    grad_norms: Dict[str, float] = {}
    all_dense_grads: List[tf.Tensor] = []

    for layer in model.layers:
        if not layer.trainable_variables:
            continue
        layer_grads: List[tf.Tensor] = []
        for var in layer.trainable_variables:
            g = var_grad_map.get(id(var))
            if g is None:
                continue
            tensor_g = tf.cast(g.values if isinstance(g, tf.IndexedSlices) else g, tf.float32)
            layer_grads.append(tensor_g)
            all_dense_grads.append(tensor_g)

        if layer_grads:
            grad_norms[layer.name] = float(tf.linalg.global_norm(layer_grads).numpy())
        else:
            grad_norms[layer.name] = 0.0

    grad_norms["global"] = float(tf.linalg.global_norm(all_dense_grads).numpy()) if all_dense_grads else 0.0
    return grad_norms


def compute_musical_domain_insights(
    model: tf.keras.Model,
    sample_batch: Tuple[tf.Tensor, tf.Tensor],
    inv_vocab: Optional[Dict[int, str]] = None,
    top_k: int = config.TRAINING_TOP_K_PREDICTIONS,
) -> Dict[str, Any]:
    """Analyze the model's output probability distribution in terms of musical event families.

    Computes how much probability mass the model allocates to NOTE_ON, NOTE_OFF,
    TIME_SHIFT, and SET_VELOCITY tokens compared to the ground-truth targets,
    measures the Shannon entropy of the predictive distribution, and extracts
    the Top-K predicted musical events for Sample #0.

    Args:
        model: tf.keras.Model being trained.
        sample_batch: Tuple of (sample_x, sample_y) tensors.
        inv_vocab: Optional ID-to-token mapping.
        top_k: Number of highest-probability predictions to extract for Sample #0.

    Returns:
        Dictionary containing category probability distributions, entropy metrics,
        and Sample #0 qualitative predictions.
    """
    sample_x, sample_y = sample_batch
    probs = model(sample_x, training=False).numpy().astype(np.float64)
    targets = sample_y.numpy().astype(np.int32)
    vocab_size = int(probs.shape[-1])

    # Partition probability mass across musical token families
    pred_cat_probs: Dict[str, float] = {cat: 0.0 for cat in _TOKEN_CATEGORIES}
    for token_id in range(vocab_size):
        cat = categorize_token_id(token_id, inv_vocab)
        if cat in pred_cat_probs:
            pred_cat_probs[cat] += float(np.mean(probs[:, token_id]))

    # Ground-truth empirical distribution in the sample batch
    target_cat_shares: Dict[str, float] = {cat: 0.0 for cat in _TOKEN_CATEGORIES}
    if len(targets) > 0:
        for tid in targets:
            cat = categorize_token_id(int(tid), inv_vocab)
            if cat in target_cat_shares:
                target_cat_shares[cat] += 1.0 / len(targets)

    # Shannon entropy H(p) = -sum p * ln(p)
    safe_probs = np.clip(probs, _EPSILON, 1.0)
    entropies = -np.sum(safe_probs * np.log(safe_probs), axis=-1)
    mean_entropy = float(np.mean(entropies))
    max_entropy = float(math.log(max(vocab_size, 1)))
    mean_top1_conf = float(np.mean(np.max(probs, axis=-1)))

    # Qualitative Top-K predictions on Sample #0
    sample0_probs = probs[0]
    k = min(top_k, vocab_size)
    top_indices = np.argsort(sample0_probs)[::-1][:k]
    sample_top_k = [
        (int(idx), describe_token(int(idx), inv_vocab), float(sample0_probs[idx]))
        for idx in top_indices
    ]

    target0_id = int(targets[0])
    target0_prob = float(sample0_probs[target0_id]) if 0 <= target0_id < vocab_size else 0.0
    sample_target = (target0_id, describe_token(target0_id, inv_vocab), target0_prob)

    return {
        "predicted_category_probs": pred_cat_probs,
        "target_category_shares": target_cat_shares,
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "mean_top1_confidence": mean_top1_conf,
        "sample_top_k": sample_top_k,
        "sample_target": sample_target,
    }


# ── Custom Academic Training Callback ─────────────────────────────────────────


class AcademicTrainingCallback(tf.keras.callbacks.Callback):
    """Keras callback providing an academic step-by-step demonstration of model training.

    Features:
        - Pre-training architecture table with per-layer roles and theoretical
          random-guessing baselines (Cross-Entropy ln(V) and Perplexity V).
        - Live tqdm batch progress bar displaying real-time loss, perplexity,
          accuracy, and token throughput (tokens/sec).
        - Detailed per-epoch diagnostic report covering:
            1. Optimization & Generalization metrics (Loss, Perplexity, Accuracy,
               deltas, throughput, and EarlyStopping patience tracking).
            2. Internal Layer Diagnostics (weight L2 norms, standard deviations,
               epoch parameter updates ||ΔW||_2, and BPTT gradient norms ||∇_W L||_2).
            3. Musical Domain Insights (probability mass distribution across
               NOTE_ON, NOTE_OFF, TIME_SHIFT, SET_VELOCITY vs. ground truth,
               Shannon entropy, and Top-K predictions on a sample sequence).
        - Post-training progression summary table across all completed epochs.
    """

    def __init__(
        self,
        total_epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        seq_len: int = config.SEQUENCE_LENGTH,
        steps_per_epoch: Optional[int] = None,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        sample_batch: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        inv_vocab: Optional[Dict[int, str]] = None,
        enable_progress_bar: bool = True,
    ) -> None:
        """Initialize the academic training callback.

        Args:
            total_epochs: Total planned training epochs.
            batch_size: Number of sequences per training batch.
            seq_len: Sequence length in tokens per sample.
            steps_per_epoch: Optional cap on batches per epoch.
            patience: EarlyStopping patience for status display.
            sample_batch: Optional (sample_x, sample_y) batch for gradient and
                musical domain diagnostics at the end of each epoch.
            inv_vocab: Optional ID-to-token mapping for musical event descriptions.
            enable_progress_bar: Whether to render a live tqdm batch bar per epoch.
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.patience = patience
        self.sample_batch = sample_batch
        self.inv_vocab = inv_vocab
        self.enable_progress_bar = enable_progress_bar

        self.epoch_records: List[Dict[str, Any]] = []
        self._prev_weights: Optional[Dict[str, List[np.ndarray]]] = None
        self._best_val_loss: float = float("inf")
        self._best_epoch: int = 0
        self._wait_epochs: int = 0
        self._train_start_time: float = 0.0
        self._epoch_start_time: float = 0.0
        self._batches_seen: int = 0
        self._tokens_processed: int = 0
        self._pbar: Optional[tqdm] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log model architecture breakdown, theoretical baselines, and snapshot initial weights."""
        self._train_start_time = time.perf_counter()
        self.epoch_records.clear()
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._wait_epochs = 0

        _, self._prev_weights = compute_layer_weight_stats(self.model)
        layer_summaries = summarize_model_layers(self.model)

        total_params = int(self.model.count_params())
        mem_mb = (total_params * _BYTES_PER_FLOAT32) / _BYTES_PER_MB
        vocab_size = int(self.model.output_shape[-1])
        random_loss = math.log(max(vocab_size, 1))
        random_ppl = float(vocab_size)
        random_acc = _PERCENT_SCALE / max(vocab_size, 1)

        sep = "═" * _BANNER_WIDTH
        sub = "─" * _BANNER_WIDTH
        logger.info(sep)
        logger.info("STAGE 2/4 — MODEL ARCHITECTURE & THEORETICAL BASELINES")
        logger.info(sep)
        logger.info(
            "  %-14s │ %-12s │ %-20s │ %10s │ %6s │ %s",
            "Layer",
            "Type",
            "Output Tensor Shape",
            "Params",
            "Share",
            "Mathematical Role in Forward Pass",
        )
        logger.info("  %s", sub)
        for item in layer_summaries:
            logger.info(
                "  %-14s │ %-12s │ %-20s │ %10s │ %5.1f%% │ %s",
                item["name"],
                item["type"],
                str(item["output_shape"]),
                f"{item['params']:,}",
                item["pct_params"],
                item["role"],
            )
        logger.info("  %s", sub)
        logger.info(
            "  Total Trainable Parameters : %s (~%.2f MB in float32)",
            f"{total_params:,}",
            mem_mb,
        )
        logger.info(
            "  Uniform Random Baseline    : Loss = ln(%d) = %.4f nats  │  Perplexity = %.1f  │  Accuracy = %.2f%%",
            vocab_size,
            random_loss,
            random_ppl,
            random_acc,
        )
        logger.info(sep)
        logger.info("STAGE 3/4 — ITERATIVE OPTIMIZATION (BACKPROPAGATION THROUGH TIME)")
        logger.info(sep)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Start epoch timers and initialize the per-batch tqdm progress bar."""
        self._epoch_start_time = time.perf_counter()
        self._batches_seen = 0
        self._tokens_processed = 0

        if self.enable_progress_bar:
            self._pbar = tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1:02d}/{self.total_epochs:02d} [Training]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update batch throughput counters and refresh live progress bar metrics."""
        logs = logs or {}
        self._batches_seen += 1
        self._tokens_processed += self.batch_size * self.seq_len

        if self._pbar is not None:
            loss = float(logs.get("loss", 0.0))
            ppl = math.exp(min(loss, config.MAX_EXP_LOSS))
            acc = float(logs.get("accuracy", 0.0)) * _PERCENT_SCALE
            elapsed = max(time.perf_counter() - self._epoch_start_time, _EPSILON)
            tok_per_sec = self._tokens_processed / elapsed

            self._pbar.update(1)
            self._pbar.set_postfix(
                loss=f"{loss:.4f}",
                ppl=f"{ppl:.1f}",
                acc=f"{acc:.1f}%",
                tok_s=f"{tok_per_sec:,.0f}",
            )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Compute and log extended metrics, internal layer diagnostics, and musical insights."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        logs = logs or {}
        epoch_duration = max(time.perf_counter() - self._epoch_start_time, _EPSILON)
        steps = max(self._batches_seen, 1)
        ms_per_step = (epoch_duration / steps) * _MS_PER_SECOND
        seq_per_sec = (steps * self.batch_size) / epoch_duration
        tok_per_sec = self._tokens_processed / epoch_duration

        train_loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", train_loss))
        train_acc = float(logs.get("accuracy", 0.0)) * _PERCENT_SCALE
        val_acc = float(logs.get("val_accuracy", train_acc / _PERCENT_SCALE)) * _PERCENT_SCALE
        train_ppl = math.exp(min(train_loss, config.MAX_EXP_LOSS))
        val_ppl = math.exp(min(val_loss, config.MAX_EXP_LOSS))

        prev_record = self.epoch_records[-1] if self.epoch_records else None
        train_loss_delta = train_loss - prev_record["train_loss"] if prev_record else 0.0
        val_loss_delta = val_loss - prev_record["val_loss"] if prev_record else 0.0
        train_ppl_delta = train_ppl - prev_record["train_ppl"] if prev_record else 0.0
        val_ppl_delta = val_ppl - prev_record["val_ppl"] if prev_record else 0.0
        train_acc_delta = train_acc - prev_record["train_acc"] if prev_record else 0.0
        val_acc_delta = val_acc - prev_record["val_acc"] if prev_record else 0.0

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch = epoch + 1
            self._wait_epochs = 0
            is_best = True
            status_str = "★ NEW BEST (checkpoint saved to best_model.keras)"
        else:
            self._wait_epochs += 1
            is_best = False
            status_str = f"No val_loss improvement — EarlyStopping patience: {self._wait_epochs}/{self.patience}"

        weight_stats, self._prev_weights = compute_layer_weight_stats(self.model, self._prev_weights)
        total_update_l2 = float(np.linalg.norm([item["update_l2"] for item in weight_stats])) if weight_stats else 0.0

        grad_norms: Dict[str, float] = {}
        musical_insights: Optional[Dict[str, Any]] = None
        if self.sample_batch is not None:
            grad_norms = compute_sample_gradient_norms(self.model, self.sample_batch)
            musical_insights = compute_musical_domain_insights(
                self.model, self.sample_batch, inv_vocab=self.inv_vocab
            )

        record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "duration_sec": epoch_duration,
            "ms_per_step": ms_per_step,
            "tokens_per_sec": tok_per_sec,
            "sequences_per_sec": seq_per_sec,
            "is_best": is_best,
            "wait_epochs": self._wait_epochs,
            "weight_stats": weight_stats,
            "total_update_l2": total_update_l2,
            "grad_norms": grad_norms,
            "musical_insights": musical_insights,
            "status": status_str,
        }
        self.epoch_records.append(record)
        self._log_epoch_report(record, prev_record is not None, (
            train_loss_delta,
            val_loss_delta,
            train_ppl_delta,
            val_ppl_delta,
            train_acc_delta,
            val_acc_delta,
        ))

    def _log_epoch_report(
        self,
        record: Dict[str, Any],
        has_prev: bool,
        deltas: Tuple[float, float, float, float, float, float],
    ) -> None:
        """Format and log the multi-section academic report for a completed epoch."""
        t_loss_d, v_loss_d, t_ppl_d, v_ppl_d, t_acc_d, v_acc_d = deltas
        sub = "─" * _BANNER_WIDTH

        def _fmt_delta(val: float, lower_is_better: bool = True, is_pct: bool = False) -> str:
            if not has_prev:
                return "   (init)   "
            arrow = "↓" if val < 0 else ("↑" if val > 0 else "→")
            suffix = "%" if is_pct else ""
            return f"{val:+.4f}{suffix} {arrow}" if not is_pct else f"{val:+.2f}{suffix} {arrow}"

        logger.info(sub)
        logger.info(
            "EPOCH %02d/%02d SUMMARY  │  Time: %.2fs (%.1f ms/step)  │  Speed: %s tok/s (%.1f seq/s)",
            record["epoch"],
            self.total_epochs,
            record["duration_sec"],
            record["ms_per_step"],
            f"{record['tokens_per_sec']:,.0f}",
            record["sequences_per_sec"],
        )
        logger.info("  Status: %s", record["status"])
        logger.info("")
        logger.info("  [1] Optimization & Generalization Metrics:")
        logger.info(
            "      %-26s │ %12s (%12s) │ %12s (%12s)",
            "Metric",
            "Train",
            "Δ Train",
            "Validation",
            "Δ Val",
        )
        logger.info("      ───────────────────────────┼─────────────────────────────┼─────────────────────────────")
        logger.info(
            "      %-26s │ %12.4f (%12s) │ %12.4f (%12s)",
            "Cross-Entropy Loss (nats)",
            record["train_loss"],
            _fmt_delta(t_loss_d, lower_is_better=True),
            record["val_loss"],
            _fmt_delta(v_loss_d, lower_is_better=True),
        )
        logger.info(
            "      %-26s │ %12.2f (%12s) │ %12.2f (%12s)",
            "Perplexity exp(loss)",
            record["train_ppl"],
            _fmt_delta(t_ppl_d, lower_is_better=True),
            record["val_ppl"],
            _fmt_delta(v_ppl_d, lower_is_better=True),
        )
        logger.info(
            "      %-26s │ %11.2f%% (%12s) │ %11.2f%% (%12s)",
            "Top-1 Accuracy",
            record["train_acc"],
            _fmt_delta(t_acc_d, lower_is_better=False, is_pct=True),
            record["val_acc"],
            _fmt_delta(v_acc_d, lower_is_better=False, is_pct=True),
        )

        if record["weight_stats"]:
            logger.info("")
            logger.info("  [2] Internal Layer Diagnostics (Weights, Updates & BPTT Gradients):")
            logger.info(
                "      %-14s │ %-10s │ %10s │ %10s │ %11s │ %11s │ %11s",
                "Layer",
                "Type",
                "||W||_2",
                "Std(W)",
                "||ΔW||_2",
                "||ΔW||/||W||",
                "||∇_W L||_2",
            )
            logger.info(
                "      ───────────────┼────────────┼────────────┼────────────┼─────────────┼─────────────┼────────────"
            )
            grad_norms = record["grad_norms"]
            for w_item in record["weight_stats"]:
                lname = w_item["layer"]
                g_str = f"{grad_norms[lname]:11.4e}" if lname in grad_norms else "        N/A"
                logger.info(
                    "      %-14s │ %-10s │ %10.4f │ %10.4f │ %11.4e │ %11.4e │ %s",
                    lname,
                    w_item["type"],
                    w_item["weight_l2"],
                    w_item["weight_std"],
                    w_item["update_l2"],
                    w_item["relative_update"],
                    g_str,
                )

        insights = record["musical_insights"]
        if insights is not None:
            logger.info("")
            logger.info(
                "  [3] Musical Domain Insights (Output Entropy H = %.2f / %.2f nats │ Mean Top-1 Conf = %.1f%%):",
                insights["mean_entropy"],
                insights["max_entropy"],
                insights["mean_top1_confidence"] * _PERCENT_SCALE,
            )
            pred_probs = insights["predicted_category_probs"]
            true_shares = insights["target_category_shares"]
            for cat in _TOKEN_CATEGORIES:
                p_mass = pred_probs.get(cat, 0.0)
                t_share = true_shares.get(cat, 0.0)
                bar = _format_ascii_bar(p_mass)
                logger.info(
                    "      %-13s : [%s] Pred: %5.1f%%  (Batch Target Share: %5.1f%%)",
                    cat,
                    bar,
                    p_mass * _PERCENT_SCALE,
                    t_share * _PERCENT_SCALE,
                )

            target_id, target_desc, target_prob = insights["sample_target"]
            top_preds_str = "  │  ".join(
                f"#{rank}: {desc} ({prob * _PERCENT_SCALE:.1f}%)"
                for rank, (_, desc, prob) in enumerate(insights["sample_top_k"], start=1)
            )
            logger.info(
                "      Sample #0 Next-Token Target : %s (Model assigned P = %.1f%%)",
                target_desc,
                target_prob * _PERCENT_SCALE,
            )
            logger.info("      Sample #0 Top-%d Predictions : %s", len(insights["sample_top_k"]), top_preds_str)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log the final multi-epoch progression table and training summary."""
        if not self.epoch_records:
            return

        total_time = max(time.perf_counter() - self._train_start_time, _EPSILON)
        sep = "═" * _BANNER_WIDTH
        sub = "─" * _BANNER_WIDTH

        first_rec = self.epoch_records[0]
        best_rec = min(self.epoch_records, key=lambda r: r["val_loss"])

        logger.info(sep)
        logger.info("STAGE 4/4 — POST-TRAINING ACADEMIC SUMMARY")
        logger.info(sep)
        logger.info(
            "  Completed %d/%d epochs in %.2fs  │  Best Epoch: #%d (val_loss = %.4f, val_ppl = %.2f)",
            len(self.epoch_records),
            self.total_epochs,
            total_time,
            best_rec["epoch"],
            best_rec["val_loss"],
            best_rec["val_ppl"],
        )
        logger.info(
            "  Progression (Epoch 1 -> Best) : Val Loss %.4f -> %.4f (%+.4f)  │  Val Perplexity %.2f -> %.2f",
            first_rec["val_loss"],
            best_rec["val_loss"],
            best_rec["val_loss"] - first_rec["val_loss"],
            first_rec["val_ppl"],
            best_rec["val_ppl"],
        )
        logger.info("  %s", sub)
        logger.info(
            "  %5s │ %10s │ %10s │ %10s │ %9s │ %11s │ %11s │ %s",
            "Epoch",
            "Train Loss",
            "Val Loss",
            "Val PPL",
            "Val Acc",
            "||∇_W L||_2",
            "||ΔW||_2",
            "Checkpoint",
        )
        logger.info("  %s", sub)
        for rec in self.epoch_records:
            global_grad = rec["grad_norms"].get("global", 0.0)
            ckpt_mark = "★ BEST" if rec["is_best"] else f"wait {rec['wait_epochs']}/{self.patience}"
            logger.info(
                "  %5d │ %10.4f │ %10.4f │ %10.2f │ %8.2f%% │ %11.4e │ %11.4e │ %s",
                rec["epoch"],
                rec["train_loss"],
                rec["val_loss"],
                rec["val_ppl"],
                rec["val_acc"],
                global_grad,
                rec["total_update_l2"],
                ckpt_mark,
            )
        logger.info(sep)


# ── Callback Builder & Training Loop ──────────────────────────────────────────


def build_callbacks(
    model_dir: str = config.MODEL_DIR,
    log_dir: str = config.LOG_DIR,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    total_epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    seq_len: int = config.SEQUENCE_LENGTH,
    steps_per_epoch: Optional[int] = None,
    sample_batch: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    inv_vocab: Optional[Dict[int, str]] = None,
    enable_progress_bar: bool = True,
) -> List[tf.keras.callbacks.Callback]:
    """Build the standard set of training callbacks including the academic progress callback.

    Args:
        model_dir: Directory where the best model checkpoint is saved.
        log_dir: Root directory for TensorBoard logs.
        patience: Number of epochs with no val_loss improvement before stopping.
        total_epochs: Total number of planned epochs for progress display.
        batch_size: Batch size for throughput calculations.
        seq_len: Sequence length in tokens for throughput calculations.
        steps_per_epoch: Optional cap on training batches per epoch.
        sample_batch: Optional reference batch (x, y) for per-epoch diagnostics.
        inv_vocab: Optional ID-to-token mapping for musical event descriptions.
        enable_progress_bar: Whether to show a live tqdm batch bar during epochs.

    Returns:
        List of [ModelCheckpoint, EarlyStopping, TensorBoard, AcademicTrainingCallback] callbacks.
    """
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=0,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, "fit", run_id),
        histogram_freq=1,
        update_freq="epoch",
    )

    academic_progress = AcademicTrainingCallback(
        total_epochs=total_epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        steps_per_epoch=steps_per_epoch,
        patience=patience,
        sample_batch=sample_batch,
        inv_vocab=inv_vocab,
        enable_progress_bar=enable_progress_bar,
    )

    return [checkpoint, early_stop, tensorboard, academic_progress]


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
    inv_vocab: Optional[Dict[int, str]] = None,
    enable_progress_bar: bool = True,
) -> tf.keras.callbacks.History:
    """Compile the model, log the pipeline walkthrough, and run the training loop.

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
        inv_vocab: Optional ID-to-token mapping for decoding sample tokens.
        enable_progress_bar: Whether to render live tqdm batch bars during training.

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

    pipeline_info = inspect_pipeline_sample(train_ds, train_ready, inv_vocab=inv_vocab)
    batch_size = pipeline_info["batch_size"] if pipeline_info else config.BATCH_SIZE
    seq_len = pipeline_info["seq_len"] if pipeline_info else config.SEQUENCE_LENGTH
    sample_batch = pipeline_info["sample_batch"] if pipeline_info else None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    logger.info("Model compiled. lr=%.4e  epochs=%d  patience=%d", learning_rate, epochs, patience)

    callbacks = build_callbacks(
        model_dir=model_dir,
        log_dir=log_dir,
        patience=patience,
        total_epochs=epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        steps_per_epoch=steps_per_epoch,
        sample_batch=sample_batch,
        inv_vocab=inv_vocab,
        enable_progress_bar=enable_progress_bar,
    )

    history = model.fit(
        train_ready,
        validation_data=val_ready,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=0,
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
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        dest="max_files",
        help="Limit number of sequence files per split (default: all).",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        dest="steps_per_epoch",
        help="Cap on training batches per epoch (default: full epoch).",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=None,
        dest="validation_steps",
        help="Cap on validation batches per epoch (default: full val set).",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        dest="mixed_precision",
        help="Enable mixed_float16 global policy (recommended on Ampere+ GPUs).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        dest="no_progress_bar",
        help="Disable the live per-batch tqdm bar (keep structured epoch tables).",
    )
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
    val_ds = build_dataset(val_paths, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
        dropout_rate=args.dropout_rate,
    )

    train(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        use_mixed_precision=args.mixed_precision,
        enable_progress_bar=not args.no_progress_bar,
    )
