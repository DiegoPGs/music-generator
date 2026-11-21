"""Training loop with a narrated, human-readable progress report, checkpointing, and TensorBoard logging.

Console output is written in Spanish and designed so that anyone running the
script can follow what the model is doing:

    ETAPA 1/4  Data preparation walkthrough with a real decoded example.
    ETAPA 2/4  Model architecture in plain language plus the random-guess baseline.
    ETAPA 3/4  Live progress bar and a compact per-epoch report with an interpretation.
    ETAPA 4/4  Final summary: why training stopped, best epoch, and next steps.

Passing verbose=True (CLI: --verbose) adds the technical diagnostics: tensor
shapes, per-layer parameter table, weight/update/gradient norms, and entropy.
"""

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
from src.console import (CATEGORY_LABELS_ES, ascii_bar, describe_token_es,
                         format_duration, pitch_to_note_name, rule, section,
                         setup_cli_logging, trend_arrow)
from src.dataset import build_dataset, get_sequence_paths
from src.model import build_model
from src.preprocessing import build_vocabulary, dequantize_velocity

logger = logging.getLogger(__name__)

_TOKEN_CATEGORIES: Tuple[str, ...] = ("NOTE_ON", "NOTE_OFF", "TIME_SHIFT", "SET_VELOCITY")
_DEFAULT_INV_VOCAB: Dict[int, str] = {idx: tok for tok, idx in build_vocabulary().items()}
_EPSILON = 1e-12
_BYTES_PER_FLOAT32 = 4
_BYTES_PER_MB = 1024 * 1024
_MS_PER_SECOND = 1000.0
_PERCENT_SCALE = 100.0
_NUM_STAGES = 4
_BEST_MODEL_FILENAME = "best_model.keras"

# Backwards-compatible aliases (these helpers now live in src/console.py).
_pitch_to_note_name = pitch_to_note_name
_format_ascii_bar = ascii_bar


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Semilla aleatoria fijada en %d (los resultados serán reproducibles).", seed)


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


def describe_token(token_id: int, inv_vocab: Optional[Dict[int, str]] = None) -> str:
    """Format a token ID with its raw name and a compact musical annotation (technical view).

    Examples:
        NOTE_ON_60      -> 'NOTE_ON_60 (C4)'
        NOTE_OFF_60     -> 'NOTE_OFF_60 (C4)'
        TIME_SHIFT_5    -> 'TIME_SHIFT_5 (50ms)'
        SET_VELOCITY_16 -> 'SET_VELOCITY_16 (vel~66)'

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
        return f"{token_str} ({pitch_to_note_name(pitch)})"
    if token_str.startswith("NOTE_OFF_"):
        pitch = int(token_str[len("NOTE_OFF_"):])
        return f"{token_str} ({pitch_to_note_name(pitch)})"
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


def _describe_friendly(token_id: int, inv_vocab: Optional[Dict[int, str]]) -> str:
    """Plain-Spanish description of a token ID using the default vocabulary when none is given."""
    return describe_token_es(token_id, inv_vocab if inv_vocab is not None else _DEFAULT_INV_VOCAB)


# ── Diagnostic Helpers ────────────────────────────────────────────────────────


def inspect_pipeline_sample(
    raw_ds: tf.data.Dataset,
    prepared_ds: tf.data.Dataset,
    inv_vocab: Optional[Dict[int, str]] = None,
    context_tokens: int = config.TRAINING_DEMO_CONTEXT_TOKENS,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Inspect a sample batch and narrate how the training data is built (ETAPA 1/4).

    Extracts one batch from both the raw sliding-window dataset and the prepared
    next-token classification dataset, logs a plain-language explanation with a
    real decoded example, and returns the inspected metadata.

    Args:
        raw_ds: Dataset from build_dataset() yielding (input_seq, target_seq).
        prepared_ds: Dataset from prepare_for_training() yielding (input_seq, next_token).
        inv_vocab: Optional ID-to-token dictionary for decoding sample tokens.
        context_tokens: Number of trailing context tokens from Sample #0 to display.
        verbose: Whether to also log tensor shapes, dtypes, and raw token IDs.

    Returns:
        Dictionary with pipeline shapes, dtypes, sample token walkthrough, and
        'sample_batch' (prep_x, prep_y), or None if the dataset is empty.
    """
    raw_sample = next(iter(raw_ds), None)
    prep_sample = next(iter(prepared_ds), None)
    if raw_sample is None or prep_sample is None:
        logger.warning("El conjunto de entrenamiento está vacío: no hay ejemplos que mostrar.")
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

    section(logger, "PREPARACIÓN DE LOS DATOS", 1, _NUM_STAGES)
    logger.info("  Cada pieza de piano se convirtió en una lista de eventos musicales:")
    logger.info("  tocar una nota, soltarla, esperar un momento o cambiar la intensidad.")
    logger.info(f"  El modelo lee ventanas de {seq_len} eventos seguidos y aprende a adivinar el siguiente.")
    logger.info(
        f"  En cada paso de entrenamiento procesa un lote de {batch_size} ventanas "
        f"({tokens_per_batch:,} eventos en total)."
    )
    logger.info("")
    logger.info(f"  Ejemplo real de los datos (últimos {len(tail_ids)} de {seq_len} eventos de contexto):")
    start_pos = max(0, seq_len - len(tail_ids))
    for offset, tid in enumerate(tail_ids):
        line = f"    evento {start_pos + offset + 1:>3} │ {_describe_friendly(tid, inv_vocab)}"
        if verbose:
            line += f"   [ID {tid}, {describe_token(tid, inv_vocab)}]"
        logger.info(line)
    logger.info("    ───────────┼──────────────────────────────────")
    target_line = f"    siguiente  │ {_describe_friendly(target_id, inv_vocab)}   ← lo que el modelo debe adivinar"
    if verbose:
        target_line += f"   [ID {target_id}, {target_description}]"
    logger.info(target_line)

    if verbose:
        logger.info("")
        logger.info("  Detalle técnico de tensores:")
        logger.info(
            "    Ventana cruda      : input_seq %s (%s) -> target_seq %s (%s)",
            info["raw_input_shape"],
            info["input_dtype"],
            info["raw_target_shape"],
            info["target_dtype"],
        )
        logger.info(
            "    Lote preparado     : X %s -> y %s  (y = target_seq[:, -1])",
            info["prepared_input_shape"],
            info["prepared_target_shape"],
        )

    return info


def summarize_model_layers(model: tf.keras.Model) -> List[Dict[str, Any]]:
    """Extract per-layer parameter counts, output shapes, and roles (technical and plain Spanish).

    Args:
        model: Compiled or uncompiled tf.keras.Model.

    Returns:
        List of dictionaries describing each layer in the model. Each entry has
        'name', 'type', 'output_shape', 'params', 'pct_params', 'role'
        (technical) and 'role_es' (plain-language Spanish explanation).
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
            role = "Recibe lotes de IDs de eventos (batch, seq_len)"
            role_es = "Recibe la ventana de eventos como números (IDs)."
        elif isinstance(layer, tf.keras.layers.Embedding):
            role = f"Mapea IDs [0..{layer.input_dim - 1}] -> vectores densos R^{layer.output_dim}"
            role_es = f"Convierte cada evento en un vector de {layer.output_dim} números que el modelo aprende."
        elif isinstance(layer, tf.keras.layers.LSTM):
            if layer.return_sequences:
                role = f"Codificador recurrente (emite h_t en R^{layer.units} en cada paso)"
                role_es = f"Lee los eventos en orden y guarda una memoria de lo que ha sonado ({layer.units} unidades)."
            else:
                role = f"Codificador secuencia->vector (estado final h_T en R^{layer.units})"
                role_es = f"Resume toda la ventana en un estado final con el contexto ({layer.units} unidades)."
        elif isinstance(layer, tf.keras.layers.Dropout):
            role = f"Regularización: anula {layer.rate * _PERCENT_SCALE:.0f}% de activaciones al entrenar"
            role_es = (
                f"Al entrenar apaga al azar el {layer.rate * _PERCENT_SCALE:.0f}% de las señales "
                "para que no memorice."
            )
        elif isinstance(layer, tf.keras.layers.Dense):
            role = f"Proyección lineal a {layer.units} logits + Softmax"
            role_es = f"Da una probabilidad a cada uno de los {layer.units} eventos posibles: la predicción."
        else:
            role = "Transformación intermedia"
            role_es = "Transformación intermedia."

        summaries.append(
            {
                "name": layer.name,
                "type": layer_type,
                "output_shape": out_shape,
                "params": params,
                "pct_params": pct,
                "role": role,
                "role_es": role_es,
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
        and Sample #0 qualitative predictions. 'sample_top_k' and 'sample_target'
        hold (token_id, technical_description, probability) tuples.
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


def interpret_epoch(
    record: Dict[str, Any],
    prev_record: Optional[Dict[str, Any]],
    patience: int = config.EARLY_STOPPING_PATIENCE,
    plateau_delta: float = config.TRAINING_PLATEAU_DELTA,
    overfit_gap: float = config.TRAINING_OVERFIT_GAP,
) -> Dict[str, str]:
    """Turn an epoch's metrics into a one-sentence, plain-Spanish interpretation.

    Rules are evaluated in priority order:
        first        -> First epoch: establishes the reference point.
        stopping     -> Patience exhausted: EarlyStopping will halt training.
        overfitting  -> Train loss falls while validation loss rises by more than plateau_delta.
        gap          -> Validation loss exceeds train loss by more than overfit_gap.
        improving    -> Validation loss dropped by more than plateau_delta.
        plateau      -> Validation loss changed by at most plateau_delta.
        worse        -> Validation loss rose (without a train/val divergence).

    Args:
        record: Current epoch record with 'train_loss', 'val_loss', 'wait_epochs',
            and optionally 'best_epoch'.
        prev_record: Previous epoch record, or None for the first epoch.
        patience: EarlyStopping patience.
        plateau_delta: Minimum val_loss improvement considered real progress.
        overfit_gap: val_loss - train_loss gap flagged as possible overfitting.

    Returns:
        Dictionary with 'kind' (one of the rule names above) and 'message'.
    """
    train_loss = float(record["train_loss"])
    val_loss = float(record["val_loss"])
    best_epoch = record.get("best_epoch", record.get("epoch", 0))

    if prev_record is None:
        return {
            "kind": "first",
            "message": "Primera época completada: estos valores son la referencia para comparar las siguientes.",
        }

    if int(record.get("wait_epochs", 0)) >= patience:
        return {
            "kind": "stopping",
            "message": (
                f"Se agotó la paciencia ({patience} épocas sin mejorar): el entrenamiento se detiene "
                f"y se conserva el mejor modelo (época {best_epoch})."
            ),
        }

    train_delta = train_loss - float(prev_record["train_loss"])
    val_delta = val_loss - float(prev_record["val_loss"])

    if train_delta < 0 and val_delta > plateau_delta:
        return {
            "kind": "overfitting",
            "message": (
                "Posible sobreajuste: mejoró con las piezas de entrenamiento pero empeoró con piezas nuevas. "
                "Si se repite varias épocas, está memorizando en lugar de generalizar."
            ),
        }

    gap = val_loss - train_loss
    if gap > overfit_gap:
        return {
            "kind": "gap",
            "message": (
                f"Atención: la pérdida de validación está {gap:.3f} por encima de la de entrenamiento; "
                "si la brecha sigue creciendo, el modelo está memorizando."
            ),
        }

    if val_delta < -plateau_delta:
        return {
            "kind": "improving",
            "message": "El modelo sigue aprendiendo y lo aprendido también funciona con piezas que nunca vio.",
        }

    if abs(val_delta) <= plateau_delta:
        return {
            "kind": "plateau",
            "message": (
                f"El aprendizaje se está estancando: la validación casi no cambió ({val_delta:+.4f}). "
                "Es normal hacia el final del entrenamiento."
            ),
        }

    return {
        "kind": "worse",
        "message": (
            "La validación empeoró en esta época; puede ser una fluctuación normal. "
            "Si se repite, el early stopping detendrá el entrenamiento."
        ),
    }


# ── Training Progress Callback ────────────────────────────────────────────────


class AcademicTrainingCallback(tf.keras.callbacks.Callback):
    """Keras callback that narrates training in plain Spanish, with optional technical detail.

    Default view:
        - ETAPA 2/4: the model's layers in plain language, size, and the
          random-guessing baseline (accuracy 1/V, perplexity V), followed by a
          short guide on how to read the metrics.
        - ETAPA 3/4: a live tqdm bar per epoch and a compact per-epoch block with
          train/validation loss, accuracy and perplexity, checkpoint status,
          estimated remaining time, an interpretation sentence (see
          interpret_epoch), and what the model predicts for a sample sequence.
        - ETAPA 4/4: why training ended, best epoch, overall progress versus
          the baseline, a progression table, and the next command to run.

    Verbose view (verbose=True) additionally logs the technical layer table,
    throughput, per-layer weight/update/gradient norms, output entropy, and
    gradient/update columns in the final table.
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
        verbose: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initialize the training progress callback.

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
            verbose: Whether to log the technical diagnostics in addition to the default view.
            checkpoint_path: Where ModelCheckpoint saves the best model (shown to the user).
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
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path or os.path.join(config.MODEL_DIR, _BEST_MODEL_FILENAME)

        self.epoch_records: List[Dict[str, Any]] = []
        self._prev_weights: Optional[Dict[str, List[np.ndarray]]] = None
        self._best_val_loss: float = float("inf")
        self._best_epoch: int = 0
        self._wait_epochs: int = 0
        self._train_start_time: float = 0.0
        self._epoch_start_time: float = 0.0
        self._batches_seen: int = 0
        self._tokens_processed: int = 0
        self._steps_last_epoch: Optional[int] = None
        self._vocab_size: int = config.VOCAB_SIZE
        self._pbar: Optional[tqdm] = None

    # ── Keras hooks ──────────────────────────────────────────────────────────

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log the model explanation, baseline, metric guide, and snapshot initial weights."""
        self._train_start_time = time.perf_counter()
        self.epoch_records.clear()
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._wait_epochs = 0

        _, self._prev_weights = compute_layer_weight_stats(self.model)
        layer_summaries = summarize_model_layers(self.model)

        total_params = int(self.model.count_params())
        mem_mb = (total_params * _BYTES_PER_FLOAT32) / _BYTES_PER_MB
        self._vocab_size = int(self.model.output_shape[-1])
        random_loss = math.log(max(self._vocab_size, 1))
        random_acc = _PERCENT_SCALE / max(self._vocab_size, 1)

        section(logger, "EL MODELO", 2, _NUM_STAGES)
        logger.info("  Cómo procesa cada ventana, de la entrada a la predicción:")
        visible_layers = [item for item in layer_summaries if item["type"] != "InputLayer"]
        for idx, item in enumerate(visible_layers, start=1):
            logger.info(f"    {idx}. {item['name']:<10} → {item['role_es']}")
        logger.info("")
        logger.info(f"  Tamaño: {total_params:,} parámetros ajustables (~{mem_mb:.1f} MB).")
        logger.info(
            f"  Punto de partida: adivinando al azar acertaría {random_acc:.2f}% de las veces y dudaría "
            f"entre {self._vocab_size} opciones (pérdida {random_loss:.2f})."
        )

        if self.verbose:
            logger.info("")
            logger.info("  Detalle técnico por capa:")
            logger.info(
                "    %-12s │ %-12s │ %-16s │ %10s │ %6s │ %s",
                "Capa", "Tipo", "Forma de salida", "Parámetros", "Parte", "Rol matemático",
            )
            for item in layer_summaries:
                logger.info(
                    "    %-12s │ %-12s │ %-16s │ %10s │ %5.1f%% │ %s",
                    item["name"],
                    item["type"],
                    str(item["output_shape"]),
                    f"{item['params']:,}",
                    item["pct_params"],
                    item["role"],
                )

        self._log_metric_guide()

        section(logger, "ENTRENAMIENTO", 3, _NUM_STAGES)
        logger.info(
            f"  Hasta {self.total_epochs} épocas. Una época = una pasada por los datos de entrenamiento"
            + (f" (limitada a {self.steps_per_epoch} lotes)." if self.steps_per_epoch else ".")
        )
        logger.info(
            f"  Se detiene antes si la validación no mejora durante {self.patience} épocas seguidas "
            "(early stopping)."
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Start epoch timers and initialize the per-batch tqdm progress bar."""
        self._epoch_start_time = time.perf_counter()
        self._batches_seen = 0
        self._tokens_processed = 0

        if self.enable_progress_bar:
            self._pbar = tqdm(
                total=self.steps_per_epoch or self._steps_last_epoch,
                desc=f"Época {epoch + 1}/{self.total_epochs}",
                unit=" lotes",
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
            acc = float(logs.get("accuracy", 0.0)) * _PERCENT_SCALE
            elapsed = max(time.perf_counter() - self._epoch_start_time, _EPSILON)
            tok_per_sec = self._tokens_processed / elapsed

            self._pbar.update(1)
            self._pbar.set_postfix(
                {"pérdida": f"{loss:.3f}", "precisión": f"{acc:.1f}%", "eventos/s": f"{tok_per_sec:,.0f}"}
            )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Compute epoch metrics and diagnostics, then log the epoch report."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        logs = logs or {}
        epoch_duration = max(time.perf_counter() - self._epoch_start_time, _EPSILON)
        steps = max(self._batches_seen, 1)
        self._steps_last_epoch = self._batches_seen or None
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

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch = epoch + 1
            self._wait_epochs = 0
            is_best = True
            status_str = f"✔ Mejor modelo hasta ahora → guardado en {self.checkpoint_path}"
        else:
            self._wait_epochs += 1
            is_best = False
            status_str = (
                f"✘ Sin mejora en validación ({self._wait_epochs} de {self.patience} épocas de paciencia). "
                f"El mejor sigue siendo la época {self._best_epoch}."
            )

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
            "best_epoch": self._best_epoch,
            "wait_epochs": self._wait_epochs,
            "weight_stats": weight_stats,
            "total_update_l2": total_update_l2,
            "grad_norms": grad_norms,
            "musical_insights": musical_insights,
            "status": status_str,
        }
        record["interpretation"] = interpret_epoch(record, prev_record, patience=self.patience)
        self.epoch_records.append(record)
        self._log_epoch_report(record, prev_record)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log the final summary: why training ended, best epoch, progression table, and next steps."""
        if not self.epoch_records:
            return

        total_time = max(time.perf_counter() - self._train_start_time, _EPSILON)
        first_rec = self.epoch_records[0]
        last_rec = self.epoch_records[-1]
        best_rec = min(self.epoch_records, key=lambda r: r["val_loss"])
        completed = len(self.epoch_records)

        section(logger, "RESUMEN DEL ENTRENAMIENTO", 4, _NUM_STAGES)
        logger.info(
            f"  Duración total: {format_duration(total_time)} · Épocas completadas: {completed} de {self.total_epochs}"
        )
        if completed >= self.total_epochs:
            reason = "se completaron todas las épocas planeadas."
        elif last_rec["wait_epochs"] >= self.patience:
            reason = f"early stopping — la validación no mejoró durante {self.patience} épocas seguidas."
        else:
            reason = "el entrenamiento se interrumpió antes de completar las épocas planeadas."
        logger.info(f"  Motivo de fin: {reason}")
        logger.info(
            f"  Mejor época: #{best_rec['epoch']} · pérdida de validación {best_rec['val_loss']:.4f} · "
            f"precisión {best_rec['val_acc']:.2f}% · perplejidad {best_rec['val_ppl']:.1f}"
        )
        logger.info(
            f"  Progreso: la perplejidad de validación pasó de {first_rec['val_ppl']:.1f} (época 1) "
            f"a {best_rec['val_ppl']:.1f} (mejor época)."
        )
        if best_rec["val_ppl"] > 0:
            factor = self._vocab_size / best_rec["val_ppl"]
            logger.info(
                f"  En otras palabras: ahora duda entre ~{best_rec['val_ppl']:.0f} opciones en lugar de "
                f"{self._vocab_size} (≈{factor:.0f}× mejor que adivinar al azar)."
            )

        logger.info("")
        header = (
            f"  {'Época':>5} │ {'Pérdida ent.':>12} │ {'Pérdida val.':>12} │ "
            f"{'Precisión val.':>14} │ {'Perplej. val.':>13}"
        )
        if self.verbose:
            header += f" │ {'||∇_W L||_2':>11} │ {'||ΔW||_2':>11}"
        logger.info(header + " │ Nota")
        rule(logger)
        for rec in self.epoch_records:
            line = (
                f"  {rec['epoch']:>5} │ {rec['train_loss']:>12.4f} │ {rec['val_loss']:>12.4f} │ "
                f"{rec['val_acc']:>13.2f}% │ {rec['val_ppl']:>13.2f}"
            )
            if self.verbose:
                line += f" │ {rec['grad_norms'].get('global', 0.0):>11.4e} │ {rec['total_update_l2']:>11.4e}"
            if rec["epoch"] == best_rec["epoch"]:
                note = "★ mejor"
            elif rec["epoch"] == 1:
                note = "referencia"
            elif rec["is_best"]:
                note = "mejoró"
            else:
                note = f"sin mejora {rec['wait_epochs']}/{self.patience}"
            logger.info(f"{line} │ {note}")
        rule(logger)
        logger.info(f"  Modelo guardado: {self.checkpoint_path} (pesos de la época {best_rec['epoch']})")
        logger.info("  Siguiente paso — generar música con el modelo entrenado:")
        logger.info(
            f"    python -m src.generate --model {self.checkpoint_path} "
            f"--output {os.path.join(config.MIDI_OUTPUT_DIR, 'generado.midi')}"
        )
        logger.info("  Para ver las curvas de aprendizaje: tensorboard --logdir %s", config.LOG_DIR)

    # ── Report helpers ───────────────────────────────────────────────────────

    def _log_metric_guide(self) -> None:
        """Log a short, one-time explanation of the metrics shown during training."""
        logger.info("")
        logger.info("  CÓMO LEER LAS MÉTRICAS")
        logger.info("  • Pérdida: qué tan 'sorprendido' está el modelo por el evento correcto. Más baja = mejor.")
        logger.info("  • Precisión: % de veces que su primera opción es exactamente el evento correcto.")
        logger.info(
            f"  • Perplejidad: entre cuántas opciones duda en promedio. Al azar serían {self._vocab_size}; "
            "más baja = mejor."
        )
        logger.info("  • Entrenamiento vs. validación: la validación usa piezas que el modelo nunca estudia.")
        logger.info("    Si solo mejora el entrenamiento, el modelo está memorizando (sobreajuste).")

    def _estimate_remaining(self, epoch_number: int) -> Optional[float]:
        """Estimate remaining seconds as mean epoch duration × epochs left (None if no epochs left)."""
        remaining_epochs = self.total_epochs - epoch_number
        if remaining_epochs <= 0 or not self.epoch_records:
            return None
        mean_duration = float(np.mean([r["duration_sec"] for r in self.epoch_records]))
        return mean_duration * remaining_epochs

    def _log_epoch_report(self, record: Dict[str, Any], prev_record: Optional[Dict[str, Any]]) -> None:
        """Format and log the per-epoch report (plain view, plus technical sections when verbose)."""

        def _change(key: str, lower_is_better: bool, fmt: str) -> str:
            if prev_record is None:
                return ""
            delta = record[key] - prev_record[key]
            return f"({format(delta, fmt)} {trend_arrow(delta, lower_is_better)})"

        rule(logger)
        remaining = self._estimate_remaining(record["epoch"])
        timing = f"duró {format_duration(record['duration_sec'])}"
        if remaining is not None:
            timing += f" · tiempo restante estimado: hasta {format_duration(remaining)}"
        logger.info(f"Época {record['epoch']}/{self.total_epochs} · {timing}")
        logger.info(f"  {'':<12} {'Entrenamiento':<30} {'Validación':<30}")
        logger.info(
            f"  {'Pérdida':<12} {record['train_loss']:<7.4f} {_change('train_loss', True, '+.4f'):<22} "
            f"{record['val_loss']:<7.4f} {_change('val_loss', True, '+.4f')}"
        )
        train_acc_str = f"{record['train_acc']:.2f}%"
        val_acc_str = f"{record['val_acc']:.2f}%"
        logger.info(
            f"  {'Precisión':<12} {train_acc_str:<7} {_change('train_acc', False, '+.2f'):<22} "
            f"{val_acc_str:<7} {_change('val_acc', False, '+.2f')}"
        )
        logger.info(
            f"  {'Perplejidad':<12} {record['train_ppl']:<30.1f} {record['val_ppl']:.1f}  "
            f"(al azar: {self._vocab_size})"
        )
        logger.info(f"  {record['status']}")
        logger.info(f"  Interpretación: {record['interpretation']['message']}")

        insights = record["musical_insights"]
        if insights is not None:
            top_str = "  ".join(
                f"{rank}) {_describe_friendly(tid, self.inv_vocab)} {prob * _PERCENT_SCALE:.1f}%"
                for rank, (tid, _, prob) in enumerate(insights["sample_top_k"], start=1)
            )
            target_id, _, target_prob = insights["sample_target"]
            logger.info("  Qué predice para el ejemplo de la Etapa 1:")
            logger.info(f"    Sus apuestas   : {top_str}")
            logger.info(
                f"    Respuesta real : {_describe_friendly(target_id, self.inv_vocab)} "
                f"(le dio {target_prob * _PERCENT_SCALE:.1f}% de probabilidad)"
            )
            logger.info(f"  {'Tipo de evento':<24} {'Lo que predice el modelo':<33} Real en el lote")
            for cat in _TOKEN_CATEGORIES:
                p_mass = insights["predicted_category_probs"].get(cat, 0.0)
                t_share = insights["target_category_shares"].get(cat, 0.0)
                logger.info(
                    f"  {CATEGORY_LABELS_ES[cat]:<24} [{ascii_bar(p_mass)}] {p_mass * _PERCENT_SCALE:5.1f}%"
                    f"{'':<4} {t_share * _PERCENT_SCALE:5.1f}%"
                )

        if self.verbose:
            self._log_technical_epoch_details(record)

    def _log_technical_epoch_details(self, record: Dict[str, Any]) -> None:
        """Log throughput, per-layer weight/update/gradient norms, and entropy (verbose view only)."""
        logger.info("  [Detalle técnico]")
        logger.info(
            f"    Rendimiento: {record['ms_per_step']:.1f} ms/lote · {record['tokens_per_sec']:,.0f} eventos/s · "
            f"{record['sequences_per_sec']:.1f} ventanas/s"
        )
        if record["weight_stats"]:
            logger.info(
                "    %-12s │ %-10s │ %10s │ %10s │ %11s │ %12s │ %11s",
                "Capa", "Tipo", "||W||_2", "Std(W)", "||ΔW||_2", "||ΔW||/||W||", "||∇_W L||_2",
            )
            grad_norms = record["grad_norms"]
            for w_item in record["weight_stats"]:
                lname = w_item["layer"]
                g_str = f"{grad_norms[lname]:11.4e}" if lname in grad_norms else "        N/A"
                logger.info(
                    "    %-12s │ %-10s │ %10.4f │ %10.4f │ %11.4e │ %12.4e │ %s",
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
            logger.info(
                "    Entropía de salida H = %.2f / %.2f nats · Confianza media top-1 = %.1f%%",
                insights["mean_entropy"],
                insights["max_entropy"],
                insights["mean_top1_confidence"] * _PERCENT_SCALE,
            )
            logger.info(
                "    Top-%d (tokens): %s",
                len(insights["sample_top_k"]),
                "  │  ".join(f"{desc} ({prob * _PERCENT_SCALE:.1f}%)" for _, desc, prob in insights["sample_top_k"]),
            )


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
    verbose: bool = False,
) -> List[tf.keras.callbacks.Callback]:
    """Build the standard set of training callbacks including the progress-report callback.

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
        verbose: Whether the progress report includes technical diagnostics.

    Returns:
        List of [ModelCheckpoint, EarlyStopping, TensorBoard, AcademicTrainingCallback] callbacks.
    """
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, _BEST_MODEL_FILENAME)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
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

    progress_report = AcademicTrainingCallback(
        total_epochs=total_epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        steps_per_epoch=steps_per_epoch,
        patience=patience,
        sample_batch=sample_batch,
        inv_vocab=inv_vocab,
        enable_progress_bar=enable_progress_bar,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
    )

    return [checkpoint, early_stop, tensorboard, progress_report]


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
    verbose: bool = False,
) -> tf.keras.callbacks.History:
    """Compile the model, narrate the data pipeline, and run the training loop.

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
        verbose: Whether to include technical diagnostics in the console report.

    Returns:
        tf.keras.callbacks.History object with per-epoch metrics.
    """
    if not tf.config.list_physical_devices("GPU"):
        logger.warning("No se detectó una GPU: el entrenamiento correrá en CPU y puede tardar bastante más.")
        if use_mixed_precision:
            logger.warning(
                "La precisión mixta (--mixed-precision) está activa sin GPU: en CPU será más lenta, no más rápida."
            )

    train_ready = prepare_for_training(train_ds)
    val_ready = prepare_for_training(val_ds)

    pipeline_info = inspect_pipeline_sample(train_ds, train_ready, inv_vocab=inv_vocab, verbose=verbose)
    batch_size = pipeline_info["batch_size"] if pipeline_info else config.BATCH_SIZE
    seq_len = pipeline_info["seq_len"] if pipeline_info else config.SEQUENCE_LENGTH
    sample_batch = pipeline_info["sample_batch"] if pipeline_info else None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    logger.debug("Modelo compilado: lr=%.4e · épocas=%d · paciencia=%d", learning_rate, epochs, patience)

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
        verbose=verbose,
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

    logger.debug("Entrenamiento terminado. Mejor val_loss: %.4f", min(history.history["val_loss"]))
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
        help="Disable the live per-batch tqdm bar (keep the per-epoch reports).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also show technical diagnostics (tensor shapes, weight/gradient norms, entropy).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_cli_logging(verbose=args.verbose)

    section(logger, "ENTRENAMIENTO DEL MODELO LSTM")
    logger.info(
        f"  Configuración: hasta {args.epochs} épocas · lotes de {args.batch_size} · ventanas de {args.seq_len} "
        f"eventos · tasa de aprendizaje {args.learning_rate:g} · paciencia {args.patience}"
    )
    logger.info(
        f"  Modelo: {args.num_lstm_layers} capas LSTM de {args.lstm_units} unidades · dropout {args.dropout_rate:g}"
    )

    if args.seed is not None:
        set_seeds(args.seed)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Precisión mixta activada (mixed_float16): más rápido en GPUs modernas.")

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
        verbose=args.verbose,
    )
