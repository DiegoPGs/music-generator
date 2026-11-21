"""MIDI to event-token sequence preprocessing for LSTM training."""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

import config
from src.console import format_duration, section, setup_cli_logging

logger = logging.getLogger(__name__)


# ── Vocabulary ────────────────────────────────────────────────────────────────


def build_vocabulary() -> Dict[str, int]:
    """Build the fixed 388-token vocabulary for event-based MIDI representation.

    Token layout (in ID order):
        NOTE_ON_0..127, NOTE_OFF_0..127, TIME_SHIFT_1..100, SET_VELOCITY_0..31

    Returns:
        Dict mapping token string to integer ID (0-indexed, contiguous).
    """
    vocab: Dict[str, int] = {}
    idx = 0
    for pitch in range(128):
        vocab[f"NOTE_ON_{pitch}"] = idx
        idx += 1
    for pitch in range(128):
        vocab[f"NOTE_OFF_{pitch}"] = idx
        idx += 1
    for step in range(1, config.MAX_TIME_SHIFT_BINS + 1):
        vocab[f"TIME_SHIFT_{step}"] = idx
        idx += 1
    for vel_bin in range(config.VELOCITY_BINS):
        vocab[f"SET_VELOCITY_{vel_bin}"] = idx
        idx += 1
    assert len(vocab) == config.VOCAB_SIZE, f"Expected {config.VOCAB_SIZE} tokens, got {len(vocab)}"
    return vocab


def save_vocabulary(vocab: Dict[str, int], path: str) -> None:
    """Serialize vocabulary to a JSON file.

    Args:
        vocab: Token-to-ID mapping from build_vocabulary.
        path: Destination file path (parent directories are created if needed).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)
    logger.info("  Vocabulario guardado: %d tipos de evento posibles → %s", len(vocab), path)


def load_vocabulary(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary from JSON and return forward and inverse mappings.

    Args:
        path: Path to JSON vocabulary file produced by save_vocabulary.

    Returns:
        Tuple of (token_to_id, id_to_token).

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vocabulary not found: {path}")
    with open(path) as f:
        vocab: Dict[str, int] = json.load(f)
    inv_vocab: Dict[int, str] = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


# ── Quantization ──────────────────────────────────────────────────────────────


def quantize_velocity(velocity: int) -> int:
    """Map MIDI velocity (0–127) to a velocity bin (0 to VELOCITY_BINS-1).

    Args:
        velocity: MIDI velocity 0–127.

    Returns:
        Bin index in range [0, VELOCITY_BINS).
    """
    return min(int(velocity) * config.VELOCITY_BINS // 128, config.VELOCITY_BINS - 1)


def dequantize_velocity(vel_bin: int) -> int:
    """Map a velocity bin back to a representative MIDI velocity.

    Args:
        vel_bin: Bin index 0 to VELOCITY_BINS-1.

    Returns:
        MIDI velocity in range [1, 127].
    """
    bin_width = 128 // config.VELOCITY_BINS
    return max(1, vel_bin * bin_width + bin_width // 2)


# ── Sustain pedal ─────────────────────────────────────────────────────────────


def _get_pedal_intervals(instrument: pretty_midi.Instrument) -> List[Tuple[float, float]]:
    """Return (press_time, release_time) intervals for CC64 sustain pedal."""
    intervals: List[Tuple[float, float]] = []
    press_time: Optional[float] = None
    for cc in sorted(instrument.control_changes, key=lambda c: c.time):
        if cc.number != 64:
            continue
        if cc.value >= 64 and press_time is None:
            press_time = cc.time
        elif cc.value < 64 and press_time is not None:
            intervals.append((press_time, cc.time))
            press_time = None
    return intervals


def _apply_sustain_pedal(instrument: pretty_midi.Instrument) -> List[pretty_midi.Note]:
    """Return notes with end times extended through active sustain pedal presses.

    Args:
        instrument: A PrettyMIDI instrument with notes and control changes.

    Returns:
        List of Note objects with adjusted end times.
    """
    pedal_intervals = _get_pedal_intervals(instrument)
    if not pedal_intervals:
        return list(instrument.notes)

    result = []
    for note in instrument.notes:
        new_end = note.end
        for press, release in pedal_intervals:
            if press < note.end <= release:
                new_end = release
                break
        result.append(pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start, end=new_end))
    return result


# ── Raw event collection ──────────────────────────────────────────────────────


@dataclass(order=True)
class _RawEvent:
    """Sortable MIDI event before tokenization.

    Sort order: time → sort_key (OFF=0 before ON=1) → pitch → velocity.
    NOTE_OFFs precede NOTE_ONs at the same tick, matching standard MIDI convention.
    """

    time: float
    sort_key: int  # 0 = note_off, 1 = note_on
    pitch: int
    velocity: int
    event_type: str


def _collect_events(midi_path: str) -> List[_RawEvent]:
    """Load a MIDI file and return all note events sorted by time."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    events: List[_RawEvent] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in _apply_sustain_pedal(instrument):
            events.append(_RawEvent(note.start, 1, note.pitch, note.velocity, "note_on"))
            events.append(_RawEvent(note.end, 0, note.pitch, 0, "note_off"))
    events.sort()
    return events


# ── Main tokenization API ─────────────────────────────────────────────────────


def midi_to_events(midi_path: str) -> List[str]:
    """Convert a MIDI file to a flat sequence of event token strings.

    Time gaps are encoded as TIME_SHIFT tokens (10 ms per bin, max 1 s each);
    gaps longer than 1 s are expressed as multiple chained TIME_SHIFT_100 tokens.
    A SET_VELOCITY token is emitted only when the quantized velocity changes.
    Sustain pedal (CC64) is pre-processed into extended note durations.

    Args:
        midi_path: Path to the .midi file.

    Returns:
        List of token strings, e.g. ['SET_VELOCITY_10', 'NOTE_ON_60', ...].
    """
    raw_events = _collect_events(midi_path)
    tokens: List[str] = []
    current_time = 0.0
    current_vel_bin = -1  # -1 forces a SET_VELOCITY on the first NOTE_ON

    for event in raw_events:
        delta_ms = max(0.0, (event.time - current_time) * 1000.0)
        if delta_ms > 0:
            remaining = int(round(delta_ms / config.TIME_SHIFT_MS_PER_BIN))
            while remaining > 0:
                step = min(remaining, config.MAX_TIME_SHIFT_BINS)
                tokens.append(f"TIME_SHIFT_{step}")
                remaining -= step
            current_time = event.time

        if event.event_type == "note_on":
            vel_bin = quantize_velocity(event.velocity)
            if vel_bin != current_vel_bin:
                tokens.append(f"SET_VELOCITY_{vel_bin}")
                current_vel_bin = vel_bin
            tokens.append(f"NOTE_ON_{event.pitch}")
        else:
            tokens.append(f"NOTE_OFF_{event.pitch}")

    return tokens


def events_to_tokens(events: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """Map event token strings to integer IDs.

    Args:
        events: List of token strings from midi_to_events.
        vocab: Token-to-ID mapping from build_vocabulary or load_vocabulary.

    Returns:
        int32 numpy array of token IDs.

    Raises:
        KeyError: If an unrecognised token string is encountered.
    """
    return np.array([vocab[e] for e in events], dtype=np.int32)


def tokens_to_events(token_ids: np.ndarray, inv_vocab: Dict[int, str]) -> List[str]:
    """Map integer token IDs back to event token strings.

    Args:
        token_ids: int32 array from events_to_tokens.
        inv_vocab: ID-to-token mapping (second element of load_vocabulary return value).

    Returns:
        List of event token strings.
    """
    return [inv_vocab[int(i)] for i in token_ids]


def events_to_midi(events: List[str], output_path: str, tempo: float = 120.0) -> None:
    """Reconstruct a MIDI file from a sequence of event token strings.

    NOTE_ON tokens without a matching NOTE_OFF are closed at the end of the sequence.

    Args:
        events: Token strings as produced by midi_to_events or the generation module.
        output_path: Destination .midi file path.
        tempo: BPM for the output file (original tempo is not recoverable from tokens).
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    current_time = 0.0
    current_velocity = dequantize_velocity(config.VELOCITY_BINS // 2)
    open_notes: Dict[int, Tuple[float, int]] = {}  # pitch -> (start_time, velocity)

    for token in events:
        if token.startswith("TIME_SHIFT_"):
            step = int(token[len("TIME_SHIFT_"):])
            current_time += step * config.TIME_SHIFT_MS_PER_BIN / 1000.0
        elif token.startswith("SET_VELOCITY_"):
            vel_bin = int(token[len("SET_VELOCITY_"):])
            current_velocity = dequantize_velocity(vel_bin)
        elif token.startswith("NOTE_ON_"):
            pitch = int(token[len("NOTE_ON_"):])
            open_notes[pitch] = (current_time, current_velocity)
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token[len("NOTE_OFF_"):])
            if pitch in open_notes:
                start, vel = open_notes.pop(pitch)
                end = max(current_time, start + 0.01)
                instrument.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))

    # Close notes that were never released
    for pitch, (start, vel) in open_notes.items():
        instrument.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=max(current_time, start + 0.01))
        )

    pm.instruments.append(instrument)
    pm.write(output_path)
    logger.debug("Wrote %d notes to %s", len(instrument.notes), output_path)


# ── Dataset processing ────────────────────────────────────────────────────────


def process_dataset(
    csv_path: str = config.MAESTRO_CSV,
    base_dir: str = config.MAESTRO_BASE_DIR,
    vocab_path: str = config.VOCAB_PATH,
    output_dir: str = config.SEQUENCES_DIR,
    splits: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Tokenize all MAESTRO MIDI files and save as int32 numpy arrays.

    Files are processed one at a time to avoid loading the full dataset into memory.
    Output mirrors the MAESTRO directory structure under output_dir, replacing
    .midi extensions with .npy. Progress and a final summary are logged in Spanish.

    Args:
        csv_path: Path to the MAESTRO metadata CSV.
        base_dir: Root directory containing the MIDI files referenced by the CSV.
        vocab_path: Path to save the vocabulary JSON.
        output_dir: Root directory for output .npy sequence files.
        splits: Splits to process. Defaults to all three (train, validation, test).

    Returns:
        Dict with 'ok' (files processed), 'errors' (files that failed), and
        'total_tokens' (events written across all files).
    """
    if splits is None:
        splits = ["train", "validation", "test"]

    section(logger, "PREPROCESAMIENTO: MIDI → SECUENCIAS DE EVENTOS")
    logger.info("  Cada archivo MIDI se convierte en una lista de eventos que el modelo puede leer:")
    logger.info("  tocar nota, soltar nota, esperar (en pasos de %d ms) y cambiar intensidad.",
                config.TIME_SHIFT_MS_PER_BIN)
    logger.info("  El pedal de sustain se aplica alargando la duración de las notas.")

    vocab = build_vocabulary()
    save_vocabulary(vocab, vocab_path)

    df = pd.read_csv(csv_path)
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    per_split = df["split"].value_counts().to_dict()
    logger.info("  Piezas a procesar: %d (%s)", len(df),
                ", ".join(f"{split}: {per_split.get(split, 0)}" for split in splits))

    start_time = time.perf_counter()
    ok, errors, total_tokens = 0, 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizando archivos MIDI", unit=" archivos"):
        midi_path = os.path.join(base_dir, row["midi_filename"])
        rel = os.path.splitext(row["midi_filename"])[0] + ".npy"
        out_path = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            token_ids = events_to_tokens(midi_to_events(midi_path), vocab)
            np.save(out_path, token_ids)
            ok += 1
            total_tokens += int(len(token_ids))
        except Exception as exc:
            logger.error("No se pudo procesar %s — %s", row["midi_filename"], exc)
            errors += 1

    elapsed = time.perf_counter() - start_time
    section(logger, "RESUMEN DEL PREPROCESAMIENTO")
    logger.info("  Piezas convertidas     : %d", ok)
    error_hint = " (revisa los mensajes de ERROR de arriba)" if errors else ""
    logger.info("  Piezas con error       : %d%s", errors, error_hint)
    logger.info("  Eventos generados      : %s", f"{total_tokens:,}")
    if ok:
        logger.info("  Promedio por pieza     : %s eventos", f"{total_tokens // ok:,}")
    logger.info("  Tiempo total           : %s", format_duration(elapsed))
    logger.info("  Secuencias guardadas en: %s", output_dir)
    logger.info("  Siguiente paso: python -m src.train")

    return {"ok": ok, "errors": errors, "total_tokens": total_tokens}


if __name__ == "__main__":
    setup_cli_logging()
    process_dataset()
