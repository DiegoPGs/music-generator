"""Human-readable console reporting helpers shared by every pipeline stage.

All user-facing messages are written in Spanish (the project's documentation
language); code, comments, and docstrings stay in English.
"""

import logging
import sys
from typing import Dict, Optional

import config

_NOTE_NAMES_EN = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NOTE_NAMES_ES = ("Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600

CATEGORY_LABELS_ES: Dict[str, str] = {
    "NOTE_ON": "Notas que empiezan",
    "NOTE_OFF": "Notas que terminan",
    "TIME_SHIFT": "Avances de tiempo",
    "SET_VELOCITY": "Cambios de intensidad",
}

_LEVEL_PREFIXES: Dict[int, str] = {
    logging.WARNING: "AVISO: ",
    logging.ERROR: "ERROR: ",
    logging.CRITICAL: "ERROR CRÍTICO: ",
}


class _FriendlyFormatter(logging.Formatter):
    """Formatter that prints INFO/DEBUG messages bare and prefixes warnings/errors in Spanish."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record, adding a Spanish severity prefix only for warnings and errors."""
        message = super().format(record)
        prefix = _LEVEL_PREFIXES.get(record.levelno, "")
        return f"{prefix}{message}"


def setup_cli_logging(verbose: bool = False) -> None:
    """Configure root logging for command-line entry points.

    INFO messages are printed without the 'INFO module:' prefix so the output
    reads like a narrated report. With verbose=True, the project's 'src'
    loggers emit DEBUG messages too (e.g. the raw Keras model summary).

    Args:
        verbose: Whether to enable DEBUG-level output for project modules.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_FriendlyFormatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("src").setLevel(logging.DEBUG if verbose else logging.INFO)


def section(logger: logging.Logger, title: str, step: Optional[int] = None, total: Optional[int] = None) -> None:
    """Log a double-ruled section banner, optionally numbered as 'ETAPA step/total'.

    Args:
        logger: Logger to write to.
        title: Section title.
        step: Optional 1-based stage number.
        total: Optional total number of stages.
    """
    heading = f"ETAPA {step}/{total} · {title}" if step is not None and total is not None else title
    logger.info("")
    logger.info("═" * config.CONSOLE_WIDTH)
    logger.info(heading)
    logger.info("═" * config.CONSOLE_WIDTH)


def rule(logger: logging.Logger) -> None:
    """Log a single horizontal rule."""
    logger.info("─" * config.CONSOLE_WIDTH)


def pitch_to_note_name(pitch: int) -> str:
    """Convert a MIDI pitch (0–127) to scientific pitch notation (e.g. 60 -> 'C4')."""
    octave = (pitch // config.MIDI_NOTES_PER_OCTAVE) - config.MIDI_OCTAVE_OFFSET
    return f"{_NOTE_NAMES_EN[pitch % config.MIDI_NOTES_PER_OCTAVE]}{octave}"


def pitch_to_note_name_es(pitch: int) -> str:
    """Convert a MIDI pitch to Spanish solfège notation with octave (e.g. 60 -> 'Do4')."""
    octave = (pitch // config.MIDI_NOTES_PER_OCTAVE) - config.MIDI_OCTAVE_OFFSET
    return f"{_NOTE_NAMES_ES[pitch % config.MIDI_NOTES_PER_OCTAVE]}{octave}"


def describe_event_es(token_str: str) -> str:
    """Describe an event token string in plain Spanish.

    Examples:
        'NOTE_ON_60'      -> 'Tocar Do4 (C4)'
        'NOTE_OFF_60'     -> 'Soltar Do4 (C4)'
        'TIME_SHIFT_5'    -> 'Esperar 50 ms'
        'SET_VELOCITY_16' -> 'Intensidad ≈ 66 (de 127)'

    Args:
        token_str: Event token such as 'NOTE_ON_60'.

    Returns:
        Human-readable Spanish description, or the token itself if unknown.
    """
    # Imported lazily to avoid a circular import (preprocessing uses this module).
    from src.preprocessing import dequantize_velocity

    if token_str.startswith("NOTE_ON_"):
        pitch = int(token_str[len("NOTE_ON_"):])
        return f"Tocar {pitch_to_note_name_es(pitch)} ({pitch_to_note_name(pitch)})"
    if token_str.startswith("NOTE_OFF_"):
        pitch = int(token_str[len("NOTE_OFF_"):])
        return f"Soltar {pitch_to_note_name_es(pitch)} ({pitch_to_note_name(pitch)})"
    if token_str.startswith("TIME_SHIFT_"):
        steps = int(token_str[len("TIME_SHIFT_"):])
        return f"Esperar {steps * config.TIME_SHIFT_MS_PER_BIN} ms"
    if token_str.startswith("SET_VELOCITY_"):
        vel_bin = int(token_str[len("SET_VELOCITY_"):])
        return f"Intensidad ≈ {dequantize_velocity(vel_bin)} (de 127)"
    return token_str


def describe_token_es(token_id: int, inv_vocab: Dict[int, str]) -> str:
    """Describe a token ID in plain Spanish using an ID-to-token mapping.

    Args:
        token_id: Integer token index.
        inv_vocab: ID-to-token mapping.

    Returns:
        Spanish description (see describe_event_es), or 'Token desconocido N'.
    """
    token_str = inv_vocab.get(int(token_id))
    if token_str is None:
        return f"Token desconocido {int(token_id)}"
    return describe_event_es(token_str)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short Spanish string (e.g. '1 h 03 min', '3 min 12 s', '4.2 s')."""
    seconds = max(0.0, float(seconds))
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f} s"
    total = int(round(seconds))
    hours, rem = divmod(total, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rem, _SECONDS_PER_MINUTE)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s"


def ascii_bar(fraction: float, width: int = config.TRAINING_BAR_WIDTH) -> str:
    """Build a fixed-width bar for a fraction in [0, 1] (values outside are clamped)."""
    clamped = max(0.0, min(1.0, float(fraction)))
    filled = int(round(clamped * width))
    return "█" * filled + "░" * (width - filled)


def trend_arrow(delta: float, lower_is_better: bool = True) -> str:
    """Return an arrow plus a Spanish verdict for a metric change.

    Args:
        delta: Change versus the previous value (current - previous).
        lower_is_better: Whether a decrease is an improvement (loss) or not (accuracy).

    Returns:
        String such as '↓ mejora', '↑ empeora', or '→ igual'.
    """
    if delta == 0:
        return "→ igual"
    improved = (delta < 0) == lower_is_better
    arrow = "↓" if delta < 0 else "↑"
    return f"{arrow} {'mejora' if improved else 'empeora'}"
