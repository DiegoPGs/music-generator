"""MIDI dataset validation: integrity checks, per-file statistics, and outlier detection."""

import logging
import os
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

_REQUIRED_CSV_COLUMNS = {"canonical_composer", "canonical_title", "split", "year", "midi_filename", "duration"}
_VALID_SPLITS = {"train", "validation", "test"}


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load and validate the MAESTRO metadata CSV.

    Args:
        csv_path: Path to the MAESTRO CSV file.

    Returns:
        Validated DataFrame with MAESTRO metadata.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MAESTRO CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    unexpected_splits = set(df["split"].unique()) - _VALID_SPLITS
    if unexpected_splits:
        logger.warning("Unexpected split values: %s", unexpected_splits)

    logger.info("Loaded metadata: %d rows, splits=%s", len(df), df["split"].value_counts().to_dict())
    return df


def extract_midi_stats(midi_path: str) -> Dict:
    """Extract per-file statistics from a MIDI file.

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        Dict with keys: duration, note_count, min_pitch, max_pitch, pitch_range,
        mean_velocity, std_velocity, mean_tempo, notes_per_second, parse_error.
        All numeric values are None and parse_error is True if parsing fails.
    """
    null_result: Dict = {
        "duration": None, "note_count": None, "min_pitch": None, "max_pitch": None,
        "pitch_range": None, "mean_velocity": None, "std_velocity": None,
        "mean_tempo": None, "notes_per_second": None, "parse_error": True,
    }
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = [note for inst in pm.instruments if not inst.is_drum for note in inst.notes]

        duration = pm.get_end_time()
        note_count = len(notes)

        if note_count == 0:
            logger.warning("No notes in %s", midi_path)
            return {**null_result, "duration": duration, "note_count": 0, "parse_error": False}

        pitches = [n.pitch for n in notes]
        velocities = [n.velocity for n in notes]
        _, tempos = pm.get_tempo_changes()
        mean_tempo = float(np.mean(tempos)) if len(tempos) > 0 else 120.0

        return {
            "duration": duration,
            "note_count": note_count,
            "min_pitch": int(np.min(pitches)),
            "max_pitch": int(np.max(pitches)),
            "pitch_range": int(np.max(pitches) - np.min(pitches)),
            "mean_velocity": float(np.mean(velocities)),
            "std_velocity": float(np.std(velocities)),
            "mean_tempo": mean_tempo,
            "notes_per_second": note_count / duration if duration > 0 else 0.0,
            "parse_error": False,
        }
    except Exception as exc:
        logger.error("Failed to parse %s: %s", midi_path, exc)
        return null_result


def build_stats_dataframe(df: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    """Process all MIDI files and return a DataFrame with per-file statistics.

    Args:
        df: MAESTRO metadata DataFrame from load_metadata.
        base_dir: Root directory of the MAESTRO dataset; midi_filename paths are relative to it.

    Returns:
        DataFrame combining original metadata columns with per-file statistics.
        The computed duration is stored as 'duration_computed' to avoid collision
        with the CSV 'duration' column (audio file duration).
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MIDI stats"):
        midi_path = os.path.join(base_dir, row["midi_filename"])
        if not os.path.exists(midi_path):
            logger.warning("MIDI not found on disk: %s", midi_path)
            stats = {k: None for k in ["duration_computed", "note_count", "min_pitch", "max_pitch",
                                        "pitch_range", "mean_velocity", "std_velocity",
                                        "mean_tempo", "notes_per_second", "parse_error"]}
        else:
            stats = extract_midi_stats(midi_path)
            stats["duration_computed"] = stats.pop("duration")
        rows.append({**row.to_dict(), **stats})

    return pd.DataFrame(rows)


def flag_outliers(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Flag files with unusual statistics using IQR-based outlier detection.

    Uses config.OUTLIER_IQR_FACTOR times the IQR beyond Q1/Q3 as the threshold.
    Also unconditionally flags parse errors and empty files.

    Args:
        stats_df: DataFrame from build_stats_dataframe.

    Returns:
        Copy of stats_df with added columns 'is_outlier' (bool) and
        'outlier_reasons' (semicolon-delimited string).
    """
    df = stats_df.copy()
    df["is_outlier"] = False
    df["outlier_reasons"] = ""

    def _iqr_bounds(series: pd.Series) -> Tuple[float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - config.OUTLIER_IQR_FACTOR * iqr, q3 + config.OUTLIER_IQR_FACTOR * iqr

    for col, label in [
        ("duration_computed", "duration"),
        ("notes_per_second", "note_density"),
        ("pitch_range", "pitch_range"),
        ("note_count", "note_count"),
    ]:
        valid = df[col].dropna()
        if valid.empty:
            continue
        low, high = _iqr_bounds(valid)
        mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
        df.loc[mask, "is_outlier"] = True
        df.loc[mask, "outlier_reasons"] += f"{label}; "

    error_mask = df["parse_error"].fillna(True).astype(bool)
    df.loc[error_mask, "is_outlier"] = True
    df.loc[error_mask, "outlier_reasons"] += "parse_error; "

    empty_mask = df["note_count"].eq(0)
    df.loc[empty_mask, "is_outlier"] = True
    df.loc[empty_mask, "outlier_reasons"] += "no_notes; "

    n = df["is_outlier"].sum()
    logger.info("Flagged %d / %d outliers (%.1f%%)", n, len(df), 100 * n / len(df))
    return df


def validate_splits(df: pd.DataFrame) -> Dict:
    """Validate train/validation/test split proportions against MAESTRO v3 expectations.

    Args:
        df: MAESTRO metadata DataFrame.

    Returns:
        Dict with keys 'counts', 'proportions', and 'total'.
    """
    counts = df["split"].value_counts().to_dict()
    total = len(df)
    proportions = {k: v / total for k, v in counts.items()}

    logger.info("Split proportions: %s", {k: f"{v:.1%}" for k, v in proportions.items()})

    expected = {"train": 0.75, "validation": 0.10, "test": 0.14}
    for split, exp in expected.items():
        if split in proportions and abs(proportions[split] - exp) > 0.05:
            logger.warning(
                "Split '%s' is %.1f%% (expected ~%.1f%%)",
                split, proportions[split] * 100, exp * 100,
            )

    return {"counts": counts, "proportions": proportions, "total": total}


def plot_distributions(stats_df: pd.DataFrame, output_dir: str) -> None:
    """Generate and save distribution histograms of MIDI dataset statistics.

    Saves five plots: duration, note count, notes per second, mean velocity,
    and pitch coverage across the dataset.

    Args:
        stats_df: DataFrame from build_stats_dataframe or flag_outliers.
        output_dir: Directory where PNG plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid = stats_df[~stats_df["parse_error"].fillna(True).astype(bool)].copy()
    sns.set_theme(style="whitegrid")

    scalar_plots = [
        ("duration_computed", "Duration (seconds)", "duration_distribution.png"),
        ("note_count", "Note Count", "note_count_distribution.png"),
        ("notes_per_second", "Notes per Second", "notes_per_second_distribution.png"),
        ("mean_velocity", "Mean Velocity", "velocity_distribution.png"),
    ]
    for col, xlabel, filename in scalar_plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(valid[col].dropna(), bins=40, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"MAESTRO v3 — {xlabel} (n={len(valid)})")
        fig.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s/%s", output_dir, filename)

    # Pitch coverage: count how many files include each MIDI pitch in their range
    pitch_counts = _aggregate_pitch_coverage(valid)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(128), pitch_counts, color="steelblue", width=1.0)
    ax.set_xlabel("MIDI Pitch (0–127)")
    ax.set_ylabel("Files using pitch")
    ax.set_title(f"MAESTRO v3 — Pitch Coverage (n={len(valid)})")
    fig.savefig(os.path.join(output_dir, "pitch_coverage.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s/pitch_coverage.png", output_dir)


def _aggregate_pitch_coverage(stats_df: pd.DataFrame) -> np.ndarray:
    """Count files that include each MIDI pitch within their min/max range."""
    counts = np.zeros(128, dtype=int)
    for _, row in stats_df.iterrows():
        if pd.notna(row["min_pitch"]) and pd.notna(row["max_pitch"]):
            counts[int(row["min_pitch"]):int(row["max_pitch"]) + 1] += 1
    return counts


def run_validation(
    csv_path: str = config.MAESTRO_CSV,
    base_dir: str = config.MAESTRO_BASE_DIR,
    plots_dir: str = config.VALIDATION_PLOTS_DIR,
) -> pd.DataFrame:
    """Run the full data validation pipeline.

    Loads metadata, extracts per-file stats, flags outliers, validates splits,
    and saves distribution plots.

    Args:
        csv_path: Path to MAESTRO CSV.
        base_dir: Root directory containing MIDI files.
        plots_dir: Directory for output plots.

    Returns:
        Annotated DataFrame with per-file stats and outlier flags.
    """
    logger.info("=== MAESTRO Data Validation ===")

    df = load_metadata(csv_path)
    split_info = validate_splits(df)
    stats_df = build_stats_dataframe(df, base_dir)
    stats_df = flag_outliers(stats_df)
    plot_distributions(stats_df, plots_dir)

    n_errors = int(stats_df["parse_error"].fillna(True).sum())
    n_outliers = int(stats_df["is_outlier"].sum())
    logger.info("=== Summary ===")
    logger.info("Total: %d | Parse errors: %d | Outliers: %d (%.1f%%)",
                len(stats_df), n_errors, n_outliers, 100 * n_outliers / len(stats_df))
    logger.info("Splits: %s", split_info["counts"])

    return stats_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = run_validation()
    outliers = results[results["is_outlier"]][["midi_filename", "outlier_reasons"]]
    print(outliers.to_string())
