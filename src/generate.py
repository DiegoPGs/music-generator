"""Autoregressive token generation and MIDI export."""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
from src.preprocessing import events_to_midi, load_vocabulary, midi_to_events

logger = logging.getLogger(__name__)


# ── Sampling ──────────────────────────────────────────────────────────────────


def sample_token(probs: np.ndarray, temperature: float = 1.0) -> int:
    """Sample the next token index from a probability distribution.

    At temperature=0 the argmax is returned (greedy/deterministic).
    Higher temperatures flatten the distribution (more random);
    lower temperatures sharpen it (more conservative).

    Args:
        probs: 1-D float array of next-token probabilities (must sum to ~1).
        temperature: Sampling temperature >= 0.

    Returns:
        Sampled token index.
    """
    if temperature == 0.0:
        return int(np.argmax(probs))

    log_probs = np.log(probs + 1e-10) / temperature
    log_probs -= log_probs.max()          # numerical stability before exp
    scaled = np.exp(log_probs)
    scaled /= scaled.sum()
    return int(np.random.choice(len(scaled), p=scaled))


# ── Seed helpers ──────────────────────────────────────────────────────────────


def seed_from_midi(midi_path: str, vocab: Dict[str, int], seq_len: int = config.SEQUENCE_LENGTH) -> List[int]:
    """Tokenize a MIDI file and return the last seq_len token IDs as a seed.

    Args:
        midi_path: Path to a .midi file to use as a seed.
        vocab: Token-to-ID mapping from load_vocabulary.
        seq_len: Number of tokens to take from the end of the file.

    Returns:
        List of integer token IDs of length min(seq_len, len(tokens)).
    """
    events = midi_to_events(midi_path)
    ids = [vocab[e] for e in events if e in vocab]
    return ids[-seq_len:]


def seed_from_random(
    vocab_size: int, seq_len: int = config.SEQUENCE_LENGTH, rng_seed: Optional[int] = None
) -> List[int]:
    """Build a random seed sequence from the vocabulary.

    Args:
        vocab_size: Total vocabulary size.
        seq_len: Number of seed tokens to generate.
        rng_seed: Optional numpy seed for reproducibility.

    Returns:
        List of random token IDs of length seq_len.
    """
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, vocab_size, size=seq_len).tolist()


# ── Core generation ───────────────────────────────────────────────────────────


def generate(
    model: tf.keras.Model,
    seed_tokens: List[int],
    inv_vocab: Dict[int, str],
    length: int = config.DEFAULT_GENERATION_LENGTH,
    temperature: float = config.DEFAULT_TEMPERATURE,
    seq_len: int = config.SEQUENCE_LENGTH,
) -> List[str]:
    """Generate event token strings autoregressively from a seed sequence.

    At each step the model receives the last seq_len tokens as context and
    predicts a probability distribution over the next token. One token is
    sampled and appended; the process repeats for `length` steps.

    Args:
        model: Trained tf.keras.Model from build_model().
        seed_tokens: Starting context as a list of integer token IDs.
        inv_vocab: ID-to-token mapping from load_vocabulary.
        length: Number of new tokens to generate (not counting the seed).
        temperature: Sampling temperature (0 = greedy, >1 = more random).
        seq_len: Context window size fed to the model at each step.

    Returns:
        List of event token strings for the entire sequence (seed + generated).
    """
    if not seed_tokens:
        raise ValueError("seed_tokens must not be empty.")

    token_ids = list(seed_tokens)

    for _ in tqdm(range(length), desc="Generating tokens", leave=False):
        context = token_ids[-seq_len:]
        inp = tf.constant([context], dtype=tf.int32)
        probs = model(inp, training=False).numpy()[0]
        token_ids.append(sample_token(probs, temperature))

    return [inv_vocab[t] for t in token_ids]


# ── End-to-end pipeline ───────────────────────────────────────────────────────


def load_model_and_vocab(
    model_path: str,
    vocab_path: str = config.VOCAB_PATH,
) -> Tuple[tf.keras.Model, Dict[str, int], Dict[int, str]]:
    """Load a saved Keras model and vocabulary files.

    Args:
        model_path: Path to a saved .keras model file.
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        Tuple of (model, token_to_id, id_to_token).

    Raises:
        FileNotFoundError: If either path does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    vocab, inv_vocab = load_vocabulary(vocab_path)
    logger.info("Loaded model from %s (%d params)", model_path, model.count_params())
    return model, vocab, inv_vocab


def generate_midi(
    model_path: str,
    output_path: str,
    vocab_path: str = config.VOCAB_PATH,
    length: int = config.DEFAULT_GENERATION_LENGTH,
    temperature: float = config.DEFAULT_TEMPERATURE,
    seq_len: int = config.SEQUENCE_LENGTH,
    seed_midi_path: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> None:
    """Full pipeline: load model → generate tokens → write MIDI file.

    Args:
        model_path: Path to a saved .keras model checkpoint.
        output_path: Destination .midi file path.
        vocab_path: Path to the vocabulary JSON.
        length: Number of new tokens to generate beyond the seed.
        temperature: Sampling temperature.
        seq_len: Context window size for generation.
        seed_midi_path: Optional MIDI file whose last seq_len tokens seed generation.
            When None a random seed is used instead.
        rng_seed: Random seed for reproducibility (affects both seed and sampling).
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    model, vocab, inv_vocab = load_model_and_vocab(model_path, vocab_path)

    if seed_midi_path is not None:
        logger.info("Using MIDI seed: %s", seed_midi_path)
        seed_tokens = seed_from_midi(seed_midi_path, vocab, seq_len)
    else:
        logger.info("Using random seed (rng_seed=%s)", rng_seed)
        seed_tokens = seed_from_random(len(vocab), seq_len, rng_seed)

    logger.info("Generating %d tokens at temperature=%.2f", length, temperature)
    events = generate(model, seed_tokens, inv_vocab, length=length, temperature=temperature, seq_len=seq_len)

    events_to_midi(events, output_path)
    logger.info("Wrote generated MIDI to %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate piano music with the trained LSTM model.")
    parser.add_argument("--model", required=True, help="Path to saved .keras model file.")
    parser.add_argument("--output", required=True, help="Output .midi file path.")
    parser.add_argument("--vocab", default=config.VOCAB_PATH, help="Path to vocabulary JSON.")
    parser.add_argument("--length", type=int, default=config.DEFAULT_GENERATION_LENGTH)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--seq-len", type=int, default=config.SEQUENCE_LENGTH, dest="seq_len")
    parser.add_argument("--seed-midi", default=None, dest="seed_midi_path", help="MIDI file to seed generation.")
    parser.add_argument("--seed", type=int, default=None, dest="rng_seed", help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    generate_midi(
        model_path=args.model,
        output_path=args.output,
        vocab_path=args.vocab,
        length=args.length,
        temperature=args.temperature,
        seq_len=args.seq_len,
        seed_midi_path=args.seed_midi_path,
        rng_seed=args.rng_seed,
    )
