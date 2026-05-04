# CLAUDE.md — Project Context for Claude Code

## Project Identity

This is an LSTM-based music generation project for a Deep Learning graduate course at ITAM (Mexico City). The goal is to train an LSTM model on the MAESTRO dataset (piano MIDI performances) to generate original piano music.

**Language policy**: Code, comments, docstrings, and commit messages in English. Documentation (README, notebooks markdown cells) in Spanish.

## Reference Repository

Base inspiration: [musikalkemist/generating-melodies-with-rnn-lstm](https://github.com/musikalkemist/generating-melodies-with-rnn-lstm)

Key differences from the reference:
- **Dataset**: We use MAESTRO v3.0.0 (MIDI files), NOT kern/Deutschl files
- **Representation**: Event-based tokenization (NOTE_ON, NOTE_OFF, TIME_SHIFT, SET_VELOCITY), NOT simple pitch encoding
- **Polyphony**: We preserve polyphonic piano textures, NOT monophonic melodies
- **Dynamics**: We encode velocity (loudness), the reference does not

Also consult TensorFlow's official music generation tutorial (uses MAESTRO with note-tuple approach) as a secondary reference for data loading patterns.

## Architecture Decisions

### Data Representation — Event-Based Tokenization

Each MIDI file is converted into a sequence of discrete tokens:

```
Token types:
- NOTE_ON_{pitch}    → 128 tokens (pitches 0-127)
- NOTE_OFF_{pitch}   → 128 tokens (pitches 0-127)
- TIME_SHIFT_{step}  → 100 tokens (10ms to 1000ms, in 10ms increments)
- SET_VELOCITY_{vel} → 32 tokens (velocity quantized to 32 bins)

Total vocabulary size: 128 + 128 + 100 + 32 = 388 tokens
```

This representation is based on the approach used by Magenta's PerformanceRNN. It captures timing, dynamics, and polyphony in a single flat sequence suitable for LSTM processing.

### Model Architecture

```
Input (token index) 
  → Embedding(vocab_size=388, embed_dim=256)
  → LSTM(units=512, return_sequences=True)
  → LSTM(units=512)
  → Dropout(0.3)
  → Dense(vocab_size, activation='softmax')
```

The model is trained with categorical cross-entropy to predict the next token. During inference, temperature-based sampling controls creativity vs. coherence.

### Framework

Primary: **TensorFlow/Keras 2.x** (for consistency with ITAM coursework and the reference repository). PyTorch is acceptable as an alternative if explicitly requested.

## Directory Structure

```
lstm-music-generation/
├── config.py                  # ALL hyperparams and paths centralized here
├── data/
│   ├── raw/                   # MAESTRO MIDIs (gitignored, ~5GB)
│   └── processed/             # Tokenized sequences (numpy arrays or pickle)
├── notebooks/                 # Jupyter notebooks for exploration/demos
├── src/
│   ├── __init__.py
│   ├── data_validation.py     # MIDI integrity checks, stats, outlier detection
│   ├── preprocessing.py       # MIDI → token sequences, vocabulary building
│   ├── dataset.py             # tf.data pipeline or PyTorch DataLoader
│   ├── model.py               # LSTM architecture definition
│   ├── train.py               # Training loop with callbacks and checkpointing
│   └── generate.py            # Seed → token generation → MIDI file output
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── midi/                  # Generated MIDI files
│   └── logs/                  # TensorBoard logs
└── tests/                     # Unit tests for each module
```

## Development Phases

### Phase 0: Project Scaffolding & DevOps

**Goal**: Reproducible environment, clean repo structure, basic CI guardrails — before writing any ML code.

#### 0.1 Repository initialization
- `git init`, create `.gitignore` with standard Python patterns plus project-specific exclusions:
  ```
  data/raw/
  data/processed/
  outputs/models/
  outputs/midi/
  outputs/logs/
  __pycache__/
  *.pyc
  .env
  venv/
  *.egg-info/
  .ipynb_checkpoints/
  ```
- Initial commit with `README.md`, `CLAUDE.md`, `.gitignore`

#### 0.2 Environment & dependencies
- Create `requirements.txt` with pinned versions:
  ```
  tensorflow>=2.12,<3.0
  pretty-midi>=0.2.10
  music21>=9.1
  numpy>=1.24
  pandas>=2.0
  matplotlib>=3.7
  seaborn>=0.12
  tqdm>=4.65
  pytest>=7.4
  flake8>=6.0
  black>=23.0
  isort>=5.12
  ```
- Create `Makefile` with standard targets:
  ```makefile
  .PHONY: setup lint format test clean

  setup:
      python -m venv venv
      . venv/bin/activate && pip install -r requirements.txt

  lint:
      flake8 src/ tests/ --max-line-length 120
      isort --check-only src/ tests/

  format:
      black src/ tests/ --line-length 120
      isort src/ tests/

  test:
      pytest tests/ -v

  clean:
      rm -rf __pycache__ .pytest_cache outputs/logs/*
      find . -name "*.pyc" -delete
  ```

#### 0.3 Directory scaffolding
- Create all directories from the project structure:
  ```bash
  mkdir -p data/{raw,processed} src notebooks outputs/{models,midi,logs} tests
  touch src/__init__.py tests/__init__.py
  ```
- Create `config.py` with all hyperparameters and paths (see config.py Specifications)

#### 0.4 Pre-commit hooks (optional but recommended)
- Configure `.pre-commit-config.yaml`:
  ```yaml
  repos:
    - repo: https://github.com/psf/black
      rev: 23.12.1
      hooks:
        - id: black
          args: [--line-length=120]
    - repo: https://github.com/pycqa/isort
      rev: 5.13.2
      hooks:
        - id: isort
    - repo: https://github.com/pycqa/flake8
      rev: 7.0.0
      hooks:
        - id: flake8
          args: [--max-line-length=120]
  ```

#### 0.5 Smoke test
- Verify the environment works end-to-end:
  ```bash
  make setup
  make lint
  make test    # should pass with 0 tests collected (no tests yet)
  python -c "import pretty_midi; import tensorflow as tf; print('OK')"
  ```

#### Completion criteria for Phase 0
- [ ] Repo initialized with clean first commit
- [ ] `make setup` creates a working venv with all dependencies
- [ ] `make lint` and `make format` run without errors on empty `src/`
- [ ] `make test` executes pytest (0 collected is fine)
- [ ] `config.py` exists with all constants defined
- [ ] All directories from the project structure exist
- [ ] `.gitignore` properly excludes data, models, and caches

---

### Phase 1: Data Validation (`src/data_validation.py`)
- Load MAESTRO metadata CSV, verify all MIDI files exist and parse without errors
- Use `pretty_midi` to extract per-file statistics: duration, note count, pitch range, velocity distribution, estimated tempo
- Generate summary statistics and flag outliers (extremely short/long pieces, unusual pitch ranges)
- Produce histogram plots: pitch distribution, velocity distribution, duration distribution, notes per second
- Validate train/validation/test split proportions from MAESTRO metadata

### Phase 2: Preprocessing (`src/preprocessing.py`)
- Implement `midi_to_events(midi_path) -> List[str]` to convert a MIDI file into a token sequence
- Quantize time deltas into TIME_SHIFT bins (10ms resolution, max 1s per token — chain for longer gaps)
- Quantize velocity into 32 bins
- Build and serialize vocabulary mapping `{token_string: int_id}`
- Convert all MAESTRO training split files and save processed sequences
- Implement `events_to_midi(events, output_path)` for the reverse conversion

### Phase 3: Dataset Pipeline (`src/dataset.py`)
- Create sliding-window sequences of fixed length (default: 64 tokens) as training samples
- Input: sequence[0:N-1], Target: sequence[1:N] (next-token prediction)
- Handle padding and batching
- For TensorFlow: use `tf.data.Dataset` with `.shuffle().batch().prefetch()`
- For PyTorch: implement `torch.utils.data.Dataset` with `DataLoader`

### Phase 4: Model (`src/model.py`)
- Define the LSTM model as specified in Architecture Decisions above
- Accept hyperparameters from `config.py` (embed_dim, lstm_units, dropout_rate, num_layers)
- Include model summary logging

### Phase 5: Training (`src/train.py`)
- Training loop with:
  - Loss: `SparseCategoricalCrossentropy` (if using integer targets) or `CategoricalCrossentropy`
  - Optimizer: Adam with learning_rate from config (default: 0.001)
  - Callbacks: ModelCheckpoint (save best), EarlyStopping (patience=5), TensorBoard
  - Validation at end of each epoch using MAESTRO validation split
- CLI interface with argparse for key hyperparameters

### Phase 6: Generation (`src/generate.py`)
- Load trained model and vocabulary mapping
- Accept a seed sequence (random or user-provided)
- Generate tokens autoregressively with temperature sampling
- Convert generated token sequence back to MIDI using `events_to_midi()`
- CLI interface: `--temperature`, `--length`, `--seed`, `--output`

## config.py Specifications

```python
# All magic numbers live here. Nothing hardcoded in other modules.

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MAESTRO_CSV = "data/raw/maestro-v3.0.0.csv"
VOCAB_PATH = "data/processed/vocabulary.json"
MODEL_DIR = "outputs/models"
MIDI_OUTPUT_DIR = "outputs/midi"
LOG_DIR = "outputs/logs"

# Tokenization
TIME_SHIFT_MS_PER_BIN = 10       # 10ms per time shift bin
MAX_TIME_SHIFT_BINS = 100        # max 1 second per TIME_SHIFT token
VELOCITY_BINS = 32               # quantize velocity into 32 levels
VOCAB_SIZE = 388                 # 128+128+100+32

# Training
SEQUENCE_LENGTH = 64             # tokens per training sample
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
EMBEDDING_DIM = 256
LSTM_UNITS = 512
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5

# Generation
DEFAULT_TEMPERATURE = 1.0
DEFAULT_GENERATION_LENGTH = 512
```

## Coding Standards

- **Type hints** on all function signatures
- **Docstrings** (Google style) on all public functions and classes
- **No magic numbers** — everything from `config.py`
- **Logging** via Python `logging` module, not print statements (except notebooks)
- **Reproducibility**: set random seeds (numpy, tensorflow/torch) at the start of training
- Notebooks are for exploration and visualization only; all reusable logic lives in `src/`
- Tests use `pytest`

## Key Libraries

```
pretty_midi          # MIDI parsing and creation (preferred over mido for this project)
music21              # Music theory utilities (key detection, transposition) if needed
tensorflow>=2.12     # Primary DL framework
numpy
pandas               # MAESTRO metadata handling
matplotlib, seaborn  # Visualization
tqdm                 # Progress bars
pytest               # Testing
```

## Common Pitfalls to Avoid

1. **Do NOT use music21 for MIDI parsing** — it's slow on large files. Use `pretty_midi` for MIDI I/O. Reserve music21 for music-theory operations (key detection, transposition) only if needed.
2. **Do NOT load all MIDI files into memory at once** — MAESTRO is ~1,300 files. Process and serialize incrementally.
3. **Do NOT ignore velocity** — velocity is essential for expressive piano music. The reference repo ignores it; we don't.
4. **Do NOT use one-hot encoding for input** — with vocab_size=388, use an Embedding layer instead. One-hot is wasteful.
5. **Handle sustain pedal (CC64)** — MAESTRO contains pedal events. Decide early whether to include them as tokens or preprocess them into extended note durations. Default: extend note durations based on pedal state.
6. **Sequence length tuning** — 64 tokens captures ~2-4 seconds of music. If musical coherence is poor, experiment with 128 or 256. Document the tradeoff: longer sequences = more memory, slower training.
7. **TIME_SHIFT chaining** — pauses longer than 1 second require multiple consecutive TIME_SHIFT tokens. This is by design.

## Git Conventions

- Branch naming: `feature/<name>`, `fix/<name>`
- Commit messages: imperative mood, concise (e.g., "Add event tokenizer for MIDI files")
- Never commit data files — `data/raw/` and `data/processed/` are gitignored
- Never commit model checkpoints — `outputs/models/` is gitignored

---

## Project Status — 2026-05-02

All six development phases are **complete**. The full pipeline is implemented and tested.

### Phase completion

| Phase | Module | Tests | Status |
|-------|--------|-------|--------|
| 0 | Scaffolding, config, Makefile | — | Done |
| 1 | `src/data_validation.py` | 19 | Done |
| 2 | `src/preprocessing.py` | 30 | Done |
| 3 | `src/dataset.py` | 13 | Done |
| 4 | `src/model.py` | 25 | Done |
| 5 | `src/train.py` | 23 | Done |
| 6 | `src/generate.py` | 27 | Done |

**Total: 137/137 tests passing. Lint clean (flake8 + isort).**

### Runtime notes

- System Python is 3.14.4, which is incompatible with TensorFlow. The venv uses `/usr/bin/python3.10`. Run `make setup` to create it.
- `tensorboard>=2.12` is required and included in `requirements.txt`.

### Next steps

The codebase is ready for a real training run on the MAESTRO v3.0.0 dataset:

1. Download MAESTRO v3.0.0 into `data/raw/maestro-v3.0.0/`
2. Run `python -m src.preprocessing` to tokenize all MIDI files → `data/processed/sequences/`
3. Run `python -m src.train` (or `python src/train.py`) to train the model
4. Run `python src/generate.py --model outputs/models/best_model.keras --output outputs/midi/gen.midi`
- Never commit model checkpoints — `outputs/models/` is gitignored

## Training Optimization Notes

### Fast iteration mode

For quick experiments during development, use the `--fast` flag:

```bash
./run_pipeline.sh --fast
```

This limits training to 100 files, 100 steps/epoch, 20 validation steps, and 10 epochs. A full training run on the complete MAESTRO dataset should use the defaults (no `--fast` flag).

### Key performance levers

| Lever | Flag / Config | Default | Fast mode |
|-------|--------------|---------|-----------|
| Window shift | `DATASET_WINDOW_SHIFT` in config.py | 16 | 16 |
| Max files | `--max-files` | all | 100 |
| Steps per epoch | `--steps-per-epoch` | all | 100 |
| Mixed precision | `--mixed-precision` | off | off (enable manually if GPU supports it) |
| Batch size | `--batch-size` | 64 | 64 |
| Architecture | `--lstm-units` / `--num-layers` | 512 / 2 | 512 / 2 |

### Mixed precision

Enable with `--mixed-precision` when training on GPUs with float16 tensor cores (NVIDIA Ampere+). This roughly halves training time. Do NOT use on CPU — it will be slower.

### TFRecords (future optimization)

The current pipeline uses `tf.numpy_function` to load `.npy` files, which has Python-to-TF overhead. Converting to TFRecords would improve I/O throughput for very large runs but adds preprocessing complexity. Not implemented yet.