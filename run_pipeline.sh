#!/usr/bin/env bash
# Full pipeline: setup → preprocess → train → generate
# Usage:
#   ./run_pipeline.sh                        # full run, default settings
#   ./run_pipeline.sh --skip-preprocess      # skip tokenization if already done
#   ./run_pipeline.sh --temperature 0.8 --length 1024

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_PREPROCESS=0
TEMPERATURE=1.0
LENGTH=512
SEED=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --temperature)     TEMPERATURE="$2"; shift 2 ;;
    --length)          LENGTH="$2"; shift 2 ;;
    --seed)            SEED="--seed $2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Python detection ──────────────────────────────────────────────────────────
for PY in python3.10 python3.11 python3 python; do
  if command -v "$PY" &>/dev/null; then
    VER=$("$PY" -c "import sys; print(sys.version_info[:2])")
    # Accept (3, 8) through (3, 11)
    if "$PY" -c "import sys; assert (3,8) <= sys.version_info[:2] <= (3,11)" 2>/dev/null; then
      PYTHON="$PY"
      break
    fi
  fi
done

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: No compatible Python (3.8–3.11) found. TensorFlow requires Python ≤ 3.11."
  echo "Install Python 3.10 or 3.11 and re-run."
  exit 1
fi
echo "Using Python: $PYTHON ($VER)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -d venv ]]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv venv
fi
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ── Directory structure ───────────────────────────────────────────────────────
mkdir -p data/raw data/processed/sequences outputs/models outputs/midi outputs/logs

# ── Dataset check ─────────────────────────────────────────────────────────────
MAESTRO_CSV="data/raw/maestro-v3.0.0/maestro-v3.0.0.csv"
if [[ ! -f "$MAESTRO_CSV" ]]; then
  echo "ERROR: MAESTRO dataset not found at $MAESTRO_CSV"
  echo "Place the extracted maestro-v3.0.0 folder under data/raw/ and re-run."
  exit 1
fi
echo "MAESTRO dataset found."

# ── Preprocessing ─────────────────────────────────────────────────────────────
if [[ $SKIP_PREPROCESS -eq 0 ]]; then
  echo ""
  echo "=== Step 1/3: Preprocessing (MIDI → token sequences) ==="
  python src/preprocessing.py
else
  echo "Skipping preprocessing (--skip-preprocess)."
  if [[ ! -f data/processed/vocabulary.json ]]; then
    echo "ERROR: data/processed/vocabulary.json missing. Run without --skip-preprocess first."
    exit 1
  fi
fi

# ── Training ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2/3: Training ==="
python src/train.py $SEED

# ── Generation ────────────────────────────────────────────────────────────────
MODEL="outputs/models/best_model.keras"
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Model checkpoint not found at $MODEL. Training may have failed."
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT="outputs/midi/generated_${TIMESTAMP}.midi"

echo ""
echo "=== Step 3/3: Generating MIDI ==="
python src/generate.py \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --vocab data/processed/vocabulary.json \
  --length "$LENGTH" \
  --temperature "$TEMPERATURE" \
  $SEED

echo ""
echo "Done. Output saved to: $OUTPUT"
