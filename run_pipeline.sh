#!/usr/bin/env bash
# Full pipeline: setup → preprocess → train → generate
# Usage:
#   ./run_pipeline.sh                        # full run, default settings
#   ./run_pipeline.sh --skip-preprocess      # skip tokenization if already done
#   ./run_pipeline.sh --temperature 0.8 --length 1024
#   ./run_pipeline.sh --fast                 # quick smoke run (small subset, few epochs)
#   ./run_pipeline.sh --verbose              # also show technical training diagnostics

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_PREPROCESS=0
TEMPERATURE=1.0
LENGTH=512
SEED=""
FAST_FLAG=""
VERBOSE_FLAG=""

# Hide TensorFlow's internal C++ INFO/WARNING logs so the narrated report stays readable.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --temperature)     TEMPERATURE="$2"; shift 2 ;;
    --length)          LENGTH="$2"; shift 2 ;;
    --seed)            SEED="--seed $2"; shift 2 ;;
    --verbose)         VERBOSE_FLAG="--verbose"; shift ;;
    --fast)
        FAST_FLAG="--max-files 100 --steps-per-epoch 100 --validation-steps 20 --epochs 10"
        shift ;;
    *) echo "Argumento desconocido: $1"; exit 1 ;;
  esac
done

step_header() {
  # $1 = step label, $2 = one-line explanation
  echo ""
  echo "################################################################################"
  echo "# $1"
  echo "# $2"
  echo "################################################################################"
}

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
  echo "ERROR: No se encontró un Python compatible (3.8–3.11). TensorFlow requiere Python ≤ 3.11."
  echo "Instala Python 3.10 o 3.11 y vuelve a ejecutar."
  exit 1
fi
echo "Python: $PYTHON ($VER)"
if [[ -n "$FAST_FLAG" ]]; then
  echo "Modo rápido (--fast): 100 piezas, 100 lotes por época, hasta 10 épocas. Útil para probar, no para un modelo final."
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -d venv ]]; then
  echo "Creando entorno virtual (venv)..."
  "$PYTHON" -m venv venv
fi
source venv/bin/activate

echo "Instalando dependencias (puede tardar la primera vez)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ── Directory structure ───────────────────────────────────────────────────────
mkdir -p data/raw data/processed/sequences outputs/models outputs/midi outputs/logs

# ── Dataset check ─────────────────────────────────────────────────────────────
MAESTRO_CSV="data/raw/maestro-v3.0.0/maestro-v3.0.0.csv"
if [[ ! -f "$MAESTRO_CSV" ]]; then
  echo "ERROR: No se encontró el dataset MAESTRO en $MAESTRO_CSV"
  echo "Coloca la carpeta extraída maestro-v3.0.0 dentro de data/raw/ y vuelve a ejecutar."
  exit 1
fi
echo "Dataset MAESTRO encontrado."

# ── Preprocessing ─────────────────────────────────────────────────────────────
if [[ $SKIP_PREPROCESS -eq 0 ]]; then
  step_header "PASO 1/3 · Preprocesamiento (MIDI → secuencias de eventos)" \
              "Convierte cada interpretación de piano en una lista de eventos que el modelo puede leer."
  python -m src.preprocessing
else
  echo "Se omite el preprocesamiento (--skip-preprocess)."
  if [[ ! -f data/processed/vocabulary.json ]]; then
    echo "ERROR: Falta data/processed/vocabulary.json. Ejecuta primero sin --skip-preprocess."
    exit 1
  fi
fi

# ── Training ──────────────────────────────────────────────────────────────────
step_header "PASO 2/3 · Entrenamiento del modelo LSTM" \
            "El modelo aprende a predecir el siguiente evento musical; verás su progreso época por época."
python -m src.train $SEED $FAST_FLAG $VERBOSE_FLAG

# ── Generation ────────────────────────────────────────────────────────────────
MODEL="outputs/models/best_model.keras"
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: No se encontró el modelo en $MODEL. Es posible que el entrenamiento haya fallado."
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT="outputs/midi/generated_${TIMESTAMP}.midi"

step_header "PASO 3/3 · Generación de música" \
            "El modelo entrenado escribe una pieza nueva, evento por evento, y la guarda como MIDI."
python -m src.generate \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --vocab data/processed/vocabulary.json \
  --length "$LENGTH" \
  --temperature "$TEMPERATURE" \
  $SEED $VERBOSE_FLAG

echo ""
echo "################################################################################"
echo "# Listo."
echo "#   Modelo entrenado : $MODEL"
echo "#   Pieza generada   : $OUTPUT"
echo "#   Curvas (TensorBoard): tensorboard --logdir outputs/logs"
echo "################################################################################"
