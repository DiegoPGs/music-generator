# LSTM Music Generation with MAESTRO

Generación de música para piano mediante redes LSTM entrenadas con el dataset MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization).

Proyecto final para la materia de Deep Learning — Maestría en Ciencias de la Computación, ITAM.

## Descripción

Este proyecto implementa un modelo de aprendizaje profundo basado en LSTM (Long Short-Term Memory) capaz de generar secuencias musicales originales para piano. El modelo aprende patrones melódicos, rítmicos y dinámicos a partir de performances reales de piano contenidas en el dataset MAESTRO, y genera nuevas composiciones en formato MIDI.

### Enfoque técnico

El pipeline se compone de cuatro fases:

1. **Validación y exploración de datos** — Verificación de integridad del dataset, análisis estadístico de las piezas (distribución de notas, duraciones, velocidades, tempos) y detección de outliers.
2. **Preprocesamiento y tokenización** — Conversión de archivos MIDI a secuencias de eventos tokenizados (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`, `SET_VELOCITY`) y construcción del vocabulario.
3. **Entrenamiento del modelo** — Arquitectura LSTM con embedding layer, entrenada para predecir el siguiente token en la secuencia.
4. **Generación e inferencia** — Sampling con temperatura a partir de una semilla (seed) para producir secuencias musicales nuevas, con conversión de vuelta a MIDI.

## Dataset

**MAESTRO v3.0.0** — Aproximadamente 200 horas de performances de piano virtuoso, alineadas con precisión entre MIDI y audio.

- Fuente: [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)
- Formato: Archivos MIDI + WAV con metadata en CSV/JSON
- Splits predefinidos: train / validation / test
- Licencia: Creative Commons Attribution Non-Commercial Share-Alike 4.0

## Estructura del proyecto

```
lstm-music-generation/
├── CLAUDE.md                  # Instrucciones para Claude Code
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias Python
├── config.py                  # Hiperparámetros y rutas centralizadas
├── data/
│   ├── raw/                   # Archivos MIDI de MAESTRO (no versionados)
│   └── processed/             # Secuencias tokenizadas serializadas
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   └── 03_training_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_validation.py     # Carga, validación e inspección de MIDIs
│   ├── preprocessing.py       # Tokenización MIDI → secuencias de eventos
│   ├── dataset.py             # Dataset/DataLoader para PyTorch o tf.data
│   ├── model.py               # Arquitectura LSTM
│   ├── train.py               # Loop de entrenamiento
│   └── generate.py            # Inferencia y conversión a MIDI
├── outputs/
│   ├── models/                # Checkpoints del modelo
│   ├── midi/                  # MIDIs generados
│   └── logs/                  # Logs de entrenamiento (TensorBoard)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_generation.py
└── .gitignore
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/DiegoPGs/music-generator
cd lstm-music-generation

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

- Python 3.10+
- TensorFlow 2.x o PyTorch 2.x
- pretty_midi
- music21
- numpy, pandas, matplotlib, seaborn
- tqdm

## Uso rápido

```bash
# 1. Descargar MAESTRO y colocar MIDIs en data/raw/
# 2. Validar datos
python -m src.data_validation

# 3. Preprocesar (tokenizar)
python -m src.preprocessing

# 4. Entrenar
python -m src.train --epochs 50 --batch-size 64

# 5. Generar
python -m src.generate --temperature 1.0 --length 512 --output outputs/midi/generated.mid
```

## Referencia base

Este proyecto toma como referencia conceptual la serie [Generating Melodies with RNN-LSTM](https://github.com/musikalkemist/generating-melodies-with-rnn-lstm) de Valerio Velardo (The Sound of AI), adaptándola significativamente:

- Se usa el dataset MAESTRO (MIDI) en lugar de kern files del Deutschl corpus.
- Se implementa tokenización event-based en lugar de representación por nota simple.
- Se preserva polifonía y dinámica (velocity) del piano.
- La arquitectura soporta secuencias más largas y vocabulario más rico.

También se consulta el [tutorial oficial de TensorFlow para generación musical con RNN](https://www.tensorflow.org/tutorials/audio/music_generation) como referencia complementaria.

## Autores

- Diego Ignacio Puente Gallegos — ITAM, MCC
- Francisco Amando Gómez Domínguez — ITAM, MCC
- Gustavo Adrían Pardo Martínez — ITAM, MCD

## Licencia

MIT
