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
python -m src.generate --model outputs/models/best_model.keras --temperature 1.0 --length 512 --output outputs/midi/generated.mid
```

O bien, todo el pipeline de una vez: `./run_pipeline.sh` (agrega `--fast` para una prueba corta y `--verbose` para ver los diagnósticos técnicos).

## Cómo leer la salida del entrenamiento

Todos los mensajes del programa están en español y explican qué está pasando. El entrenamiento se narra en cuatro etapas:

1. **Preparación de los datos**: cómo se arma cada ejemplo de entrenamiento, con un fragmento real decodificado ("Tocar Do4", "Esperar 50 ms", …) y el evento que el modelo debe adivinar.
2. **El modelo**: qué hace cada capa en lenguaje sencillo, cuántos parámetros tiene y cuál es el punto de partida si adivinara al azar.
3. **Entrenamiento**: una barra de progreso por época y, al terminar cada una, un bloque como este:

```
Época 3/50 · duró 4 min 12 s · tiempo restante estimado: hasta 3 h 17 min
               Entrenamiento                  Validación
  Pérdida      2.8117  (-0.0376 ↓ mejora)     2.9240  (-0.1228 ↓ mejora)
  Precisión    15.04%  (+1.56 ↑ mejora)       16.67%  (-2.08 ↓ empeora)
  Perplejidad  16.6                           18.6  (al azar: 388)
  ✔ Mejor modelo hasta ahora → guardado en outputs/models/best_model.keras
  Interpretación: El modelo sigue aprendiendo y lo aprendido también funciona con piezas que nunca vio.
  Qué predice para el ejemplo de la Etapa 1:
    Sus apuestas   : 1) Esperar 500 ms 19.3%  2) Esperar 250 ms 16.7%  3) Tocar Si4 (B4) 7.5%
    Respuesta real : Soltar Mi4 (E4) (le dio 5.3% de probabilidad)
  Tipo de evento           Lo que predice el modelo          Real en el lote
  Notas que empiezan       [█████░░░░░░░░░░░░░░░]  26.6%      21.9%
  ...
```

4. **Resumen**: por qué terminó (todas las épocas o *early stopping*), cuál fue la mejor época, cuánto mejoró respecto al azar, una tabla con todas las épocas y el comando para generar música.

| Métrica | Qué significa | Mejor si… |
|---------|---------------|-----------|
| Pérdida | Qué tan "sorprendido" está el modelo por el evento correcto | baja |
| Precisión | % de veces que su primera opción es exactamente el evento correcto | sube |
| Perplejidad | Entre cuántas opciones duda en promedio (al azar serían 388) | baja |
| Entrenamiento vs. validación | La validación usa piezas que el modelo nunca estudia | ambas mejoran juntas |

La línea **Interpretación** resume la época en una frase:

- *Sigue aprendiendo*: la pérdida de validación bajó de forma clara.
- *Se está estancando*: la validación casi no cambió (umbral `TRAINING_PLATEAU_DELTA` en `config.py`).
- *Posible sobreajuste*: mejora en entrenamiento pero empeora en validación, o la brecha entre ambas supera `TRAINING_OVERFIT_GAP`.
- *Se agotó la paciencia*: después de `EARLY_STOPPING_PATIENCE` épocas sin mejorar, el entrenamiento se detiene y se conserva el mejor modelo.

Con `--verbose` (en `src.train`, `src.generate` o `run_pipeline.sh`) se agregan los diagnósticos técnicos: formas de los tensores, tabla de capas, normas de pesos, actualizaciones y gradientes (`||W||`, `||ΔW||`, `||∇_W L||`), entropía de la salida y el `model.summary()` de Keras. Las mismas métricas quedan en TensorBoard: `tensorboard --logdir outputs/logs`.

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
- Juan Ignacio Hernández Muziño — ITAM, MCC

## Licencia

MIT
