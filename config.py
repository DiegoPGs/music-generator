# All magic numbers live here. Nothing hardcoded in other modules.

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MAESTRO_BASE_DIR = "data/raw/maestro-v3.0.0"
MAESTRO_CSV = "data/raw/maestro-v3.0.0/maestro-v3.0.0.csv"
VOCAB_PATH = "data/processed/vocabulary.json"
MODEL_DIR = "outputs/models"
MIDI_OUTPUT_DIR = "outputs/midi"
LOG_DIR = "outputs/logs"
VALIDATION_PLOTS_DIR = "outputs/logs/validation_plots"
SEQUENCES_DIR = "data/processed/sequences"

# Data Validation
OUTLIER_IQR_FACTOR = 3.0          # IQR multiplier for outlier detection

# Tokenization
TIME_SHIFT_MS_PER_BIN = 10       # 10ms per time shift bin
MAX_TIME_SHIFT_BINS = 100        # max 1 second per TIME_SHIFT token
VELOCITY_BINS = 32               # quantize velocity into 32 levels
VOCAB_SIZE = 388                 # 128 NOTE_ON + 128 NOTE_OFF + 100 TIME_SHIFT + 32 SET_VELOCITY

# Dataset pipeline
DATASET_SHUFFLE_BUFFER = 10000   # window-level shuffle buffer size
# Stride between sliding windows. 1 = maximum data but high redundancy and slow epochs;
# higher values = fewer samples, faster epochs, less inter-sample correlation.
DATASET_WINDOW_SHIFT = 16

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

# Fast iteration (set via CLI flags; these are just reference values)
FAST_MAX_FILES = 100             # use ~100 files for quick experiments
FAST_LSTM_UNITS = 256            # lighter architecture for prototyping
FAST_EMBEDDING_DIM = 128         # lighter embedding for prototyping
FAST_EPOCHS = 10                 # fewer epochs for smoke tests
