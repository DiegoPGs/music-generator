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
VOCAB_SIZE = 388                 # 128 NOTE_ON + 128 NOTE_OFF + 100 TIME_SHIFT + 32 SET_VELOCITY

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
