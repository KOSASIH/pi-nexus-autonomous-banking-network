# config.py

class Config:
    """Configuration class for PiBrain."""

    # General settings
    PROJECT_NAME = "PiBrain"
    VERSION = "1.0.0"

    # Data settings
    DATA_DIR = "data"
    MODEL_DIR = "models"
    LOG_DIR = "logs"

    # Model settings
    MODEL_TYPE = "neural_network"
    HIDDEN_LAYERS = 2
    HIDDEN_UNITS = 128
    OUTPUT_UNITS = 10
    ACTIVATION_FUNCTION = "relu"
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "mean_squared_error"

    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Evaluation settings
    EVAL_METRICS = ["accuracy", "precision", "recall", "f1_score"]

    # Other settings
    RANDOM_SEED = 42
    VERBOSE = True
