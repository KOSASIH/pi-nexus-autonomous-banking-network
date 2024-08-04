import os

# Project constants
PROJECT_NAME = 'DAICTD'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data constants
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

# Model constants
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_NAME = ' threat_detection_model'
MODEL_CHECKPOINT_FILE = 'checkpoint.pth'

# Training constants
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Logging constants
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_FILE = 'train.log'

# Plotting constants
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')
LOSS_PLOT_FILE = 'loss_curve.png'
ACCURACY_PLOT_FILE = 'accuracy_curve.png'
CONFUSION_MATRIX_PLOT_FILE = 'confusion_matrix.png'
