import os

TRAIN_NAME = f"SUDOKU"

OUTPUT_DIR = os.path.join("output", TRAIN_NAME)
TENSORBOARD_TRAIN_LOG = os.path.join(OUTPUT_DIR, "logs", "train")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "training.log")

MODEL_NAME = "sudoku.zip"
CHECKPOINT_NAME = "checkpoint"

MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_NAME)

# generator parameters
START_LEVEL = 1
MAX_LEVEL = 12
UNIQUE = False
PARTIAL_PROB = 0.0
PARTIAL_MODE = 0

# training parameters
MAX_TIMESTEPS = 50
TOTAL_TIMESTEPS = 500000000
NUM_AGENTS = 10
CHECK_FREQ = 500
REWARD_THRESHOLD = 0.90

# reward parameters
REWARD_SOLVED = 10.0
REWARD_VALID_MOVE_BASE = 0.1
REWARD_INVALID_MOVE = -0.5

# visualization parameters
RENDER_GUI = False

