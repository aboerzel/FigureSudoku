import os

OUTPUT_DIR = os.path.join("output")

TAU = 0.005                 # Soft update of target networks
OU_THETA = 0.15             # theta for OU noise
BATCH_SIZE = 512            # batch size
BUFFER_SIZE = 100000        # Replay buffer size

MAX_EPISODES = 2000000       # Total number of episodes to train (default: 3000)
MAX_STEPS_PER_EPISODE = 200  # Max timesteps in a single episode (default: 100)
AVG_SCORE_WINDOW = 100       # Sliding window size for average score calculation
