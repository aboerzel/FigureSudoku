import os

OUTPUT_DIR = os.path.join("output")

CRITIC_LR = 0.002           # Learning rate of the critic
ACTOR_LR = 0.001            # Learning rate of the actor
GAMMA = 0.99                # Discount factor for future rewards
TAU = 0.005                 # Soft update of target networks
OU_NOISE = [[0.0, 0.2],     # [mu, sigma] for axes X, Y, Z
            [0.0, 0.2],
            [0.0, 0.2],
            [0.0, 0.2]]
OU_THETA = 0.15             # theta for OU noise
BATCH_SIZE = 64             # Minibatch size
BUFFER_SIZE = 50000         # Replay buffer size

MAX_EPISODES = 100000         # Total number of episodes to train (default: 3000)
MAX_STEPS_PER_EPISODE = 5000 # Max timesteps in a single episode (default: 100)
TARGET_SCORE = 10.0          # Goal score at which the problem is solved
AVG_SCORE_WINDOW = 100       # Sliding window size for average score calculation
