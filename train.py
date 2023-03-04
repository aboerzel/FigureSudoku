import gym
import numpy as np
#from sb3_contrib import RecurrentPPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from figure_sudoko_env import FigureSudokuEnv, Reward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, DDPG, A2C

from stable_baselines3.ppo import MlpPolicy
from gym.envs.registration import register


class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        # Overwrite the done signal when
        if self.current_step >= self.max_steps:
            done = True
            #reward = Reward.LOSS.value
            # Update the info dict to signal that the limit was exceeded
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def render(self, **kwargs):
        self.env.render(**kwargs)


class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self):
        """
        Reset the environment
        """
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info


MODEL_PATH = "output/sudoku"
MAX_TIMESTEPS = 200
EPISODES = 1000000


def train_sudoku(gui, stop):
    # create environment
    env = FigureSudokuEnv(level=5, gui=gui)
    env = TimeLimitWrapper(env, max_steps=MAX_TIMESTEPS)
    #env = NormalizeActionWrapper(env)
    check_env(env)

    # Example for the FigureSudoku environment
    #env_id = "FigureSudoku-v1"
    #register(
    #    # unique identifier for the env `name-version`
    #    id=env_id,
    #    # path to the class for creating the env
    #    # Note: entry_point also accept a class as input (and not only a string)
    #    entry_point="figure_sudoko_env:FigureSudokuEnv",
    #    # Max number of steps per episode, using a `TimeLimitWrapper`
    ##    max_episode_steps=200,
    #)

    #env1 = gym.make(env_id, num_env=8)

    #model = SAC("MlpPolicy", env=env, verbose=1, batch_size=256, learning_rate=3e-5, tau=0.005, ent_coef='auto_0.9', use_sde=True, tensorboard_log="runs", device="auto")

    model = A2C("MlpPolicy", env=env, verbose=1, learning_rate=3e-5, tensorboard_log="runs", device="cuda")
    #model = PPO(MlpPolicy, env=env, verbose=1, batch_size=64, use_sde=False, learning_rate=3e-5, tensorboard_log="runs", device="cuda")
    #model = PPO.load(MODEL_PATH)
    #model.set_env(env)

    #for epoch in range(1, EPISODES):
    #model.learn(total_timesteps=EPISODES, reset_num_timesteps=False)
    #model.learn(total_timesteps=EPISODES, reset_num_timesteps=False, progress_bar=True)
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(MODEL_PATH)

    del model  # delete trained model to demonstrate loading
    model = PPO.load(MODEL_PATH)

    obs = env.reset()
    for i in range(30):
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
