import os

import gym
import numpy as np
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

import config
from callbacks import SaveOnBestTrainingRewardCallback
from figure_sudoko_env import FigureSudokuEnv, Reward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, DDPG, A2C

from stable_baselines3.ppo import MlpPolicy


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


def make_sudoku_env(env_id, level):
    env = FigureSudokuEnv(level=level, gui=None)
    env = TimeLimitWrapper(env, max_steps=config.MAX_TIMESTEPS)
    check_env(env)
    env = Monitor(env, f'{config.OUTPUT_DIR}/env{env_id}')
    return env


def make_env(env_id, level):
    def _thunk():
        env = make_sudoku_env(env_id=env_id, level=level)
        return env

    return _thunk


def make_vec_env(num_envs, level):
    envs = SubprocVecEnv([make_env(env_id=i, level=level) for i in range(num_envs)])
    #envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=256, clip_reward=1)
    return envs


if __name__ == '__main__':
    vec_env = make_vec_env(config.NUM_AGENTS, config.LEVEL)

    if os.path.isfile(config.MODEL_PATH):
        model = A2C.load(config.MODEL_PATH, verbose=1, tensorboard_log=config.TENSORBOARD_LOG, device="cuda")
        model.set_env(env=vec_env)
    else:
        model = A2C(MlpPolicy, env=vec_env, use_rms_prop=False, use_sde=False, verbose=1, tensorboard_log=config.TENSORBOARD_LOG, device="cuda")
        #model = TRPO(MlpPolicy, env=vec_env, use_sde=False, verbose=1, tensorboard_log=config.TENSORBOARD_LOG, device="cuda")
        #model = SAC("MlpPolicy", env=env, verbose=1, batch_size=256, learning_rate=3e-5, tau=0.005, ent_coef='auto_0.9', use_sde=True, tensorboard_log=config.TENSORBOARD_LOG, device="auto")
        #model = PPO(MlpPolicy, env=env, verbose=1, batch_size=64, use_sde=False, learning_rate=3e-5, tensorboard_log=config.TENSORBOARD_LOG, device="auto")

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=config.OUTPUT_DIR, model_name=config.MODEL_NAME)
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

    #stats_path = os.path.join(config.OUTPUT_DIR, "vec_normalize.pkl")
    #vec_env.save(stats_path)
    print('Training finished!')
