import os
import torch as th
import torch.nn as nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
import config

from callbacks import SaveOnBestTrainingRewardCallback, CurriculumCallback
from figure_sudoko_env import FigureSudokuEnv

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SudokuCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(SudokuCNN, self).__init__(observation_space, features_dim)
        # Input: (Batch, 10, 4, 4)
        self.cnn = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=1), # Dimension expansion
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass with reshaped dummy input
        with th.no_grad():
            dummy_input = th.as_tensor(observation_space.sample()[None]).float()
            dummy_input = dummy_input.view(-1, 10, 4, 4)
            n_flatten = self.cnn(dummy_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Reshape flat observations back to (Channels, Height, Width)
        # observations shape: (Batch, 160) -> (Batch, 10, 4, 4)
        observations = observations.view(-1, 10, 4, 4)
        return self.linear(self.cnn(observations))


def make_sudoku_env(env_id, level, render_gui=False):
    env = FigureSudokuEnv(
        env_id=env_id, 
        level=level, 
        max_steps=config.MAX_TIMESTEPS, 
        render_gui=render_gui,
        unique=config.UNIQUE,
        partial_prob=config.PARTIAL_PROB,
        partial_mode=config.PARTIAL_MODE,
        reward_solved=config.REWARD_SOLVED,
        reward_valid_move_base=config.REWARD_VALID_MOVE_BASE,
        reward_invalid_move=config.REWARD_INVALID_MOVE
    )
    check_env(env)
    env = Monitor(env, f'{config.OUTPUT_DIR}/train_{env_id}')
    return env


def make_env(env_id, level, render_gui=False):
    def _thunk():
        env = make_sudoku_env(env_id=env_id, level=level, render_gui=render_gui)
        return env

    return _thunk


def make_vec_env(num_envs, level, render_gui=False):
    if render_gui:
        envs = DummyVecEnv([make_env(env_id=i, level=level, render_gui=render_gui) for i in range(num_envs)])
    else:
        envs = SubprocVecEnv([make_env(env_id=i, level=level, render_gui=render_gui) for i in range(num_envs)], start_method='spawn')
    return envs


if __name__ == '__main__':

    # PPO Hyperparameters
    initial_learning_rate = 1e-4 
    n_steps = 4096 
    batch_size = 1024 
    n_epochs = 5      # Reduced from 10 to improve stability and speed up iterations
    target_kl = 0.02  # Added to prevent too large policy updates (the "dips")
    ent_coef = 0.01 
    vf_coef = 0.5
    gamma = 0.995 

    lr_schedule = linear_schedule(initial_learning_rate)

    policy_kwargs = dict(
        features_extractor_class=SudokuCNN,
        features_extractor_kwargs=dict(features_dim=256),
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[256], vf=[256])
    )

    train_env = make_vec_env(config.NUM_AGENTS, config.START_LEVEL, render_gui=config.RENDER_GUI)

    if os.path.isfile(config.MODEL_PATH):
        custom_objects = {
            'learning_rate': lr_schedule,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'target_kl': target_kl,
            'gamma': gamma,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'policy_kwargs': policy_kwargs
        }
        model = MaskablePPO.load(config.MODEL_PATH, env=train_env, custom_objects=custom_objects, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")
    else:
        model = MaskablePPO(MaskableActorCriticPolicy, env=train_env, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, target_kl=target_kl, learning_rate=lr_schedule, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=config.CHECK_FREQ, log_dir=config.OUTPUT_DIR, model_name=config.MODEL_NAME, checkpoint_name=config.CHECKPOINT_NAME, verbose=1)

    curriculum_callback = CurriculumCallback(
        check_freq=config.CHECK_FREQ, 
        reward_threshold=config.REWARD_THRESHOLD, 
        log_dir=config.OUTPUT_DIR,
        reward_solved=config.REWARD_SOLVED,
        start_level=config.START_LEVEL,
        max_level=config.MAX_LEVEL,
        unique=config.UNIQUE,
        partial_prob=config.PARTIAL_PROB,
        partial_mode=config.PARTIAL_MODE,
        verbose=1
    )

    callback = CallbackList([save_best_model_callback, curriculum_callback])

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback, progress_bar=False)

    print('Training finished!')
