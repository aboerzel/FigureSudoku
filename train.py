import os
import torch as th
import torch.nn as nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
import config

from callbacks import SaveOnBestTrainingRewardCallback, CurriculumCallback
from figure_sudoku_env import FigureSudokuEnv

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable
import re


def get_last_level(log_path: str, default_level: int = None) -> int:
    """
    Versucht das zuletzt erreichte Level aus der Log-Datei zu extrahieren.
    Liest die Datei speichereffizient von hinten und unterstützt UTF-8 und UTF-16.
    """
    if default_level is not None:
        return default_level

    default_level = 1

    if not os.path.exists(log_path):
        return default_level
    
    try:
        size = os.path.getsize(log_path)
        if size == 0:
            return default_level
            
        with open(log_path, 'rb') as f:
            # Versuche die letzten 20000 Bytes zu lesen (ca. 100-200 Zeilen)
            read_size = min(size, 20000)
            f.seek(-read_size, os.SEEK_END)
            data = f.read()
            
            # Kodierung erkennen: Wenn viele Null-Bytes vorhanden sind, ist es wahrscheinlich UTF-16
            if b'\x00' in data:
                try:
                    # UTF-16 LE (oft bei PowerShell > Umleitung)
                    # Wir müssen sicherstellen, dass wir an einer geraden Grenze anfangen,
                    # falls wir nicht die ganze Datei lesen.
                    # Aber decode('utf-16') mit ignore sollte robust sein.
                    text = data.decode('utf-16', errors='ignore')
                except:
                    text = data.decode('utf-8', errors='ignore')
            else:
                text = data.decode('utf-8', errors='ignore')
                
            lines = text.splitlines()
            for line in reversed(lines):
                match = re.search(r"level (\d+)", line)
                if match:
                    return int(match.group(1))
    except Exception as e:
        print(f"Warnung: Konnte letztes Level nicht aus Log lesen: {e}")
    
    return default_level


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
            sample_obs = observation_space.sample()
            if isinstance(sample_obs, tuple): # Gymnasium sample can return info sometimes, but usually just the value
                sample_obs = sample_obs[0]
            dummy_input = th.as_tensor(sample_obs[None]).float()
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
    initial_learning_rate = 5e-5 
    n_steps = 4096 
    batch_size = 1024 
    n_epochs = 5      # Reduced from 10 to improve stability and speed up iterations
    target_kl = 0.05  # Increased from 0.02 to allow for more policy updates before early stopping
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

    # Versuche das letzte Level aus dem Log zu laden, falls wir ein Modell fortsetzen
    current_start_level = config.START_LEVEL
    if os.path.isfile(config.MODEL_PATH):
        current_start_level = get_last_level(config.LOG_FILE_PATH, config.START_LEVEL)
        
        print(f"Fortsetzen des Trainings erkannt. Starte bei Level: {current_start_level}")

    train_env = make_vec_env(config.NUM_AGENTS, current_start_level, render_gui=config.RENDER_GUI)

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
        model = MaskablePPO.load(config.MODEL_PATH, env=train_env, custom_objects=custom_objects, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda", verbose=1)
    else:
        model = MaskablePPO(MaskableActorCriticPolicy, env=train_env, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, target_kl=target_kl, learning_rate=lr_schedule, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, policy_kwargs=policy_kwargs, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda", verbose=1)

    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=config.CHECK_FREQ, log_dir=config.OUTPUT_DIR, model_name=config.MODEL_NAME, checkpoint_name=config.CHECKPOINT_NAME, verbose=1)

    curriculum_callback = CurriculumCallback(
        check_freq=config.CHECK_FREQ, 
        reward_threshold=config.REWARD_THRESHOLD, 
        log_dir=config.OUTPUT_DIR,
        reward_solved=config.REWARD_SOLVED,
        start_level=current_start_level,
        max_level=config.MAX_LEVEL,
        verbose=1
    )

    callback = CallbackList([save_best_model_callback, curriculum_callback])

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback, progress_bar=False)

    print('Training finished!')
