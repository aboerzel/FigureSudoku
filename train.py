import os
import torch as th

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
import config

from callbacks import SaveOnBestTrainingRewardCallback, CurriculumCallback
from figure_sudoko_env import FigureSudokuEnv

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_checker import check_env


def make_sudoku_env(env_id, level, render_gui=False):
    env = FigureSudokuEnv(env_id=env_id, level=level, max_steps=config.MAX_TIMESTEPS, render_gui=render_gui)
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
    learning_rate = 1e-4 # Lower for more stable convergence in higher levels
    n_steps = 2048 # Increased for more data per update
    batch_size = 512 # Increased for stability
    ent_coef = 0.01 # Standard entropy
    vf_coef = 0.5
    gamma = 0.995 # Slightly higher gamma for longer horizon logic

    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[512, 512], vf=[512, 512])
    )

    train_env = make_vec_env(config.NUM_AGENTS, config.START_LEVEL, render_gui=config.RENDER_GUI)

    if os.path.isfile(config.MODEL_PATH):
        custom_objects = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'gamma': gamma,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'policy_kwargs': policy_kwargs
        }
        model = MaskablePPO.load(config.MODEL_PATH, env=train_env, custom_objects=custom_objects, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")
    else:
        model = MaskablePPO(MaskableActorCriticPolicy, env=train_env, n_steps=n_steps, batch_size=batch_size, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=config.CHECK_FREQ, log_dir=config.OUTPUT_DIR, model_name=config.MODEL_NAME, checkpoint_name=config.CHECKPOINT_NAME)
    curriculum_callback = CurriculumCallback(check_freq=config.CHECK_FREQ, reward_threshold=config.REWARD_THRESHOLD, log_dir=config.OUTPUT_DIR)

    callback = CallbackList([save_best_model_callback, curriculum_callback])

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback, progress_bar=False)

    print('Training finished!')
