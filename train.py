import os
import numpy as np
import config

from callbacks import SaveOnBestTrainingRewardCallback
from figure_sudoko_env import FigureSudokuEnv, Reward

from sb3_contrib import RecurrentPPO, TRPO, ARS
from sb3_contrib.ars.policies import ARSPolicy
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO, SAC, DDPG, A2C, HER, HerReplayBuffer


def make_sudoku_env(env_id, level):
    env = FigureSudokuEnv(level=level, max_steps=config.MAX_TIMESTEPS)
    check_env(env)
    env = Monitor(env, f'{config.OUTPUT_DIR}/train_{env_id}')
    return env


def make_env(env_id, level):
    def _thunk():
        env = make_sudoku_env(env_id=env_id, level=level)
        return env

    return _thunk


def make_vec_env(num_envs, level):
    envs = SubprocVecEnv([make_env(env_id=i, level=level) for i in range(num_envs)], start_method='spawn')
    return envs


if __name__ == '__main__':

    learning_rate = 3e-4
    gamma = 0.99
    target_entropy = 'auto' # 0.95
    ent_coef = 0.02
    vf_coef = 0.5
    use_sde = True
    buffer_size = int(1e6)
    batch_size = 256
    tau = 0.005
    learning_starts = 1000
    episodes = 500000
    timesteps = 256 * episodes

    train_env = make_vec_env(config.NUM_AGENTS, config.LEVEL)
    train_env = VecNormalize(venv=train_env, norm_obs=True, norm_reward=False)

    if os.path.isfile(config.MODEL_PATH):
        custom_objects = {'learning_rate': learning_rate, 'gamma': gamma, 'use_sde': use_sde, 'ent_coef': ent_coef, 'vf_coef': vf_coef}

        #model = A2C.load(config.MODEL_PATH, env=train_env, device="cuda", custom_objects=custom_objects, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG)
        #model = PPO.load(config.MODEL_PATH, env=train_env, device="cuda", custom_objects=custom_objects, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG)
    else:
        #model = A2C(MlpPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #model = TRPO(MlpPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, policy_kwargs=dict(net_arch=[256, 256, 256]), device="cuda")

        #model = ARS(ARSPolicy, env=train_env, learning_rate=0.0001, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #replay_buffer = HerReplayBuffer(env=train_env, buffer_size=20000, n_sampled_goal=4, goal_selection_strategy='future')
        #model = PPO(MlpPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #model = RecurrentPPO(MlpLstmPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        # Erstelle einen Experience Replay Buffer
        #buffer = ReplayBuffer(buffer_size=buffer_size, observation_space=train_env.observation_space, action_space=train_env.action_space, device="cuda")

        # Erstelle einen Ornstein-Uhlenbeck-Action-Noise-Prozess
        action_noise = None  # OrnsteinUhlenbeckActionNoise(mean=np.zeros(train_env.action_space.shape[0]), sigma=0.2 * np.ones(train_env.action_space.shape[0]))

        #model = SAC(SACPolicy, env=train_env, action_noise=action_noise, buffer_size=buffer_size, learning_rate=learning_rate, gamma=gamma, target_entropy=target_entropy, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, policy_kwargs=dict(net_arch=[256, 256, 256]), device="cuda")
        #model = SAC(SACPolicy, env=train_env, action_noise=action_noise, learning_starts=learning_starts, batch_size=batch_size, learning_rate=learning_rate, gamma=gamma, target_entropy=target_entropy, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")
        #model = SAC(SACPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, target_entropy=target_entropy, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")
        model = SAC(SACPolicy, env=train_env, action_noise=action_noise, learning_rate=learning_rate, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #model = DDPG(TD3Policy, env=train_env, train_freq=20, batch_size=512, learning_rate=learning_rate, gamma=gamma, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")
        #model = DDPG(TD3Policy, env=train_env, verbose=1, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #train_env = SubprocVecEnv([lambda: make_sudoku_env(env_id=i, level=config.LEVEL) for i in range(config.NUM_AGENTS)], start_method='spawn')
        #train_env = SubprocVecEnv([lambda: FigureSudokuEnv(level=config.LEVEL, max_steps=config.MAX_TIMESTEPS) for i in range(config.NUM_AGENTS)])
        #train_env = VecNormalize(venv=train_env, norm_obs=True, norm_reward=False)

    eval_env = DummyVecEnv([lambda: Monitor(FigureSudokuEnv(level=config.LEVEL, max_steps=config.MAX_TIMESTEPS), f'{config.OUTPUT_DIR}/eval')])
    eval_env = VecNormalize(venv=eval_env, norm_obs=True, norm_reward=False)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=Reward.SOLVED.value, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=config.BEST_EVAL_MODEL_PATH, log_path=config.TENSORBOARD_EVAL_LOG, eval_freq=config.EVAL_FREQ, callback_on_new_best=callback_on_best)

    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=config.CHECK_FREQ, log_dir=config.OUTPUT_DIR, model_name=config.MODEL_NAME, checkpoint_name=config.CHECKPOINT_NAME)

    callback = CallbackList([save_best_model_callback, eval_callback])

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    print('Training finished!')
