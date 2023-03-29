import os

import gym
from sb3_contrib import RecurrentPPO, TRPO, ARS
from sb3_contrib.ars.policies import ARSPolicy
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.sac.policies import SACPolicy

import config
from callbacks import SaveOnBestTrainingRewardCallback
from figure_sudoko_env import FigureSudokuEnv, Reward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, DDPG, A2C, HER, HerReplayBuffer

from stable_baselines3.ppo import MlpPolicy


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
    envs = SubprocVecEnv([make_env(env_id=i, level=level) for i in range(num_envs)])
    return envs


if __name__ == '__main__':

    learning_rate = 3e-6
    gamma = 0.99
    target_entropy = 0.95
    ent_coef = 0.05
    vf_coef = 0.5
    use_sde = True

    train_env = make_vec_env(config.NUM_AGENTS, config.LEVEL)

    if os.path.isfile(config.MODEL_PATH):
        custom_objects = {'learning_rate': learning_rate, 'gamma': gamma, 'use_sde': use_sde, 'ent_coef': ent_coef, 'vf_coef': vf_coef}

        #model = A2C.load(config.MODEL_PATH, env=train_env, device="cuda", custom_objects=custom_objects, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG)
        model = PPO.load(config.MODEL_PATH, env=train_env, device="cuda", custom_objects=custom_objects, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG)
    else:
        #model = A2C(MlpPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #model = TRPO(MlpPolicy, env=train_env, use_sde=False, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")
        #model = ARS(ARSPolicy, env=train_env, learning_rate=0.0001, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #model = SAC(MlpPolicy, env=train_env, verbose=1, batch_size=256, learning_rate=3e-5, tau=0.005, ent_coef='auto_0.9', use_sde=True, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="auto")

        #replay_buffer = HerReplayBuffer(env=train_env, buffer_size=20000, n_sampled_goal=4, goal_selection_strategy='future')
        #model = PPO(MlpPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        #model = RecurrentPPO(MlpLstmPolicy, env=train_env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, vf_coef=vf_coef, use_sde=use_sde, verbose=1, tensorboard_log=config.TENSORBOARD_TRAIN_LOG, device="cuda")

        model = SAC(SACPolicy,
                    env=train_env,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    target_entropy=target_entropy,
                    use_sde=use_sde,
                    verbose=1,
                    tensorboard_log=config.TENSORBOARD_TRAIN_LOG,
                    #policy_kwargs=dict(net_arch=[256, 256, 256]),
                    device="cuda")

    eval_env = FigureSudokuEnv(level=config.LEVEL, max_steps=config.MAX_TIMESTEPS)
    eval_env = Monitor(eval_env, f'{config.OUTPUT_DIR}/eval')

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=Reward.SOLVED.value, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=config.BEST_EVAL_MODEL_PATH, log_path=config.TENSORBOARD_EVAL_LOG, eval_freq=config.EVAL_FREQ, callback_on_new_best=callback_on_best)

    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=config.CHECK_FREQ, log_dir=config.OUTPUT_DIR, model_name=config.MODEL_NAME, checkpoint_name=config.CHECKPOINT_NAME)

    callback = CallbackList([save_best_model_callback, eval_callback])

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

    print('Training finished!')
