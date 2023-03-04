import config
from figure_sudoko_env import FigureSudokuEnv, Reward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, DDPG, A2C


def play_sudoku(gui, level, stop):
    # create environment
    env = FigureSudokuEnv(level=level, gui=gui)
    check_env(env)

    model = A2C.load(config.MODEL_PATH)

    obs = env.reset()
    for i in range(30):
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
