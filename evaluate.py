from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from figure_sudoko_env import FigureSudokuEnv


class SudokuEnvironmentFactory:
    def __init__(self,  normalization_param_file):
        self.normalization_param_file = normalization_param_file

    @staticmethod
    def make_sudoku_env(level):
        def _thunk():
            env = FigureSudokuEnv(level=level)
            return env

        return _thunk

    def create(self, level):
        # create environment
        env = DummyVecEnv(env_fns=[SudokuEnvironmentFactory.make_sudoku_env(level)])
        env = VecNormalize.load(self.normalization_param_file, venv=env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
        return env
