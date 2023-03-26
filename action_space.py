import random

from gym.spaces import Discrete, MultiDiscrete


class SudokoActionSpace(Discrete):
    def __init__(self, n, env):
        super().__init__(n=n)
        self.env = env

    def sample(self):
        possible_actions = self.env.get_possible_actions()
        return random.choice(possible_actions)


class SudokuMultiDiscreteActionSpace(MultiDiscrete):
    def __init__(self, nvec, dtype, env):
        super().__init__(nvec=nvec, dtype=dtype)
        self.env = env

    def sample(self):
        possible_actions = self.env.get_possible_actions()
        return random.choice(possible_actions)
