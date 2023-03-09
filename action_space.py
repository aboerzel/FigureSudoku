import random

from gym.spaces import Discrete


class SudokoActionSpace(Discrete):
    def __init__(self, n, env):
        super().__init__(n=n)
        self.env = env

    def sample(self):
        possible_actions = self.env.get_possible_actions()
        return random.choice(possible_actions)
