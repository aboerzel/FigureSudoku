import random
from collections import deque


class ExperienceBuffer:
    def __init__(self, buff_len):
        self.buff = deque(maxlen=buff_len)

    def add(self, observation):
        self.buff.append(observation)

    def sample(self, batch_size):
        out = random.sample(self.buff, batch_size)
        return list(zip(*out))  # transpose list

    def clear(self):
        self.buff.clear()

    def __len__(self):
        return len(self.buff)
