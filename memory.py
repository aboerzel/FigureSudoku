import random
from collections import deque


class ExperienceBuffer:
    def __init__(self, buff_len, batch_size):
        self.buff = deque(maxlen=buff_len)
        self.batch_size = batch_size

    def add(self, observation):
        self.buff.append(observation)

    def sample(self):
        out = random.sample(self.buff, self.batch_size)
        return list(zip(*out))  # transpose list

    def clear(self):
        self.buff.clear()

    def __len__(self):
        return len(self.buff)
