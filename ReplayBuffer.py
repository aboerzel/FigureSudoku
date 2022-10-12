import numpy as np
from collections import namedtuple

from torch import from_numpy


class ReplayBuffer:

    def __init__(self, capacity=5000):
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.buffer = np.zeros(capacity, dtype=type(self.experience))
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def add_transition(self, state, action, reward, next_state, done):
        self.buffer[self.head_idx] = self.experience(state, action, reward, next_state, done)
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)

        minibatch = self.buffer[self.indices]

        states = from_numpy(np.vstack([e.state for e in minibatch if e is not None])).float()
        actions = from_numpy(np.vstack([e.action for e in minibatch if e is not None])).long()
        rewards = from_numpy(np.vstack([e.reward for e in minibatch if e is not None])).float()
        next_states = from_numpy(np.vstack([e.next_state for e in minibatch if e is not None])).float()
        dones = from_numpy(np.vstack([e.done for e in minibatch if e is not None]).astype(np.uint8))

        return states, actions, rewards, next_states, dones

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count
