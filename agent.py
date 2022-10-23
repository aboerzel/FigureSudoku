import os

import torch
import torch.nn.functional as F
import copy
import itertools
import numpy as np
from torch import tensor
from torch.optim import Adam

from network import Actor, Critic


class SACDiscrete:
    def __init__(self, state_size, action_size, gamma=0.99, tau=1e-3, actor_lr=5e-4, critic_lr=1e-4,
                 target_entropy_scale=0.5, alpha=0.8, alpha_lr=1e-5, hidden=512):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.dtype = torch.float32
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden = hidden
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_entropy_scale = target_entropy_scale
        self.alpha = alpha
        self.alpha_lr = alpha_lr

        print(f"GPU available: {torch.cuda.is_available()}")

        self.actor = Actor(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden)
        self.critic_1 = Critic(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden)
        self.critic_2 = Critic(state_size=self.state_size, action_size=self.action_size, hidden_size=self.hidden)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.actor.to(self.device)
        self.critic_1.to(self.device)
        self.target_critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_2.to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt_1 = Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_opt_2 = Adam(self.critic_2.parameters(), lr=self.critic_lr)

        for p in itertools.chain(self.target_critic_1.parameters(), self.target_critic_2.parameters()):
            p.requires_grad = False

        # adaptive temperature
        self.target_entropy = -np.log((1.0 / self.action_size)) * self.target_entropy_scale
        self.init_alpha = np.log(self.alpha)

        self.alpha_log = tensor([self.init_alpha], dtype=self.dtype, device=self.device, requires_grad=True)
        self.alpha_opt = Adam([self.alpha_log], lr=self.alpha_lr)
        self.alpha = torch.exp(self.alpha_log)

    def _compute_alpha(self, state):
        action, prob, log_prob = self.actor(state, deterministic=False, log_proba=True)

        loss = torch.mean((-1 * self.alpha_log * (torch.sum(log_prob * prob, -1) + self.target_entropy)))

        return loss

    def _compute_target(self, reward, next_state, done):
        with torch.no_grad():
            a_prime, prob, log_prob = self.actor(next_state, deterministic=False, log_proba=True)

            q1_targ = self.target_critic_1(next_state)
            q2_targ = self.target_critic_2(next_state)
            q_min = torch.min(q1_targ, q2_targ)

            q = torch.sum((q_min - self.alpha * log_prob) * prob, -1)
            target = reward + self.gamma * (1 - done) * q

        return target

    def _compute_actor_loss(self, state):
        action, prob, log_prob = self.actor(state, deterministic=False, log_proba=True)

        q1 = self.critic_1(state)
        q2 = self.critic_2(state)
        q_min = torch.min(q1, q2)

        loss = torch.mean(torch.sum((q_min - self.alpha.detach() * log_prob) * prob, -1))

        return -1 * loss

    def _compute_critic_loss(self, state, action, next_state, reward, done):
        target = self._compute_target(reward, next_state, done)

        q1 = self.critic_1(state).gather(1, action.reshape(-1, 1).long()).view(-1)
        q2 = self.critic_2(state).gather(1, action.reshape(-1, 1).long()).view(-1)

        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        return loss

    def _soft_update(self, target_net, source_net):
        for tp, lp in zip(target_net.parameters(), source_net.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * lp.data)

    def update(self, batch):
        state, action, reward, next_state, done = batch

        state = tensor(state, dtype=self.dtype, device=self.device)
        next_state = tensor(next_state, dtype=self.dtype, device=self.device)
        reward = tensor(reward, dtype=self.dtype, device=self.device)
        done = tensor(done, device=self.device, dtype=self.dtype)
        action = tensor(action, device=self.device, dtype=self.dtype)

        critic_loss = self._compute_critic_loss(state, action, next_state, reward, done)
        self.critic_opt_1.zero_grad()
        self.critic_opt_2.zero_grad()
        critic_loss.backward()
        self.critic_opt_1.step()
        self.critic_opt_2.step()

        actor_loss = self._compute_actor_loss(state)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = self._compute_alpha(state)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = torch.exp(self.alpha_log)

        # Critic soft update
        with torch.no_grad():
            self._soft_update(self.target_critic_1, self.critic_1)
            self._soft_update(self.target_critic_2, self.critic_2)

        return actor_loss.item(), critic_loss.item()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), f'{path}/actor.pt')
        torch.save(self.critic_1.state_dict(), f'{path}/critic_1.pt')
        torch.save(self.critic_2.state_dict(), f'{path}/critic_2.pt')

    def load_model(self, path):
        actor_model_path = f'{path}/actor.pt'
        critic1_model_path = f'{path}/critic_1.pt'
        critic2_model_path = f'{path}/critic_2.pt'

        if os.path.exists(actor_model_path):
            self.actor.load_state_dict(torch.load(actor_model_path, map_location=self.device))

        if os.path.exists(critic1_model_path):
            self.critic_1.load_state_dict(torch.load(critic1_model_path, map_location=self.device))

        if os.path.exists(critic2_model_path):
            self.critic_2.load_state_dict(torch.load(critic2_model_path, map_location=self.device))
        print('Weights are loaded')

    def get_alpha(self):
        return self.alpha.item()

    def get_action(self, state, deterministic=False, log_proba=False):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
            action, _, _ = self.actor(state, deterministic, log_proba)

        return action.item()
