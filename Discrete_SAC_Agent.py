import os
import random

import numpy as np
import torch

from Network import Network
from ReplayBuffer import ReplayBuffer


class SACAgent:

    ALPHA_INITIAL = 1.
    REPLAY_BUFFER_BATCH_SIZE = 100
    DISCOUNT_RATE = 0.99
    LEARNING_RATE = 0.0001
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.005

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.critic_local = Network(input_dimension=self.state_size, output_dimension=self.action_size)
        self.critic_local2 = Network(input_dimension=self.state_size, output_dimension=self.action_size)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Network(input_dimension=self.state_size, output_dimension=self.action_size)
        self.critic_target2 = Network(input_dimension=self.state_size, output_dimension=self.action_size)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Network(input_dimension=self.state_size, output_dimension=self.action_size, output_activation=torch.nn.Softmax(dim=1))
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(capacity=5000)

        self.target_entropy = 0.98 * -np.log(1 / self.action_size)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state, possible_actions, eps):
        # Epsilon-greedy action selection
        if random.random() > eps:
            action = self.get_action(state)
            return action
        else:
            action = random.choice(possible_actions)
            return action

    def get_action(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.argmax(action_probabilities)
        return discrete_action

    def train_networks(self, state, action, reward, next_state, done):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(state, action, reward, next_state, done)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)

            critic_loss, critic2_loss = self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities)).sum(dim=1).unsqueeze(-1)
            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values

        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)

    def load(self, filepath):

        if not os.path.isfile(filepath):
            return

        # load weights from file
        state_dict = torch.load(filepath)
        self.actor_local.load_state_dict(state_dict)
        self.actor_local.eval()

    def save(self, filepath):
        # Save weights to file
        torch.save(self.actor_local.state_dict(), filepath)
