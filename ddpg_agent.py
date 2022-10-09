import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import min_max_norm

from memory import Buffer
from ou_noise import OUActionNoise
from shapes import Geometry, Color


class DDPGAgent:
    def __init__(self, state_size, action_dims, batch_size, ou_noise, ou_theta, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005, buffer_capacity=100000):
        self.action_dims = action_dims

        self.actor_model = self.build_actor(state_size, action_dims)
        self.critic_model = self.build_critic(state_size, len(action_dims))

        self.target_actor = self.build_actor(state_size, action_dims)
        self.target_critic = self.build_critic(state_size, len(action_dims))

        self.ou_noise = OUActionNoise(mean=ou_noise[:, 0], std_deviation=ou_noise[:, 1], theta=ou_theta)

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.buffer = Buffer(state_size, action_dims, batch_size, self.actor_model, self.critic_model, self.target_actor, self.target_critic, actor_optimizer, critic_optimizer, gamma=gamma, tau=tau, buffer_capacity=buffer_capacity)

    def act(self, state, train=False):
        predicted_action = tf.squeeze(self.actor_model(np.array([state]))).numpy()

        if train:
            # Adding noise to action
            noise = self.ou_noise()
            predicted_action += noise

        # We make sure action is within bounds
        legal_action = np.round(np.clip(predicted_action, self.action_dims[:, 0], self.action_dims[:, 1]), decimals=0).astype(int)

        return legal_action

    def step(self, state, action, reward, next_state, done):
        self.buffer.record((state, action, reward, next_state))
        self.buffer.learn()

    def reset(self):
        self.ou_noise.reset()

    @staticmethod
    def build_actor(state_size, action_dims):
        out_init = tf.random_uniform_initializer(minval=0.0, maxval=0.03)
        #out_init = tf.random_uniform_initializer(minval=-0.0000001, maxval=0.0000001)

        inputs = Input(shape=(state_size,))
        hidden = Dense(256, activation="relu")(inputs)
        hidden = Dense(256, activation="relu")(hidden)
        outputs = Dense(len(action_dims), activation="sigmoid", kernel_initializer=out_init)(hidden)

        action_output = outputs * action_dims[:, 1]  # upper bound

        model = Model(inputs=inputs, outputs=action_output)
        return model

    @staticmethod
    def build_critic(state_size, action_size):
        # State as input
        state_input = Input(shape=state_size)
        state_out = Dense(64, activation="relu")(state_input)
        state_out = Dense(128, activation="relu")(state_out)

        # Action as input
        action_input = Input(shape=action_size)
        action_out = Dense(16, activation="relu")(action_input)
        action_out = Dense(32, activation="relu")(action_out)
        action_out = Dense(64, activation="relu")(action_out)
        action_out = Dense(128, activation="relu")(action_out)

        # Both are passed through separate layer before concatenating
        concat = Concatenate()([state_out, action_out])

        out = Dense(256, activation="relu")(concat)
        out = Dense(256, activation="relu")(out)
        outputs = Dense(action_size)(out)

        # Outputs single value for give state-action
        model = Model([state_input, action_input], outputs)
        return model

    def load_model_weights(self, output_dir):
        actor_model_file = os.path.normpath(os.path.join(output_dir, 'actor.h5'))
        critic_model_file = os.path.normpath(os.path.join(output_dir, 'critic.h5'))
        target_actor_model_file = os.path.normpath(os.path.join(output_dir, 'target_actor.h5'))
        target_critic_model_file = os.path.normpath(os.path.join(output_dir, 'target_critic.h5'))

        if pathlib.Path(actor_model_file).is_file():
            self.actor_model.load_weights(actor_model_file)

        if pathlib.Path(critic_model_file).is_file():
            self.critic_model.load_weights(critic_model_file)

        if pathlib.Path(target_actor_model_file).is_file():
            self.target_actor.load_weights(target_actor_model_file)
        else:
            self.target_actor.set_weights(self.actor_model.get_weights())

        if pathlib.Path(target_critic_model_file).is_file():
            self.target_critic.load_weights(target_critic_model_file)
        else:
            self.target_critic.set_weights(self.critic_model.get_weights())

    def save_model_weights(self, output_dir):
        actor_model_file = os.path.normpath(os.path.join(output_dir, 'actor.h5'))
        critic_model_file = os.path.normpath(os.path.join(output_dir, 'critic.h5'))
        target_actor_model_file = os.path.normpath(os.path.join(output_dir, 'target_actor.h5'))
        target_critic_model_file = os.path.normpath(os.path.join(output_dir, 'target_critic.h5'))
        # position_prediction_model_file = os.path.normpath(os.path.join(output_dir, 'axis_position_prediction_model.h5'))

        # Save trained weights
        self.actor_model.save_weights(actor_model_file)
        self.critic_model.save_weights(critic_model_file)

        self.target_actor.save_weights(target_actor_model_file)
        self.target_critic.save_weights(target_critic_model_file)

        # save as position prediction model
        # self.actor_model.save(position_prediction_model_file)

