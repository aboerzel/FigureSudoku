from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from figure_sudoko_env import FigureSudokuEnv
from hyperparameters import *
from ddpg_agent import DDPGAgent
from shapes import Geometry, Color


def train_sudoku(gui, stop):
    # create environment
    geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
    colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
    env = FigureSudokuEnv(geometries, colors, gui=gui)
    state_size = env.state_size
    action_dims = env.actions_dims

    agent = DDPGAgent(state_size=state_size, action_dims=action_dims, batch_size=BATCH_SIZE, buffer_capacity=BUFFER_SIZE,
                      ou_noise=np.array(OU_NOISE), ou_theta=OU_THETA,
                      actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU)

    agent.load_model_weights(OUTPUT_DIR)

    # hyperparameter
    start_episode = 1
    start_level = 1

    # score parameter
    warmup_episodes = start_episode + 2 * AVG_SCORE_WINDOW
    scores_deque = deque(maxlen=AVG_SCORE_WINDOW)
    avg_score = -99999
    best_avg_score = avg_score

    level = start_level

    writer = SummaryWriter()

    for episode in range(start_episode, MAX_EPISODES + 1):
        if stop():
            break

        state = env.reset(level=level)  # reset the environment
        episode_score = 0
        for timestep in range(1, MAX_STEPS_PER_EPISODE + 1):
            action = agent.act(state, train=True)
            print(f'Episode {episode:08d} - Step {timestep:04d}\tAction: {action[0]} {action[1]} {action[2]} {action[3]}', end='\r')
            next_state, reward, done = env.step(np.array([action[0], action[1], Geometry(action[2]), Color(action[3])]))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_score += reward
            if done:
                print(f'Episode {episode:08d} - Step {timestep:04d}\tEpisode Score: {episode_score:.2f}\tdone!')
                break

        # average score over the last n epochs
        scores_deque.append(episode_score)
        avg_score = np.mean(scores_deque)

        writer.add_scalar("episode score", episode_score, episode)
        writer.add_scalar("avg score", avg_score, episode)

        if episode % 10 == 0:
            print(f'\rEpisode {episode:08d}\tAverage Score: {avg_score:.2f}')
            agent.save_model_weights(OUTPUT_DIR)

        # print(f'Episode {episode:06d}\tAvg Score: {avg_score:.2f}\tBest Avg Score: {best_avg_score:.2f}')

        # save best weights
        if episode > warmup_episodes and avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save_model_weights(OUTPUT_DIR)
            print(f'Episode {episode:08d}\tWeights saved!\tBest Avg Score: {best_avg_score:.2f}')

        # stop training if target score was reached
        if episode > warmup_episodes and avg_score >= TARGET_SCORE:
            agent.save_model_weights(OUTPUT_DIR)
            print(f'\nEnvironment solved in {episode} episodes!\tAverage Score: {avg_score:.2f}')
            break

    agent.save_model_weights(OUTPUT_DIR)
    print(f'Training finished!\tBest Avg Score: {best_avg_score:.2f}')
