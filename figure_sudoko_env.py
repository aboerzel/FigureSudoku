import itertools
import numpy as np
import gym
from enum import Enum

from gym.spaces import Box, Discrete, MultiDiscrete, Tuple, MultiBinary

from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator


class Reward(Enum):
    FAILED = -1.0
    CONTINUE = 5.0
    SOLVED = 5.0
    TIME = 2.0


class FigureSudokuEnv(gym.Env):

    def __init__(self, env_id=0, level=1, max_steps=None, gui=None):
        super(FigureSudokuEnv, self).__init__()
        self.env_id = env_id
        self.level = level

        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

        self.gui = gui

        self.geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
        self.colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        self.figures = np.array(list(itertools.product(self.geometries, self.colors)))
        fields = np.array(list(itertools.product(np.arange(self.rows), np.arange(self.cols))))
        self.actions = np.array(list(itertools.product(self.figures, fields)), dtype=object)
        self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])
        self.solved_state = self.state

        #self.action_space = Box(shape=(1,), low=-1.0, high=1.0, dtype=np.float32)
        #self.action_space = Box(shape=(1,), low=0, high=len(self.actions)-1, dtype=np.int32)
        self.action_space = Discrete(n=len(self.actions)-1)

        state_size = int(self.state.shape[0] * self.state.shape[1] * self.state.shape[2])
        geometry_values = [e.value for e in Geometry]
        color_values = [e.value for e in Color]
        low = min(np.min(geometry_values), np.min(color_values))
        high = max(np.max(geometry_values), np.max(color_values))

        self.observation_space = Box(shape=(state_size,), low=low, high=high, dtype=np.int32)

        self.reward_range = (Reward.FAILED.value, Reward.SOLVED.value + Reward.TIME.value)

        self.generator = SudokuGenerator(self.geometries, self.colors)

    def reset(self):
        initial_items = (self.rows * self.cols) - self.level
        self.solved_state, self.state = self.generator.generate(initial_items=initial_items)

        if self.gui is not None:
            self.gui.display_state(self.state)

        # Reset the counter
        self.current_step = 0

        return self.state.flatten() #.astype(np.float32)

    def reset_with_level(self, level):
        initial_items = (self.rows * self.cols) - level
        self.solved_state, self.state = self.generator.generate(initial_items=initial_items)

        if self.gui is not None:
            self.gui.display_state(self.state)

        # Reset the counter
        self.current_step = 0

        return self.state.flatten() #.astype(np.float32)

    def render(self, **kwargs):
        # update gui
        if self.gui is not None:
            self.gui.display_state(self.state)

    def get_possible_actions(self):
        state = self.state.reshape(16, 2)

        # get used figures
        used_figures = state[np.logical_and(state[:, 0] != Geometry.EMPTY.value, state[:, 1] != Color.EMPTY.value)]
        used_figures = [[Geometry(f[0]), Color(f[1])] for f in used_figures]

        # get used cells
        used_cells_flatten = np.array([a for a in np.where(np.logical_and(state[:, 0] == Geometry.EMPTY.value, state[:, 1] == Color.EMPTY.value))]).squeeze(axis=0)

        used_cells = []
        for x in used_cells_flatten:
            row = int(x / self.rows)
            col = x % self.cols
            used_cells.append([row, col])

        possible_actions = self.actions.copy()

        # filter out used figures
        possible_actions = [a for a in possible_actions if
                            len([f for f in used_figures if a[0][0] == f[0] and a[0][1] == f[1]]) == 0]
        # filter out used cells
        possible_actions = [a for a in possible_actions if
                            len([c for c in used_cells if a[1][0] == c[0] and a[1][1] == c[1]]) != 0]

        # get indices
        possible_actions_ind = np.where([True if len([a for a in possible_actions if
                                                      a[0][0] == b[0][0] and a[0][1] == b[0][1] and a[1][0] == b[1][0]
                                                      and a[1][1] == b[1][1]]) > 0 else False for b in self.actions])[0]

        return possible_actions_ind

    def step(self, action):
        #target_action = self.actions[self.denormalize_action(action[0])]
        #target_action = self.actions[int(action[0])]
        #target_action = self.actions[int(action[0])]
        target_action = self.actions[action]

        self.current_step += 1

        reward = Reward.CONTINUE.value
        done = False
        info = {}

        # check if the action is valid
        if self.is_valid_action(target_action):
            # perform action if it is valid
            (geometry, color) = target_action[0]
            (row, col) = target_action[1]
            self.state[row][col] = [geometry.value, color.value]

            if self.gui is not None:
                self.gui.display_state(self.state)

            # check game solved or failed
            solved = FigureSudokuEnv.is_done(self.state)
            failed = not solved and self.is_game_finished()

            # finish the game when the game is won or lost
            done = solved or failed

            if failed:
                reward = Reward.FAILED.value

            if solved:
                time_reward = Reward.TIME.value * (self.level / self.current_step)
                reward = Reward.SOLVED.value + time_reward
                print(f"{self.env_id:02d}: SOLVED - Reward: {reward:.2f}")

            if not failed and not solved:
                solve_reward = ((self.level - FigureSudokuEnv.get_empty_fields(self.state)) / self.level) ** 2
                reward = Reward.CONTINUE.value * solve_reward

        else:
            reward = Reward.FAILED.value

        # Overwrite the done signal when
        if self.max_steps is not None and self.current_step >= self.max_steps:
            done = True
            info['time_limit_reached'] = True

        return self.state.flatten(), reward, done, info

    def is_game_finished(self):
        possible_actions = self.get_possible_actions()
        finished = True
        for action in possible_actions:
            if self.is_valid_action(self.actions[action]):
                finished = False
                break
        return finished

    def is_valid_action(self, action):
        (geometry, color) = action[0]
        (row, col) = action[1]

        if not FigureSudokuEnv.is_figure_available(self.state, geometry, color):
            return False

        if not FigureSudokuEnv.is_field_empty(self.state, row, col):
            return False

        if not FigureSudokuEnv.can_move(self.state, row, col, geometry, color):
            return False

        return True

    def denormalize_action(self, normalized_action):
        # Shift the value from the range of -1 to +1 to the range of 0 to 1.
        action = (normalized_action + 1.0) / 2.0

        # Scale the value from the range of 0 to 1 to the range of 0 to 255.
        action *= (len(self.actions)-1)

        return int(action)

    def normalize_action(self, action):
        # Convert the value to a float between 0 and 1.
        action /= (len(self.actions)-1)

        # Shift the value from the range of 0 to 1 to the range of -1 to +1.
        return (action * 2.0) - 1.0

    @staticmethod
    def is_field_empty(state, row, col):
        return state[row][col][0] == Geometry.EMPTY.value or state[row][col][1] == Color.EMPTY.value

    @staticmethod
    def is_figure_available(state, geometry, color):
        state = state.reshape(state.shape[0] * state.shape[1], 2)
        return len(np.where(np.logical_and(state[:, 0] == geometry.value, state[:, 1] == color.value))[0]) == 0

    @staticmethod
    def get_empty_fields(state):
        state = state.reshape(state.shape[0] * state.shape[1], 2)
        return len(np.where(np.logical_or(state[:, 0] == Geometry.EMPTY.value, state[:, 1] == Color.EMPTY.value))[0])

    @staticmethod
    def is_done(state):
        return FigureSudokuEnv.get_empty_fields(state) == 0

    @staticmethod
    def can_move(state, row, col, geometry, color):
        for field in state[row]:
            if field[0] == geometry.value:
                return False
            if field[1] == color.value:
                return False

        for field in np.array(state)[:, col]:
            if field[0] == geometry.value:
                return False
            if field[1] == color.value:
                return False

        return True
