from enum import Enum

import numpy as np

from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator


class Reward(Enum):
    FORBIDDEN = -10
    CONTINUE = 0
    DONE = 10


class FigureSudokuEnv:

    def __init__(self, geometries, colors, gui=None):
        self.geometries = geometries
        self.colors = colors
        self.gui = gui
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])

        self.state_size = len(self.state.flatten())
        self.actions_dims = np.array([[0, self.rows - 1], [0, self.cols - 1], [Geometry.CIRCLE.value, Geometry.HEXAGON.value], [Color.RED.value, Color.YELLOW.value]])

        self.generator = SudokuGenerator(geometries, colors)

    def reset(self, level=1):
        initial_items = (self.rows * self.cols) - level
        self.state = self.generator.generate(initial_items=initial_items)[1]

        # update gui
        if self.gui is not None:
            self.gui.display_state(self.state)
        return self.state.flatten()

    def step(self, action):
        row, col, geometry, color = action

        temp_state = [geometry.value, color.value]

        if not FigureSudokuEnv.is_figure_available(self.state, geometry, color):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        if not FigureSudokuEnv.is_field_empty(self.state, row, col):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        if not FigureSudokuEnv.can_move(self.state, row, col, geometry, color):
            return self.state.flatten(), Reward.FORBIDDEN.value, False

        self.state[row][col] = temp_state
        done = FigureSudokuEnv.is_done(self.state)
        reward = Reward.DONE.value if done else Reward.CONTINUE.value

        if self.gui is not None:
            self.gui.display_state(self.state)

        return self.state.flatten(), reward, done

    @staticmethod
    def is_field_empty(state, row, col):
        return state[row][col][0] == Geometry.EMPTY.value or state[row][col][1] == Color.EMPTY.value

    @staticmethod
    def is_figure_available(state, geometry, color):
        state = state.reshape(state.shape[0] * state.shape[1], 2)
        return len(np.where(np.logical_and(state[:, 0] == geometry.value, state[:, 1] == color.value))[0]) == 0

    @staticmethod
    def is_done(state):
        state = state.reshape(state.shape[0] * state.shape[1], 2)
        return len(np.where(np.logical_or(state[:, 0] == Geometry.EMPTY.value, state[:, 1] == Color.EMPTY.value))[0]) == 0

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
