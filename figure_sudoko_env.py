import itertools
import numpy as np
import gym

from gym.spaces import Box, Discrete

from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator
from visualizer import SudokuVisualizer


class FigureSudokuEnv(gym.Env):

    def __init__(self, env_id=0, level=1, max_steps=None, render_gui=False):
        super(FigureSudokuEnv, self).__init__()
        self.env_id = env_id
        self.level = level

        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0
        self.episode = 0

        # episode statistics
        self.rewards = []
        self.valids = 0

        self.geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
        self.colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])
        self.rows = len(self.geometries)
        self.cols = len(self.colors)

        if render_gui:
            self.gui = SudokuVisualizer(env_id=self.env_id, rows=self.rows, cols=self.cols, level=self.level)
        else:
            self.gui = None
        self.figures = np.array(list(itertools.product(self.geometries, self.colors)))
        fields = np.array(list(itertools.product(np.arange(self.rows), np.arange(self.cols))))
        self.actions = np.array(list(itertools.product(self.figures, fields)), dtype=object)
        self.state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])
        self.solved_state = self.state

        self.action_space = Discrete(n=len(self.actions))

        # Observation Space: Flattened one-hot encoding
        # 4x4 cells * (5 Geometries + 5 Colors) = 160 elements
        self.observation_space = Box(low=0, high=1, shape=(self.rows * self.cols * 10,), dtype=np.float32)

        self.generator = SudokuGenerator(self.geometries, self.colors)

    def _get_obs(self):
        # Convert state (4, 4, 2) to one-hot encoding (4, 4, 10) then flatten to (160,)
        obs = np.zeros((self.rows, self.cols, 10), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                g_val = int(self.state[r, c, 0])
                c_val = int(self.state[r, c, 1])
                obs[r, c, g_val] = 1.0  # Geometry one-hot (0-4)
                obs[r, c, 5 + c_val] = 1.0  # Color one-hot (5-9)
        return obs.flatten()

    def reset(self):
        self.current_step = 0
        self.episode += 1

        self.rewards = []
        self.valids = 0

        initial_items = (self.rows * self.cols) - self.level
        self.solved_state, self.state = self.generator.generate(initial_items=initial_items)

        if self.gui is not None:
            self.gui.clear_visual_feedback()
            self.gui.display_state(self.state)

        return self._get_obs()

    def reset_with_level(self, level):
        self.level = level
        initial_items = (self.rows * self.cols) - level
        self.solved_state, self.state = self.generator.generate(initial_items=initial_items)

        if self.gui is not None:
            self.gui.clear_visual_feedback()
            self.gui.update_title(level)
            self.gui.display_state(self.state)

        # Reset the counter
        self.current_step = 0

        return self._get_obs()

    def render(self, **kwargs):
        # update gui
        if self.gui is not None:
            self.gui.display_state(self.state)

    def close(self):
        if self.gui is not None and hasattr(self.gui, 'close'):
            self.gui.close()

    def action_masks(self):
        mask = np.zeros(len(self.actions), dtype=bool)
        for i, action in enumerate(self.actions):
            if self.is_valid_action(action):
                mask[i] = True
        return mask

    def step(self, action):
        self.current_step += 1

        target_action = self.actions[action]

        reward = 0.0
        done = False

        # check if the action is valid
        if self.is_valid_action(target_action):
            self.valids += 1
            
            # perform action if it is valid
            self.perform_action(target_action)

            # Reward for a valid move
            # Base reward of 0.1 to encourage any valid move.
            # Plus a progress bonus: 1.0 divided by total cells (16), so each field is worth 0.0625.
            reward = 0.1 + (1.0 / (self.rows * self.cols)) 

            # check game solved
            done = self.is_done(self.state)
            if done:
                reward += 2.0 # High reward for completing the puzzle
                if self.gui is not None:
                    self.gui.show_success()
        else:
            reward = -0.5 # Slightly softer penalty for invalid moves to encourage exploration

        finished = self.is_game_finished()

        self.rewards.append(reward)
        mean_reward = np.mean(np.array(self.rewards))

        # An episode ends when no further move is possible or the maximum number of time steps is reached
        episode_over = done or finished or (self.max_steps is not None and self.current_step >= self.max_steps)

        if episode_over and not done:
            if self.gui is not None:
                self.gui.show_failure()

        info = {}
        # Provide the action mask for MaskablePPO
        info["action_mask"] = self.action_masks()

        if done:
            print(f"agent {self.env_id:02d} - episode {self.episode:05d} - step {self.current_step:04d} - level {self.level:02d} : Action: {action:03d} - Valids: {self.valids:04d} - Mean Reward: {mean_reward:.5f} - DONE", flush=True)
        elif episode_over:
            print(f"agent {self.env_id:02d} - episode {self.episode:05d} - step {self.current_step:04d} - level {self.level:02d} : Action: {action:03d} - Valids: {self.valids:04d} - Mean Reward: {mean_reward:.5f}", flush=True)
            if self.max_steps is not None and self.current_step >= self.max_steps:
                info['time_limit_reached'] = True

        return self._get_obs(), reward, episode_over, info

    def is_game_finished(self):
        # Wenn kein leeres Feld mehr da ist, ist das Spiel vorbei
        if FigureSudokuEnv.get_empty_fields(self.state) == 0:
            return True
            
        # Wenn es keine validen Aktionen mehr gibt, obwohl noch Felder frei sind
        mask = self.action_masks()
        return not np.any(mask)

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

    def perform_action(self, action):
        (geometry, color) = action[0]
        (row, col) = action[1]
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color
        self.state[row][col] = [g_val, c_val]

        if self.gui is not None:
            self.gui.display_state(self.state)

    @staticmethod
    def is_field_empty(state, row, col):
        return state[row][col][0] == Geometry.EMPTY.value

    @staticmethod
    def is_figure_available(state, geometry, color):
        # state is (4, 4, 2)
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color
        
        exists = np.any((state[:, :, 0] == g_val) & (state[:, :, 1] == c_val))
        return not exists

    @staticmethod
    def get_empty_fields(state):
        return np.sum(state[:, :, 0] == Geometry.EMPTY.value)

    @staticmethod
    def is_done(state):
        return FigureSudokuEnv.get_empty_fields(state) == 0

    @staticmethod
    def can_move(state, row, col, geometry, color):
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color

        # Check row
        if np.any(state[row, :, 0] == g_val) or np.any(state[row, :, 1] == c_val):
            return False
        
        # Check column
        if np.any(state[:, col, 0] == g_val) or np.any(state[:, col, 1] == c_val):
            return False

        return True
