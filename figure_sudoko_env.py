import itertools
import numpy as np
import gym

from gym.spaces import Box, Discrete

from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator
from visualizer import SudokuVisualizer


class FigureSudokuEnv(gym.Env):

    def __init__(self, env_id=0, level=1, max_steps=None, render_gui=False, unique=False, partial_prob=0.0, partial_mode=0,
                 reward_solved=10.0, reward_valid_move_base=0.1, reward_invalid_move=-0.5):
        super(FigureSudokuEnv, self).__init__()
        self.env_id = env_id
        self.level = level
        self.unique = unique
        self.partial_prob = partial_prob
        self.partial_mode = partial_mode
        self.reward_solved = reward_solved
        self.reward_valid_move_base = reward_valid_move_base
        self.reward_invalid_move = reward_invalid_move

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
        self.state = np.full((self.rows, self.cols, 2), Geometry.EMPTY.value, dtype=np.int32)
        self.solved_state = self.state.copy()

        self.action_space = Discrete(n=len(self.actions))

        # Observation Space: Flattened one-hot encoding
        # 4x4 cells * (5 Geometries + 5 Colors) = 160 elements
        # We use a 1D vector to avoid unnecessary image-related warnings in SB3
        self.observation_space = Box(low=0, high=1, shape=(self.rows * self.cols * 10,), dtype=np.float32)

        self.generator = SudokuGenerator(self.geometries, self.colors)

    def _get_obs(self):
        # Convert state (4, 4, 2) to one-hot encoding (10, 4, 4) then flatten to (160,)
        obs = np.zeros((10, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                g_val = int(self.state[r, c, 0])
                c_val = int(self.state[r, c, 1])
                obs[g_val, r, c] = 1.0  # Geometry one-hot (0-4)
                obs[5 + c_val, r, c] = 1.0  # Color one-hot (5-9)
        return obs.flatten()

    def reset(self):
        self.current_step = 0
        self.episode += 1

        self.rewards = []
        self.valids = 0

        initial_items = (self.rows * self.cols) - self.level
        self.solved_state, self.state = self.generator.generate(
            initial_items=initial_items, 
            unique=self.unique,
            partial_prob=self.partial_prob, 
            partial_mode=self.partial_mode
        )

        if self.gui is not None:
            self.gui.clear_visual_feedback()
            self.gui.update_title(self.level)
            self.gui.display_state(self.state)

        return self._get_obs()

    def reset_with_level(self, level, unique=None, partial_prob=None, partial_mode=None):
        self.level = level
        if unique is not None:
            self.unique = unique
        if partial_prob is not None:
            self.partial_prob = partial_prob
        if partial_mode is not None:
            self.partial_mode = partial_mode
        return self.reset()

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
            # Base reward to encourage any valid move, plus a progress bonus.
            reward = self.reward_valid_move_base + (1.0 / (self.rows * self.cols)) 

            # check game solved
            done = self.is_done(self.state)
            if done:
                reward += self.reward_solved
                if self.gui is not None:
                    self.gui.show_success()
        else:
            reward = self.reward_invalid_move

        # Calculate action mask once for efficiency
        mask = self.action_masks()

        # Check if the game is finished (no more moves possible)
        finished = False
        if not done:
            if FigureSudokuEnv.get_empty_fields(self.state) == 0:
                finished = True
            else:
                finished = not np.any(mask)

        self.rewards.append(reward)
        mean_reward = np.mean(np.array(self.rewards))

        # An episode ends when no further move is possible or the maximum number of time steps is reached
        episode_over = done or finished or (self.max_steps is not None and self.current_step >= self.max_steps)

        if episode_over and not done:
            if self.gui is not None:
                self.gui.show_failure()

        info = {"action_mask": mask}

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
        
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color

        # Check if the cell is already fully occupied
        if self.state[row, col, 0] != Geometry.EMPTY.value and self.state[row, col, 1] != Color.EMPTY.value:
            return False

        # If geometry is pre-filled, the action's geometry must match
        if self.state[row, col, 0] != Geometry.EMPTY.value and self.state[row, col, 0] != g_val:
            return False
        
        # If color is pre-filled, the action's color must match
        if self.state[row, col, 1] != Color.EMPTY.value and self.state[row, col, 1] != c_val:
            return False

        if not FigureSudokuEnv.is_figure_available(self.state, geometry, color):
            return False

        if not FigureSudokuEnv.can_move(self.state, row, col, geometry, color):
            return False

        return True

    def perform_action(self, action):
        (geometry, color) = action[0]
        (row, col) = action[1]
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color
        self.state[row, col] = [g_val, c_val]

        if self.gui is not None:
            self.gui.display_state(self.state)

    @staticmethod
    def is_field_empty(state, row, col):
        # A field is considered empty if it's missing either geometry or color
        return state[row, col, 0] == Geometry.EMPTY.value or state[row, col, 1] == Color.EMPTY.value

    @staticmethod
    def is_figure_available(state, geometry, color):
        # state is (4, 4, 2)
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color
        
        exists = np.any((state[:, :, 0] == g_val) & (state[:, :, 1] == c_val))
        return not exists

    @staticmethod
    def get_empty_fields(state):
        # A field is considered empty if it's missing either geometry or color
        return np.sum((state[:, :, 0] == Geometry.EMPTY.value) | (state[:, :, 1] == Color.EMPTY.value))

    @staticmethod
    def is_done(state):
        # Done means all cells have both geometry and color
        return np.all(state[:, :, 0] != Geometry.EMPTY.value) and np.all(state[:, :, 1] != Color.EMPTY.value)

    @staticmethod
    def can_move(state, row, col, geometry, color):
        g_val = geometry.value if hasattr(geometry, 'value') else geometry
        c_val = color.value if hasattr(color, 'value') else color

        # Check row, excluding the current cell
        row_state_g = np.delete(state[row, :, 0], col)
        row_state_c = np.delete(state[row, :, 1], col)
        if np.any(row_state_g == g_val) or np.any(row_state_c == c_val):
            return False
        
        # Check column, excluding the current cell
        col_state_g = np.delete(state[:, col, 0], row)
        col_state_c = np.delete(state[:, col, 1], row)
        if np.any(col_state_g == g_val) or np.any(col_state_c == c_val):
            return False

        return True
