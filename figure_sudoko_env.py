import itertools
import numpy as np
import gym

from gym.spaces import Box, Discrete

from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator
from visualizer import SudokuVisualizer


class FigureSudokuEnv(gym.Env):

    def __init__(self, env_id=0, level=1, max_steps=None, render_gui=False, unique=False, partial_prob=0.0, partial_mode=1,
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
        self.state_size = self.rows * self.cols

        if render_gui:
            self.gui = SudokuVisualizer(env_id=self.env_id, rows=self.rows, cols=self.cols, level=self.level)
        else:
            self.gui = None
        self.figures = np.array([(int(g), int(c)) for g, c in itertools.product(self.geometries, self.colors)])
        fields = np.array(list(itertools.product(np.arange(self.rows), np.arange(self.cols))))
        self.actions = list(itertools.product(self.figures, fields))
        self.action_figures = np.array([a[0] for a in self.actions], dtype=np.int32)
        self.action_fields = np.array([a[1] for a in self.actions], dtype=np.int32)

        self.state = np.full((self.rows, self.cols, 2), Geometry.EMPTY.value, dtype=np.int32)
        self.solved_state = self.state.copy()

        self.action_space = Discrete(n=len(self.actions))

        # Observation Space: Flattened one-hot encoding
        # 4x4 cells * (5 Geometries + 5 Colors) = 160 elements
        # We use a 1D vector to avoid unnecessary image-related warnings in SB3
        self.observation_space = Box(low=0, high=1, shape=(self.rows * self.cols * 10,), dtype=np.float32)

        self.generator = SudokuGenerator(self.geometries, self.colors)
        self._action_mask = None

    def _get_obs(self):
        # Convert state (4, 4, 2) to one-hot encoding (10, 4, 4) then flatten to (160,)
        obs = np.zeros((10, self.rows, self.cols), dtype=np.float32)
        
        # Efficiently fill one-hot using indexing
        rows, cols = np.indices((self.rows, self.cols))
        obs[self.state[:, :, 0], rows, cols] = 1.0
        obs[5 + self.state[:, :, 1], rows, cols] = 1.0
        
        return obs.flatten()

    def reset(self):
        self.current_step = 0
        self.episode += 1
        self._action_mask = None

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

    def action_masks(self) -> np.ndarray:
        if self._action_mask is not None:
            return self._action_mask

        board_g = self.state[:, :, 0]
        board_c = self.state[:, :, 1]
        
        # 1. Figure availability (only completed figures count as used)
        mask_completed = (board_g != 0) & (board_c != 0)
        used_figs = np.zeros((5, 5), dtype=bool)
        for g, c in zip(board_g[mask_completed], board_c[mask_completed]):
            used_figs[g, c] = True
            
        # 2. Row/Col constraints pre-calculation
        row_has_g = np.zeros((self.rows, 5), dtype=bool)
        row_has_c = np.zeros((self.rows, 5), dtype=bool)
        col_has_g = np.zeros((self.cols, 5), dtype=bool)
        col_has_c = np.zeros((self.cols, 5), dtype=bool)
        
        for r in range(self.rows):
            for c in range(self.cols):
                gv, cv = board_g[r, c], board_c[r, c]
                if gv != 0:
                    row_has_g[r, gv] = True
                    col_has_g[c, gv] = True
                if cv != 0:
                    row_has_c[r, cv] = True
                    col_has_c[c, cv] = True
        
        masks = np.ones(len(self.actions), dtype=bool)
        for i in range(len(self.actions)):
            gv, cv = self.action_figures[i]
            r, c = self.action_fields[i]
            
            # Figure already used?
            if used_figs[gv, cv]:
                masks[i] = False
                continue
            
            # Field occupied with different geometry or color?
            curr_g, curr_c = board_g[r, c], board_c[r, c]
            if (curr_g != 0 and curr_g != gv) or (curr_c != 0 and curr_c != cv):
                masks[i] = False
                continue
            
            # Field already full?
            if curr_g != 0 and curr_c != 0:
                masks[i] = False
                continue

            # Sudoku rules (check if figure's G or C exists elsewhere in Row/Col)
            if (row_has_g[r, gv] and curr_g != gv) or (row_has_c[r, cv] and curr_c != cv):
                masks[i] = False
                continue
            if (col_has_g[c, gv] and curr_g != gv) or (col_has_c[c, cv] and curr_c != cv):
                masks[i] = False
                continue
        
        self._action_mask = masks
        return self._action_mask

    def step(self, action):
        self.current_step += 1

        # Get action mask for the current state
        mask_before = self.action_masks()
        is_valid = mask_before[action]

        target_action = self.actions[action]
        reward = 0.0
        done = False

        if is_valid:
            self.valids += 1
            self.perform_action(target_action)

            # Reward for a valid move
            empty_fields = FigureSudokuEnv.get_empty_fields(self.state)
            reward = self.reward_valid_move_base * (1 + empty_fields / self.state_size)

            done = FigureSudokuEnv.is_done(self.state)
            if done:
                reward += self.reward_solved
                if self.gui is not None:
                    self.gui.show_success()
        else:
            reward = self.reward_invalid_move

        # Get action mask for the new state (cached or recomputed)
        mask_after = self.action_masks()

        # Check if the game is finished (no more moves possible)
        finished = False
        if not done:
            if not np.any(mask_after):
                finished = True
            elif FigureSudokuEnv.get_empty_fields(self.state) == 0:
                finished = True

        self.rewards.append(reward)
        mean_reward = np.mean(self.rewards)

        # An episode ends when no further move is possible or the maximum number of time steps is reached
        episode_over = done or finished or (self.max_steps is not None and self.current_step >= self.max_steps)

        if episode_over and not done:
            if self.gui is not None:
                self.gui.show_failure()

        info = {"action_mask": mask_after}

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
        self._action_mask = None

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
