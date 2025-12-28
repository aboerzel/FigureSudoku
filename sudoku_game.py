import math
import random
import time
from threading import Thread
from tkinter import *
import tkinter as tk
import numpy as np
from sb3_contrib import MaskablePPO

import config
from figure_sudoku_env import FigureSudokuEnv
from shapes import Geometry, Color

class GridCell:
    def __init__(self, board, row, col, width=80, height=80):
        self.board = board
        self.row = row
        self.col = col

        self.width = width
        self.height = height

        self.x = col * self.width
        self.y = row * self.height

        self.rect = board.create_rectangle(self.x, self.y, self.x + width, self.y + height, outline="black", fill="white")
        self.board.tag_bind(self.rect, "<Button-1>", self.clicked)

        self.shape = None

    def clicked(self, event):
        return

        if self.shape is None:
            #self.board.itemconfig(self.rect, fill='green', outline='red')
            shape = self.get_random_shape()
            color = self.get_random_color()
            self.set_shape(shape, color)
        else:
            #self.board.itemconfig(self.rect, fill='orange', outline='gray')
            self.clear()

        print(f'Cell {self.row + 1} {self.col + 1} clicked.')

    def set_shape(self, geometry, color):
        self.clear()
        # Reset background to white
        self.board.itemconfig(self.rect, fill='white')
        
        if geometry != Geometry.EMPTY.value and color != Color.EMPTY.value:
            # Full figure: normal display
            self.shape = self.get_shape(geometry, color)
        elif geometry != Geometry.EMPTY.value:
            # Geometry only: display as gray shape
            self.shape = self.get_shape(geometry, None)
        elif color != Color.EMPTY.value:
            # Color only: fill background with that color
            bg_color = self.get_color(color)
            self.board.itemconfig(self.rect, fill=bg_color)

    def clear(self):
        if self.shape is not None:
            self.board.delete(self.shape)
            self.shape = None

    def create_triangle(self, color='red'):
        r = 15
        ha = self.height - 2 * r
        a = 2 * ha / math.sqrt(3)

        mx = self.width / 2 + self.x
        my = self.height / 2 + self.y

        ax = mx - a / 2
        ay = my + ha / 2

        bx = mx + a / 2
        by = my + ha / 2

        cx = mx
        cy = my - ha / 2

        points = [ax, ay, bx, by, cx, cy]
        shape = self.board.create_polygon(points, smooth=False, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_circle(self, color='red'):
        r = 15
        x1 = self.x + r
        y1 = self.y + r
        x2 = self.x + self.width - r
        y2 = self.y + self.height - r
        shape = self.board.create_oval(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_quadrat(self, color='red'):
        r = 15
        x1 = self.x + r
        y1 = self.y + r
        x2 = self.x + self.width - r
        y2 = self.y + self.height - r
        shape = self.board.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_hexagon(self, color='red'):
        r = 15
        a = (self.width - (2 * r)) / 2
        ri = math.sqrt(3) * a / 2

        mx = self.width / 2 + self.x
        my = self.height / 2 + self.y

        fx = mx - a
        fy = my

        ax = mx - (a/2)
        ay = my + ri

        bx = mx + (a/2)
        by = my + ri

        cx = mx + a
        cy = my

        dx = mx + (a/2)
        dy = my - ri

        ex = mx - (a/2)
        ey = my - ri

        points = [ax, ay, bx, by, cx, cy, dx, dy, ex, ey, fx, fy]
        shape = self.board.create_polygon(points, smooth=False, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    @staticmethod
    def get_random_shape():
        n = random.randint(0, 3)
        return {
            0: Geometry.QUADRAT,
            1: Geometry.TRIANGLE,
            2: Geometry.CIRCLE,
            3: Geometry.HEXAGON
        }[n]

    @staticmethod
    def get_random_color():
        n = random.randint(0, 3)
        return {
            0: Color.RED,
            1: Color.GREEN,
            2: Color.YELLOW,
            3: Color.BLUE
        }[n]

    @staticmethod
    def get_color(color):
        return {
            Color.RED.value: 'red',
            Color.GREEN.value: 'green',
            Color.YELLOW.value: 'yellow',
            Color.BLUE.value: 'blue'
        }[color]

    def get_shape(self, shape, color):
        func = {
            Geometry.QUADRAT.value: self.create_quadrat,
            Geometry.TRIANGLE.value: self.create_triangle,
            Geometry.CIRCLE.value: self.create_circle,
            Geometry.HEXAGON.value: self.create_hexagon
        }.get(shape)
        if func:
            color_str = 'lightgray' if color is None else self.get_color(color)
            return func(color=color_str)
        return None


class SudokuApp(tk.Tk):
    def __init__(self, model, env):
        super().__init__()

        self.model = model
        self.env = env
        self.level = 3

        self.rows = env.rows
        self.cols = env.cols
        self.cell_width = self.cell_height = 82

        self.width = self.cell_width * self.cols + 120
        self.height = self.cell_height * self.rows

        self.geometry(f"{self.width}x{self.height}")
        self.title('Figure Sudoku')
        self.resizable(False, False)

        # configure the grid
        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=3)

        self.grid = np.empty((self.rows, self.cols), dtype=object)

        self.create_board()

        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.stop_solve = False
        self.game_state = None
        self.obs = None

    def create_game(self):
        self.level = self.level_slider.get()
        self.obs, _info = self.env.reset_with_level(
            level=self.level,
            unique=True,
            partial_prob=self.partial_prob_slider.get(),
            partial_mode=self.partial_mode_slider.get()
        )
        self.game_state = self.env.state.copy()
        self.display_state(self.game_state)

    def solve_game(self):
        if self.game_state is not None:
            #self.solve(self.game_state)
            Thread(target=self.solve, args=[]).start()

    def close_window(self):
        self.stop_solve = True
        self.destroy()

    def create_board(self):

        board = Canvas(self)

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col] = GridCell(board, row, col, width=self.cell_width, height=self.cell_height)

        board.pack(fill=BOTH, expand=True)

        sidebar = Frame(board, width=100, bg='grey')

        self.reset_button = Button(sidebar, text="New Game", command=self.create_game)
        self.reset_button.pack(anchor=CENTER, padx=5, pady=5)

        self.solve_button = Button(sidebar, text="Solve", command=self.solve_game)
        self.solve_button.pack(anchor=CENTER, padx=5, pady=5)

        self.level_slider = Scale(sidebar, from_=1, to=self.rows * self.cols, orient=HORIZONTAL, label="Level", bg='grey', highlightthickness=0)
        self.level_slider.set(self.level)
        self.level_slider.pack(anchor=CENTER, padx=5, pady=5)

        self.partial_prob_slider = Scale(sidebar, from_=0.0, to=1.0, resolution=0.1, orient=HORIZONTAL, label="Partial Prob", bg='grey', highlightthickness=0)
        self.partial_prob_slider.set(config.PARTIAL_PROB)
        self.partial_prob_slider.pack(anchor=CENTER, padx=5, pady=5)

        self.partial_mode_slider = Scale(sidebar, from_=0, to=2, orient=HORIZONTAL, label="Partial Mode", bg='grey', highlightthickness=0)
        self.partial_mode_slider.set(config.PARTIAL_MODE)
        self.partial_mode_slider.pack(anchor=CENTER, padx=5, pady=5)

        sidebar.pack(anchor=E, fill=Y, expand=False, side=RIGHT)

    def display_state(self, state):
        self.after(0, self._display_state, state)

    def _display_state(self, state):
        # state ist hier bereits (rows, cols, 2) direkt aus env.state
        for row in range(self.rows):
            for col in range(self.cols):
                (geometry, color) = state[row][col]
                self.grid[row][col].set_shape(geometry, color)

    def _set_controls_state(self, state):
        self.after(0, self.__set_controls_state, state)

    def __set_controls_state(self, state):
        controls = [
            self.solve_button, self.reset_button, self.level_slider,
            self.partial_prob_slider, self.partial_mode_slider
        ]
        for control in controls:
            control.config(state=state)

    def solve(self):
        self._set_controls_state(DISABLED)
        actions = []
        done = False

        for move_count in range(1, self.level + 1):
            if self.stop_solve:
                break

            action_masks = self.env.action_masks()
            action, _ = self.model.predict(self.obs, action_masks=action_masks, deterministic=True)
            actions.append(action)

            self.obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.game_state = self.env.state.copy()
            self.display_state(self.game_state)
            time.sleep(0.5)

            if done:
                print(f'Sudoku solved in {move_count} moves! {np.array(actions)}')
                break
        else:
            if not done:
                print(f'Sudoku could not be solved within the maximum number of {self.level} moves!')

        if not self.stop_solve:
            self._set_controls_state(NORMAL)


if __name__ == "__main__":
    model = MaskablePPO.load(config.MODEL_PATH)
    env = FigureSudokuEnv(
        reward_solved=config.REWARD_SOLVED,
        reward_valid_move_base=config.REWARD_VALID_MOVE_BASE,
        reward_invalid_move=config.REWARD_INVALID_MOVE
    )
        
    app = SudokuApp(model, env)
    app.mainloop()
