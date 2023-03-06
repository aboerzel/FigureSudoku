import math
import random
from threading import Thread
from tkinter import *
import tkinter as tk
import numpy as np
from stable_baselines3 import A2C

import config
from figure_sudoko_env import FigureSudokuEnv
from shapes import Geometry, Color
from sudoku_generator import SudokuGenerator


class GridCell:
    def __init__(self, board, row, col, width=80, height=80):
        self.board = board
        self.row = row
        self.col = col

        self.width = width
        self.height = height

        self.x = row * self.width
        self.y = col * self.height

        self.rect = board.create_rectangle(self.y, self.x, self.y + height, self.x + width, outline="black", fill="white")
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

        if geometry != Geometry.EMPTY.value and color != Color.EMPTY.value:
            self.shape = self.get_shape(geometry, color)

    def clear(self):
        if self.shape is not None:
            self.board.delete(self.shape)
            self.shape = None

    def create_triangle(self, color='red'):
        r = 15
        ha = self.height - 2 * r
        a = 2 * ha / math.sqrt(3)

        mx = self.width / 2 + self.y
        my = self.height / 2 + self.x

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
        x1 = self.y + r
        y1 = self.x + r
        x2 = self.y + self.width - r
        y2 = self.x + self.height - r
        shape = self.board.create_oval(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_quadrat(self, color='red'):
        r = 15
        x1 = self.y + r
        y1 = self.x + r
        x2 = self.y + self.width - r
        y2 = self.x + self.height - r
        shape = self.board.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        return shape

    def create_hexagon(self, color='red'):
        r = 15
        a = (self.width - (2 * r)) / 2
        ri = math.sqrt(3) * a / 2

        mx = self.width / 2 + self.y
        my = self.height / 2 + self.x

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
        return {
            Geometry.QUADRAT.value: self.create_quadrat,
            Geometry.TRIANGLE.value: self.create_triangle,
            Geometry.CIRCLE.value: self.create_circle,
            Geometry.HEXAGON.value: self.create_hexagon
        }[shape](color=self.get_color(color))


class SudokuApp(tk.Tk):
    def __init__(self, model, env, generator):
        super().__init__()

        self.model = model
        self.env = env
        self.generator = generator

        self.rows = 4
        self.cols = 4
        self.cell_width = self.cell_height = 80

        self.width = self.cell_width * self.cols
        self.height = self.cell_height * self.rows

        self.geometry(f"{self.width}x{self.height}")
        self.title('Figure Sudoku')
        self.resizable(False, False)

        # configure the grid
        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=3)

        self.grid = np.array([x for x in [[None] * self.rows] * self.cols])

        self.create_board()

        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.stop_train = False

        level = 3
        game_state = self.new_game(level=level)

        self.train_thread = Thread(target=self.solve, args=game_state).start()

    def close_window(self):
        self.stop_train = True
        self.destroy()

    def create_board(self):

        board = Canvas(self)

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col] = GridCell(board, row, col, width=self.cell_width, height=self.cell_height)

        board.pack(fill=BOTH, expand=1)

    def display_state(self, state):
        for row in range(self.rows):
            for col in range(self. cols):
                (geometry, color) = state[row][col]
                self.grid[row][col].set_shape(geometry, color)

    def new_game(self, level):
        initial_items = (self.rows * self.cols) - level
        _, self.state = self.generator.generate(initial_items=initial_items)
        self.display_state(self.state)
        return self.state.flatten()

    def solve(self, state, max_attempts=10):
        for i in range(1, max_attempts+1):
            action, _states = self.model.predict(state, deterministic=True)
            print(action)
            state, reward, done, info = self.env.step(action)
            self.display_state(state)
            if done:
                print(f'solved in {i} attempts!')
                return

        print(f'could not be solved with the maximum number of {max_attempts} attempts!')


if __name__ == "__main__":
    model = A2C.load(config.MODEL_PATH)
    env = FigureSudokuEnv()
    generator = SudokuGenerator(env.geometries, env.colors)

    app = SudokuApp(model, env, generator)
    app.mainloop()

