import tkinter as tk
from tkinter import Canvas, BOTH
import math
import numpy as np
from shapes import Geometry, Color

class GridCell:
    def __init__(self, board, row, col, width=80, height=80):
        self.board = board
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.x = row * self.width
        self.y = col * self.height
        # Draw background rectangle
        self.rect = board.create_rectangle(self.y, self.x, self.y + height, self.x + width, outline="black", fill="white")
        self.shape = None

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
        points = [mx - a / 2, my + ha / 2, mx + a / 2, my + ha / 2, mx, my - ha / 2]
        return self.board.create_polygon(points, smooth=False, fill=color, outline='')

    def create_circle(self, color='red'):
        r = 15
        return self.board.create_oval(self.y + r, self.x + r, self.y + self.width - r, self.x + self.height - r, fill=color, outline='')

    def create_quadrat(self, color='red'):
        r = 15
        return self.board.create_rectangle(self.y + r, self.x + r, self.y + self.width - r, self.x + self.height - r, fill=color, outline='')

    def create_hexagon(self, color='red'):
        r = 15
        a = (self.width - (2 * r)) / 2
        ri = math.sqrt(3) * a / 2
        mx, my = self.width / 2 + self.y, self.height / 2 + self.x
        points = [mx - (a/2), my + ri, mx + (a/2), my + ri, mx + a, my, mx + (a/2), my - ri, mx - (a/2), my - ri, mx - a, my]
        return self.board.create_polygon(points, smooth=False, fill=color, outline='')

    def get_color(self, color):
        return {
            Color.RED.value: 'red',
            Color.GREEN.value: 'green',
            Color.YELLOW.value: 'yellow',
            Color.BLUE.value: 'blue'
        }.get(color, 'white')

    def get_shape(self, shape, color):
        funcs = {
            Geometry.QUADRAT.value: self.create_quadrat,
            Geometry.TRIANGLE.value: self.create_triangle,
            Geometry.CIRCLE.value: self.create_circle,
            Geometry.HEXAGON.value: self.create_hexagon
        }
        func = funcs.get(shape)
        if func:
            return func(color=self.get_color(color))
        return None

_root = None

class SudokuVisualizer:
    def __init__(self, env_id=0, rows=4, cols=4, level=1):
        global _root
        try:
            if _root is None:
                _root = tk.Tk()
                _root.withdraw()
            
            self.env_id = env_id
            self.rows = rows
            self.cols = cols
            self.level = level
            self.cell_width = self.cell_height = 80
            self.width = self.cell_width * self.cols
            self.height = self.cell_height * self.rows
            
            # Use Toplevel for additional windows
            self.window = tk.Toplevel(_root)
            self.window.withdraw()
            
            # Grid layout for windows
            cols_per_row = 3
            x_offset = (env_id % cols_per_row) * (self.width + 40) + 50
            y_offset = (env_id // cols_per_row) * (self.height + 80) + 50
            
            self.window.geometry(f"{self.width}x{self.height}+{x_offset}+{y_offset}")
            self.update_title(level)
            self.window.resizable(False, False)
            
            try:
                self.window.attributes("-topmost", True)
            except:
                pass

            self.canvas = Canvas(self.window, width=self.width, height=self.height, bg='white')
            self.canvas.pack(fill=BOTH, expand=True)

            self.grid = np.empty((self.rows, self.cols), dtype=object)
            for r in range(self.rows):
                for c in range(self.cols):
                    self.grid[r, c] = GridCell(self.canvas, r, c, width=self.cell_width, height=self.cell_height)
            
            self.window.deiconify()
            _root.update()
        except Exception as e:
            print(f"Could not initialize GUI for agent {env_id}: {e}")
            self.window = None

    def update_title(self, level):
        if self.window:
            self.level = level
            self.window.title(f'Agent {self.env_id} (Level: {level})')

    def display_state(self, state):
        if not self.window:
            return
        try:
            state = state.reshape(self.rows, self.cols, 2)
            for r in range(self.rows):
                for c in range(self.cols):
                    g, col = state[r, c]
                    self.grid[r, c].set_shape(g, col)
            _root.update()
        except Exception as e:
            self.window = None

    def clear_visual_feedback(self):
        if not self.window:
            return
        try:
            self.canvas.configure(bg='white')
            for r in range(self.rows):
                for c in range(self.cols):
                    self.canvas.itemconfig(self.grid[r, c].rect, fill='white')
            _root.update()
        except Exception as e:
            pass

    def show_success(self):
        if not self.window:
            return
        try:
            self.canvas.configure(bg='lightgreen')
            for r in range(self.rows):
                for c in range(self.cols):
                    self.canvas.itemconfig(self.grid[r, c].rect, fill='lightgreen')
            _root.update()
        except Exception as e:
            pass

    def show_failure(self):
        if not self.window:
            return
        try:
            self.canvas.configure(bg='#ffcccc') # Light red
            for r in range(self.rows):
                for c in range(self.cols):
                    self.canvas.itemconfig(self.grid[r, c].rect, fill='#ffcccc')
            _root.update()
        except Exception as e:
            pass

    def close(self):
        if self.window:
            try:
                self.window.destroy()
            except:
                pass
            self.window = None
