import math
import random
import time
from threading import Thread
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import numpy as np
from sb3_contrib import MaskablePPO

import config
from figure_sudoku_env import FigureSudokuEnv
from shapes import Geometry, Color

class GridCell:
    def __init__(self, board, row, col, width=80, height=80, app=None):
        self.board = board
        self.row = row
        self.col = col
        self.app = app

        self.width = width
        self.height = height

        self.x = col * self.width
        self.y = row * self.height

        # Schachbrett-Hintergrund für bessere Orientierung
        bg_color = "#f9f9f9" if (row + col) % 2 == 0 else "#ffffff"
        self.rect = board.create_rectangle(self.x, self.y, self.x + width, self.y + height, outline="#cccccc", fill=bg_color, width=1)
        self.board.tag_bind(self.rect, "<Button-1>", self.clicked)
        self.board.tag_bind(self.rect, "<ButtonRelease-1>", self.released)
        self.board.tag_bind(self.rect, "<Button-3>", self.right_clicked)
        self.board.tag_bind(self.rect, "<Enter>", self.enter)
        self.board.tag_bind(self.rect, "<Leave>", self.leave)

        self.shape = None
        self.preview_shape = None

    def clicked(self, event):
        if self.app and hasattr(self.app, 'grid') and \
           self.row < self.app.rows and self.col < self.app.cols and \
           self.app.grid[self.row, self.col] == self:
            if hasattr(self.app, 'cell_clicked'):
                self.app.cell_clicked(self.row, self.col)

    def released(self, event):
        if self.app and hasattr(self.app, 'grid') and \
           self.row < self.app.rows and self.col < self.app.cols and \
           self.app.grid[self.row, self.col] == self:
            if hasattr(self.app, 'cell_released'):
                self.app.cell_released(self.row, self.col)

    def right_clicked(self, event):
        if self.app and hasattr(self.app, 'grid') and \
           self.row < self.app.rows and self.col < self.app.cols and \
           self.app.grid[self.row, self.col] == self:
            if hasattr(self.app, 'cell_right_clicked'):
                self.app.cell_right_clicked(self.row, self.col)

    def enter(self, event):
        if self.app and hasattr(self.app, 'grid') and \
           self.row < self.app.rows and self.col < self.app.cols and \
           self.app.grid[self.row, self.col] == self:
            if hasattr(self.app, 'cell_enter'):
                self.app.cell_enter(self.row, self.col)

    def leave(self, event):
        if self.app and hasattr(self.app, 'grid') and \
           self.row < self.app.rows and self.col < self.app.cols and \
           self.app.grid[self.row, self.col] == self:
            if hasattr(self.app, 'cell_leave'):
                self.app.cell_leave(self.row, self.col)

    def set_shape(self, geometry, color):
        self.clear()
        # Reset background
        bg_color = "#f9f9f9" if (self.row + self.col) % 2 == 0 else "#ffffff"
        self.board.itemconfig(self.rect, fill=bg_color)
        
        if geometry != Geometry.EMPTY.value and color != Color.EMPTY.value:
            # Full figure: normal display
            self.shape = self.get_shape(geometry, color)
        elif geometry != Geometry.EMPTY.value:
            # Geometry only: display as a solid gray shape
            self.shape = self.get_shape(geometry, None)
        elif color != Color.EMPTY.value:
            # Color only: display as a dashed rectangle with cross-hatch
            padding = 5
            x1 = self.x + padding
            y1 = self.y + padding
            x2 = self.x + self.width - padding
            y2 = self.y + self.height - padding
            self.shape = self.board.create_rectangle(x1, y1, x2, y2, outline=self.get_color(color), width=1, dash=(2, 2), stipple='gray25', fill=self.get_color(color))
            self.board.tag_bind(self.shape, "<Button-1>", self.clicked)
            self.board.tag_bind(self.shape, "<ButtonRelease-1>", self.released)
            self.board.tag_bind(self.shape, "<Button-3>", self.right_clicked)
            self.board.tag_bind(self.shape, "<Enter>", self.enter)
            self.board.tag_bind(self.shape, "<Leave>", self.leave)
        else:
            self.shape = None

    def set_preview(self, geometry, color, valid=True):
        self.clear_preview()
        
        # Anforderung: "als wäre sie schon ausgeführt, nur mit einer entsprechenden umrandung"
        # Wir ändern die Hintergrundfarbe der Zelle und zeichnen die Form
        if valid:
            self.board.itemconfig(self.rect, fill="#e8f5e9") # Light green for valid preview
            outline_color = "#4caf50"
        else:
            self.board.itemconfig(self.rect, fill="#ffebee") # Light red for invalid preview
            outline_color = "#e53935"

        # Preview shape drawing
        if geometry != Geometry.EMPTY.value and color != Color.EMPTY.value:
            self.preview_shape = self.get_shape(geometry, color, dash=(4, 2))
        elif geometry != Geometry.EMPTY.value:
            self.preview_shape = self.get_shape(geometry, None, dash=(4, 2))
        elif color != Color.EMPTY.value:
            # Color only: display as a dashed rectangle with cross-hatch
            padding = 5
            x1 = self.x + padding
            y1 = self.y + padding
            x2 = self.x + self.width - padding
            y2 = self.y + self.height - padding
            self.preview_shape = self.board.create_rectangle(x1, y1, x2, y2, outline=self.get_color(color), width=1, dash=(2, 2), stipple='gray12', fill=self.get_color(color))
        
        # Add a thick border to indicate preview
        tag = f"preview_border_{self.row}_{self.col}"
        padding = 2
        self.board.create_rectangle(
            self.x + padding, self.y + padding, 
            self.x + self.width - padding, self.y + self.height - padding,
            outline=outline_color, width=2, tags=tag
        )
        
        if self.preview_shape:
            if isinstance(self.preview_shape, str): # if it's a tag
                self.preview_border = tag
            else:
                self.preview_border = tag
        else:
            self.preview_border = tag

    def clear_preview(self):
        # Reset background to normal
        bg_color = "#f9f9f9" if (self.row + self.col) % 2 == 0 else "#ffffff"
        self.board.itemconfig(self.rect, fill=bg_color)
        
        if self.preview_shape is not None:
            self.board.delete(self.preview_shape)
            self.preview_shape = None
            
        if hasattr(self, 'preview_border') and self.preview_border:
            self.board.delete(self.preview_border)
            self.preview_border = None

    def clear(self):
        if self.shape is not None:
            self.board.delete(self.shape)
            self.shape = None

    def create_triangle(self, color='red', dash=None):
        padding = self.height * 0.2
        ha = self.height - 2 * padding
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
        if dash:
            shape = self.board.create_polygon(points, smooth=False, fill='', outline=color, width=2, dash=dash)
        else:
            shape = self.board.create_polygon(points, smooth=False, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        self.board.tag_bind(shape, "<ButtonRelease-1>", self.released)
        self.board.tag_bind(shape, "<Button-3>", self.right_clicked)
        self.board.tag_bind(shape, "<Enter>", self.enter)
        self.board.tag_bind(shape, "<Leave>", self.leave)
        return shape

    def create_circle(self, color='red', dash=None):
        padding = self.height * 0.2
        x1 = self.x + padding
        y1 = self.y + padding
        x2 = self.x + self.width - padding
        y2 = self.y + self.height - padding
        if dash:
            shape = self.board.create_oval(x1, y1, x2, y2, fill='', outline=color, width=2, dash=dash)
        else:
            shape = self.board.create_oval(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        self.board.tag_bind(shape, "<ButtonRelease-1>", self.released)
        self.board.tag_bind(shape, "<Button-3>", self.right_clicked)
        self.board.tag_bind(shape, "<Enter>", self.enter)
        self.board.tag_bind(shape, "<Leave>", self.leave)
        return shape

    def create_quadrat(self, color='red', dash=None):
        padding = self.height * 0.2
        x1 = self.x + padding
        y1 = self.y + padding
        x2 = self.x + self.width - padding
        y2 = self.y + self.height - padding
        if dash:
            shape = self.board.create_rectangle(x1, y1, x2, y2, fill='', outline=color, width=2, dash=dash)
        else:
            shape = self.board.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        self.board.tag_bind(shape, "<ButtonRelease-1>", self.released)
        self.board.tag_bind(shape, "<Button-3>", self.right_clicked)
        self.board.tag_bind(shape, "<Enter>", self.enter)
        self.board.tag_bind(shape, "<Leave>", self.leave)
        return shape

    def create_hexagon(self, color='red', dash=None):
        padding = self.height * 0.2
        a = (self.width - (2 * padding)) / 2
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
        if dash:
            shape = self.board.create_polygon(points, smooth=False, fill='', outline=color, width=2, dash=dash)
        else:
            shape = self.board.create_polygon(points, smooth=False, fill=color, outline='')
        self.board.tag_bind(shape, "<Button-1>", self.clicked)
        self.board.tag_bind(shape, "<ButtonRelease-1>", self.released)
        self.board.tag_bind(shape, "<Button-3>", self.right_clicked)
        self.board.tag_bind(shape, "<Enter>", self.enter)
        self.board.tag_bind(shape, "<Leave>", self.leave)
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

    def get_shape(self, shape, color, dash=None):
        func = {
            Geometry.QUADRAT.value: self.create_quadrat,
            Geometry.TRIANGLE.value: self.create_triangle,
            Geometry.CIRCLE.value: self.create_circle,
            Geometry.HEXAGON.value: self.create_hexagon
        }.get(shape)
        if func:
            color_str = 'lightgray' if color is None else self.get_color(color)
            return func(color=color_str, dash=dash)
        return None


class SudokuApp(tk.Tk):
    def __init__(self, model, env):
        super().__init__()

        self.model = model
        self.env = env
        self.level = 10

        self.rows = env.rows
        self.cols = env.cols
        
        self.sidebar_width = 200
        # Mindesthöhe für die Sidebar (ca. 450px) sicherstellen
        self.height = max(450, 30) + 30 # Platz für Sidebar + Statusleiste
        # Da wir wollen, dass das Board die Höhe der Sidebar ausfüllt:
        available_board_height = max(450, 450) # Die Höhe der Sidebar ohne Statusleiste
        self.cell_height = self.cell_width = available_board_height // self.rows

        self.height = (self.cell_height * self.rows) + 30
        self.width = self.cell_width * self.cols + self.sidebar_width
        self.help_sidebar_width = 660
        self.help_sidebar_visible = False
        
        # Drag and Drop State
        self.dragging_item = None # (type, value) where type is 'geometry' or 'color'
        self.drag_widget = None # The visual representation being dragged
        self.last_hovered_cell = None # (row, col)

        self.geometry(f"{self.width}x{self.height}")
        self.title('Figure Sudoku')
        self.resizable(False, False)

        # Style colors
        self.bg_sidebar = "#333333"
        self.fg_sidebar = "#ffffff"
        self.accent_color = "#4a90e2"

        self.grid = np.empty((self.rows, self.cols), dtype=object)

        self.create_board()

        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.stop_solve = False
        self.game_state = np.full((self.rows, self.cols, 2), Geometry.EMPTY.value, dtype=np.int32)
        self.initial_state = self.game_state.copy()
        self.obs = None
        
        # Initialisiere das erste Spiel
        self.after(100, self.create_game)

    def create_game(self):
        self.set_status("Generiere Spiel...")
        self.level = self.level_slider.get()
        self.obs, _info = self.env.reset_with_level(level=self.level)
        self.game_state = self.env.state.copy()
        self.initial_state = self.game_state.copy()
        self.display_state(self.game_state)
        self.set_status("Bereit")

    def solve_game(self):
        if self.game_state is not None:
            Thread(target=self.solve, args=[]).start()

    def close_window(self):
        self.stop_solve = True
        self.destroy()

    def set_status(self, text):
        self.after(0, lambda: self.status_label.config(text=f" Status: {text}"))

    def update_partial_prob_label(self, value):
        self.partial_prob_label.config(text=f"Häufigkeit: {value}%")

    def update_level_label(self, value):
        self.level_label.config(text=f"Level: {value}")

    def update_partial_mode_label(self, value):
        self.partial_mode_label.config(text=f"Anzahl Felder: {value}")

    def toggle_help(self):
        if self.help_sidebar_visible:
            self.help_sidebar.pack_forget()
            self.help_sidebar_visible = False
            new_width = self.width
            self.help_button.config(text="?")
        else:
            self.help_sidebar.pack(side=RIGHT, fill=Y, expand=False)
            self.help_sidebar_visible = True
            new_width = self.width + self.help_sidebar_width
            self.help_button.config(text="✕")
        
        self.geometry(f"{new_width}x{self.height}")

    def start_drag(self, event, item_type, value):
        self.dragging_item = (item_type, value)
        # Create a small floating widget to follow the mouse
        self.drag_widget = tk.Canvas(self, width=30, height=30, bg="white", highlightthickness=1, highlightbackground="black")
        cell = GridCell(self.drag_widget, 0, 0, width=30, height=30, app=None)
        if item_type == 'geometry':
            cell.set_shape(value, Color.EMPTY.value) # Neutraler Hintergrund für Formen
        else:
            cell.set_shape(Geometry.CIRCLE.value, value) # Show color on a circle
        
        self.drag_widget.place(x=event.x_root - self.winfo_rootx(), y=event.y_root - self.winfo_rooty(), anchor=CENTER)
        self.set_status(f"Ziehe {item_type}...")

    def on_drag(self, event):
        if self.drag_widget:
            # Update drag widget position
            x = event.x_root - self.winfo_rootx()
            y = event.y_root - self.winfo_rooty()
            self.drag_widget.place(x=x, y=y)
            
            # Use board_canvas's winfo_root to get coordinates relative to the board
            bx = event.x_root - self.board_canvas.winfo_rootx()
            by = event.y_root - self.board_canvas.winfo_rooty()
            
            cell_found = None
            if 0 <= bx < self.cell_width * self.cols and 0 <= by < self.cell_height * self.rows:
                col = int(bx // self.cell_width)
                row = int(by // self.cell_height)
                cell_found = (row, col)
            
            if cell_found != self.last_hovered_cell:
                if self.last_hovered_cell:
                    self.cell_leave(*self.last_hovered_cell)
                if cell_found:
                    self.cell_enter(*cell_found)
                self.last_hovered_cell = cell_found

    def stop_drag(self, event):
        # Determine cell one last time using root coordinates to be precise
        bx = event.x_root - self.board_canvas.winfo_rootx()
        by = event.y_root - self.board_canvas.winfo_rooty()
        
        target_cell = None
        if 0 <= bx < self.cell_width * self.cols and 0 <= by < self.cell_height * self.rows:
            col = int(bx // self.cell_width)
            row = int(by // self.cell_height)
            target_cell = (row, col)

        if self.drag_widget:
            self.drag_widget.destroy()
            self.drag_widget = None
        
        # Clear any existing preview before making the move
        if self.last_hovered_cell:
            self.cell_leave(*self.last_hovered_cell)
        
        # Check if we dropped on a cell
        if target_cell:
            # First clear preview on the target cell as well to avoid artifacts
            self.cell_leave(*target_cell)
            self.cell_released(*target_cell)
        
        self.last_hovered_cell = None
        self.dragging_item = None
        self.set_status("Bereit")

    def clear_drag_item(self):
        self.dragging_item = None

    def cell_enter(self, row, col):
        if self.dragging_item and self.game_state is not None:
            item_type, value = self.dragging_item
            curr_g, curr_c = self.game_state[row, col]
            
            new_g = curr_g
            new_c = curr_c
            if item_type == 'geometry':
                new_g = value
            else:
                new_c = value
            
            # Calculate target based on current cell content
            actual_g, actual_c = self._calculate_target(row, col, new_g, new_c)
            
            # Check if this addition would be valid
            valid = self._is_move_valid(row, col, actual_g, actual_c)
            
            # We show preview regardless, but the visual style (color/border) 
            # will indicate if it's valid or not (light blue vs light red)
            self.grid[row, col].set_preview(actual_g, actual_c, valid=valid)

    def cell_leave(self, row, col):
        self.grid[row, col].clear_preview()

    def _calculate_target(self, row, col, new_g, new_c):
        # Wir geben einfach die neuen Werte zurück, da wir nun auch 
        # vorbelegte Felder aktualisieren wollen. 
        # Die Validierung erfolgt später in _is_move_valid.
        return new_g, new_c

    def create_board(self):
        # Main container
        main_container = Frame(self)
        main_container.pack(fill=BOTH, expand=True)

        # Center Board Canvas container
        board_container = Frame(main_container, width=self.cell_width * self.cols)
        board_container.pack(side=LEFT, fill=BOTH, expand=True)

        # Board Canvas - füllt den board_container aus
        canvas_width = self.cell_width * self.cols
        canvas_height = self.cell_height * self.rows
        self.board_canvas = Canvas(board_container, width=canvas_width, height=canvas_height, highlightthickness=0)
        self.board_canvas.pack(expand=True)

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col] = GridCell(self.board_canvas, row, col, width=self.cell_width, height=self.cell_height, app=self)

        # Sidebar (Links)
        sidebar = Frame(main_container, width=self.sidebar_width, bg=self.bg_sidebar)
        sidebar.pack(side=LEFT, fill=Y, expand=False)
        sidebar.pack_propagate(False)

        # Help Sidebar (Rechts, initial versteckt)
        self.help_sidebar = Frame(main_container, width=self.help_sidebar_width, bg="#f0f0f0", bd=1, relief=SUNKEN)
        self.help_sidebar.pack_propagate(False)

        # Help Sidebar Header with Close Button
        help_header = Frame(self.help_sidebar, bg="#e0e0e0", height=30)
        help_header.pack(side=TOP, fill=X)
        help_header.pack_propagate(False)
        Label(help_header, text="ANLEITUNG", font=("Arial", 10, "bold"), bg="#e0e0e0", fg="#333333").pack(side=LEFT, padx=10)
        Button(help_header, text="✕", command=self.toggle_help, bg="#e0e0e0", fg="#333333", 
               font=("Arial", 10), relief="flat", bd=0, activebackground="#cccccc").pack(side=RIGHT, padx=5)

        # Content for Help Sidebar (rest of the code follows)
        help_canvas = Canvas(self.help_sidebar, bg="#f0f0f0", highlightthickness=0)
        scrollbar = Scrollbar(self.help_sidebar, orient=VERTICAL, command=help_canvas.yview)
        scrollable_frame = Frame(help_canvas, bg="#f0f0f0")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: help_canvas.configure(scrollregion=help_canvas.bbox("all"))
        )

        help_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=self.help_sidebar_width-25)
        help_canvas.configure(yscrollcommand=scrollbar.set)

        help_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Spalten-Container für 3-spaltiges Layout
        columns_container = Frame(scrollable_frame, bg="#f0f0f0")
        columns_container.pack(fill=X, expand=True)
        
        col_padding = 10
        left_col = Frame(columns_container, bg="#f0f0f0")
        left_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, col_padding), anchor=N)
        
        mid_col = Frame(columns_container, bg="#f0f0f0")
        mid_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, col_padding), anchor=N)

        right_col = Frame(columns_container, bg="#f0f0f0")
        right_col.pack(side=LEFT, fill=BOTH, expand=True, anchor=N)

        def add_help_section(parent, title, text=None):
            col_width = (self.help_sidebar_width - 80) // 3
            Label(parent, text=title, font=("Arial", 10, "bold"), bg="#f0f0f0", fg="#333333", justify=LEFT, wraplength=col_width).pack(anchor=W, pady=(12, 2))
            if text:
                Label(parent, text=text, font=("Arial", 9), bg="#f0f0f0", fg="#555555", justify=LEFT, wraplength=col_width).pack(anchor=W, pady=(0, 5))

        def add_visual_example(parent, geometry, color, label_text):
            frame = Frame(parent, bg="#f0f0f0")
            frame.pack(anchor=W, pady=2, fill=X)
            
            cv_size = 25
            cv = Canvas(frame, width=cv_size, height=cv_size, bg="#f0f0f0", highlightthickness=0)
            cv.pack(side=LEFT)
            
            # Temporary GridCell-like drawing
            cell = GridCell(cv, 0, 0, width=cv_size, height=cv_size, app=self)
            cell.set_shape(geometry, color)
            
            Label(frame, text=label_text, font=("Arial", 9), bg="#f0f0f0", fg="#555555").pack(side=LEFT, padx=5)

        # Linke Spalte (1)
        add_help_section(left_col, "DAS SPIELPRINZIP", 
            "Figure-Sudoku ist eine Variante des klassischen Sudokus, bei der anstelle von Zahlen eine Kombination aus Form und Farbe verwendet wird.\n\n"
            "Ziel ist es, das 4x4 Gitter so zu füllen, dass jede Figur (Form + Farbe) genau einmal vorkommt und die Sudoku-Regeln eingehalten werden.")

        add_help_section(left_col, "DIE REGELN")
        col_w = (self.help_sidebar_width - 80) // 3
        Label(left_col, text="• Jedes Feld muss am Ende eine eindeutige Figur enthalten.", font=("Arial", 9), bg="#f0f0f0", fg="#555555", justify=LEFT, wraplength=col_w).pack(anchor=W)
        Label(left_col, text="• Jede Form darf pro Zeile/Spalte nur einmal vorkommen.", font=("Arial", 9), bg="#f0f0f0", fg="#555555", justify=LEFT, wraplength=col_w).pack(anchor=W)
        Label(left_col, text="• Jede Farbe darf pro Zeile/Spalte nur einmal vorkommen.", font=("Arial", 9), bg="#f0f0f0", fg="#555555", justify=LEFT, wraplength=col_w).pack(anchor=W)
        Label(left_col, text="• Jede Kombination ist im gesamten Gitter einzigartig.", font=("Arial", 9), bg="#f0f0f0", fg="#555555", justify=LEFT, wraplength=col_w).pack(anchor=W)

        # Mittlere Spalte (2)
        add_help_section(mid_col, "FORMEN")
        add_visual_example(mid_col, Geometry.CIRCLE.value, Color.RED.value, "Kreis")
        add_visual_example(mid_col, Geometry.QUADRAT.value, Color.RED.value, "Quadrat")
        add_visual_example(mid_col, Geometry.TRIANGLE.value, Color.RED.value, "Dreieck")
        add_visual_example(mid_col, Geometry.HEXAGON.value, Color.RED.value, "Hexagon")

        add_help_section(mid_col, "FARBEN")
        add_visual_example(mid_col, Geometry.CIRCLE.value, Color.RED.value, "Rot")
        add_visual_example(mid_col, Geometry.CIRCLE.value, Color.GREEN.value, "Grün")
        add_visual_example(mid_col, Geometry.CIRCLE.value, Color.BLUE.value, "Blau")
        add_visual_example(mid_col, Geometry.CIRCLE.value, Color.YELLOW.value, "Gelb")

        # Rechte Spalte (3)
        add_help_section(right_col, "TEILBELEGUNGEN (Level 11+)",
            "Manchmal sind Felder nur teilweise vorgegeben:")
        add_visual_example(right_col, Geometry.CIRCLE.value, Color.EMPTY.value, "Farbe fehlt")
        add_visual_example(right_col, Geometry.EMPTY.value, Color.RED.value, "Form fehlt")

        add_help_section(right_col, "STEUERUNG",
            "• Neues Spiel: Startet eine neue Runde.\n"
            "• Lösen: Lässt die KI das Rätsel lösen.\n"
            "• Level-Slider: Stellt die Schwierigkeit (1-12) ein.\n"
            "• Ziehen von Formen/Farben: Platziert diese auf dem Feld.\n"
            "• Rechtsklick auf Feld: Öffnet Menü zum Löschen von Zügen.")

        # Buttons with style
        button_style = {"width": 18, "pady": 5, "bg": self.accent_color, "fg": "white", "font": ("Arial", 9, "bold"), "relief": "flat"}
        
        self.reset_button = Button(sidebar, text="Neues Spiel", command=self.create_game, **button_style)
        self.reset_button.pack(padx=10, pady=(20, 5))

        self.solve_button = Button(sidebar, text="Lösen", command=self.solve_game, **button_style)
        self.solve_button.pack(padx=10, pady=5)
        if self.model is None:
            self.solve_button.config(state=DISABLED)

        # Manual Entry UI (Drag and Drop)
        Label(sidebar, text="MANUELLES SETZEN", font=("Arial", 8, "bold"), bg=self.bg_sidebar, fg="#888888").pack(pady=(20, 5))
        Label(sidebar, text="(Ziehe Form/Farbe auf Feld)", font=("Arial", 7), bg=self.bg_sidebar, fg="#666666").pack(pady=(0, 5))
        
        # Shapes selection
        shapes_frame = Frame(sidebar, bg=self.bg_sidebar)
        shapes_frame.pack(pady=5)
        
        for g in [Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON]:
            # Preview canvas for shape as a draggable source
            ps = Canvas(shapes_frame, width=30, height=30, bg=self.bg_sidebar, highlightthickness=0)
            ps.pack(side=LEFT, padx=5)
            cell = GridCell(ps, 0, 0, width=30, height=30, app=self)
            cell.set_shape(g.value, Color.EMPTY.value) # Neutraler Hintergrund für Formen
            
            # Bind drag events to the canvas and its elements
            ps.bind("<Button-1>", lambda e, g=g: self.start_drag(e, 'geometry', g.value))
            ps.bind("<B1-Motion>", self.on_drag)
            ps.bind("<ButtonRelease-1>", self.stop_drag)

        # Colors selection
        colors_frame = Frame(sidebar, bg=self.bg_sidebar)
        colors_frame.pack(pady=5)
        
        for c in [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]:
            # Color indicator as a draggable source
            ps = Canvas(colors_frame, width=30, height=30, bg=self.bg_sidebar, highlightthickness=0)
            ps.pack(side=LEFT, padx=5)
            # Use a circle for color dragging
            cell = GridCell(ps, 0, 0, width=30, height=30, app=self)
            cell.create_circle(color=GridCell.get_color(c.value))
            
            # Die ID ist hier nicht so wichtig, da wir nur draggen
            # aber wir müssen sicherstellen, dass die Events auf dem Canvas oder den Items liegen
            ps.bind("<Button-1>", lambda e, c=c: self.start_drag(e, 'color', c.value))
            ps.bind("<B1-Motion>", self.on_drag)
            ps.bind("<ButtonRelease-1>", self.stop_drag)

        # Sliders with labels
        slider_style = {"bg": self.bg_sidebar, "fg": self.fg_sidebar, "highlightthickness": 0, "orient": HORIZONTAL, "troughcolor": "#555555", "length": 160}

        self.level_label = Label(sidebar, text=f"Level: {self.level}", bg=self.bg_sidebar, fg=self.fg_sidebar, font=("Arial", 11, "bold"))
        self.level_label.pack(pady=(20, 0))
        self.level_slider = Scale(sidebar, from_=1, to=12, showvalue=0, command=self.update_level_label, **slider_style)
        self.level_slider.set(self.level)
        self.level_slider.pack(padx=10, pady=(0, 5))

        # Hilfe-Button unten in der Sidebar
        self.help_button = Button(sidebar, text="?", command=self.toggle_help, 
                                 bg=self.bg_sidebar, fg="#aaaaaa", font=("Arial", 12, "bold"), 
                                 relief="flat", bd=0, activebackground="#444444", activeforeground="#ffffff")
        self.help_button.pack(side=BOTTOM, pady=10)

        # Status Bar
        self.status_bar = Frame(self, height=30, bg="#eeeeee", relief=SUNKEN, bd=1)
        self.status_bar.pack(side=BOTTOM, fill=X)
        self.status_label = Label(self.status_bar, text=" Status: Bereit", bg="#eeeeee", font=("Arial", 8))
        self.status_label.pack(side=LEFT)

    def cell_clicked(self, row, col):
        # We handle this in cell_released for Drag and Drop
        pass

    def cell_released(self, row, col):
        if self.game_state is None or not self.dragging_item:
            return
        
        item_type, value = self.dragging_item
        curr_g, curr_c = self.game_state[row, col]
        
        new_g = curr_g
        new_c = curr_c

        if item_type == 'geometry':
            new_g = value
        else:
            new_c = value
        
        target_g, target_c = self._calculate_target(row, col, new_g, new_c)

        # Validate move - also check if it's different from current state
        if self._is_move_valid(row, col, target_g, target_c):
            # Check if something actually changed
            if target_g != curr_g or target_c != curr_c:
                self.game_state[row, col] = [target_g, target_c]
                self.env.state[row, col] = [target_g, target_c]
                self.env.invalidate_action_mask()
                self.display_state(self.game_state)
                
                if FigureSudokuEnv.is_done(self.game_state):
                    self.set_status("Gelöst! Glückwunsch!")
                    messagebox.showinfo("Erfolg", "Du hast das Sudoku gelöst!")
            else:
                self.set_status("Bereit")
        else:
            self.set_status("Ungültiger Zug!")
            # Clear preview when drop fails
            self.grid[row, col].clear_preview()

    def cell_right_clicked(self, row, col):
        if self.game_state is None:
            return

        # Erstelle ein Kontextmenü
        menu = tk.Menu(self, tearoff=0)
        
        curr_g, curr_c = self.game_state[row, col]
        
        # Zeige Löschen-Option an, wenn das Feld nicht leer ist
        if curr_g != Geometry.EMPTY.value or curr_c != Color.EMPTY.value:
            menu.add_command(label="Belegung löschen", command=lambda r=row, c=col: self._delete_cell_content(r, c))
            
        if menu.index('end') is not None: # Wenn mindestens ein Eintrag vorhanden ist
            # Hole Mausposition
            x = self.winfo_pointerx()
            y = self.winfo_pointery()
            menu.post(x, y)

    def _delete_cell_content(self, row, col):
        self.game_state[row, col] = [Geometry.EMPTY.value, Color.EMPTY.value]
        self.env.state[row, col] = [Geometry.EMPTY.value, Color.EMPTY.value]
        self.env.invalidate_action_mask()
        self.display_state(self.game_state)
        self.set_status("Bereit")

    def _is_move_valid(self, row, col, g, c):
        # Wir prüfen die Sudoku Regeln
        # 1. Figur Einzigartigkeit im Gitter (nur wenn vollständig)
        if g != Geometry.EMPTY.value and c != Color.EMPTY.value:
            for r in range(self.rows):
                for _c in range(self.cols):
                    if r == row and _c == col: continue
                    if self.game_state[r, _c, 0] == g and self.game_state[r, _c, 1] == c:
                        return False
        
        # 2. Form Einzigartigkeit in Zeile/Spalte (nur wenn g gesetzt)
        if g != Geometry.EMPTY.value:
            for i in range(4):
                if i != col and self.game_state[row, i, 0] == g: return False
                if i != row and self.game_state[i, col, 0] == g: return False
            
        # 3. Farbe Einzigartigkeit in Zeile/Spalte (nur wenn c gesetzt)
        if c != Color.EMPTY.value:
            for i in range(4):
                if i != col and self.game_state[row, i, 1] == c: return False
                if i != row and self.game_state[i, col, 1] == c: return False
            
        return True

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
            self.solve_button, self.reset_button, self.level_slider
        ]
        for control in controls:
            control.config(state=state)

    def show_info(self, title, message):
        self.after(0, lambda: messagebox.showinfo(title, message))

    def solve(self):
        self._set_controls_state(DISABLED)
        self.set_status("Löse Sudoku...")
        actions = []
        solved = False

        # Aktuelle Beobachtung aus dem Environment holen (falls manuell gespielt wurde)
        self.obs = self.env._get_obs()

        for move_count in range(1, config.MAX_TIMESTEPS + 1):
            if self.stop_solve:
                break

            action_masks = self.env.action_masks()
            action, _ = self.model.predict(self.obs, action_masks=action_masks, deterministic=True)
            actions.append(action)

            self.obs, _, terminated, truncated, _ = self.env.step(action)
            
            # Das Sudoku ist gelöst, wenn FigureSudokuEnv.is_done wahr zurückgibt.
            # terminated kann auch wahr sein, wenn keine Züge mehr möglich sind (Fehlschlag).
            solved = FigureSudokuEnv.is_done(self.env.state)
            
            self.game_state = self.env.state.copy()
            self.display_state(self.game_state)
            time.sleep(0.2)

            if terminated or truncated:
                break

        if not self.stop_solve:
            if solved:
                self.set_status(f"Gelöst in {len(actions)} Zügen!")
                print(f'Sudoku solved in {len(actions)} moves! {np.array(actions)}')
            else:
                self.set_status("Lösen fehlgeschlagen")
                print(f'Sudoku could not be solved within the maximum number of {config.MAX_TIMESTEPS} moves!')
            
            self._set_controls_state(NORMAL)


if __name__ == "__main__":
    try:
        model = MaskablePPO.load(config.MODEL_PATH)
        print(f"Modell erfolgreich geladen von {config.MODEL_PATH}")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        model = None
        
    env = FigureSudokuEnv(level=config.START_LEVEL)
        
    app = SudokuApp(model, env)
    
    if model is None:
        app.after(500, lambda: messagebox.showwarning(
            "Modell nicht gefunden", 
            f"Das trainierte Modell konnte nicht unter '{config.MODEL_PATH}' gefunden werden.\n\n"
            "Der 'Lösen'-Button wird deaktiviert. Du kannst das Spiel aber weiterhin manuell spielen."
        ))
        
    app.mainloop()
