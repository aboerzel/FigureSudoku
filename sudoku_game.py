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
    def __init__(self, board, row, col, width=80, height=80):
        self.board = board
        self.row = row
        self.col = col

        self.width = width
        self.height = height

        self.x = col * self.width
        self.y = row * self.height

        # Schachbrett-Hintergrund für bessere Orientierung
        bg_color = "#f9f9f9" if (row + col) % 2 == 0 else "#ffffff"
        self.rect = board.create_rectangle(self.x, self.y, self.x + width, self.y + height, outline="#cccccc", fill=bg_color, width=1)
        self.board.tag_bind(self.rect, "<Button-1>", self.clicked)

        self.shape = None

    def clicked(self, event):
        return

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
            # Color only: display as a thick colored dashed frame/border inside the cell
            # This indicates that the color is fixed but the shape is missing
            padding = 10
            x1 = self.x + padding
            y1 = self.y + padding
            x2 = self.x + self.width - padding
            y2 = self.y + self.height - padding
            color_str = self.get_color(color)
            
            tag = f"shape_{self.row}_{self.col}"
            
            # The dashed rectangle
            self.board.create_rectangle(
                x1, y1, x2, y2, 
                outline=color_str, width=3, dash=(4, 4), tags=tag
            )
            
            # Hatching (Schraffur) - diagonal lines
            spacing = 8
            # We want lines from top-left to bottom-right: y = x + offset
            # The range of offset:
            # Min: y1 - x2
            # Max: y2 - x1
            for offset in range(int(y1 - x2), int(y2 - x1), spacing):
                # Intersection with x=x1: y = x1 + offset
                # Intersection with x=x2: y = x2 + offset
                # Intersection with y=y1: x = y1 - offset
                # Intersection with y=y2: x = y2 - offset
                
                lx1 = max(x1, y1 - offset)
                ly1 = lx1 + offset
                lx2 = min(x2, y2 - offset)
                ly2 = lx2 + offset
                
                if lx1 < lx2:
                    # Clip to bounds for safety
                    clx1 = max(x1, min(x2, lx1))
                    cly1 = max(y1, min(y2, ly1))
                    clx2 = max(x1, min(x2, lx2))
                    cly2 = max(y1, min(y2, ly2))
                    
                    if abs(clx1 - clx2) > 1 or abs(cly1 - cly2) > 1:
                        self.board.create_line(clx1, cly1, clx2, cly2, fill=color_str, width=1, tags=tag)
            
            # Ensure lines are on top
            self.board.tag_raise(tag)
            
            self.board.tag_bind(tag, "<Button-1>", self.clicked)
            self.shape = tag

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
        self.game_state = None
        self.obs = None

    def create_game(self):
        self.set_status("Generiere Spiel...")
        self.level = self.level_slider.get()
        self.obs, _info = self.env.reset_with_level(level=self.level)
        self.game_state = self.env.state.copy()
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
        board_canvas = Canvas(board_container, width=canvas_width, height=canvas_height, highlightthickness=0)
        board_canvas.pack(expand=True)

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col] = GridCell(board_canvas, row, col, width=self.cell_width, height=self.cell_height)

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
            cell = GridCell(cv, 0, 0, width=cv_size, height=cv_size)
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
            "• Level-Slider: Stellt die Schwierigkeit (1-12) ein.")

        # Buttons with style
        button_style = {"width": 18, "pady": 5, "bg": self.accent_color, "fg": "white", "font": ("Arial", 9, "bold"), "relief": "flat"}
        
        self.reset_button = Button(sidebar, text="Neues Spiel", command=self.create_game, **button_style)
        self.reset_button.pack(padx=10, pady=(20, 5))

        self.solve_button = Button(sidebar, text="Lösen", command=self.solve_game, **button_style)
        self.solve_button.pack(padx=10, pady=5)

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
    model = MaskablePPO.load(config.MODEL_PATH)
    env = FigureSudokuEnv(level=config.START_LEVEL)
        
    app = SudokuApp(model, env)
    app.mainloop()
