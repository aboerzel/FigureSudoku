import itertools
import random
import numpy as np

from shapes import Geometry, Color


class SudokuGenerator:
    def __init__(self, geometries, colors):
        self.geometries = geometries
        self.colors = colors
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        self.state_size = self.rows * self.cols

        self.shapes = list(itertools.product(geometries, colors))

    def generate(self, initial_items=4):
        solved = False
        state = []

        while not solved:

            state = np.array([x for x in [[(Geometry.EMPTY.value, Color.EMPTY.value)] * self.rows] * self.cols])
            possibilities = [[self.shapes for i in range(self.cols)] for j in range(self.rows)]

            # random select first cell, geometry and color
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.rows - 1)
            geometry = random.choice(self.geometries)
            color = random.choice(self.colors)

            while True:
                state[row][col] = np.array([geometry.value, color.value])
                possibilities[row][col] = []

                for r in range(self.rows):
                    for c in range(self.cols):
                        possibilities[r][c] = [item for item in possibilities[r][c] if not (item[0] == geometry and item[1] == color)]

                for i in range(self.cols):
                    possibilities[row][i] = [item for item in possibilities[row][i] if item[0] != geometry and item[1] != color]

                for i in range(self.rows):
                    possibilities[i][col] = [item for item in possibilities[i][col] if item[0] != geometry and item[1] != color]

                min_length = 999
                for r in range(self.rows):
                    for c in range(self.cols):
                        length = len(possibilities[r][c])
                        if 0 < length <= min_length:
                            min_length = length

                cells = []
                for r in range(self.rows):
                    for c in range(self.cols):
                        if len(possibilities[r][c]) == min_length:
                            cells.append((r, c))

                if len(cells) < 1:
                    break

                (row, col) = random.choice(cells)
                possible_shapes = possibilities[row][col]
                (geometry, color) = random.choice(possible_shapes)
                #print(geometry, color)

            solved = np.all(state.reshape(self.state_size, 2)[:, 0] != Geometry.EMPTY.value) and np.all(state.reshape(self.state_size, 2)[:, 1] != Color.EMPTY.value)
            #print(f'solved: {solved}')

        init_state = np.copy(state.reshape(self.state_size, 2))
        idx = np.random.choice(range(self.state_size), initial_items, replace=False)
        init_state[np.delete(range(self.state_size), idx, axis=0)] = [Geometry.EMPTY.value, Color.EMPTY.value]
        init_state = init_state.reshape(self.rows, self.cols, 2)
        #print("final state:")
        #print(state)
        #print()
        #print("init state:")
        #print(init_state)

        return state, init_state


#env = FigureSudokuEnv()
#generator = SudokuGenerator(env.geometries, env.colors)
#print(generator.generate(initial_items=4))
