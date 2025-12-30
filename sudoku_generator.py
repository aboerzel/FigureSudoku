import itertools
import random
import numpy as np
from copy import deepcopy

from shapes import Geometry, Color


MAX_ATTEMPTS = 100
HCDS_TOLERANCE = 0.5

LEVEL_HCDS_TARGET = {
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0,
    5: 2.5,
    6: 3.0,
    7: 3.5,
    8: 4.0,
    9: 4.5,
    10: 5.0,
    11: 6.0,
    12: 7.0,
}


class SudokuGenerator:
    def __init__(self, geometries, colors):
        self.geometries = [int(g) for g in geometries]
        self.colors = [int(c) for c in colors]
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        self.state_size = self.rows * self.cols
        self.all_shapes = list(itertools.product(self.geometries, self.colors))

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def generate(self, level: int):
        target = LEVEL_HCDS_TARGET[level]

        partial_target = 0
        if level == 11:
            partial_target = 1
        elif level == 12:
            partial_target = 2

        best_state = None
        best_distance = float("inf")

        for _ in range(MAX_ATTEMPTS):
            solved_state = self._generate_full_solution()
            init_state = solved_state.copy()

            indices = list(range(self.state_size))
            random.shuffle(indices)

            partial_used = 0
            
            # Aktueller hcds vorab
            current_hcds = 0.0

            for idx in indices:
                r, c = idx // self.cols, idx % self.cols
                
                # Wenn wir schon nah genug am Ziel sind, aufhören
                # Aber nur wenn wir keine partials mehr brauchen
                if current_hcds >= target and partial_used >= partial_target:
                    break
                
                if init_state[r, c, 0] == Geometry.EMPTY.value and init_state[r, c, 1] == Color.EMPTY.value:
                    continue
                
                old_val = init_state[r, c].copy()

                # 1) Versuch: komplett entfernen
                init_state[r, c] = [Geometry.EMPTY.value, Color.EMPTY.value]
                if self._has_unique_solution(init_state):
                    current_hcds += 0.5 # entspricht empty * 0.5
                    continue

                init_state[r, c] = old_val

                # 2) Versuch: partiell machen (nur wenn noch nicht am Limit)
                if partial_used < partial_target:
                    # Sicherstellen dass es noch nicht partiell ist
                    if old_val[0] != Geometry.EMPTY.value and old_val[1] != Color.EMPTY.value:
                        init_state[r, c] = self._make_partial(old_val)
                        if self._has_unique_solution(init_state):
                            partial_used += 1
                            current_hcds += 1.5 # entspricht partial * 1.5
                        else:
                            init_state[r, c] = old_val

            if partial_used != partial_target:
                continue

            hcds = self._compute_hcds(init_state)

            if abs(hcds - target) <= HCDS_TOLERANCE:
                return solved_state, init_state

            distance = abs(hcds - target)
            if distance < best_distance:
                best_distance = distance
                best_state = deepcopy(init_state)

        return solved_state, best_state

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================
    def _generate_full_solution(self):
        state = np.full((self.rows, self.cols, 2), Geometry.EMPTY.value, dtype=int)
        available = set(self.all_shapes)
        self._solve(state, available)
        return state

    def _has_unique_solution(self, state):
        available = set(self.all_shapes)
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] != Geometry.EMPTY.value and state[r, c, 1] != Color.EMPTY.value:
                    shape = (int(state[r, c, 0]), int(state[r, c, 1]))
                    if shape in available:
                        available.remove(shape)

        return self._count_solutions(state, available, limit=2) == 1

    def _make_partial(self, cell):
        g, c = cell
        if random.choice([True, False]):
            return [g, Color.EMPTY.value]
        return [Geometry.EMPTY.value, c]

    # ==========================================================
    # DIFFICULTY METRIC (HCDS – heuristisch, menschlich)
    # ==========================================================
    def _compute_hcds(self, state):
        empty = 0
        partial = 0

        for r in range(self.rows):
            for c in range(self.cols):
                g, col = state[r, c]
                if g == Geometry.EMPTY.value and col == Color.EMPTY.value:
                    empty += 1
                elif g == Geometry.EMPTY.value or col == Color.EMPTY.value:
                    partial += 1

        # einfache, robuste Metrik
        # empty * 0.4 + partial * 1.2
        # Bei 16 Zellen: 16 * 0.4 = 6.4 max bei leeren Zellen (unmöglich da Lösung eindeutig sein muss)
        # 16 - (4+3+2+1) = 6 Zellen müssen mindestens belegt sein für Eindeutigkeit? Meist mehr.
        return empty * 0.5 + partial * 1.5

    # ==========================================================
    # SOLVER (UNVERÄNDERT, AUS IHREM CODE)
    # ==========================================================
    def _count_solutions(self, state, available_shapes, limit=2):
        empty_coords = []
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] == Geometry.EMPTY.value or state[r, c, 1] == Color.EMPTY.value:
                    empty_coords.append((r, c))

        if not empty_coords:
            return 1

        # MRV (Minimum Remaining Values)
        best_cell = None
        best_poss = None
        min_poss = 999

        for r, c in empty_coords:
            eg, ec = int(state[r, c, 0]), int(state[r, c, 1])
            poss = []
            for g, col in available_shapes:
                if eg != Geometry.EMPTY.value and eg != g:
                    continue
                if ec != Color.EMPTY.value and ec != col:
                    continue
                if self._is_safe(state, r, c, g, col):
                    poss.append((g, col))
            
            if not poss:
                return 0
            
            if len(poss) < min_poss:
                min_poss = len(poss)
                best_cell = (r, c)
                best_poss = poss
                if min_poss == 1:
                    break

        r, c = best_cell
        count = 0
        for g, col in best_poss:
            old_g, old_c = state[r, c]
            state[r, c] = [g, col]
            available_shapes.remove((g, col))

            count += self._count_solutions(state, available_shapes, limit)

            available_shapes.add((g, col))
            state[r, c] = [old_g, old_c]

            if count >= limit:
                return count

        return count

    def _solve(self, state, available_shapes):
        empty_coords = []
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] == Geometry.EMPTY.value or state[r, c, 1] == Color.EMPTY.value:
                    empty_coords.append((r, c))

        if not empty_coords:
            return True

        # MRV
        best_cell = None
        best_poss = None
        min_poss = 999

        for r, c in empty_coords:
            eg, ec = int(state[r, c, 0]), int(state[r, c, 1])
            poss = []
            for g, col in available_shapes:
                if eg != Geometry.EMPTY.value and eg != g:
                    continue
                if ec != Color.EMPTY.value and ec != col:
                    continue
                if self._is_safe(state, r, c, g, col):
                    poss.append((g, col))
            
            if not poss:
                return False
            
            if len(poss) < min_poss:
                min_poss = len(poss)
                best_cell = (r, c)
                best_poss = poss
                if min_poss == 1:
                    break

        r, c = best_cell
        random.shuffle(best_poss)
        for g, col in best_poss:
            old_g, old_c = state[r, c]
            state[r, c] = [g, col]
            available_shapes.remove((g, col))

            if self._solve(state, available_shapes):
                return True

            available_shapes.add((g, col))
            state[r, c] = [old_g, old_c]

        return False

    def _is_safe(self, state, r, c, g, col):
        for i in range(self.cols):
            if i != c:
                if state[r, i, 0] == g or state[r, i, 1] == col:
                    return False
        for i in range(self.rows):
            if i != r:
                if state[i, c, 0] == g or state[i, c, 1] == col:
                    return False
        return True

def print_grid(state):
    for r in range(state.shape[0]):
        row = []
        for c in range(state.shape[1]):
            g, col = state[r, c]
            if g == Geometry.EMPTY.value and col == Color.EMPTY.value:
                row.append(" . ")
            elif g == Geometry.EMPTY.value:
                row.append(f"(?,{col})")
            elif col == Color.EMPTY.value:
                row.append(f"({g},?)")
            else:
                row.append(f"({g},{col})")
        print(" ".join(row))
    print()


if __name__ == "__main__":
    # erlaubte Geometrien & Farben (4x4)
    geometries = np.array([Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON])
    colors = np.array([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])

    generator = SudokuGenerator(geometries, colors)

    level = 12  # maximal schwierig
    solved, puzzle = generator.generate(level)

    print(f"=== Level {level} – Startzustand ===")
    print_grid(puzzle)

    print(f"=== Level {level} – Lösung ===")
    print_grid(solved)

