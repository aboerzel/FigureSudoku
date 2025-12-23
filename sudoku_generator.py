import itertools
import random
import numpy as np

from shapes import Geometry, Color


class SudokuGenerator:
    def __init__(self, geometries, colors):
        # Konvertiere Enums/Objekte zu Integers für konsistente Vergleiche
        self.geometries = [int(g) for g in geometries]
        self.colors = [int(c) for c in colors]
        self.rows = len(self.geometries)
        self.cols = len(self.colors)
        self.state_size = self.rows * self.cols
        # Alle 16 möglichen Kombinationen aus Geometrie und Farbe
        self.all_shapes = list(itertools.product(self.geometries, self.colors))

    def generate(self, initial_items=4):
        """
        Generiert ein gültiges, gelöstes Figure-Sudoku und entfernt dann 
        Elemente, um den Startzustand für den Agenten zu erstellen.
        """
        state = np.full((self.rows, self.cols, 2), Geometry.EMPTY.value, dtype=int)
        available_shapes = set(self.all_shapes)
        
        # Versuche das Gitter vollständig zu füllen (Backtracking + MRV)
        if not self._solve(state, available_shapes):
            # Fallback (sollte bei 4x4 theoretisch nie nötig sein)
            return self.generate(initial_items)
            
        solved_state = state.copy()
        
        # Erzeuge den initialen Zustand durch Entfernen von Elementen
        init_state = solved_state.copy().reshape(-1, 2)
        
        # Bestimme, wie viele Elemente übrig bleiben sollen
        num_to_keep = max(0, min(initial_items, self.state_size))
        keep_indices = np.random.choice(range(self.state_size), num_to_keep, replace=False)
        
        # Maske für die zu leerenden Felder
        mask = np.ones(self.state_size, dtype=bool)
        mask[keep_indices] = False
        
        # Felder auf EMPTY setzen
        init_state[mask] = [Geometry.EMPTY.value, Color.EMPTY.value]
        init_state = init_state.reshape(self.rows, self.cols, 2)
        
        return solved_state, init_state

    def _solve(self, state, available_shapes):
        """
        Backtracking-Algorithmus mit Minimum Remaining Values (MRV) Heuristik.
        """
        # Finde alle leeren Felder und berechne deren Möglichkeiten
        empty_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] == Geometry.EMPTY.value:
                    possibilities = [
                        s for s in available_shapes 
                        if self._is_safe(state, r, c, s[0], s[1])
                    ]
                    empty_cells.append(((r, c), possibilities))
        
        # Wenn kein leeres Feld mehr da ist, haben wir eine Lösung
        if not empty_cells:
            return True
            
        # MRV: Wähle das Feld mit den wenigsten Möglichkeiten (beschleunigt die Suche)
        empty_cells.sort(key=lambda x: len(x[1]))
        (r, c), possibilities = empty_cells[0]
        
        # Wenn ein Feld keine Möglichkeiten hat, ist dieser Pfad eine Sackgasse
        if not possibilities:
            return False
            
        # Probiere die Möglichkeiten in zufälliger Reihenfolge
        random.shuffle(possibilities)
        for g, col in possibilities:
            state[r, c] = [g, col]
            available_shapes.remove((g, col))
            
            if self._solve(state, available_shapes):
                return True
                
            # Backtrack: Zustand zurücksetzen
            available_shapes.add((g, col))
            state[r, c] = [Geometry.EMPTY.value, Color.EMPTY.value]
            
        return False

    def _is_safe(self, state, r, c, g, col):
        """Prüft, ob das Platzieren einer Figur gegen Sudoku-Regeln verstößt."""
        # Einfache Schleifen sind bei 4x4 oft schneller als NumPy
        for i in range(self.cols):
            if state[r, i, 0] == g or state[r, i, 1] == col:
                return False
        for i in range(self.rows):
            if state[i, c, 0] == g or state[i, c, 1] == col:
                return False
        return True
