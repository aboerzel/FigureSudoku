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

    def generate(self, initial_items=4, unique=False, partial_prob=0.0, partial_mode=0):
        """
        Generiert ein gültiges, gelöstes Figure-Sudoku und entfernt dann 
        Elemente, um den Startzustand für den Agenten zu erstellen.
        """
        state = np.full((self.rows, self.cols, 2), Geometry.EMPTY.value, dtype=int)
        available_shapes = set(self.all_shapes)
        
        # Versuche das Gitter vollständig zu füllen (Backtracking + MRV)
        if not self._solve(state, available_shapes):
            # Fallback (sollte bei 4x4 theoretisch nie nötig sein)
            return self.generate(initial_items, unique, partial_prob, partial_mode)
            
        solved_state = state.copy()
        
        if not unique:
            # Klassisches Verhalten: Zufälliges Entfernen ohne Eindeutigkeitsprüfung
            init_state = solved_state.copy().reshape(-1, 2)
            num_to_keep = max(0, min(initial_items, self.state_size))
            keep_indices = np.random.choice(range(self.state_size), num_to_keep, replace=False)
            mask = np.ones(self.state_size, dtype=bool)
            mask[keep_indices] = False
            init_state[mask] = [Geometry.EMPTY.value, Color.EMPTY.value]
            init_state = init_state.reshape(self.rows, self.cols, 2)
        else:
            # Eindeutigkeits-Modus: Versuche ein Rätsel mit genau einer Lösung zu finden
            init_state = solved_state.copy()
            indices = list(range(self.state_size))
            random.shuffle(indices)
            
            num_removed = 0
            max_to_remove = self.state_size - initial_items
            
            for idx in indices:
                if num_removed >= max_to_remove:
                    break
                    
                r, c = idx // self.cols, idx % self.cols
                old_val = init_state[r, c].copy()
                
                # Provisorisch entfernen
                init_state[r, c] = [Geometry.EMPTY.value, Color.EMPTY.value]
                
                # Prüfen, ob noch genau eine Lösung existiert
                # Wir müssen die aktuell verfügbaren Figuren berechnen
                current_available = set(self.all_shapes)
                for row in range(self.rows):
                    for col in range(self.cols):
                        if init_state[row, col, 0] != Geometry.EMPTY.value:
                            shape = (int(init_state[row, col, 0]), int(init_state[row, col, 1]))
                            if shape in current_available:
                                current_available.remove(shape)
                
                if self.count_solutions(init_state.copy(), current_available, limit=2) == 1:
                    num_removed += 1
                else:
                    # Nicht eindeutig -> Figur wieder einsetzen
                    init_state[r, c] = old_val
        
        # Teilbelegungen hinzufügen (bereits gesetzte Felder reduzieren)
        if partial_prob > 0:
            if random.random() < partial_prob:
                num_to_modify = partial_mode
                
                if num_to_modify > 0:
                    # Finde Felder, die aktuell vollständig belegt sind
                    filled_indices = []
                    for r in range(self.rows):
                        for c in range(self.cols):
                            if init_state[r, c, 0] != Geometry.EMPTY.value and init_state[r, c, 1] != Color.EMPTY.value:
                                filled_indices.append((r, c))
                    
                    if filled_indices:
                        # Wähle zufällig Felder aus den bereits gesetzten aus
                        num_actual = min(num_to_modify, len(filled_indices))
                        chosen_cells = random.sample(filled_indices, num_actual)
                            
                        for r, c in chosen_cells:
                            # Zufällig entweder Form oder Farbe verwerfen
                            if random.choice([True, False]):
                                # Farbe verwerfen -> Nur Form bleibt
                                init_state[r, c, 1] = Color.EMPTY.value
                            else:
                                # Form verwerfen -> Nur Farbe bleibt
                                init_state[r, c, 0] = Geometry.EMPTY.value
        
        return solved_state, init_state

    def count_solutions(self, state, available_shapes, limit=2):
        """
        Zählt die Anzahl der Lösungen für einen gegebenen Zustand.
        Bricht ab, wenn das Limit erreicht ist.
        """
        # Finde alle leeren oder unvollständigen Felder und berechne deren Möglichkeiten
        empty_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                # Feld ist leer, wenn Geometrie ODER Farbe fehlt
                if state[r, c, 0] == Geometry.EMPTY.value or state[r, c, 1] == Color.EMPTY.value:
                    existing_g = int(state[r, c, 0])
                    existing_c = int(state[r, c, 1])
                    
                    possibilities = []
                    for s in available_shapes:
                        # Respektiere bereits gesetzte Attribute (Teilbelegungen)
                        if existing_g != Geometry.EMPTY.value and existing_g != s[0]:
                            continue
                        if existing_c != Color.EMPTY.value and existing_c != s[1]:
                            continue
                            
                        if self._is_safe(state, r, c, s[0], s[1]):
                            possibilities.append(s)
                    
                    empty_cells.append(((r, c), possibilities))
        
        # Wenn kein leeres Feld mehr da ist, haben wir eine Lösung gefunden
        if not empty_cells:
            return 1
            
        # MRV Heuristik
        empty_cells.sort(key=lambda x: len(x[1]))
        (r, c), possibilities = empty_cells[0]
        
        count = 0
        for g_val, c_val in possibilities:
            old_val = state[r, c].copy()
            state[r, c] = [g_val, c_val]
            available_shapes.remove((g_val, c_val))
            
            count += self.count_solutions(state, available_shapes, limit)
            
            # Backtrack
            available_shapes.add((g_val, c_val))
            state[r, c] = old_val
            
            if count >= limit:
                return count
                
        return count

    def _solve(self, state, available_shapes):
        """
        Backtracking-Algorithmus mit Minimum Remaining Values (MRV) Heuristik.
        """
        # Finde alle leeren oder unvollständigen Felder und berechne deren Möglichkeiten
        empty_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] == Geometry.EMPTY.value or state[r, c, 1] == Color.EMPTY.value:
                    existing_g = int(state[r, c, 0])
                    existing_c = int(state[r, c, 1])
                    
                    possibilities = []
                    for s in available_shapes:
                        # Respektiere bereits gesetzte Attribute (Teilbelegungen)
                        if existing_g != Geometry.EMPTY.value and existing_g != s[0]:
                            continue
                        if existing_c != Color.EMPTY.value and existing_c != s[1]:
                            continue
                            
                        if self._is_safe(state, r, c, s[0], s[1]):
                            possibilities.append(s)
                            
                    empty_cells.append(((r, c), possibilities))
        
        # Wenn kein leeres Feld mehr da ist, haben wir eine Lösung
        if not empty_cells:
            return True
            
        # MRV: Wähle das Feld mit den wenigsten Möglichkeiten
        empty_cells.sort(key=lambda x: len(x[1]))
        (r, c), possibilities = empty_cells[0]
        
        # Wenn ein Feld keine Möglichkeiten hat, ist dieser Pfad eine Sackgasse
        if not possibilities:
            return False
            
        # Probiere die Möglichkeiten in zufälliger Reihenfolge
        random.shuffle(possibilities)
        for g_val, c_val in possibilities:
            old_val = state[r, c].copy()
            state[r, c] = [g_val, c_val]
            available_shapes.remove((g_val, c_val))
            
            if self._solve(state, available_shapes):
                return True
                
            # Backtrack: Zustand zurücksetzen
            available_shapes.add((g_val, c_val))
            state[r, c] = old_val
            
        return False

    def _is_safe(self, state, r, c, g, col):
        """Prüft, ob das Platzieren einer Figur gegen Sudoku-Regeln verstößt."""
        # Zeilenprüfung (aktuelle Spalte ausschließen)
        for i in range(self.cols):
            if i != c:
                if state[r, i, 0] == g or state[r, i, 1] == col:
                    return False
        # Spaltenprüfung (aktuelle Zeile ausschließen)
        for i in range(self.rows):
            if i != r:
                if state[i, c, 0] == g or state[i, c, 1] == col:
                    return False
        return True
