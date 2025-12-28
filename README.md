# 🧩 FigureSudoku RL-Agent

Dieses Projekt demonstriert den Einsatz von **Reinforcement Learning** (Bestärkendes Lernen), um eine komplexe Sudoku-Variante zu lösen. Anstelle von Zahlen verwendet dieses Sudoku geometrische **Formen** und **Farben**, was die logischen Anforderungen an den Agenten erhöht.

---

## 🎨 Das Spielkonzept

Das **FigureSudoku** basiert auf einem 4x4-Gitter. Jedes Feld muss eine eindeutige Kombination aus einer Form und einer Farbe enthalten.

### Die Attribute:
*   **Geometrien:** 🔵 Kreis, 🟥 Quadrat, ▲ Dreieck, ⬢ Hexagon
*   **Farben:** ❤️ Rot, 💚 Grün, 💙 Blau, 💛 Gelb

### Die Regeln:
1.  Jedes Feld muss eine Figur (Form + Farbe) enthalten.
2.  In jeder **Reihe** und jeder **Spalte** darf jede Form nur einmal vorkommen.
3.  In jeder **Reihe** und jeder **Spalte** darf jede Farbe nur einmal vorkommen.
4.  Jede Kombination (z.B. "Roter Kreis") darf im gesamten Gitter nur einmal existieren.
5.  **Teilvorgaben:** Es ist möglich, dass Felder nur mit einer Form (ohne Farbe) oder nur mit einer Farbe (ohne Form) vorbelegt sind. Der Agent muss dann die jeweils fehlende Komponente logisch korrekt ergänzen.

---

## 🚀 Die KI-Architektur

Der Agent nutzt modernste Deep-Learning-Techniken, um die Spielregeln von Grund auf zu lernen:

*   **Algorithmus:** `MaskablePPO` (Proximal Policy Optimization). Dank **Action Masking** lernt der Agent keine ungültigen Züge, was das Training massiv beschleunigt.
*   **CNN (Convolutional Neural Network) mit Residual Blocks (ResNet):** Da Sudoku-Regeln auf räumlichen Abhängigkeiten (Zeilen/Spalten) basieren, nutzt der Agent Faltungsschichten. ResNet-Blöcke helfen dabei, auch tieferliegende Abhängigkeiten ohne Informationsverlust zu lernen.
*   **Observation Space:** Ein 3D-Tensor (10 Kanäle), der One-Hot-kodiert die Positionen aller Formen und Farben repräsentiert (flattened auf 160 Eingänge).
*   **Action Space:** Insgesamt 256 diskrete Aktionen. Jede Aktion entspricht der Kombination aus einer bestimmten Figur (16 Möglichkeiten) und einem Zielfeld (16 Felder).
*   **Action Masking:** Da in jedem Zustand nur wenige der 256 Aktionen regelkonform sind, werden ungültige Züge (z.B. doppelte Farbe in einer Reihe) maskiert. Der Agent wählt nur aus den verbleibenden validen Optionen.
*   **Curriculum Learning:** Das Training startet bei Level 1 (fast gelöst) und steigert automatisch den Schwierigkeitsgrad bis Level 12 (viele leere Felder), sobald der Agent eine definierte Erfolgsquote (einstellbar über `REWARD_THRESHOLD`) erreicht.
*   **Fortsetzbarkeit:** Das Training erkennt automatisch vorhandene Modelle und setzt das Curriculum-Level basierend auf dem letzten Log-Eintrag in der in `config.py` definierten Log-Datei (`LOG_FILE_PATH`) fort.
*   **Backtracking-Generator:** Die Rätsel werden mithilfe eines Backtracking-Algorithmus generiert, der sicherstellt, dass die Aufgaben lösbar sind und optional eine eindeutige Lösung besitzen.

---

## 🧠 Funktionsweise des Agenten

Der Lösungsprozess folgt einem klassischen RL-Zyklus:

1.  **Beobachtung:** Der Agent sieht das aktuelle 4x4-Gitter als One-Hot-Vektor.
2.  **Maskierung:** Die Umgebung berechnet alle regelkonformen Züge basierend auf den Sudoku-Regeln.
3.  **Entscheidung:** Das neuronale Netz bewertet die validen Aktionen und wählt die Erfolgversprechendste aus.
4.  **Belohnung:** Für jeden korrekten Zug erhält der Agent einen kleinen Reward. Das Lösen des gesamten Rätsels gibt einen großen Bonus.
5.  **Lernen:** Über PPO optimiert der Agent seine Strategie, um die kumulierte Belohnung zu maximieren.

---

## 📂 Projektstruktur

```text
FigureSudoku/
├── 📄 config.py             # Zentrale Konfiguration (Hyperparameter, Level, etc.)
├── 📄 train.py              # Hauptskript zum Starten des KI-Trainings
├── 📄 figure_sudoko_env.py  # Die Gymnasium-Umgebung (Logik & Rewards)
├── 📄 sudoku_generator.py   # Backtracking-Algorithmus zur Rätsel-Generierung (mit optionaler Eindeutigkeitsprüfung)
├── 📄 sudoku_game.py        # Grafische Oberfläche zum Spielen & Evaluieren
├── 📄 visualizer.py         # Live-Visualisierung während des Trainings
├── 📄 callbacks.py          # Logik für Curriculum Learning & Modell-Speicherung
├── 📄 shapes.py             # Definitionen der Formen und Farben (Enums)
└── 📁 output/               # Gespeicherte Modelle, Logs und Checkpoints
```

---

## ⚙️ Konfiguration (`config.py`)

Die zentralen Einstellungen des Projekts werden in der `config.py` vorgenommen. Hier eine Übersicht der wichtigsten Parameter:

### 🧩 Generator (Rätsel-Erstellung)
*   `START_LEVEL`: Level, bei dem das Training beginnt (Anzahl leere Felder bzw. Felder ohne vollständige Figur). [Bereich: `1` bis `16`]
*   `MAX_LEVEL`: Das Ziel-Level (höchste Schwierigkeit). [Bereich: `1` bis `16`, aktuell `12`]
*   `UNIQUE`: Stellt sicher, dass jedes generierte Rätsel nur genau eine gültige Lösung hat. [Werte: `True`, `False`]
*   `PARTIAL_PROB`: Wahrscheinlichkeit (`0.0` bis `1.0`), dass in einem Rätsel Teilvorgaben (nur Farbe oder nur Form) generiert werden. Erhöht die Komplexität, da der Agent fehlende Attribute ergänzen muss.
*   `PARTIAL_MODE`: Bestimmt die Anzahl der Teilvorgaben pro Rätsel:
    *   `0`: Deaktiviert.
    *   `1`: Genau 2 Felder werden als Teilvorgaben markiert.
    *   `2`: Zufällig 1 bis 2 Felder werden als Teilvorgaben markiert.

### ⚡ Training & Hyperparameter
*   `NUM_AGENTS`: Anzahl der parallelen Trainings-Umgebungen. [Bereich: `>= 1`]
*   `REWARD_THRESHOLD`: Die benötigte Erfolgsquote (z.B. `0.90` für 90%), um in das nächste Level aufzusteigen. [Bereich: `0.0` bis `1.0`]
*   `CHECK_FREQ`: Intervall (in Schritten), in dem die Erfolgsquote geprüft und Modelle zwischengespeichert werden. [Bereich: `>= 1`]
*   `TOTAL_TIMESTEPS`: Die Gesamtdauer des Trainings (Gesamtzahl der Schritte über alle Agenten). [Bereich: `>= 1`]
*   `MAX_TIMESTEPS`: Maximale Anzahl an Schritten pro Episode. Verhindert Endlosschleifen bei unlösbaren Zuständen.

### 🏆 Belohnungssystem (Rewards)
*   `REWARD_SOLVED`: Belohnung für ein komplett gelöstes Sudoku. [Typ: `Float`, empfohlen: `> 0`]
*   `REWARD_VALID_MOVE_BASE`: Basisbelohnung für einen korrekten Setzvorgang. Die tatsächliche Belohnung ist dynamisch und wird mit der Anzahl der leeren Felder skaliert: `base * (1 + empty_fields / state_size)`. Dies fördert gezielte Züge auf einem leeren Board.
*   `REWARD_INVALID_MOVE`: Strafe für den Versuch, eine Figur entgegen der Regeln zu platzieren. [Typ: `Float`, empfohlen: `< 0`]

### 🖼️ Visualisierung
*   `RENDER_GUI`: Aktiviert die Live-Anzeige der Agenten während des Trainings. [Werte: `True`, `False`]

---

## 🛠 Setup & Installation

### Voraussetzungen:
*   Python 3.8+
*   Anaconda oder venv (empfohlen)

### Installation der Abhängigkeiten:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training starten

Um den Agenten zu trainieren, führe einfach die `train.py` aus. Die Konfiguration kann in der `config.py` angepasst werden (z.B. `NUM_AGENTS` für Parallelisierung).

```bash
python train.py | Tee-Object -FilePath output/SUDOKU/training.log
```

### Monitoring mit TensorBoard:
Während das Training läuft, kannst du den Fortschritt (Erfolgsquote, Reward) live verfolgen. Der Pfad ist in `config.TENSORBOARD_TRAIN_LOG` definiert:
```bash
# Beispiel (standardmäßig):
tensorboard --logdir output/SUDOKU/logs/train --port 6006
```
Öffne dann `http://localhost:6006` in deinem Browser.

---

## 🎮 Den Agenten beobachten (Test/Demo)

Wenn du sehen möchtest, wie die trainierte KI ein Rätsel löst, kannst du die GUI nutzen:

1.  Stelle sicher, dass ein trainiertes Modell im `output`-Ordner liegt (siehe `config.MODEL_PATH`).
2.  Starte das Spiel:
```bash
python sudoku_game.py
```
3.  Wähle den Schwierigkeitsgrad über den **"Level"-Slider** aus.
4.  Klicke auf **"New Game"** und dann auf **"Solve"**, um den Agenten beim Lösen zuzusehen.

---

## 📊 Visualisierung des Trainings
Wenn in der `config.py` der Parameter `RENDER_GUI = True` gesetzt ist, öffnet das Training für jeden Agenten ein eigenes Fenster. So kannst du live beobachten, wie die KI verschiedene Strategien ausprobiert.

---
*Entwickelt als Experimentierfeld für Reinforcement Learning in komplexen Constraint-Umgebungen.*
