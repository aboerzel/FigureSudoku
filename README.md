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

---

## 🚀 Die KI-Architektur

Der Agent nutzt modernste Deep-Learning-Techniken, um die Spielregeln von Grund auf zu lernen:

*   **Algorithmus:** `MaskablePPO` (Proximal Policy Optimization). Dank **Action Masking** lernt der Agent keine ungültigen Züge, was das Training massiv beschleunigt.
*   **Neuronales Netz:** Ein **CNN (Convolutional Neural Network)** mit **Residual Blocks (ResNet)**. Dies erlaubt der KI, räumliche Zusammenhänge zwischen Reihen und Spalten wie ein menschliches Auge zu erfassen.
*   **Curriculum Learning:** Das Training startet bei Level 1 (fast gelöst) und steigert automatisch den Schwierigkeitsgrad bis Level 10 (viele leere Felder), sobald der Agent eine Erfolgsquote von 98% erreicht.
*   **Observation Space:** Ein 3D-Tensor (10 Kanäle), der One-Hot-kodiert die Positionen aller Formen und Farben repräsentiert.

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

## 🛠 Setup & Installation

### Voraussetzungen:
*   Python 3.8+
*   Anaconda oder venv (empfohlen)

### Installation der Abhängigkeiten:
```bash
pip install torch stable-baselines3 sb3-contrib gym==0.21.0 numpy
```

---

## 🏋️ Training starten

Um den Agenten zu trainieren, führe einfach die `train.py` aus. Die Konfiguration kann in der `config.py` angepasst werden (z.B. `NUM_AGENTS` für Parallelisierung).

```bash
python train.py
```

### Monitoring mit TensorBoard:
Während das Training läuft, kannst du den Fortschritt (Erfolgsquote, Reward) live verfolgen:
```bash
tensorboard --logdir output/SUDOKU/logs/train --port 6006
```
Öffne dann `http://localhost:6006` in deinem Browser.

---

## 🎮 Den Agenten beobachten (Test/Demo)

Wenn du sehen möchtest, wie die trainierte KI ein Rätsel löst, kannst du die GUI nutzen:

1.  Stelle sicher, dass ein trainiertes Modell im `output`-Ordner liegt.
2.  Starte das Spiel:
```bash
python sudoku_game.py --level 10
```
3.  Klicke auf **"New Game"** und dann auf **"Solve"**, um den Agenten beim Lösen zuzusehen.

---

## 📊 Visualisierung des Trainings
Wenn in der `config.py` der Parameter `RENDER_GUI = True` gesetzt ist, öffnet das Training für jeden Agenten ein eigenes Fenster. So kannst du live beobachten, wie die KI verschiedene Strategien ausprobiert.

---
*Entwickelt als Experimentierfeld für Reinforcement Learning in komplexen Constraint-Umgebungen.*
