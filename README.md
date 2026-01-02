# 🧩 Figure-Sudoku RL-Agent

<img src="./documentation/screenshot.png" width="500" alt="Figure-Sudoku Screenshot">

This project demonstrates the use of **Reinforcement Learning** to solve a complex Sudoku variant. Instead of numbers, this Sudoku uses geometric **shapes** and **colors**, increasing the logical requirements for the agent.

---

## 🎨 Game Concept

**Figure-Sudoku** is based on a 4x4 grid. Each cell must contain a unique combination of a shape and a color.

### Attributes:
*   **Geometries:** 🔵 Circle, 🟥 Square, ▲ Triangle, ⬢ Hexagon
*   **Colors:** ❤️ Red, 💚 Green, 💙 Blue, 💛 Yellow

### Rules:
1.  Each cell must contain a figure (shape + color).
2.  In each **row** and each **column**, each shape may only appear once.
3.  In each **row** and each **column**, each color may only appear once.
4.  Each combination (e.g., "Red Circle") may only exist once in the entire grid.
5.  **Partial specifications:** It is possible for cells to be pre-filled with only a shape (without color) or only a color (without shape). The agent must then logically complete the missing component.

---

## 🚀 AI Architecture

The agent uses state-of-the-art deep learning techniques to learn the game rules from scratch:

*   **Libraries:** Uses **Stable Baselines3 (v2.0+)** and **Gymnasium**, the current standard interface for Reinforcement Learning.
*   **Algorithm:** `MaskablePPO` (Proximal Policy Optimization). Thanks to **Action Masking**, the agent does not learn invalid moves, which massively accelerates training.
*   **CNN (Convolutional Neural Network) with Residual Blocks (ResNet):** Since Sudoku rules are based on spatial dependencies (rows/columns), the agent uses convolutional layers. ResNet blocks help learn deeper dependencies without loss of information.
*   **Observation Space:** A 3D tensor (10 channels) representing the positions of all shapes and colors in one-hot encoding (flattened to 160 inputs).
*   **Action Space:** A total of 256 discrete actions. Each action corresponds to the combination of a specific figure (16 possibilities) and a target cell (16 cells).
*   **Action Masking:** Since only a few of the 256 actions are compliant with the rules in any state, the project uses **Action Masking**. This prevents the agent from even considering invalid moves (e.g., duplicate color in a row). This dramatically reduces the search space and stabilizes training (see section [Action Masking (Detailed Explanation)](#-action-masking-detailed-explanation)).
*   **Curriculum Learning:** Training starts at Level 1 (almost solved) and automatically increases the difficulty up to Level 12 (many empty cells) once the agent reaches a defined success rate (adjustable via `REWARD_THRESHOLD`).
*   **Resumability:** Training automatically detects existing models. The starting level is primarily controlled via `START_LEVEL` in `config.py`. If this value is set to `None`, the level is automatically determined from the last log entry (`LOG_FILE_PATH`) (with fallback to Level 1).
*   **Puzzle Generator:** Puzzles are generated using a highly optimized backtracking algorithm (`sudoku_generator.py`). This uses the **HCDS metric** (Human-Centric Difficulty System) to specifically generate difficulty levels from 1 to 12 that reflect human perception of difficulty. It ensures that every puzzle has a unique solution. From Level 11 onwards, partial specifications (Partial Shapes) are also supported.

---

## 🧠 How the Agent Works

The solution process follows a classic RL cycle:

1.  **Observation:** The agent sees the current 4x4 grid as a one-hot vector.
2.  **Masking:** The environment calculates all rule-compliant moves based on the Sudoku rules.
3.  **Decision:** The neural network evaluates the valid actions and selects the most promising one.
4.  **Reward:** The agent receives a small reward for each correct move. Solving the entire puzzle gives a large bonus.
5.  **Learning:** Via PPO, the agent optimizes its strategy to maximize the cumulative reward.

---

## 🧩 Puzzle Generator & Difficulty (HCDS)

The new generator (`sudoku_generator.py`) is based on the **Human-Centric Difficulty System (HCDS)**. It has been specifically optimized to reflect human perception of difficulty across levels 1-12 and to enable fast puzzle generation (approx. 3s for Level 12) even at high difficulty levels while guaranteeing uniqueness.

### Difficulty Levels (1-12):
*   **Levels 1-10:** Difficulty scales linearly by removing cells until the target HCDS value is reached.
*   **Level 11:** Additionally contains **one partial specification** (only shape or only color).
*   **Level 12:** Contains **two partial specifications**.

### Performance Features:
*   **MRV Heuristic (Minimum Remaining Values):** Speeds up uniqueness testing through intelligent selection of the next cell in backtracking.
*   **In-Place Backtracking:** Minimizes memory allocations and CPU load.
*   **Incremental HCDS Calculation:** Efficient assessment of difficulty during the generation process.

---

## 🛡️ Action Masking (Detailed Explanation)

Action Masking is a crucial technique for the efficiency of this agent. Since the Action Space is very large with **256 actions**, but often **less than 5%** of the moves are legal in any game state, a standard RL agent would take an extremely long time to learn even the basic rules (e.g., "no red twice in a column") through pure *trial & error*.

### How it works:
Before the agent selects an action, the environment (`FigureSudokuEnv.action_masks()`) calculates a binary vector (the mask). For each of the 256 actions, it checks:

1.  **Cell Occupancy:** Is the target cell already occupied by another figure? (Or does the chosen figure match an existing partial specification?)
2.  **Figure Availability:** Has the combination of shape and color (e.g., "Blue Circle") already been placed elsewhere in the grid?
3.  **Sudoku Constraints (Row/Column):** Does the chosen shape or color already exist in the target row or target column?

### Why MaskablePPO?
In a standard PPO algorithm, the agent would also choose invalid actions, receive a negative reward, and then laboriously learn to avoid these actions.
**MaskablePPO**, on the other hand, uses the mask directly in the probability distribution of the policy:
*   Invalid actions receive a probability of **exactly zero**.
*   The agent "sees" only the legal options during decision-making.
*   **Advantage:** The neural network does not have to waste capacity learning the hard rules of the game, but can immediately concentrate on the **solution strategy**.

---

### 📂 Project Structure

```text
FigureSudoku/
├── 📄 config.py             # Central configuration (hyperparameters, levels, etc.)
├── 📄 train.py              # Main script to start AI training
├── 📄 figure_sudoku_env.py  # The Gymnasium environment (logic & rewards)
├── 📄 sudoku_generator.py   # Highly optimized generator with HCDS metric & uniqueness check
├── 📄 sudoku_game.py        # Graphical desktop interface (Tkinter)
├── 📄 streamlit_app.py      # Modern web application (Streamlit)
├── 📄 visualizer.py         # Live visualization during training
├── 📄 callbacks.py          # Logic for curriculum learning & model saving
├── 📄 shapes.py             # Definitions of shapes and colors (Enums)
├── 📁 documentation/        # Project documentation & videos
└── 📁 output/               # Saved models, logs, and checkpoints
```

---

## 🎬 Demo

Here you can see the RL agent in action as it solves a Figure-Sudoku step-by-step:

<div align="center">
  <video src="./documentation/solving_sudoku_game.mp4" width="600" controls autoplay loop muted>
    Your browser does not support the video tag.
  </video>
  <p><i>Agent solving a Figure-Sudoku (RL MaskablePPO)</i></p>
</div>

> **Note:** If the video above does not start automatically, you can view it directly here: [Open Demo Video](./documentation/solving_sudoku_game.mp4)

---

## ⚙️ Configuration (`config.py`)

The central settings of the project are made in `config.py`. Here is an overview of the most important parameters:

### 🧩 Generator (Puzzle Creation)
*   `START_LEVEL`: Determines the starting level for training. If a value (1-12) is specified, it is used fixed (manual override). If set to `None`, the level is automatically determined from the log file when resuming training (fallback: Level 1). [Range: `1` to `12` or `None`]
*   `MAX_LEVEL`: The target level (highest difficulty). [Range: `1` to `12`]

### ⚡ Training & Hyperparameters
*   `NUM_AGENTS`: Number of parallel training environments. [Range: `>= 1`]
*   `REWARD_THRESHOLD`: The required success rate (e.g., `0.90` for 90%) to advance to the next level. [Range: `0.0` to `1.0`]
*   `CHECK_FREQ`: Interval (in steps) at which the success rate is checked and models are cached. [Range: `>= 1`]
*   `TOTAL_TIMESTEPS`: Total duration of training (total number of steps across all agents). [Range: `>= 1`]
*   `MAX_TIMESTEPS`: Maximum number of steps per episode. Prevents infinite loops in unsolvable states.

### 🏆 Reward System

The reward system is designed to guide the agent towards an efficient and rule-compliant solution path. It consists of three main components:

1.  **`REWARD_SOLVED` (Current: `10.0`)**:
    *   **Purpose:** The "Holy Grail". This is the maximum reward the agent receives when the entire grid is filled in compliance with the rules.
    *   **Why this value?** It must be significantly higher than the sum of individual moves so that the agent prioritizes the overall goal (solving). Even at the highest difficulty level (Level 12), the sum of all valid individual move rewards is only about `2.45`, which means that `REWARD_SOLVED` (10.0) is still worth more than four times that. This ensures that the agent remains motivated to solve the puzzle completely even with complex puzzles.

2.  **`REWARD_VALID_MOVE_BASE` (Current: `0.1`)**:
    *   **Purpose:** Reward for each correct move.
    *   **Dynamic Scaling:** The actual reward is calculated as: `base * (1 + empty_cells / grid_size)`.
    *   **Why this logic?** Due to the scaling, the agent receives a higher reward for moves on an empty board (where there are many possibilities) than for moves on an almost full board. This motivates the agent to make "difficult" decisions correctly at an early stage. The base value of `0.1` is small enough to allow "reward shaping" without overshadowing the ultimate goal.

3.  **`REWARD_INVALID_MOVE` (Current: `-0.5`)**:
    *   **Purpose:** Punishment for illegal moves (although these are largely prevented by action masking).
    *   **Why this value?** The penalty is chosen to be moderately negative. Since the agent uses `MaskablePPO`, it rarely encounters invalid moves in the action space, but the penalty serves as additional security for the learning stability of the policy.

---

## 🖼️ Visualization (Training)

Progress during training can be visualized in two ways:

*   **Live-GUI:** If `config.RENDER_GUI = True` is set, the game state of the agents is displayed live in a window (`visualizer.py`).
*   **TensorBoard:** Detailed metrics (reward, success rate, training loss) are logged.

---

## 🛠 Setup & Installation

### Prerequisites:
*   Python 3.8+
*   Anaconda or venv (recommended)
*   CUDA-capable GPU (recommended for training, e.g., CUDA 11.8)

### Installing Dependencies:

1.  **PyTorch with CUDA support (example for CUDA 11.8):**
    ```bash
    pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    ```

2.  **Remaining Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏋️ Starting Training

To train the agent, simply run `train.py`. The configuration can be adjusted in `config.py` (e.g., `NUM_AGENTS` for parallelization).

```bash
python train.py | Tee-Object -FilePath output/SUDOKU/training.log
```

### Monitoring with TensorBoard:
While training is running, you can follow the progress (success rate, reward) live. The path is defined in `config.TENSORBOARD_TRAIN_LOG`:
```bash
# Example (default):
tensorboard --logdir output/SUDOKU/logs/train --port 6006
```
Then open `http://localhost:6006` in your browser.

---

## 🎮 Playing & Watching the Agent

Two interfaces are available to play the game yourself or watch the AI solving it.

### 🌐 Web Application (Streamlit) - Recommended
A modern, interactive web interface that runs in the browser.
```bash
streamlit run streamlit_app.py
```

### 🖥️ Desktop Application (Tkinter)
The classic version with drag & drop functionality.
```bash
python sudoku_game.py
```

### Instructions:
1.  Make sure a trained model is in the `output` folder (see `config.MODEL_PATH`).
2.  Select the difficulty level using the **"Level" slider**.
3.  Click on **"New Game"** (or generate a new puzzle).
4.  Click on **"Solve"** to watch the agent solve it, or play yourself!

---

## 📊 Training Visualization (Live)
If the parameter `RENDER_GUI = True` is set in `config.py`, the training opens a separate window for each agent (`visualizer.py`). This allows you to watch live as the AI tries out different strategies.

---

## 📄 License & Author

*   **Author:** Andreas Börzel
*   **GitHub:** [Figure-Sudoku](https://github.com/aboerzel/FigureSudoku)
*   **License:** [MIT License](LICENSE) (or see file header)

*Developed as an experimental field for reinforcement learning in complex constraint environments.*
