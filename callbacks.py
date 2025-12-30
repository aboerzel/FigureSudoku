import os

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results


class CurriculumCallback(BaseCallback):
    """
    Callback for increasing the difficulty level of the environment
    based on the success rate.
    """
    def __init__(self, check_freq: int, reward_threshold: float, log_dir: str, reward_solved: float, start_level: int = 1, max_level: int = 16, 
                 verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.log_dir = log_dir
        self.reward_solved = reward_solved
        self.current_level = start_level
        self.max_level = max_level
        self.episodes_at_start_of_level = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Hole Ergebnisse der letzten Episoden
            try:
                results = load_results(self.log_dir)
                if len(results) == 0:
                    return True
                
                # 'r' ist die Spalte für Belohnungen in Monitor-Dateien
                y = results['r'].values
                num_episodes = len(y)
                episodes_at_current_level = num_episodes - self.episodes_at_start_of_level
                
                if episodes_at_current_level >= 100: # Mindestens 100 Episoden auf dem aktuellen Level
                    last_rewards = y[-100:]
                    
                    # Schwellenwert für Erfolg (Reward >= self.reward_solved bedeutet Episode abgeschlossen)
                    # Wir nutzen eine kleine Toleranz für Float-Vergleiche
                    success_rate = np.mean(last_rewards >= (self.reward_solved - 1e-3))
                    
                    if self.verbose > 0:
                        print(f"Curriculum Check: Level {self.current_level} - Success Rate: {success_rate:.2f} (Window: 100, Episodes at Level: {episodes_at_current_level})", flush=True)

                    if success_rate > self.reward_threshold and self.current_level < self.max_level:
                        self.current_level += 1
                        self.episodes_at_start_of_level = num_episodes
                        if self.verbose > 0:
                            print(f"Success rate {success_rate:.2f} > {self.reward_threshold}. Increasing difficulty level to: {self.current_level}", flush=True)
                        
                        # Level in allen Umgebungen aktualisieren
                        # reset_with_level setzt das neue Level; es wird beim nächsten automatischen Reset wirksam.
                        self.training_env.env_method("reset_with_level",
                                                     level=self.current_level)
                elif num_episodes > 0 and self.verbose > 1:
                    print(f"Waiting for more episodes at level {self.current_level} (current: {episodes_at_current_level}/100)", flush=True)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Curriculum update failed: {e}", flush=True)
        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, model_name: str, checkpoint_name: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.model_save_path = os.path.join(log_dir, model_name)
        self.checkpoint_save_path = os.path.join(log_dir, checkpoint_name)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            try:
                self.model.save(self.checkpoint_save_path)
            except:
                print(f'Saving model checkpoint {self.checkpoint_save_path} failed.')

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}", flush=True)
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}", flush=True)

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.model_save_path}", flush=True)
                    self.model.save(self.model_save_path)

        return True
