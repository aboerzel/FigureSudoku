import os

import numpy as np

import config
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results


class CurriculumCallback(BaseCallback):
    """
    Callback for increasing the difficulty level of the environment
    based on the mean reward.
    """
    def __init__(self, check_freq: int, reward_threshold: float, log_dir: str, verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.log_dir = log_dir
        self.current_level = config.LEVEL

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Hole Ergebnisse der letzten Episoden
            try:
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(y) >= 100: # Betrachte mindestens die letzten 100 Episoden
                    last_rewards = y[-100:]
                    # Eine Episode gilt als erfolgreich (DONE), wenn der Bonus von 1.0 erreicht wurde.
                    # Da jeder valide Zug 0.1 gibt, ist der Reward bei Erfolg immer >= 1.0.
                    success_rate = np.mean([1 if r >= 1.0 else 0 for r in last_rewards])
                    
                    if success_rate > self.reward_threshold and self.current_level < config.MAX_LEVEL:
                        self.current_level += 1
                        if self.verbose > 0:
                            print(f"Success rate {success_rate:.2f} > {self.reward_threshold}. Increasing difficulty level to: {self.current_level}", flush=True)
                        
                        # Level in allen Umgebungen aktualisieren
                        # Wir nutzen env_method um die Methode in den Subprozessen aufzurufen
                        self.training_env.env_method("reset_with_level", self.current_level)
                elif len(y) > 0 and self.verbose > 1:
                    print(f"Waiting for more episodes to evaluate curriculum (current: {len(y)}/100)", flush=True)
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
