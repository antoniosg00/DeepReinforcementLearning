from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from typing import Callable

import numpy as np

# Importing classes from my module
# from libs.CustomPPO_CNN import CustomCombinedExtractor, CustomActorCriticPolicy
from libs.CustomPPO_LSTM import CustomCombinedExtractor, CustomActorCriticPolicy

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule


def exponential_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Learning rate scheme with exponential decay. I wanted it to depend only on the initial and final values (much easier to understand).

    params:
        initial_value: initial learning rate.
        final_value: Final learning rate.
    return: 
        Scheme that calculates the current learning rate based on the remaining progress.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (start) to 0 (end of training).

        param:
            progress_remaining: remaining progress.
        return:
            Current learning rate.
        """    
        return final_value * np.exp(progress_remaining * np.log(initial_value/final_value))

    return func

class StockTradingAgent:
    def __init__(self, env, val_env, initial_lr=5e-4, final_value=1e-6):
        # Configuration of PPO model policy parameters
        policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor
        )

        # Decay of the learning rate
        self.lr_schedule = exponential_schedule(initial_value=initial_lr, final_value=final_value)

        # Creation of the DummyVecEnv-wrapped environment for the model and the evaluation environment
        self.env = DummyVecEnv([lambda: env])
        self.val_env = DummyVecEnv([lambda: val_env])

        # PPO model initialization with custom policy and other parameters
        self.model = PPO(CustomActorCriticPolicy, policy_kwargs=policy_kwargs, env=self.env, verbose=0, tensorboard_log='logs_board\\', learning_rate=self.lr_schedule)

    def train(self, total_timesteps=50000, trial=0):
        assert isinstance(trial, int), f"Trial expected: {int}, but received {type(trial)}" 
        # Setting up a callback to save the model periodically during training
        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps // 100, save_path='logs\\',
                                                 name_prefix='model_checkpoint')
        
        eval_callback = EvalCallback(eval_env=self.val_env, best_model_save_path='logs\\', log_path="logs\\", 
                                     eval_freq=500, deterministic=True, render=False)
        
        # Training and saving of the PPO model for the specified number of time steps
        self.model.learn(total_timesteps=total_timesteps, callback=None, progress_bar=True) # To use callbacks, add callback=[eval_callback, checkpoint_callback]
        self.model.save("logs_saves\\ppo_stock_trading"+str(trial))

    def load_model(self, model_path): # path = "ppo_stock_trading"
        # Load the model from the specified file
        self.model = PPO.load(model_path)

    def evaluate(self, steps=1000):
        # Reset the environment for a new evaluation
        obs = self.val_env.reset()
        episode_reward = 0
        episode_infos = []
        actions = []

        # Simulation of model execution in the environment for a specified number of steps
        for i in range(steps): # We see the result of 1000 days of trading
            action, _ = self.model.predict(obs)
            actions.append(action)
            obs, rewards, dones, info = self.val_env.step(action)
            # We add the pass reward to the episode total
            episode_reward += rewards
            episode_infos.append(info)
            
            #We check if an episode has been completed (done).
            if dones:
                print(f"Episode finished after {i+1} steps with total reward: {episode_reward}")
                break
            
        return episode_infos, actions