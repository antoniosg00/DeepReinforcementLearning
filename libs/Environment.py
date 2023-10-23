import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

class StockEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, df, n_obs=10, transaction_rate=0.01, initial_worth=1000, render_mode=None):
        """
        Class constructor.

        Parameters:
        - df: DataFrame, the data set.
        - n_obs: int, number of observations in history.
        - transaction_cost: float, transaction cost as a percentage.
        - render_mode: str, render mode (optional).
        """
        assert isinstance(df, pd.DataFrame), "df must be a DataFrame"

        self.df = df
        self.n_examples = df.shape[0]
        self.n_features = df.shape[1]
        self.n_obs = n_obs

        self.transaction_rate = transaction_rate # Percentage commission on any transaction
        self.initial_worth = initial_worth # Initial net worth in each episode

        n_info = 3 # Informative scalars that we take as observation of the environment 

        # Definition of the observation and action space
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(0, 1, shape=(self.n_features, self.n_obs), dtype=np.float32),
                "other_info": spaces.Box(0, 1, shape=(n_info,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(n=21) # 0, 5, ..., 95, 100 % of net worth in the form of shares (similar to https://gym-trading-env.readthedocs.io/)

        # Verification of the rendering mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def _get_obs(self):
        # Obtaining observation
        history_window = np.transpose(self.df.values[self.current_step - self.n_obs + 1:self.current_step + 1, :]) # -> we transpose it so that it has a shape accepted by torch (channels, width)
        other_info = np.array([self.account_balance, self.shares_held, self.net_worth])
        return {"history": history_window, "other_info": other_info}
    
    def _get_info(self):
        # Obtaining the enclosed information
        info_dict = {
            "account_balance": self.account_balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth
        }
        return info_dict

    def _exe_action(self, action):
        # Execution of the action chosen by the agent
        # action (a scalar in [0,20]) -> 0, 5, ..., 95, 100 % of net worth in the form of shares
        percentage = action*0.05
        self.net_worth = self.account_balance + self.shares_held * self.df['Close'].iloc[self.current_step]
        self.new_shares_held = percentage * self.net_worth / self.df['Close'].iloc[self.current_step] # Allow fractional shares?
        self.account_balance = (1-percentage) * self.net_worth

        # We apply the fees for the transaction
        transaction_cost = (self.new_shares_held - self.shares_held) * self.df['Close'].iloc[self.current_step] * self.transaction_rate # 1% transaction rate
        self.net_worth -= transaction_cost
        self.account_balance -= transaction_cost
        self.shares_held = self.new_shares_held # We return the value of the auxiliary used for the transaction_cost calculation.

        # We add net worth in this step to history
        self.net_worth_historial.append(self.net_worth)

    def _get_reward(self):
        # Obtaining agent's reward (mix of ideas -> improvable)
        # Calculate change in total net worth
        short_worth_delta = self.net_worth_historial[-1] - self.net_worth_historial[-2]
        reward = short_worth_delta / self.net_worth_historial[-2] * 100 # Percentage of growth

        return reward
        
    def reset(self, seed=None):
        # Restart the environment and return the initial observation
        np.random.seed(seed)

        self.current_step = np.random.randint(low=self.n_obs-1, high=self.n_examples, size=None, dtype=int) # interval [low, high)
        
        self.net_worth_historial = []
        self.account_balance = self.initial_worth
        self.shares_held = 0
        self.net_worth = self.initial_worth
        self.net_worth_historial.append(self.net_worth)

        # We obtain a new random observation
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        # We take a step in the environment according to the action provided.
        self._exe_action(action)
        reward = self._get_reward()
        
        self.current_step += 1 # We add 1 to the step count for this episode.
        # If the last instance of the data tensor is reached, it returns to the first instance without resetting the self.current_step counter.
        if self.current_step >= len(self.df):
            self.current_step = self.n_obs-1 
        
        observation = self._get_obs()
        info = self._get_info()

        # We truncate the episode if a minimum return of 2% has not been achieved in one year.
        truncated = False
        if len(self.net_worth_historial) % 365 == 0: # Once a year
            if (self.net_worth_historial[-1] - self.net_worth_historial[-365])/self.net_worth_historial[-365] * 100 < 2.:
                truncated = True
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, False, truncated, info # (observation, reward, terminated, truncated, info)
