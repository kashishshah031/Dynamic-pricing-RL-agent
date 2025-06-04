import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        low = np.array([0.0, 0.0])
        high = np.array([1.0, 1.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.price_levels = [5, 8, 10, 12, 15]
        
    def reset(self):
        self.current_demand_level = np.random.uniform(0.4, 1.0)
        self.current_competition_price = np.random.uniform(5, 15)
        self.current_step = 0  # ADD THIS
        obs = np.array([self.current_demand_level, self.current_competition_price], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        chosen_price = self.price_levels[action]
        #demand = self.current_demand_level - 0.1 * (chosen_price - self.current_competition_price)
        demand = self.current_demand_level * np.exp(-0.15 * (chosen_price - self.current_competition_price))
        demand = np.clip(demand, 0, 1)

        #demand = np.clip(demand, 0, 1)
        sales_volume = demand * 100
        revenue = chosen_price * sales_volume
        #reward = revenue
        reward = revenue - 0.1 * abs(chosen_price - self.current_competition_price)

        
        # Simulate next demand and competition price
        self.current_demand_level = np.random.uniform(0.6, 1.0)
        self.current_competition_price = np.random.uniform(5, 15)
        
        obs = np.array([self.current_demand_level, self.current_competition_price], dtype=np.float32)
        
        self.current_step += 1
        done = self.current_step >= 10  # 10-step episode
        
        return obs, reward, done, False, {}

