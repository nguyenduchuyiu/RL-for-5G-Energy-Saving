"""Gym wrapper for FiveGEnvironment to enable SubprocVecEnv"""

try:
    import gymnasium as gym
    USE_GYMNASIUM = True
except ImportError:
    import gym
    USE_GYMNASIUM = False

import numpy as np
from typing import Optional, Dict, Tuple, Any

from .environment import FiveGEnvironment


class FiveGGymWrapper(gym.Env):
    """
    Gym wrapper for FiveGEnvironment
    Enables compatibility with stable-baselines3 and SubprocVecEnv
    """
    
    def __init__(self, scenario: str = 'indoor_hotspot', seed: int = 42):
        super().__init__()
        
        self.env = FiveGEnvironment(scenario=scenario, seed=seed)
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.env.action_dim,),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.state_dim,),
            dtype=np.float32
        )
        
        self.scenario = scenario
        self.seed_value = seed
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        if seed is not None:
            self.seed_value = seed
        
        obs, info = self.env.reset(seed=self.seed_value, options=options)
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        return self.env.step(action)
    
    def render(self):
        """Render (not implemented for 5G environment)"""
        pass
    
    def close(self):
        """Close environment"""
        pass
    
    def seed(self, seed: int = None):
        """Set random seed"""
        if seed is not None:
            self.seed_value = seed
        return [self.seed_value]


def make_env(scenario: str, seed: int):
    """
    Utility function to create environment instance
    Required for SubprocVecEnv
    """
    def _init():
        env = FiveGGymWrapper(scenario=scenario, seed=seed)
        return env
    return _init

