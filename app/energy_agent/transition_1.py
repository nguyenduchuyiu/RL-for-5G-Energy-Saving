# energy_agent/transition.py
import numpy as np
from collections import namedtuple

# Transition with env_id to track which environment it came from
Transition = namedtuple('Transition', [
    'state', 'action', 'reward', 'done', 'log_prob', 'value', 'env_id'
])

class TrajectoryBuffer:
    """A buffer for collecting experiences for on-policy algorithms like PPO."""
    def __init__(self):
        self.memory = []

    def add(self, transition: Transition):
        """Saves a transition."""
        self.memory.append(transition)

    def get_all_and_clear(self):
        """Returns all transitions and clears the memory."""
        # Chuyển đổi list các transition thành các batch dữ liệu
        batch = self.memory
        self.memory = [] # Xóa buffer ngay sau khi lấy dữ liệu
        
        # Unpack the batch
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        dones = np.array([t.done for t in batch])
        log_probs = np.array([t.log_prob for t in batch])
        values = np.array([t.value for t in batch])
        env_ids = np.array([t.env_id for t in batch])
        
        return states, actions, rewards, dones, log_probs, values, env_ids

    def __len__(self):
        return len(self.memory)