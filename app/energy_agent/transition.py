# energy_agent/transition.py
import numpy as np
from collections import namedtuple

# Transition includes next_state, env_id, per-step cost, and per-step action mask (active cells)
Transition = namedtuple('Transition', [
    'state', 'action', 'mask', 'reward', 'cost', 'next_state', 'done', 'log_prob', 'value', 'env_id'
])

class TrajectoryBuffer:
    """A buffer for collecting experiences for on-policy algorithms like PPO."""
    def __init__(self, capacity=None):
        self.memory = []
        self.capacity = capacity

    def add(self, transition: Transition):
        """Save a transition."""
        self.memory.append(transition)
        if self.capacity is not None and len(self.memory) > self.capacity:
            # keep last capacity transitions (FIFO)
            self.memory = self.memory[-self.capacity:]

    def get_all_and_clear(self):
        """Return lists (in insertion order) and clear the memory."""
        batch = self.memory
        self.memory = []

        if len(batch) == 0:
            return [], [], [], [], [], [], [], [], []

        states = [t.state for t in batch]
        actions = [t.action for t in batch]
        masks = [t.mask for t in batch]
        rewards = [t.reward for t in batch]
        costs = [t.cost for t in batch]
        next_states = [t.next_state for t in batch]
        dones = [t.done for t in batch]
        log_probs = [t.log_prob for t in batch]
        values = [t.value for t in batch]
        env_ids = [t.env_id for t in batch]

        return states, actions, masks, rewards, costs, next_states, dones, log_probs, values, env_ids

    def __len__(self):
        return len(self.memory)
