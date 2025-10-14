# energy_agent/critic.py

import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Mạng Critic (Q-function) cho DrQ-v2/SAC.
    Ước tính giá trị Q của cặp (state, action).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Ghép state và action làm đầu vào
        x = torch.cat([state, action], dim=1)
        return self.net(x)