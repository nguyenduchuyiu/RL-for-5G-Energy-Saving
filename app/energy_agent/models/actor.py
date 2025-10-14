# energy_agent/actor.py

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """
    Mạng Actor (Policy) cho DrQ-v2/SAC.
    Ánh xạ state tới một phân phối xác suất của action.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            # Ở chế độ đánh giá, lấy hành động có xác suất cao nhất
            z = mean
        else:
            # Ở chế độ huấn luyện, lấy mẫu (reparameterization trick)
            z = normal.rsample()
        
        # Áp dụng tanh để đưa action về khoảng [-1, 1]
        action = torch.tanh(z)
        
        # Tính log_prob theo công thức của SAC
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob