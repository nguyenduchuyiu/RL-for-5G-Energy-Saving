import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    """
    Critic network with GRU for handling sequential states.
    """
    
    def __init__(self, state_dim, hidden_dim=256, 
                 gru_hidden_dim=128, gru_num_layers=1, activation='relu'):
        """
        Initialize Critic network with GRU.
        
        Args:
            state_dim (int): Dimension of a single state input at one timestep.
            hidden_dim (int): Hidden layer dimension for the MLP part.
            gru_hidden_dim (int): Hidden dimension for the GRU layer.
            gru_num_layers (int): Number of GRU layers to stack.
            activation (str): Activation function type.
        """
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # --- THAY ĐỔI 1: Thêm lớp GRU ---
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True
        )
        
        # --- THAY ĐỔI 2: Lớp MLP đầu tiên nhận đầu vào từ GRU ---
        self.fc1 = nn.Linear(gru_hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Activation function không đổi
        if activation == 'relu':
            self.activation_fn = F.relu
        # ... (các activation khác)

        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # --- THAY ĐỔI 3: Sửa đổi hoàn toàn hàm forward ---
    def forward(self, state, hidden_state=None):
        """
        Forward pass through critic network.
        
        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, sequence_length, state_dim).
            hidden_state (torch.Tensor, optional): Initial hidden state for the GRU. Defaults to None.
            
        Returns:
            tuple: (estimated_state_value, new_hidden_state)
        """
        gru_out, new_hidden_state = self.gru(state, hidden_state)
        
        # Lấy đầu ra của timestep cuối cùng
        x = gru_out[:, -1, :] 
        
        x = self.activation_fn(self.ln1(self.fc1(x)))
        x = self.activation_fn(self.ln2(self.fc2(x)))
        x = self.activation_fn(self.ln3(self.fc3(x)))
        
        value = self.value_head(x)
        
        return value, new_hidden_state