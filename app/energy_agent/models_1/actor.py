import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """
    Actor network with GRU for handling sequential states.
    """
    
    # Thêm các tham số cho GRU
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 gru_hidden_dim=128, gru_num_layers=1, activation='relu'):
        """
        Initialize Actor network with GRU.
        
        Args:
            state_dim (int): Dimension of a single state input at one timestep.
            action_dim (int): Dimension of the output action.
            hidden_dim (int): Hidden layer dimension for the MLP part.
            gru_hidden_dim (int): Hidden dimension for the GRU layer.
            gru_num_layers (int): Number of GRU layers to stack.
            activation (str): Activation function type.
        """
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # --- THAY ĐỔI 1: Thêm lớp GRU ---
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True  # Rất quan trọng! Giúp input có dạng (batch, seq, features)
        )
        
        # --- THAY ĐỔI 2: Lớp MLP đầu tiên nhận đầu vào từ GRU ---
        # Kích thước đầu vào của fc1 bây giờ là gru_hidden_dim
        self.fc1 = nn.Linear(gru_hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Action heads không đổi
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.action_logstd = nn.Linear(hidden_dim // 2, action_dim)

        # Activation function không đổi
        if activation == 'relu':
            self.activation_fn = F.relu
        if activation == 'tanh':
            self.activation_fn = torch.tanh
        if activation == 'elu':
            self.activation_fn = F.elu
        else:
            self.activation_fn = F.relu

        self.init_weights()

    def init_weights(self):
        # Hàm này không cần thay đổi nhiều, lớp GRU có cơ chế khởi tạo riêng khá tốt
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)
        
        nn.init.constant_(self.action_logstd.weight, 0.0)
        nn.init.constant_(self.action_logstd.bias, -0.5)

    # --- THAY ĐỔI 3: Sửa đổi hoàn toàn hàm forward ---
    def forward(self, state, hidden_state=None):
        """
        Forward pass through actor network.
        
        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, sequence_length, state_dim).
            hidden_state (torch.Tensor, optional): Initial hidden state for the GRU. Defaults to None.
            
        Returns:
            tuple: (action_mean, action_logstd, new_hidden_state)
        """
        # 1. Đưa state và hidden_state qua lớp GRU
        # gru_out shape: (batch_size, sequence_length, gru_hidden_dim)
        # new_hidden_state shape: (num_layers, batch_size, gru_hidden_dim)
        gru_out, new_hidden_state = self.gru(state, hidden_state)
        
        # 2. Chỉ lấy đầu ra của timestep cuối cùng trong chuỗi để đưa ra quyết định
        # x shape: (batch_size, gru_hidden_dim)
        x = gru_out[:, -1, :] 
        
        # 3. Đưa qua các lớp MLP như bình thường với layer norm
        x = self.activation_fn(self.ln1(self.fc1(x)))
        x = self.activation_fn(self.ln2(self.fc2(x)))
        x = self.activation_fn(self.ln3(self.fc3(x)))
        
        # Use tanh for more stable output
        action_mean = (torch.tanh(self.action_mean(x)) + 1.0) / 2.0
        action_logstd = torch.clamp(self.action_logstd(x), min=-20, max=2)
        
        # Trả về cả hidden_state mới để agent có thể lưu lại cho timestep tiếp theo
        return action_mean, action_logstd, new_hidden_state