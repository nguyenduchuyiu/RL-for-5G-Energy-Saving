# energy_agent/rl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import logging
import random
from datetime import datetime

# Import các lớp đã định nghĩa
from .models.actor import Actor
from .models.critic import Critic
from .state_normalizer import StateNormalizer

# DATA AUGMENTATION FOR DrQ-v2
class RandomShiftsAug(nn.Module):
    """
    Lớp thực hiện gia tăng dữ liệu bằng cách dịch chuyển ngẫu nhiên.
    Thêm một padding nhỏ và sau đó cắt (crop) ngẫu nhiên.
    """
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c = x.size()
        assert c % 2 == 0 # Kích thước state phải là số chẵn
        x = x.reshape(n, c, 1, 1) # Chuyển về dạng ảnh 2D giả
        x = F.pad(x, [self.pad] * 4)
        n, c, h, w = x.size()
        w1 = torch.randint(0, self.pad * 2 + 1, (n,))
        h1 = torch.randint(0, self.pad * 2 + 1, (n,))
        cropped = torch.empty((n, c, h - 2 * self.pad, w - 2 * self.pad), device=x.device)
        for i, (x_i, w1_i, h1_i) in enumerate(zip(x, w1, h1)):
            cropped[i] = x_i[:, h1_i:h1_i + h - 2 * self.pad, w1_i:w1_i + w - 2 * self.pad]
        return cropped.reshape(n, c)


class ReplayBuffer:
    """Bộ đệm kinh nghiệm off-policy."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        print("Initializing RL Agent using DrQ-v2")
        self.n_cells = n_cells
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        self.state_dim = 17 + 14 + (n_cells * 12)
        # Để dùng augmentation, state_dim cần là số chẵn. Thêm 1 chiều nếu cần.
        if self.state_dim % 2 != 0:
            self.state_dim += 1

        self.action_dim = n_cells
        
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)
        
        # DrQ-v2 Augmentation
        self.aug = RandomShiftsAug(pad=4).to(self.device)

        # Khởi tạo mạng
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic1_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3)

        # Tự động điều chỉnh Alpha (Entropy)
        self.target_entropy = -self.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 32 # Batch size nhỏ hơn cho 300 step
        self.buffer_capacity = int(1e5) # Buffer vẫn nên lớn
        
        self.buffer = ReplayBuffer(self.buffer_capacity)
        
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        
        self.setup_logging(log_file)
        self.logger.info(f"DrQ-v2 Agent: State dim={self.state_dim}, Action dim={self.action_dim}, Device={self.device}")

    def setup_logging(self, log_file):
        self.logger = logging.getLogger('DrQv2Agent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def start_scenario(self):
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.logger.info(f"Starting episode {self.total_episodes}")
    
    def end_scenario(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        self.logger.info(f"Episode {self.total_episodes} ended: Steps={self.episode_steps}, Reward={self.current_episode_reward:.2f}, Avg100={avg_reward:.2f}")

    def get_action(self, state):
        with torch.no_grad():
            state_norm = self.state_normalizer.normalize(state)
            
            # Đệm 0 nếu state_dim lẻ
            if len(state_norm) < self.state_dim:
                state_norm = np.append(state_norm, 0)
                
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic=(not self.training_mode))
        
        # Chuyển từ [-1, 1] sang [0, 1]
        action_rescaled = (action + 1) / 2.0
        return action_rescaled.cpu().numpy().flatten()

    def calculate_reward(self, prev_state, action, current_state):
            """
            1. Log-Barrier cho Drop Rate & Latency, đọc ngưỡng từ state.
            2. Thêm thành phần thưởng/phạt dựa trên Chất lượng Tín hiệu (SINR)
            để dạy agent tư duy phòng ngừa.
            3. Giữ nguyên Reward Scaling.
            """
            if prev_state is None: return 0.0
            
            current_state = np.array(current_state).flatten()
            prev_state = np.array(prev_state).flatten()

            # Indices
            SIM_START, NET_START = 0, 17
            
            # --- Metrics & Thresholds read directly from state ---
            current_energy = current_state[NET_START + 0]
            prev_energy = prev_state[NET_START + 0]
            
            current_drop_rate = current_state[SIM_START + 11]
            DROP_RATE_THRESHOLD = current_state[SIM_START + 11]

            current_latency = current_state[SIM_START + 12]
            LATENCY_THRESHOLD = current_state[SIM_START + 12]

            # --- 1. Energy Saving Reward ---
            energy_reward = (prev_energy - current_energy) * 0.1

            # --- 2. Generalizable Penalties for Hard Constraints ---
            # Log-Barrier cho Drop Rate
            margin_dr = DROP_RATE_THRESHOLD - current_drop_rate
            drop_penalty = -50.0 if margin_dr <= 0.1 else np.log(margin_dr)
            
            # Phạt tuyến tính đơn giản cho Latency
            latency_penalty = -max(0, current_latency - LATENCY_THRESHOLD) * 0.1

            # --- 3. NEW: Proactive Signal Quality Reward ---
            # Lấy ra chỉ số SINR của tất cả các cell đang hoạt động
            start_idx = SIM_START + 17 + 14
            sinr_values = []
            num_active_cells = int(current_state[0])
            for i in range(num_active_cells):
                # Index của avgSINR cho cell i. 12 là số feature mỗi cell.
                sinr_idx = start_idx + i * 12 + 9 
                if sinr_idx < len(current_state):
                    sinr_values.append(current_state[sinr_idx])

            signal_quality_reward = 0.0
            if sinr_values:
                avg_sinr = np.mean(sinr_values)
                # SINR tốt thường > 10 dB. Chúng ta sẽ thưởng/phạt xung quanh mốc này.
                # Dùng hàm tanh để tạo ra một phần thưởng/phạt mượt mà, giới hạn trong [-1, 1]
                # sau đó nhân với một hệ số nhỏ.
                signal_quality_reward = np.tanh((avg_sinr - 10) / 5) * 0.5 

            total_reward = energy_reward + drop_penalty + latency_penalty + signal_quality_reward

            # --- 4. CRITICAL STEP: REWARD SCALING ---
            REWARD_SCALE = 10.0
            return float(total_reward / REWARD_SCALE)

    def update(self, state, action, next_state, done):
        if not self.training_mode: return
        
        reward = self.calculate_reward(state, action, next_state)
        self.current_episode_reward += reward
        self.episode_steps += 1
        self.total_steps += 1
        
        safe_action = np.array(action).flatten()
        
        state_norm = self.state_normalizer.normalize(state)
        next_state_norm = self.state_normalizer.normalize(next_state)

        # Đệm 0 nếu state_dim lẻ
        if len(state_norm) < self.state_dim:
            state_norm = np.append(state_norm, 0)
            next_state_norm = np.append(next_state_norm, 0)

        self.buffer.add(state_norm, safe_action, reward, next_state_norm, done)
        
        if len(self.buffer) > self.batch_size:
            self.train()

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # --- Data Augmentation ---
        states = self.aug(states)
        next_states = self.aug(next_states)
        
        # --- Cập nhật Critic ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            alpha = self.log_alpha.exp()
            target_q = rewards + (1 - dones) * self.gamma * (q_next - alpha * next_log_prob)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Cập nhật Actor và Alpha ---
        pi, log_prob = self.actor.sample(states)
        q1_pi = self.critic1(states, pi)
        q2_pi = self.critic2(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (alpha.detach() * log_prob - min_q_pi).mean()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Cập nhật mạng Target ---
        with torch.no_grad():
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = f"drqv2_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.logger.info(f"Model loaded from {filepath}")

    def set_training_mode(self, training):
        self.training_mode = training
        self.actor.train(training)
        self.critic1.train(training)
        self.critic2.train(training)
        self.logger.info(f"Training mode set to {training}")

    def get_stats(self):
            """Lấy thông số thống kê quá trình huấn luyện (tùy chọn, để debug)."""
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            
            return {
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps,
                'avg_reward_last_100': avg_reward,
                'buffer_size': len(self.buffer),
                'training_mode': self.training_mode,
                'current_episode_reward': self.current_episode_reward
        }