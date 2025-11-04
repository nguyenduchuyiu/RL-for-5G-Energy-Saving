import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from datetime import datetime
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .transition import Transition, TrajectoryBuffer
from .models import Actor
from .models import Critic

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = yaml.safe_load(open('config.yaml'))

class RunningNorm:
    """
    Adaptive normalizer for augmented state vectors.
    Tracks running mean and variance for each feature (column-wise).
    Works for both global + per-cell stacked features.
    """
    def __init__(self, state_dim, eps=1e-8):
        self.mean = np.zeros(state_dim, dtype=np.float32)
        self.var = np.ones(state_dim, dtype=np.float32)
        self.count = eps
        self.eps = eps

    def update(self, x):
        """
        Update running mean/var statistics.
        Args:
            x (np.ndarray): batch of states, shape (batch_size, state_dim)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        """
        Normalize input state(s) using running mean/var.
        Args:
            x (np.ndarray): state or batch, shape (..., state_dim)
        Returns:
            np.ndarray: normalized state(s)
        """
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)

    def denormalize(self, x):
        """
        Reverse normalization (optional, for debugging).
        """
        return x * np.sqrt(self.var) + self.mean



class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False, max_cells=100):
        """
        Initialize PPO agent for 5G energy saving
        
        Args:
            n_cells (int): Number of cells to control
            n_ues (int): Number of UEs in network
            max_time (int): Maximum simulation time steps
            log_file (str): Path to log file
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.n_cells = int(n_cells)
        self.max_cells = int(max_cells)
        self.max_time = int(max_time)
        self.use_gpu = config['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
                
        # 7 global features + 10 augmented features + 8*max_cells local features
        self.state_dim = 7 + 10 + 8*max_cells
        
        # Power ratio for each cell
        self.action_dim = self.max_cells
        
        # Parallel envs
        self.n_envs = int(config.get('n_envs', 1))
                                
        # PPO hyperparameters
        self.gamma = config['gamma']
        self.lambda_gae = config['lambda_gae']
        self.clip_epsilon = config['clip_epsilon']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        self.buffer_size = config['buffer_size']
        self.hidden_dim = config['hidden_dim']
        self.entropy_coef = config['entropy_coef']
        # Lagrangian PPO hyperparameters
        self.lambda_lr = float(config['lambda_lr'])
        self.lambda_multiplier = float(config['lambda_init'])
        self.lambda_max = float(config['lambda_max'])
        self.cost_target = float(config['cost_target'])
        # Practical tolerance for constraint satisfaction and early stop control
        self.cost_tolerance = float(config['cost_tolerance'])
        self.cost_stop_patience = int(config['cost_stop_patience'])
        self.cost_ok_streak = 0
                
        # use augmented state dimension for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.hidden_dim).to(self.device)
        # Separate cost critic for Lagrangian PPO
        self.cost_critic = Critic(self.state_dim, self.hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=config['critic_lr'])
        self.training_mode = config['training_mode']
        self.checkpoint_load_path = config['checkpoint_load_path']
        self.checkpoint_save_path = config['checkpoint_save_path']
        
        # Experience buffer (per-env, flat with env_id)
        self.buffer = TrajectoryBuffer()
        
        # Running normalization
        # use state_dim before padding for running normalization
        self.running_norm = RunningNorm(state_dim=7 + 10 + 8*n_cells)
        
        # Scale episode step target by number of envs to keep similar update cadence
        self.step_per_episode = max(1, int(config['buffer_size'] // max(1, self.n_envs)))
        self.total_episodes = int(self.max_time / self.step_per_episode)
        self.current_episode = 1
        
        # Per-env action and log_prob tracking
        self.current_padded_actions = {i: np.ones(self.max_cells) * 0.7 for i in range(self.n_envs)}
        self.last_log_probs = {i: 0.0 for i in range(self.n_envs)}
        
        # Track previous state for delta features (temporal info)
        self.prev_states = {i: None for i in range(self.n_envs)}
        
        # Metrics tracking (Lagrangian PPO essentials)
        self.metrics = {
            'reward': [],           # mean reward per update
            'cost': [],             # mean cost per update
            'lambda': [],           # lagrange multiplier
            'actor_loss': [],
            'critic_loss': [],
            'cost_critic_loss': [],
            'entropy': []
        }
        
                
        self.setup_logging(log_file)
        
        if config['use_gpu'] and not torch.cuda.is_available():
            self.logger.warning("GPU requested but CUDA not available, using CPU instead")
        
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
        
        if os.path.exists(self.checkpoint_load_path):
            self.load_model(self.checkpoint_load_path)
            self.logger.info(f"Loaded checkpoint from {self.checkpoint_load_path}")
        else:
            self.logger.info(f"No checkpoint found at {self.checkpoint_load_path}")
    
    def augment_state(self, current_state_raw, prev_state_raw=None):
        """
        Selective augmentation (raw, no normalization, no padding).
        Returns 1D numpy array of floats.
        """
        cur = np.asarray(current_state_raw).flatten()
        n = int(self.n_cells)

        NETWORK_START_IDX = 17
        CELL_START_IDX = 31  # column-stacked base

        # --- safe global reads ---
        drop_threshold = float(cur[11])
        latency_threshold = float(cur[12])
        cpu_threshold = float(cur[13])
        prb_threshold = float(cur[14])

        active_cells = float(cur[NETWORK_START_IDX + 1])
        current_drop_rate = float(cur[NETWORK_START_IDX + 2])
        current_latency = float(cur[NETWORK_START_IDX + 3])
        total_traffic = float(cur[NETWORK_START_IDX + 4])
        total_tx_power = float(cur[NETWORK_START_IDX + 12])
        current_energy = float(cur[NETWORK_START_IDX + 0])

        # prev-safe
        if prev_state_raw is not None:
            prev = np.asarray(prev_state_raw).flatten()
            prev_energy = float(prev[NETWORK_START_IDX + 0])
            prev_drop_rate = float(prev[NETWORK_START_IDX + 2])
            prev_latency = float(prev[NETWORK_START_IDX + 3])
        else:
            prev_energy = current_energy
            prev_drop_rate = current_drop_rate
            prev_latency = current_latency

        # --- per-cell extraction (column-stacked) ---
        def read_block(offset):
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = CELL_START_IDX + offset * n
            return np.asarray(cur[start:start + n], dtype=np.float32)

        per_cell_total_traffic_demand = read_block(10)   # block offset 10
        per_cell_tx_power             = read_block(5)    # block offset 5
        per_cell_energy_consumption   = read_block(6)    # block offset 6
        per_cell_load_ratio           = read_block(11)   # block offset 11
        per_cell_prb_usage            = read_block(1)    # block offset 1
        per_cell_cpu_usage            = read_block(0)    # block offset 0
        per_cell_avg_sinr             = read_block(9)    # block offset 9

        # safe aggregates
        if n > 0:
            max_cpu_usage = float(np.max(per_cell_cpu_usage))
            max_prb_usage = float(np.max(per_cell_prb_usage))
            avg_cell_load = float(np.mean(per_cell_load_ratio))
            all_cell_loads = per_cell_load_ratio.copy().astype(np.float32)
        else:
            max_cpu_usage = 0.0
            max_prb_usage = 0.0
            avg_cell_load = 0.0
            all_cell_loads = np.zeros(0, dtype=np.float32)

        # --- global features (raw) ---
        global_features = np.array([
            total_traffic,
            active_cells,
            total_tx_power,
            current_drop_rate,
            current_latency,
            max_cpu_usage,
            max_prb_usage,
        ], dtype=np.float32)

        # --- augmented scalars (raw) ---
        dist_to_drop_thresh = drop_threshold - current_drop_rate
        dist_to_latency_thresh = latency_threshold - current_latency
        dist_to_cpu_thresh = cpu_threshold - max_cpu_usage
        dist_to_prb_thresh = prb_threshold - max_prb_usage

        drop_rate_delta = current_drop_rate - prev_drop_rate
        latency_delta = current_latency - prev_latency
        energy_delta = current_energy - prev_energy

        safe_active_cells = max(1.0, active_cells)
        load_per_active_cell = total_traffic / safe_active_cells

        # power_efficiency: use log1p to avoid explosion; keep raw scale
        power_efficiency = np.log1p(total_traffic) / (np.log1p(total_tx_power) + 1e-6)
        # clamp to reasonable raw bounds so downstream code doesn't explode
        power_efficiency = float(np.clip(power_efficiency, -1e6, 1e6))

        lambda_multiplier = float(getattr(self, "lambda_multiplier", 0.0))

        augmented_features = np.array([
            dist_to_drop_thresh,
            dist_to_latency_thresh,
            dist_to_cpu_thresh,
            dist_to_prb_thresh,
            drop_rate_delta,
            latency_delta,
            energy_delta,
            load_per_active_cell,
            power_efficiency,
            lambda_multiplier,
        ], dtype=np.float32)

        # local correlation (hotspot) = per cell load deviation from mean
        load_deltas = (all_cell_loads - avg_cell_load).astype(np.float32) if n > 0 else np.zeros(0, dtype=np.float32)

        # --- assemble per-cell flat block (concatenate blocks in chosen order) ---
        # order: traffic, txPower, energy, loadRatio, prbUsage, cpuUsage, avgSINR, load_delta
        per_cell_blocks = [
            per_cell_total_traffic_demand.astype(np.float32),
            per_cell_tx_power.astype(np.float32),
            per_cell_energy_consumption.astype(np.float32),
            per_cell_load_ratio.astype(np.float32),
            per_cell_prb_usage.astype(np.float32),
            per_cell_cpu_usage.astype(np.float32),
            per_cell_avg_sinr.astype(np.float32),
            load_deltas.astype(np.float32),
        ]

        # concat safely even if n == 0
        if len(per_cell_blocks) > 0:
            per_cell_concat = np.concatenate(per_cell_blocks, axis=0) if per_cell_blocks[0].size > 0 else np.zeros(0, dtype=np.float32)
        else:
            per_cell_concat = np.zeros(0, dtype=np.float32)

        # --- final augmented state vector (raw, not normalized, not padded) ---
        augmented_state = np.concatenate([
            global_features,
            augmented_features,
            per_cell_concat
        ], axis=0).astype(np.float32)

        return augmented_state

    def pad_state(self, augmented_state):
        """
        Pad augmented_state (raw, not normalized) to fixed-length blocks per max_cells.
        Expected input layout (column-stacked per-cell blocks):
        [ global_features (7),
            augmented_features (10),
            block0 (n_cells), block1 (n_cells), ..., block7 (n_cells) ]
        Where block order is:
        0: per_cell_total_traffic_demand
        1: per_cell_tx_power
        2: per_cell_energy_consumption
        3: per_cell_load_ratio
        4: per_cell_prb_usage
        5: per_cell_cpu_usage
        6: per_cell_avg_sinr
        7: load_deltas

        Returns:
        padded_state: 1D np.float32 array of length 17 + 8*max_cells
        """
        arr = np.asarray(augmented_state).flatten().astype(np.float32)
        GLOBAL_CNT = 7
        AUG_CNT = 10
        PER_CELL_BLOCKS = 8
        n = int(self.n_cells)
        max_cells = int(self.max_cells)

        expected_len = GLOBAL_CNT + AUG_CNT + PER_CELL_BLOCKS * n
        if arr.size != expected_len:
            raise ValueError(f"augmented_state length mismatch: got {arr.size}, expected {expected_len} "
                            f"(check self.n_cells={n} and augmented_state layout)")

        head = arr[: GLOBAL_CNT + AUG_CNT]
        base = GLOBAL_CNT + AUG_CNT

        pad_each = max_cells - n
        padded_blocks = []
        for feat_idx in range(PER_CELL_BLOCKS):
            start = base + feat_idx * n
            end = start + n
            if n == 0:
                padded_block = np.zeros(max_cells, dtype=np.float32)
            else:
                block = arr[start:end]
                if pad_each > 0:
                    padded_block = np.concatenate([block, np.zeros(pad_each, dtype=np.float32)])
                else:
                    # already full size or n == max_cells
                    padded_block = block[:max_cells]
            padded_blocks.append(padded_block)

        # concatenate blocks in feature order: block0_padded, block1_padded, ...
        cells_padded = np.concatenate(padded_blocks, axis=0) if padded_blocks else np.zeros(0, dtype=np.float32)

        padded_state = np.concatenate([head, cells_padded], axis=0).astype(np.float32)
        return padded_state


    
    def setup_logging(self, log_file):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PPOAgent')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler only (no file logging)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def start_scenario(self):
        print("===================Starting scenario===================")
        self.start_episode()
    
    def end_scenario(self):
        print("===================Ending scenario===================")
        if self.training_mode:
            self.save_plots()
            self.save_model(self.checkpoint_save_path)
    
    def start_episode(self):
        self.logger.info(f"Starting episode: {self.current_episode}")

        
        # Reset per-env states
        for i in range(self.n_envs):
            self.current_padded_actions[i] = np.ones(self.max_cells) * 0.7
            self.last_log_probs[i] = 0.0

    
    def end_episode(self):
        self.train()
        self.current_episode += 1
        if self.current_episode < self.total_episodes:
            self.start_episode()
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def get_action(self, state, env_id=0):
        """
        Get action from policy network.
        Correctly computes log_prob *after* forcing/clamping actions.
        """

        # 1. Xử lý State
        raw_state = np.array(state).flatten()   
        augmented_state = self.augment_state(raw_state)
        self.running_norm.update(augmented_state) # Giữ ở đây (VÀ XÓA TRONG HÀM UPDATE)
        normalized_augmented_state = self.running_norm.normalize(augmented_state)
        padded_augmented_state = self.pad_state(normalized_augmented_state)
        state_tensor = torch.FloatTensor(padded_augmented_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 2. Lấy Phân phối (Distribution)
            action_mean, action_logstd = self.actor(state_tensor)
            # Dải clamp hẹp để tránh std quá nhỏ/lớn
            action_logstd = torch.clamp(action_logstd, min=-3.0, max=-0.2)
            action_std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, action_std)

            # 3. Lấy Action "Gợi ý"
            if self.training_mode:
                action_tensor = dist.sample() # Sample nếu training
            else:
                action_tensor = action_mean # Dùng mean nếu inference

            # 4. Ép (Force) Action (TẤT CẢ TRONG PYTORCH)
            action_final_tensor = action_tensor.clone()
            n = int(self.n_cells)
            
            # (Bỏ post-processing theo traffic: không ép cell idle về 0)

            # Ép cell padding (ngoài n_cells) về 0
            if self.max_cells > n:
                action_final_tensor[0, n:] = 0.0

            # Clamp action cuối cùng về [0, 1]
            action_final_tensor = torch.clamp(action_final_tensor, 0.0, 1.0)
            
            # 5. TÍNH LOG_PROB TRÊN ACTION CUỐI CÙNG (A_forced)
            if self.training_mode:
                # Tính log_prob của action_final_tensor (là action sẽ được thực thi)
                log_prob_per_dim = dist.log_prob(action_final_tensor)
                # Tổng log_prob theo n cell
                log_prob_sum = float(log_prob_per_dim[0, :n].sum().cpu().numpy())
            else:
                log_prob_sum = 0.0
            
            # 6. Lưu Action và Log_Prob đã đồng bộ
            action_final_np = action_final_tensor.cpu().numpy().flatten()
            self.current_padded_actions[env_id] = action_final_np
            self.last_log_probs[env_id] = log_prob_sum
            
            # Trả về action đã cắt ngắn cho môi trường
            truncated_action = action_final_np[:n]
            
            if not self.training_mode:
                self.prev_states[env_id] = raw_state.copy()
            
            # if np.random.random() < 0.2:
            #     print(f"Action: {truncated_action}")
                
            return truncated_action

    
    ## OPTIONAL: Modify reward calculation as needed
    def calculate_reward(self, prev_state, action, current_state, env_id=0):
        """
        Pure Lagrangian reward with dense signal:
        - Objective: minimize total energy (or per-traffic energy)
        - Constraint (QoS) handled externally by lambda update
        """
        if prev_state is None: 
            return 0.0

        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()

        NETWORK_START_IDX = 17
        
        total_traffic_demand = current_state[NETWORK_START_IDX + 4]
        total_energy_delta = current_state[NETWORK_START_IDX + 0] - prev_state[NETWORK_START_IDX + 0]
        energy_scale = config['energy_reward_scale']
        reward = 0.0
        if total_traffic_demand < 1e-6:
            reward = -0.01 * np.clip(total_energy_delta, -100, 100)
        else:
            reward = -energy_scale * total_energy_delta / total_traffic_demand

        # Clip để tránh outlier
        reward = np.clip(reward, -100.0, 100.0)

        if env_id == 0 and np.random.random() < 0.02:
            print(f"Energy Delta={total_energy_delta:.4f}, Total Traffic Demand={total_traffic_demand:.4f}, Reward={reward:.4f}")

        return float(reward)


    def calculate_cost(self, current_state, env_id=0):
        current_state = np.array(current_state).flatten()

        NETWORK_START_IDX = 17
        CELL_START_IDX = 17 + 14
        n = int(self.n_cells)
        cur = np.asarray(current_state).flatten()

        drop = current_state[NETWORK_START_IDX + 2]
        latency = current_state[NETWORK_START_IDX + 3]
        drop_th = current_state[11]
        latency_th = current_state[12]
        cpu_th = current_state[13]
        prb_th = current_state[14]

        # --- per-cell extraction (column-stacked) ---
        def read_block(offset):
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = CELL_START_IDX + offset * n
            return np.asarray(cur[start:start + n], dtype=np.float32)

        max_cpu = np.max(read_block(0))
        max_prb = np.max(read_block(1))

        def _violation_penalty(x, th):
            violation_ratio = max(0.0, (x - th) / max(1e-6, th))
            return violation_ratio

        drop_violation = _violation_penalty(drop, drop_th)
        latency_violation = _violation_penalty(latency, latency_th)
        cpu_violation = _violation_penalty(max_cpu, cpu_th)
        prb_violation = _violation_penalty(max_prb, prb_th)

        qos_cost = drop_violation + latency_violation + cpu_violation + prb_violation

        qos_cost = np.clip(qos_cost, 0.0, 100.0)
        qos_cost = qos_cost * config['cost_scale']

        if env_id == 0 and np.random.random() < 0.02:
            print(f"Drop={drop:.4f}%, Latency={latency:.4f}ms, CPU={max_cpu:.4f}%, PRB={max_prb:.4f}%")
            print(f"QoS Cost: {qos_cost:.4f}")
            print(f"Drop Violation={drop_violation:.4f}, Latency Violation={latency_violation:.4f}, "
                f"CPU Violation={cpu_violation:.4f}, PRB Violation={prb_violation:.4f}")

        return float(qos_cost)


    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def update(self, state, action, next_state, done, env_id=0):
        """
        Update agent with experience
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
            env_id: Environment id for parallel execution
        """
        if not self.training_mode:
            return
        
        action = self.current_padded_actions.get(env_id, np.ones(self.max_cells) * 0.7)
        
        # Calculate actual reward and cost using state as prev_state and next_state as current
        actual_reward = self.calculate_reward(state, action, next_state, env_id)
        actual_cost = self.calculate_cost(next_state, env_id)
        
        # Convert inputs to numpy if needed
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        
        # Ensure proper shapes
        raw_state = np.array(state).flatten().copy()
        augmented_state = self.augment_state(raw_state)
        normalized_augmented_state = self.running_norm.normalize(augmented_state)
        padded_augmented_state = self.pad_state(normalized_augmented_state)
        
        action = np.array(action).flatten()
        # Get value estimates
        state_tensor = torch.FloatTensor(padded_augmented_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
        
        # Create transition
        # Build transition with next_state retained (normalized/padded/augmented like state)
        raw_next_state = np.array(next_state).flatten().copy()
        next_augmented_state = self.augment_state(raw_next_state)
        normalized_next_augmented_state = self.running_norm.normalize(next_augmented_state)
        padded_next_augmented_state = self.pad_state(normalized_next_augmented_state)

        transition = Transition(
            state=padded_augmented_state,
            action=action,
            reward=actual_reward,
            cost=actual_cost,
            next_state=padded_next_augmented_state,
            done=done,
            log_prob=self.last_log_probs.get(env_id, 0.0),
            value=value,
            env_id=env_id
        )
        
        self.buffer.add(transition)
        
        # Update prev_states for next step (important for delta features)
        self.prev_states[env_id] = raw_state.copy()
        
        # Check for episode termination
        if done:
            # Natural episode termination - reset prev_state for this env
            self.prev_states[env_id] = None
            self.logger.info("Episode terminated naturally (done=True)")
        elif len(self.buffer.memory) >= self.buffer_size and self.training_mode:
            # Forced episode termination due to buffer size
            self.end_episode()
    
    def train(self):
        """Train PPO using experiences collected across parallel envs (per-env GAE)."""
        if len(self.buffer.memory) < self.batch_size:
            return

        # Fetch and clear buffer
        states, actions, rewards, costs, next_states, dones, old_log_probs, values, env_ids = self.buffer.get_all_and_clear()

        # convert lists -> numpy arrays / tensors
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        costs = np.asarray(costs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        old_log_probs = np.asarray(old_log_probs, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        env_ids_np = np.asarray(env_ids, dtype=np.int32)

        # To tensors
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        costs_tensor = torch.FloatTensor(costs).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        env_ids_np = np.array(env_ids)

        # Compute GAE per env
        unique_envs = np.unique(env_ids_np)
        all_advantages = torch.zeros_like(rewards_tensor)
        all_returns = torch.zeros_like(values_tensor)
        all_cost_advantages = torch.zeros_like(costs_tensor)
        all_cost_returns = torch.zeros_like(costs_tensor)

        # Precompute cost value predictions with no grad for all states (targets)
        states_tensor_full = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            cost_values_full = self.cost_critic(states_tensor_full).squeeze()

        for env_id in unique_envs:
            env_mask = (env_ids_np == env_id)
            env_indices = np.where(env_mask)[0]

            env_rewards = rewards_tensor[env_mask]
            env_values = values_tensor[env_mask]
            env_dones = dones_tensor[env_mask]

            # Bootstrap with critic on last state of this env
            # Use stored next_state for bootstrap when available, else last state
            last_idx = env_indices[-1]
            bootstrap_input = next_states[last_idx] if (next_states is not None and next_states[last_idx] is not None) else states[last_idx]
            last_state = torch.FloatTensor(bootstrap_input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(last_state).squeeze()
                next_cost_value = self.cost_critic(last_state).squeeze()

            advantages = torch.zeros_like(env_rewards)
            cost_advantages = torch.zeros_like(costs_tensor[env_mask])
            last_adv = 0.0
            last_cost_adv = 0.0
            # Append next_value conceptually by indexing t+1 at end
            for t in reversed(range(len(env_rewards))):
                non_terminal = 1.0 - env_dones[t]
                nv = next_value if t == len(env_rewards) - 1 else env_values[t + 1]
                delta = env_rewards[t] + self.gamma * nv * non_terminal - env_values[t]
                last_adv = delta + self.gamma * self.lambda_gae * non_terminal * last_adv
                advantages[t] = last_adv
                # Cost side GAE
                cv_t = cost_values_full[env_indices[t]]
                nv_c = next_cost_value if t == len(env_rewards) - 1 else cost_values_full[env_indices[t + 1]]
                delta_c = costs_tensor[env_mask][t] + self.gamma * nv_c * non_terminal - cv_t
                last_cost_adv = delta_c + self.gamma * self.lambda_gae * non_terminal * last_cost_adv
                cost_advantages[t] = last_cost_adv

            returns = advantages + env_values
            all_advantages[env_mask] = advantages
            all_returns[env_mask] = returns
            cost_returns = cost_advantages + cost_values_full[env_mask]
            all_cost_advantages[env_mask] = cost_advantages
            all_cost_returns[env_mask] = cost_returns

        # Dataset tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = all_advantages
        returns_tensor = all_returns
        cost_advantages_tensor = all_cost_advantages
        cost_returns_tensor = all_cost_returns

        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()
        if len(cost_advantages_tensor) > 1:
            cost_advantages_tensor = (cost_advantages_tensor - cost_advantages_tensor.mean()) / (cost_advantages_tensor.std() + 1e-8)
        else:
            cost_advantages_tensor = cost_advantages_tensor - cost_advantages_tensor.mean()

        advantages_tensor = torch.clamp(advantages_tensor, min=-10.0, max=10.0)
        cost_advantages_tensor = torch.clamp(cost_advantages_tensor, min=-10.0, max=10.0)
        
        dataset = torch.utils.data.TensorDataset(
            states_tensor, actions_tensor,
            old_log_probs_tensor, advantages_tensor, returns_tensor,
            cost_advantages_tensor, cost_returns_tensor
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        final_entropy = 0.0
        final_actor_loss = 0.0
        final_critic_loss = 0.0
        final_cost_critic_loss = 0.0

        for epoch in range(self.ppo_epochs):
            for batch in loader:
                (batch_states, batch_actions, batch_old_log_probs,
                 batch_advantages, batch_returns,
                 batch_cost_advantages, batch_cost_returns) = batch
                action_mean, action_logstd = self.actor(batch_states)
                action_logstd = torch.clamp(action_logstd, min=-3.0, max=-0.2)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)

                n = int(self.n_cells)
                log_prob_per_dim = dist.log_prob(batch_actions)
                new_log_probs = log_prob_per_dim[:, :n].sum(-1)
                entropy = dist.entropy()[:, :n].sum(-1).mean()

                # Clamp log-ratio to avoid exponential blow-up (tighter)
                log_ratio = (new_log_probs - batch_old_log_probs).clamp(-2.0, 2.0)
                ratio = torch.exp(log_ratio)
                # Reward surrogate
                surr1_r = ratio * batch_advantages
                surr2_r = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                L_reward = torch.min(surr1_r, surr2_r).mean()
                # Cost surrogate
                surr1_c = ratio * batch_cost_advantages
                surr2_c = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_cost_advantages
                L_cost = torch.min(surr1_c, surr2_c).mean()
                # Lagrangian objective (maximize reward, minimize cost)
                lagrange = torch.clamp(torch.tensor(self.lambda_multiplier, device=self.device), 0.0, self.lambda_max)
                # Entropy bonus for exploration
                actor_loss = -(L_reward - lagrange * L_cost + self.entropy_coef * entropy)
                current_values = self.critic(batch_states).view(-1)
                critic_loss = nn.MSELoss()(current_values, batch_returns.view(-1).detach())
                # Cost critic loss
                current_cost_values = self.cost_critic(batch_states).view(-1)
                cost_critic_loss = nn.MSELoss()(current_cost_values, batch_cost_returns.view(-1).detach())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 0.5)
                self.cost_critic_optimizer.step()

                final_entropy = entropy.item()
                final_actor_loss = actor_loss.item()
                final_critic_loss = critic_loss.item()
                final_cost_critic_loss = cost_critic_loss.item()

        # Dual ascent update for lambda (compute per-env mean cost (proxy for episodic cost))
        env_cost_means = []
        for env_id in unique_envs:
            env_mask = (env_ids_np == env_id)
            if env_mask.sum() == 0:
                continue
            env_cost = costs[env_mask]  # costs is numpy array earlier
            env_cost_means.append(float(np.mean(env_cost)))
        if len(env_cost_means) == 0:
            J_c = 0.0
        else:
            J_c = float(np.mean(env_cost_means))
        # Dual ascent with slack: allow tolerance around the target
        target_with_tol = self.cost_target + self.cost_tolerance
        self.lambda_multiplier = float(np.clip(self.lambda_multiplier + self.lambda_lr * (J_c - target_with_tol), 0.0, self.lambda_max))
        
        # Tolerance-based stopping rule (consecutive updates under target+tol)
        if J_c <= target_with_tol:
            self.cost_ok_streak += 1
        else:
            self.cost_ok_streak = 0
        
        # Mean reward across buffer
        J_r = float(np.mean(rewards)) if len(rewards) > 0 else 0.0

        self.metrics['lambda'].append(self.lambda_multiplier)
        self.metrics['reward'].append(J_r)
        self.metrics['cost'].append(J_c)
        self.metrics['entropy'].append(final_entropy)
        self.metrics['actor_loss'].append(final_actor_loss)
        self.metrics['critic_loss'].append(final_critic_loss)
        self.metrics['cost_critic_loss'].append(final_cost_critic_loss)
        self.logger.info(
            f"Train: J_r={J_r:.4f}, J_c={J_c:.4f}, Target+tol={target_with_tol:.4f}, Streak={self.cost_ok_streak}/{self.cost_stop_patience}, Lambda={self.lambda_multiplier:.4f}, "
            f"Actor Loss={final_actor_loss:.4f}, Critic Loss={final_critic_loss:.4f}, Cost Critic Loss={final_cost_critic_loss:.4f}, Entropy={final_entropy:.4f}"
        )

    
    def save_model(self, filepath=None):
        """Save model parameters"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            self.logger.info(f"Created directory: {parent_dir}")
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'cost_critic_state_dict': self.cost_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
            'episodes_trained': self.total_episodes,
            'lambda_multiplier': self.lambda_multiplier,
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'cost_critic_state_dict' in checkpoint:
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'cost_critic_optimizer_state_dict' in checkpoint:
            self.cost_critic_optimizer.load_state_dict(checkpoint['cost_critic_optimizer_state_dict'])
        if self.training_mode:
            self.current_episode = checkpoint['episodes_trained'] + 1
            self.total_episodes = checkpoint['episodes_trained'] + self.total_episodes
            print(f"current_episode: {self.current_episode}, total_episodes: {self.total_episodes}")
        if 'lambda_multiplier' in checkpoint:
            self.lambda_multiplier = float(checkpoint['lambda_multiplier'])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.logger.info(f"Training mode set to {training}")
    
    def save_plots(self):
        """Save concise Lagrangian PPO metrics plots."""
        if not any(self.metrics.values()):
            return

        def ma(data, window=10):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(f'Episode {self.total_episodes} - Lagrangian PPO Metrics', fontsize=16)

        # Reward
        if self.metrics['reward']:
            axes[0,0].plot(self.metrics['reward'], alpha=0.4, label='Reward')
            if len(self.metrics['reward']) >= 5:
                axes[0,0].plot(ma(self.metrics['reward']), label='Reward MA', linewidth=2)
            axes[0,0].set_title('Mean Reward per Update')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend(fontsize=8)

        # Cost
        if self.metrics['cost']:
            axes[0,1].plot(self.metrics['cost'], alpha=0.4, label='Cost', color='red')
            if len(self.metrics['cost']) >= 5:
                axes[0,1].plot(ma(self.metrics['cost']), label='Cost MA', linewidth=2, color='darkred')
            # Draw target and tolerance line
            axes[0,1].axhline(self.cost_target, color='gray', linestyle='--', linewidth=1, label='Cost Target')
            axes[0,1].axhline(self.cost_target + getattr(self, 'cost_tolerance', 0.0), color='gray', linestyle=':', linewidth=1, label='Target + tol')
            axes[0,1].set_title('Mean Cost per Update')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend(fontsize=8)

        # Lambda
        if self.metrics['lambda']:
            axes[0,2].plot(self.metrics['lambda'], label='Lambda', color='purple')
            axes[0,2].set_title('Lagrange Multiplier λ')
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].legend(fontsize=8)

        # Losses
        has_losses = any([
            len(self.metrics['actor_loss']) > 0,
            len(self.metrics['critic_loss']) > 0,
            len(self.metrics['cost_critic_loss']) > 0
        ])
        if has_losses:
            if self.metrics['actor_loss']:
                axes[1,0].plot(self.metrics['actor_loss'], alpha=0.4, label='Actor')
            if self.metrics['critic_loss']:
                axes[1,0].plot(self.metrics['critic_loss'], alpha=0.4, label='Critic')
            if self.metrics['cost_critic_loss']:
                axes[1,0].plot(self.metrics['cost_critic_loss'], alpha=0.4, label='Cost Critic')
            axes[1,0].set_title('Losses')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend(fontsize=8)

        # Entropy
        if self.metrics['entropy']:
            axes[1,1].plot(self.metrics['entropy'], alpha=0.4, label='Entropy', color='teal')
            if len(self.metrics['entropy']) >= 5:
                axes[1,1].plot(ma(self.metrics['entropy']), label='Entropy MA', linewidth=2, color='darkcyan')
            axes[1,1].set_title('Policy Entropy')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend(fontsize=8)

        # Hide last subplot if unused
        axes[1,2].axis('off')

        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract model name from checkpoint_save_path
        model_name = ""
        if self.checkpoint_save_path and self.checkpoint_save_path.endswith('.pth'):
            model_name = os.path.splitext(os.path.basename(self.checkpoint_save_path))[0]
        
        out_path = f'plots/{model_name}_{timestamp}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Metrics plot saved to {out_path}")