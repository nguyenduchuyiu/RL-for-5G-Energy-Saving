# energy_agent/rl_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
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
from .state_normalizer import StateNormalizer

config = yaml.safe_load(open('app/energy_agent/config.yaml'))

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
        
        self.original_state_dim = 17 + 14 + (self.max_cells * 12)
        
        # 10 global features + 2*max_cells local features
        self.state_dim = self.original_state_dim + 10 + 2*self.max_cells
        
        # Power ratio for each cell
        self.action_dim = self.max_cells
                
        self.actor_hidden_state = None
        self.critic_hidden_state = None
                
        # PPO hyperparameters
        self.gamma = config['gamma']
        self.lambda_gae = config['lambda_gae']
        self.clip_epsilon = config['clip_epsilon']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        self.buffer_size = config['buffer_size']
        self.hidden_dim = config['hidden_dim']
        
        # Normalization parameters, use original state dimension and n_cells for state normalizer
        self.state_normalizer = StateNormalizer(self.original_state_dim, n_cells=self.n_cells)
        
        # use augmented state dimension for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
                
        # reward coefficients
        self.energy_coeff = config['energy_coeff']
        self.drop_magnitude_penalty_coef = config['drop_magnitude_penalty_coef']
        self.latency_magnitude_penalty_coef = config['latency_magnitude_penalty_coef']
        self.improvement_coeff = config['improvement_coeff']
        self.violation_event_penalty = config['violation_event_penalty']
        self.cpu_magnitude_penalty_coef = config['cpu_magnitude_penalty_coef']
        self.prb_magnitude_penalty_coef = config['prb_magnitude_penalty_coef']
        self.energy_consumption_penalty = config['energy_consumption_penalty']
        self.entropy_coef = config.get('entropy_coef', 0.001)
        self.training_mode = config['training_mode']
        self.checkpoint_path = config['checkpoint_path']
        self.chunk_len = config['chunk_len']
        # Experience buffer
        self.buffer = TrajectoryBuffer()
        
        self.step_per_episode = config['buffer_size']
        self.total_episodes = int(self.max_time / self.step_per_episode)
        self.current_episode = 1
        
        self.current_padded_action = np.ones(self.max_cells) * 0.7
        self.last_padded_action = self.current_padded_action
        self.last_padded_action = np.zeros(self.action_dim)
        self.last_log_prob = np.zeros(1)
        
        # Metrics tracking
        self.metrics = {'drop_rate': [], 'latency': [], 'energy_efficiency_reward': [], 'drop_penalty': [], 
                       'latency_penalty': [], 'cpu_penalty': [], 'prb_penalty': [], 'drop_improvement': [], 
                       'latency_improvement': [], 'total_reward': [], 'actor_loss': [], 'critic_loss': [], 'entropy': []}
        
        self.episodic_metrics = {'total_reward': 0.0, 
                                 'drop_penalty': 0.0, 'latency_penalty': 0.0, 
                                 'cpu_penalty': 0.0, 'prb_penalty': 0.0, 'drop_improvement': 0.0, 
                                 'latency_improvement': 0.0, 'energy_consumption_penalty': 0.0, 'energy_efficiency_reward': 0.0}
                
        self.setup_logging(log_file)
        
        if config['use_gpu'] and not torch.cuda.is_available():
            self.logger.warning("GPU requested but CUDA not available, using CPU instead")
        
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
        
        if os.path.exists(self.checkpoint_path):
            self.load_model(self.checkpoint_path)
            self.logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            self.logger.info(f"Fine-tuning with learning rate 3e-5 for actor and 1e-4 for critic")
            self.actor_optimizer.param_groups[0]['lr'] = 3e-5
            self.critic_optimizer.param_groups[0]['lr'] = 1e-4
        else:
            self.logger.info(f"No checkpoint found at {self.checkpoint_path}")
        
    def normalize_augmented_features(self, raw_new_features):
        """Normalizes only the newly created features using predefined bounds."""

        bounds = {
            'ue_density': [0, 50],
            'isd': [0, 2000],
            'base_power': [0, 1500],
            'dist_to_drop_thresh': [-10, 10],
            'dist_to_latency_thresh': [-100, 100],
            'dist_to_cpu_thresh': [-100, 100],
            'dist_to_prb_thresh': [-100, 100],
            'load_per_active_cell': [0, 5000],
            'power_efficiency': [0, 1000],
            'load_deltas': [-500, 500],
            'ue_deltas': [-50, 50],
            'energy_consumption': [0, 0.01],
        }

        def normalize(val, min_v, max_v):
            val = np.clip(val, min_v, max_v)
            return (val - min_v) / (max_v - min_v + 1e-8)

        global_features = raw_new_features[:10]
        load_deltas = raw_new_features[10 : 10 + self.n_cells]
        ue_deltas = raw_new_features[10 + self.n_cells :]

        norm_global = np.array([
            normalize(global_features[0], *bounds['ue_density']),
            normalize(global_features[1], *bounds['isd']),
            normalize(global_features[2], *bounds['base_power']),
            normalize(global_features[3], *bounds['dist_to_drop_thresh']),
            normalize(global_features[4], *bounds['dist_to_latency_thresh']),
            normalize(global_features[5], *bounds['dist_to_cpu_thresh']),
            normalize(global_features[6], *bounds['dist_to_prb_thresh']),
            normalize(global_features[7], *bounds['load_per_active_cell']),
            normalize(global_features[8], *bounds['power_efficiency']),
            normalize(global_features[9], *bounds['energy_consumption']),
        ])
        
        norm_load_deltas = normalize(load_deltas, *bounds['load_deltas'])
        norm_ue_deltas = normalize(ue_deltas, *bounds['ue_deltas'])

        return norm_global, norm_load_deltas, norm_ue_deltas
    
    def augment_state(self, current_state_raw, normalized_padded_state):
        """
        Create new features from raw state to provide context and objectives for the agent.
        
        Args:
            current_state_raw (np.ndarray): Raw state vector, not normalized, not padded.
            normalized_padded_state (np.ndarray): Normalized and padded state vector.

        Returns:
            np.ndarray: Augmented state vector
        """
        # --- 1. Analyze raw state ---
        NETWORK_START_IDX = 17
        CELL_FEATURES_START_IDX = 31 # 17 (sim) + 14 (net)
        CELL_FEATURE_COUNT = 12

        # Global features from simulation_features
        total_ues_config = current_state_raw[1]
        isd = current_state_raw[6]
        base_power = current_state_raw[9]
        drop_threshold = current_state_raw[11]
        latency_threshold = current_state_raw[12]
        cpu_threshold = current_state_raw[13]
        prb_threshold = current_state_raw[14]

        # Global features from network_features
        active_cells = current_state_raw[NETWORK_START_IDX + 1]
        current_drop_rate = current_state_raw[NETWORK_START_IDX + 2]
        current_latency = current_state_raw[NETWORK_START_IDX + 3]
        total_traffic = current_state_raw[NETWORK_START_IDX + 4]
        total_tx_power = current_state_raw[NETWORK_START_IDX + 12]

        # Analyze data from each cell
        max_cpu_usage = 0
        max_prb_usage = 0
        avg_cell_load = 0
        avg_cell_ues = 0

        CELL_FEATURE_COUNT = 12
        CPU_FEATURE_IDX = CELL_FEATURES_START_IDX
        PRB_FEATURE_IDX = CELL_FEATURES_START_IDX + CELL_FEATURE_COUNT
        LOAD_FEATURE_IDX = CELL_FEATURES_START_IDX + 2 * CELL_FEATURE_COUNT
        UE_FEATURE_IDX = CELL_FEATURES_START_IDX + 4 * CELL_FEATURE_COUNT
        max_cpu_usage = np.max(current_state_raw[CPU_FEATURE_IDX:CPU_FEATURE_IDX + self.n_cells])
        max_prb_usage = np.max(current_state_raw[PRB_FEATURE_IDX:PRB_FEATURE_IDX + self.n_cells])
        all_cell_loads = current_state_raw[LOAD_FEATURE_IDX:LOAD_FEATURE_IDX + self.n_cells]
        all_cell_ues = current_state_raw[UE_FEATURE_IDX:UE_FEATURE_IDX + self.n_cells]
        avg_cell_load = np.mean(all_cell_loads)
        avg_cell_ues = np.mean(all_cell_ues)

        # --- 2. Create new features ---
        # UE density, a golden metric to distinguish Rural and Urban/Indoor
        ue_density = total_ues_config / (self.max_cells + 1e-6)

        # Distance to dangerous zones
        dist_to_drop_thresh = drop_threshold - current_drop_rate
        dist_to_latency_thresh = latency_threshold - current_latency
        dist_to_cpu_thresh = cpu_threshold - max_cpu_usage
        dist_to_prb_thresh = prb_threshold - max_prb_usage
        
        # Network efficiency
        load_per_active_cell = total_traffic / (active_cells + 1e-6)
        # Load-Power efficiency: how much traffic is served per Watt of transmit power
        power_efficiency = total_traffic / (total_tx_power + 1e-6)
        
        # Energy consumption
        # Convert power (W) to energy (kWh)
        time_step = current_state_raw[3]  # seconds
        current_energy = current_state_raw[NETWORK_START_IDX + 0]
        energy_consumption = (current_energy / 1000) * (time_step / 3600)
        
        # Local correlation features (Hotspot/Coldspot)
        load_deltas = all_cell_loads - avg_cell_load
        ue_deltas = all_cell_ues - avg_cell_ues

        new_features = np.concatenate([
            np.array([
            ue_density, isd, base_power, 
            dist_to_drop_thresh, dist_to_latency_thresh, dist_to_cpu_thresh, dist_to_prb_thresh,
            load_per_active_cell, power_efficiency, energy_consumption
            ]), 
            load_deltas, 
            ue_deltas
        ])
        
        # normalize new features
        norm_global, norm_load_deltas, norm_ue_deltas = self.normalize_augmented_features(new_features)
        # padding load_deltas and ue_deltas to max_cells length
        padded_deltas = np.zeros(self.max_cells - self.n_cells)
        norm_load_deltas = np.concatenate([norm_load_deltas, padded_deltas])
        norm_ue_deltas = np.concatenate([norm_ue_deltas, padded_deltas])
        # --- 3. Assemble final State Vector ---
        
        # Create new feature array
        augmented_state = np.concatenate([
            normalized_padded_state,
            norm_global,
            norm_load_deltas,
            norm_ue_deltas
        ])
        
        # Concatenate all: original state + new global features + new local features
        return augmented_state
    
    def pad_state(self, state):
        """
        Pad state for feature-block layout:
        state layout:
        [ simulation(17), network(14), f0_all_cells(n), f1_all_cells(n), ..., f11_all_cells(n) ]
        After padding, each feature block has length max_cells.

        Args:
            state: 1D array-like (length = 17+14 + n_cells*12)
            n_cells: actual number of cells (<= self.max_cells)
        Returns:
            padded_state: 1D np.array length = 17 + 14 + self.max_cells * 12
        """
        state = np.asarray(state).flatten().astype(np.float32)

        SIM_CNT = 17
        NET_CNT = 14
        CELL_FEATURE_COUNT = 12
        max_cells = int(self.max_cells)
        n = int(self.n_cells)

        # basic checks
        if n < 0 or n > max_cells:
            raise ValueError(f"n_cells must be 0 <= n_cells <= max_cells ({max_cells}), got {n}")

        expected_cur_len = SIM_CNT + NET_CNT + n * CELL_FEATURE_COUNT
        if state.size != expected_cur_len:
            raise ValueError(f"state length mismatch: got {state.size}, expected {expected_cur_len} "
                            "(check CELL_START_IDX, n_cells and layout)")

        # head (simulation + network) stay the same
        head = state[: SIM_CNT + NET_CNT ]

        # start index for cell feature blocks
        base = SIM_CNT + NET_CNT

        pad_each = max_cells - n
        if pad_each == 0:
            return state  # already at max

        padded_blocks = []
        for feat_idx in range(CELL_FEATURE_COUNT):
            start = base + feat_idx * n
            end = start + n
            block = state[start:end]
            if n == 0:
                # block is empty -> just zeros
                padded_block = np.zeros(max_cells, dtype=np.float32)
            else:
                padded_block = np.concatenate([block, np.zeros(pad_each, dtype=np.float32)])
            padded_blocks.append(padded_block)

        # concatenate blocks in feature order: f0_all_cells_padded, f1_all_cells_padded, ...
        cells_padded = np.concatenate(padded_blocks, axis=0)

        padded_state = np.concatenate([head, cells_padded], axis=0)
        return padded_state

    
    def normalize_state(self, state):
        """Normalize state vector to [0, 1] range"""
        return self.state_normalizer.normalize(state)
    
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
        print("Starting scenario...")
    
    def end_scenario(self):
        print("Ending scenario...")
        self.save_plots()
        self.save_model(self.checkpoint_path)
    
    def start_episode(self):
        print(f"Starting episode: {self.current_episode}")
        self.episodic_metrics['total_reward'] = 0.0
        self.episodic_metrics['drop_penalty'] = 0.0
        self.episodic_metrics['latency_penalty'] = 0.0
        self.episodic_metrics['cpu_penalty'] = 0.0
        self.episodic_metrics['prb_penalty'] = 0.0
        self.episodic_metrics['drop_improvement'] = 0.0
        self.episodic_metrics['latency_improvement'] = 0.0
        self.episodic_metrics['energy_efficiency_reward'] = 0.0
        self.episodic_metrics['energy_consumption_penalty'] = 0.0
        
        self.actor_hidden_state = None
        self.critic_hidden_state = None
        
        self.last_padded_action = np.ones(self.max_cells) * 0.7
        self.current_padded_action = self.last_padded_action.copy()

    
    def end_episode(self):
        self.logger.info(f"Episode ={self.current_episode},\n"
                         f"Episode Reward={self.episodic_metrics['total_reward'] / self.buffer_size :.2f},\n"
                         f"Drop Penalty={self.episodic_metrics['drop_penalty'] / self.buffer_size:.2f},\n"
                         f"Latency Penalty={self.episodic_metrics['latency_penalty'] / self.buffer_size:.2f},\n"
                         f"CPU Penalty={self.episodic_metrics['cpu_penalty'] / self.buffer_size:.2f},\n"
                         f"PRB Penalty={self.episodic_metrics['prb_penalty'] / self.buffer_size:.2f},\n"
                         f"Drop Improvement={self.episodic_metrics['drop_improvement'] / self.buffer_size:.2f},\n"
                         f"Latency Improvement={self.episodic_metrics['latency_improvement'] / self.buffer_size:.2f},\n"
                         f"Energy Efficiency Reward={self.episodic_metrics['energy_efficiency_reward'] / self.buffer_size:.2f},\n"
                         f"Energy Consumption Penalty={self.episodic_metrics['energy_consumption_penalty'] / self.buffer_size:.2f},\n"
        )        
        self.train()
        self.current_episode += 1
        
        if self.current_episode < self.total_episodes:
            self.start_episode()
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def get_action(self, state):
        """
        Get action from policy network
        
        Args:
            state: State vector from MATLAB interface
            
        Returns:
            action: Power ratios for each cell [0, 1]
        """

        raw_state = np.array(state).flatten()   
        normalized_state = self.normalize_state(raw_state)
        normalized_padded_state = self.pad_state(normalized_state)
        state = self.augment_state(raw_state, normalized_padded_state)
                
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            
            action_mean, action_logstd, next_actor_hidden = self.actor(state_tensor, self.actor_hidden_state)
            self.actor_hidden_state = next_actor_hidden
            
            if self.training_mode:
                # Sample from policy during training
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
            else:
                # Use mean during evaluation
                action = action_mean
        
        # Clamp actions to [0, 1] range
        action = torch.clamp(action, 0.0, 1.0)
        
        if self.training_mode:
            with torch.no_grad():
                log_prob_per_dim = dist.log_prob(action)  # shape (1, action_dim)
                # mask active dims
                active_count = float(self.n_cells)
                avg_log_prob = (log_prob_per_dim[:, :self.n_cells].sum(dim=-1) / (active_count + 1e-8)).cpu().numpy().flatten()[0]        
        else:
            avg_log_prob = 0.0
        
        # update current padded action
        self.current_padded_action = action.cpu().numpy().flatten()
        
        # truncate action to n_cells length
        truncated_action = action.squeeze()[:self.n_cells]
        
        # Store for experience replay
        self.last_padded_state = state
        self.last_log_prob = np.array([avg_log_prob])
        
        # return truncated action for environment
        return truncated_action.cpu().numpy().flatten()
    
    ## OPTIONAL: Modify reward calculation as needed
    def calculate_reward(self, prev_state, action, current_state):
        """
        Calculate reward based on energy efficiency and QoS constraints.
        
        Args:
            prev_state: Previous state vector
            action: Action taken (power ratios)
            current_state: Current state vector
            
        Returns:
            float: Calculated reward value
        """
        if prev_state is None:
            return 0.0

        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()

        # State structure indices
        CELL_START_IDX = 17 + 14
        NETWORK_START_IDX = 17
                
        # Extract current and previous metrics
        current_energy = current_state[NETWORK_START_IDX + 0]
        prev_energy = prev_state[NETWORK_START_IDX + 0]
        
        current_drop_rate = current_state[NETWORK_START_IDX + 2]
        prev_drop_rate = prev_state[NETWORK_START_IDX + 2]
        
        current_latency = current_state[NETWORK_START_IDX + 3]
        prev_latency = prev_state[NETWORK_START_IDX + 3]

        CELL_FEATURE_COUNT = 12
        CPU_FEATURE_IDX = CELL_START_IDX
        PRB_FEATURE_IDX = CELL_START_IDX + CELL_FEATURE_COUNT
        max_cpu_usage = np.max(current_state[CPU_FEATURE_IDX:CPU_FEATURE_IDX + self.n_cells])
        max_prb_usage = np.max(current_state[PRB_FEATURE_IDX:PRB_FEATURE_IDX + self.n_cells])
        
        # Extract thresholds from simulation features
        drop_threshold = current_state[11]  # dropCallThreshold
        latency_threshold = current_state[12]  # latencyThreshold
        cpu_threshold = current_state[13]
        prb_threshold = current_state[14]
        
        # --- Energy Reward ---
        energy_efficiency = prev_energy - current_energy 
        energy_efficiency_reward = self.energy_coeff * energy_efficiency
        
        # Convert power (W) to energy (kWh)
        time_step = current_state[3]  # seconds
        energy_consumption = (current_energy / 1000) * (time_step / 3600)
        energy_consumption_penalty = self.energy_consumption_penalty * energy_consumption

        # --- Penalty for Drop Rate (Combined) ---
        drop_penalty, latency_penalty, cpu_penalty, prb_penalty = 0.0, 0.0, 0.0, 0.0

        if current_drop_rate > drop_threshold:
            excess = current_drop_rate - drop_threshold
            drop_penalty = self.violation_event_penalty - self.drop_magnitude_penalty_coef * (excess**2)

        if current_latency > latency_threshold:
            excess = current_latency - latency_threshold
            latency_penalty = self.violation_event_penalty - self.latency_magnitude_penalty_coef * (excess**2)
            
        if max_cpu_usage > cpu_threshold:
            excess = max_cpu_usage - cpu_threshold
            cpu_penalty = self.violation_event_penalty - self.cpu_magnitude_penalty_coef * (excess**2)

        if max_prb_usage > prb_threshold:
            excess = max_prb_usage - prb_threshold
            prb_penalty = self.violation_event_penalty - self.prb_magnitude_penalty_coef * (excess**2)

        # --- Reward for Improvement (MODIFIED V2) ---

        # 1. Define "Safe Zones" (e.g., 80% or 90% of the threshold)
        safe_drop_threshold = drop_threshold * 0.9
        safe_latency_threshold = latency_threshold * 0.9

        # 2. Calculate improvement value as before
        drop_improvement_val = prev_drop_rate - current_drop_rate
        latency_improvement_val = (prev_latency - current_latency) * 0.1
        
        drop_improvement = 0.0
        latency_improvement = 0.0

        # 3. LOGIC MỚI: Chỉ tính 'improvement' (thưởng hoặc phạt)
        #    nếu state (trước hoặc sau) nằm ngoài vùng an toàn.
        
        if (prev_drop_rate > safe_drop_threshold) or (current_drop_rate > safe_drop_threshold):
            # Nếu state trước (prev) KHÔNG an toàn, HOẶC state hiện tại (current) KHÔNG an toàn
            # thì chúng ta mới "quan tâm" đến việc di chuyển (improvement/degradation).
            drop_improvement = self.improvement_coeff * drop_improvement_val
        
        # Ngược lại (else):
        # Nếu cả prev_drop_rate VÀ current_drop_rate đều NẰM TRONG vùng an toàn
        # (ví dụ: 0.5% và 0.8%, đều < 0.9%)
        # thì drop_improvement = 0.0.
        # Agent sẽ không bị phạt khi đi từ 0.5% -> 0.8%.
        
        if (prev_latency > safe_latency_threshold) or (current_latency > safe_latency_threshold):
            latency_improvement = self.improvement_coeff * latency_improvement_val

        # --- Total Reward ---
        total_reward = (
            energy_efficiency_reward
            + drop_penalty
            + latency_penalty
            + cpu_penalty
            + prb_penalty
            + drop_improvement
            + latency_improvement
            + energy_consumption_penalty
        )
        
        # Update episodic metrics (accumulated from all envs)
        self.episodic_metrics['total_reward'] += total_reward
        self.episodic_metrics['drop_penalty'] += drop_penalty
        self.episodic_metrics['latency_penalty'] += latency_penalty
        self.episodic_metrics['cpu_penalty'] += cpu_penalty
        self.episodic_metrics['prb_penalty'] += prb_penalty
        self.episodic_metrics['drop_improvement'] += drop_improvement
        self.episodic_metrics['latency_improvement'] += latency_improvement
        self.episodic_metrics['energy_consumption_penalty'] += energy_consumption_penalty
        self.episodic_metrics['energy_efficiency_reward'] += energy_efficiency_reward
        
        self.metrics['drop_rate'].append(current_drop_rate)
        self.metrics['latency'].append(current_latency)
        self.metrics['energy_efficiency_reward'].append(energy_efficiency_reward)
        self.metrics['drop_penalty'].append(drop_penalty)
        self.metrics['latency_penalty'].append(latency_penalty)
        self.metrics['cpu_penalty'].append(cpu_penalty)
        self.metrics['prb_penalty'].append(prb_penalty)
        self.metrics['drop_improvement'].append(drop_improvement)
        self.metrics['latency_improvement'].append(latency_improvement)
        self.metrics['total_reward'].append(total_reward)

        return total_reward
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def update(self, state, action, next_state, done):
        """
        Update agent with experience
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
        """
        if not self.training_mode:
            return
        
        # pad action to max_cells length from raw action vector
        padded_action = self.current_padded_action
        
        # Calculate actual reward using raw state
        actual_reward = self.calculate_reward(state, padded_action, next_state)
        
        # Convert inputs to numpy if needed
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(padded_action, 'numpy'):
            padded_action = padded_action.numpy()
        
        
        # Ensure proper shapes
        raw_state = np.array(state).copy().flatten()
        normalized_state = self.normalize_state(raw_state)
        normalized_padded_state = self.pad_state(normalized_state)
        state = self.augment_state(raw_state, normalized_padded_state)
        action = np.array(padded_action).flatten()
        
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value_tensor, next_critic_hidden = self.critic(state_tensor, self.critic_hidden_state)
            value = value_tensor.item()
            
            self.critic_hidden_state = next_critic_hidden
        
        # Create transition
        transition = Transition(
            state=state,
            action=action,
            reward=actual_reward,
            done=done,
            log_prob=getattr(self, 'last_log_prob', np.array([0.0]))[0],
            value=value
        )
                
        self.buffer.add(transition)
        
        # update last padded action
        self.last_padded_action = self.current_padded_action
        
        if len(self.buffer) >= self.buffer_size and self.training_mode:
            self.end_episode()
            
    
    def compute_gae_for_trajectory(self, rewards, values, dones, next_value):
            """Compute GAE for a single trajectory."""
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            
            # Thêm next_value vào cuối để tính GAE dễ hơn
            extended_values = torch.cat([values, next_value.unsqueeze(0)])
            
            for t in reversed(range(len(rewards))):
                non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * extended_values[t+1] * non_terminal - extended_values[t]
                advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * non_terminal * last_advantage
            
            returns = advantages + values
            return advantages, returns
    
    def train(self):
        # Sequence-aware PPO training (chunked TBPTT) for GRU actor/critic
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get_all_and_clear()

        if len(states) == 0:
            return

        # Bootstrap value for the last state (decoupled from online hidden state)
        last_state = torch.FloatTensor(states[-1]).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value, _ = self.critic(last_state, None)
            next_value = next_value.squeeze()

        # Tensors for GAE
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Compute GAE over the collected trajectory(ies)
        advantages, returns = self.compute_gae_for_trajectory(rewards_tensor, values_tensor, dones_tensor, next_value)

        # Move to numpy for chunking
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        oldlp_np = np.asarray(old_log_probs, dtype=np.float32)
        adv_np = advantages.detach().cpu().numpy().astype(np.float32)
        ret_np = returns.detach().cpu().numpy().astype(np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)

        # Normalize advantages globally
        if len(adv_np) > 1:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        else:
            adv_np = adv_np - adv_np.mean()

        # Segment by episode boundaries using dones
        traj_bounds = []
        start_idx = 0
        for t in range(len(states_np)):
            is_last = (t == len(states_np) - 1)
            if is_last or dones_np[t] > 0.5:
                end_idx = t + 1
                if end_idx - start_idx > 0:
                    traj_bounds.append((start_idx, end_idx))
                start_idx = end_idx

        # Build fixed-length chunks (drop remainders)
        CHUNK_LEN = min(self.chunk_len, len(states_np))
        chunks = []  # list of tuples (states[L,D], actions[L,A], oldlp[L], adv[L], ret[L])
        for (s_idx, e_idx) in traj_bounds:
            seg_len = e_idx - s_idx
            n_full = seg_len // CHUNK_LEN
            for k in range(n_full):
                a = s_idx + k * CHUNK_LEN
                b = a + CHUNK_LEN
                chunks.append((
                    states_np[a:b],
                    actions_np[a:b],
                    oldlp_np[a:b],
                    adv_np[a:b],
                    ret_np[a:b],
                ))

        if not chunks:
            return

        rng = np.random.default_rng()

        avg_actor_loss, avg_critic_loss = 0.0, 0.0
        for epoch in range(self.ppo_epochs):
            rng.shuffle(chunks)
            epoch_actor_losses = []
            epoch_critic_losses = []

            # Number of chunks per minibatch equals PPO minibatch size (counted by chunks)
            chunks_per_batch = max(1, int(self.batch_size))
            for i in range(0, len(chunks), chunks_per_batch):
                batch = chunks[i:i + chunks_per_batch]

                # Stack to tensors: (B, L, ...)
                bs = torch.FloatTensor(np.stack([c[0] for c in batch], axis=0)).to(self.device)
                ba = torch.FloatTensor(np.stack([c[1] for c in batch], axis=0)).to(self.device)
                bolp = torch.FloatTensor(np.stack([c[2] for c in batch], axis=0)).to(self.device)
                badv = torch.FloatTensor(np.stack([c[3] for c in batch], axis=0)).to(self.device)
                bret = torch.FloatTensor(np.stack([c[4] for c in batch], axis=0)).to(self.device)

                seq_len = bs.size(1)
                # Explicitly reshape PPO tensors to (B, L)
                bolp = bolp.view(bs.size(0), seq_len)
                badv = badv.view(bs.size(0), seq_len)
                bret = bret.view(bs.size(0), seq_len)

                # Actor forward through time to get per-timestep log-probs
                hidden_actor = None
                step_log_probs = []  # list of (B,)
                step_entropies = []  # list of (B,)
                for t in range(seq_len):
                    inp_t = bs[:, t:t+1, :]
                    mean_t, logstd_t, hidden_actor = self.actor(inp_t, hidden_actor)
                    if isinstance(hidden_actor, tuple):
                        hidden_actor = tuple(h.detach() for h in hidden_actor)
                    else:
                        hidden_actor = hidden_actor.detach()
                    logstd_t = torch.clamp(logstd_t, min=-20.0, max=2.0)
                    dist_t = torch.distributions.Normal(mean_t, torch.exp(logstd_t))

                    log_probs_all_t = dist_t.log_prob(ba[:, t, :])  # (B, action_dim)
                    mask = torch.zeros_like(log_probs_all_t)
                    mask[:, :self.n_cells] = 1.0
                    active_counts = mask.sum(dim=-1).clamp(min=1.0)
                    avg_log_prob_t = (log_probs_all_t * mask).sum(dim=-1) / active_counts  # (B,)
                    step_log_probs.append(avg_log_prob_t)

                    entropy_all_t = dist_t.entropy()  # (B, action_dim)
                    avg_entropy_t = (entropy_all_t * mask).sum(dim=-1) / active_counts  # (B,)
                    step_entropies.append(avg_entropy_t)

                new_log_probs = torch.stack(step_log_probs, dim=1)  # (B, L)
                entropy = torch.stack(step_entropies, dim=1)  # (B, L)
                # Track mean entropy for monitoring
                self.metrics['entropy'].append(float(entropy.mean().detach().cpu().item()))

                # Ensure shapes match for PPO terms
                bolp = bolp.view_as(new_log_probs)
                badv = badv.view_as(new_log_probs)
                
                log_ratio = new_log_probs - bolp
                log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
                ratio = torch.exp(log_ratio)

                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * badv
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy.mean()

                # Critic forward through time to get per-timestep values
                hidden_critic = None
                value_steps = []
                for t in range(seq_len):
                    inp_t = bs[:, t:t+1, :]
                    value_t, hidden_critic = self.critic(inp_t, hidden_critic)
                    value_steps.append(value_t.squeeze(-1))  # (B,)
                values_pred = torch.stack(value_steps, dim=1)  # (B, L)

                critic_loss = 0.5 * nn.MSELoss()(values_pred, bret)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                # Detach/reset hidden states after each minibatch update
                if isinstance(hidden_actor, tuple):
                    hidden_actor = tuple(h.detach() for h in hidden_actor)
                elif hidden_actor is not None:
                    hidden_actor = hidden_actor.detach()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                if isinstance(hidden_critic, tuple):
                    hidden_critic = tuple(h.detach() for h in hidden_critic)
                elif hidden_critic is not None:
                    hidden_critic = hidden_critic.detach()

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())

            avg_actor_loss = float(np.mean(epoch_actor_losses)) if epoch_actor_losses else 0.0
            avg_critic_loss = float(np.mean(epoch_critic_losses)) if epoch_critic_losses else 0.0
            self.logger.info(f"Epoch {epoch+1}/{self.ppo_epochs}: Actor loss={avg_actor_loss:.4f}, Critic loss={avg_critic_loss:.4f}")

        # Track losses
        self.metrics['actor_loss'].append(avg_actor_loss)
        self.metrics['critic_loss'].append(avg_critic_loss)
        self.logger.info(f"Training completed: Actor loss={avg_actor_loss:.4f}, Critic loss={avg_critic_loss:.4f}")
    
    def save_model(self, filepath=None):
        """Save model parameters"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.logger.info(f"Training mode set to {training}")
    
    def save_plots(self):
        """Save metrics plots"""
        if not any(self.metrics.values()):
            return
        
        def ma(data, window=20):
            if len(data) < window: return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Episode {self.total_episodes} Metrics')
        
        # Entropy and Energy metrics
        if self.metrics['entropy']:
            axes[0,0].plot(self.metrics['entropy'], alpha=0.3, label='Entropy')
            axes[0,0].plot(ma(self.metrics['entropy']), label='Entropy MA', linewidth=2)
            axes[0,0].set_ylabel('Entropy'); axes[0,0].legend()
        
        # Reward components
        if self.metrics['energy_efficiency_reward']:
            axes[0,1].plot(ma(self.metrics['energy_efficiency_reward']), label='Energy Efficiency Reward')
            axes[0,1].plot(ma(self.metrics['drop_penalty']), label='Drop Penalty')
            axes[0,1].plot(ma(self.metrics['latency_penalty']), label='Latency Penalty')
            axes[0,1].set_ylabel('Reward Components'); axes[0,1].legend()
        
        # Total reward
        if self.metrics['total_reward']:
            axes[1,0].plot(self.metrics['total_reward'], alpha=0.3)
            axes[1,0].plot(ma(self.metrics['total_reward']), linewidth=2)
            axes[1,0].set_ylabel('Total Reward'); axes[1,0].set_xlabel('Step')
        
        # Losses
        if self.metrics['actor_loss']:
            axes[1,1].plot(self.metrics['actor_loss'], alpha=0.3, label='Actor')
            axes[1,1].plot(self.metrics['critic_loss'], alpha=0.3, label='Critic')
            if len(self.metrics['actor_loss']) >= 5:
                axes[1,1].plot(ma(self.metrics['actor_loss'], 5), label='Actor MA', linewidth=2)
                axes[1,1].plot(ma(self.metrics['critic_loss'], 5), label='Critic MA', linewidth=2)
            axes[1,1].set_ylabel('Loss'); axes[1,1].set_xlabel('Training Step'); axes[1,1].legend()
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/episode_{timestamp}.png', dpi=300)
        plt.close()
        self.logger.info(f"Plots saved to plots/episode_{timestamp}.png")