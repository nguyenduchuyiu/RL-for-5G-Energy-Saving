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

from .transition import Transition, TransitionBuffer
from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        
        # 9 global features + 5 delta features + 2*max_cells local features
        self.state_dim = self.original_state_dim + 9 + 5 + 2*self.max_cells
        
        # Power ratio for each cell
        self.action_dim = self.max_cells
                                
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
        self.training_mode = config['training_mode']
        self.checkpoint_path = config['checkpoint_path']
        # Experience buffer
        self.buffer = TransitionBuffer(capacity=self.buffer_size)
        
        self.step_per_episode = config['buffer_size']
        self.total_episodes = int(self.max_time / self.step_per_episode)
        self.current_episode = 1
        
        # Action and log_prob tracking
        self.current_padded_action = np.ones(self.max_cells) * 0.7
        self.last_log_prob = 0
        
        # Track previous state for delta features (temporal info)
        self.prev_states = None
        
        # Metrics tracking
        self.metrics = {'drop_rate': [], 'latency': [], 'cpu': [], 'prb': [], 'energy_efficiency_reward': [], 
                       'drop_improvement': [], 'latency_improvement': [], 'cpu_improvement': [], 'prb_improvement': [],
                       'stability_penalty': [], 'energy_consumption_penalty': [], 'violation_penalty': [], 'warning_reward': [],
                       'total_reward': [], 'actor_loss': [], 'critic_loss': [], 'entropy': []}
        
                
        self.setup_logging(log_file)
        
        if config['use_gpu'] and not torch.cuda.is_available():
            self.logger.warning("GPU requested but CUDA not available, using CPU instead")
        
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
        
        if os.path.exists(self.checkpoint_path):
            self.load_model(self.checkpoint_path)
            self.logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            # self.logger.info(f"Fine-tuning with learning rate 3e-5 for actor and 1e-4 for critic")
            # self.actor_optimizer.param_groups[0]['lr'] = 3e-5
            # self.critic_optimizer.param_groups[0]['lr'] = 1e-4
        else:
            self.logger.info(f"No checkpoint found at {self.checkpoint_path}")
    
    def normalize_augmented_features(self, global_features, 
                                     load_deltas, ue_deltas, 
                                     delta_features):
        """Normalizes only the newly created features using predefined bounds.
        
        Uses consistent bounds with StateNormalizer where applicable.
        """

        bounds = {
            # Reuse from StateNormalizer.simulation_bounds
            'ue_density': [0, 20],          
            'isd': [100, 2000],                 # Same as StateNormalizer
            'base_power': [100, 1000],          # Same as StateNormalizer
            
            # Distance to thresholds (new features)
            'dist_to_drop_thresh': [-20, 10],  # Can go negative (over threshold)
            'dist_to_latency_thresh': [-200, 100],  # Based on latencyThreshold bounds
            'dist_to_cpu_thresh': [-30, 95],   # Based on cpuThreshold bounds  
            'dist_to_prb_thresh': [-30, 95],   # Based on prbThreshold bounds
            
            # Efficiency metrics (new features)
            'load_per_active_cell': [0, 5000/1],  # totalTraffic / 1 active cell
            'power_efficiency': [0, 5000/100],     # totalTraffic / min power
            
            # Cell deltas (spatial)
            'load_deltas': [-1000, 1000],      # Cell load deviation
            'ue_deltas': [-50, 50],            # Cell UE deviation
            
            # Temporal deltas (time-series trends) - NEW
            'drop_rate_delta': [-20, 20],     # Based on avgDropRate bounds
            'latency_delta': [-200, 200],     # Based on avgLatency bounds  
            'energy_delta': [-5000, 5000],    # Based on totalEnergy bounds
            'active_cells_delta': [-50, 50],  # Based on activeCells bounds
            'energy_consumption_delta': [-1, 1],
        }

        def normalize(val, min_v, max_v):
            val = np.clip(val, min_v, max_v)
            return (val - min_v) / (max_v - min_v + 1e-8)

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
        ])
        
        norm_load_deltas = normalize(load_deltas, *bounds['load_deltas'])
        norm_ue_deltas = normalize(ue_deltas, *bounds['ue_deltas'])
        
        norm_delta_features = np.array([
            normalize(delta_features[0], *bounds['drop_rate_delta']),
            normalize(delta_features[1], *bounds['latency_delta']),
            normalize(delta_features[2], *bounds['energy_delta']), 
            normalize(delta_features[3], *bounds['active_cells_delta']),
            normalize(delta_features[4], *bounds['energy_consumption_delta'])
        ])

        return norm_global, norm_load_deltas, norm_ue_deltas, norm_delta_features
    
    def augment_state(self, current_state_raw, normalized_padded_state, prev_state_raw=None):
        """
        Create new features from raw state to provide context and objectives for the agent.
        
        Args:
            current_state_raw (np.ndarray): Raw state vector, not normalized, not padded.
            normalized_padded_state (np.ndarray): Normalized and padded state vector.
            prev_state_raw (np.ndarray): Previous raw state for delta computation

        Returns:
            np.ndarray: Augmented state vector
        """
        # --- 1. Analyze raw state ---
        NETWORK_START_IDX = 17
        CELL_FEATURES_START_IDX = 31 # 17 (sim) + 14 (net)

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

        CPU_FEATURE_IDX = CELL_FEATURES_START_IDX
        PRB_FEATURE_IDX = CELL_FEATURES_START_IDX + self.n_cells
        LOAD_FEATURE_IDX = CELL_FEATURES_START_IDX + 2 * self.n_cells
        UE_FEATURE_IDX = CELL_FEATURES_START_IDX + 4 * self.n_cells
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
        
        # Local correlation features (Hotspot/Coldspot)
        load_deltas = all_cell_loads - avg_cell_load
        ue_deltas = all_cell_ues - avg_cell_ues
        
        # Temporal delta features (trends) - substitute for GRU memory
        current_energy = current_state_raw[NETWORK_START_IDX + 0]
        if prev_state_raw is not None:
            prev_drop_rate = prev_state_raw[NETWORK_START_IDX + 2]
            prev_latency = prev_state_raw[NETWORK_START_IDX + 3]
            prev_energy = prev_state_raw[NETWORK_START_IDX + 0]
            prev_active_cells = prev_state_raw[NETWORK_START_IDX + 1]
            
            drop_rate_delta = current_drop_rate - prev_drop_rate
            latency_delta = current_latency - prev_latency
            energy_delta = current_energy - prev_energy
            active_cells_delta = active_cells - prev_active_cells
            # Convert power (W) to energy (kWh)
            time_step = current_state_raw[3]  # seconds
            current_tx_power = current_state_raw[NETWORK_START_IDX + 12]
            prev_tx_power = prev_state_raw[NETWORK_START_IDX + 12]
            energy_consumption_delta = (current_tx_power - prev_tx_power) / 1000 * (time_step / 3600)
        else:
            # First step, no deltas
            drop_rate_delta = 0.0
            latency_delta = 0.0
            energy_delta = 0.0
            active_cells_delta = 0.0
            energy_consumption_delta = 0.0

        delta_features = np.array([drop_rate_delta, latency_delta, energy_delta, active_cells_delta])

        global_features = np.array([
            ue_density, isd, base_power, 
            dist_to_drop_thresh, dist_to_latency_thresh, dist_to_cpu_thresh, dist_to_prb_thresh,
            load_per_active_cell, power_efficiency
            ]) 
        delta_features = np.array([drop_rate_delta, latency_delta, energy_delta, active_cells_delta, energy_consumption_delta])
        load_deltas = np.array(load_deltas)
        ue_deltas = np.array(ue_deltas)
        
        
        # normalize new features
        (norm_global, 
         norm_load_deltas, 
         norm_ue_deltas,
         norm_delta_features) = self.normalize_augmented_features(global_features, 
                                                                  load_deltas, 
                                                                  ue_deltas, 
                                                                  delta_features)
        
        # padding load_deltas and ue_deltas to max_cells length
        padded_deltas = np.zeros(self.max_cells - self.n_cells)
        norm_load_deltas = np.concatenate([norm_load_deltas, padded_deltas])
        norm_ue_deltas = np.concatenate([norm_ue_deltas, padded_deltas])
        # --- 3. Assemble final State Vector ---
        
        # Create new feature array
        augmented_state = np.concatenate([
            normalized_padded_state,
            norm_global,
            norm_delta_features,  # Temporal info
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
        print("===================Starting scenario===================")
        self.start_episode()
    
    def end_scenario(self):
        print("===================Ending scenario===================")
        # Force the last episode to end properly
        self.force_episode_termination()
        self.save_plots()
        self.save_model(self.checkpoint_path)
    
    def force_episode_termination(self):
        """Force termination of current episode by marking last transition as done=True"""
        # Use safe API to get last transition
        last = self.buffer.get_last_n(1)
        if len(last) == 0:
            return
            
        last_transition = last[-1]
        if not last_transition.done:
            # Create a new transition with done=True
            corrected_transition = Transition(
                state=last_transition.state,
                action=last_transition.action,
                reward=last_transition.reward,
                next_state=last_transition.next_state,
                done=True,  # Force done=True for episode termination
                log_prob=last_transition.log_prob,
                value=last_transition.value
            )
            
            # Replace last transition using safe API
            flat = self.buffer.get_all_flat(include_current=True)
            flat[-1] = corrected_transition
            self.buffer.clear()
            for t in flat:
                self.buffer.add(t)
            self.logger.info("Forced episode termination: marked last transition as done=True")
        
        # Train with the corrected buffer
        if len(self.buffer) >= self.batch_size:
            self.train()
    
    def start_episode(self):
        print(f"Starting episode: {self.current_episode}")

        
        # Reset per-env states
        self.current_padded_action = np.ones(self.max_cells) * 0.7
        self.last_log_prob = 0.0

    
    def end_episode(self):
        # Ensure the last transition in buffer is marked as done=True using safe API
        last = self.buffer.get_last_n(1)
        if len(last) > 0:
            last_transition = last[-1]
            if not last_transition.done:
                # Create a new transition with done=True
                corrected_transition = Transition(
                    state=last_transition.state,
                    action=last_transition.action,
                    reward=last_transition.reward,
                    next_state=last_transition.next_state,
                    done=True,  # Force done=True for episode termination
                    log_prob=last_transition.log_prob,
                    value=last_transition.value
                )
                
                # Replace last transition using safe API
                flat = self.buffer.get_all_flat(include_current=True)
                flat[-1] = corrected_transition
                self.buffer.clear()
                for t in flat:
                    self.buffer.add(t)
        
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
        # Use prev_state to compute deltas (DON'T update yet, wait for update())
        state = self.augment_state(raw_state, normalized_padded_state, self.prev_states)
                
        # prepare tensor: shape (1, state_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)
            if self.training_mode:
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
            else:
                action = action_mean

        action = torch.clamp(action, 0.0, 1.0)

        if self.training_mode:
            with torch.no_grad():
                # compute summed log_prob over active action dims (do NOT average)
                n = int(self.n_cells)
                log_prob_per_dim = dist.log_prob(action)  # shape (1, action_dim)
                log_prob_sum = float(log_prob_per_dim[0, :n].sum().cpu().numpy())
        else:
            log_prob_sum = 0.0

        self.current_padded_action = action.cpu().numpy().flatten()
        truncated_action = action.squeeze()[:self.n_cells]
        self.last_log_prob = log_prob_sum
        if not self.training_mode:
            self.prev_states = raw_state.copy()
        return truncated_action.cpu().numpy().flatten()

    
    ## OPTIONAL: Modify reward calculation as needed
    def calculate_reward(self, prev_state, action, current_state):
        """
        Directional reward including CPU and PRB:
        - Nếu bất kỳ QoS metric (drop, latency, cpu, prb) vượt safe -> thưởng gradient (giảm metric)
        - Nếu tất cả QoS nằm trong safe -> thưởng tiết kiệm năng lượng, phạt nhẹ khi gần danger zone
        - Không có phạt cứng cho violation; chỉ thưởng hướng cải thiện / phạt nhẹ hướng ra ngoài
        """
        if prev_state is None:
            return 0.0

        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        action = np.array(action).flatten()

        # indices
        CELL_START_IDX = 17 + 14
        NETWORK_START_IDX = 17
        CELL_FEATURE_COUNT = 12

        # network-level
        prev_energy = prev_state[NETWORK_START_IDX + 0]
        current_energy = current_state[NETWORK_START_IDX + 0]

        prev_drop = prev_state[NETWORK_START_IDX + 2]
        current_drop = current_state[NETWORK_START_IDX + 2]

        prev_latency = prev_state[NETWORK_START_IDX + 3]
        current_latency = current_state[NETWORK_START_IDX + 3]

        drop_th = current_state[11]
        latency_th = current_state[12]
        cpu_th = current_state[13]
        prb_th = current_state[14]

        # per-cell maxima (current & prev)
        CPU_FEATURE_IDX = CELL_START_IDX
        PRB_FEATURE_IDX = CELL_START_IDX + self.n_cells
        max_cpu = np.max(current_state[CPU_FEATURE_IDX:CPU_FEATURE_IDX + self.n_cells])
        max_prb = np.max(current_state[PRB_FEATURE_IDX:PRB_FEATURE_IDX + self.n_cells])
        prev_max_cpu = np.max(prev_state[CPU_FEATURE_IDX:CPU_FEATURE_IDX + self.n_cells])
        prev_max_prb = np.max(prev_state[PRB_FEATURE_IDX:PRB_FEATURE_IDX + self.n_cells])

        # safe / danger cutoffs
        safe_drop = 0.9 * drop_th
        safe_latency = 0.9 * latency_th
        safe_cpu = 0.9 * cpu_th
        safe_prb = 0.9 * prb_th

        danger_drop = 0.8 * drop_th
        danger_latency = 0.8 * latency_th
        danger_cpu = 0.8 * cpu_th
        danger_prb = 0.8 * prb_th

        # gradients (positive when improve)
        drop_grad = prev_drop - current_drop
        latency_grad = (prev_latency - current_latency) / max(1e-6, latency_th)
        cpu_grad = (prev_max_cpu - max_cpu) / max(1e-6, cpu_th)
        prb_grad = (prev_max_prb - max_prb) / max(1e-6, prb_th)
        energy_grad = (prev_energy - current_energy) / max(1e-6, current_energy)

        # --- REWARD DECOMPOSITION ---
        # reward khi prev hoặc current unsafe — tức khi agent đang/đã cải thiện từ trạng thái xấu
        drop_improvement = config['qos_grad_coeff'] * drop_grad if (prev_drop > safe_drop or current_drop > safe_drop) else 0.0
        latency_improvement = config['qos_grad_coeff'] * latency_grad if (prev_latency > safe_latency or current_latency > safe_latency) else 0.0
        cpu_improvement = config['cpu_grad_coeff'] * cpu_grad if (prev_max_cpu > safe_cpu or max_cpu > safe_cpu) else 0.0
        prb_improvement = config['prb_grad_coeff'] * prb_grad if (prev_max_prb > safe_prb or max_prb > safe_prb) else 0.0

        drop_improvement = np.clip(drop_improvement, -10.0, 10.0)
        latency_improvement = np.clip(latency_improvement, -10.0, 10.0)
        cpu_improvement = np.clip(cpu_improvement, -10.0, 10.0)
        prb_improvement = np.clip(prb_improvement, -10.0, 10.0)

        energy_efficiency_reward = 0.0
        stability_pen = 0.0
        energy_consumption_penalty = 0.0
        violation_penalty = 0.0

        if (current_drop <= safe_drop) \
            and (current_latency <= safe_latency) \
            and (max_cpu <= safe_cpu) \
            and (max_prb <= safe_prb):

            # QoS safe → optimize energy
            energy_efficiency_reward = config['energy_grad_coeff'] * energy_grad + config['baseline_reward']

            # discourage moving too close to danger zone
            if current_drop > danger_drop:
                stability_pen += (current_drop - danger_drop) / max(1e-6, drop_th)
            if current_latency > danger_latency:
                stability_pen += (current_latency - danger_latency) / max(1e-6, latency_th)
            if max_cpu > danger_cpu:
                stability_pen += (max_cpu - danger_cpu) / max(1e-6, cpu_th)
            if max_prb > danger_prb:
                stability_pen += (max_prb - danger_prb) / max(1e-6, prb_th)
                
            stability_pen = config['stability_penalty'] * stability_pen

            # --- ENERGY COST ---
            time_step = current_state[3]
            energy_consumption = (current_energy / 1000) * (time_step / 3600)
            energy_consumption_penalty = -config['energy_consumption_penalty'] * energy_consumption + config['baseline_reward']

        # --- ABSOLUTE VIOLATION PENALTY (applies regardless of safe block) ---
        # Penalize how far each QoS metric exceeds its threshold (normalized)
        drop_violation = max(0.0, (current_drop - drop_th) / max(1e-6, drop_th))
        latency_violation = max(0.0, (current_latency - latency_th) / max(1e-6, latency_th))
        cpu_violation = max(0.0, (max_cpu - cpu_th) / max(1e-6, cpu_th))
        prb_violation = max(0.0, (max_prb - prb_th) / max(1e-6, prb_th))
        violation_penalty = drop_violation + latency_violation + cpu_violation + prb_violation 
        violation_penalty = config['violation_penalty'] * violation_penalty

        if prev_drop > drop_th or current_drop > drop_th:
            violation_penalty -= config['baseline_reward']
        if prev_latency > latency_th or current_latency > latency_th:
            violation_penalty -= config['baseline_reward']
        if prev_max_cpu > cpu_th or max_cpu > cpu_th:
            violation_penalty -= config['baseline_reward']
        if prev_max_prb > prb_th or max_prb > prb_th:
            violation_penalty -= config['baseline_reward']
        
        warning_reward = 0.0
        if (current_drop < drop_th and current_drop > danger_drop)\
        or (current_latency < latency_th and current_latency > danger_latency)\
        or (max_cpu < cpu_th and max_cpu > danger_cpu)\
        or (max_prb < prb_th and max_prb > danger_prb):
            warning_reward = config['baseline_reward'] / 4
        
        # --- TOTAL REWARD ---
        total_reward = (
            drop_improvement
            + latency_improvement
            + cpu_improvement
            + prb_improvement
            + energy_efficiency_reward
            + stability_pen 
            + violation_penalty 
            + warning_reward
            + energy_consumption_penalty
        )

        # --- METRICS LOGGING ---
        if np.random.random() < 0.02:
            print("total_reward: ", f"{total_reward:.5f}", 
                "\ndrop_improvement: ", f"{drop_improvement:.5f}", 
                "\nlatency_improvement: ", f"{latency_improvement:.5f}", 
                "\ncpu_improvement: ", f"{cpu_improvement:.5f}", 
                "\nprb_improvement: ", f"{prb_improvement:.5f}", 
                "\nenergy_efficiency_reward: ", f"{energy_efficiency_reward:.5f}", 
                "\nstability_penalty: ", f"{stability_pen:.5f}", 
                "\nviolation_penalty: ", f"{violation_penalty:.5f}", 
                "\nenergy_consumption_penalty: ", f"{energy_consumption_penalty:.5f}",
                "\nwarning_reward: ", f"{warning_reward:.5f}")
            
            self.metrics['drop_rate'].append(current_drop)
            self.metrics['latency'].append(current_latency)
            self.metrics['cpu'].append(max_cpu)
            self.metrics['prb'].append(max_prb)
            self.metrics['energy_efficiency_reward'].append(energy_efficiency_reward)
            self.metrics['drop_improvement'].append(drop_improvement)
            self.metrics['latency_improvement'].append(latency_improvement)
            self.metrics['cpu_improvement'].append(cpu_improvement)
            self.metrics['prb_improvement'].append(prb_improvement)
            self.metrics['stability_penalty'].append(stability_pen)
            self.metrics['energy_consumption_penalty'].append(energy_consumption_penalty)
            self.metrics['violation_penalty'].append(violation_penalty)
            self.metrics['warning_reward'].append(warning_reward)
            self.metrics['total_reward'].append(total_reward)

        return float(np.clip(total_reward, -100.0, 100.0))
    
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
        
        action = self.current_padded_action
        
        # Calculate actual reward using state as prev_state and next_state as current
        actual_reward = self.calculate_reward(state, action, next_state)
        
        # Convert inputs to numpy if needed
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        
        # Ensure proper shapes
        raw_state = np.array(state).flatten().copy()
        state = self.normalize_state(raw_state)
        state = self.pad_state(state)
        state = self.augment_state(raw_state, state, self.prev_states)
        
        action = np.array(action).flatten()
        raw_next_state = np.array(next_state).flatten().copy()
        next_state = self.normalize_state(raw_next_state)
        next_state = self.pad_state(next_state)
        next_state = self.augment_state(raw_next_state, next_state, raw_state)
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
            next_value = self.critic(next_state_tensor).cpu().numpy().flatten()[0]
        
        # Create transition
        transition = Transition(
            state=state,
            action=action,
            reward=actual_reward,
            next_state=next_state,
            done=done,
            log_prob=self.last_log_prob,
            value=value
        )
        
        self.buffer.add(transition)
        
        # Update prev_states for next step (important for delta features)
        self.prev_states = raw_state.copy()
        
        # Check for episode termination
        if done:
            # Natural episode termination - reset prev_states
            self.prev_states = None
            self.logger.info("Episode terminated naturally (done=True)")
        elif len(self.buffer) >= self.step_per_episode and self.training_mode:
            # Forced episode termination due to buffer size
            self.end_episode()
    
    def compute_gae_from_trajectories(self, transitions):
        """
        Compute advantages & returns per-trajectory.
        transitions: list of Transition in insertion order
        Returns: advantages (np array), returns (np array), values (np array)
        """
        # split into trajectories by done flags
        trajs = []
        cur = []
        for t in transitions:
            cur.append(t)
            if t.done:
                trajs.append(cur)
                cur = []
        if len(cur) > 0:
            # if last trajectory didn't end with done, still treat as trajectory
            trajs.append(cur)

        all_advantages = []
        all_returns = []
        all_values = []

        for traj in trajs:
            # collect arrays
            rewards = np.array([t.reward for t in traj], dtype=np.float32)
            values = np.array([t.value for t in traj], dtype=np.float32)
            dones = np.array([t.done for t in traj], dtype=np.float32)
            # compute next_values: V_{t+1} for each t
            # for last step next_value = 0 if done else critic(next_state)
            # but transitions store next_state; easier: compute next_values by shifting values and append last_next_value
            # we'll compute last_next_value via critic on last next_state
            # fetch next_values array
            # prepare next_state tensors
            with torch.no_grad():
                next_states = np.array([t.next_state for t in traj], dtype=np.float32)
                if len(next_states) > 0:
                    ns_tensor = torch.FloatTensor(next_states).to(self.device)
                    next_vals = self.critic(ns_tensor).cpu().numpy().flatten()
                else:
                    next_vals = np.array([], dtype=np.float32)

            # Now compute deltas using next_vals and dones
            advantages = np.zeros_like(rewards)
            last_adv = 0.0
            for t in reversed(range(len(rewards))):
                next_non_terminal = 1.0 - dones[t]
                next_value = next_vals[t] if (t < len(next_vals)) else 0.0
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                advantages[t] = last_adv = delta + self.gamma * self.lambda_gae * next_non_terminal * last_adv

            returns = advantages + values

            all_advantages.append(advantages)
            all_returns.append(returns)
            all_values.append(values)

        # concat trajectories back to arrays
        advantages = np.concatenate(all_advantages, axis=0) if len(all_advantages) > 0 else np.array([], dtype=np.float32)
        returns = np.concatenate(all_returns, axis=0) if len(all_returns) > 0 else np.array([], dtype=np.float32)
        values = np.concatenate(all_values, axis=0) if len(all_values) > 0 else np.array([], dtype=np.float32)

        return advantages, returns, values

    def train(self):
        """Train PPO using full buffer aggregated into trajectories"""
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.get_all_flat()  # insertion order
        # get arrays
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        old_log_probs = np.array([t.log_prob for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)

        # compute advantages & returns per-trajectory
        advantages, returns, _ = self.compute_gae_from_trajectories(transitions)

        # normalize advantages
        if len(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        data_size = len(states)
        
        # Initialize variables to track metrics from the last batch
        final_entropy = 0.0
        final_actor_loss = 0.0
        final_critic_loss = 0.0
        
        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(data_size)
            for start in range(0, data_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_log_probs = old_log_probs_tensor[idx]
                batch_advantages = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]

                # forward
                action_mean, action_logstd = self.actor(batch_states)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                # sum only active action dims (first n_cells)
                n = int(self.n_cells)
                log_prob_per_dim = dist.log_prob(batch_actions)  # (batch, action_dim)
                new_log_probs = log_prob_per_dim[:, :n].sum(-1)
                
                # Calculate entropy for tracking
                entropy = dist.entropy()[:, :n].sum(-1).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                # Track metrics from the last epoch and last batch
                if epoch == self.ppo_epochs - 1 and start + self.batch_size >= data_size:
                    final_entropy = entropy.item()
                    final_actor_loss = actor_loss.item()
                    final_critic_loss = critic_loss.item()

        # Store final metrics
        self.metrics['entropy'].append(final_entropy)
        self.metrics['actor_loss'].append(final_actor_loss)
        self.metrics['critic_loss'].append(final_critic_loss)

        # clear buffer after on-policy update
        self.buffer.clear()
        self.logger.info(f"Training completed: Actor loss={final_actor_loss:.4f}, Critic loss={final_critic_loss:.4f}, Entropy={final_entropy:.4f}")

    
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
        """Save comprehensive metrics plots showing all reward components"""
        if not any(self.metrics.values()):
            return
        
        def ma(data, window=20):
            if len(data) < window: return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Create a larger figure with more subplots for comprehensive view
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Episode {self.total_episodes} - Comprehensive Metrics Dashboard', fontsize=16)
        
        # 1. QoS Improvement Rewards (top-left)
        if any([self.metrics['drop_improvement'], self.metrics['latency_improvement'], 
                self.metrics['cpu_improvement'], self.metrics['prb_improvement']]):
            if self.metrics['drop_improvement']:
                axes[0,0].plot(ma(self.metrics['drop_improvement']), label='Drop Improvement', linewidth=2)
            if self.metrics['latency_improvement']:
                axes[0,0].plot(ma(self.metrics['latency_improvement']), label='Latency Improvement', linewidth=2)
            if self.metrics['cpu_improvement']:
                axes[0,0].plot(ma(self.metrics['cpu_improvement']), label='CPU Improvement', linewidth=2)
            if self.metrics['prb_improvement']:
                axes[0,0].plot(ma(self.metrics['prb_improvement']), label='PRB Improvement', linewidth=2)
            axes[0,0].set_title('QoS Improvement Rewards')
            axes[0,0].set_ylabel('Reward Value')
            axes[0,0].legend(fontsize=8)
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Energy & Efficiency Rewards (top-center)
        if self.metrics['energy_efficiency_reward']:
            axes[0,1].plot(ma(self.metrics['energy_efficiency_reward']), label='Energy Efficiency', linewidth=2, color='green')
            if self.metrics['energy_consumption_penalty']:
                axes[0,1].plot(ma(self.metrics['energy_consumption_penalty']), label='Energy Consumption Penalty', linewidth=2, color='red')
            axes[0,1].set_title('Energy-Related Rewards')
            axes[0,1].set_ylabel('Reward Value')
            axes[0,1].legend(fontsize=8)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Penalty Components (top-right)
        if any([self.metrics['stability_penalty'], self.metrics['violation_penalty']]):
            if self.metrics['stability_penalty']:
                axes[0,2].plot(ma(self.metrics['stability_penalty']), label='Stability Penalty', linewidth=2, color='orange')
            if self.metrics['violation_penalty']:
                axes[0,2].plot(ma(self.metrics['violation_penalty']), label='Violation Penalty', linewidth=2, color='red')
            if self.metrics['warning_reward']:
                axes[0,2].plot(ma(self.metrics['warning_reward']), label='Warning Reward', linewidth=2, color='yellow')
            axes[0,2].set_title('Penalty & Warning Components')
            axes[0,2].set_ylabel('Penalty/Reward Value')
            axes[0,2].legend(fontsize=8)
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Total Reward (middle-left)
        if self.metrics['total_reward']:
            axes[1,0].plot(self.metrics['total_reward'], alpha=0.3, color='blue')
            axes[1,0].plot(ma(self.metrics['total_reward']), linewidth=2, color='darkblue', label='Total Reward MA')
            axes[1,0].set_title('Total Reward')
            axes[1,0].set_ylabel('Total Reward')
            axes[1,0].legend(fontsize=8)
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. QoS Metrics (middle-center)
        if any([self.metrics['drop_rate'], self.metrics['latency'], self.metrics['cpu'], self.metrics['prb']]):
            if self.metrics['drop_rate']:
                axes[1,1].plot(ma(self.metrics['drop_rate']), label='Drop Rate (%)', linewidth=2)
            if self.metrics['latency']:
                # Normalize latency for better visualization
                latency_norm = np.array(self.metrics['latency']) / 100  # Scale down for visualization
                axes[1,1].plot(ma(latency_norm), label='Latency (ms/100)', linewidth=2)
            if self.metrics['cpu']:
                axes[1,1].plot(ma(self.metrics['cpu']), label='Max CPU (%)', linewidth=2)
            if self.metrics['prb']:
                axes[1,1].plot(ma(self.metrics['prb']), label='Max PRB (%)', linewidth=2)
            axes[1,1].set_title('QoS Metrics')
            axes[1,1].set_ylabel('Metric Value')
            axes[1,1].legend(fontsize=8)
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Training Metrics (middle-right)
        if any([self.metrics['actor_loss'], self.metrics['critic_loss'], self.metrics['entropy']]):
            if self.metrics['actor_loss']:
                axes[1,2].plot(self.metrics['actor_loss'], alpha=0.3, label='Actor Loss', color='red')
                if len(self.metrics['actor_loss']) >= 5:
                    axes[1,2].plot(ma(self.metrics['actor_loss'], 5), label='Actor Loss MA', linewidth=2, color='darkred')
            if self.metrics['critic_loss']:
                axes[1,2].plot(self.metrics['critic_loss'], alpha=0.3, label='Critic Loss', color='blue')
                if len(self.metrics['critic_loss']) >= 5:
                    axes[1,2].plot(ma(self.metrics['critic_loss'], 5), label='Critic Loss MA', linewidth=2, color='darkblue')
            axes[1,2].set_title('Training Losses')
            axes[1,2].set_ylabel('Loss Value')
            axes[1,2].legend(fontsize=8)
            axes[1,2].grid(True, alpha=0.3)
        
        # 7. Entropy (bottom-left)
        if self.metrics['entropy']:
            axes[2,0].plot(self.metrics['entropy'], alpha=0.3, label='Entropy', color='purple')
            axes[2,0].plot(ma(self.metrics['entropy']), label='Entropy MA', linewidth=2, color='darkmagenta')
            axes[2,0].set_title('Policy Entropy')
            axes[2,0].set_ylabel('Entropy')
            axes[2,0].set_xlabel('Training Step')
            axes[2,0].legend(fontsize=8)
            axes[2,0].grid(True, alpha=0.3)
        
        # 8. Reward Components Stacked (bottom-center)
        if any([self.metrics['energy_efficiency_reward'], self.metrics['drop_improvement'], 
                self.metrics['stability_penalty'], self.metrics['violation_penalty']]):
            reward_components = []
            labels = []
            colors = []
            
            if self.metrics['energy_efficiency_reward']:
                reward_components.append(ma(self.metrics['energy_efficiency_reward']))
                labels.append('Energy Efficiency')
                colors.append('green')
            if self.metrics['drop_improvement']:
                reward_components.append(ma(self.metrics['drop_improvement']))
                labels.append('Drop Improvement')
                colors.append('blue')
            if self.metrics['stability_penalty']:
                reward_components.append(ma(self.metrics['stability_penalty']))
                labels.append('Stability Penalty')
                colors.append('orange')
            if self.metrics['violation_penalty']:
                reward_components.append(ma(self.metrics['violation_penalty']))
                labels.append('Violation Penalty')
                colors.append('red')
            
            # Stack the components
            if reward_components:
                axes[2,1].stackplot(range(len(reward_components[0])), *reward_components, 
                                  labels=labels, colors=colors, alpha=0.7)
                axes[2,1].set_title('Reward Components (Stacked)')
                axes[2,1].set_ylabel('Cumulative Reward')
                axes[2,1].set_xlabel('Step')
                axes[2,1].legend(fontsize=8, loc='upper left')
                axes[2,1].grid(True, alpha=0.3)
        
        # 9. Reward Distribution (bottom-right)
        if self.metrics['total_reward']:
            axes[2,2].hist(self.metrics['total_reward'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[2,2].axvline(np.mean(self.metrics['total_reward']), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(self.metrics["total_reward"]):.2f}')
            axes[2,2].set_title('Total Reward Distribution')
            axes[2,2].set_xlabel('Reward Value')
            axes[2,2].set_ylabel('Frequency')
            axes[2,2].legend(fontsize=8)
            axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/comprehensive_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Comprehensive plots saved to plots/comprehensive_metrics_{timestamp}.png")