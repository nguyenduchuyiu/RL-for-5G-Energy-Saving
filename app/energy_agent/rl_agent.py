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

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = yaml.safe_load(open('config.yaml'))

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
        
        # 9 global features + 4 delta features + 2*max_cells local features
        self.state_dim = self.original_state_dim + 9 + 4 + 2*self.max_cells
        
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
        self.lambda_lr = float(config.get('lambda_lr', 1e-3))
        self.lambda_multiplier = float(config.get('lambda_init', 1.0))
        self.lambda_max = float(config.get('lambda_max', 10.0))
        self.cost_target = float(config.get('cost_target', 0.0))
        
        # Normalization parameters, use original state dimension and n_cells for state normalizer
        self.state_normalizer = StateNormalizer(self.original_state_dim, n_cells=self.n_cells)
        
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
            # self.logger.info(f"Fine-tuning with learning rate 3e-5 for actor and 1e-4 for critic")
            # self.actor_optimizer.param_groups[0]['lr'] = 3e-5
            # self.critic_optimizer.param_groups[0]['lr'] = 1e-4
        else:
            self.logger.info(f"No checkpoint found at {self.checkpoint_load_path}")

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
        else:
            # First step, no deltas
            drop_rate_delta = 0.0
            latency_delta = 0.0
            energy_delta = 0.0
            active_cells_delta = 0.0

        delta_features = np.array([drop_rate_delta, latency_delta, energy_delta, active_cells_delta])

        global_features = np.array([
            ue_density, isd, base_power, 
            dist_to_drop_thresh, dist_to_latency_thresh, dist_to_cpu_thresh, dist_to_prb_thresh,
            load_per_active_cell, power_efficiency
            ]) 
        delta_features = np.array([drop_rate_delta, latency_delta, energy_delta, active_cells_delta])
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
        Get action from policy network
        
        Args:
            state: State vector from MATLAB interface
            env_id: Environment id for parallel execution
            
        Returns:
            action: Power ratios for each cell [0, 1]
        """

        raw_state = np.array(state).flatten()   
        normalized_state = self.normalize_state(raw_state)
        normalized_padded_state = self.pad_state(normalized_state)
        # Use prev_state to compute deltas (DON'T update yet, wait for update())
        state = self.augment_state(raw_state, normalized_padded_state, self.prev_states.get(env_id))
                
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

        self.current_padded_actions[env_id] = action.cpu().numpy().flatten()
        truncated_action = action.squeeze()[:self.n_cells]
        self.last_log_probs[env_id] = log_prob_sum
        if not self.training_mode:
            self.prev_states[env_id] = raw_state.copy()
        return truncated_action.cpu().numpy().flatten()

    
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

        NETWORK_START_IDX = 17  # adjust if different
        idx_total_energy = NETWORK_START_IDX + 0   # totalEnergy

        prev_energy = prev_state[idx_total_energy]
        curr_energy = current_state[idx_total_energy]

        energy_delta = curr_energy - prev_energy
        energy_scale = config.get('energy_grad_coeff', 1000.0)
        reward = -energy_scale * energy_delta / max(1e-6, prev_energy)


        # Clip để tránh outlier
        reward = np.clip(reward, -1000.0, 1000.0)

        if env_id == 0 and np.random.random() < 0.02:
            print(f"ΔEnergy={energy_delta:.4f}, Reward={reward:.4f}")

        return float(reward)



    def calculate_cost(self, current_state, env_id=0):
        current_state = np.array(current_state).flatten()

        NETWORK_START_IDX = 17
        CELL_START_IDX = 17 + 14

        drop = current_state[NETWORK_START_IDX + 2]
        latency = current_state[NETWORK_START_IDX + 3]
        drop_th = current_state[11]
        latency_th = current_state[12]
        cpu_th = current_state[13]
        prb_th = current_state[14]

        CPU_FEATURE_IDX = CELL_START_IDX
        PRB_FEATURE_IDX = CELL_START_IDX + self.n_cells
        max_cpu = np.max(current_state[CPU_FEATURE_IDX:CPU_FEATURE_IDX + self.n_cells])
        max_prb = np.max(current_state[PRB_FEATURE_IDX:PRB_FEATURE_IDX + self.n_cells])

        def _violation_penalty(x, th):
            return max(0.0, (x - th) / max(1e-6, th))

        drop_violation = _violation_penalty(drop, drop_th)
        latency_violation = _violation_penalty(latency, latency_th)
        cpu_violation = _violation_penalty(max_cpu, cpu_th)
        prb_violation = _violation_penalty(max_prb, prb_th)

        qos_cost = (
            config.get('drop_cost_coeff', 1.0) * drop_violation +
            config.get('latency_cost_coeff', 1.0) * latency_violation +
            config.get('cpu_cost_coeff', 1.0) * cpu_violation +
            config.get('prb_cost_coeff', 1.0) * prb_violation
        )

        qos_cost = np.clip(qos_cost, 0.0, 1000.0)

        if env_id == 0 and np.random.random() < 0.02:
            print(f"qos_cost: {qos_cost:.4f}")
            print(f"drop_vio={drop_violation:.4f}, lat_vio={latency_violation:.4f}, "
                f"cpu_vio={cpu_violation:.4f}, prb_vio={prb_violation:.4f}")

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
        state = self.normalize_state(raw_state)
        state = self.pad_state(state)
        state = self.augment_state(raw_state, state, self.prev_states.get(env_id))
        
        action = np.array(action).flatten()
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
        
        # Create transition
        # Build transition with next_state retained (normalized/padded/augmented like state)
        raw_next_state = np.array(next_state).flatten().copy()
        next_state_norm = self.normalize_state(raw_next_state)
        next_state_pad = self.pad_state(next_state_norm)
        next_state_aug = self.augment_state(raw_next_state, next_state_pad, raw_state)

        transition = Transition(
            state=state,
            action=action,
            reward=actual_reward,
            cost=actual_cost,
            next_state=next_state_aug,
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

        dataset = torch.utils.data.TensorDataset(
            states_tensor, actions_tensor, old_log_probs_tensor, advantages_tensor, returns_tensor,
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
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)

                n = int(self.n_cells)
                log_prob_per_dim = dist.log_prob(batch_actions)
                new_log_probs = log_prob_per_dim[:, :n].sum(-1)
                entropy = dist.entropy()[:, :n].sum(-1).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
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
                actor_loss = -(L_reward - lagrange * L_cost) - self.entropy_coef * entropy

                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns.detach())
                # Cost critic loss
                current_cost_values = self.cost_critic(batch_states).squeeze()
                cost_critic_loss = nn.MSELoss()(current_cost_values, batch_cost_returns.detach())

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
        self.lambda_multiplier = float(np.clip(self.lambda_multiplier + self.lambda_lr * (J_c - self.cost_target), 0.0, self.lambda_max))
        
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
            f"Train: J_r={J_r:.4f}, J_c={J_c:.4f}, lambda={self.lambda_multiplier:.4f}, "
            f"actor={final_actor_loss:.4f}, critic={final_critic_loss:.4f}, cost_critic={final_cost_critic_loss:.4f}, ent={final_entropy:.4f}"
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
            # Draw target line
            axes[0,1].axhline(self.cost_target, color='gray', linestyle='--', linewidth=1, label='Cost Target')
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