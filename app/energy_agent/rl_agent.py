# energy_agent/rl_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import os
from datetime import datetime
import random
import time

from .transition import Transition, TransitionBuffer
from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer

class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False, n_cells_max=100):
        """
        PPO-based agent tuned for fast learning in the competition.
        Adds:
         - warm-start heuristic
         - smoothing + hysteresis
         - augmented state (prev_action + moving averages) via adapter
         - aggressive, small-batch PPO updates for fast learning
        """
        print("Initializing RL Agent (patched for fast PPO)")
        seed = 42
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # fixed maximum cells for padding/masking across scenarios

        self.n_cells_max = n_cells_max

        # base state dims (fixed with n_cells_max to enable weight sharing)
        self.state_dim_base = 17 + 14 + (self.n_cells_max * 12)

        # augmentation: prev_action (n_cells_max) + drop_ma + latency_ma
        self.aug_extra = self.n_cells_max + 2
        self.aug_state_dim = self.state_dim_base + self.aug_extra

        # normalizer set to n_cells_max (padding fills missing cells)
        self.state_normalizer = StateNormalizer(self.state_dim_base, n_cells=self.n_cells_max)

        # networks: keep actor/critic input size fixed to state_dim_base
        # create adapter that maps augmented state -> base state dimension
        self.adapter = nn.Sequential(
            nn.Linear(self.aug_state_dim, max(128, self.state_dim_base)),
            nn.ReLU(),
            nn.Linear(max(128, self.state_dim_base), self.state_dim_base)
        ).to(self.device)

        # create actor & critic (smaller hidden dims for fast learning)
        hidden_size = 128
        self.actor = Actor(self.state_dim_base, self.n_cells_max, hidden_dim=hidden_size).to(self.device)
        self.critic = Critic(self.state_dim_base, hidden_dim=hidden_size).to(self.device)


        # optimizers
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + list(self.adapter.parameters()), lr=3e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        # PPO hyperparameters (fast-learning)
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 6
        self.buffer_size = 2048
        self.update_every = 32  # update every N env steps
        self.batch_size = 32

        # experience buffer
        self.buffer = TransitionBuffer(self.buffer_size)

        # bookkeeping
        self.training_mode = os.environ.get("ES_TRAINING_MODE", "1") != "0"
        print(f"Training mode: {self.training_mode}")
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0

        # smoothing & warm-start
        # masks and padded vectors use n_cells_max
        self.action_mask = np.concatenate([np.ones(self.n_cells), np.zeros(self.n_cells_max - self.n_cells)]).astype(np.float32)
        self.prev_action = np.ones(self.n_cells_max) * 0.7  # safe initial power ratio (padded)
        self.noise_scale = 0.04  # exploration noise scale

        # moving averages for drop & latency
        self.drop_ma_window = deque(maxlen=5)
        self.latency_ma_window = deque(maxlen=5)

        # reward penalty coefficients
        self.drop_penalty_coef = 10.0
        self.latency_penalty_coef = 2.0
        self.energy_coeff = 1.0

        self.setup_logging(log_file)
        self.logger.info(f"Patched PPO Agent initialized: {n_cells} cells, {n_ues} UEs, device={self.device}")

        # timing debug
        self._time_check = True
        
        # checkpointing (auto load/save across scenarios)
        self.checkpoint_path = os.environ.get("ES_CHECKPOINT_PATH", os.path.join("models", "ppo_model.pth"))
        self.auto_checkpoint = os.environ.get("ES_AUTO_CHECKPOINT", "1") != "0"
        try:
            ckpt_dir = os.path.dirname(self.checkpoint_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
        except Exception:
            pass

    def _pad1d(self, x, max_len):
        arr = np.array(x, dtype=np.float32).flatten()
        if arr.shape[0] >= max_len:
            return arr[:max_len]
        pad = np.zeros(max_len - arr.shape[0], dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def _pad_state_base(self, base_state):
        # pad or truncate raw base state vector to fixed state_dim_base
        return self._pad1d(base_state, self.state_dim_base)
        
    def setup_logging(self, log_file):
        self.logger = logging.getLogger('PatchedPPOAgent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def normalize_state(self, state):
        """Normalize base part of state using provided StateNormalizer"""
        return self.state_normalizer.normalize(state)

    def _build_augmented_state(self, base_state):
        """
        Build augmented state: [normalized_base | prev_action | drop_ma | latency_ma]
        - base_state: raw state vector (un-normalized)
        """
        # normalized base on padded vector
        base_padded = self._pad_state_base(np.array(base_state).flatten())
        norm_base = self.normalize_state(base_padded)
        # prev_action already in [0,1], padded length n_cells_max
        prev_action = np.clip(self.prev_action, 0.0, 1.0)
        # moving averages (if empty, use current state's values)
        # extract current drop_rate and latency from base_state using known indices
        base = np.array(base_state).flatten()
        network_start_idx = 17
        drop_rate_idx = network_start_idx + 2   # Index 19
        latency_idx = network_start_idx + 3 # Index 20
        current_drop = base[drop_rate_idx] if drop_rate_idx < len(base) else 0.0
        current_latency = base[latency_idx] if latency_idx < len(base) else 0.0
        mean_drop_rate = np.mean(base[11])
        mean_latency = np.mean(base[12])


        # update deques if called during step/update
        if len(self.drop_ma_window) == 0:
            drop_ma = current_drop
            latency_ma = current_latency
        else:
            drop_ma = np.mean(self.drop_ma_window)
            latency_ma = np.mean(self.latency_ma_window)

        # normalize drop_ma, latency_ma to roughly [0,1] using thresholds from state (fallback values)
        # We'll clip using plausible ranges
        drop_ma_norm = np.clip(drop_ma / mean_drop_rate, 0.0, 1.0)
        latency_ma_norm = np.clip(latency_ma / mean_latency, 0.0, 1.0)

        aug = np.concatenate([
            norm_base.astype(np.float32),
            prev_action.astype(np.float32),
            np.array([drop_ma_norm, latency_ma_norm], dtype=np.float32)
        ], axis=0)

        return aug

    def start_scenario(self):
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        # Reset global counters and randomness to make scenarios independent
        self.total_steps = 0
        self.prev_action = np.ones(self.n_cells_max) * 0.7
        self.drop_ma_window.clear()
        self.latency_ma_window.clear()
        self.logger.info(f"Starting episode {self.total_episodes}")
        
        # auto-load weights to continue training across scenarios
        if self.auto_checkpoint and isinstance(self.checkpoint_path, str) and os.path.isfile(self.checkpoint_path):
            try:
                self.load_model(self.checkpoint_path)
                self.logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint {self.checkpoint_path}: {e}")

    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        self.logger.info(f"Episode {self.total_episodes} ended: Steps={self.episode_steps}, Reward={self.current_episode_reward:.2f}, Avg100={avg_reward:.2f}")
        # run final training if any
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
        
        # auto-save weights for cross-scenario training continuity
        if self.auto_checkpoint and isinstance(self.checkpoint_path, str):
            try:
                self.save_model(self.checkpoint_path)
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint {self.checkpoint_path}: {e}")

    def get_action(self, state):
        """
        Return action for environment in [0,1] range.
        Implements:
         - warm-start heuristic for first few steps
         - actor policy (mean + std) sampling
         - smoothing + hysteresis
        """
        t0 = time.perf_counter() if self._time_check else None

        base_state = np.array(state).flatten()
        # actor-only action (no warm-start heuristic, no smoothing/hysteresis)
        aug_state = self._build_augmented_state(base_state)
        with torch.no_grad():
            aug_tensor = torch.FloatTensor(aug_state).unsqueeze(0).to(self.device)
            adapted = self.adapter(aug_tensor)  # -> [1, state_dim_base]
            action_mean, action_logstd = self.actor(adapted)
            if self.training_mode:
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                sampled = dist.rsample()
                # masked log-prob over valid action dimensions
                mask = torch.from_numpy(self.action_mask).to(self.device).unsqueeze(0)
                log_prob = (dist.log_prob(sampled) * mask).sum(-1)
            else:
                sampled = action_mean
                log_prob = torch.zeros(1).to(self.device)
            action_full = sampled.squeeze(0).cpu().numpy()
            if self.training_mode:
                action_full = action_full + np.random.normal(0, self.noise_scale, size=action_full.shape)
            action_full = np.clip(action_full, 0.0, 1.0)
            self.last_log_prob = log_prob.cpu().numpy() if 'log_prob' in locals() else np.array([0.0])

        # update prev_action for next step & store
        self.prev_action = action_full.copy()
        self.last_state = base_state
        self.last_action = action_full.copy()

        if self._time_check:
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000.0
            if elapsed > 50.0:
                self.logger.warning(f"get_action took {elapsed:.1f} ms (exceeds 50ms)")

        # return only first n_cells to environment
        return action_full[:self.n_cells].copy()

    def calculate_reward(self, prev_state, action, current_state):
            """Modified reward: energy saving scaled, heavy penalties on drop/latency violations"""
            if prev_state is None:
                return 0.0

            prev_state = np.array(prev_state).flatten()
            current_state = np.array(current_state).flatten()

            # indices according to original mapping
            simulation_start_idx = 0
            network_start_idx = 17  # after simulation features

            # extract metrics (careful with bounds)
            def safe_get(arr, idx, default=0.0):
                return arr[idx] if idx < len(arr) else default

            # energy is in network features at index (network start + 0)
            # [CORRECT]
            current_energy = safe_get(current_state, network_start_idx + 0, 0.0)
            prev_energy = safe_get(prev_state, network_start_idx + 0, current_energy)

            # connected UEs at network offset 5
            # [FIXED] Was simulation_start_idx + 5 (carrierFrequency)
            current_connected_ues = safe_get(current_state, network_start_idx + 5, 0.0) 
            prev_connected_ues = safe_get(prev_state, network_start_idx + 5, current_connected_ues)

            # drop rate & latency at network offsets 2 & 3
            # [CORRECT]
            current_drop_rate = safe_get(current_state, network_start_idx + 2, 0.0)
            prev_drop_rate = safe_get(prev_state, network_start_idx + 2, current_drop_rate)

            # [FIXED] Was simulation_start_idx + 12 (latencyThreshold)
            current_latency = safe_get(current_state, network_start_idx + 3, 0.0) 
            prev_latency = safe_get(prev_state, network_start_idx + 3, current_latency)

            # thresholds available in simulation features (indices within the first 17)
            # [FIXED] Was 10 (idlePower)
            drop_threshold = safe_get(current_state, 11, 5.0)  # default 5%
            # [FIXED] Was 11 (dropCallThreshold)
            latency_threshold = safe_get(current_state, 12, 50.0)  # default 50ms
            # note: based on original mapping these may differ; use defaults if missing

            # energy reward (positive when energy reduced)
            energy_change = prev_energy - current_energy
            energy_reward = self.energy_coeff * energy_change

            # heavy penalties
            drop_penalty = 0.0
            if current_drop_rate > drop_threshold:
                drop_penalty = self.drop_penalty_coef * (current_drop_rate - drop_threshold)

            latency_penalty = 0.0
            if current_latency > latency_threshold:
                latency_penalty = self.latency_penalty_coef * (current_latency - latency_threshold) / max(1.0, latency_threshold)

            # connection stability
            connection_reward = (current_connected_ues - prev_connected_ues) * 5.0

            # improvement bonuses
            drop_improvement = (prev_drop_rate - current_drop_rate) * 2.0
            latency_improvement = (prev_latency - current_latency) * 0.1

            # Scale the reward to avoid large values
            scaling_factor = 1000.0
            reward = (energy_reward - drop_penalty - latency_penalty + connection_reward + drop_improvement + latency_improvement) / scaling_factor
            
            # clip to reasonable bounds
            reward = float(np.clip(reward, -10.0, 1.0))

            # update moving averages used for augmentation
            try:
                self.drop_ma_window.append(current_drop_rate)
                self.latency_ma_window.append(current_latency)
            except Exception:
                pass

            # debug print (light)
            if self.total_steps % 50 == 0:
                self.logger.info(f"Reward components: E={energy_reward:.2f}, DropPen={drop_penalty:.2f}, LatPen={latency_penalty:.2f}, Conn={connection_reward:.2f}")

            return reward

    def update(self, state, action, next_state, done):
        """
        Called every env step by simulator.
        - compute reward (we use prev state as reference)
        - store transition in buffer
        - trigger training every self.update_every steps
        """

        actual_reward = self.calculate_reward(state, action, next_state)

        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += actual_reward

        if not self.training_mode:
            return
        # convert inputs if torch tensors
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()

        # shapes & normalization: pad to base dims and n_cells_max
        state_base = self.normalize_state(self._pad_state_base(np.array(state).flatten()))
        action_arr = self._pad1d(np.array(action).flatten(), self.n_cells_max)
        next_state_base = self.normalize_state(self._pad_state_base(np.array(next_state).flatten()))

        # value estimate via critic (on adapted input)
        try:
            # build augmented inputs for critic
            # but for value estimate we need adapted base input; use adapter
            # build augmented raw (we must supply raw base state to _build_augmented_state)
            aug_state = self._build_augmented_state(state)
            aug_tensor = torch.FloatTensor(aug_state).unsqueeze(0).to(self.device)
            adapted = self.adapter(aug_tensor)
            with torch.no_grad():
                value = self.critic(adapted).cpu().numpy().flatten()[0]
        except Exception:
            value = 0.0

        # store log_prob (from last action if available)
        log_prob_val = getattr(self, 'last_log_prob', np.array([0.0]))
        log_prob_scalar = float(log_prob_val[0]) if hasattr(log_prob_val, '__len__') else float(log_prob_val)

        transition = Transition(
            state=state_base,
            action=action_arr,
            reward=actual_reward,
            next_state=next_state_base,
            done=done,
            log_prob=log_prob_scalar,
            value=value,
            action_mask=self.action_mask.copy()
        )

        self.buffer.add(transition)

        # perform training periodically
        if self.total_steps % self.update_every == 0 and len(self.buffer) >= self.batch_size:
            self.train()

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = next_values[t] if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_adv = delta + self.gamma * self.lambda_gae * next_non_terminal * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def train(self):
        """PPO training using minibatch sampling of recent transitions (stabilized)"""
        if len(self.buffer) < self.batch_size:
            return

        # sample up to recent_k transitions to bias recent data
        recent_k = min(len(self.buffer), 1024)
        transitions = self.buffer.get_last_n(recent_k)

        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        old_log_probs = np.array([t.log_prob for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)
        action_masks = np.array([t.action_mask for t in transitions], dtype=np.float32)

        # --- Reward normalization (simple, per-batch) ---
        # Prevent extremely large reward scale blowing up value updates.
        if rewards.std() > 1e-8:
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_norm = rewards - rewards.mean()
        # use normalized rewards for GAE but keep original for logging if needed
        rewards_for_gae = rewards_norm

        # compute next_values using critic on adapted inputs
        with torch.no_grad():
            pad_prev = np.tile(self.prev_action, (next_states.shape[0], 1))
            drop_ma_val = np.mean(self.drop_ma_window) if len(self.drop_ma_window) > 0 else 0.0
            latency_ma_val = np.mean(self.latency_ma_window) if len(self.latency_ma_window) > 0 else 0.0
            pad_ma = np.tile(np.array([np.clip(drop_ma_val/20.0,0,1.0), np.clip(latency_ma_val/200.0,0,1.0)]), (next_states.shape[0],1))
            next_aug = np.concatenate([next_states, pad_prev, pad_ma], axis=1)
            next_aug_tensor = torch.FloatTensor(next_aug).to(self.device)
            adapted_next = self.adapter(next_aug_tensor)
            next_values_batch = self.critic(adapted_next).cpu().numpy().flatten()

        # advantages & returns (use normalized rewards)
        advantages, returns = self.compute_gae(rewards_for_gae, values, next_values_batch, dones)
        # normalize advantages (important)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert to tensors
        dataset_size = len(states)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        old_values_tensor = torch.FloatTensor(values).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        masks_tensor = torch.FloatTensor(action_masks).to(self.device)

        actor_loss_val = 0.0
        critic_loss_val = 0.0

        # Training hyperparams sanity:
        # If dataset small, reduce epochs to avoid overfitting/noise amplification.
        ppo_epochs = max(1, min(self.ppo_epochs, 6))

        for epoch in range(ppo_epochs):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if len(idx) == 0:
                    continue

                # ---- Batch ----
                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_log_probs = old_log_probs_tensor[idx]
                batch_adv = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]
                batch_old_values = old_values_tensor[idx] 
                batch_masks = masks_tensor[idx]

                drop_ma_val = np.mean(self.drop_ma_window) if len(self.drop_ma_window) > 0 else 0.0
                latency_ma_val = np.mean(self.latency_ma_window) if len(self.latency_ma_window) > 0 else 0.0
                # Create augmentation for the current batch only
                pad_prev = torch.FloatTensor(np.tile(self.prev_action, (batch_states.size(0),1))).to(self.device)
                pad_ma = torch.FloatTensor(np.tile(np.array([
                    np.clip(drop_ma_val/20.0,0,1.0),
                    np.clip(latency_ma_val/200.0,0,1.0)
                ]), (batch_states.size(0),1))).to(self.device)
                batch_aug = torch.cat([batch_states, pad_prev, pad_ma], dim=1)


                 # ---- Forward ----
                adapted = self.adapter(batch_aug)

                # actor
                action_mean, action_logstd = self.actor(adapted)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = (dist.log_prob(batch_actions) * batch_masks).sum(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic
                v_pred = self.critic(adapted.detach()).squeeze()
                v_old = batch_old_values.squeeze()
                v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -self.clip_epsilon, self.clip_epsilon)
                value_loss1 = (v_pred - batch_returns).pow(2)
                value_loss2 = (v_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # ---- Update ----
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.adapter.parameters()), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_loss_val = actor_loss.item()
                critic_loss_val = value_loss.item()


            # end minibatch loop

        # logging summary diagnostics
        try:
            with torch.no_grad():
                # quick diagnostics on last batch
                mean_reward = float(rewards.mean())
                mean_value = float(v_pred.mean().cpu().numpy()) if 'v_pred' in locals() else 0.0
                adv_std = float(advantages.std()) if advantages.size > 0 else 0.0
                ratio_mean = float(ratio.mean().cpu().numpy()) if 'ratio' in locals() else 1.0
                self.logger.info(f"PPO fast-train: Actor loss={actor_loss_val:.4f}, Critic loss={critic_loss_val:.4f}, mean_reward={mean_reward:.4f}, mean_value={mean_value:.4f}, adv_std={adv_std:.4f}, ratio_mean={ratio_mean:.4f}")
        except Exception:
            self.logger.info(f"PPO fast-train: Actor loss={actor_loss_val:.4f}, Critic loss={critic_loss_val:.4f}")


    def save_model(self, filepath=None):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        # ensure directory exists
        try:
            dirpath = os.path.dirname(filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
        except Exception:
            pass
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'adapter_state_dict': self.adapter.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'adapter_state_dict' in checkpoint:
            try:
                self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
            except Exception:
                self.logger.warning("Adapter weights mismatch; skipping.")
        if 'actor_optimizer_state_dict' in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            except Exception:
                self.logger.warning("Optimizer state mismatch; skipping.")
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.logger.info(f"Model loaded from {filepath}")

    def set_training_mode(self, training):
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.adapter.train(training)
        self.logger.info(f"Training mode set to {training}")

    def get_stats(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer),
            'training_mode': self.training_mode,
            'episode_steps': self.episode_steps,
            'current_episode_reward': self.current_episode_reward
        }
