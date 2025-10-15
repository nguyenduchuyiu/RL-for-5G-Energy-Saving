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

from .transition import Transition, TransitionBuffer
from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer

class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        """
        PPO-based agent tuned for fast learning in the competition.
        Adds:
         - warm-start heuristic
         - smoothing + hysteresis
         - augmented state (prev_action + moving averages) via adapter
         - aggressive, small-batch PPO updates for fast learning
        """
        print("Initializing RL Agent (patched for fast PPO)")
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # base state dims (keep consistent with simulator mapping)
        self.state_dim_base = 17 + 14 + (n_cells * 12)  # original expected by normalizer & models

        # augmentation: prev_action (n_cells) + drop_ma + latency_ma
        self.aug_extra = n_cells + 2
        self.aug_state_dim = self.state_dim_base + self.aug_extra

        # normalizer unchanged (works on base part)
        self.state_normalizer = StateNormalizer(self.state_dim_base, n_cells=n_cells)

        # networks: keep actor/critic input size = state_dim_base (no change to models)
        # create adapter that maps augmented state -> base state dimension
        self.adapter = nn.Sequential(
            nn.Linear(self.aug_state_dim, max(128, self.state_dim_base)),
            nn.ReLU(),
            nn.Linear(max(128, self.state_dim_base), self.state_dim_base)
        ).to(self.device)

        # create actor & critic (smaller hidden dims for fast learning)
        hidden_size = 128
        self.actor = Actor(self.state_dim_base, n_cells, hidden_dim=hidden_size).to(self.device)
        self.critic = Critic(self.state_dim_base, hidden_dim=hidden_size).to(self.device)

        # optimizers
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + list(self.adapter.parameters()), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # PPO hyperparameters (fast-learning)
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        self.batch_size = 32
        self.buffer_size = 2048
        self.update_every = 8  # update every N env steps

        # experience buffer
        self.buffer = TransitionBuffer(self.buffer_size)

        # bookkeeping
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0

        # smoothing & warm-start
        self.prev_action = np.ones(self.n_cells) * 0.7  # safe initial power ratio
        self.smoothing_alpha = 0.6
        self.hysteresis_threshold = 0.05
        self.warm_start_steps = 40  # use heuristic for first N steps
        self.noise_scale = 0.04  # exploration noise scale

        # moving averages for drop & latency
        self.drop_ma_window = deque(maxlen=5)
        self.latency_ma_window = deque(maxlen=5)

        # reward penalty coefficients
        self.drop_penalty_coef = 1000.0
        self.latency_penalty_coef = 200.0
        self.energy_coeff = 10.0

        self.setup_logging(log_file)
        self.logger.info(f"Patched PPO Agent initialized: {n_cells} cells, {n_ues} UEs, device={self.device}")

        # timing debug
        self._time_check = True

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
        # normalized base
        norm_base = self.normalize_state(np.array(base_state).flatten())
        # prev_action already in [0,1]
        prev_action = np.clip(self.prev_action, 0.0, 1.0)
        # moving averages (if empty, use current state's values)
        # extract current drop_rate and latency from base_state using known indices
        try:
            base = np.array(base_state).flatten()
            drop_idx = 17 + 2  # caution: original mapping had network features starting at 18, avgDropRate index = 18+2? 
            # safer: use simulation indices (drop rate and latency are at simulation offsets)
            sim_drop_idx = 11  # within simulation features (0-based)
            sim_latency_idx = 12
            current_drop = base[sim_drop_idx] if sim_drop_idx < len(base) else 0.0
            current_latency = base[sim_latency_idx] if sim_latency_idx < len(base) else 0.0
        except Exception:
            current_drop = 0.0
            current_latency = 0.0

        # update deques if called during step/update
        if len(self.drop_ma_window) == 0:
            drop_ma = current_drop
            latency_ma = current_latency
        else:
            drop_ma = np.mean(self.drop_ma_window)
            latency_ma = np.mean(self.latency_ma_window)

        # normalize drop_ma, latency_ma to roughly [0,1] using thresholds from state (fallback values)
        # We'll clip using plausible ranges
        drop_ma_norm = np.clip(drop_ma / 20.0, 0.0, 1.0)
        latency_ma_norm = np.clip(latency_ma / 200.0, 0.0, 1.0)

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
        self.prev_action = np.ones(self.n_cells) * 0.7
        self.drop_ma_window.clear()
        self.latency_ma_window.clear()
        self.logger.info(f"Starting episode {self.total_episodes}")

    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        self.logger.info(f"Episode {self.total_episodes} ended: Steps={self.episode_steps}, Reward={self.current_episode_reward:.2f}, Avg100={avg_reward:.2f}")
        # run final training if any
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()

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
        # heuristic warm-start for early steps
        if self.total_steps < self.warm_start_steps:
            # compute avg load from cell features (loadRatio at last feature of each cell)
            try:
                start_idx = 17 + 14
                load_indices = [start_idx + i*12 + 11 for i in range(self.n_cells)]  # loadRatio index in cell block
                loads = [base_state[i] for i in load_indices if i < len(base_state)]
                avg_load = np.mean(loads) if len(loads) > 0 else 0.5
            except Exception:
                avg_load = 0.5
            base = 0.45 if avg_load < 0.2 else 0.7
            heuristic = np.clip(base + np.random.normal(0, self.noise_scale, size=self.n_cells), 0.0, 1.0)
            action = heuristic
        else:
            # build augmented state, map to base dim via adapter, feed actor
            aug_state = self._build_augmented_state(base_state)
            with torch.no_grad():
                aug_tensor = torch.FloatTensor(aug_state).unsqueeze(0).to(self.device)
                adapted = self.adapter(aug_tensor)  # -> [1, state_dim_base]
                action_mean, action_logstd = self.actor(adapted)
                if self.training_mode:
                    action_std = torch.exp(action_logstd)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    sampled = dist.rsample()
                    log_prob = dist.log_prob(sampled).sum(-1)
                else:
                    sampled = action_mean
                    log_prob = torch.zeros(1).to(self.device)
                action = sampled.squeeze(0).cpu().numpy()
                # action produced by actor is already in [0,1] due to sigmoid in Actor
                # add small gaussian exploration noise
                if self.training_mode:
                    action = action + np.random.normal(0, self.noise_scale, size=action.shape)
                    action = np.clip(action, 0.0, 1.0)

                # store last log prob for buffer use
                self.last_log_prob = log_prob.cpu().numpy() if 'log_prob' in locals() else np.array([0.0])

        # smoothing + hysteresis
        smoothed = self.smoothing_alpha * self.prev_action + (1.0 - self.smoothing_alpha) * action
        delta = smoothed - self.prev_action
        small_change_mask = np.abs(delta) < self.hysteresis_threshold
        smoothed[small_change_mask] = self.prev_action[small_change_mask]
        smoothed = np.clip(smoothed, 0.0, 1.0)

        # update prev_action for next step & store
        self.prev_action = smoothed.copy()
        self.last_state = base_state
        self.last_action = smoothed.copy()

        if self._time_check:
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000.0
            if elapsed > 50.0:
                self.logger.warning(f"get_action took {elapsed:.1f} ms (exceeds 50ms)")

        return smoothed.copy()

    def calculate_reward(self, prev_state, action, current_state):
        """Modified reward: energy saving scaled, heavy penalties on drop/latency violations"""
        if prev_state is None:
            return 0.0

        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()

        # indices according to original mapping
        current_simulation_start = 0
        current_network_start = 17  # after simulation features

        # extract metrics (careful with bounds)
        def safe_get(arr, idx, default=0.0):
            return arr[idx] if idx < len(arr) else default

        # energy is in network features at index (network start + 0)
        current_energy = safe_get(current_state, current_network_start + 0, 0.0)
        prev_energy = safe_get(prev_state, current_network_start + 0, current_energy)

        # connected UEs at simulation offset 5
        current_connected_ues = safe_get(current_state, current_simulation_start + 5, 0.0)
        prev_connected_ues = safe_get(prev_state, current_simulation_start + 5, current_connected_ues)

        # drop rate & latency at simulation offsets 11 & 12
        current_drop_rate = safe_get(current_state, current_simulation_start + 11, 0.0)
        prev_drop_rate = safe_get(prev_state, current_simulation_start + 11, current_drop_rate)

        current_latency = safe_get(current_state, current_simulation_start + 12, 0.0)
        prev_latency = safe_get(prev_state, current_simulation_start + 12, current_latency)

        # thresholds available in simulation features (indices within the first 17)
        drop_threshold = safe_get(current_state, 10, 5.0)  # default 5%
        latency_threshold = safe_get(current_state, 11, 50.0)  # default 50ms
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

        reward = energy_reward - drop_penalty - latency_penalty + connection_reward + drop_improvement + latency_improvement
        # clip to reasonable bounds (but very negative allowed due to heavy penalties)
        reward = float(np.clip(reward, -10000.0, 100.0))

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
        if not self.training_mode:
            return

        actual_reward = self.calculate_reward(state, action, next_state)

        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += actual_reward

        # convert inputs if torch tensors
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()

        # shapes & normalization
        state_base = self.normalize_state(np.array(state).flatten())
        action_arr = np.array(action).flatten()
        next_state_base = self.normalize_state(np.array(next_state).flatten())

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
            value=value
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
        """PPO training using minibatch sampling of recent transitions"""
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

        # compute next_values using critic on adapted inputs
        with torch.no_grad():
            # adapt next_states batch via adapter
            # need to reconstruct augmented next_state: approximate by concatenating prev_action + ma (use stored deques mean)
            # Simpler: pass next_states (base) through adapter by padding prev_action and ma with current values
            pad_prev = np.tile(self.prev_action, (next_states.shape[0], 1))
            drop_ma_val = np.mean(self.drop_ma_window) if len(self.drop_ma_window) > 0 else 0.0
            latency_ma_val = np.mean(self.latency_ma_window) if len(self.latency_ma_window) > 0 else 0.0
            pad_ma = np.tile(np.array([np.clip(drop_ma_val/20.0,0,1.0), np.clip(latency_ma_val/200.0,0,1.0)]), (next_states.shape[0],1))
            next_aug = np.concatenate([next_states, pad_prev, pad_ma], axis=1)
            next_aug_tensor = torch.FloatTensor(next_aug).to(self.device)
            adapted_next = self.adapter(next_aug_tensor)
            next_values_batch = self.critic(adapted_next).cpu().numpy().flatten()

        # advantages & returns
        advantages, returns = self.compute_gae(rewards, values, next_values_batch, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert to tensors
        dataset_size = len(states)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        actor_loss_val = 0.0
        critic_loss_val = 0.0

        # training loop: multiple epochs, minibatch sampling
        for epoch in range(self.ppo_epochs):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_log_probs = old_log_probs_tensor[idx]
                batch_adv = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]

                # adapt batch_states through adapter: need to append prev_action + ma to each (use current prev_action & ma)
                pad_prev = torch.FloatTensor(np.tile(self.prev_action, (batch_states.size(0),1))).to(self.device)
                drop_ma_val = np.mean(self.drop_ma_window) if len(self.drop_ma_window) > 0 else 0.0
                latency_ma_val = np.mean(self.latency_ma_window) if len(self.latency_ma_window) > 0 else 0.0
                pad_ma = torch.FloatTensor(np.tile(np.array([np.clip(drop_ma_val/20.0,0,1.0), np.clip(latency_ma_val/200.0,0,1.0)]), (batch_states.size(0),1))).to(self.device)
                batch_aug = torch.cat([batch_states, pad_prev, pad_ma], dim=1)
                adapted = self.adapter(batch_aug)

                # policy forward
                action_mean, action_logstd = self.actor(adapted)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss (value head expects adapted input as well)
                adapted_detached = adapted.detach()
                current_values = self.critic(adapted_detached).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)

                # update actor + adapter
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.adapter.parameters()), 0.5)
                self.actor_optimizer.step()

                # update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_loss_val = actor_loss.item()
                critic_loss_val = critic_loss.item()

        # keep buffer but remove oldest if capacity reached (buffer already deque)
        # optionally keep recent transitions only: we keep as-is
        self.logger.info(f"PPO fast-train: Actor loss={actor_loss_val:.4f}, Critic loss={critic_loss_val:.4f}")

    def save_model(self, filepath=None):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_fast_model_{timestamp}.pth"
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
