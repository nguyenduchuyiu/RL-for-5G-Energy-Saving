import numpy as np
from collections import deque, namedtuple

# Transition structure (giữ nguyên)
Transition = namedtuple('Transition', [
    'state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'
])

class TransitionBuffer:
    """
    Buffer organizes transitions as trajectories (episodes).
    - capacity: total number of transitions kept across all stored trajectories.
    - Internally stores deque of trajectories (each a list of Transition).
    - Maintains `current_traj` while an episode is being collected.
    """

    def __init__(self, capacity=2048):
        self.capacity = int(capacity)
        self.trajs = deque()            # deque of completed trajectories (old -> new)
        self.current_traj = []          # accumulating transitions for the ongoing episode
        self.total_transitions = 0      # current total transitions stored across trajs
        # optional: keep flattened cache for fast sampling (kept None and updated lazily)
        self._flat_cache = None

    # ---------- internal helpers ----------
    def _invalidate_cache(self):
        self._flat_cache = None

    def _ensure_capacity(self):
        """
        If total_transitions > capacity, pop oldest trajectories until within capacity.
        """
        while self.total_transitions > self.capacity and len(self.trajs) > 0:
            oldest = self.trajs.popleft()
            self.total_transitions -= len(oldest)
            self._invalidate_cache()
        # if still > capacity (e.g., single trajectory > capacity), trim oldest trajectory head
        if self.total_transitions > self.capacity and len(self.trajs) == 0 and len(self.current_traj) > 0:
            # trim head of current_traj (rare)
            excess = self.total_transitions - self.capacity
            if excess > 0:
                self.current_traj = self.current_traj[excess:]
                self.total_transitions -= excess
                self._invalidate_cache()

    # ---------- public API ----------
    def add(self, transition):
        """
        Add a transition to the buffer. If transition.done==True, closes the current trajectory
        and moves it to stored trajectories.
        """
        if not isinstance(transition, Transition):
            raise TypeError("Expected Transition namedtuple")

        self.current_traj.append(transition)
        self.total_transitions += 1
        self._invalidate_cache()

        if transition.done:
            # move current_traj to trajs
            self.trajs.append(list(self.current_traj))
            self.current_traj = []
            # ensure capacity (may pop old trajectories)
            self._ensure_capacity()

    def get_trajectories(self):
        """
        Return list of stored trajectories (each trajectory is list of Transition),
        in chronological order (oldest -> newest). Does NOT include the currently
        accumulating trajectory (unless you call flush_current=True).
        """
        return list(self.trajs)

    def get_all_flat(self, include_current=False):
        """
        Return flattened list of transitions in chronological order.
        If include_current=True, append current_traj at the end.
        """
        if self._flat_cache is not None and not include_current:
            return list(self._flat_cache)

        flat = []
        for traj in self.trajs:
            flat.extend(traj)
        if include_current and len(self.current_traj) > 0:
            flat.extend(self.current_traj)

        # cache flattened (without current) for reuse
        if not include_current:
            self._flat_cache = list(flat)
        return flat

    def sample(self, batch_size):
        """
        Sample transitions randomly (flat sampling).
        If not enough transitions, returns all transitions.
        """
        flat = self.get_all_flat()
        if len(flat) == 0:
            return []
        if len(flat) <= batch_size:
            return list(flat)
        idx = np.random.choice(len(flat), batch_size, replace=False)
        return [flat[i] for i in idx]

    def sample_trajectories(self, max_transitions=None):
        """
        Return trajectories sampled in chronological order until reaching ~max_transitions.
        Useful when you want to build minibatches of whole episodes.
        If max_transitions is None, returns all stored trajectories.
        """
        if max_transitions is None:
            return self.get_trajectories()

        selected = []
        count = 0
        for traj in reversed(self.trajs):  # start from newest
            if count + len(traj) > max_transitions and count > 0:
                break
            selected.append(traj)
            count += len(traj)
        return list(reversed(selected))   # return in chronological order

    def get_last_n(self, n):
        """
        Return last n transitions in chronological order (newest at end).
        Includes current_traj (most recent).
        """
        if n <= 0:
            return []
        flat = self.get_all_flat(include_current=True)
        if n >= len(flat):
            return list(flat)
        return flat[-n:]

    def clear(self):
        """Clear everything."""
        self.trajs.clear()
        self.current_traj = []
        self.total_transitions = 0
        self._invalidate_cache()

    def __len__(self):
        """Return total number of stored transitions (excluding any not-yet-closed trajectory if none)."""
        return int(self.total_transitions)

    def is_full(self):
        return self.total_transitions >= self.capacity

    def get_statistics(self):
        """Some quick stats."""
        flat = self.get_all_flat()
        if len(flat) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'is_full': False,
                'num_trajectories': len(self.trajs),
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0
            }
        rewards = np.array([t.reward for t in flat], dtype=np.float32)
        return {
            'size': len(flat),
            'capacity': self.capacity,
            'is_full': self.is_full(),
            'num_trajectories': len(self.trajs),
            'avg_reward': float(rewards.mean()),
            'min_reward': float(rewards.min()),
            'max_reward': float(rewards.max()),
            'std_reward': float(rewards.std())
        }
