"""
Prioritized Experience Replay (PER) — Sun et al. 2022, Section 4.3

Implements SumTree for O(log N) proportional sampling and a drop-in
replacement for the simple ReplayBuffer used by the BDQ agent.

References:
  - Schaul et al. (2016) "Prioritized Experience Replay"
  - Sun et al. (2022) "DeepScalper" Section 4.3
"""
import numpy as np
from typing import Tuple, Optional


class SumTree:
    """Binary sum-tree stored in a flat array for O(log N) priority sampling.

    Tree Layout (capacity=4):
        Index:    0         <- root (sum of all)
                /   \
               1     2
              / \   / \
             3   4 5   6   <- leaves (priorities)
        Data:  [0] [1] [2] [3]  <- circular data buffer

    Leaf nodes start at index (capacity - 1).
    """

    __slots__ = ('capacity', 'tree', 'data', 'data_pointer', 'size')

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float, data) -> int:
        """Add a new transition with the given priority. Returns tree index."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_idx

    def update(self, tree_idx: int, priority: float):
        """Update leaf priority and propagate change up to root."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, cumsum: float) -> Tuple[int, float, object]:
        """Retrieve leaf by cumulative sum (proportional sampling).

        Returns:
            (tree_index, priority, data)
        """
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                # Reached leaf level
                break
            if cumsum <= self.tree[left]:
                parent_idx = left
            else:
                cumsum -= self.tree[left]
                parent_idx = right

        leaf_idx = parent_idx
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self) -> float:
        """Sum of all priorities (root node)."""
        return self.tree[0]

    def min_priority(self) -> float:
        """Minimum priority among active leaves."""
        if self.size == 0:
            return 0.0
        leaf_start = self.capacity - 1
        active_leaves = self.tree[leaf_start:leaf_start + self.size]
        return float(np.min(active_leaves))


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree.

    Drop-in replacement for ReplayBuffer with additional PER semantics:
      - push() stores transitions with max_priority (new experiences are important)
      - sample() returns IS weights alongside the batch
      - update_priorities() adjusts priorities based on TD error

    Args:
        capacity: Maximum buffer size
        alpha: Prioritization exponent (0 = uniform, 1 = full priority)
        beta_start: Initial importance-sampling correction exponent
        beta_frames: Number of frames to anneal beta from beta_start to 1.0
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0  # For beta annealing

        self._max_priority = 1.0  # Initial max priority for new transitions
        self._epsilon = 1e-6      # Small constant to prevent zero priorities

    @property
    def beta(self) -> float:
        """Current IS correction exponent (annealed from beta_start → 1.0)."""
        fraction = min(self.frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def push(self, state, action, reward, next_state, done, aux_target=0.0):
        """Store transition with max_priority (ensures new experiences get sampled).
        
        Uses _max_priority directly (already alpha-exponentiated in update_priorities).
        """
        transition = (state, action, reward, next_state, done, aux_target)
        self.tree.add(self._max_priority, transition)

    def sample(self, batch_size: int):
        """Sample batch proportional to priorities.

        Returns:
            (states, actions, rewards, next_states, dones, aux_targets,
             indices, is_weights)

        indices: tree indices for update_priorities()
        is_weights: importance-sampling weights (normalized so max = 1.0)
        """
        indices = []
        priorities = []
        batch = []

        total = self.tree.total()
        if total <= 0:
            total = 1e-6  # Guard against empty buffer

        # Stratified sampling: divide [0, total) into batch_size segments
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = np.random.uniform(low, high)
            tree_idx, priority, data = self.tree.get(cumsum)

            # Guard against zero priority (shouldn't happen but be safe)
            if priority <= 0:
                priority = self._epsilon

            indices.append(tree_idx)
            priorities.append(priority)
            batch.append(data)

        # Compute IS weights: w_i = (N * P(i))^(-beta) / max_w
        priorities_arr = np.array(priorities, dtype=np.float64)
        N = self.tree.size
        probs = priorities_arr / total

        beta = self.beta

        # IS weights
        weights = (N * probs) ** (-beta)
        # Normalize by max weight for stability
        weights /= weights.max()

        # Advance frame counter for beta annealing
        self.frame += batch_size

        # Unpack batch
        state, action, reward, next_state, done, aux_target = zip(*batch)

        return (state, action, reward, next_state, done, aux_target,
                np.array(indices, dtype=np.int64),
                weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors.

        priority_i = (|td_error_i| + epsilon) ^ alpha
        """
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self._epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            # FIX FIND-2: Decay max_priority to prevent permanent bias from early high-error samples
            self._max_priority = max(self._max_priority * 0.999, priority)

    def __len__(self) -> int:
        return self.tree.size
