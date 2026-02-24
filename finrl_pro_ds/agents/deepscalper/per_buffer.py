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

    Tree Layout (capacity=4)::

        Index:    0         <- root (sum of all)
                /   \\
               1     2
              / \\   / \\
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

    def get_batch(self, cumsums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized batch retrieval — process all queries in lockstep through tree levels.

        Args:
            cumsums: 1D array of cumulative sum queries (length N).

        Returns:
            (leaf_indices, priorities) — both shape (N,).
            Data must be retrieved separately via ``self.data[leaf_idx - (capacity - 1)]``.
        """
        n = len(cumsums)
        parent_idx = np.zeros(n, dtype=np.int64)
        remaining = cumsums.copy()
        tree_len = len(self.tree)

        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            # All queries that have reached leaf level stop advancing
            at_leaf = left >= tree_len
            if at_leaf.all():
                break
            left_vals = np.where(at_leaf, 0.0, self.tree[np.minimum(left, tree_len - 1)])
            go_right = (~at_leaf) & (remaining > left_vals)
            # Go left
            parent_idx = np.where(at_leaf, parent_idx,
                         np.where(go_right, right, left))
            remaining = np.where(go_right, remaining - left_vals, remaining)

        leaf_indices = parent_idx
        priorities = self.tree[leaf_indices]
        return leaf_indices, priorities

    def batch_update(self, tree_indices: np.ndarray, new_priorities: np.ndarray):
        """Vectorized priority update — set leaves and propagate changes up.

        Handles duplicate indices correctly: when the same leaf appears multiple
        times, the last value wins and the net change is propagated once.
        """
        tree_indices = np.asarray(tree_indices, dtype=np.int64)
        new_priorities = np.asarray(new_priorities, dtype=np.float64)

        # Deduplicate: keep last occurrence for each index (last-write-wins)
        unique_indices, first_occ = np.unique(tree_indices[::-1], return_index=True)
        # first_occ gives positions in reversed array; convert to original positions
        last_occ = len(tree_indices) - 1 - first_occ
        final_priorities = new_priorities[last_occ]

        # Compute deltas from current tree values
        old_priorities = self.tree[unique_indices]
        changes = final_priorities - old_priorities
        self.tree[unique_indices] = final_priorities

        # Propagate up level by level
        current = unique_indices.copy()
        while True:
            parent = (current - 1) // 2
            # Filter out entries that have already reached or passed root
            valid = current > 0
            if not valid.any():
                break
            np.add.at(self.tree, parent[valid], changes[valid])
            current = parent

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
        """Sample batch proportional to priorities (vectorized).

        Returns:
            (states, actions, rewards, next_states, dones, aux_targets,
             indices, is_weights)

        indices: tree indices for update_priorities()
        is_weights: importance-sampling weights (normalized so max = 1.0)
        """
        total = self.tree.total()
        if total <= 0:
            total = 1e-6  # Guard against empty buffer

        # Stratified sampling: divide [0, total) into batch_size segments
        segment = total / batch_size
        lows = np.arange(batch_size, dtype=np.float64) * segment
        highs = lows + segment
        cumsums = np.random.uniform(lows, highs)

        # Vectorized tree traversal
        leaf_indices, priorities_arr = self.tree.get_batch(cumsums)

        # Guard against zero priority
        priorities_arr = np.maximum(priorities_arr, self._epsilon)

        # Retrieve data (Python objects — must iterate)
        leaf_offset = self.tree.capacity - 1
        data_indices = leaf_indices - leaf_offset
        batch = [self.tree.data[int(di)] for di in data_indices]

        # Compute IS weights: w_i = (N * P(i))^(-beta) / max_w
        N = self.tree.size
        probs = priorities_arr / total
        beta = self.beta
        weights = (N * probs) ** (-beta)
        weights /= weights.max()

        # Advance frame counter for beta annealing
        self.frame += batch_size

        # Unpack batch
        state, action, reward, next_state, done, aux_target = zip(*batch)

        return (state, action, reward, next_state, done, aux_target,
                leaf_indices.astype(np.int64),
                weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors (vectorized).

        priority_i = (|td_error_i| + epsilon) ^ alpha
        """
        indices = np.asarray(indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float64)
        priorities = (np.abs(td_errors) + self._epsilon) ** self.alpha
        self.tree.batch_update(indices, priorities)
        # FIX PERF-6: Standard max_priority update (Schaul et al. 2016).
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def __len__(self) -> int:
        return self.tree.size
