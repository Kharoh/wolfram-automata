import torch
import collections
import numpy as np
from collections import Counter


class MetricsCalculator:
    """A class to calculate statistical metrics from the CA history."""

    @staticmethod
    def calculate_density(state):
        """
        Calculates the density of '1's (ρ) in a given state.

        Args:
            state (torch.Tensor): 1D tensor of cell states (0 or 1)
        Returns:
            float: The density of '1's in the state
        """
        return torch.mean(state.float()).item()

    @staticmethod
    def calculate_sequence_density(state):
        """
        Calculates the sequence density Q_i(n) for a given state.
        Q_0(n): Fraction of runs of 0s of length n
        Q_1(n): Fraction of runs of 1s of length n

        Args:
            state (torch.Tensor): 1D tensor of cell states (0 or 1)
        Returns:
            tuple: (q0_density, q1_density) where each is a dict mapping run length n to density
        """
        size = state.shape[0]
        if size == 0:
            return {}, {}
        run_values, run_lengths = torch.unique_consecutive(state, return_counts=True)
        q0, q1 = collections.defaultdict(int), collections.defaultdict(int)
        for v, n in zip(run_values.tolist(), run_lengths.tolist()):
            if v == 0:
                q0[n] += 1
            else:
                q1[n] += 1
        q0_density = {n: count / size for n, count in q0.items()}
        q1_density = {n: count / size for n, count in q1.items()}
        return q0_density, q1_density

    @staticmethod
    def calculate_triangular_density(history, max_n=11):
        """
        Calculates the density of downward-pointing triangles in the space-time history.
        Uses 2D convolution with triangle-shaped kernels.

        Args:
            history (torch.Tensor): 2D tensor of shape (generations, size)
            max_n (int): Maximum triangle size to consider (must be odd)
        Returns:
            tuple: (t0_density, t1_density) where each is a dict mapping triangle size n to density
        """
        import torch.nn.functional as F

        generations, size = history.shape
        device = history.device
        t_counts = {0: collections.defaultdict(int), 1: collections.defaultdict(int)}
        t0_density, t1_density = {}, {}
        for n in range(3, max_n + 1, 2):
            height = (n + 1) // 2
            if generations < height or size < n:
                continue
            mask = torch.zeros((height, n), dtype=history.dtype, device=device)
            for h in range(height):
                mask[h, h : n - h] = 1
            input_ = history.unsqueeze(0).unsqueeze(0).float()
            mask_1 = mask.unsqueeze(0).unsqueeze(0).float()
            out_1 = F.conv2d(input_, mask_1, stride=1)
            out_0 = F.conv2d(1 - input_, mask_1, stride=1)
            triangle_area = mask.sum().item()
            match_1 = out_1[0, 0] == triangle_area
            match_0 = out_0[0, 0] == triangle_area
            t_counts[1][n] = match_1.sum().item()
            t_counts[0][n] = match_0.sum().item()
            num_possible_locations = (generations - height + 1) * (size - n + 1)
            t1_density[n] = (
                t_counts[1][n] / num_possible_locations
                if num_possible_locations > 0
                else 0.0
            )
            t0_density[n] = (
                t_counts[0][n] / num_possible_locations
                if num_possible_locations > 0
                else 0.0
            )
        return t0_density, t1_density

    @staticmethod
    def calculate_correlation(state, distance):
        """
        Calculates the two-point correlation for a given distance.
        C(r) = <s_i * s_{i+r}> - rho^2

        Args:
            state (torch.Tensor): 1D tensor of cell states (0 or 1)
            distance (int): Distance r between cells
        Returns:
            float: The two-point correlation at the given distance
        """
        rho = MetricsCalculator.calculate_density(state)
        state_float = state.float()
        rolled_state = torch.roll(state_float, shifts=-distance, dims=0)
        correlation = torch.mean(state_float * rolled_state) - rho**2
        return correlation

    @staticmethod
    def calculate_average_correlation(states, max_distance):
        """
        Calculates the average two-point correlation function over a batch of states.
        Args:
            states (torch.Tensor or list of torch.Tensor): 2D tensor (n_states, state_length) or list of 1D tensors
            max_distance (int): Maximum distance for correlation calculation
        Returns:
            avg_correlations (dict): Mapping from distance to average correlation value
        """
        # If input is a list, fall back to old method
        if isinstance(states, list):
            correlations_sum = collections.defaultdict(float)
            n_states = len(states)
            for final_state in states:
                for r in range(1, max_distance + 1):
                    corr = MetricsCalculator.calculate_correlation(final_state, r)
                    correlations_sum[r] += corr
            avg_correlations = {
                r: correlations_sum[r] / n_states for r in range(1, max_distance + 1)
            }
            return avg_correlations

        # Vectorized version for torch.Tensor input
        # states: shape (n_states, state_length)
        n_states = states.shape[0]
        state_float = states.float()
        rho = state_float.mean(dim=1, keepdim=True)  # shape (n_states, 1)
        avg_correlations = {}
        for r in range(1, max_distance + 1):
            rolled = torch.roll(state_float, shifts=-r, dims=1)
            mean_prod = (state_float * rolled).mean(dim=1)  # shape (n_states,)
            corr = mean_prod - (rho.squeeze(1) ** 2)
            avg_correlations[r] = corr.mean().item()
        return avg_correlations

    @staticmethod
    def calculate_lempel_ziv_complexity(word):
        """
        Computes the Lempel-Ziv complexity of a given word.
        Attention: This implementation corresponds to the original Lempel-Ziv algorithm,
        the candidate is searched in the prefix of the word, which is the extension minus the last character. It is not searched in simply word[:i] but in word[:i + v - 1].

        I think the LZ77 variant is using word[:i] instead of word[:i + v - 1].
        """
        n = len(word)
        i = 0
        count = 0
        while i < n:
            v = 1
            while i + v <= n and word[i : i + v] in word[: i + v - 1]:
                v += 1
            count += 1
            i += v
        return count

    @staticmethod
    def calculate_normalized_lempel_ziv_complexity(word):
        """
        Computes the normalized Lempel-Ziv complexity of a given word.
        Normalization is done by dividing the complexity by the length of the word.
        Args:
            word (str): Binary string (0s and 1s)
        Returns:
            float: Normalized Lempel-Ziv complexity
        """
        complexity = MetricsCalculator.calculate_lempel_ziv_complexity(word)
        return complexity / len(word) if len(word) > 0 else 0.0

    @staticmethod
    def calculate_average_lempel_ziv_complexity(words):
        """
        Computes the average Lempel-Ziv complexity for a list of words.
        Args:
            words (list of str): List of binary strings (0s and 1s)
        Returns:
            float: Average normalized Lempel-Ziv complexity
        """
        complexities = [
            MetricsCalculator.calculate_lempel_ziv_complexity(word) for word in words
        ]
        return sum(complexities) / len(complexities) if complexities else 0.0

    @staticmethod
    def calculate_average_normalized_lempel_ziv_complexity(words):
        """
        Computes the average normalized Lempel-Ziv complexity for a list of words.
        Args:
            words (list of str): List of binary strings (0s and 1s)
        Returns:
            float: Average normalized Lempel-Ziv complexity
        """
        complexities = [
            MetricsCalculator.calculate_normalized_lempel_ziv_complexity(word)
            for word in words
        ]
        return sum(complexities) / len(complexities) if complexities else 0.0

    @staticmethod
    def calculate_normalized_entropy(final_states, size):
        """
        Calculates the normalized entropy (per cell) from a list of final states.
        Args:
            final_states (list): List of final state tuples
            size (int): Number of cells in the automaton
        Returns:
            entropy_per_cell (float): Entropy per cell
            counts (Counter): Counter of unique final states
            probabilities (np.ndarray): Probabilities of each unique state
        """
        counts = Counter(final_states)
        total = sum(counts.values())
        probabilities = np.array([count / total for count in counts.values()])
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = size  # log2(2^size) = size
        min_entropy = 0.0
        entropy_per_cell = entropy / size
        return entropy_per_cell, counts, probabilities, min_entropy, max_entropy
