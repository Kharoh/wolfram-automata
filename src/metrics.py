import torch
import collections


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
