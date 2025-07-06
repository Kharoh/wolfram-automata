import torch
import matplotlib.pyplot as plt
import collections

class WolframCA:
    """
    A PyTorch implementation of Wolfram's elementary cellular automata.

    This class simulates the evolution of a 1D cellular automaton based on a
    given Wolfram rule (0-255). It uses tensor operations for efficient
    computation of the automaton's state transitions with periodic
    boundary conditions.
    """
    def __init__(self, rule_number, size=101, initial_condition='single', device='cpu'):
        """
        Initializes the cellular automaton.

        Args:
            rule_number (int): The Wolfram rule number (0-255).
            size (int): The number of cells in the automaton.
            initial_condition (str): The initial state type. Can be 'single' or 'random'.
            device (str): The torch device to run computations on ('cpu' or 'cuda').
        """
        if not 0 <= rule_number <= 255:
            raise ValueError("Rule number must be between 0 and 255.")

        self.rule_number = rule_number
        self.size = size
        self.device = device

        rule_bin = format(rule_number, '08b')
        self.rule = torch.tensor([int(bit) for bit in rule_bin], dtype=torch.int8, device=self.device).flip(0)

        if initial_condition == 'single':
            self.state = torch.zeros(size, dtype=torch.int8, device=self.device)
            self.state[size // 2] = 1
        elif initial_condition == 'random':
            self.state = torch.randint(0, 2, (size,), dtype=torch.int8, device=self.device)
        else:
            raise ValueError("initial_condition must be 'single' or 'random'.")

        self.conversion_weights = torch.tensor([4, 2, 1], dtype=torch.int8, device=self.device)

    def step(self):
        """Computes the next state of the automaton using vectorized operations."""
        left_neighbors = torch.roll(self.state, shifts=1, dims=0)
        right_neighbors = torch.roll(self.state, shifts=-1, dims=0)
        center = self.state

        neighborhoods = torch.stack([left_neighbors, center, right_neighbors])
        indices = torch.matmul(neighborhoods.T, self.conversion_weights).long()
        self.state = self.rule[indices]

    def run(self, generations):
        """
        Runs the simulation for a specified number of generations.

        Returns:
            torch.Tensor: A tensor containing the history of states.
        """
        history = torch.empty((generations, self.size), dtype=torch.int8, device=self.device)
        for i in range(generations):
            history[i] = self.state
            self.step()
        return history

class MetricsCalculator:
    """A class to calculate statistical metrics from the CA history."""

    @staticmethod
    def calculate_density(state):
        """Calculates the density of '1's (ρ) in a given state."""
        return torch.mean(state.float()).item()

    @staticmethod
    def calculate_sequence_density(state):
        """
        Calculates the sequence density Q_i(n) for a given state.

        Q_i(n) is the density of sequences of exactly n adjacent sites with the
        same value i, bordered by sites with a different value.
        """
        size = state.shape[0]
        if size == 0:
            return {}, {}

        # Detect boundaries where the value changes (including wrap-around)
        boundaries = state != torch.roll(state, shifts=-1, dims=0)
        boundary_indices = torch.where(boundaries)[0]

        # If the state is homogeneous, all cells form a single run
        if boundary_indices.shape[0] == 0:
            val = state[0].item()
            density = {size: 1.0}
            return (density, {}) if val == 0 else ({}, density)
            
        # Calculate run lengths and their corresponding values
        run_lengths = torch.diff(boundary_indices, append=boundary_indices.new_tensor([boundary_indices[0] + size]))
        run_values = state[boundary_indices]

        q0, q1 = collections.defaultdict(int), collections.defaultdict(int)
        for length, value in zip(run_lengths, run_values):
            n, v = length.item(), value.item()
            if v == 0:
                q0[n] += 1
            else:
                q1[n] += 1
        
        # Normalize counts by the total size to get densities
        q0_density = {n: count / size for n, count in q0.items()}
        q1_density = {n: count / size for n, count in q1.items()}
        
        return q0_density, q1_density

    @staticmethod
    def calculate_triangular_density(history, max_n=11):
        """
        Calculates the density of downward-pointing triangles in the space-time history.

        T_i(n) is the density of triangles of value i with base length n,
        surrounded by the opposite value.
        """
        generations, size = history.shape
        t_counts = {0: collections.defaultdict(int), 1: collections.defaultdict(int)}

        # Iterate through possible odd base lengths for simple symmetric triangles
        for n in range(3, max_n + 1, 2):
            height = (n + 1) // 2
            if generations < height:
                continue

            for t in range(generations - height + 1):
                for j in range(size - n):
                    sub_grid = history[t:t+height, j:j+n]
                    
                    # Check for a triangle of 1s bordered by 0s
                    is_triangle_1 = True
                    # Check for a triangle of 0s bordered by 1s
                    is_triangle_0 = True

                    for h in range(height):
                        row_start = h
                        row_end = n - 1 - h
                        for w in range(n):
                            is_in_triangle = (w >= row_start and w <= row_end)
                            cell_val = sub_grid[h, w]
                            
                            # Check triangle of 1s
                            if (is_in_triangle and cell_val == 0) or (not is_in_triangle and cell_val == 1):
                                is_triangle_1 = False
                            
                            # Check triangle of 0s
                            if (is_in_triangle and cell_val == 1) or (not is_in_triangle and cell_val == 0):
                                is_triangle_0 = False
                        
                        if not is_triangle_1 and not is_triangle_0:
                            break
                    
                    if is_triangle_1:
                        t_counts[1][n] += 1
                    if is_triangle_0:
                        t_counts[0][n] += 1
        
        num_possible_locations = (generations - height + 1) * (size - n + 1) if generations >= height else 0
        t0_density = {n: count / num_possible_locations for n, count in t_counts[0].items()} if num_possible_locations > 0 else {}
        t1_density = {n: count / num_possible_locations for n, count in t_counts[1].items()} if num_possible_locations > 0 else {}

        return t0_density, t1_density

def plot_history(history, title):
    """Plots the evolution of the cellular automaton."""
    fig_size_y = max(5, 10 * (history.shape[0] / history.shape[1]))
    plt.figure(figsize=(10, fig_size_y))
    plt.imshow(history.cpu().numpy(), cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Cell")
    plt.ylabel("Generation")
    plt.show()

def format_density_dict(d):
    """Formats a density dictionary for printing."""
    if not d:
        return "None"
    return ", ".join([f"n={k}: {v:.4f}" for k, v in sorted(d.items())])

if __name__ == '__main__':
    # --- Parameters ---
    SIZE = 201
    GENERATIONS = 100
    INITIAL_CONDITION = 'random' # Use 'random' or 'single'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Wolfram's Rule Categories for Demonstration ---
    RULE_CATEGORIES = {
        "Homogeneous (Class 1)": [0, 128, 255],
        "Periodic (Class 2)": [108, 170, 240],
        "Chaotic (Class 3)": [30, 90, 150],
        "Complex (Class 4)": [54, 110, 182]
    }

    # --- Run and Plot Demonstrations ---
    for category, rules in RULE_CATEGORIES.items():
        print(f"\n--- Demonstrating {category} Rules (Initial: {INITIAL_CONDITION}) ---")
        for rule in rules:
            ca = WolframCA(
                rule_number=rule,
                size=SIZE,
                initial_condition=INITIAL_CONDITION,
                device=DEVICE
            )
            history = ca.run(generations=GENERATIONS)
            final_state = history[-1]

            # --- Calculate Metrics ---
            density = MetricsCalculator.calculate_density(final_state)
            q0, q1 = MetricsCalculator.calculate_sequence_density(final_state)
            t0, t1 = MetricsCalculator.calculate_triangular_density(history)

            # --- Print Results ---
            print(f"\nMetrics for Rule {rule}:")
            print(f"  - Final State Density (ρ): {density:.4f}")
            print("  - Sequence Density Q(n) for final state:")
            print(f"    - Runs of 0s: {format_density_dict(q0)}")
            print(f"    - Runs of 1s: {format_density_dict(q1)}")
            print("  - Triangular Density T(n) in history:")
            print(f"    - Triangles of 0s: {format_density_dict(t0)}")
            print(f"    - Triangles of 1s: {format_density_dict(t1)}")

            plot_title = f"Wolfram Rule {rule} ({category}, Initial: {INITIAL_CONDITION})"
            plot_history(history, plot_title)
