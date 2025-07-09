import torch

from src.metrics import MetricsCalculator


class WolframCA:
    """
    A PyTorch implementation of Wolfram's elementary cellular automata.
    """

    def __init__(self, rule_number, size=101, initial_condition="single", device="cpu"):
        """
        Initializes the cellular automaton.

        Args:
            rule_number (int): The Wolfram rule number (0-255).
            size (int): The number of cells in the automaton. If 0, runs in infinite mode.
            initial_condition (str): The initial state type. Can be 'single' or 'random'.
            device (str): The torch device to run computations on ('cpu' or 'cuda').
        Raises:
            ValueError: If rule_number is not in [0, 255] or initial_condition is invalid.
        """
        if not 0 <= rule_number <= 255:
            raise ValueError("Rule number must be between 0 and 255.")
        self.rule_number = rule_number
        self.size = size
        self.device = device
        self.infinite_mode = size == 0
        rule_bin = format(rule_number, "08b")
        self.rule = torch.tensor(
            [int(bit) for bit in rule_bin], dtype=torch.int8, device=self.device
        ).flip(0)
        self.conversion_weights = torch.tensor(
            [4, 2, 1], dtype=torch.int8, device=self.device
        )
        if self.infinite_mode:
            # Start with a single cell in the center
            if initial_condition == "single":
                self.state = torch.tensor([1], dtype=torch.int8, device=self.device)
            elif initial_condition == "random":
                self.state = torch.randint(
                    0, 2, (1,), dtype=torch.int8, device=self.device
                )
            else:
                raise ValueError("initial_condition must be 'single' or 'random'.")
        else:
            if initial_condition == "single":
                self.state = torch.zeros(size, dtype=torch.int8, device=self.device)
                self.state[size // 2] = 1
            elif initial_condition == "random":
                self.state = torch.randint(
                    0, 2, (size,), dtype=torch.int8, device=self.device
                )
            else:
                raise ValueError("initial_condition must be 'single' or 'random'.")

    def step(self):
        """
        Computes the next state of the automaton using vectorized operations.
        Updates the state in-place.
        """
        if self.infinite_mode:
            # Pad with 0 on both sides
            padded = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int8, device=self.device),
                    self.state,
                    torch.zeros(1, dtype=torch.int8, device=self.device),
                ]
            )
            left_neighbors = torch.roll(padded, shifts=1, dims=0)
            right_neighbors = torch.roll(padded, shifts=-1, dims=0)
            center = padded
            neighborhoods = torch.stack([left_neighbors, center, right_neighbors])
            indices = torch.matmul(neighborhoods.T, self.conversion_weights).long()
            # Only take the new state for the padded region (i.e., all cells)
            self.state = self.rule[indices]
        else:
            left_neighbors = torch.roll(self.state, shifts=1, dims=0)
            right_neighbors = torch.roll(self.state, shifts=-1, dims=0)
            center = self.state
            neighborhoods = torch.stack([left_neighbors, center, right_neighbors])
            indices = torch.matmul(neighborhoods.T, self.conversion_weights).long()
            self.state = self.rule[indices]

    def run(self, generations):
        """
        Runs the simulation for a specified number of generations.

        Args:
            generations (int): Number of generations to simulate.
        Returns:
            torch.Tensor: A tensor containing the history of states (shape: generations x size) or a list of tensors for infinite mode.
        """
        if self.infinite_mode:
            history = []
            for _ in range(generations):
                history.append(self.state.clone())
                self.step()
            return history
        else:
            history = torch.empty(
                (generations, self.size), dtype=torch.int8, device=self.device
            )
            for i in range(generations):
                history[i] = self.state
                self.step()
            return history

    def run_experiments_to_csv(
        self,
        generations,
        runs,
        n_first,
        n_last,
        initial_condition=None,
        output_dir=None,
    ):
        """
        Runs N experiments, saving the first n_first and last n_last states of each run to a CSV file.
        Each line in the CSV contains the first n_first and last n_last states (flattened, comma-separated) for a run.
        Args:
            generations (int): Number of generations per run.
            runs (int): Number of experiments to run.
            n_first (int): Number of initial states to save per run.
            n_last (int): Number of final states to save per run.
            initial_condition (str, optional): If provided, overrides the instance's initial_condition for each run.
            output_dir (str, optional): Directory to save the CSV file. Defaults to current directory.
        """
        import os
        import csv

        ic = (
            initial_condition
            if initial_condition is not None
            else ("random" if torch.sum(self.state) > 1 else "single")
        )
        filename = f"{self.rule_number}_{ic}_s{self.size}_g{generations}_r{runs}_f{n_first}_l{n_last}.csv"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for _ in range(runs):
                # Reset state for each run
                if self.infinite_mode:
                    if ic == "single":
                        self.state = torch.tensor(
                            [1], dtype=torch.int8, device=self.device
                        )
                    elif ic == "random":
                        self.state = torch.randint(
                            0, 2, (1,), dtype=torch.int8, device=self.device
                        )
                    else:
                        raise ValueError(
                            "initial_condition must be 'single' or 'random'."
                        )
                    history = self.run(generations)
                    # For infinite mode, states are of increasing length
                    first_states = torch.cat([h for h in history[:n_first]]).tolist()
                    last_states = torch.cat([h for h in history[-n_last:]]).tolist()
                else:
                    if ic == "single":
                        self.state = torch.zeros(
                            self.size, dtype=torch.int8, device=self.device
                        )
                        self.state[self.size // 2] = 1
                    elif ic == "random":
                        self.state = torch.randint(
                            0, 2, (self.size,), dtype=torch.int8, device=self.device
                        )
                    else:
                        raise ValueError(
                            "initial_condition must be 'single' or 'random'."
                        )
                    history = self.run(generations)
                    first_states = ",".join(
                        [
                            "".join([str(b) for b in x])
                            for x in history[:n_first].tolist()
                        ]
                    )
                    last_states = ",".join(
                        [
                            "".join([str(b) for b in x])
                            for x in history[-n_last:].tolist()
                        ]
                    )
                writer.writerow(",".join([first_states, last_states]).split(","))
        print(f"Saved experiments to {filepath}")

    def get_properties_to_md(self):
        """
        Checks all properties of the automaton and returns them as a markdown string.
        """
        from src.properties import is_legal_wolfram, is_additive, is_additive_by_test

        rule = self.rule_number
        legal = is_legal_wolfram(rule)
        additive = is_additive(rule)
        additive_by_test = is_additive_by_test(rule)

        md = f"""## Automaton Properties\n\n\
- **Legal Wolfram Rule:** {'Yes' if legal else 'No'}\n\
- **Additive (by lookup):** {'Yes' if additive else 'No'}\n\
- **Additive (by test):** {'Yes' if additive_by_test else 'No'}\n\
\n---\n\n\
| Property                | Value |
|-------------------------|-------|
| Legal Wolfram Rule      | {'Yes' if legal else 'No'} |
| Additive (by lookup)    | {'Yes' if additive else 'No'} |
| Additive (by test)      | {'Yes' if additive_by_test else 'No'} |
"""
        return md

    def run_correlations_to_md(
        self,
        input_csv,
        max_distance=None,
        output_dir=None,
    ):
        """
        Runs multiple random initializations for the rule and computes the average two-point correlation function.
        Saves the results to a markdown file and image files imported in the markdown.

        Args:
            input_csv (str): Path to the CSV file the initial states and final states from which to compute the correlations.
            generations (int): Number of generations to run.
            runs (int): Number of random initializations.
            max_distance (int, optional): Maximum distance for correlation calculation. Defaults to None.
            output_dir (str, optional): Directory to save the markdown file. Defaults to None.
        """
        import os
        import re
        import csv
        import matplotlib.pyplot as plt

        # --- Extract n_first and n_last from filename ---
        filename = os.path.basename(input_csv)
        match = re.match(r".*_f(\d+)_l(\d+)\.csv$", filename)
        if match:
            n_first_row = int(match.group(1))
            n_last_row = int(match.group(2))
        else:
            raise ValueError(
                f"Could not extract n_first and n_last from filename: {filename}"
            )
        # --- Read CSV file and parse states ---
        all_first_states = []
        all_last_states = []
        with open(input_csv, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Each row: [first_state_1, ..., first_state_n, last_state_1, ..., last_state_n]
                first_states = row[:n_first_row]
                last_states = row[n_first_row : n_first_row + n_last_row]
                all_first_states.append(first_states)
                all_last_states.append(last_states)
        # Now all_first_states and all_last_states are lists of lists of string bit patterns
        # Convert to tensors of shape [n_first_states, n_runs, size]
        n_runs = len(all_first_states)
        n_first_states = len(all_first_states[0]) if n_runs > 0 else 0
        n_last_states = len(all_last_states[0]) if n_runs > 0 else 0
        size = len(all_first_states[0][0]) if n_runs > 0 and n_first_states > 0 else 0
        all_first_states_tensor = torch.tensor(
            [
                [[int(bit) for bit in state] for state in run_states]
                for run_states in all_first_states
            ],
            dtype=torch.int8,
        )  # shape: [n_runs, n_first_states, size]
        all_last_states_tensor = torch.tensor(
            [
                [[int(bit) for bit in state] for state in run_states]
                for run_states in all_last_states
            ],
            dtype=torch.int8,
        )  # shape: [n_runs, n_last_states, size]
        # Transpose to [n_first_states, n_runs, size] for first states, and same for last states
        all_first_states_tensor = all_first_states_tensor.permute(1, 0, 2)
        all_last_states_tensor = all_last_states_tensor.permute(1, 0, 2)
        # Calculate average correlations for each state index (across runs) in both first and last states
        avg_correlations_first = []
        avg_correlations_last = []
        state_length = (
            all_first_states_tensor.shape[2]
            if all_first_states_tensor.shape[0] > 0
            else all_last_states_tensor.shape[2]
        )
        # Determine max_distance if not provided
        if max_distance is None:
            max_distance = min(20, state_length // 2)
        # For each state index in first and last, calculate average correlation across runs
        for i in range(all_first_states_tensor.shape[0]):
            avg_corr = MetricsCalculator.calculate_average_correlation(
                all_first_states_tensor[i], max_distance
            )
            avg_correlations_first.append(avg_corr)
        for i in range(all_last_states_tensor.shape[0]):
            avg_corr = MetricsCalculator.calculate_average_correlation(
                all_last_states_tensor[i], max_distance
            )
            avg_correlations_last.append(avg_corr)
        # avg_correlations_first and avg_correlations_last are lists of dicts (distance -> value)

        # Plot first states correlations
        plt.figure(figsize=(10, 6))
        for idx, corr_dict in enumerate(avg_correlations_first):
            distances = list(corr_dict.keys())
            values = [corr_dict[d] for d in distances]
            plt.plot(distances, values, label=f"First State {idx+1}", alpha=0.7)
        plt.title(f"First States Correlations (Rule {self.rule_number})")
        plt.xlabel("Distance r")
        plt.ylabel("C(r)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        base_csv = os.path.basename(input_csv).replace(".csv", "")
        first_img_path = (
            os.path.join(output_dir, f"{base_csv}_first_correlations.png")
            if output_dir
            else f"{base_csv}_first_correlations.png"
        )
        plt.savefig(first_img_path)
        plt.close()
        # Plot last states correlations
        plt.figure(figsize=(10, 6))
        for idx, corr_dict in enumerate(avg_correlations_last):
            distances = list(corr_dict.keys())
            values = [corr_dict[d] for d in distances]
            plt.plot(distances, values, label=f"Last State {idx+1}", alpha=0.7)
        plt.title(f"Last States Correlations (Rule {self.rule_number})")
        plt.xlabel("Distance r")
        plt.ylabel("C(r)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        last_img_path = (
            os.path.join(output_dir, f"{base_csv}_last_correlations.png")
            if output_dir
            else f"{base_csv}_last_correlations.png"
        )
        plt.savefig(last_img_path)
        plt.close()

        # --- Generate markdown string showcasing the two PNG images ---
        md = f"""# Two-Point Correlation Functions for Rule {self.rule_number}\n\n \
This section presents the average two-point correlation functions for the first and last states across all runs.\n\n \
## First States Correlations\n\n \
![First States Correlations]({os.path.relpath(first_img_path, output_dir) if output_dir else first_img_path})\n\n \
## Last States Correlations\n\n \
![Last States Correlations]({os.path.relpath(last_img_path, output_dir) if output_dir else last_img_path})\n"""

        return md

    def run_complexity_to_md(
        self,
        input_csv,
        output_dir=None,
    ):
        """
        Computes and plots the average normalized Lempel-Ziv complexity for each epoch index (first and last) across all runs.
        The plot shows the average complexity for the first epochs (on the left) and the last epochs (on the right).
        Returns a markdown string showcasing the image.

        Args:
            input_csv (str): Path to the CSV file containing the initial and final states.
            output_dir (str, optional): Directory to save the markdown file and image. Defaults to None.
        """
        import os
        import re
        import csv
        import matplotlib.pyplot as plt
        from src.metrics import MetricsCalculator

        # --- Extract n_first and n_last from filename ---
        filename = os.path.basename(input_csv)
        match = re.match(r".*_f(\d+)_l(\d+)\.csv$", filename)
        if match:
            n_first_row = int(match.group(1))
            n_last_row = int(match.group(2))
        else:
            raise ValueError(
                f"Could not extract n_first and n_last from filename: {filename}"
            )
        # --- Read CSV file and parse states ---
        all_first_states = []
        all_last_states = []
        with open(input_csv, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                first_states = row[:n_first_row]
                last_states = row[n_first_row : n_first_row + n_last_row]
                all_first_states.append(first_states)
                all_last_states.append(last_states)
        # Now all_first_states and all_last_states are lists of lists of string bit patterns
        n_runs = len(all_first_states)
        n_first_states = len(all_first_states[0]) if n_runs > 0 else 0
        n_last_states = len(all_last_states[0]) if n_runs > 0 else 0

        # Compute average normalized Lempel-Ziv complexity for each epoch index
        avg_complexities_first = []
        avg_complexities_last = []
        for i in range(n_first_states):
            words = [all_first_states[run][i] for run in range(n_runs)]
            avg_c = (
                MetricsCalculator.calculate_average_normalized_lempel_ziv_complexity(
                    words
                )
            )
            avg_complexities_first.append(avg_c)
        for i in range(n_last_states):
            words = [all_last_states[run][i] for run in range(n_runs)]
            avg_c = (
                MetricsCalculator.calculate_average_normalized_lempel_ziv_complexity(
                    words
                )
            )
            avg_complexities_last.append(avg_c)

        # Plot: x-axis is epoch index (first epochs on left, last epochs on right), y-axis is avg complexity
        plt.figure(figsize=(12, 6))
        x_first = list(range(1, n_first_states + 1))
        x_last = list(range(n_first_states + 1, n_first_states + n_last_states + 1))
        plt.plot(x_first, avg_complexities_first, label="First Epochs", marker="o")
        plt.plot(x_last, avg_complexities_last, label="Last Epochs", marker="o")
        plt.xlabel("Epoch Index")
        plt.ylabel("Average Normalized Lempel-Ziv Complexity")
        plt.title(f"Average Normalized Lempel-Ziv Complexity (Rule {self.rule_number})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        base_csv = os.path.basename(input_csv).replace(".csv", "")
        img_path = (
            os.path.join(output_dir, f"{base_csv}_lz_complexity.png")
            if output_dir
            else f"{base_csv}_lz_complexity.png"
        )
        plt.savefig(img_path)
        plt.close()

        # --- Generate markdown string showcasing the PNG image ---
        md = f"""# Lempel-Ziv Complexity for Rule {self.rule_number}\n\n \
This section presents the average normalized Lempel-Ziv complexity for the first and last epochs across all runs.\n\n \
![Lempel-Ziv Complexity]({os.path.relpath(img_path, output_dir) if output_dir else img_path})\n"""
        return md

    def run_entropy_to_md(
        self,
        input_csv,
        output_dir=None,
    ):
        """
        Computes and plots the normalized entropy per cell for each epoch index (first and last) across all runs.
        The plot shows the entropy for the first epochs (on the left) and the last epochs (on the right).
        Returns a markdown string showcasing the image.

        Args:
            input_csv (str): Path to the CSV file containing the initial and final states.
            output_dir (str, optional): Directory to save the markdown file and image. Defaults to None.
        """
        from src.metrics import MetricsCalculator
        import os
        import re
        import csv
        import matplotlib.pyplot as plt

        # --- Extract n_first and n_last from filename ---
        filename = os.path.basename(input_csv)
        match = re.match(r".*_f(\d+)_l(\d+)\.csv$", filename)
        if match:
            n_first_row = int(match.group(1))
            n_last_row = int(match.group(2))
        else:
            raise ValueError(
                f"Could not extract n_first and n_last from filename: {filename}"
            )
        # --- Read CSV file and parse states ---
        all_first_states = []
        all_last_states = []
        with open(input_csv, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                n_first = n_first_row
                n_last = n_last_row
                first_states = row[:n_first]
                last_states = row[n_first : n_first + n_last]
                all_first_states.append(first_states)
                all_last_states.append(last_states)
        # Now all_first_states and all_last_states are lists of lists of string bit patterns
        n_runs = len(all_first_states)
        n_first_states = len(all_first_states[0]) if n_runs > 0 else 0
        n_last_states = len(all_last_states[0]) if n_runs > 0 else 0
        size = len(all_first_states[0][0]) if n_runs > 0 and n_first_states > 0 else 0

        # Compute normalized entropy for each epoch index
        entropy_first = []
        entropy_last = []
        for i in range(n_first_states):
            # For each epoch index, collect the state at that index across all runs
            states_at_i = [
                tuple(s) for s in [all_first_states[run][i] for run in range(n_runs)]
            ]
            entropy_per_cell, *_ = MetricsCalculator.calculate_normalized_entropy(
                states_at_i, size
            )
            entropy_first.append(entropy_per_cell)
        for i in range(n_last_states):
            states_at_i = [
                tuple(s) for s in [all_last_states[run][i] for run in range(n_runs)]
            ]
            entropy_per_cell, *_ = MetricsCalculator.calculate_normalized_entropy(
                states_at_i, size
            )
            entropy_last.append(entropy_per_cell)

        # Plot: x-axis is epoch index (first epochs on left, last epochs on right), y-axis is entropy
        plt.figure(figsize=(12, 6))
        x_first = list(range(1, n_first_states + 1))
        x_last = list(range(n_first_states + 1, n_first_states + n_last_states + 1))
        plt.plot(x_first, entropy_first, label="First Epochs", marker="o")
        plt.plot(x_last, entropy_last, label="Last Epochs", marker="o")
        plt.xlabel("Epoch Index")
        plt.ylabel("Normalized Entropy per Cell")
        plt.title(f"Normalized Entropy per Cell (Rule {self.rule_number})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        base_csv = os.path.basename(input_csv).replace(".csv", "")
        img_path = (
            os.path.join(output_dir, f"{base_csv}_entropy.png")
            if output_dir
            else f"{base_csv}_entropy.png"
        )
        plt.savefig(img_path)
        plt.close()

        # --- Generate markdown string showcasing the PNG image ---
        md = f"""# Normalized Entropy for Rule {self.rule_number}\n\n \
This section presents the normalized entropy per cell for the first and last epochs across all runs.\n\n \
![Normalized Entropy]({os.path.relpath(img_path, output_dir) if output_dir else img_path})\n"""
        return md

    def run_analysis_to_md(
        self,
        input_csv,
        output_dir=None,
    ):
        """
        Runs all analyses (correlations, complexity, entropy) and generates a markdown file with the results.
        The markdown includes experiment metadata and automaton properties at the top, and links to the generated images.

        Args:
            input_csv (str): Path to the CSV file containing the initial and final states.
            output_dir (str, optional): Directory to save the markdown file and images. Defaults to None.
        """
        import os
        import re
        import random
        import matplotlib.pyplot as plt
        import numpy as np
        import csv

        # --- Extract experiment details from filename ---
        filename = os.path.basename(input_csv)
        # Example: 18_random_s13_g300_r5000_f13_l13.csv
        match = re.match(
            r"(\d+)_([a-zA-Z]+)_s(\d+)_g(\d+)_r(\d+)_f(\d+)_l(\d+)\.csv$", filename
        )
        if match:
            rule = int(match.group(1))
            ic = match.group(2)
            size = int(match.group(3))
            generations = int(match.group(4))
            runs = int(match.group(5))
            n_first = int(match.group(6))
            n_last = int(match.group(7))
        else:
            rule = self.rule_number
            ic = None
            size = self.size
            generations = None
            runs = None
            n_first = None
            n_last = None

        # --- Build experiment metadata markdown ---
        metadata_md = f"""# Analysis for Wolfram Rule {self.rule_number}\n\n\
## Experiment Details\n\
- **Rule:** {rule}\n\
- **Initial Condition:** {ic if ic is not None else ''}\n\
- **Size:** {size}\n\
- **Generations:** {generations if generations is not None else ''}\n\
- **Runs:** {runs if runs is not None else ''}\n\
- **First States per Run:** {n_first if n_first is not None else ''}\n\
- **Last States per Run:** {n_last if n_last is not None else ''}\n\n\n"""

        # --- Get automaton properties markdown ---
        properties_md = self.get_properties_to_md()

        # --- Sample images of first/last states for 5 random runs ---
        sample_imgs_md = ""
        sample_img_paths = []
        if runs is not None and n_first is not None and n_last is not None:
            # Read all states from CSV
            all_first_states = []
            all_last_states = []
            with open(input_csv, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    first_states = row[:n_first]
                    last_states = row[n_first : n_first + n_last]
                    all_first_states.append(first_states)
                    all_last_states.append(last_states)
            n_samples = min(5, len(all_first_states))
            sample_indices = random.sample(range(len(all_first_states)), n_samples)
            base_csv = os.path.basename(input_csv).replace(".csv", "")
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            for idx, run_idx in enumerate(sample_indices):
                first = all_first_states[run_idx]
                last = all_last_states[run_idx]
                # Convert to numpy arrays for plotting
                first_arr = np.array([[int(bit) for bit in state] for state in first])
                last_arr = np.array([[int(bit) for bit in state] for state in last])
                # Plot: first states on top, last states on bottom
                fig, ax = plt.subplots(figsize=(8, 4))
                combined = np.vstack([first_arr, last_arr])
                ax.imshow(combined, cmap="binary", aspect="auto")
                ax.axhline(
                    len(first_arr) - 0.5, color="red", linestyle="--", linewidth=1
                )
                ax.set_title(f"Sample Run {run_idx+1} (Rule {self.rule_number})")
                ax.set_xlabel("Cell Index")
                ax.set_ylabel(f"First {n_first} (top), Last {n_last} (bottom)")
                ax.set_yticks(
                    [0, len(first_arr) - 1, len(first_arr), len(combined) - 1]
                )
                ax.set_yticklabels(
                    ["1", str(n_first), str(n_first + 1), str(n_first + n_last)]
                )
                plt.tight_layout()
                img_path = (
                    os.path.join(output_dir, f"{base_csv}_sample_run_{run_idx+1}.png")
                    if output_dir
                    else f"{base_csv}_sample_run_{run_idx+1}.png"
                )
                plt.savefig(img_path)
                plt.close()
                sample_img_paths.append(img_path)
            # Markdown for sample images
            sample_imgs_md = (
                "# Sample Automaton Runs\n\n"
                + "Below are five sample runs, showing the first states (top) and last states (bottom) for each run. The red line separates the two.\n\n"
            )
            for img_path in sample_img_paths:
                rel_path = (
                    os.path.relpath(img_path, output_dir) if output_dir else img_path
                )
                sample_imgs_md += f"![Sample Run]({rel_path})\n\n"

        # --- Run analyses ---
        correlations_md = self.run_correlations_to_md(input_csv, output_dir=output_dir)
        complexity_md = self.run_complexity_to_md(input_csv, output_dir=output_dir)
        entropy_md = self.run_entropy_to_md(input_csv, output_dir=output_dir)

        # --- Save markdown file with same name as CSV plus _analysis.md ---
        base_csv = os.path.basename(input_csv).replace(".csv", "")
        md_filename = f"{base_csv}_analysis.md"
        full_md = (
            metadata_md
            + properties_md
            + "\n\n"
            + sample_imgs_md
            + "\n\n"
            + correlations_md
            + "\n\n"
            + complexity_md
            + "\n\n"
            + entropy_md
        )
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, md_filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(full_md)
            print(f"Analysis markdown saved to {path}")
        else:
            print(full_md)
            return full_md
