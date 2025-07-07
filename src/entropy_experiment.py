import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src.wolfram_ca import WolframCA


def run_entropy_experiment(
    rule_number, size=10, generations=20, n_runs=1000, device="cpu", plot_top_k=5
):
    """
    Runs multiple random initializations for a given rule and computes the entropy of the final state distribution.
    Args:
        rule_number (int): Wolfram rule number (0-255)
        size (int): Number of cells in the automaton
        generations (int): Number of generations to run
        n_runs (int): Number of random initializations
        device (str): 'cpu' or 'cuda'
        plot_top_k (int): Number of most probable configurations to plot
    """
    final_states = []
    for _ in range(n_runs):
        ca = WolframCA(
            rule_number, size=size, initial_condition="random", device=device
        )
        history = ca.run(generations=generations)
        final_state = history[-1].cpu().numpy()
        final_states.append(tuple(final_state.tolist()))

    # Count occurrences of each unique final state
    counts = Counter(final_states)
    total = sum(counts.values())
    probabilities = np.array([count / total for count in counts.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = size  # log2(2^size) = size
    min_entropy = 0.0

    # Make entropy extensive by dividing by the number of cells
    entropy_per_cell = entropy / size

    # Sort by probability (descending)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_items[:plot_top_k]
    top_states = [np.array(state) for state, _ in top_k]
    top_probs = [count / total for _, count in top_k]

    # Plot the most probable configurations
    plt.figure(figsize=(2 * plot_top_k, 3))
    for i, (state, prob) in enumerate(zip(top_states, top_probs)):
        plt.subplot(1, plot_top_k, i + 1)
        plt.imshow(state.reshape(1, -1), cmap="binary", aspect="auto")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"P={prob:.2f}")
    plt.suptitle(f"Top {plot_top_k} Final Configurations (Rule {rule_number})")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

    print(
        f"Rule {rule_number} (size={size}, generations={generations}, runs={n_runs}):"
    )
    print(
        f"  Entropy per cell: {entropy_per_cell:.4f} (min: {min_entropy/size}, max: {max_entropy/size})"
    )
    print(f"  Number of unique final configurations: {len(counts)} out of {2**size}")
    return entropy_per_cell


if __name__ == "__main__":
    RULE = 30
    SIZE = 10
    GENERATIONS = 20
    N_RUNS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run_entropy_experiment(
        RULE, size=SIZE, generations=GENERATIONS, n_runs=N_RUNS, device=DEVICE
    )
