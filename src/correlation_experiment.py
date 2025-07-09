import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.wolfram_ca import WolframCA
from src.metrics import MetricsCalculator


def run_correlation_experiment(
    rule_number,
    size=10,
    generations=20,
    n_runs=1000,
    device="cpu",
    max_distance=None,
    plot=True,
):
    """
    Runs multiple random initializations for a given rule and computes the average two-point correlation function over all runs.
    Args:
        rule_number (int): Wolfram rule number (0-255)
        size (int): Number of cells in the automaton
        generations (int): Number of generations to run
        n_runs (int): Number of random initializations
        device (str): 'cpu' or 'cuda'
        max_distance (int): Maximum distance for correlation calculation (default: size//2)
        plot (bool): Whether to plot the average correlation function
    Returns:
        avg_correlations (dict): Mapping from distance to average correlation value
    """
    if max_distance is None:
        max_distance = min(20, size // 2)  # Calculate up to 20 or half the size
    else:
        max_distance = min(max_distance, 20, size // 2)
    all_final_states = []
    for _ in range(n_runs):
        ca = WolframCA(
            rule_number, size=size, initial_condition="random", device=device
        )
        history = ca.run(generations=generations)
        final_state = history[-1]
        all_final_states.append(final_state)
    avg_correlations = MetricsCalculator.calculate_average_correlation(
        all_final_states, max_distance
    )

    # Plotting: show automaton and correlation side by side
    if plot:
        import matplotlib.gridspec as gridspec
        import numpy as np

        plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
        # Plot a sample automaton (last run)
        ax0 = plt.subplot(gs[0])
        ax0.imshow(
            history.cpu().numpy(), cmap="binary", interpolation="nearest", aspect="auto"
        )
        ax0.set_title(f"Sample Automaton (Rule {rule_number})")
        ax0.set_xlabel("Cell")
        ax0.set_ylabel("Generation")
        # Plot correlation
        ax1 = plt.subplot(gs[1])
        distances = list(avg_correlations.keys())
        values = [avg_correlations[r] for r in distances]
        ax1.plot(distances, values, marker="o")
        ax1.set_xlabel("Distance r")
        ax1.set_ylabel("Average Correlation C(r)")
        ax1.set_title(f"Average Two-Point Correlation\n(Rule {rule_number})")
        ax1.grid(True)
        plt.tight_layout()
        plt.show()

    print(
        f"Rule {rule_number} (size={size}, generations={generations}, runs={n_runs}):"
    )
    print("  Average two-point correlations:")
    for r, val in avg_correlations.items():
        print(f"    r={r}: C(r)={val:.4f}")
    return avg_correlations


if __name__ == "__main__":
    RULE = 30
    SIZE = 10
    GENERATIONS = 20
    N_RUNS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run_correlation_experiment(
        RULE, size=SIZE, generations=GENERATIONS, n_runs=N_RUNS, device=DEVICE
    )
