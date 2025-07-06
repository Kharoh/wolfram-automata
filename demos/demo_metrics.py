import torch
import random

from src.wolfram_ca import WolframCA
from src.metrics import MetricsCalculator
from src.plotting import plot_all_metrics, format_density_dict
from config.rules import RULE_CATEGORIES


def demo_metrics():
    """
    Run metrics demonstrations across different Wolfram rule categories.
    This function initializes cellular automata for each rule in the specified categories,
    runs them for a specified number of generations, calculates various metrics,
    and plots the results.
    """
    # --- Parameters ---
    SIZE = 201
    GENERATIONS = 100
    INITIAL_CONDITION = "random"  # Use 'random' or 'single'

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Run and Plot Demonstrations ---
    for category, rules in RULE_CATEGORIES.items():
        print(
            f"\n--- Demonstrating {category} Rules (Initial: {INITIAL_CONDITION}) ---"
        )
        for rule in rules:
            ca = WolframCA(
                rule_number=rule,
                size=SIZE,
                initial_condition=INITIAL_CONDITION,
                device=DEVICE,
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

            plot_title = (
                f"Wolfram Rule {rule} ({category}, Initial: {INITIAL_CONDITION})"
            )
            plot_all_metrics(
                history, final_state, t0, t1, q0, q1, rule, category, plot_title
            )
