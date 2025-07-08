import torch
import random
from config.rules import RULE_CATEGORIES
from src.lempel_ziv_experiment import run_lempel_ziv_experiment


def demo_lempel_ziv():
    """
    Run Lempel-Ziv complexity experiments across different Wolfram rule categories (on CA final states).
    This function samples a maximum of MAX_RULES_PER_CATEGORY rules from each category,
    runs Lempel-Ziv experiments, and prints the results in a summary table.
    """
    SIZE = 20
    GENERATIONS = 40
    N_RUNS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_RULES_PER_CATEGORY = 5  # Change as desired
    lz_results = {}
    print(
        "\n--- Lempel-Ziv Complexity Comparison Across Rule Categories (CA Final States) ---"
    )
    for category, rules in RULE_CATEGORIES.items():
        print(f"\nCategory: {category}")
        lz_results[category] = []
        sampled_rules = random.sample(rules, min(MAX_RULES_PER_CATEGORY, len(rules)))
        for rule in sampled_rules:
            print(f"\nRule {rule}:")
            complexities = run_lempel_ziv_experiment(
                rule_number=rule,
                size=SIZE,
                generations=GENERATIONS,
                n_runs=N_RUNS,
                device=DEVICE,
                # plot=True,
                plot=False,  # Set to False to avoid plotting in the demo
            )
            mean_lz = sum(complexities) / len(complexities)
            lz_results[category].append((rule, mean_lz))
    # Print summary table
    print("\n--- Lempel-Ziv Complexity Summary Table (Mean Normalized) ---")
    for category, results in lz_results.items():
        print(f"\n{category}")
        for rule, mean_lz in results:
            print(f"  Rule {rule}: Mean Normalized LZ = {mean_lz:.4f}")
