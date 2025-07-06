import torch
import random
from config.rules import RULE_CATEGORIES
from src.entropy_experiment import run_entropy_experiment


def demo_entropy():
    """
    Run entropy experiments across different Wolfram rule categories.
    This function samples a maximum of MAX_RULES_PER_CATEGORY rules from each category,
    runs entropy experiments, and prints the results in a summary table.
    """

    # --- Entropy Comparison Across Rule Categories ---
    SIZE = 10
    GENERATIONS = 20
    N_RUNS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_RULES_PER_CATEGORY = 5  # Change as desired
    entropy_results = {}
    print("\n--- Entropy Comparison Across Rule Categories ---")
    for category, rules in RULE_CATEGORIES.items():
        print(f"\nCategory: {category}")
        entropy_results[category] = []
        # Randomly sample up to MAX_RULES_PER_CATEGORY rules
        sampled_rules = random.sample(rules, min(MAX_RULES_PER_CATEGORY, len(rules)))
        for rule in sampled_rules:
            print(f"\nRule {rule}:")
            entropy = run_entropy_experiment(
                rule_number=rule,
                size=SIZE,
                generations=GENERATIONS,
                n_runs=N_RUNS,
                device=DEVICE,
                plot_top_k=5,
            )
            entropy_results[category].append((rule, entropy))
    # Print summary table
    print("\n--- Entropy Summary Table ---")
    for category, results in entropy_results.items():
        print(f"\n{category}")
        for rule, entropy in results:
            print(f"  Rule {rule}: Entropy = {entropy:.4f}")
