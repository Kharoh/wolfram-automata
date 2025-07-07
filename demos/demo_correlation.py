import torch
import random
from config.rules import RULE_CATEGORIES
from src.correlation_experiment import run_correlation_experiment


def demo_correlation():
    """
    Run correlation experiments across different Wolfram rule categories.
    This function samples a maximum of MAX_RULES_PER_CATEGORY rules from each category,
    runs correlation experiments, and prints the results in a summary table.
    """

    SIZE = 20
    GENERATIONS = 40
    N_RUNS = 2000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_RULES_PER_CATEGORY = 5  # Change as desired
    correlation_results = {}
    print("\n--- Correlation Comparison Across Rule Categories ---")
    for category, rules in RULE_CATEGORIES.items():
        print(f"\nCategory: {category}")
        correlation_results[category] = []
        sampled_rules = random.sample(rules, min(MAX_RULES_PER_CATEGORY, len(rules)))
        for rule in sampled_rules:
            print(f"\nRule {rule}:")
            avg_corr = run_correlation_experiment(
                rule_number=rule,
                size=SIZE,
                generations=GENERATIONS,
                n_runs=N_RUNS,
                device=DEVICE,
                plot=True,
            )
            correlation_results[category].append((rule, avg_corr))
    # Print summary table
    print("\n--- Correlation Summary Table (C(r=1)) ---")
    for category, results in correlation_results.items():
        print(f"\n{category}")
        for rule, avg_corr in results:
            c1 = avg_corr.get(1, float("nan"))
            print(f"  Rule {rule}: C(r=1) = {c1:.4f}")
