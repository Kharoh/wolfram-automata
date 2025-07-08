import torch
import numpy as np
import matplotlib.pyplot as plt
from src.wolfram_ca import WolframCA


def lempel_ziv_complexity(word):
    """
    Computes the Lempel-Ziv complexity of a given word.
    Attention: This implementation corresponds to the original Lempel-Ziv algorithm,
    the candidate is searched in the prefix of the word, which is the extension minus the last character. It is not searched in simply word[:i] but in word[:i + v - 1].
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


def run_lempel_ziv_experiment(
    rule_number, size=10, generations=20, n_runs=1000, device="cpu", plot=True
):
    """
    Runs multiple random initializations for a given rule and computes the normalized Lempel-Ziv complexity of the final state distribution.
    Args:
        rule_number (int): Wolfram rule number (0-255)
        size (int): Number of cells in the automaton
        generations (int): Number of generations to run
        n_runs (int): Number of random initializations
        device (str): 'cpu' or 'cuda'
        plot (bool): Whether to plot histogram
    Returns:
        complexities (list): List of normalized complexities
    """
    complexities = []
    for _ in range(n_runs):
        ca = WolframCA(
            rule_number, size=size, initial_condition="random", device=device
        )
        history = ca.run(generations=generations)
        final_state = history[-1].cpu().numpy()
        word = "".join(str(int(x)) for x in final_state)
        complexity = lempel_ziv_complexity(word)
        normalized = complexity / size
        complexities.append(normalized)
    if plot:
        plt.hist(complexities, bins=20, color="orchid", edgecolor="black")
        plt.xlabel("Normalized Lempel-Ziv Complexity")
        plt.ylabel("Frequency")
        plt.title(
            f"Lempel-Ziv Complexity of CA Final States\n(Rule {rule_number}, size={size}, runs={n_runs})"
        )
        plt.grid(True)
        plt.show()
    print(
        f"\nRule {rule_number} (size={size}, generations={generations}, runs={n_runs}):"
    )
    print(f"  Mean normalized Lempel-Ziv complexity: {np.mean(complexities):.4f}")
    print(f"  Std: {np.std(complexities):.4f}")
    print(f"  Min: {np.min(complexities):.4f}, Max: {np.max(complexities):.4f}")
    return complexities


if __name__ == "__main__":
    RULE = 30
    SIZE = 10
    GENERATIONS = 20
    N_RUNS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run_lempel_ziv_experiment(
        RULE, size=SIZE, generations=GENERATIONS, n_runs=N_RUNS, device=DEVICE
    )
