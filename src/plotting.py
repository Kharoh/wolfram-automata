import matplotlib.pyplot as plt
from src.metrics import MetricsCalculator


def plot_history(history, title):
    fig_size_y = max(5, 10 * (history.shape[0] / history.shape[1]))
    plt.figure(figsize=(10, fig_size_y))
    plt.imshow(history.cpu().numpy(), cmap="binary", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Cell")
    plt.ylabel("Generation")
    plt.show()


def plot_correlation(final_state, max_distance, rule, category):
    rs = list(range(1, max_distance + 1))
    correlations = [MetricsCalculator.calculate_correlation(final_state, r) for r in rs]
    plt.figure(figsize=(8, 4))
    plt.plot(rs, correlations, marker="o")
    plt.title(f"2-Cell Correlation for Rule {rule} ({category})")
    plt.xlabel("Distance r")
    plt.ylabel("C(r)")
    plt.grid(True)
    plt.show()


def plot_triangle_density(t0, t1, rule, category):
    ns = sorted(set(t0.keys()) | set(t1.keys()))
    t0_vals = [t0.get(n, 0.0) for n in ns]
    t1_vals = [t1.get(n, 0.0) for n in ns]
    plt.figure(figsize=(8, 4))
    plt.plot(ns, t0_vals, marker="o", label="Triangles of 0s")
    plt.plot(ns, t1_vals, marker="s", label="Triangles of 1s")
    plt.title(f"Triangle Density for Rule {rule} ({category})")
    plt.xlabel("Triangle size n")
    plt.ylabel("Density T(n)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_metrics(history, final_state, t0, t1, q0, q1, rule, category, plot_title):
    import matplotlib.gridspec as gridspec

    plt.style.use("seaborn-v0_8-darkgrid")
    max_corr_distance = min(30, final_state.shape[0] // 2)
    rs = list(range(1, max_corr_distance + 1))
    correlations = [MetricsCalculator.calculate_correlation(final_state, r) for r in rs]
    ns_tri = sorted(set(t0.keys()) | set(t1.keys()))
    t0_vals = [t0.get(n, 0.0) for n in ns_tri]
    t1_vals = [t1.get(n, 0.0) for n in ns_tri]
    ns_seq = sorted(set(q0.keys()) | set(q1.keys()))
    q0_vals = [q0.get(n, 0.0) for n in ns_seq]
    q1_vals = [q1.get(n, 0.0) for n in ns_seq]
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1.3, 1], height_ratios=[2, 1], wspace=0.25, hspace=0.3
    )
    # Main automata plot (top left, spans both columns)
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(
        history.cpu().numpy(), cmap="Greys", interpolation="nearest", aspect="auto"
    )
    ax0.set_title(plot_title, fontsize=16, fontweight="bold")
    ax0.set_xlabel("Cell", fontsize=12)
    ax0.set_ylabel("Generation", fontsize=12)
    ax0.set_aspect(history.shape[0] / history.shape[1])
    ax0.tick_params(axis="both", which="major", labelsize=10)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    # Sequence density (directly below automata)
    ax_seq = plt.subplot(gs[1, 0])
    ax_seq.plot(
        ns_seq,
        q0_vals,
        marker="o",
        color="#56B4E9",
        linewidth=2,
        markersize=6,
        label="Runs of 0s",
    )
    ax_seq.plot(
        ns_seq,
        q1_vals,
        marker="s",
        color="#F0E442",
        linewidth=2,
        markersize=6,
        label="Runs of 1s",
    )
    ax_seq.set_title("Sequence Density Q(n)", fontsize=14, fontweight="bold")
    ax_seq.set_xlabel("Run length n", fontsize=12)
    ax_seq.set_ylabel("Density Q(n)", fontsize=12)
    ax_seq.legend(fontsize=11, frameon=False)
    ax_seq.grid(True, linestyle="--", alpha=0.7)
    ax_seq.tick_params(axis="both", which="major", labelsize=10)
    ax_seq.spines["top"].set_visible(False)
    ax_seq.spines["right"].set_visible(False)
    # 2-cell correlation (top right)
    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(rs, correlations, marker="o", color="#0072B2", linewidth=2, markersize=6)
    ax1.set_title("2-Cell Correlation", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Distance r", fontsize=12)
    ax1.set_ylabel("C(r)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.tick_params(axis="both", which="major", labelsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # Triangle density (bottom right)
    ax2 = plt.subplot(gs[1, 1])
    ax2.plot(
        ns_tri,
        t0_vals,
        marker="o",
        color="#D55E00",
        linewidth=2,
        markersize=6,
        label="Triangles of 0s",
    )
    ax2.plot(
        ns_tri,
        t1_vals,
        marker="s",
        color="#009E73",
        linewidth=2,
        markersize=6,
        label="Triangles of 1s",
    )
    ax2.set_title("Triangle Density", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Triangle size n", fontsize=12)
    ax2.set_ylabel("Density T(n)", fontsize=12)
    ax2.legend(fontsize=11, frameon=False)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.tick_params(axis="both", which="major", labelsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.style.use("default")


def format_density_dict(d):
    if not d:
        return "None"
    return ", ".join([f"n={k}: {v:.4f}" for k, v in sorted(d.items())])
