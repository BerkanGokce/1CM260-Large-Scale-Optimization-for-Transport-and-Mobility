
from pathlib import Path
import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from UFL import UFL_Problem, LagrangianHeuristic


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "Instances"
RESULTS_DIR = BASE_DIR / "Results"

INSTANCE_NAMES = ["MO1", "MO2", "MO3", "MO4", "MO5"]

MAX_ITERATIONS = 5000
STAGNATION_LIMIT = 20
INITIAL_BETA = 1.5


def load_instance(name):
    """Load and validate a UFL benchmark instance."""

    instance_path = INSTANCE_DIR / name

    if not instance_path.is_file():
        raise FileNotFoundError(
            f"Instance file not found: {instance_path}"
        )

    previous_dir = Path.cwd()

    try:
        os.chdir(BASE_DIR)
        instance = UFL_Problem.readInstance(name)
    finally:
        os.chdir(previous_dir)

    if instance.c.shape != (
        instance.n_markets,
        instance.n_facilities
    ):
        raise ValueError(
            "Invalid transportation cost matrix"
        )

    if instance.f.shape != (instance.n_facilities,):
        raise ValueError(
            "Invalid facility opening cost vector"
        )

    return instance


def plot_convergence(name, history):
    """Plot best bounds and annotate the final solution quality."""

    iterations = np.array([
        row["iteration"] for row in history
    ])

    bestLB = np.array([
        row["best_LB"] for row in history
    ])

    bestUB = np.array([
        row["best_UB"] for row in history
    ])

    finalLB = bestLB[-1]
    finalUB = bestUB[-1]

    final_gap = (
        (finalUB - finalLB)
        / max(1.0, abs(finalUB))
        * 100
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.step(
        iterations,
        bestLB,
        where="post",
        color="tab:blue",
        linewidth=2,
        label="Best lower bound"
    )

    ax.step(
        iterations,
        bestUB,
        where="post",
        color="tab:orange",
        linewidth=2,
        label="Best upper bound"
    )

    ax.scatter(
        iterations[-1],
        finalLB,
        color="tab:blue",
        s=45,
        zorder=5
    )

    ax.scatter(
        iterations[-1],
        finalUB,
        color="tab:orange",
        s=45,
        zorder=5
    )

    result_text = (
        f"Best lower bound: {finalLB:,.2f}\n"
        f"Best upper bound: {finalUB:,.2f}\n"
        f"Best solution cost: {finalUB:,.2f}\n"
        f"Relative gap: {final_gap:.2f}%"
    )

    ax.text(
        0.98,
        0.70,
        result_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.95
        }
    )

    ax.set_title(
        f"{name}: Lagrangian Heuristic Convergence",
        fontsize=15
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective value", fontsize=12)

    ax.set_xlim(1, max(2, len(history)))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()

    figure_path = RESULTS_DIR / f"{name}_convergence.png"

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_history(name, history):
    """Save iteration results and Lagrangian multipliers."""

    # Save iteration statistics
    history_path = RESULTS_DIR / f"{name}_history.csv"

    scalar_fields = [
        key for key in history[0].keys()
        if key != "multipliers"
    ]

    with history_path.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=scalar_fields
        )

        writer.writeheader()

        for row in history:
            writer.writerow({
                key: row[key] for key in scalar_fields
            })

    # Save multiplier values separately
    multiplier_path = RESULTS_DIR / f"{name}_multipliers.csv"

    n_customers = len(history[0]["multipliers"])

    with multiplier_path.open("w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            ["iteration"]
            + [f"lambda_{i+1}" for i in range(n_customers)]
        )

        for row in history:
            writer.writerow(
                [row["iteration"]]
                + row["multipliers"].tolist()
            )


def plot_multiplier_evolution(name, history, n_customers=8):
    """Plot selected Lagrangian multipliers over iterations."""

    iterations = np.array([
        row["iteration"] for row in history
    ])

    multipliers = np.array([
        row["multipliers"] for row in history
    ])

    # Select customers with the largest multiplier variation
    variation = np.std(multipliers, axis=0)

    selected = np.argsort(variation)[-n_customers:]

    fig, ax = plt.subplots(figsize=(11, 6))

    for i in selected:
        ax.plot(
            iterations,
            multipliers[:, i],
            linewidth=1.5,
            label=f"Customer {i+1}"
        )

    ax.set_title(
        f"{name}: Lagrangian Multiplier Evolution",
        fontsize=15
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Multiplier value", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / f"{name}_multipliers.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_subgradient_diagnostics(name, history):
    """Plot subgradient norm and Polyak step size."""

    iterations = np.array([
        row["iteration"] for row in history
    ])

    norms = np.array([
        row["subgradient_norm"] for row in history
    ])

    step_sizes = np.array([
        row["step_size"] for row in history
    ])

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        iterations,
        norms,
        color="tab:blue",
        linewidth=1.5,
        label="Subgradient norm"
    )

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel(
        "Subgradient norm",
        color="tab:blue",
        fontsize=12
    )

    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()

    ax2.plot(
        iterations,
        step_sizes,
        color="tab:orange",
        linewidth=1.5,
        alpha=0.8,
        label="Polyak step size"
    )

    ax2.set_ylabel(
        "Step size",
        color="tab:orange",
        fontsize=12
    )

    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(
        f"{name}: Subgradient and Step-Size Diagnostics",
        fontsize=15
    )

    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / f"{name}_subgradient.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    """Run all benchmark instances and report their results."""

    RESULTS_DIR.mkdir(exist_ok=True)

    summary = []
    total_start = time.perf_counter()

    print("\nLagrangian Heuristic Benchmark Results")
    print("=" * 112)

    header = (
        f"{'Instance':<10}"
        f"{'Best LB':>14}"
        f"{'Best UB':>14}"
        f"{'Gap (%)':>12}"
        f"{'Iterations':>12}"
        f"{'Open Fac.':>12}"
        f"{'Time (s)':>12}"
        f"  {'Termination'}"
    )

    print(header)
    print("-" * 112)

    for name in INSTANCE_NAMES:

        instance = load_instance(name)

        heuristic = LagrangianHeuristic(instance)

        start_time = time.perf_counter()

        (   bestLB,
            bestUB,
            best_solution,
            history,
            stop_reason
        ) = heuristic.runHeuristic(
            max_iterations=MAX_ITERATIONS,
            stagnation_limit=STAGNATION_LIMIT,
            initial_beta=INITIAL_BETA
        )

        execution_time = time.perf_counter() - start_time

        if best_solution is None:
            raise RuntimeError(
                "No feasible solution was found"
            )

        if not best_solution.isFeasible():
            raise RuntimeError(
                "The final solution is infeasible"
            )

        gap_percent = ((bestUB - bestLB) / max(1.0, abs(bestUB)) * 100)

        plot_convergence(name, history)
        plot_multiplier_evolution(name, history)
        plot_subgradient_diagnostics(name, history)
        save_history(name, history)

        open_facilities = int(np.sum(best_solution.y))

        result = {
            "instance": name,
            "best_LB": bestLB,
            "best_UB": bestUB,
            "gap_percent": gap_percent,
            "iterations": len(history),
            "open_facilities": open_facilities,
            "time_seconds": execution_time,
            "stop_reason": stop_reason
        }

        summary.append(result)

        print(
            f"{name:<10}"
            f"{bestLB:>14,.4f}"
            f"{bestUB:>14,.4f}"
            f"{gap_percent:>12.4f}"
            f"{len(history):>12}"
            f"{open_facilities:>12}"
            f"{execution_time:>12.4f}"
            f"  {stop_reason}"
        )

    summary_path = RESULTS_DIR / "summary.csv"

    with summary_path.open("w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))

        writer.writeheader()
        writer.writerows(summary)

    total_time = time.perf_counter() - total_start

    print("=" * 112)
    print(f"Total execution time: {total_time:.4f} seconds")
    print(f"Maximum iterations: {MAX_ITERATIONS}")
    print(f"Initial beta: {INITIAL_BETA}")
    print(f"Stagnation limit: {STAGNATION_LIMIT}")
    print(f"Results directory: {RESULTS_DIR}")
    print("\nAll benchmark experiments completed.")


if __name__ == "__main__":
    main()
