import csv
import os
import random
import re
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from contextlib import contextmanager
from statistics import mean, stdev
from tabulate import tabulate
import matplotlib.pyplot as plt

from problem import TWO_E_CVRP
from solution import Solution
from constructive import RandomInsertion, ClarkeWrightSavings
from visualize import SolutionVisualizer


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
INSTANCES_DIR = BASE_DIR / "instances"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_INSTANCE = "Ca1-2,3,15.txt"
BASE_SEED = 42
N_RUNS = 30
# Match filenames such as Ca1-2,3,15.txt and Cb2-2,3,100.txt.
INSTANCE_PATTERN = re.compile(
    r"^C[a-d]\d+-\d+,\d+,\d+\.txt$",
    re.IGNORECASE
)
# ============================================================
# Problem loading
# ============================================================


@contextmanager
def working_directory(directory):
    """
    Temporarily change the working directory.
    Restores the original directory even if an exception occurs.
    """
    previous_directory = Path.cwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous_directory)


def load_problem(instance_path):
    """
    Load an instance using its filename only.
    This accommodates the existing problem.py parser, which
    expects the instance name rather than an absolute path.
    """
    instance_path = Path(instance_path).resolve()
    if not instance_path.is_file():
        raise FileNotFoundError(
            f"Instance not found: {instance_path}"
        )
    with working_directory(instance_path.parent):
        problem = TWO_E_CVRP(instance_path.name)
    return problem
# ============================================================
# Constructive heuristic execution
# ============================================================


def run_constructive(instance_path, method, seed=BASE_SEED):
    """
    Construct one solution using the specified heuristic.
    Parameters
    ----------
    instance_path : str or Path
        Path to the instance.
    method : str
        "random" or "cw".
    seed : int
        Random seed.
    Returns
    -------
    solution : Solution
        Constructed solution.
    runtime : float
        Construction time in seconds.
    """
    # Load a fresh problem for each run.
    problem = load_problem(instance_path)
    random_gen = random.Random(seed)
    solution = Solution(problem)
    start_time = time.perf_counter()
    if method == "random":
        RandomInsertion(solution).construct(random_gen)
    elif method == "cw":
        ClarkeWrightSavings(solution).construct(random_gen)
    else:
        raise ValueError(
            f"Unknown constructive heuristic: {method}"
        )
    solution.validate_solution()
    solution.compute_cost()
    runtime = time.perf_counter() - start_time
    return solution, runtime


def print_solution_summary(solution, method, runtime):
    """
    Print basic statistics for a constructed solution.
    """
    total_distance = sum(
        route.distance
        for route in (
            solution.routes_first + solution.routes_second
        )
    )
    print(f"\n{method}")
    print("-" * 50)
    print(f"Total cost:            {solution.cost:.2f}")
    print(f"First-echelon routes:  {len(solution.routes_first)}")
    print(f"Second-echelon routes: {len(solution.routes_second)}")
    print(f"Total distance:        {total_distance:.2f}")
    print(f"Construction time:     {runtime:.6f} s")
# ============================================================
# Individual visualization: Clarke--Wright
# ============================================================


def main_cw(instance=DEFAULT_INSTANCE, seed=BASE_SEED):
    """
    Run and visualize one Clarke--Wright solution.
    """
    instance_path = INSTANCES_DIR / instance
    instance_name = Path(instance).stem
    solution, runtime = run_constructive(
        instance_path=instance_path,
        method="cw",
        seed=seed
    )
    print_solution_summary(
        solution,
        "Clarke--Wright Constructive Heuristic",
        runtime
    )
    visualizer = SolutionVisualizer(solution)
    visualizer.plot(
        echelon="both",
        show_arrows=True,
        show_labels=True,
        show_loads=False,
        title=f"Clarke--Wright Constructive Solution\nInstance: {instance_name}",
        save_path=RESULTS_DIR / f"clarke_wright_{instance_name}.png"
    )
    return solution
# ============================================================
# Individual visualization: Random Insertion
# ============================================================


def main_insertion(instance=DEFAULT_INSTANCE, seed=BASE_SEED):
    """
    Run and visualize one Random Insertion solution.
    """
    instance_path = INSTANCES_DIR / instance
    instance_name = Path(instance).stem
    solution, runtime = run_constructive(
        instance_path=instance_path,
        method="random",
        seed=seed
    )
    print_solution_summary(
        solution,
        "Random Insertion Constructive Heuristic",
        runtime
    )
    visualizer = SolutionVisualizer(solution)
    visualizer.plot(
        echelon="both",
        show_arrows=True,
        show_labels=True,
        show_loads=False,
        title=f"Random Insertion Constructive Solution\nInstance: {instance_name}",
        save_path=RESULTS_DIR / f"random_insertion_{instance_name}.png"
    )
    return solution
# ============================================================
# Benchmark one instance
# ============================================================


def benchmark_instance(
    instance_path,
    n_runs=N_RUNS,
    base_seed=BASE_SEED
):
    """
    Benchmark both heuristics on one instance.
    Both methods use the same set of random seeds.
    Each run starts with a fresh problem and solution.
    """
    results = {}
    for method in ("random", "cw"):
        costs = []
        runtimes = []
        for run in range(n_runs):
            seed = base_seed + run
            solution, runtime = run_constructive(
                instance_path=instance_path,
                method=method,
                seed=seed
            )
            costs.append(solution.cost)
            runtimes.append(runtime)
        results[method] = {
            "avg": mean(costs),
            "min": min(costs),
            "max": max(costs),
            "std": stdev(costs) if n_runs > 1 else 0.0,
            "time": mean(runtimes)
        }
    random_avg = results["random"]["avg"]
    cw_avg = results["cw"]["avg"]
    improvement = (
        (random_avg - cw_avg) / random_avg * 100
        if random_avg != 0 else 0.0
    )
    return {
        "instance": Path(instance_path).stem,
        "random": results["random"],
        "cw": results["cw"],
        "improvement": improvement
    }
# ============================================================
# Save benchmark results
# ============================================================


def save_results(all_results, n_runs, base_seed):
    """
    Save benchmark results to TXT and CSV files.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULTS_DIR / "constructive_results.txt"
    csv_path = RESULTS_DIR / "constructive_results.csv"
    # --------------------------------------------------------
    # Formatted TXT report
    # --------------------------------------------------------
    header = (
        f"{'Instance':<22}"
        f"{'Random avg.':>14}"
        f"{'Random min.':>14}"
        f"{'CW avg.':>14}"
        f"{'CW min.':>14}"
        f"{'Improv. (%)':>14}"
    )
    separator = "-" * len(header)
    lines = [
        "2E-CVRP CONSTRUCTIVE HEURISTIC COMPARISON",
        "=" * len(header),
        f"Instances evaluated : {len(all_results)}",
        f"Runs per heuristic  : {n_runs}",
        f"Random seed range   : {base_seed} - {base_seed + n_runs - 1}",
        "",
        header,
        separator
    ]
    for result in all_results:
        lines.append(
            f"{result['instance']:<22}"
            f"{result['random']['avg']:>14.2f}"
            f"{result['random']['min']:>14.2f}"
            f"{result['cw']['avg']:>14.2f}"
            f"{result['cw']['min']:>14.2f}"
            f"{result['improvement']:>14.2f}"
        )
    # Average across instances.
    avg_random = mean(
        result["random"]["avg"]
        for result in all_results
    )
    avg_random_min = mean(
        result["random"]["min"]
        for result in all_results
    )
    avg_cw = mean(
        result["cw"]["avg"]
        for result in all_results
    )
    avg_cw_min = mean(
        result["cw"]["min"]
        for result in all_results
    )
    avg_improvement = mean(
        result["improvement"]
        for result in all_results
    )
    lines.append(separator)
    lines.append(
        f"{'Average':<22}"
        f"{avg_random:>14.2f}"
        f"{avg_random_min:>14.2f}"
        f"{avg_cw:>14.2f}"
        f"{avg_cw_min:>14.2f}"
        f"{avg_improvement:>14.2f}"
    )
    # --------------------------------------------------------
    # Additional statistics
    # --------------------------------------------------------
    cw_wins = sum(
        result["cw"]["avg"] < result["random"]["avg"]
        for result in all_results
    )
    random_wins = sum(
        result["random"]["avg"] < result["cw"]["avg"]
        for result in all_results
    )
    ties = len(all_results) - cw_wins - random_wins
    avg_random_time = mean(
        result["random"]["time"]
        for result in all_results
    )
    avg_cw_time = mean(
        result["cw"]["time"]
        for result in all_results
    )
    lines.extend([
        "",
        "ADDITIONAL STATISTICS",
        separator,
        f"Clarke--Wright wins: {cw_wins}",
        f"Random Insertion wins: {random_wins}",
        f"Ties: {ties}",
        "",
        f"Average Random Insertion runtime (s): {avg_random_time:.6f}",
        f"Average Clarke--Wright runtime (s): {avg_cw_time:.6f}",
        "",
        "Positive improvement indicates lower cost for Clarke--Wright.",
        "The Average row contains arithmetic means across instances."
    ])
    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )
    # --------------------------------------------------------
    # CSV export
    # --------------------------------------------------------
    fieldnames = [
        "instance",
        "random_avg",
        "random_min",
        "random_max",
        "random_std",
        "random_time",
        "cw_avg",
        "cw_min",
        "cw_max",
        "cw_std",
        "cw_time",
        "improvement_pct"
    ]
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                "instance": result["instance"],
                "random_avg": result["random"]["avg"],
                "random_min": result["random"]["min"],
                "random_max": result["random"]["max"],
                "random_std": result["random"]["std"],
                "random_time": result["random"]["time"],
                "cw_avg": result["cw"]["avg"],
                "cw_min": result["cw"]["min"],
                "cw_max": result["cw"]["max"],
                "cw_std": result["cw"]["std"],
                "cw_time": result["cw"]["time"],
                "improvement_pct": result["improvement"]
            })
    print("\nBenchmark results saved:")
    print(f"  TXT: {txt_path}")
    print(f"  CSV: {csv_path}")
# ============================================================
# Print saved CSV as a formatted terminal table
# ============================================================


def print_results_table(
    csv_path=RESULTS_DIR / "constructive_results.csv"
):
    """
    Read the benchmark CSV and print a formatted comparison
    table in the terminal.
    """
    df = pd.read_csv(csv_path)
    # Select and rename columns for display.
    columns = {
        "instance": "Instance",
        "random_avg": "Random Avg.",
        "random_min": "Random Min.",
        "cw_avg": "CW Avg.",
        "cw_min": "CW Min.",
        "improvement_pct": "Improvement (%)"
    }
    table = df[list(columns.keys())].rename(
        columns=columns
    )
    # Add an average row.
    average_row = {
        "Instance": "Average",
        "Random Avg.": table["Random Avg."].mean(),
        "Random Min.": table["Random Min."].mean(),
        "CW Avg.": table["CW Avg."].mean(),
        "CW Min.": table["CW Min."].mean(),
        "Improvement (%)": table["Improvement (%)"].mean()
    }
    table = pd.concat(
        [table, pd.DataFrame([average_row])],
        ignore_index=True
    )
    # Format numeric columns to two decimal places.
    for column in table.columns:
        if column != "Instance":
            table[column] = table[column].map(
                lambda value: f"{value:.2f}"
            )
    print("\n" + "=" * 100)
    print("CONSTRUCTIVE HEURISTIC COMPARISON")
    print("=" * 100)
    print(
        tabulate(
            table,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            disable_numparse=True
        )
    )
# ============================================================
# Benchmark all instances
# ============================================================


def main_comparison(
    instances_dir=INSTANCES_DIR,
    n_runs=N_RUNS,
    base_seed=BASE_SEED
):
    """
    Benchmark Random Insertion and Clarke--Wright across
    all valid instances in the instances directory.
    """
    instances_dir = Path(instances_dir).resolve()
    if not instances_dir.is_dir():
        raise FileNotFoundError(
            f"Instance directory not found: {instances_dir}"
        )
    if n_runs < 1:
        raise ValueError(
            "n_runs must be at least 1."
        )
    # Discover valid instance files.
    instance_files = sorted(
        [
            path
            for path in instances_dir.glob("*.txt")
            if INSTANCE_PATTERN.fullmatch(path.name)
        ],
        key=lambda path: path.name
    )
    if not instance_files:
        raise FileNotFoundError(
            f"No valid instance files found in {instances_dir}"
        )
    print("\nConstructive Heuristic Benchmark")
    print("================================")
    print(f"Instances: {len(instance_files)}")
    print(f"Runs per heuristic: {n_runs}")
    print(f"Base seed: {base_seed}")
    print(f"Output: {RESULTS_DIR / 'constructive_results.txt'}\n")
    all_results = []
    # Benchmark every instance.
    for instance_path in tqdm(
        instance_files,
        desc="Benchmarking instances",
        unit="instance",
        dynamic_ncols=True
    ):
        result = benchmark_instance(
            instance_path=instance_path,
            n_runs=n_runs,
            base_seed=base_seed
        )
        all_results.append(result)
    # Save TXT and CSV reports.
    save_results(
        all_results=all_results,
        n_runs=n_runs,
        base_seed=base_seed
    )
    # Read the saved CSV and print it as a table.
    print_results_table()
    return all_results
# ============================================================
# Plot constructive heuristic comparison
# ============================================================


def plot_constructive_comparison(
    csv_path=RESULTS_DIR / "constructive_results.csv",
    save_path=RESULTS_DIR / "constructive_comparison.png"
):
    """
    Compare average solution costs of Random Insertion and
    Clarke--Wright Savings using a parity scatter plot.
    Instances are grouped by number of customers.
    The figure is saved as a high-resolution PNG image.
    """
    # Load benchmark results.
    df = pd.read_csv(csv_path)
    # Extract number of customers from instance names.
    # Example: Ca1-2,3,100 -> 100 customers.
    df["customers"] = (
        df["instance"]
        .str.extract(r",(\d+)$")[0]
        .astype(int)
    )
    # Create figure.
    fig, ax = plt.subplots(figsize=(8, 7))
    # Colors for different instance sizes.
    colors = {
        15: "tab:blue",
        50: "tab:orange",
        100: "tab:green"
    }
    # Plot instances grouped by number of customers.
    for n_customers, group in df.groupby("customers"):
        ax.scatter(
            group["random_avg"],
            group["cw_avg"],
            s=90,
            color=colors.get(n_customers, "gray"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{n_customers} customers",
            zorder=3
        )
    # Determine common axis limits.
    min_cost = min(
        df["random_avg"].min(),
        df["cw_avg"].min()
    )
    max_cost = max(
        df["random_avg"].max(),
        df["cw_avg"].max()
    )
    margin = 0.05 * (max_cost - min_cost)
    lower = min_cost - margin
    upper = max_cost + margin
    # Equal-cost reference line (y = x).
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Equal cost",
        zorder=2
    )
    # Axis settings.
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        "Random Insertion average cost",
        fontsize=13
    )
    ax.set_ylabel(
        "Clarke--Wright average cost",
        fontsize=13
    )
    ax.set_title(
        "Constructive Heuristic Comparison",
        fontsize=15,
        pad=12
    )
    ax.tick_params(axis="both", labelsize=11)
    # Grid and legend.
    ax.grid(
        True,
        linestyle=":",
        alpha=0.35
    )
    ax.legend(
        fontsize=11,
        loc="upper left",
        frameon=True
    )
    fig.tight_layout()
    # Save as a high-resolution PNG.
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"\nComparison plot saved: {save_path}")
    # Display figure.
    plt.show()
    plt.close(fig)



# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":

    # Individual constructive solution visualizations:
    main_insertion()
    main_cw()

    # Benchmark all instances
    main_comparison(instances_dir=INSTANCES_DIR, n_runs=30, base_seed=42)

    # Generate scatter plot using existing benchmark results.
    plot_constructive_comparison()
