"""Benchmark ALNS on all 2E-CVRP instances with both constructive heuristics.
"""

import contextlib
import copy
import csv
import os
import math
import random
import re
import time
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")  # Batch plots: save without opening hundreds of windows.
import matplotlib.pyplot as plt

from alns import ALNS
from problem import TWO_E_CVRP
from visualize import SolutionVisualizer


# ============================================================
# Configuration: change these values to control the benchmark
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
INSTANCES_DIR = BASE_DIR / "instances"
USE_SA = False             # False -> alns_results; True -> alns_sa_results
SA_INITIAL_TEMPERATURE = None  # None: automatically calibrated
SA_COOLING_RATE = 0.99
SA_END_TEMPERATURE = 1e-8
OUTPUT_DIR = BASE_DIR / ("alns_sa_results" if USE_SA else "alns_results")
INSTANCE_PATTERN = re.compile(r"^C[a-d]\d+-\d+,\d+,\d+\.txt$", re.IGNORECASE)
CONSTRUCTIVE_METHODS = ("savings", "random")
NUM_ITERATIONS = 200
BASE_SEED = 42
N_RUNS = 1                  # Use 30 for a larger statistical experiment.
SAVE_PLOTS = True          # One set of three plots for every run.
SAVE_ALNS_LOGS = False     # Set True to keep the per-iteration terminal logs.
PLOT_DPI = 200

METHOD_FOLDERS = {
    "savings": "cw_constructive_heur",
    "random": "random_constructive_heur",
}

ITERATION_FIELDS = [
    "instance", "constructive_method", "seed", "run", "iteration",
    "candidate_cost", "candidate_distance", "current_cost", "current_distance",
    "best_cost", "best_distance", "selected_destroy", "selected_repair",
    "score", "accepted", "new_global_best",
    "temperature", "acceptance_probability", "acceptance_reason",
    "destroy1_weight", "destroy2_weight", "destroy3_weight",
    "repair1_weight", "repair2_weight", "repair3_weight",
]
RUN_FIELDS = [
    "instance", "constructive_method", "seed", "run", "iterations",
    "initial_cost", "best_cost", "initial_distance", "best_distance",
    "absolute_improvement", "improvement_pct", "runtime_seconds",
    "sa_enabled", "sa_initial_temperature", "sa_cooling_rate", "sa_end_temperature",
    "accepted_count", "sa_accepted_count", "rejected_count",
    "initial_first_routes", "initial_second_routes", "best_first_routes",
    "best_second_routes", "improvement_count", "last_improvement_iteration",
]
SUMMARY_METRICS = [
    "initial_cost", "best_cost", "initial_distance", "best_distance",
    "absolute_improvement", "improvement_pct", "runtime_seconds",
    "improvement_count", "last_improvement_iteration",
    "accepted_count", "sa_accepted_count", "rejected_count",
]
SUMMARY_FIELDS = ["instance", "constructive_method", "n_runs", "iterations"] + [
    f"{metric}_{stat}" for metric in SUMMARY_METRICS
    for stat in ("mean", "min", "max", "std")
]


@contextlib.contextmanager
def working_directory(directory):
    """Accommodate problem.py, whose parser expects an instance filename."""
    previous = Path.cwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous)


def load_problem(instance_path):
    instance_path = Path(instance_path).resolve()
    if not instance_path.is_file():
        raise FileNotFoundError(instance_path)
    with working_directory(instance_path.parent):
        return TWO_E_CVRP(instance_path.name)


def total_distance(solution):
    """Distance traveled across both echelons, excluding fixed costs."""
    return sum(route.distance for route in solution.routes_first + solution.routes_second)


class TrackedALNS(ALNS):
    """Collect iteration data while leaving the original ALNS search unchanged."""

    def __init__(self, problem):
        super().__init__(problem)
        self.initial_solution = None
        self.history = []

    def construct_init_solution(self):
        super().construct_init_solution()
        self.initial_solution = copy.deepcopy(self.current_solution)
        self.history = []
        self._record(iteration=0, candidate=None)

    def update_weights(self):
        # Parent execute() calls this just after check_if_accept().
        previous_best = self.history[-1]["best_cost"]
        previous_current = self.history[-1]["current_cost"]
        # Temperature at which the candidate was evaluated (before cooling).
        evaluated_temperature = (self.temperature / self.sa_cooling_rate
                                 if self.use_sa else None)
        super().update_weights()
        self._record(iteration=len(self.history), candidate=self.temp_solution,
                     previous_best=previous_best, previous_current=previous_current,
                     evaluated_temperature=evaluated_temperature)

    def _record(self, iteration, candidate, previous_best=None,
                previous_current=None, evaluated_temperature=None):
        current = self.current_solution
        best = self.best_solution
        is_candidate = candidate is not None
        candidate_cost = candidate.cost if is_candidate else current.cost
        accepted = is_candidate and self.accepted
        self.history.append({
            "iteration": iteration,
            "candidate_cost": candidate_cost,
            "candidate_distance": total_distance(candidate) if is_candidate else total_distance(current),
            "current_cost": current.cost,
            "current_distance": total_distance(current),
            "best_cost": best.cost,
            "best_distance": total_distance(best),
            "destroy_weights": dict(self.destroy_weights),
            "repair_weights": dict(self.repair_weights),
            "selected_destroy": self.selected_destroy if is_candidate else "",
            "selected_repair": self.selected_repair if is_candidate else "",
            "score": self.score if is_candidate else "",
            "accepted": int(accepted),
            "temperature": evaluated_temperature if is_candidate else self.temperature,
            "acceptance_probability": self.acceptance_probability if is_candidate else "",
            "acceptance_reason": self.acceptance_reason if is_candidate else "initial",
            "new_global_best": int(is_candidate and best.cost < previous_best),
        })


def _draw_solution(ax, solution, title, echelon="both"):
    viz = SolutionVisualizer(solution)
    if echelon in ("both", "first"):
        viz._plot_routes(ax, solution.routes_first, True, True, False)
    if echelon in ("both", "second"):
        viz._plot_routes(ax, solution.routes_second, False, True, False)
    viz._plot_locations(ax, show_labels=False)
    ax.set_title(f"{title}\nCost: {solution.cost:.2f} | Distance: {total_distance(solution):.2f}", fontsize=12)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.25)


def plot_route_comparison(algo, output_dir, echelon="both"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    _draw_solution(axes[0], algo.initial_solution, "Before ALNS (initial)", echelon)
    _draw_solution(axes[1], algo.best_solution, "After ALNS (best found)", echelon)
    SolutionVisualizer(algo.best_solution)._add_legend(axes[1], echelon)
    fig.suptitle("2E-CVRP: Route comparison", fontsize=16)
    fig.savefig(output_dir / "routes_before_after.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_operator_weights(algo, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    iterations = [row["iteration"] for row in algo.history]
    names = {
        "destroy_weights": ("Destroy operator weights", axes[0],
                            {"destroy1": "Random", "destroy2": "Worst", "destroy3": "Shaw"}),
        "repair_weights": ("Repair operator weights", axes[1],
                           {"repair1": "Greedy", "repair2": "Regret-2", "repair3": "Random"}),
    }
    for key, (title, ax, labels) in names.items():
        for name in algo.history[0][key]:
            ax.plot(iterations, [row[key][name] for row in algo.history],
                    label=f"{name} — {labels.get(name, name)}", linewidth=1.6)
        ax.set_title(title)
        ax.set_ylabel("Adaptive weight")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("ALNS iteration")
    fig.savefig(output_dir / "operator_weights.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(algo, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    iterations = [row["iteration"] for row in algo.history]
    for ax, metric, title in (
        (axes[0], "cost", "Total solution cost"),
        (axes[1], "distance", "Total traveled distance (both echelons)"),
    ):
        for kind, label, style in (
            ("candidate", "Generated candidate", {"alpha": 0.45, "linewidth": 1.0}),
            ("current", "Accepted current", {"linewidth": 1.7}),
            ("best", "Best so far", {"linewidth": 2.0, "linestyle": "--"}),
        ):
            ax.plot(iterations, [row[f"{kind}_{metric}"] for row in algo.history],
                    label=label, **style)
        ax.set_title(title)
        ax.set_ylabel("Cost" if metric == "cost" else "Distance")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("ALNS iteration")
    fig.savefig(output_dir / "convergence.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def iteration_csv_rows(algo, instance_name, method, seed, run_number):
    for row in algo.history:
        result = {
            "instance": instance_name,
            "constructive_method": method,
            "seed": seed,
            "run": run_number,
        }
        for field in ITERATION_FIELDS:
            if field in row and field not in ("destroy_weights", "repair_weights"):
                result[field] = row[field]
        for operator, weight in row["destroy_weights"].items():
            result[f"{operator}_weight"] = weight
        for operator, weight in row["repair_weights"].items():
            result[f"{operator}_weight"] = weight
        yield result


def run_alns(instance_path, method, seed, run_number):
    """Execute one independent ALNS run and return its history and summary."""
    problem = load_problem(instance_path)
    algo = TrackedALNS(problem)
    algo.constructive_method = method
    algo.num_iter = NUM_ITERATIONS
    algo.random_gen = random.Random(seed)
    algo.use_sa = USE_SA
    algo.sa_initial_temperature = SA_INITIAL_TEMPERATURE
    algo.sa_cooling_rate = SA_COOLING_RATE
    algo.sa_end_temperature = SA_END_TEMPERATURE

    output_dir = OUTPUT_DIR / METHOD_FOLDERS[method] / instance_path.stem / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if SAVE_ALNS_LOGS:
        with (output_dir / "alns_log.txt").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                algo.execute()
    else:
        with (output_dir / "alns_log.txt").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                algo.execute()
        (output_dir / "alns_log.txt").unlink()
    runtime = time.perf_counter() - start
    # Initial history row is captured during construction, before SA calibration.
    if USE_SA:
        algo.history[0]["temperature"] = (
            algo.sa_initial_temperature if algo.sa_initial_temperature is not None
            else max(1e-8, 0.05 * algo.initial_solution.cost / math.log(2))
        )

    if SAVE_PLOTS:
        plot_route_comparison(algo, output_dir)
        plot_operator_weights(algo, output_dir)
        plot_convergence(algo, output_dir)

    initial = algo.initial_solution
    best = algo.best_solution
    improvement = initial.cost - best.cost
    improvement_iterations = [r["iteration"] for r in algo.history if r["new_global_best"]]
    result = {
        "instance": instance_path.stem,
        "constructive_method": method,
        "seed": seed,
        "run": run_number,
        "iterations": NUM_ITERATIONS,
        "initial_cost": initial.cost,
        "best_cost": best.cost,
        "initial_distance": total_distance(initial),
        "best_distance": total_distance(best),
        "absolute_improvement": improvement,
        "improvement_pct": 100 * improvement / initial.cost if initial.cost else 0.0,
        "runtime_seconds": runtime,
        "sa_enabled": USE_SA,
        "sa_initial_temperature": algo.history[0]["temperature"],
        "sa_cooling_rate": algo.sa_cooling_rate if USE_SA else "",
        "sa_end_temperature": algo.sa_end_temperature if USE_SA else "",
        "accepted_count": sum(r["accepted"] for r in algo.history),
        "sa_accepted_count": sum(r["acceptance_reason"] == "sa_accepted" for r in algo.history),
        "rejected_count": sum(r["acceptance_reason"] == "rejected" for r in algo.history),
        "initial_first_routes": len(initial.routes_first),
        "initial_second_routes": len(initial.routes_second),
        "best_first_routes": len(best.routes_first),
        "best_second_routes": len(best.routes_second),
        "improvement_count": len(improvement_iterations),
        "last_improvement_iteration": improvement_iterations[-1] if improvement_iterations else 0,
    }
    return algo, result


def summarize_runs(run_rows):
    """Compute sample standard deviations across seeds for each instance/method."""
    groups = {}
    for row in run_rows:
        groups.setdefault((row["instance"], row["constructive_method"]), []).append(row)
    summaries = []
    for (instance, method), rows in sorted(groups.items()):
        result = {
            "instance": instance,
            "constructive_method": method,
            "n_runs": len(rows),
            "iterations": NUM_ITERATIONS,
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
            result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(result)
    return summaries


def main():
    if N_RUNS < 1:
        raise ValueError("N_RUNS must be at least 1")
    if NUM_ITERATIONS < 1:
        raise ValueError("NUM_ITERATIONS must be at least 1")
    if not INSTANCES_DIR.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {INSTANCES_DIR}")
    instance_files = sorted(
        (p for p in INSTANCES_DIR.glob("*.txt") if INSTANCE_PATTERN.fullmatch(p.name)),
        key=lambda p: p.name,
    )
    if not instance_files:
        raise FileNotFoundError(f"No matching instances in {INSTANCES_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iteration_path = OUTPUT_DIR / "alns_iteration_results.csv"
    runs_path = OUTPUT_DIR / "alns_run_results.csv"
    summary_path = OUTPUT_DIR / "alns_summary_results.csv"
    all_runs = []
    total = len(instance_files) * len(CONSTRUCTIVE_METHODS) * N_RUNS
    count = 0
    print(f"Running {total} ALNS runs: {len(instance_files)} instances × "
          f"{len(CONSTRUCTIVE_METHODS)} heuristics × {N_RUNS} seeds")
    print(f"Iterations per run: {NUM_ITERATIONS}")

    # Write each run immediately: the large iteration CSV does not need to be
    # accumulated in memory and remains available if the batch is interrupted.
    with iteration_path.open("w", newline="", encoding="utf-8") as it_file, \
         runs_path.open("w", newline="", encoding="utf-8") as run_file:
        it_writer = csv.DictWriter(it_file, fieldnames=ITERATION_FIELDS)
        run_writer = csv.DictWriter(run_file, fieldnames=RUN_FIELDS)
        it_writer.writeheader()
        run_writer.writeheader()

        for instance_path in instance_files:
            for method in CONSTRUCTIVE_METHODS:
                for run_idx in range(N_RUNS):
                    seed = BASE_SEED + run_idx
                    count += 1
                    print(f"[{count}/{total}] {instance_path.stem} | {method} | seed={seed}", flush=True)
                    algo, result = run_alns(instance_path, method, seed, run_idx + 1)
                    it_writer.writerows(iteration_csv_rows(
                        algo, instance_path.stem, method, seed, run_idx + 1
                    ))
                    run_writer.writerow(result)
                    it_file.flush()
                    run_file.flush()
                    all_runs.append(result)
                    print(f"  initial={result['initial_cost']:.2f} | "
                          f"best={result['best_cost']:.2f} | "
                          f"improvement={result['improvement_pct']:.2f}% | "
                          f"runtime={result['runtime_seconds']:.2f}s", flush=True)

    summaries = summarize_runs(all_runs)
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    print("\nBenchmark complete. CSV files:")
    print(f"  {iteration_path}")
    print(f"  {runs_path}")
    print(f"  {summary_path}")
    print("\nMean final costs by instance and heuristic:")
    for row in summaries:
        print(f"  {row['instance']:<22} {row['constructive_method']:<8} "
              f"{row['best_cost_mean']:.2f} ± {row['best_cost_std']:.2f}")


if __name__ == "__main__":
    main()
