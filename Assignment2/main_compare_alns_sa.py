"""
Compare regular ALNS against ALNS + simulated annealing from saved CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["instance", "constructive_method", "seed", "run"]
METHODS = ["random", "savings"]
LABELS = {"random": "Random Insertion", "savings": "CW Savings"}
COLORS = {"Regular ALNS": "#386cb0", "ALNS + SA": "#e07a32", "Tie": "#888888"}


def load_run_data(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} run CSV not found: {path}")
    df = pd.read_csv(path)
    required = set(KEY + ["initial_cost", "best_cost", "runtime_seconds"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} run CSV missing columns: {sorted(missing)}")
    if df.duplicated(KEY).any():
        raise ValueError(f"{label} run CSV has duplicate instance/method/seed/run rows")
    df["constructive_method"] = df["constructive_method"].str.lower().str.strip()
    return df


def build_comparison(regular: pd.DataFrame, sa: pd.DataFrame) -> pd.DataFrame:
    merged = regular.merge(sa, on=KEY, how="outer", suffixes=("_regular", "_sa"), indicator=True, validate="one_to_one")
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        print(f"WARNING: {len(unmatched)} unmatched runs; only matched pairs are compared.")
        print(unmatched[KEY + ["_merge"]].to_string(index=False))
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge").copy()
    if merged.empty:
        raise ValueError("No matching instance, constructive method, seed and run in the two CSVs")
    if not np.allclose(merged["initial_cost_regular"], merged["initial_cost_sa"], rtol=1e-9, atol=1e-6):
        print("WARNING: Some paired runs have different initial costs.")
    merged["cost_delta"] = merged["best_cost_sa"] - merged["best_cost_regular"]
    merged["cost_delta_pct"] = 100 * merged["cost_delta"] / merged["best_cost_regular"]
    merged["winner"] = np.select(
        [merged["cost_delta"] < -1e-6, merged["cost_delta"] > 1e-6],
        ["ALNS + SA", "Regular ALNS"], default="Tie"
    )
    return merged.sort_values(KEY).reset_index(drop=True)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def parity_plot(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    markers = {"random": "o", "savings": "s"}
    for method in sorted(df["constructive_method"].unique()):
        part = df[df["constructive_method"] == method]
        ax.scatter(part["best_cost_regular"], part["best_cost_sa"], s=46,
                   alpha=.78, marker=markers.get(method, "^"), label=LABELS.get(method, method))
    bounds = np.r_[df["best_cost_regular"].to_numpy(), df["best_cost_sa"].to_numpy()]
    lower, upper = bounds.min(), bounds.max()
    pad = max((upper - lower) * .04, 1)
    ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad],
            "k--", lw=1.2, label="Equal best cost")
    ax.set(xlim=(lower-pad, upper+pad), ylim=(lower-pad, upper+pad),
           xlabel="Regular ALNS best cost", ylabel="ALNS + SA best cost",
           title="Best-cost comparison: regular ALNS vs SA")
    ax.set_aspect("equal", adjustable="box")
    ax.text(.03, .96, "Above line: regular ALNS better\nBelow line: SA better",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, out, "best_cost_parity")


def wins_percentage_plot(df: pd.DataFrame, out: Path):
    methods = [m for m in METHODS if m in df["constructive_method"].unique()]
    methods += [m for m in sorted(df["constructive_method"].unique()) if m not in methods]
    categories = ["ALNS + SA", "Regular ALNS", "Tie"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for cat in categories:
        counts = np.array([((df["constructive_method"] == m) & (df["winner"] == cat)).sum() for m in methods])
        totals = np.array([(df["constructive_method"] == m).sum() for m in methods])
        percentages = 100 * counts / totals
        bars = ax.barh(y, percentages, left=left, label=cat, color=COLORS[cat], height=.56)
        for bar, n, pct in zip(bars, counts, percentages):
            if n and pct >= 6:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                        f"{pct:.1f}%\n({n})", ha="center", va="center", color="white" if cat != "Tie" else "black", fontsize=10)
        left += percentages
    ax.set(yticks=y, yticklabels=[LABELS.get(m, m) for m in methods], xlim=(0, 100),
           xlabel="Share of matched runs (%)", title="Which algorithm achieves the lower best cost?")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "best_cost_wins_percentage")


def per_instance_difference(df: pd.DataFrame, out: Path):
    methods = [m for m in METHODS if m in df["constructive_method"].unique()]
    methods += [m for m in sorted(df["constructive_method"].unique()) if m not in methods]
    # For multiple seeds, average the paired percentage differences for each instance/method.
    grouped = df.groupby(["instance", "constructive_method"], as_index=False)["cost_delta_pct"].mean()
    instances = sorted(grouped["instance"].unique())
    x = np.arange(len(instances))
    width = .8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(12, len(instances)*.62), 6))
    for idx, method in enumerate(methods):
        sub = grouped[grouped["constructive_method"] == method].set_index("instance")
        values = sub.reindex(instances)["cost_delta_pct"].to_numpy()
        ax.bar(x - .4 + width*(idx+.5), values, width=width, label=LABELS.get(method, method))
    ax.axhline(0, color="black", lw=1)
    ax.set(xticks=x, xticklabels=instances, ylabel="Best-cost difference: SA vs regular (%)",
           title="Effect of SA on best cost by instance")
    ax.tick_params(axis="x", rotation=65)
    ax.text(.01, .97, "Negative = SA better | Positive = regular ALNS better",
            transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "per_instance_cost_difference_pct")


def load_iteration_data(path: Path, label: str) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"Skipping convergence plots: {label} iteration CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    required = set(KEY + ["iteration", "best_cost"])
    if required - set(df.columns):
        print(f"Skipping convergence plots: {label} iteration CSV missing {sorted(required-set(df.columns))}")
        return None
    df["constructive_method"] = df["constructive_method"].str.lower().str.strip()
    return df


def convergence_plots(reg: pd.DataFrame, sa: pd.DataFrame, pairs: pd.DataFrame, out: Path, max_plots: int):
    # Only compare histories for matching runs, and plot best-so-far cost (not current cost).
    for _, pair in pairs.head(max_plots).iterrows():
        filters = {k: pair[k] for k in KEY}
        fig, ax = plt.subplots(figsize=(8, 5))
        ok = True
        for frame, label in [(reg, "Regular ALNS"), (sa, "ALNS + SA")]:
            subset = frame
            for k, value in filters.items():
                subset = subset[subset[k] == value]
            if subset.empty:
                ok = False
                break
            subset = subset.sort_values("iteration")
            ax.plot(subset["iteration"], subset["best_cost"], label=label, lw=1.8, color=COLORS[label])
        if not ok:
            plt.close(fig)
            continue
        ax.set(xlabel="Iteration", ylabel="Best-so-far cost",
               title=f"Convergence: {pair['instance']} | {LABELS.get(pair['constructive_method'], pair['constructive_method'])} | seed {pair['seed']}")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        name = f"convergence_{pair['instance']}_{pair['constructive_method']}_seed_{pair['seed']}_run_{pair['run']}"
        save(fig, out, name)


# Edit these settings directly; no command-line arguments are required.
ROOT = Path(__file__).resolve().parent
REGULAR_RUN_CSV = ROOT / "alns_results/alns_run_results.csv"
SA_RUN_CSV = ROOT / "alns_sa_results/alns_run_results.csv"
REGULAR_ITERATIONS_CSV = ROOT / "alns_results/alns_iteration_results.csv"
SA_ITERATIONS_CSV = ROOT / "alns_sa_results/alns_iteration_results.csv"
OUTPUT_DIR = ROOT / "alns_comparison_results"
MAX_CONVERGENCE_PLOTS = 48  # Set to 0 to disable convergence plots.


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regular = load_run_data(REGULAR_RUN_CSV, "Regular ALNS")
    sa = load_run_data(SA_RUN_CSV, "ALNS + SA")
    comparison = build_comparison(regular, sa)
    comparison.to_csv(OUTPUT_DIR / "paired_results.csv", index=False)
    summary = (comparison.groupby("constructive_method")
               .agg(matched_runs=("winner", "size"),
                    sa_wins=("winner", lambda s: (s == "ALNS + SA").sum()),
                    regular_wins=("winner", lambda s: (s == "Regular ALNS").sum()),
                    ties=("winner", lambda s: (s == "Tie").sum()),
                    mean_sa_minus_regular_pct=("cost_delta_pct", "mean"))
               .reset_index())
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    parity_plot(comparison, OUTPUT_DIR)
    wins_percentage_plot(comparison, OUTPUT_DIR)
    per_instance_difference(comparison, OUTPUT_DIR)
    if MAX_CONVERGENCE_PLOTS > 0:
        regular_iter = load_iteration_data(REGULAR_ITERATIONS_CSV, "Regular ALNS")
        sa_iter = load_iteration_data(SA_ITERATIONS_CSV, "ALNS + SA")
        if regular_iter is not None and sa_iter is not None:
            convergence_plots(regular_iter, sa_iter, comparison, OUTPUT_DIR, MAX_CONVERGENCE_PLOTS)
    print(f"\nMatched runs: {len(comparison)}")
    print(summary.to_string(index=False))
    print(f"\nSaved plots (PNG) and CSVs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
