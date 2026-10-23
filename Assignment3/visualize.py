from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from main import (
    INSTANCE_NAMES,
    RESULTS_DIR,
    MAX_ITERATIONS,
    STAGNATION_LIMIT,
    INITIAL_BETA,
    load_instance
)

from UFL import LagrangianHeuristic


SHOW_CLOSED_FACILITIES = True
SHOW_CUSTOMER_LABELS = False
SHOW_FACILITY_LABELS = True
RANDOM_SEED = 42


def make_positions(solution):
    """Create schematic facility and customer positions."""

    rng = np.random.default_rng(RANDOM_SEED)

    open_ids = np.flatnonzero(solution.y == 1)
    closed_ids = np.flatnonzero(solution.y == 0)
    assignments = np.argmax(solution.x, axis=1)

    facility_positions = {}
    customer_positions = np.zeros(
        (solution.instance.n_markets, 2)
    )

    n_open = len(open_ids)
    ring_radius = max(5.0, 2.2 * n_open / np.pi)

    angles = (
        np.linspace(0, 2 * np.pi, n_open, endpoint=False)
        + np.pi / 2
    )

    for j, angle in zip(open_ids, angles):

        center = ring_radius * np.array([
            np.cos(angle),
            np.sin(angle)
        ])

        facility_positions[j] = center

        customer_ids = np.flatnonzero(assignments == j)

        if len(customer_ids) == 0:
            continue

        customer_angles = np.linspace(
            0, 2 * np.pi, len(customer_ids), endpoint=False
        )

        rng.shuffle(customer_angles)

        radii = 0.8 + 1.4 * rng.random(len(customer_ids))

        customer_positions[customer_ids] = (
            center
            + np.column_stack((
                radii * np.cos(customer_angles),
                radii * np.sin(customer_angles)
            ))
        )

    if SHOW_CLOSED_FACILITIES and len(closed_ids):

        cols = min(20, len(closed_ids))
        spacing = 0.48
        y_base = -ring_radius - 3.5

        for index, j in enumerate(closed_ids):

            col = index % cols
            row = index // cols

            facility_positions[j] = np.array([
                (col - (cols - 1) / 2) * spacing,
                y_base - row * spacing
            ])

    return facility_positions, customer_positions, assignments


def plot_network(name, solution, output_dir=RESULTS_DIR):
    """Plot and save the facility-customer assignment network."""

    if not solution.isFeasible():
        raise ValueError("The solution is infeasible.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    facilities, customers, assignments = make_positions(solution)

    open_ids = np.flatnonzero(solution.y == 1)
    closed_ids = np.flatnonzero(solution.y == 0)

    fig, ax = plt.subplots(figsize=(12, 10))

    palette = plt.get_cmap("tab20")

    facility_colors = {
        j: palette(index % 20)
        for index, j in enumerate(open_ids)
    }

    for i, j in enumerate(assignments):

        ax.plot(
            [customers[i, 0], facilities[j][0]],
            [customers[i, 1], facilities[j][1]],
            color=facility_colors[j],
            linewidth=1.0,
            alpha=0.45,
            zorder=1
        )

    for j in open_ids:

        assigned = np.flatnonzero(assignments == j)
        color = facility_colors[j]

        if len(assigned):

            ax.scatter(
                customers[assigned, 0],
                customers[assigned, 1],
                s=34,
                c=[color],
                alpha=0.9,
                edgecolors="none",
                zorder=2
            )

            if SHOW_CUSTOMER_LABELS:

                for i in assigned:
                    ax.annotate(
                        str(i + 1),
                        customers[i],
                        fontsize=11,
                        alpha=0.8
                    )

        ax.scatter(
            *facilities[j],
            marker="s",
            s=290,
            c=[color],
            edgecolors="black",
            linewidths=1.2,
            zorder=4
        )

        if SHOW_FACILITY_LABELS:

            ax.annotate(
                f"F{j + 1} ({len(assigned)})",
                facilities[j],
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=15,
                fontweight="bold",
                zorder=5
            )

    if SHOW_CLOSED_FACILITIES and len(closed_ids):

        coords = np.array([
            facilities[j] for j in closed_ids
        ])

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            marker="s",
            s=30,
            c="lightgray",
            edgecolors="gray",
            linewidths=0.4,
            zorder=2
        )

        ax.text(
            coords[:, 0].min(),
            coords[:, 1].max() + 0.65,
            f"Closed facilities ({len(closed_ids)})",
            fontsize=15,
            color="dimgray"
        )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            markerfacecolor="tab:green",
            markeredgecolor="black",
            markersize=12,
            label="Open facility"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            markersize=8,
            label="Customer"
        ),
        Line2D(
            [0], [0],
            color="gray",
            linewidth=1.2,
            label="Customer assignment"
        )
    ]

    if SHOW_CLOSED_FACILITIES:

        legend_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor="lightgray",
                markeredgecolor="gray",
                markersize=8,
                label="Closed facility"
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=14,
        framealpha=0.95
    )

    ax.set_title(
        f"{name}: Facility–customer assignment network\n"
        f"{len(open_ids)} open facilities | "
        f"{len(assignments)} customers | "
        f"Solution cost: {solution.getCosts():,.2f}",
        fontsize=22,
        pad=25
    )

    all_points = np.vstack((
        np.array(list(facilities.values())),
        customers
    ))

    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    ax.set_xlim(
        x_min - 0.18 * x_range,
        x_max + 0.55 * x_range
    )

    ax.set_ylim(
        y_min - 0.12 * y_range,
        y_max + 0.12 * y_range
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.tight_layout()

    path = output_dir / f"{name}_network.png"

    fig.savefig(
        path,
        dpi=200
    )

    plt.close(fig)

    return path


def main():
    """Generate network figures for all benchmark instances."""

    for name in INSTANCE_NAMES:

        instance = load_instance(name)

        heuristic = LagrangianHeuristic(instance)

        (
            best_lb,
            best_ub,
            solution,
            history,
            stop_reason
        ) = heuristic.runHeuristic(
            max_iterations=MAX_ITERATIONS,
            stagnation_limit=STAGNATION_LIMIT,
            initial_beta=INITIAL_BETA
        )

        if solution is None:
            raise RuntimeError(
                f"No feasible solution found for {name}."
            )

        path = plot_network(name, solution)

        print(
            f"{name}: upper bound {best_ub:.2f} | "
            f"network saved to {path}"
        )


if __name__ == "__main__":
    main()
