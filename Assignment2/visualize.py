
"""
Visualization of solutions to the two-echelon vehicle routing problem.

Supports solutions constructed by Random Insertion, Clarke--Wright,
or Adaptive Large Neighborhood Search (ALNS).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class SolutionVisualizer:
    """
    Visualize a two-echelon vehicle routing solution.

    Parameters
    ----------
    solution : Solution
        A completed solution containing first- and second-echelon routes.
    """

    def __init__(self, solution):

        self.solution = solution
        self.problem = solution.problem

        self.depots = self.problem.depots
        self.satellites = self.problem.satellites
        self.customers = self.problem.customers

        # Font sizes used throughout the visualization.
        self.font_sizes = {
            "title": 20,
            "axis_label": 17,
            "tick": 14,
            "legend": 14,
            "node_label": 12,
            "route_load": 12,
            "cost": 14
        }

    def plot(
        self,
        echelon="both",
        show_arrows=True,
        show_labels=False,
        show_loads=False,
        title=None,
        instance_name=None,
        save_path=None,
        figsize=(12, 9),
        show=True
    ):
        """
        Plot the vehicle routes and network locations.

        Parameters
        ----------
        echelon : str
            "both", "first", or "second".
        show_arrows : bool
            Whether to draw arrows indicating travel direction.
        show_labels : bool
            Whether to display depot, satellite, and customer IDs.
        show_loads : bool
            Whether to display route loads near their first stop.
        title : str or None
            Custom figure title.
        instance_name : str or None
            Name of the problem instance, displayed below the title.
        save_path : str or None
            File path for saving the figure.
        figsize : tuple
            Figure dimensions.
        show : bool
            Whether to display the figure interactively.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """

        if echelon not in ("both", "first", "second"):
            raise ValueError(
                "echelon must be 'both', 'first', or 'second'."
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Plot first-echelon routes.
        if echelon in ("both", "first"):
            self._plot_routes(
                ax,
                self.solution.routes_first,
                is_first_echelon=True,
                show_arrows=show_arrows,
                show_loads=show_loads
            )

        # Plot second-echelon routes.
        if echelon in ("both", "second"):
            self._plot_routes(
                ax,
                self.solution.routes_second,
                is_first_echelon=False,
                show_arrows=show_arrows,
                show_loads=show_loads
            )

        # Plot nodes above the routes.
        self._plot_locations(ax, show_labels)

        # Axis labels.
        ax.set_xlabel(
            "X coordinate",
            fontsize=self.font_sizes["axis_label"]
        )

        ax.set_ylabel(
            "Y coordinate",
            fontsize=self.font_sizes["axis_label"]
        )

        # Axis tick labels.
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.font_sizes["tick"]
        )

        # Preserve the spatial proportions of the network.
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.25)

        # Plot title.
        if title is None:
            title = "Two-Echelon Vehicle Routing Solution"

        if instance_name is not None:
            title = f"{title}\nInstance: {Path(instance_name).stem}"

        ax.set_title(
            title,
            fontsize=self.font_sizes["title"],
            pad=18
        )

        # Display solution cost if available.
        cost = getattr(self.solution, "cost", None)

        if cost is not None:
            ax.text(
                0.01,
                0.99,
                f"Total cost: {cost:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=self.font_sizes["cost"],
                bbox=dict(
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.9
                ),
                zorder=20
            )

        # Add legend.
        self._add_legend(ax, echelon)

        fig.tight_layout()

        # Save the figure if a path is provided.
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        return fig, ax

    def _plot_locations(self, ax, show_labels):
        """
        Plot depots, satellites, and customers using filled markers.
        """

        # Depots: blue filled squares.
        for depot_ID, depot in self.depots.items():

            x, y = depot["location"]

            ax.scatter(
                x, y,
                marker="s",
                s=180,
                facecolor="royalblue",
                edgecolor="black",
                linewidth=1.2,
                zorder=10
            )

            if show_labels:
                ax.annotate(
                    str(depot_ID),
                    (x, y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=self.font_sizes["node_label"],
                    zorder=11
                )

        # Satellites: green filled triangles.
        for satellite_ID, satellite in self.satellites.items():

            x, y = satellite["location"]

            ax.scatter(
                x, y,
                marker="^",
                s=200,
                facecolor="forestgreen",
                edgecolor="black",
                linewidth=1.2,
                zorder=10
            )

            if show_labels:
                ax.annotate(
                    str(satellite_ID),
                    (x, y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=self.font_sizes["node_label"],
                    zorder=11
                )

        # Customers: red filled circles.
        for customer_ID, customer in self.customers.items():

            x, y = customer["location"]

            ax.scatter(
                x, y,
                marker="o",
                s=65,
                facecolor="red",
                edgecolor="black",
                linewidth=0.7,
                zorder=10
            )

            if show_labels:
                ax.annotate(
                    str(customer_ID),
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=self.font_sizes["node_label"],
                    zorder=11
                )

    def _plot_routes(
        self,
        ax,
        routes,
        is_first_echelon,
        show_arrows,
        show_loads
    ):
        """
        Plot first- or second-echelon routes.

        First-echelon routes: black dashed lines.
        Second-echelon routes: black solid lines.
        """

        if not routes:
            return

        color = "black"

        linestyle = "--" if is_first_echelon else "-"
        linewidth = 1.8 if is_first_echelon else 1.3

        for route in routes:

            locations = route.locations

            if len(locations) < 2:
                continue

            # Draw consecutive route segments.
            for start, end in zip(locations[:-1], locations[1:]):

                x1, y1 = start
                x2, y2 = end

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linestyle=linestyle,
                    color=color,
                    linewidth=linewidth,
                    alpha=0.85,
                    zorder=2
                )

                if show_arrows:
                    self._draw_arrow(
                        ax,
                        start,
                        end,
                        color,
                        linestyle
                    )

            # Optionally display route loads.
            if show_loads and len(locations) > 2:

                route_load = sum(route.served_load.values())

                x, y = locations[1]

                ax.annotate(
                    f"L={route_load:g}",
                    (x, y),
                    xytext=(8, -12),
                    textcoords="offset points",
                    fontsize=self.font_sizes["route_load"],
                    color="black",
                    zorder=11
                )

    def _draw_arrow(
        self,
        ax,
        start,
        end,
        color,
        linestyle
    ):
        """
        Draw a directional arrow along a route segment.
        """

        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1

        if math.hypot(dx, dy) < 1e-12:
            return

        # Position the arrow between 40% and 60% of the segment.
        arrow_start = (
            x1 + 0.40 * dx,
            y1 + 0.40 * dy
        )

        arrow_end = (
            x1 + 0.60 * dx,
            y1 + 0.60 * dy
        )

        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                mutation_scale=12
            ),
            zorder=4
        )

    def _add_legend(self, ax, echelon):
        """
        Add a legend for location types and route types.
        """

        handles = [
            Line2D(
                [0], [0],
                marker="s",
                color="black",
                markerfacecolor="royalblue",
                markersize=11,
                linestyle="None",
                label="Depot"
            ),
            Line2D(
                [0], [0],
                marker="^",
                color="black",
                markerfacecolor="forestgreen",
                markersize=11,
                linestyle="None",
                label="Satellite"
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="black",
                markerfacecolor="red",
                markersize=9,
                linestyle="None",
                label="Customer"
            )
        ]

        if echelon in ("both", "first"):
            handles.append(
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="--",
                    linewidth=1.8,
                    label="First-echelon route"
                )
            )

        if echelon in ("both", "second"):
            handles.append(
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="-",
                    linewidth=1.3,
                    label="Second-echelon route"
                )
            )

        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=self.font_sizes["legend"],
            frameon=False
        )
