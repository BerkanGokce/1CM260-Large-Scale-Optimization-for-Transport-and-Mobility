"""Repair operators for the two-echelon capacitated VRP."""

import math

from route import Route


class Repair:
    """Customer insertion operators and shared first-echelon rebuilding."""

    @staticmethod
    def repair1(solution, removed_customers, random_gen):
        """Repair by global greedy cheapest insertion."""
        pending = Repair._prepare_removed_customers(
            solution,
            removed_customers
        )

        while pending:
            customer_choices = []

            for customer_ID in pending:
                options = Repair._second_echelon_options(
                    solution,
                    customer_ID
                )
                if not options:
                    raise ValueError(
                        f"No feasible insertion for customer {customer_ID}."
                    )

                best_option = options[0]
                customer_choices.append(
                    (best_option["cost"], customer_ID, best_option)
                )

            _, customer_ID, option = min(
                customer_choices,
                key=lambda item: (
                    item[0],
                    item[1],
                    Repair._option_sort_key(item[2])
                )
            )
            Repair._apply_second_echelon_option(
                solution,
                customer_ID,
                option
            )
            pending.remove(customer_ID)

        Repair._finalize_repair(solution)

    @staticmethod
    def repair2(solution, removed_customers, random_gen):
        """Repair by regret-2 insertion using distinct route alternatives."""
        pending = Repair._prepare_removed_customers(
            solution,
            removed_customers
        )

        while pending:
            customer_choices = []

            for customer_ID in pending:
                options = Repair._second_echelon_options(
                    solution,
                    customer_ID
                )
                alternatives = Repair._distinct_route_alternatives(options)

                if not alternatives:
                    raise ValueError(
                        f"No feasible insertion for customer {customer_ID}."
                    )

                best_option = alternatives[0]

                if len(alternatives) == 1:
                    # A customer with no backup insertion is handled first.
                    priority_group = 0
                    regret = 0.0
                else:
                    priority_group = 1
                    regret = (
                        alternatives[1]["cost"]
                        - best_option["cost"]
                    )

                customer_choices.append(
                    (
                        priority_group,
                        -regret,
                        best_option["cost"],
                        customer_ID,
                        best_option
                    )
                )

            _, _, _, customer_ID, option = min(
                customer_choices,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                    Repair._option_sort_key(item[4])
                )
            )
            Repair._apply_second_echelon_option(
                solution,
                customer_ID,
                option
            )
            pending.remove(customer_ID)

        Repair._finalize_repair(solution)

    @staticmethod
    def repair3(solution, removed_customers, random_gen):
        """Repair by uniformly randomized feasible insertion."""
        pending = Repair._prepare_removed_customers(
            solution,
            removed_customers
        )

        while pending:
            customer_ID = random_gen.choice(pending)
            options = Repair._second_echelon_options(
                solution,
                customer_ID
            )

            if not options:
                raise ValueError(
                    f"No feasible insertion for customer {customer_ID}."
                )

            option = random_gen.choice(options)
            Repair._apply_second_echelon_option(
                solution,
                customer_ID,
                option
            )
            pending.remove(customer_ID)

        Repair._finalize_repair(solution)

    @staticmethod
    def rebuild_first_echelon(solution):
        """
        Rebuild first-echelon supply from actual second-echelon assignments.

        Every candidate delivers the maximum useful amount allowed by its
        remaining route capacity and satellite demand. Candidates are ranked
        by incremental distance per delivered unit, then by higher capacity
        utilization and lower raw distance. This prevents a tiny split from
        winning solely because it has a small absolute insertion distance.
        """
        satellite_by_location = {
            tuple(data["location"]): satellite_ID
            for satellite_ID, data in solution.satellites.items()
        }
        if len(satellite_by_location) != len(solution.satellites):
            raise ValueError(
                "First-echelon reconstruction requires unique satellite "
                "coordinates."
            )

        satellite_demand = {
            satellite_ID: 0
            for satellite_ID in solution.satellites
        }

        for route in solution.routes_second:
            satellite_ID = satellite_by_location.get(
                tuple(route.locations[0])
            )
            if satellite_ID is None:
                raise ValueError(
                    "Second-echelon route does not originate at a satellite."
                )

            satellite_demand[satellite_ID] += sum(
                route.served_load.values()
            )

        # Only first-echelon routes are discarded. The repaired customer
        # routes remain unchanged.
        solution.routes_first = []

        remaining_demand = {
            satellite_ID: demand
            for satellite_ID, demand in satellite_demand.items()
            if demand > 0
        }

        if not remaining_demand:
            return

        capacity = solution.problem.capacity_first
        if capacity <= 0:
            raise ValueError("First-echelon capacity must be positive.")
        if not solution.depots:
            raise ValueError(
                "Cannot reconstruct first-echelon routes without a depot."
            )

        while remaining_demand:
            options = Repair._first_echelon_options(
                solution,
                remaining_demand,
                capacity
            )
            if not options:
                raise ValueError(
                    "Unable to supply all satellite demand in the first "
                    "echelon."
                )

            option = min(options, key=Repair._first_option_sort_key)
            delivered = option["delivery"]
            if delivered <= 0:
                raise ValueError(
                    "First-echelon reconstruction made no delivery progress."
                )

            Repair._apply_first_echelon_option(solution, option)

            satellite_ID = option["satellite_id"]
            remaining = remaining_demand[satellite_ID] - delivered

            if remaining <= 0 or math.isclose(
                remaining,
                0.0,
                rel_tol=1e-12,
                abs_tol=1e-12
            ):
                del remaining_demand[satellite_ID]
            else:
                remaining_demand[satellite_ID] = remaining

    @staticmethod
    def _prepare_removed_customers(solution, removed_customers):
        """Validate and return the pending customer IDs in supplied order."""
        pending = list(removed_customers)

        if len(pending) != len(set(pending)):
            raise ValueError("Removed customer IDs must be distinct.")

        unknown = [
            customer_ID
            for customer_ID in pending
            if customer_ID not in solution.customers
        ]
        if unknown:
            raise ValueError(f"Unknown removed customers: {unknown}.")

        served = {
            customer_ID
            for route in solution.routes_second
            for customer_ID in route.served_load
        }
        already_served = [
            customer_ID
            for customer_ID in pending
            if customer_ID in served
        ]
        if already_served:
            raise ValueError(
                "Cannot repair customers that are still served: "
                f"{already_served}."
            )

        return pending

    @staticmethod
    def _second_echelon_options(solution, customer_ID):
        """Enumerate every feasible existing-route position and new route."""
        customer = solution.customers[customer_ID]
        customer_location = customer["location"]
        demand = customer["demand"]
        capacity = solution.problem.capacity_second
        options = []

        if capacity <= 0:
            raise ValueError("Second-echelon capacity must be positive.")

        for route_index, route in enumerate(solution.routes_second):
            route_load = sum(route.served_load.values())
            if route_load + demand > capacity:
                continue

            for position in range(1, len(route.locations)):
                previous_location = route.locations[position - 1]
                next_location = route.locations[position]
                added_distance = (
                    solution.problem.get_distance(
                        previous_location,
                        customer_location
                    )
                    + solution.problem.get_distance(
                        customer_location,
                        next_location
                    )
                    - solution.problem.get_distance(
                        previous_location,
                        next_location
                    )
                )
                options.append(
                    {
                        "cost": max(0.0, added_distance),
                        "kind": "existing",
                        "route_index": route_index,
                        "position": position,
                        "satellite_id": None,
                    }
                )

        if demand <= capacity:
            for satellite_ID in sorted(solution.satellites):
                route_distance = 2 * solution.problem.distance_second[
                    "satellite-customer"
                ][satellite_ID, customer_ID]
                options.append(
                    {
                        "cost": route_distance,
                        "kind": "new",
                        "route_index": None,
                        "position": 1,
                        "satellite_id": satellite_ID,
                    }
                )

        options.sort(key=Repair._option_sort_key)
        return options

    @staticmethod
    def _option_sort_key(option):
        """Deterministic ordering for equal-cost insertion options."""
        if option["kind"] == "existing":
            return (
                option["cost"],
                0,
                option["route_index"],
                option["position"]
            )

        return (
            option["cost"],
            1,
            option["satellite_id"],
            option["position"]
        )

    @staticmethod
    def _distinct_route_alternatives(options):
        """Keep the cheapest position for each existing or potential route."""
        alternatives = {}

        for option in options:
            if option["kind"] == "existing":
                key = ("existing", option["route_index"])
            else:
                key = ("new", option["satellite_id"])

            if key not in alternatives:
                alternatives[key] = option

        return sorted(
            alternatives.values(),
            key=Repair._option_sort_key
        )

    @staticmethod
    def _apply_second_echelon_option(solution, customer_ID, option):
        """Apply one selected customer insertion option."""
        customer_location = solution.customers[customer_ID]["location"]
        demand = solution.customers[customer_ID]["demand"]

        if option["kind"] == "existing":
            route_index = option["route_index"]
            route = solution.routes_second[route_index]
            locations = route.locations.copy()
            locations.insert(option["position"], customer_location)
            served_load = route.served_load.copy()
            served_load[customer_ID] = demand

            updated_route = Route(
                solution.problem,
                locations,
                served_load,
                False
            )
            updated_route.customers = set(served_load)
            solution.routes_second[route_index] = updated_route
            return

        satellite_ID = option["satellite_id"]
        satellite_location = solution.satellites[satellite_ID]["location"]
        new_route = Route(
            solution.problem,
            [satellite_location, customer_location, satellite_location],
            {customer_ID: demand},
            False
        )
        new_route.customers = {customer_ID}
        solution.routes_second.append(new_route)

    @staticmethod
    def _first_echelon_options(solution, remaining_demand, capacity):
        """Enumerate maximum-useful-load first-echelon insertions."""
        options = []

        for satellite_ID in sorted(remaining_demand):
            demand = remaining_demand[satellite_ID]
            satellite_location = solution.satellites[satellite_ID]["location"]

            for route_index, route in enumerate(solution.routes_first):
                if satellite_ID in route.served_load:
                    continue

                available_capacity = (
                    capacity - sum(route.served_load.values())
                )
                delivery = min(demand, available_capacity)
                if delivery <= 0:
                    continue

                best_position = None
                best_distance = None

                for position in range(1, len(route.locations)):
                    previous_location = route.locations[position - 1]
                    next_location = route.locations[position]
                    added_distance = (
                        solution.problem.get_distance(
                            previous_location,
                            satellite_location
                        )
                        + solution.problem.get_distance(
                            satellite_location,
                            next_location
                        )
                        - solution.problem.get_distance(
                            previous_location,
                            next_location
                        )
                    )
                    added_distance = max(0.0, added_distance)

                    if (
                        best_distance is None
                        or (added_distance, position)
                        < (best_distance, best_position)
                    ):
                        best_distance = added_distance
                        best_position = position

                options.append(
                    Repair._make_first_option(
                        kind="existing",
                        satellite_ID=satellite_ID,
                        delivery=delivery,
                        added_distance=best_distance,
                        capacity=capacity,
                        route_index=route_index,
                        position=best_position,
                        depot_ID=None
                    )
                )

            delivery = min(demand, capacity)

            for depot_ID in sorted(solution.depots):
                route_distance = 2 * solution.problem.distance_first[
                    "depot-satellite"
                ][depot_ID, satellite_ID]
                options.append(
                    Repair._make_first_option(
                        kind="new",
                        satellite_ID=satellite_ID,
                        delivery=delivery,
                        added_distance=route_distance,
                        capacity=capacity,
                        route_index=None,
                        position=1,
                        depot_ID=depot_ID
                    )
                )

        return options

    @staticmethod
    def _make_first_option(
        kind,
        satellite_ID,
        delivery,
        added_distance,
        capacity,
        route_index,
        position,
        depot_ID
    ):
        """Build a first-echelon option with its utilization-aware score."""
        return {
            "kind": kind,
            "satellite_id": satellite_ID,
            "delivery": delivery,
            "added_distance": added_distance,
            "distance_per_unit": added_distance / delivery,
            "utilization": delivery / capacity,
            "route_index": route_index,
            "position": position,
            "depot_id": depot_ID,
        }

    @staticmethod
    def _first_option_sort_key(option):
        """Deterministically rank utilization-aware first-echelon options."""
        if option["kind"] == "existing":
            source_key = (0, option["route_index"], option["position"])
        else:
            source_key = (1, option["depot_id"], option["position"])

        return (
            option["distance_per_unit"],
            -option["utilization"],
            option["added_distance"],
            option["satellite_id"],
            source_key
        )

    @staticmethod
    def _apply_first_echelon_option(solution, option):
        """Apply one first-echelon satellite delivery option."""
        satellite_ID = option["satellite_id"]
        satellite_location = solution.satellites[satellite_ID]["location"]
        delivery = option["delivery"]

        if option["kind"] == "existing":
            route_index = option["route_index"]
            route = solution.routes_first[route_index]
            locations = route.locations.copy()
            locations.insert(option["position"], satellite_location)
            served_load = route.served_load.copy()
            served_load[satellite_ID] = delivery

            solution.routes_first[route_index] = Route(
                solution.problem,
                locations,
                served_load,
                True
            )
            return

        depot_ID = option["depot_id"]
        depot_location = solution.depots[depot_ID]["location"]
        solution.routes_first.append(
            Route(
                solution.problem,
                [depot_location, satellite_location, depot_location],
                {satellite_ID: delivery},
                True
            )
        )

    @staticmethod
    def _finalize_repair(solution):
        """Rebuild supply, synchronize costs, and validate completeness."""
        Repair.rebuild_first_echelon(solution)

        for route in solution.routes_first + solution.routes_second:
            route.cost = route.compute_cost()

        solution.validate_solution()
        solution.compute_cost()
