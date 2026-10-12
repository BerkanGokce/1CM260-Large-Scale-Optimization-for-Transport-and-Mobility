
"""Destroy operators for the two-echelon capacitated VRP."""

import math


class Destroy:
    """Customer-level removal operators used by the ALNS."""

    # Shaw relatedness weights. Lower scores mean more closely related.
    # Geography is deliberately the dominant component.
    SHAW_DISTANCE_WEIGHT = 0.60
    SHAW_DEMAND_WEIGHT = 0.20
    SHAW_SATELLITE_WEIGHT = 0.20

    @staticmethod
    def destroy1(solution, remove_count, random_gen):
        """
        Random Removal.

        Uniformly select customers from all currently served customers
        and remove them from their second-echelon routes.

        Returns the IDs of removed customers.
        """

        # Collect all currently served customer IDs.
        available_customers = Destroy._served_customer_ids(solution)

        # Randomly select customers without replacement.
        selected_customers = random_gen.sample(
            available_customers,
            min(max(remove_count, 0), len(available_customers))
        )

        # Remove each selected customer from its route.
        for customer_ID in selected_customers:

            Destroy._remove_customer_by_id(solution, customer_ID)

        # Remove empty second-echelon routes.
        Destroy.remove_empty_routes(solution)

        return selected_customers

    @staticmethod
    def destroy2(solution, remove_count, random_gen):
        """
        Worst removal based on marginal second-echelon travel distance.

        Savings are recalculated after every removal. The customer with the
        largest saving is removed, with random selection only among numerical
        ties.
        """
        target_count = min(
            max(remove_count, 0),
            len(Destroy._served_customer_ids(solution))
        )
        removed_customers = []

        while len(removed_customers) < target_count:
            savings = Destroy._removal_savings(solution)
            if not savings:
                break

            highest_saving = max(item[1] for item in savings)
            tied_candidates = [
                item
                for item in savings
                if math.isclose(
                    item[1],
                    highest_saving,
                    rel_tol=1e-12,
                    abs_tol=1e-12
                )
            ]
            customer_ID, _, route, position = random_gen.choice(
                tied_candidates
            )

            Destroy._remove_customer_by_id(
                solution,
                customer_ID,
                route_hint=route,
                position_hint=position
            )
            removed_customers.append(customer_ID)

        Destroy.remove_empty_routes(solution)
        return removed_customers

    @staticmethod
    def destroy3(solution, remove_count, random_gen):
        """
        Shaw related removal adapted to the 2E-CVRP.

        Relatedness combines normalized geographic distance, normalized
        demand difference, and a same-satellite indicator. A random removed
        customer is used as the reference in each iteration.
        """
        available_customers = Destroy._served_customer_ids(solution)
        target_count = min(
            max(remove_count, 0),
            len(available_customers)
        )

        if target_count == 0:
            Destroy.remove_empty_routes(solution)
            return []

        satellite_assignments = Destroy._satellite_assignments(solution)
        maximum_distance = Destroy._maximum_customer_distance(
            solution,
            available_customers
        )
        demands = [
            solution.customers[customer_ID]["demand"]
            for customer_ID in available_customers
        ]
        demand_range = max(demands) - min(demands)

        seed = random_gen.choice(available_customers)
        removed_customers = [seed]
        remaining_customers = [
            customer_ID
            for customer_ID in available_customers
            if customer_ID != seed
        ]
        Destroy._remove_customer_by_id(solution, seed)

        while len(removed_customers) < target_count:
            reference = random_gen.choice(removed_customers)
            scores = [
                (
                    customer_ID,
                    Destroy._relatedness(
                        solution,
                        reference,
                        customer_ID,
                        satellite_assignments,
                        maximum_distance,
                        demand_range
                    )
                )
                for customer_ID in remaining_customers
            ]

            best_score = min(score for _, score in scores)
            tied_customers = [
                customer_ID
                for customer_ID, score in scores
                if math.isclose(
                    score,
                    best_score,
                    rel_tol=1e-12,
                    abs_tol=1e-12
                )
            ]
            customer_ID = random_gen.choice(tied_customers)

            Destroy._remove_customer_by_id(solution, customer_ID)
            removed_customers.append(customer_ID)
            remaining_customers.remove(customer_ID)

        Destroy.remove_empty_routes(solution)
        return removed_customers

    @staticmethod
    def remove_empty_routes(solution):
        """
        Remove second-echelon routes that serve no customers.
        """
        solution.routes_second = [
            route
            for route in solution.routes_second
            if route.served_load and len(route.locations) > 2
        ]

    @staticmethod
    def _served_customer_ids(solution):
        """Return distinct served customer IDs in route traversal order."""
        customer_IDs = []
        seen = set()

        for route in solution.routes_second:
            for customer_ID in route.served_load:
                if customer_ID not in seen:
                    customer_IDs.append(customer_ID)
                    seen.add(customer_ID)

        return customer_IDs

    @staticmethod
    def _remove_customer_by_id(
        solution,
        customer_ID,
        route_hint=None,
        position_hint=None
    ):
        """
        Remove an exact customer ID and one corresponding coordinate stop.

        Route.remove_customer accepts only a coordinate and is ambiguous for
        the duplicate customer coordinates present in the supplied instances.
        """
        if customer_ID not in solution.customers:
            raise ValueError(f"Unknown customer {customer_ID}.")

        customer_location = solution.customers[customer_ID]["location"]

        routes = (
            [route_hint]
            if route_hint is not None
            else solution.routes_second
        )

        for route in routes:
            if route not in solution.routes_second:
                continue
            if customer_ID not in route.served_load:
                continue

            if (
                position_hint is not None
                and 0 < position_hint < len(route.locations) - 1
                and route.locations[position_hint] == customer_location
            ):
                position = position_hint
            else:
                position = next(
                    (
                        index
                        for index in range(1, len(route.locations) - 1)
                        if route.locations[index] == customer_location
                    ),
                    None
                )
            if position is None:
                raise ValueError(
                    f"Customer {customer_ID} is recorded on a route "
                    "but has no matching route stop."
                )

            route.locations.pop(position)
            del route.served_load[customer_ID]

            if hasattr(route, "customers"):
                route.customers.discard(customer_ID)

            route.cost = route.compute_cost()
            return

        raise ValueError(f"Customer {customer_ID} is not currently served.")

    @staticmethod
    def _removal_savings(solution):
        """Calculate current marginal-distance savings for every customer."""
        savings = []

        for route in solution.routes_second:
            unused_positions = set(range(1, len(route.locations) - 1))

            for customer_ID in route.served_load:
                customer_location = solution.customers[customer_ID]["location"]
                position = next(
                    (
                        index
                        for index in sorted(unused_positions)
                        if route.locations[index] == customer_location
                    ),
                    None
                )
                if position is None:
                    raise ValueError(
                        f"Customer {customer_ID} is recorded on a route "
                        "but has no matching route stop."
                    )

                unused_positions.remove(position)
                previous_location = route.locations[position - 1]
                next_location = route.locations[position + 1]
                saving = (
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
                savings.append((customer_ID, saving, route, position))

        return savings

    @staticmethod
    def _satellite_assignments(solution):
        """Map each served customer to its current route's satellite."""
        satellite_by_location = {
            tuple(data["location"]): satellite_ID
            for satellite_ID, data in solution.satellites.items()
        }
        assignments = {}

        for route in solution.routes_second:
            satellite_ID = satellite_by_location.get(tuple(route.locations[0]))
            if satellite_ID is None:
                raise ValueError(
                    "Second-echelon route does not originate at a satellite."
                )

            for customer_ID in route.served_load:
                assignments[customer_ID] = satellite_ID

        return assignments

    @staticmethod
    def _maximum_customer_distance(solution, customer_IDs):
        """Return the largest pairwise distance used for normalization."""
        maximum_distance = 0.0

        for index, first_ID in enumerate(customer_IDs):
            first_location = solution.customers[first_ID]["location"]

            for second_ID in customer_IDs[index + 1:]:
                second_location = solution.customers[second_ID]["location"]
                maximum_distance = max(
                    maximum_distance,
                    solution.problem.get_distance(
                        first_location,
                        second_location
                    )
                )

        return maximum_distance

    @staticmethod
    def _relatedness(
        solution,
        first_ID,
        second_ID,
        satellite_assignments,
        maximum_distance,
        demand_range
    ):
        """Return a normalized Shaw score; lower values are more related."""
        first_customer = solution.customers[first_ID]
        second_customer = solution.customers[second_ID]

        if maximum_distance > 0:
            distance_component = solution.problem.get_distance(
                first_customer["location"],
                second_customer["location"]
            ) / maximum_distance
        else:
            distance_component = 0.0

        if demand_range > 0:
            demand_component = abs(
                first_customer["demand"] - second_customer["demand"]
            ) / demand_range
        else:
            demand_component = 0.0

        satellite_component = (
            0.0
            if satellite_assignments[first_ID]
            == satellite_assignments[second_ID]
            else 1.0
        )

        return (
            Destroy.SHAW_DISTANCE_WEIGHT * distance_component
            + Destroy.SHAW_DEMAND_WEIGHT * demand_component
            + Destroy.SHAW_SATELLITE_WEIGHT * satellite_component
        )
