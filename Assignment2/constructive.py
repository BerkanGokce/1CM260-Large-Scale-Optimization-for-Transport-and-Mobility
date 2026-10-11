
"""Constructive heuristics for the two-echelon capacitated VRP."""

import copy

from route import Route
from solution import Solution


class RandomInsertion:
    """Build routes on an existing Solution using a constructive heuristic."""

    def __init__(self, solution):
        self.solution = solution

    @classmethod
    def build(cls, problem, random_gen):
        """Create and return a complete Solution."""
        solution = Solution(problem)
        cls(solution).construct(random_gen)
        solution.compute_cost()
        return solution

    def construct(self, random_gen):
        """Run the constructive heuristic on the associated solution."""
        self.do_random_insertion(random_gen)
        return self.solution

    def do_random_insertion(self, random_gen):
        """
        Construct first- and second-echelon routes using random insertion.

        Customers are first randomly inserted into second-echelon routes.
        The resulting satellite demands are then used to construct the
        first-echelon routes.
        """
        # Construct the second-echelon routes.
        self.do_random_insertion_second(random_gen)

        # Construct the first-echelon routes.
        self.do_random_insertion_first(random_gen)

    def do_random_insertion_first(self, random_gen):
        """
        Construct first-echelon routes by randomly inserting satellites.
        """
        # Initialize the list of first-echelon routes.
        self.solution.routes_first = []

        # Initialize the lists of served and unserved satellites.
        # Preserve the professor's original algorithm: include all satellites.
        not_served = list(self.solution.satellites)
        served = []

        # Determine the total demand associated with each satellite from
        # the second-echelon routes.
        demand_not_served = {
            satellite_ID: 0
            for satellite_ID in self.solution.satellites
        }

        for route in self.solution.routes_second:
            satellite = route.locations[0]
            satellite_ID = next(
                key
                for key, value in self.solution.satellites.items()
                if value["location"] == satellite
            )

            total_demand = sum(route.served_load.values())
            demand_not_served[satellite_ID] += total_demand

        # Assign satellite demand until all satellites are served.
        satellites = copy.deepcopy(self.solution.satellites)

        for satellite in satellites.values():
            satellite["demand"] = 0

        while not_served:
            # Select a random satellite and determine its remaining demand.
            satellite_ID = random_gen.choice(not_served)
            load_max = demand_not_served[satellite_ID]

            # Consider the existing routes as potential insertion routes.
            routes = self.solution.routes_first.copy()
            inserted = False

            while routes:
                # Select a random route.
                route = random_gen.choice(routes)

                # Compute the remaining capacity for the selected route.
                remaining_cap = (
                    self.solution.problem.capacity_first
                    - sum(route.served_load.values())
                )

                if remaining_cap <= 0:
                    break

                # Assign the maximum load that can be added to the route.
                if load_max > remaining_cap:
                    load = remaining_cap
                else:
                    load = load_max

                satellites[satellite_ID]["demand"] = load

                # Try to insert the satellite at its best feasible position.
                after_insertion, feasible_insertion = (
                    self.solution.greedy_insertion(
                        route, satellites, satellite_ID, True
                    )
                )

                # Update the route if a feasible insertion was found.
                if feasible_insertion:
                    inserted = True
                    self.solution.routes_first.remove(route)
                    self.solution.routes_first.append(after_insertion)
                    break
                else:
                    routes.remove(route)

            # If no existing route can accommodate the load, create a new route.
            if not inserted:
                # Select a depot to start and end the new route.
                depot_ID = random_gen.choice(list(self.solution.depots))
                depot_loc = self.solution.depots[depot_ID]["location"]

                locations = [
                    depot_loc,
                    self.solution.satellites[satellite_ID]["location"],
                    depot_loc
                ]

                # Assign the maximum load that can be served by a new vehicle.
                if load_max > self.solution.problem.capacity_first:
                    load = self.solution.problem.capacity_first
                else:
                    load = load_max

                satellites[satellite_ID]["demand"] = load

                # Create and add the new first-echelon route.
                demand = {
                    satellite_ID: satellites[satellite_ID]["demand"]
                }

                new_route = Route(
                    self.solution.problem,
                    locations,
                    demand,
                    True
                )

                self.solution.routes_first.append(new_route)

            # Update the remaining demand and the list of unserved satellites.
            demand_not_served[satellite_ID] -= load

            if demand_not_served[satellite_ID] <= 0:
                served.append(satellite_ID)
                not_served.remove(satellite_ID)

    def do_random_insertion_second(self, random_gen):
        """
        Construct second-echelon routes by randomly inserting customers.
        """
        # Initialize the list of second-echelon routes.
        self.solution.routes_second = []

        # Initialize the lists of served and unserved customers.
        not_served = list(self.solution.customers)
        served = []

        # Insert customers until all customers are served.
        while not_served:
            # Select a random customer.
            customer_ID = random_gen.choice(not_served)

            # Consider the existing routes as potential insertion routes.
            routes = self.solution.routes_second.copy()
            inserted = False

            while routes:
                # Select a random route.
                route = random_gen.choice(routes)

                # Try to insert the customer at its best feasible position.
                after_insertion, feasible_insertion = (
                    self.solution.greedy_insertion(
                        route,
                        self.solution.customers,
                        customer_ID,
                        False
                    )
                )

                # Update the route if a feasible insertion was found.
                if feasible_insertion:
                    inserted = True
                    self.solution.routes_second.remove(route)
                    self.solution.routes_second.append(after_insertion)
                    break
                else:
                    routes.remove(route)

            # If no existing route can accommodate the customer, create a new route.
            if not inserted:
                # Select a satellite to start and end the new route.
                satellite_ID = random_gen.choice(
                    list(self.solution.satellites)
                )

                satellite_loc = (
                    self.solution.satellites[satellite_ID]["location"]
                )

                locations = [
                    satellite_loc,
                    self.solution.customers[customer_ID]["location"],
                    satellite_loc
                ]

                # Create and add the new second-echelon route.
                demand = {
                    customer_ID: self.solution.customers[customer_ID]["demand"]
                }

                new_route = Route(
                    self.solution.problem,
                    locations,
                    demand,
                    False
                )

                new_route.customers = {customer_ID}
                self.solution.routes_second.append(new_route)

            # Mark the customer as served.
            served.append(customer_ID)
            not_served.remove(customer_ID)


class ClarkeWrightSavings:
    """Build routes on an existing Solution using a constructive heuristic."""

    def __init__(self, solution):
        self.solution = solution

    @classmethod
    def build(cls, problem, random_gen):
        """Create and return a complete Solution."""
        solution = Solution(problem)
        cls(solution).construct(random_gen)
        solution.compute_cost()
        return solution

    def construct(self, random_gen):
        """Run the constructive heuristic on the associated solution."""
        self.do_clarke_wright(random_gen)
        return self.solution

    def do_clarke_wright(self, random_gen):
        """
        Construct both echelons using distance-biased customer assignment
        and the parallel Clarke--Wright savings algorithm.

        Full first-echelon truckloads are allocated before merging residual
        satellite demands, following the original oversized-demand approach.
        """
        assignments = self.assign_customers_to_satellites_cw(random_gen)

        self.clarke_wright_second(assignments, random_gen)
        self.clarke_wright_first(random_gen)

        self.solution.validate_solution()

    def assign_customers_to_satellites_cw(self, random_gen):
        """
        Assign each customer to one satellite by distance-biased roulette wheel.

        Inverse-distance weighting is an implementation choice; the referenced
        2E-VRP paper does not specify its exact probability formula.
        """
        if not self.solution.satellites:
            raise ValueError("Clarke--Wright requires at least one satellite.")

        assignments = {
            satellite_ID: []
            for satellite_ID in self.solution.satellites
        }

        for customer_ID in self.solution.customers:
            satellite_IDs = list(self.solution.satellites)

            distances = [
                self.solution.problem.distance_second["satellite-customer"][
                    (s, customer_ID)
                ]
                for s in satellite_IDs
            ]

            # A coincident satellite gets all the probability mass.
            zero_distance = [
                s
                for s, distance in zip(satellite_IDs, distances)
                if distance == 0
            ]

            if zero_distance:
                selected = random_gen.choice(zero_distance)
            else:
                weights = [1.0 / distance for distance in distances]
                threshold = random_gen.random() * sum(weights)
                selected = satellite_IDs[-1]

                for satellite_ID, weight in zip(satellite_IDs, weights):
                    threshold -= weight

                    if threshold <= 0:
                        selected = satellite_ID
                        break

            assignments[selected].append(customer_ID)

        return assignments

    def clarke_wright_second(self, assignments, random_gen):
        """
        Build second-echelon routes with endpoint-feasible savings merges.
        """
        self.solution.routes_second = []

        capacity = self.solution.problem.capacity_second
        distances = self.solution.problem.distance_second

        if capacity <= 0:
            raise ValueError("Second-echelon capacity must be positive.")

        for satellite_ID, customer_IDs in assignments.items():
            # Keep route sequences as IDs until all merges are finished.
            routes = [[customer_ID] for customer_ID in customer_IDs]

            for customer_ID in customer_IDs:
                if self.solution.customers[customer_ID]["demand"] > capacity:
                    raise ValueError(
                        f"Customer {customer_ID} exceeds second-echelon capacity."
                    )

            # Calculate Clarke--Wright savings for customer pairs.
            savings = []

            for index, i in enumerate(customer_IDs):
                for j in customer_IDs[index + 1:]:
                    saving = (
                        distances["satellite-customer"][(satellite_ID, i)]
                        + distances["satellite-customer"][(satellite_ID, j)]
                        - distances["customer-customer"][(i, j)]
                    )

                    savings.append((saving, i, j))

            # Randomize equal-saving ties, then sort by decreasing savings.
            random_gen.shuffle(savings)
            savings.sort(key=lambda item: item[0], reverse=True)

            # Merge routes whenever the saving is feasible.
            for saving, i, j in savings:
                if saving < 0:
                    continue

                route_i = next(
                    (r for r in routes if i in r),
                    None
                )

                route_j = next(
                    (r for r in routes if j in r),
                    None
                )

                if route_i is None or route_j is None or route_i is route_j:
                    continue

                # Both customers must be route endpoints.
                if i not in (route_i[0], route_i[-1]):
                    continue

                if j not in (route_j[0], route_j[-1]):
                    continue

                # Check the second-echelon capacity constraint.
                combined_load = sum(
                    self.solution.customers[k]["demand"]
                    for k in route_i + route_j
                )

                if combined_load > capacity:
                    continue

                # Orient routes so that i is last and j is first.
                oriented_i = route_i if route_i[-1] == i else route_i[::-1]
                oriented_j = route_j if route_j[0] == j else route_j[::-1]

                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(oriented_i + oriented_j)

            # Convert route sequences into Route objects.
            satellite_loc = self.solution.satellites[satellite_ID]["location"]

            for customer_route in routes:
                locations = (
                    [satellite_loc]
                    + [
                        self.solution.customers[i]["location"]
                        for i in customer_route
                    ]
                    + [satellite_loc]
                )

                demand = {
                    i: self.solution.customers[i]["demand"]
                    for i in customer_route
                }

                route = Route(
                    self.solution.problem,
                    locations,
                    demand,
                    False
                )

                route.customers = set(customer_route)
                self.solution.routes_second.append(route)

    def clarke_wright_first(self, random_gen):
        """
        Allocate full truckloads, then merge residual first-echelon routes.

        The nearest-depot assignment is an explicit multi-depot adaptation.
        """
        self.solution.routes_first = []

        capacity = self.solution.problem.capacity_first
        distances = self.solution.problem.distance_first

        if capacity <= 0:
            raise ValueError("First-echelon capacity must be positive.")

        if not self.solution.depots:
            raise ValueError("Clarke--Wright requires at least one depot.")

        # Aggregate demand from actual second-echelon routes.
        satellite_demands = {
            s: 0
            for s in self.solution.satellites
        }

        satellite_by_location = {
            tuple(data["location"]): s
            for s, data in self.solution.satellites.items()
        }

        for route in self.solution.routes_second:
            satellite_ID = satellite_by_location[
                tuple(route.locations[0])
            ]

            satellite_demands[satellite_ID] += sum(
                route.served_load.values()
            )

        # Residual route sequences and their loads, grouped by depot.
        residual_by_depot = {
            d: []
            for d in self.solution.depots
        }

        for satellite_ID, demand in satellite_demands.items():
            if demand <= 0:
                continue

            # Assign the satellite to its nearest depot.
            depot_ID = min(
                self.solution.depots,
                key=lambda d: distances["depot-satellite"][
                    (d, satellite_ID)
                ]
            )

            depot_loc = self.solution.depots[depot_ID]["location"]
            satellite_loc = self.solution.satellites[satellite_ID]["location"]

            # Original Clarke--Wright oversized-demand preprocessing.
            full_count, residual = divmod(demand, capacity)

            # Allocate full-capacity first-echelon routes.
            for _ in range(int(full_count)):
                self.solution.routes_first.append(
                    Route(
                        self.solution.problem,
                        [depot_loc, satellite_loc, depot_loc],
                        {satellite_ID: capacity},
                        True
                    )
                )

            # Retain the residual demand for savings-based merging.
            if residual > 0:
                residual_by_depot[depot_ID].append(
                    ([satellite_ID], {satellite_ID: residual})
                )

        # Apply Clarke--Wright independently for each depot.
        for depot_ID, routes in residual_by_depot.items():
            satellites = [
                sequence[0]
                for sequence, _ in routes
            ]

            savings = []

            for index, s in enumerate(satellites):
                for t in satellites[index + 1:]:
                    saving = (
                        distances["depot-satellite"][(depot_ID, s)]
                        + distances["depot-satellite"][(depot_ID, t)]
                        - distances["satellite-satellite"][(s, t)]
                    )

                    savings.append((saving, s, t))

            # Randomize equal-saving ties, then sort by decreasing savings.
            random_gen.shuffle(savings)
            savings.sort(key=lambda item: item[0], reverse=True)

            # Merge feasible first-echelon routes.
            for saving, s, t in savings:
                if saving < 0:
                    continue

                route_s = next(
                    (r for r in routes if s in r[0]),
                    None
                )

                route_t = next(
                    (r for r in routes if t in r[0]),
                    None
                )

                if route_s is None or route_t is None or route_s is route_t:
                    continue

                seq_s, load_s = route_s
                seq_t, load_t = route_t

                # Both satellites must be route endpoints.
                if s not in (seq_s[0], seq_s[-1]):
                    continue

                if t not in (seq_t[0], seq_t[-1]):
                    continue

                # Check the first-echelon capacity constraint.
                combined_load = (
                    sum(load_s.values()) + sum(load_t.values())
                )

                if combined_load > capacity:
                    continue

                # Orient routes so that s is last and t is first.
                oriented_s = seq_s if seq_s[-1] == s else seq_s[::-1]
                oriented_t = seq_t if seq_t[0] == t else seq_t[::-1]

                routes.remove(route_s)
                routes.remove(route_t)

                routes.append(
                    (
                        oriented_s + oriented_t,
                        {**load_s, **load_t}
                    )
                )

            # Convert merged sequences into Route objects.
            depot_loc = self.solution.depots[depot_ID]["location"]

            for sequence, demand in routes:
                locations = (
                    [depot_loc]
                    + [
                        self.solution.satellites[s]["location"]
                        for s in sequence
                    ]
                    + [depot_loc]
                )

                self.solution.routes_first.append(
                    Route(
                        self.solution.problem,
                        locations,
                        demand,
                        True
                    )
                )
