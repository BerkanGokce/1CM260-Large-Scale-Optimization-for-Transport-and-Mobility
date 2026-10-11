
"""
Classes and methods for representing, modifying, evaluating, and validating
solutions to the two-echelon capacitated vehicle routing problem (2E-CVRP).

This module provides the shared solution representation used by the
constructive heuristics and the Adaptive Large Neighborhood Search (ALNS).

Original template by Yasmine Beck.
"""

import math
from collections import Counter
from route import Route


class Solution:
    """
    Represents a complete or partial solution to the 2E-CVRP.

    A solution consists of first-echelon routes transporting goods from
    depots to satellites and second-echelon routes transporting goods
    from satellites to customers.

    This class stores the solution and provides shared operations for
    route modification, cost calculation, and feasibility validation.

    Attributes
    ----------
    problem : object
        Problem instance containing locations, demands, capacities,
        vehicle costs, and distance matrices.
    depots : dict
        Dictionary containing the depots and their locations.
    satellites : dict
        Dictionary containing the satellites and their locations.
    customers : dict
        Dictionary containing the customers, their locations, and demands.
    routes_first : list
        List of first-echelon Route objects.
    routes_second : list
        List of second-echelon Route objects.
    cost : float
        Total solution cost, including route distances, vehicle costs,
        and satellite handling costs.
    """

    def __init__(self, problem):
        """
        Initialize a solution associated with the given problem instance.

        Parameters
        ----------
        problem : TWO_E_CVRP
            Problem instance to be solved.
        """

        # Store the problem instance and its location data.
        self.problem = problem
        self.depots = problem.depots
        self.satellites = problem.satellites
        self.customers = problem.customers

        # Initialize route collections and solution cost.
        self.routes_first = []
        self.routes_second = []
        self.cost = 0

    def greedy_insertion(self, route, locations, location_ID, is_first_echelon):
        """
        Find the best feasible insertion position for a location in a route.

        The method evaluates every possible insertion position and selects
        the one that produces the shortest resulting route, provided that
        the vehicle capacity constraint is satisfied.

        The original route is not modified.

        Parameters
        ----------
        route : Route
            Existing route into which the location may be inserted.
        locations : dict
            Dictionary of customers or satellites containing location
            coordinates and demands.
        location_ID : int
            ID of the customer or satellite to be inserted.
        is_first_echelon : bool
            True for first-echelon routes, False for second-echelon routes.

        Returns
        -------
        best_insert : Route
            Route resulting from the best feasible insertion.
            Returns the original route if insertion is infeasible.
        feasible_insertion : bool
            True if insertion is feasible, False otherwise.
        """

        # Prevent inserting the same satellite twice into a first-echelon route.
        if is_first_echelon and location_ID in route.served_load:
            return route, False

        # Determine the demand of the location to be inserted.
        load = locations[location_ID]["demand"]

        # Calculate the total load currently served by the route.
        served_load = sum(route.served_load.values())

        # Select the vehicle capacity according to the echelon.
        if is_first_echelon:
            capacity = self.problem.capacity_first
        else:
            capacity = self.problem.capacity_second

        # Check whether the additional load would exceed vehicle capacity.
        if served_load + load > capacity:
            # The vehicle does not have sufficient capacity.
            return route, False

        # Retrieve the coordinates of the location to be inserted.
        location = locations[location_ID]["location"]

        # Evaluate all insertion positions between the route endpoints.
        positions = range(1, len(route.locations))
        min_distance = math.inf

        for position in positions:

            # Copy the existing route data to avoid modifying the original.
            demand = route.served_load.copy()
            route_locations = route.locations.copy()

            # Insert the location and its load at the current position.
            route_locations.insert(position, location)
            demand[location_ID] = load

            # Construct the candidate route and calculate its cost/distance.
            after_insertion = Route(
                self.problem,
                route_locations,
                demand,
                is_first_echelon
            )

            # Keep the insertion with the shortest resulting route.
            if after_insertion.distance < min_distance:
                best_insert = after_insertion
                min_distance = after_insertion.distance

        return best_insert, True

    def do_random_removal(self, remove_cnt, random_gen):
        """
        Randomly remove a specified number of customers from the solution.

        Customers are removed from randomly selected second-echelon routes.
        The method stops when the requested number of customers has been
        removed or no further customers are available.

        This method modifies the existing routes directly. It does not
        reconstruct first-echelon routes or return removed customer IDs.

        Parameters
        ----------
        remove_cnt : int
            Maximum number of customers to remove.
        random_gen : random.Random
            Random number generator used for route and customer selection.
        """

        # Create a list of candidate second-echelon routes.
        routes = self.routes_second.copy()
        removed = 0

        # Remove customers until the requested number has been removed
        # or no more customers can be removed.
        while removed < remove_cnt:

            # Select a random route from the remaining candidate routes.
            if routes:
                route = random_gen.choice(routes)
            else:
                # No routes with removable customers remain.
                break

            # Remove a random customer if the route contains one.
            if len(route.locations) > 2:

                # Select a customer, excluding the satellite endpoints.
                customer = random_gen.choice(route.locations[1:-1])

                # Remove the customer and update the route cost.
                route.remove_customer(customer)
                removed += 1

            else:
                # Exclude routes that contain no customers.
                routes.remove(route)

    def compute_cost(self):
        """
        Compute the total cost of the solution from all vehicle routes.

        The cost includes first- and second-echelon route distances,
        fixed vehicle costs, and satellite handling costs.

        Individual route costs are calculated by the Route class.
        """

        # Sum the costs of all first- and second-echelon routes.
        self.cost = sum(
            route.cost
            for route in self.routes_first + self.routes_second
        )

    def validate_solution(self):
        """
        Validate the feasibility of a complete 2E-CVRP solution.

        The following constraints are checked:
        1. Every customer is served exactly once.
        2. Each customer receives its required demand.
        3. Second-echelon vehicle capacities are respected.
        4. First-echelon vehicle capacities are respected.
        5. The supply delivered to each satellite equals the demand
           served by its second-echelon routes.

        Raises
        ------
        ValueError
            If any of the above feasibility constraints are violated.
        """

        # --------------------------------------------------
        # 1. Customer coverage and second-echelon capacities.
        # --------------------------------------------------

        # Initialize the service count of every customer to zero.
        customer_counts = {
            customer_ID: 0
            for customer_ID in self.customers
        }

        # Examine every second-echelon route.
        for route in self.routes_second:

            # Ensure the route starts and ends at the same valid satellite.
            if route.is_first_echelon or len(route.locations) < 3:
                raise ValueError("Invalid second-echelon route structure.")
            if route.locations[0] != route.locations[-1] or tuple(route.locations[0]) not in {
                tuple(data["location"]) for data in self.satellites.values()
            }:
                raise ValueError("Second-echelon route must start and end at a satellite.")

            # Served customer IDs must agree with the intermediate stops.
            expected_stops = Counter(
                tuple(self.customers[customer_ID]["location"])
                for customer_ID in route.served_load
                if customer_ID in self.customers
            )
            actual_stops = Counter(tuple(stop) for stop in route.locations[1:-1])
            if actual_stops != expected_stops:
                raise ValueError("Second-echelon route stops do not match served customers.")

            # Check the second-echelon vehicle capacity.
            if sum(route.served_load.values()) > self.problem.capacity_second:
                raise ValueError(
                    "Second-echelon route exceeds vehicle capacity."
                )

            # Verify each customer and its delivered demand.
            for customer_ID, load in route.served_load.items():

                # Check whether the customer exists in the problem.
                if customer_ID not in customer_counts:
                    raise ValueError(
                        f"Unknown customer {customer_ID} in second-echelon route."
                    )

                # Check whether the delivered load matches customer demand.
                if load != self.customers[customer_ID]["demand"]:
                    raise ValueError(
                        f"Incorrect delivered demand for customer {customer_ID}."
                    )

                # Record that the customer is served by this route.
                customer_counts[customer_ID] += 1

        # Verify that every customer is served exactly once.
        for customer_ID, count in customer_counts.items():
            if count != 1:
                raise ValueError(
                    f"Customer {customer_ID} is served {count} times."
                )

        # --------------------------------------------------
        # 2. Calculate demand leaving each satellite.
        # --------------------------------------------------

        # Initialize the total outgoing demand of every satellite.
        satellite_demand = {
            satellite_ID: 0
            for satellite_ID in self.satellites
        }

        # Create a lookup from satellite coordinates to satellite IDs.
        satellite_by_location = {
            tuple(data["location"]): satellite_ID
            for satellite_ID, data in self.satellites.items()
        }

        # Aggregate the demand served by each satellite's routes.
        for route in self.routes_second:

            # Identify the satellite where the route starts.
            satellite_ID = satellite_by_location[
                tuple(route.locations[0])
            ]

            # Add the route's delivered customer demand.
            satellite_demand[satellite_ID] += sum(
                route.served_load.values()
            )

        # --------------------------------------------------
        # 3. First-echelon capacities and satellite supply.
        # --------------------------------------------------

        # Initialize the total incoming supply of every satellite.
        satellite_supply = {
            satellite_ID: 0
            for satellite_ID in self.satellites
        }

        # Examine every first-echelon route.
        for route in self.routes_first:

            # Ensure the route starts and ends at the same valid depot.
            if not route.is_first_echelon or len(route.locations) < 3:
                raise ValueError("Invalid first-echelon route structure.")
            if route.locations[0] != route.locations[-1] or tuple(route.locations[0]) not in {
                tuple(data["location"]) for data in self.depots.values()
            }:
                raise ValueError("First-echelon route must start and end at a depot.")

            # Served satellite IDs must agree with the intermediate stops.
            expected_stops = Counter(
                tuple(self.satellites[satellite_ID]["location"])
                for satellite_ID in route.served_load
                if satellite_ID in self.satellites
            )
            actual_stops = Counter(tuple(stop) for stop in route.locations[1:-1])
            if actual_stops != expected_stops:
                raise ValueError("First-echelon route stops do not match served satellites.")

            # Check the first-echelon vehicle capacity.
            if sum(route.served_load.values()) > self.problem.capacity_first:
                raise ValueError(
                    "First-echelon route exceeds vehicle capacity."
                )

            # Aggregate the goods delivered to each satellite.
            for satellite_ID, load in route.served_load.items():

                # Check whether the satellite exists in the problem.
                if satellite_ID not in satellite_supply:
                    raise ValueError(
                        f"Unknown satellite {satellite_ID} in first-echelon route."
                    )

                satellite_supply[satellite_ID] += load

        # --------------------------------------------------
        # 4. Check satellite supply-demand balance.
        # --------------------------------------------------

        # Every satellite must receive exactly the amount of goods
        # required by its second-echelon customer routes.
        for satellite_ID in self.satellites:

            if not math.isclose(
                satellite_supply[satellite_ID],
                satellite_demand[satellite_ID],
                rel_tol=1e-9,
                abs_tol=1e-9
            ):
                raise ValueError(
                    f"Supply-demand mismatch at satellite {satellite_ID}: "
                    f"supply={satellite_supply[satellite_ID]}, "
                    f"demand={satellite_demand[satellite_ID]}"
                )
