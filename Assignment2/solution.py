"""
Methods for constructing and modifying solutions using ALNS.

Original template by Yasmine Beck.
"""

# Global imports.
import copy
import math

# Local imports.
from route import Route


class Solution:
    """
    Represents a solution to a two-echelon vehicle routing problem.

    Attributes
    ----------
    problem : object
        Problem instance corresponding to the solution.
    depots : dict
        Dictionary containing the depots and their locations.
    satellites : dict
        Dictionary containing the satellites, their locations, and their demands.
    customers : dict
        Dictionary containing the customers, their locations, and their demands.
    routes_first : list
        List of routes used by first-echelon vehicles.
    routes_second : list
        List of routes used by second-echelon vehicles.
    cost : float
        Total cost of the solution, including distance, vehicle, and handling
        costs.
    """
    def __init__(self, problem):
        self.problem = problem
        self.depots = problem.depots
        self.satellites = problem.satellites
        self.customers = problem.customers
        
    def do_random_insertion(self, random_gen):
        """
        Construct first- and second-echelon routes using random insertion.

        Customers are first randomly inserted into second-echelon routes.
        The resulting satellite demands are then used to construct the
        first-echelon routes.
        """
        # Construct the secon-echelon routes.
        self.do_random_insertion_second(random_gen)
        
        # Construct the first-echelon routes.
        self.do_random_insertion_first(random_gen)

    def do_random_insertion_first(self, random_gen):
        """
        Construct first-echelon routes by randomly inserting satellites.
        """
        # Initialize the list of first-echelon routes.
        self.routes_first = []
        
        # Initialize the lists of served and unserved satellites.
        not_served = list(self.satellites)
        served = []

        # Determine the total demand associated with each satellite from
        # the second-echelon routes.
        demand_not_served = {satellite_ID: 0 for satellite_ID in self.satellites}
        for route in self.routes_second:
            satellite = route.locations[0]
            satellite_ID = next(
                key for key, value in self.satellites.items()
                if value["location"] == satellite
            )
            total_demand = sum(route.served_load.values())
            demand_not_served[satellite_ID] += total_demand
            
        # Assign satellite demand until all satellites are served.
        satellites = copy.deepcopy(self.satellites)
        for satellite in satellites.values():
            satellite["demand"] = 0
        while not_served:
            # Select a random satellite and determine its remaining demand.
            satellite_ID = random_gen.choice(not_served)
            load_max = demand_not_served[satellite_ID]

            # Consider the existing routes as potential insertion routes.
            routes = self.routes_first.copy()
            inserted = False
            while routes:
                # Select a random route.
                route = random_gen.choice(routes)

                # Compute the remaining capacity for the selected route.
                remaining_cap = self.problem.capacity_first - sum(route.served_load.values())
                if remaining_cap <= 0:
                    break

                # Assign the maximum load that can be added to the route.
                if load_max > remaining_cap:
                     load = remaining_cap
                else:
                     load = load_max
                satellites[satellite_ID]["demand"] = load

                # Try to insert the satellite at its best feasible position.
                after_insertion, feasible_insertion = self.greedy_insertion(
                    route, satellites, satellite_ID, True
                )
                
                # Update the route if a feasible insertion was found.
                if feasible_insertion:
                    inserted = True
                    self.routes_first.remove(route)
                    self.routes_first.append(after_insertion)
                    break
                else:
                    routes.remove(route)
                    
            # If no existing route can accommodate the load, create a new route.
            if not inserted:
                # Select a depot to start and end the new route.
                depot_ID = random_gen.choice(list(self.depots))
                depot_loc = self.depots[depot_ID]["location"]
                locations = [depot_loc, self.satellites[satellite_ID]["location"], depot_loc]
                
                # Assign the maximum load that can be served by a new vehicle.
                if load_max > self.problem.capacity_first:
                     load = self.problem.capacity_first
                else:
                     load = load_max
                satellites[satellite_ID]["demand"] = load

                # Create and add the new first-echelon route.
                demand = {satellite_ID: satellites[satellite_ID]["demand"]}
                new_route = Route(self.problem, locations, demand, True)
                self.routes_first.append(new_route)
                
            # Update the remaining demand and the list of unserved satellites.
            demand_not_served[satellite_ID] -= load
            self.satellites = satellites
            if demand_not_served[satellite_ID] <= 0:
                served.append(satellite_ID)
                not_served.remove(satellite_ID)
                
    def do_random_insertion_second(self, random_gen):
        """
        Construct second-echelon routes by randomly inserting customers.
        """
        # Initialize the list of second-echelon routes.
        self.routes_second = []

        # Initialize the lists of served and unserved customers.
        not_served = list(self.customers)
        served = []
        
        # Insert customers until all customers are served.
        while not_served:
            # Select a random customer.
            customer_ID = random_gen.choice(not_served)
            
            # Consider the existing routes as potential insertion routes.
            routes = self.routes_second.copy()
            inserted = False
            while routes:
                # Select a random route.
                route = random_gen.choice(routes)

                # Try to insert the customer at its best feasible position.
                after_insertion, feasible_insertion = self.greedy_insertion(
                    route, self.customers, customer_ID, False
                )

                # Update the route if a feasible insertion was found.
                if feasible_insertion:
                    inserted = True
                    self.routes_second.remove(route)
                    self.routes_second.append(after_insertion)
                    break
                else:
                    routes.remove(route)
                    
            # If no existing route can accommodate the customer, create a new route.
            if not inserted:
                # Select a satellite to start and end the new route.
                satellite_ID = random_gen.choice(list(self.satellites))
                satellite_loc = self.satellites[satellite_ID]["location"]
                locations = [satellite_loc, self.customers[customer_ID]["location"], satellite_loc]

                # Create and add the new second-echelon route.
                demand = {customer_ID: self.customers[customer_ID]["demand"]}
                new_route = Route(self.problem, locations, demand, False)
                new_route.customers = {customer_ID}
                self.routes_second.append(new_route)
                
            # Mark the customer as served.
            served.append(customer_ID)
            not_served.remove(customer_ID)

    def greedy_insertion(self, route, locations, location_ID, is_first_echelon):
        """
        Insert a location at the position that results in the shortest route.

        Returns the resulting route and a Boolean indicating whether the
        location can be inserted without exceeding vehicle capacity.
        """
        # Check whether the location can be served by the selected route.
        load = locations[location_ID]["demand"]
        served_load = sum(route.served_load.values())
        if is_first_echelon:
            capacity = self.problem.capacity_first
        else:
            capacity = self.problem.capacity_second        
        if served_load + load > capacity:
            # The vehicle does not have sufficient capacity.
            return route, False
                
        # Evaluate all possible insertion positions.
        location = locations[location_ID]["location"]
        positions = range(1, len(route.locations))
        min_distance = math.inf
        for position in positions:
            demand = route.served_load.copy()
            route_locations = route.locations.copy()
            
            # Insert the location and its load at the current position.
            route_locations.insert(position, location)
            demand[location_ID] = load
            after_insertion = Route(self.problem, route_locations, demand, is_first_echelon)
            
            # Keep the insertion with the shortest resulting route.
            if after_insertion.distance < min_distance:
                best_insert = after_insertion
                min_distance = after_insertion.distance
        return best_insert, True
    
    def do_random_removal(self, remove_cnt, random_gen):
        """
        Randomly remove a given number of customers from the solution.
        """
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
                customer = random_gen.choice(route.locations[1:-1])
                route.remove_customer(customer)
                removed += 1
            else:
                # Exclude routes that contain no customers.
                routes.remove(route)
                
    def compute_cost(self):
        """
        Compute the total cost of the solution from all vehicle routes.
        """
        self.cost = sum(route.cost for route in self.routes_first + self.routes_second)
