"""
Classes and methods for representing and manipulating vehicle routes.

Original template by Yasmine Beck.
"""


class Route:
    """
    Represents a vehicle route for either the first or second echelon.

    Attributes
    ----------
    problem : object
        Problem instance used to compute distances and costs.
    locations : list
        Sequence of locations visited by the vehicle. The first and last
        locations are the depot for a first-echelon route or the satellite
        for a second-echelon route.
    served_load : dict
        Load served at each intermediate location in the route.
    is_first_echelon : Boolean
        Indicates whether the route belongs to the first echelon.
    distance : float
        Total distance of the route.
    cost : float
        Total cost of the route, including distance, vehicle, and, for
        first-echelon routes, handling costs.
    """

    def __init__(self, problem, locations, load, is_first_echelon):
        self.problem = problem
        self.satellite_locations = [satellite["location"] for satellite in problem.satellites.values()]
        self.customer_locations = [customer["location"] for customer in problem.customers.values()]
        self.locations = locations
        self.served_load = load
        self.is_first_echelon = is_first_echelon
        
        # Compute the cost of the route.
        self.cost = self.compute_cost()

    def compute_distance(self):
        """
        Compute the total distance of the route.
        """
        # Compute the distance from the start location to the first
        # intermediate location and from the last intermediate location
        # back to the start location.        
        if self.is_first_echelon:
            depot_ID = self.get_location_id(self.locations[0], self.problem.depots)[0]
            first_satellite_ID = self.get_location_id(self.locations[1], self.problem.satellites)[0]
            last_satellite_ID = self.get_location_id(self.locations[-2], self.problem.satellites)[0]
            distance = (
                self.problem.distance_first["depot-satellite"][depot_ID, first_satellite_ID]
                + self.problem.distance_first["depot-satellite"][depot_ID, last_satellite_ID]
            )
        else:
            satellite_ID = self.get_location_id(self.locations[0], self.problem.satellites)[0]
            first_customer_ID = self.get_location_id(self.locations[1], self.problem.customers)[0]
            last_customer_ID = self.get_location_id(self.locations[-2], self.problem.customers)[0]
            distance = (
                self.problem.distance_second["satellite-customer"][satellite_ID, first_customer_ID]
                + self.problem.distance_second["satellite-customer"][satellite_ID, last_customer_ID]
            )

        # Add the distances between consecutive intermediate locations.
        for idx in range(1, len(self.locations)):
            previous = self.locations[idx - 1]
            current = self.locations[idx]
            if self.is_first_echelon:
                if previous in self.satellite_locations and current in self.satellite_locations:
                    previous_ID = self.get_location_id(previous, self.problem.satellites)[0]
                    current_ID = self.get_location_id(current, self.problem.satellites)[0]
                    distance += self.problem.distance_first["satellite-satellite"][previous_ID, current_ID]
            else:
                if previous in self.customer_locations and current in self.customer_locations:
                    previous_ID = self.get_location_id(previous, self.problem.customers)[0]
                    current_ID = self.get_location_id(current, self.problem.customers)[0]
                    distance += self.problem.distance_second["customer-customer"][previous_ID, current_ID]
        return distance

    def remove_customer(self, customer):
        """
        Remove a customer and its served load from the route.
        """
        # Determine the customer ID.
        customer_IDs = self.get_location_id(customer, self.problem.customers)
        customer_ID = next(customer_ID for customer_ID in customer_IDs if customer_ID in self.served_load)
        
        # Determine the demand served at the customer.
        demand = self.served_load[customer_ID]
        
        # Remove the customer and its served load from the route.
        for position in range(1, len(self.locations) - 1):
            if self.locations[position] == customer:
                self.locations.pop(position)
                break
        del self.served_load[customer_ID]
        
        # Update the costs of the route.
        self.cost = self.compute_cost()
        
    def get_location_id(self, location, location_dict):
        """
        Determine the ID of a given depot, satellite, or customer location.
        """
        location_ID = [key for key, val in location_dict.items() if val["location"] == location]
        return location_ID

    def compute_cost(self):    
        """
        Compute the total cost of the route.

        First-echelon routes include distance, vehicle, and handling costs.
        Second-echelon routes include distance and vehicle costs.
        """
        if len(self.locations) < 3:
            # No customer/satellite served.
            return 0

        self.distance = self.compute_distance()
        cost = self.distance
        if self.is_first_echelon:
            cost += self.problem.cost_first + self.problem.cost_handling*sum(self.served_load.values())
        else:
            cost += self.problem.cost_second        
        return cost
