"""
Data structures and methods for representing and preparing 2E-CVRP instances.

Original template by Yasmine Beck.
"""

# Global imports.
import math


class TWO_E_CVRP:
    """
    Represents an instance of the two-echelon capacitated vehicle routing problem (2E-CVRP).

    The class reads an instance from a file, stores the locations and demands
    of customers, satellites, and depots, and computes the distances required
    for the first and second echelons.
    
    Attributes
    ----------
    num_depots : int
        Number of depots in the instance.
    num_satellites : int
        Number of satellites in the instance.
    num_customers : int
        Number of customers in the instance.
    depots : dict
        Dictionary containing the depots and their locations.
    satellites : dict
        Dictionary containing the satellites and their locations.
    customers : dict
        Dictionary containing the customers, their locations, and their demands.
    distance_first : dict
        Distances used for routing in the first echelon.
    distance_second : dict
        Distances used for routing in the second echelon.
    capacity_first : int
        Capacity of first-echelon vehicles.
    cost_first : int
        Cost per first-echelon vehicle.
    capacity_second : int
        Capacity of second-echelon vehicles.
    cost_second : int
        Cost per second-echelon vehicle.
    cost_handling : int
        Handling cost per unit of demand transshipped at a satellite.
    """
    
    def __init__(self, path_to_file):
        self.parse_instance(path_to_file)
        self.compute_distance_matrix()
        
        # Set the vehicle capacities and costs for both echelons and the handling cost at satellites.
        # Note: These values are based on the information provided in the assignment.
        self.capacity_first = 200
        self.cost_first = 50
        self.capacity_second = 50
        self.cost_second = 25
        self.cost_handling = 5

    def parse_instance(self, path_to_file):
        """
        Parse the depot, satellite, and customer data of a 2E-CVRP instance from a file.
        """
        # Get the characteristics of the instance from its file name.
        instance_name = path_to_file.rsplit("/", 1)[-1][:-4]
        self.num_depots = int(instance_name[4])
        self.num_satellites = int(instance_name[6])
        self.num_customers = int(instance_name[8:])

        # Initialize dictionaries for the depots, satellites, and customers.
        self.depots = {}
        self.satellites = {}
        self.customers = {}

        # Read and parse the instance data.
        with open(f"{path_to_file}") as data_file:
            lines = [[int(value) for value in line.split()] for line in data_file.readlines()]
            for line_cnt, data in enumerate(lines):
                if line_cnt < self.num_customers:
                    # Parse customer data.
                    customer_ID = line_cnt + 1
                    self.customers[customer_ID] = {
                        "location": (data[0], data[1]),
                        "demand": data[4]
                    }
                elif line_cnt < self.num_customers + self.num_satellites:
                    # Parse satellite data.
                    satellite_ID = line_cnt - self.num_customers + 1
                    self.satellites[satellite_ID] = {"location": (data[0], data[1])}
                else:
                    # Parse depot data.
                    depot_ID = line_cnt - self.num_customers - self.num_satellites + 1
                    self.depots[depot_ID] = {"location": (data[0], data[1])}

    def compute_distance_matrix(self):
        """
        Compute the distances between depots and satellites (first echelon)
        and between satellites and customers (second echelon).
        """
        # Initialize dictionaries for the first- and second-echelon distances.
        self.distance_first = {
            "depot-satellite": {},
            "satellite-satellite": {},
        }
        self.distance_second = {
            "satellite-customer": {},
            "customer-customer": {},
        }

        # Compute all first- and second-echelon distances.
        for satellite_ID, satellite in self.satellites.items():
            for depot_ID, depot in self.depots.items():
                distance = self.get_distance(depot["location"], satellite["location"])
                self.distance_first["depot-satellite"][depot_ID, satellite_ID] = distance
            for customer_ID, customer in self.customers.items():
                distance = self.get_distance(customer["location"], satellite["location"])
                self.distance_second["satellite-customer"][satellite_ID, customer_ID] = distance
            for other_satellite_ID, other_satellite in self.satellites.items():
                if satellite_ID == other_satellite_ID:
                    distance = 0
                else:
                    distance = self.get_distance(satellite["location"], other_satellite["location"])
                self.distance_first["satellite-satellite"][satellite_ID, other_satellite_ID] = distance
                
        for customer_ID, customer in self.customers.items():
            for other_customer_ID, other_customer in self.customers.items():
                if customer_ID == other_customer_ID:
                    distance = 0
                else:
                    distance = self.get_distance(customer["location"], other_customer["location"])
                self.distance_second["customer-customer"][customer_ID, other_customer_ID] = distance
            
    def get_distance(self, location1, location2):
        """
        Compute the Euclidean distance between two locations.
        """
        dx = location1[0] - location2[0]
        dy = location1[1] - location2[1]
        return math.sqrt(dx**2 + dy**2)
