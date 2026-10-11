"""
Main file for running the ALNS algorithm.

This script reads a 2E-CVRP instance, initializes the ALNS algorithm,
executes it using the specified number of destroy and repair operators,
and prints the resulting best solution.

Original template by Yasmine Beck.
"""

# Local imports.
from problem import TWO_E_CVRP
from alns import ALNS

# Read the test instance.
path_to_file = "instances/Ca1-2,3,15.txt"
instance = TWO_E_CVRP(path_to_file)

# Build and execute the ALNS.
algo = ALNS(instance)
algo.execute()

# Print the best solution.
print("\nFirst-echelon routes:")
for route in algo.best_solution.routes_first:
    print(f"  Locations: {route.locations}, Served load: {sum(route.served_load.values())}")

print("\nSecond-echelon routes:")
for route in algo.best_solution.routes_second:
    print(f"  Locations: {route.locations}, Served load: {sum(route.served_load.values())}")
