"""
Adaptive large neighborhood search (ALNS) algorithm.

This module implements the ALNS framework for constructing and improving
solutions through destroy and repair operations.

Original template by Yasmine Beck.
"""

# Global imports.
import random
import copy
import time

# Local imports.
from solution import Solution


class ALNS:
    """
    Implements an adaptive large neighborhood search (ALNS) algorithm.

    The algorithm constructs an initial solution and iteratively improves it
    by destroying and repairing the current solution. It maintains the
    current and best solutions.

    Attributes
    ----------
    problem : object
        Problem instance to be solved.
    num_destroy_ops : int
        Number of available destroy operators.
    num_repair_ops : int
        Number of available repair operators.
    random_gen : object
        Random number generator used by the ALNS.
    num_iter : int
        Number of ALNS iterations.
    min_size_nbh : int
        Minimum number of elements removed during a destroy operation.
    max_size_nbh : int
        Maximum number of elements removed during a destroy operation.
    current_solution : object
        Current solution accepted by the ALNS.
    temp_solution : object
        Temporary solution generated during the current iteration.
    best_solution : object
        Best solution found by the ALNS.
    best_cost : float
        Cost of the best solution found.
    """
    def __init__(self, problem, num_destroy_ops, num_repair_ops):
        self.problem = problem
        self.num_destroy_ops = num_destroy_ops
        self.num_repair_ops = num_repair_ops
        self.random_gen = random.Random(42)
        self.num_iter = 200
        self.min_size_nbh = 1
        self.max_size_nbh = 45
        
    def execute(self):
        """
        Main method that executes the ALNS.
        """
        start_time = time.time()
        
        # Construct an initial solution.
        self.construct_init_solution()

        # Repeatedly destroy and repair the current solution.
        for iteration in range(self.num_iter):
            # Copy the current solution.
            self.temp_solution = copy.deepcopy(self.current_solution)
            
            # Decide on the size of the neighborhood.
            size_nbh = self.random_gen.randint(self.min_size_nbh, self.max_size_nbh)
            
            # Execute the destroy and repair operations.
            self.destroy(size_nbh)
            self.repair()
            
            # Compute the cost of the newly found solution.
            self.temp_solution.compute_cost()
            print(f"Iteration {iteration}: found solution with cost {self.temp_solution.cost}")
            
            # Determine whether the new solution is accepted.
            self.check_if_accept()
            
            # Update the weights for choosing the destroy and repair operators.
            self.update_weights()
            
        runtime = time.time() - start_time
        print(f"Terminated. Final cost: {self.best_cost}, Runtime: {runtime:.3f} seconds")

    def construct_init_solution(self):
        """
        Construct an initial solution using random insertion.
        """
        # Initialize the solution.
        self.current_solution = Solution(self.problem)
        
        # Generate the first- and second-echelon routes using random insertion.
        self.current_solution.do_random_insertion(self.random_gen)
        
        # Compute the cost of the current solution.
        self.current_solution.compute_cost()

        # Initialize the best solution found so far and its associated cost.
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_solution.cost
        print(f"Constructed initial solution with cost: {self.best_cost}")
        
    def destroy(self, size_nbh):
        """
        Perform the destroy operations.
        Can be extended with weights and additional destroy operators.
        """
        self.temp_solution.do_random_removal(size_nbh, self.random_gen)
    
    def repair(self):
        """
        Perform the repair operations.
        Can be extended with weights and additional destroy operators.
        """
        self.temp_solution.do_random_insertion(self.random_gen)

    def check_if_accept(self):
        """
        Check whether the newly found solution is accepted.
        """
        # Accept if a global best solution is found.
        if self.temp_solution.cost < self.best_cost:
            # Update the solution and its cost.
            self.best_cost = self.temp_solution.cost
            self.best_solution = copy.deepcopy(self.temp_solution)
            self.current_solution = copy.deepcopy(self.temp_solution)
            print("Found new global best solution.")
        
        # Accept if a better solution than the current one is found.
        if self.temp_solution.cost < self.current_solution.cost:
            self.current_solution = copy.deepcopy(self.temp_solution)
            
    def update_weights(self):
        """
        Update the weights for the destroy and repair operators.
        """
        pass
    
