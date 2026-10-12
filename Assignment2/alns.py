
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
import math

# Local imports.
from solution import Solution
from constructive import RandomInsertion, ClarkeWrightSavings
from destroy import Destroy
from repair import Repair


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
    destroy_operators : dict
        Available destroy operators.
    repair_operators : dict
        Available repair operators.
    destroy_weights : dict
        Adaptive weights assigned to destroy operators.
    repair_weights : dict
        Adaptive weights assigned to repair operators.
    lambda_decay : float
        Decay parameter controlling the sensitivity of weight updates.
    """

    def __init__(self, problem):
        self.problem = problem
        self.random_gen = random.Random(42)
        self.num_iter = 200
        self.min_size_nbh = 1
        self.max_size_nbh = 45

        self.constructive_method = "random"

        # Optional simulated annealing (lecture: exp(-delta/T), T <- alpha*T).
        self.use_sa = False
        self.sa_initial_temperature = None  # Auto: 5% cost increase accepted with p=0.5
        self.sa_cooling_rate = 0.99
        self.sa_end_temperature = 1e-8
        self.temperature = None
        self.accepted = False
        self.acceptance_probability = 0.0
        self.acceptance_reason = ""

        # Adaptive operator selection parameters
        self.lambda_decay = 0.9

        # Scores assigned according to the outcome of an iteration
        self.omega_1 = 10  # New global best
        self.omega_2 = 5   # Improved current solution
        self.omega_3 = 2   # Accepted without improvement
        self.omega_4 = 0   # Rejected

        # Register the available destroy operators.
        # Additional operators can be added without changing the ALNS loop.
        self.destroy_operators = {
            "destroy1": Destroy.destroy1,
            "destroy2": Destroy.destroy2,
            "destroy3": Destroy.destroy3,
        }

        # Register the available repair operators.
        # Additional operators can be added without changing the ALNS loop.
        self.repair_operators = {
            "repair1": Repair.repair1,
            "repair2": Repair.repair2,
            "repair3": Repair.repair3,
        }

        # Determine the number of available operators automatically.
        self.num_destroy_ops = len(self.destroy_operators)
        self.num_repair_ops = len(self.repair_operators)

        # Initialize all destroy and repair operator weights to 1.
        self.destroy_weights = {
            name: 1.0 for name in self.destroy_operators
        }

        self.repair_weights = {
            name: 1.0 for name in self.repair_operators
        }

    def execute(self):
        """
        Main method that executes the ALNS.
        """
        start_time = time.time()

        # Construct an initial solution.
        self.construct_init_solution()

        if self.use_sa:
            if not 0 < self.sa_cooling_rate < 1:
                raise ValueError("sa_cooling_rate must be between 0 and 1.")
            if self.sa_initial_temperature is None:
                self.temperature = max(
                    1e-8, 0.05 * self.current_solution.cost / math.log(2)
                )
            else:
                if self.sa_initial_temperature <= 0:
                    raise ValueError("sa_initial_temperature must be positive.")
                self.temperature = float(self.sa_initial_temperature)
            if self.sa_end_temperature <= 0:
                raise ValueError("sa_end_temperature must be positive.")
        else:
            self.temperature = None

        # Repeatedly destroy and repair the current solution.
        for iteration in range(self.num_iter):
            # Copy the current solution.
            self.temp_solution = copy.deepcopy(self.current_solution)

            # Decide on the size of the neighborhood.
            size_nbh = self.random_gen.randint(
                self.min_size_nbh,
                min(self.max_size_nbh, len(self.problem.customers))
            )

            # Execute the destroy and repair operations.
            self.destroy(size_nbh)
            self.repair()

            # Validate the repaired solution.
            self.temp_solution.validate_solution()

            # Compute the cost of the newly found solution.
            self.temp_solution.compute_cost()
            print(f"Iteration {iteration}: found solution with cost {self.temp_solution.cost}")

            # Determine whether the new solution is accepted.
            self.check_if_accept()

            # Cool after acceptance, as in the lecture notes.
            if self.use_sa:
                self.temperature *= self.sa_cooling_rate

            # Update the weights for choosing the destroy and repair operators.
            self.update_weights()

            # Display the selected operators and their assigned score.
            print(
                f"Destroy: {self.selected_destroy} | "
                f"Repair: {self.selected_repair} | "
                f"Score: {self.score}"
            )

        runtime = time.time() - start_time
        print(f"Terminated. Final cost: {self.best_cost}, Runtime: {runtime:.3f} seconds")

        # Display the final adaptive weights of all operators.
        print("\nFinal destroy operator weights:")
        for name, weight in self.destroy_weights.items():
            print(f"  {name}: {weight:.4f}")

        print("\nFinal repair operator weights:")
        for name, weight in self.repair_weights.items():
            print(f"  {name}: {weight:.4f}")

    def construct_init_solution(self):
        """
        Construct an initial solution using the Random Insertion
        constructive heuristic.
        """

        # Initialize the solution.
        self.current_solution = Solution(self.problem)

        if self.constructive_method == "random":
            RandomInsertion(self.current_solution).construct(self.random_gen)

        elif self.constructive_method == "savings":
            ClarkeWrightSavings(self.current_solution).construct(self.random_gen)

        else:
            raise ValueError(f"Unknown constructive method: {self.constructive_method}")

        # Validate the constructed solution.
        self.current_solution.validate_solution()

        # Compute the cost of the current solution.
        self.current_solution.compute_cost()

        # Initialize the best solution found so far and its associated cost.
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_solution.cost

        print(f"Constructed initial solution with cost: {self.best_cost}")

    def select_operator(self, operators, weights):
        """
        Select an operator using roulette-wheel probabilities
        proportional to its adaptive weight.
        """

        # Retrieve the available operators and their current weights.
        names = list(operators.keys())
        probabilities = [weights[name] for name in names]

        # Use uniform selection if all operator weights are zero.
        if sum(probabilities) == 0:
            probabilities = [1.0] * len(names)

        # Select an operator using roulette-wheel selection.
        selected = self.random_gen.choices(
            names,
            weights=probabilities,
            k=1
        )[0]

        return selected

    def destroy(self, size_nbh):
        """
        Perform the destroy operations.
        Select a destroy operator based on its adaptive weight.
        """

        # Select a destroy operator using roulette-wheel selection.
        self.selected_destroy = self.select_operator(
            self.destroy_operators,
            self.destroy_weights
        )

        # Execute the selected destroy operator.
        self.removed_customers = self.destroy_operators[self.selected_destroy](
            self.temp_solution,
            size_nbh,
            self.random_gen
        )

    def repair(self):
        """
        Perform the repair operations.
        Select a repair operator based on its adaptive weight.
        """

        # Select a repair operator using roulette-wheel selection.
        self.selected_repair = self.select_operator(
            self.repair_operators,
            self.repair_weights
        )

        # Execute the selected repair operator.
        self.repair_operators[self.selected_repair](
            self.temp_solution,
            self.removed_customers,
            self.random_gen
        )

    def check_if_accept(self):
        """Accept improvements, and optionally accept worse candidates via SA."""
        candidate_cost = self.temp_solution.cost
        current_cost = self.current_solution.cost
        self.accepted = False
        self.acceptance_probability = 0.0
        self.acceptance_reason = "rejected"

        if candidate_cost < self.best_cost:
            self.best_cost = candidate_cost
            self.best_solution = copy.deepcopy(self.temp_solution)
            self.current_solution = copy.deepcopy(self.temp_solution)
            self.score = self.omega_1
            self.accepted = True
            self.acceptance_probability = 1.0
            self.acceptance_reason = "global_best"
            print("Found new global best solution.")
        elif candidate_cost < current_cost:
            self.current_solution = copy.deepcopy(self.temp_solution)
            self.score = self.omega_2
            self.accepted = True
            self.acceptance_probability = 1.0
            self.acceptance_reason = "improved_current"
        elif self.use_sa and self.temperature > self.sa_end_temperature:
            delta = candidate_cost - current_cost
            self.acceptance_probability = math.exp(-delta / self.temperature)
            if self.random_gen.random() < self.acceptance_probability:
                self.current_solution = copy.deepcopy(self.temp_solution)
                self.score = self.omega_3
                self.accepted = True
                self.acceptance_reason = "sa_accepted"
            else:
                self.score = self.omega_4
        else:
            self.score = self.omega_4

    def update_weights(self):
        """
        Update adaptive destroy and repair operator weights
        using exponential smoothing.
        """

        lam = self.lambda_decay
        score = self.score

        # Update selected destroy operator
        self.destroy_weights[self.selected_destroy] = lam * self.destroy_weights[self.selected_destroy] + (1 - lam) * score

        # Update selected repair operator
        self.repair_weights[self.selected_repair] = lam * self.repair_weights[self.selected_repair] + (1 - lam) * score
        
