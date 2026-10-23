# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np


class UFL_Problem:
    """
    Class that represent a problem instance of the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    f : numpy array
        the yearly fixed operational costs of all facilities
    c : numpy 2-D array (matrix)
        the yearly transportation cost of delivering all demand from markets to facilities
    n_markets : int
        number of all markets.
    n_facilities : int
        number of all available locations.
    """

    def __init__(self, f, c, n_markets, n_facilities):

        self.f = f
        self.c = c
        self.n_markets = n_markets
        self.n_facilities = n_facilities

    def __str__(self):
        return f" Uncapacitated Facility Location Problem: {self.n_markets} markets, {self.n_facilities} facilities"

    def readInstance(fileName):
        """
        Read the instance fileName

        Parameters
        ----------
        fileName : str
            instance name in the folder Instance.

        Returns
        -------
        UFL Object

        """
        # Read filename
        f = open(f"Instances/{fileName}")
        n_line = 0
        n_markets = 0
        n_facilities = 0
        n_row = 0
        for line in f.readlines():
            asList = line.replace(" ", "_").split("_")
            if line:
                if n_line == 0:
                    n_markets = int(asList[0])
                    n_facilities = int(asList[1])
                    f_j = np.empty(n_facilities)
                    c_ij = np.empty((n_markets, n_facilities))
                elif n_line <= n_facilities:  # For facilities
                    index = n_line - 1
                    f_j[index] = asList[1]
                else:
                    if len(asList) == 1:
                        n_row += 1
                        demand_i = float(asList[0])
                        n_column = 0
                    else:
                        for i in range(len(asList) - 1):
                            c_ij[n_row - 1, n_column] = (
                                demand_i * float(asList[i])
                            )
                            n_column += 1
            n_line += 1
        f.close()
        return UFL_Problem(f_j, c_ij, n_markets, n_facilities) 


class UFL_Solution: 
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    y : numpy array
        binary array indicating whether facilities are open
    x : numpy 2-D array (matrix)
        fraction of demand from markets sourced from facilities
    instance: UFL_Problem
        the problem instance
    """ 
    
    def __init__(self, y, x, instance):
        self.y = y
        self.x = x
        self.instance = instance

    def isFeasible(self): 
        """
        Method that checks whether the solution is feasible
        
        Returns true if feasible, false otherwise
        """
        # Check dimensions
        if self.y.shape != (self.instance.n_facilities,):
            return False

        if self.x.shape != (self.instance.n_markets, self.instance.n_facilities):
            return False

        # Check binary facility decisions
        if not np.all((self.y == 0) | (self.y == 1)):
            return False

        # Check assignment bounds: 0 <= x_ij <= y_j
        if np.any(self.x < -1e-9) or np.any(self.x > self.y + 1e-9):
            return False

        # Check that each customer is fully supplied
        if not np.allclose(np.sum(self.x, axis=1), 1):
            return False

        return True

    
    def getCosts(self): 
        """
        Method that computes and returns the costs of the solution
        """
        transportation_cost = np.sum(self.instance.c * self.x)
        opening_cost = np.sum(self.instance.f * self.y)

        return float(transportation_cost + opening_cost)
    
class LagrangianHeuristic: 
    """
    Class used for the Lagrangian Heuristic
        
    Attributes
    ----------
    instance : UFL_Problem
        the problem instance
    """
    
    def __init__(self,instance):
        self.instance = instance

        # Parameters for the Polyak subgradient update
        self.beta = 1.5
        self.bestUB = float("inf")
        
    def computeTheta(self,labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the optimal value of the Lagrangian problem
        """

        # Adjust transportation costs using customer multipliers
        adjusted_costs = self.instance.c - labda[:, np.newaxis]

        # Only negative adjusted costs improve the relaxed objective
        negative_costs = np.minimum(adjusted_costs, 0)

        # Cost contribution if each facility is opened
        opening_contributions = self.instance.f + np.sum(
            negative_costs, axis=0
        )

        # Compare opening each facility against keeping it closed
        facility_contributions = np.minimum(opening_contributions, 0)

        # Lagrangian lower bound
        theta = np.sum(labda) + np.sum(facility_contributions)

        return float(theta)

    
    def computeLagrangianSolution(self,labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the Lagrangian solution (as a UFL_Solution)
        """

        # Adjust transportation costs using customer multipliers
        adjusted_costs = self.instance.c - labda[:, np.newaxis]

        # Calculate the cost contribution if each facility opens
        opening_contributions = self.instance.f + np.sum(
            np.minimum(adjusted_costs, 0), axis=0
        )

        # Open facilities only when their contribution is negative
        y = (opening_contributions < 0).astype(int)

        # Assign customers where adjusted costs are negative
        # and the corresponding facility is open
        x = ((adjusted_costs < 0) & (y[np.newaxis, :] == 1)).astype(int)

        # Return the Lagrangian solution
        return UFL_Solution(y, x, self.instance)

    
    def convertToFeasibleSolution(self,lagr_solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """

        # Copy facility decisions from the Lagrangian solution
        y = lagr_solution.y.copy()

        # If no facility is open, select the cheapest facility
        # considering both opening and transportation costs
        if np.sum(y) == 0:
            total_costs = self.instance.f + np.sum(
                self.instance.c, axis=0
            )

            cheapest_facility = np.argmin(total_costs)
            y[cheapest_facility] = 1

        # Identify all open facilities
        open_facilities = np.where(y == 1)[0]

        # Find the cheapest open facility for each customer
        cheapest_assignments = open_facilities[
            np.argmin(self.instance.c[:, open_facilities], axis=1)
        ]

        # Construct the assignment matrix
        x = np.zeros(
            (self.instance.n_markets, self.instance.n_facilities),
            dtype=int
        )

        x[np.arange(self.instance.n_markets), cheapest_assignments] = 1

        # Return the repaired feasible solution
        return UFL_Solution(y, x, self.instance)

    
    def updateMultipliers(self, labda_old, lagr_solution):
        """
        Updates Lagrangian multipliers using Polyak-type
        subgradient ascent and returns a new array of Lagrangian multipliers
        """

        # Calculate assignment violations for each customer
        g = 1 - np.sum(lagr_solution.x, axis=1)

        # Squared norm of the subgradient
        g_norm_squared = np.sum(g ** 2)

        # Avoid division by zero
        if g_norm_squared == 0:
            return labda_old.copy()

        # Calculate the current Lagrangian lower bound
        currentLB = self.computeTheta(labda_old)

        # Obtain a feasible upper bound if none exists yet
        if not np.isfinite(self.bestUB):
            feasible_solution = self.convertToFeasibleSolution(
                lagr_solution
            )
            self.bestUB = feasible_solution.getCosts()

        # Calculate the Polyak step size
        step_size = (
            self.beta * max(0.0, self.bestUB - currentLB)
            / g_norm_squared
        )

        # Update multipliers
        labda_new = labda_old + step_size * g

        return labda_new

    
    def runHeuristic(self, max_iterations=1000, stagnation_limit=20, initial_beta=1.5):
        """Run the Lagrangian heuristic and record iteration diagnostics."""

        labda = np.min(self.instance.c, axis=1).astype(float)

        self.beta = initial_beta
        self.bestUB = float("inf")

        bestLB = float("-inf")
        best_solution = None

        stagnation_counter = 0
        history = []
        stop_reason = "Maximum iterations reached"

        for k in range(1, max_iterations + 1):

            # Compute current Lagrangian lower bound
            currentLB = self.computeTheta(labda)

            if currentLB > bestLB + 1e-9:
                bestLB = currentLB
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Solve the Lagrangian relaxation
            lagr_solution = self.computeLagrangianSolution(labda)

            # Construct a feasible solution
            feasible_solution = self.convertToFeasibleSolution(lagr_solution)

            if not feasible_solution.isFeasible():
                raise RuntimeError("The repaired solution is infeasible")

            currentUB = feasible_solution.getCosts()

            if currentUB < self.bestUB:
                self.bestUB = currentUB
                best_solution = feasible_solution

            if bestLB > self.bestUB + 1e-7:
                raise RuntimeError("Lower bound exceeds upper bound")

            # Calculate relative optimality gap
            gap = (
                (self.bestUB - bestLB)
                / max(1.0, abs(self.bestUB))
            )

            # Calculate subgradient and its norm
            g = 1 - np.sum(lagr_solution.x, axis=1)

            g_norm_squared = float(np.sum(g ** 2))
            g_norm = float(np.sqrt(g_norm_squared))

            # Check termination conditions
            gap_reached = gap <= 1e-6
            zero_subgradient = g_norm_squared == 0

            # Reduce beta after prolonged stagnation
            if (
                not gap_reached
                and not zero_subgradient
                and stagnation_counter >= stagnation_limit
            ):
                self.beta *= 0.5
                stagnation_counter = 0

            # Calculate Polyak step size for diagnostics
            if gap_reached or zero_subgradient:
                step_size = 0.0
            else:
                step_size = (
                    self.beta
                    * max(0.0, self.bestUB - currentLB)
                    / g_norm_squared
                )

            # Record iteration results and multiplier diagnostics
            history.append({
                "iteration": k,
                "current_LB": currentLB,
                "current_UB": currentUB,
                "best_LB": bestLB,
                "best_UB": self.bestUB,
                "gap_percent": gap * 100,
                "beta": self.beta,
                "subgradient_norm": g_norm,
                "step_size": step_size,
                "multiplier_mean": float(np.mean(labda)),
                "multiplier_std": float(np.std(labda)),
                "multipliers": labda.copy()
            })

            # Stop if optimality gap is sufficiently small
            if gap_reached:
                stop_reason = "Optimality gap reached"
                break

            # Stop if subgradient is zero
            if zero_subgradient:
                stop_reason = "Zero subgradient"
                break

            # Update Lagrangian multipliers
            labda = self.updateMultipliers(labda, lagr_solution)

        return (bestLB, self.bestUB, best_solution, history, stop_reason)
