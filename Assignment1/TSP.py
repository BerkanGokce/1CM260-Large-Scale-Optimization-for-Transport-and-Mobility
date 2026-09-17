# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math 
import os
import time


class Point2D: 
    """Class for representing a point in 2D space"""
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
    
    #method that computes the rounded euclidian distance between two 2D points
    def getDistance(c1,c2): 
        dx = c1.x-c2.x
        dy = c1.y-c2.y
        return math.sqrt(dx**2+dy**2)

class TSP:
    """
    Class for representing a Traveling Salesman Problem
    
    Attributes
    ----------
    nCities : int
        the number of cities
    cities : list of ints
        the cities, all represented by integers
    distMatrix : 2D array
        matrix with all distances between cities. Distance between city i and city j is distMatrix[i-1][j]
    
    """
    def __init__(self,tspFileName):
        """
        Reads a .tsp file and constructs an instance. 
        We assume that it is an Euclidian TSP

        Parameters
        ----------
        tspFileName : str
            name of the file
        """
        points = list() #add all points to list
        f = open(tspFileName)
        for line in f.readlines()[6:-1]: #start reading from line 7, skip last line
            asList = line.split()
            floatList = list(map(float,asList))

            id = int(floatList[0])-1 #convert to int, subtract 1 because Python indices start from 0
            x = floatList[1]
            y = floatList[2]

            c = Point2D(id,x,y)
            points.append(c)
        f.close()
        
        #print("Read in all points, start computing distance matrix")
        
        self.nCities = len(points)
        self.cities = list(range(self.nCities))
        
        #compute distance matrix, assume Euclidian TSP
        self.distMatrix = np.zeros((self.nCities,self.nCities)) #init as nxn matrix
        for i in range(self.nCities):
            for j in range(i+1,self.nCities):
                distItoJ = Point2D.getDistance(points[i], points[j])
                self.distMatrix[i,j] = distItoJ
                self.distMatrix[j,i] = distItoJ
        
        #print("Finished computing distance matrix")


    def getTour_NN(self,start):
        """
        Performs the nearest neighbour algorithm

        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied.

        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)
    
        #print("Start computing NN tour")
        for i in range(self.nCities-1):
            curCity = tour[i]
            closestDist = -1 #initialize with -1
            closestCity = None #initialize with None

            #find closest city not yet in tour
            for j in notInTour:
                dist = self.distMatrix[curCity][j]
                if dist<closestDist or closestCity is None:
                    #update the closest city and distance
                    closestDist = dist
                    closestCity = j
            
            tour.append(closestCity)
            notInTour.remove(closestCity)
        
        #print("Finished computing NN tour")

        return tour
    
    def getCitiesCopy(self): 
        return self.cities.copy()
        
    def evaluateSolution(self,tour):
        if self.isFeasible(tour):
            costs = self.computeCosts(tour)
            print("The solution is feasible with costs "+str(costs))
        else: 
            print("The solution is infeasible")
    
    def isFeasible(self,tour):
        """
        Checks if tour is feasible

        Parameters
        ----------
        tour : list of integers
            order in which cities are visited. For a 4-city TSP, an example tour is [3, 1, 4, 2]

        Returns
        -------
        bool
            TRUE if feasible, FALSE if infeasible.

        """
        #first check if the length of the tour is correct
        if len(tour)!=self.nCities:
            print("Length of tour incorrect")
            return False
        else: 
            #check if all cities in the tour
            for city in self.cities:
                if city not in tour:
                    return False
        return True
    
    def computeCosts(self,tour):
        """
        Computes the costs of a tour

        Parameters
        ----------
        tour : list of integers
            order of cities.

        Returns
        -------
        costs : int
            costs of tour.

        """
        costs = 0
        for i in range(len(tour)-1):
            costs += self.distMatrix[tour[i],tour[i+1]]
            
        # add the costs to complete the tour back to the start
        costs += self.distMatrix[tour[-1],tour[0]]
        return costs
    
    ##############################################################################   
    ### Part 3 - Method to execute the expensive insertion algorithm

    def getTour_ExpensiveInsertion(self, start):
        """
        Performs the expensive insertion heuristic.

        Parameters
        ----------
        start : int
            Starting point of the tour.

        Returns
        -------
        tour : list of ints
            Order in which the cities are visited.
        """

        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)

        # Find the city farthest from the starting city
        farthestCity = None
        farthestDist = -1

        for city in notInTour:
            dist = self.distMatrix[start][city]

            if dist > farthestDist:
                farthestDist = dist
                farthestCity = city

        # Add the farthest city to the initial tour
        tour.append(farthestCity)
        notInTour.remove(farthestCity)

        # Continue until all cities have been added to the tour
        while notInTour:

            # Step 1: Select the city whose minimum distance to a city
            # already in the tour is the largest
            selectedCity = None
            largestMinDist = -1

            for city in notInTour:

                minDist = float("inf")

                # Find the minimum distance from this city
                # to any city already in the tour
                for tourCity in tour:
                    dist = self.distMatrix[city][tourCity]

                    if dist < minDist:
                        minDist = dist

                # Keep the city with the largest minimum distance
                if minDist > largestMinDist:
                    largestMinDist = minDist
                    selectedCity = city

            # Step 2: Find the insertion position that causes the
            # smallest increase in total tour length
            bestPosition = None
            smallestIncrease = float("inf")

            for i in range(len(tour)):

                city1 = tour[i]

                # % handles the edge from the last city back to the first
                city2 = tour[(i + 1) % len(tour)]

                increase = (
                    self.distMatrix[city1][selectedCity]
                    + self.distMatrix[selectedCity][city2]
                    - self.distMatrix[city1][city2]
                )

                if increase < smallestIncrease:
                    smallestIncrease = increase
                    bestPosition = i + 1

            # Insert the selected city at the best position
            tour.insert(bestPosition, selectedCity)
            notInTour.remove(selectedCity)

        return tour
    
    ##############################################################################  
    ### Part 4 - Method to execute the GRASPed expensive insertion algorithm

    def getTour_GRASPedInsertion(self, start, alpha, iterations):
        """
        Performs GRASPed expensive insertion multiple times and
        returns the best tour found.

        The city-selection step is randomized using a Restricted
        Candidate List (RCL), while the insertion-position step
        remains greedy.

        Parameters
        ----------
        start : int
            Starting point of the tour.

        alpha : float
            Tolerance used to construct the RCL.
            Must be between 0 and 1.
            For example, alpha = 0.10 allows candidates whose score
            is within 10% of the best score.

        iterations : int
            Number of randomized GRASP constructions to perform.

        Returns
        -------
        bestTour : list of ints
            Best tour found over all iterations.
        """

        # Check parameters
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1.")

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")

        bestTour = None
        bestCost = float("inf")

        # Run the randomized construction multiple times
        for _ in range(iterations):

            tour = [start]
            notInTour = self.cities.copy()
            notInTour.remove(start)

            # Initial step:
            # Find the city farthest from the starting city
            farthestCity = None
            farthestDist = -1

            for city in notInTour:
                dist = self.distMatrix[start][city]

                if dist > farthestDist:
                    farthestDist = dist
                    farthestCity = city

            # Add the farthest city to the initial tour
            tour.append(farthestCity)
            notInTour.remove(farthestCity)

            # Continue until all cities have been added
            while notInTour:

                # Step 1: Calculate the minimum distance to the current
                # tour for every city not yet in the tour
                candidateScores = {}

                for city in notInTour:

                    minDist = float("inf")

                    for tourCity in tour:
                        dist = self.distMatrix[city][tourCity]

                        if dist < minDist:
                            minDist = dist

                    candidateScores[city] = minDist

                # Step 2: Construct the Restricted Candidate List (RCL)
                bestScore = max(candidateScores.values())

                threshold = (1 - alpha) * bestScore

                RCL = [
                    city
                    for city, score in candidateScores.items()
                    if score >= threshold
                ]

                # Randomly select one city from the RCL
                selectedCity = int(np.random.choice(RCL))

                # Step 3: Greedily find the insertion position that
                # causes the smallest increase in tour length
                bestPosition = None
                smallestIncrease = float("inf")

                for i in range(len(tour)):

                    city1 = tour[i]
                    city2 = tour[(i + 1) % len(tour)]

                    increase = (
                        self.distMatrix[city1][selectedCity]
                        + self.distMatrix[selectedCity][city2]
                        - self.distMatrix[city1][city2]
                    )

                    if increase < smallestIncrease:
                        smallestIncrease = increase
                        bestPosition = i + 1

                # Insert selected city into its cheapest position
                tour.insert(bestPosition, selectedCity)
                notInTour.remove(selectedCity)

            # Evaluate this GRASP construction
            cost = self.computeCosts(tour)

            # Keep it if it is the best solution found so far
            if cost < bestCost:
                bestCost = cost
                bestTour = tour.copy()

        return bestTour
    
    ##############################################################################  
    ### Part 5 - Methods to check whether a given tour is 2-optimal and to make a given tour 2-optimal

    def isTwoOpt(self, tour):
        """
        Checks whether a given tour is 2-optimal.

        A tour is 2-optimal if no pair of non-adjacent edges can be
        replaced by a 2-exchange that decreases the total tour length.

        Parameters
        ----------
        tour : list of ints
            Order in which the cities are visited.

        Returns
        -------
        bool
            True if the tour is 2-optimal, False otherwise.
        """

        n = len(tour)

        # Consider the first edge (a, b)
        for i in range(n):

            a = tour[i]
            b = tour[(i + 1) % n]

            # Consider the second edge (c, d)
            for j in range(i + 2, n):

                c = tour[j]
                d = tour[(j + 1) % n]

                # Skip adjacent edges:
                # when i = 0 and j = n-1, the two edges share tour[0]
                if i == 0 and j == n - 1:
                    continue

                # Cost of the two current edges
                currentCost = (
                    self.distMatrix[a][b]
                    + self.distMatrix[c][d]
                )

                # Cost after the 2-exchange
                newCost = (
                    self.distMatrix[a][c]
                    + self.distMatrix[b][d]
                )

                # If an improving 2-exchange exists,
                # the tour is not 2-optimal
                if newCost < currentCost:
                    return False

        # No improving 2-exchange was found
        return True

    def makeTwoOpt(self, tour):
        """
        Applies 2-opt exchanges using a best-improvement strategy
        until the tour is 2-optimal.

        Parameters
        ----------
        tour : list of ints
            Order in which the cities are visited.

        Returns
        -------
        tour : list of ints
            A 2-optimal tour.
        """

        # Work on a copy so the original tour is not modified
        tour = tour.copy()
        n = len(tour)

        # Continue until no improving 2-exchange exists
        while not self.isTwoOpt(tour):

            bestImprovement = 0
            bestI = None
            bestJ = None

            # Consider first edge (a, b)
            for i in range(n):

                a = tour[i]
                b = tour[(i + 1) % n]

                # Consider second edge (c, d)
                for j in range(i + 2, n):

                    # Skip adjacent first and last edges
                    if i == 0 and j == n - 1:
                        continue

                    c = tour[j]
                    d = tour[(j + 1) % n]

                    # Cost of current edges
                    currentCost = (
                        self.distMatrix[a][b]
                        + self.distMatrix[c][d]
                    )

                    # Cost after the 2-exchange
                    newCost = (
                        self.distMatrix[a][c]
                        + self.distMatrix[b][d]
                    )

                    # Change in cost caused by the exchange
                    improvement = newCost - currentCost

                    # Keep the best improving exchange found
                    if improvement < bestImprovement:
                        bestImprovement = improvement
                        bestI = i
                        bestJ = j

            # Apply the best 2-exchange found
            if bestI is not None:
                tour[bestI + 1:bestJ + 1] = reversed(
                    tour[bestI + 1:bestJ + 1]
                )
        return tour

    ##############################################################################  
    ### Part 6 - Method to apply 2-exchanges to a tour until no improving 2-exchanges remains using AI

    def makeTwoOpt_AI(self, tour):
        """
        Improve a given TSP tour using the 2-opt local search heuristic.

        The 2-opt heuristic repeatedly considers pairs of edges in the current
        tour. If removing two edges and reconnecting the tour in the alternative
        way decreases the total tour length, the corresponding section of the
        tour is reversed.

        The procedure continues until no improving 2-opt exchange can be found.

        Parameters
        ----------
        tour : list of int
            Current TSP tour, where each integer represents a city.

        Returns
        -------
        list of int
            A locally improved tour for which no improving 2-opt exchange remains.
        """

        # Make a copy so that the original tour is not modified
        tour = tour.copy()
        n = len(tour)

        # Continue searching until no improving 2-opt move can be found
        while True:
            improved = False

            # Select the first edge (a, b)
            for i in range(n):

                # Select the second edge (c, d).
                # Starting at i + 2 ensures that the two selected edges
                # are not adjacent.
                for j in range(i + 2, n):

                    # When i = 0 and j = n - 1, the two edges are also
                    # adjacent because the tour is closed.
                    if i == 0 and j == n - 1:
                        continue

                    # Current edges:
                    # (a, b) = (tour[i], tour[i+1])
                    # (c, d) = (tour[j], tour[j+1])
                    a = tour[i]
                    b = tour[(i + 1) % n]
                    c = tour[j]
                    d = tour[(j + 1) % n]

                    # Cost of the two edges currently in the tour
                    old_cost = (
                        self.distMatrix[a, b]
                        + self.distMatrix[c, d]
                    )

                    # Cost after the proposed 2-opt exchange:
                    # (a,b) and (c,d) are replaced by (a,c) and (b,d)
                    new_cost = (
                        self.distMatrix[a, c]
                        + self.distMatrix[b, d]
                    )

                    # Perform the exchange only if it improves the tour
                    if new_cost < old_cost:

                        # Reversing this section reconnects the tour using
                        # the new edges (a,c) and (b,d)
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])

                        improved = True
                        break

                # Restart the search from the beginning after an improvement
                if improved:
                    break

            # If a complete search found no improvement, the tour is
            # 2-opt locally optimal
            if not improved:
                break
        return tour

    ##############################################################################
    ### Part 7 - GRASPed insertion combined with 2-opt local search

    def getTour_GRASPedInsertion_2Opt(self, start, alpha, iterations):
        """
        Performs GRASPed expensive insertion combined with 2-opt local search.

        In each iteration, one randomized GRASP tour is constructed and
        subsequently improved using 2-opt. The costs before and after local
        search are stored, and the best locally improved tour is returned.

        Parameters
        ----------
        start : int
            Starting point of the tour.

        alpha : float
            Tolerance used to construct the Restricted Candidate List (RCL).

        iterations : int
            Number of GRASP + 2-opt iterations.

        Returns
        -------
        bestTour : list of ints
            Best tour found after 2-opt local search.

        costsBefore : list of floats
            Tour costs after GRASP construction and before local search.

        costsAfter : list of floats
            Tour costs after 2-opt local search.
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1.")

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")

        bestTour = None
        bestCost = float("inf")

        costsBefore = []
        costsAfter = []

        for _ in range(iterations):
            # Generate one randomized GRASP solution
            tour_GRASP = self.getTour_GRASPedInsertion(start, alpha, 10)

            # Store cost before local search
            costBefore = self.computeCosts(tour_GRASP)
            costsBefore.append(costBefore)

            # Apply 2-opt local search
            tour_2opt = self.makeTwoOpt(tour_GRASP)

            # Store cost after local search
            costAfter = self.computeCosts(tour_2opt)
            costsAfter.append(costAfter)

            # Keep the best solution found after local search
            if costAfter < bestCost:
                bestCost = costAfter
                bestTour = tour_2opt.copy()

        return bestTour, costsBefore, costsAfter


##############################################################################   
if __name__ == "__main__":

    ##############################################################################   
    ### Part 1: Select 10 TSP instances
    ############################################################################## 

    # Random seed for reproducibility
    SEED = 42
    np.random.seed(SEED)

    # Add 5 small instances
    small_instances = [
        "Instances/Small/berlin52.tsp",
        "Instances/Small/eil51.tsp",
        "Instances/Small/eil76.tsp",
        "Instances/Small/kroA100.tsp",
        "Instances/Small/kroA200.tsp"
    ]

    # Add 3 medium instances
    medium_instances = [
        "Instances/Medium/a280.tsp",
        "Instances/Medium/ali535.tsp",
        "Instances/Medium/fl417.tsp"
    ]

    # Add 2 large instances
    large_instances = [
        "Instances/Large/rl1304.tsp",
        "Instances/Large/fl1577.tsp"
    ]

    # Combine all selected instances
    instances = small_instances + medium_instances + large_instances

    ##############################################################################   
    ### Part 2 - Nearest Neighbour with different starting positions
    ############################################################################## 
    
    nn_results = {}

    for instFilename in instances:
        instance = TSP(instFilename)

        costs = []

        # Run nearest neighbour from every possible starting city
        for startPointNN in tqdm(instance.cities, desc=f"NN - {os.path.basename(instFilename)}", dynamic_ncols=True):
            tour = instance.getTour_NN(startPointNN)
            cost = instance.computeCosts(tour)
            costs.append(cost)

        # Store starting cities and corresponding costs
        # Find the best starting city
        best_idx = np.argmin(costs)
        best_starting_city = instance.cities[best_idx]
        best_cost = costs[best_idx]

        # Store results
        nn_results[instFilename] = {
            "starting_cities": instance.cities.copy(),
            "costs": costs,
            "best_starting_city": best_starting_city,
            "best_cost": best_cost
        }

    # Plot Part 2 results
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))

    axes = axes.flatten()

    for i, (instFilename, result) in enumerate(nn_results.items()):

        starting_cities = result["starting_cities"]
        costs = result["costs"]
        best_starting_city = result["best_starting_city"]

        color = f"C{i}"

        # Plot all starting cities
        axes[i].scatter(starting_cities, costs, color=color, s=15, label=f"Best starting city = {best_starting_city}")

        instance_name = os.path.basename(instFilename).replace(".tsp", "")

        axes[i].set_title(instance_name)
        axes[i].grid(alpha=0.3)
        axes[i].legend()

    # Common axis labels
    fig.supxlabel("Starting City", fontsize=16)
    fig.supylabel("Tour Cost", fontsize=16)
    fig.suptitle("Nearest Neighbour Tour Cost for Different Starting Cities", fontsize=16)

    plt.tight_layout()
    plt.show()

    ##############################################################################   
    ### Testing the implemented TSP methods on single instange for parts 2, 3, 4 and 5: berlin52.tsp
    ##############################################################################  
    instFilename = "Instances/Small/berlin52.tsp"
    inst = TSP(instFilename)
    startPoint = 39

    ### Part 2: Nearest Neighbour
    tour_NN = inst.getTour_NN(startPoint)
    print("\nNearest Neighbour:")
    print(tour_NN)
    inst.evaluateSolution(tour_NN)

    print()

    ### Part 3: Expensive Insertion
    tour_EI = inst.getTour_ExpensiveInsertion(startPoint)
    print("\nExpensive Insertion:")
    print(tour_EI)
    inst.evaluateSolution(tour_EI)

    print()

    ### Part 4: GRASPed Expensive Insertion
    alpha = 0.10
    iterations = 100
    tour_GRASPEI = inst.getTour_GRASPedInsertion(startPoint, alpha, iterations)
    print(f"\nGRASPed Expensive Insertion ({iterations} iterations):")
    print(tour_GRASPEI)
    inst.evaluateSolution(tour_GRASPEI)

    ### Part 5: 2-Optimal
    print("\nCheck whether a given tour is 2-optimal: ")
    print("NN 2-optimal:", inst.isTwoOpt(tour_NN))
    print("Expensive Insertion 2-optimal:", inst.isTwoOpt(tour_EI))
    print("GRASPed Insertion 2-optimal:", inst.isTwoOpt(tour_GRASPEI))

    print("\nApply 2-opt exchanges until the tour is 2-optimal:")
    # Apply 2-exchanges
    tour_NN_2opt = inst.makeTwoOpt(tour_NN)
    tour_EI_2opt = inst.makeTwoOpt(tour_EI)
    tour_GRASPEI_2opt = inst.makeTwoOpt(tour_GRASPEI)

    # Re-Check 2-optimal
    print("NN 2-optimal:", inst.isTwoOpt(tour_NN_2opt))
    print("Expensive Insertion 2-optimal:", inst.isTwoOpt(tour_EI_2opt))
    print("GRASPed Insertion 2-optimal:", inst.isTwoOpt(tour_GRASPEI_2opt))

    # Check feasibility of 2-optimal solutions
    print("\nCheck Feasibility of 2-optimal solutions: ")
    print("NN 2-optimal:", inst.isFeasible(tour_NN_2opt))
    print("Expensive Insertion 2-optimal:", inst.isFeasible(tour_EI_2opt))
    print("GRASPed Insertion 2-optimal:", inst.isFeasible(tour_GRASPEI_2opt))

    
    ##############################################################################   
    ### Part 6 - Compare makeTwoOpt() and makeTwoOpt_AI()
    ############################################################################## 

    comparison_results = []

    for instFilename in tqdm(instances, desc="Comparing 2-opt methods", dynamic_ncols=True):

        instance = TSP(instFilename)
        instance_name = os.path.basename(instFilename).replace(".tsp", "")

        # Reset seed
        np.random.seed(SEED)

        # Generate initial tour using Nearest Neighbour Heuristic
        initial_tour = instance.getTour_NN(startPoint)
        # Cost of the initial tour
        initial_cost = instance.computeCosts(initial_tour)

        # Test for makeTwoOpt()
        start_time = time.perf_counter()
        tour_makeTwoOpt = instance.makeTwoOpt(initial_tour.copy())
        time_makeTwoOpt = time.perf_counter() - start_time

        cost_makeTwoOpt = instance.computeCosts(tour_makeTwoOpt)
        makeTwoOpt_bool = instance.isTwoOpt(tour_makeTwoOpt)

        # Test for makeTwoOpt_AI()
        start_time = time.perf_counter()
        tour_AI = instance.makeTwoOpt_AI(initial_tour.copy())
        time_AI = time.perf_counter() - start_time

        cost_AI = instance.computeCosts(tour_AI)
        AI_TwoOpt_bool = instance.isTwoOpt(tour_AI)

        comparison_results.append({
            "instance": instance_name,
            "initial_cost": initial_cost,
            "cost_makeTwoOpt": cost_makeTwoOpt,
            "cost_AI": cost_AI,
            "makeTwoOpt_bool": makeTwoOpt_bool,
            "AI_TwoOpt_bool": AI_TwoOpt_bool,
            "time_makeTwoOpt": time_makeTwoOpt,
            "time_AI": time_AI
        })

    print("\nPart 6 - Comparison of 2-opt implementations\n")

    header = (
        f"{'Instance':<{12}}"
        f"{'Initial Cost':^{14}}"
        f"{'makeTwoOpt Cost':^{18}}"
        f"{'makeTwoOpt_AI Cost':^{20}}"
        f"{'makeTwoOpt 2-opt?':^{20}}"
        f"{'AI 2-opt?':^{14}}"
        f"{'makeTwoOpt Time':^{18}}"
        f"{'makeTwoOpt_AI Time':^{20}}"
    )

    print(header)
    print("-" * len(header))

    for result in comparison_results:
        print(
            f"{result['instance']:<{12}}"
            f"{result['initial_cost']:^{14}.2f}"
            f"{result['cost_makeTwoOpt']:^{18}.2f}"
            f"{result['cost_AI']:^{20}.2f}"
            f"{str(result['makeTwoOpt_bool']):^{20}}"
            f"{str(result['AI_TwoOpt_bool']):^{14}}"
            f"{result['time_makeTwoOpt']:^{18}.4f}"
            f"{result['time_AI']:^{20}.4f}"
        )

    print()

    ##############################################################################
    ### Part 7 - GRASPed insertion combined with 2-opt local search
    ##############################################################################

    # Select instance
    instFilename = "Instances/Small/berlin52.tsp"
    instance_name = os.path.basename(instFilename).replace(".tsp", "")
    inst = TSP(instFilename)

    # Parameters
    alpha = 0.10
    iterations = 500

    # Reset random seed for reproducibility
    np.random.seed(SEED)

    # Run GRASPed Expensive Insertion + 2-opt
    bestTour, costsBefore, costsAfter = inst.getTour_GRASPedInsertion_2Opt(startPoint, alpha, iterations)

    # Cost of the best final tour
    bestCost = inst.computeCosts(bestTour)

    # Print results
    print("\nPart 7 - GRASPed Insertion + 2-opt")
    print(f"Instance: {instance_name}")
    print(f"Number of iterations: {iterations}")
    print(f"Best tour cost: {bestCost:.2f}")
    print(f"Best tour is 2-optimal: {inst.isTwoOpt(bestTour)}")
    print(f"Best tour is feasible: {inst.isFeasible(bestTour)}")

    # Find best solutions before and after local search
    # Iteration containing the lowest GRASP cost before local search
    best_before_idx = np.argmin(costsBefore)
    # Iteration containing the lowest cost after 2-opt local search
    best_after_idx = np.argmin(costsAfter)

    print(f"Best cost before local search: {costsBefore[best_before_idx]:.2f}")
    print(f"Best cost after local search: {costsAfter[best_after_idx]:.2f}")

    # Plot results
    plt.figure(figsize=(10, 7))

    # Plot all GRASP + 2-opt iterations
    plt.scatter(costsBefore, costsAfter, facecolors="none", edgecolors="C0", s=35)

    # Mark the iteration with the best GRASP solution
    # before local search
    plt.scatter(costsBefore[best_before_idx], costsAfter[best_before_idx], facecolors="none", color="red", s=35)

    # Mark the iteration with the best solution
    # after 2-opt local search
    plt.scatter(costsBefore[best_after_idx], costsAfter[best_after_idx], facecolors="none", edgecolors="red", s=35)

    # Add y = x reference line
    min_value = min(min(costsBefore), min(costsAfter))
    max_value = max(max(costsBefore), max(costsAfter))

    plt.plot([min_value, max_value], [min_value, max_value], "--")

    # Figure formatting
    plt.xlabel("Cost Before Local Search", fontsize=16)
    plt.ylabel("Cost After Local Search", fontsize=16)
    plt.title(f"GRASPed Insertion with 2-opt Local Search ({instance_name})", fontsize=16)

    # Increase x- and y-axis tick label sizes
    plt.tick_params(axis="both", labelsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()