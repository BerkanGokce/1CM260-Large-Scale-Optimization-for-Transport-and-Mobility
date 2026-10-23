
import numpy as np
from UFL import UFL_Problem, UFL_Solution, LagrangianHeuristic


def make_test_instance():
    """Create a small UFL instance with a known optimum of 125."""

    f = np.array([60, 45])

    c = np.array([
        [20, 35],
        [15, 25],
        [30, 20]
    ])

    return UFL_Problem(f, c, 3, 2)


def test_part1():
    """Test solution feasibility and objective costs."""

    instance = make_test_instance()

    y1 = np.array([1, 0])
    x1 = np.array([
        [1, 0],
        [1, 0],
        [1, 0]
    ])

    sol1 = UFL_Solution(y1, x1, instance)

    assert sol1.isFeasible()
    assert np.isclose(sol1.getCosts(), 125)
    print("Test 1 passed: Feasible solution, cost = 125")

    x2 = np.array([
        [1, 0],
        [1, 0],
        [0, 0]
    ])

    sol2 = UFL_Solution(y1, x2, instance)

    assert not sol2.isFeasible()
    print("Test 2 passed: Unserved customer detected")

    x3 = np.array([
        [0, 1],
        [1, 0],
        [1, 0]
    ])

    sol3 = UFL_Solution(y1, x3, instance)

    assert not sol3.isFeasible()
    print("Test 3 passed: Closed facility assignment detected")

    y4 = np.array([1, 1])
    x4 = np.array([
        [1, 0],
        [1, 0],
        [0, 1]
    ])

    sol4 = UFL_Solution(y4, x4, instance)

    assert sol4.isFeasible()
    assert np.isclose(sol4.getCosts(), 160)
    print("Test 4 passed: Two open facilities, cost = 160")

    print("\nAll Part 1 tests passed!\n")


def test_part2():
    """Test the Lagrangian lower-bound calculation."""

    instance = make_test_instance()
    heuristic = LagrangianHeuristic(instance)

    labda1 = np.array([50, 30, 40])
    theta1 = heuristic.computeTheta(labda1)

    assert isinstance(theta1, float)
    assert np.isclose(theta1, 120)
    print(f"Test 1 passed: Theta = {theta1}")

    labda2 = np.array([0, 0, 0])
    theta2 = heuristic.computeTheta(labda2)

    assert np.isclose(theta2, 0)
    print(f"Test 2 passed: Theta = {theta2}")

    labda3 = np.array([60, 40, 50])
    theta3 = heuristic.computeTheta(labda3)

    assert np.isclose(theta3, 100)
    print(f"Test 3 passed: Theta = {theta3}")

    assert theta1 <= 125
    assert theta2 <= 125
    assert theta3 <= 125

    print("Test 4 passed: Lower-bound checks")
    print("\nAll Part 2 tests passed!\n")


def test_part3():
    """Test the relaxed Lagrangian solution."""

    instance = make_test_instance()
    heuristic = LagrangianHeuristic(instance)

    labda1 = np.array([50, 30, 40])
    sol1 = heuristic.computeLagrangianSolution(labda1)

    assert np.array_equal(sol1.y, [0, 0])
    assert np.array_equal(sol1.x, np.zeros((3, 2)))

    print("Test 1 passed: Both facilities closed")

    labda2 = np.array([55, 45, 20])
    sol2 = heuristic.computeLagrangianSolution(labda2)

    assert np.array_equal(sol2.y, [1, 0])
    assert np.array_equal(sol2.x, [
        [1, 0],
        [1, 0],
        [0, 0]
    ])

    print("Test 2 passed: Only facility A opens")

    labda3 = np.array([80, 80, 80])
    sol3 = heuristic.computeLagrangianSolution(labda3)

    assert np.array_equal(sol3.y, [1, 1])
    assert np.array_equal(sol3.x, np.ones((3, 2)))

    print("Test 3 passed: Both facilities open")

    for labda in [labda1, labda2, labda3]:

        sol = heuristic.computeLagrangianSolution(labda)

        lagrangian_value = (
            sol.getCosts()
            + np.dot(
                labda,
                1 - np.sum(sol.x, axis=1)
            )
        )

        theta = heuristic.computeTheta(labda)

        assert np.isclose(lagrangian_value, theta)

    print("Test 4 passed: Solution matches computeTheta()")
    print("\nAll Part 3 tests passed!\n")


def test_part4():
    """Test conversion to feasible UFL solutions."""

    instance = make_test_instance()
    heuristic = LagrangianHeuristic(instance)

    labda1 = np.array([50, 30, 40])
    lagr1 = heuristic.computeLagrangianSolution(labda1)
    sol1 = heuristic.convertToFeasibleSolution(lagr1)

    assert sol1.isFeasible()
    assert np.array_equal(sol1.y, [1, 0])
    assert np.array_equal(sol1.x, [
        [1, 0],
        [1, 0],
        [1, 0]
    ])
    assert np.isclose(sol1.getCosts(), 125)

    print("Test 1 passed: All facilities closed, repaired cost = 125")

    labda2 = np.array([50, 40, 20])
    lagr2 = heuristic.computeLagrangianSolution(labda2)
    sol2 = heuristic.convertToFeasibleSolution(lagr2)

    assert sol2.isFeasible()
    assert np.array_equal(sol2.y, [1, 0])
    assert np.array_equal(sol2.x, [
        [1, 0],
        [1, 0],
        [1, 0]
    ])
    assert np.isclose(sol2.getCosts(), 125)

    print("Test 2 passed: One facility open, repaired cost = 125")

    labda3 = np.array([80, 80, 80])
    lagr3 = heuristic.computeLagrangianSolution(labda3)
    sol3 = heuristic.convertToFeasibleSolution(lagr3)

    assert sol3.isFeasible()
    assert np.array_equal(sol3.y, [1, 1])
    assert np.array_equal(sol3.x, [
        [1, 0],
        [1, 0],
        [0, 1]
    ])
    assert np.isclose(sol3.getCosts(), 160)

    print("Test 3 passed: Both facilities open, repaired cost = 160")

    original_y = lagr1.y.copy()
    original_x = lagr1.x.copy()

    heuristic.convertToFeasibleSolution(lagr1)

    assert np.array_equal(lagr1.y, original_y)
    assert np.array_equal(lagr1.x, original_x)

    print("Test 4 passed: Original Lagrangian solution unchanged")
    print("\nAll Part 4 tests passed!\n")


def test_part5():
    """Test the Polyak subgradient multiplier update."""

    instance = make_test_instance()
    heuristic = LagrangianHeuristic(instance)

    heuristic.beta = 1.5
    heuristic.bestUB = 125.0

    labda1 = np.array([50., 30., 40.])
    lagr1 = heuristic.computeLagrangianSolution(labda1)

    new_labda1 = heuristic.updateMultipliers(
        labda1,
        lagr1
    )

    # LB = 120, UB = 125, g = [1, 1, 1]
    # Step size = 1.5 * (125 - 120) / 3 = 2.5
    expected1 = np.array([52.5, 32.5, 42.5])

    assert np.allclose(new_labda1, expected1)

    print("Test 1 passed: Undersupplied customers increase multipliers")

    labda2 = np.array([80., 80., 80.])
    lagr2 = heuristic.computeLagrangianSolution(labda2)

    new_labda2 = heuristic.updateMultipliers(
        labda2,
        lagr2
    )

    # LB = 10, UB = 125, g = [-1, -1, -1]
    # Step size = 1.5 * (125 - 10) / 3 = 57.5
    expected2 = np.array([22.5, 22.5, 22.5])

    assert np.isclose(heuristic.computeTheta(labda2), 10)
    assert np.allclose(new_labda2, expected2)

    print("Test 2 passed: Oversupplied customers decrease multipliers")

    y3 = np.array([1, 0])
    x3 = np.array([
        [1, 0],
        [1, 0],
        [1, 0]
    ])

    lagr3 = UFL_Solution(y3, x3, instance)
    labda3 = np.array([30., 25., 35.])

    new_labda3 = heuristic.updateMultipliers(
        labda3,
        lagr3
    )

    assert np.array_equal(new_labda3, labda3)

    print("Test 3 passed: Zero subgradient handled")

    original_labda = labda1.copy()

    heuristic.updateMultipliers(labda1, lagr1)

    assert np.array_equal(labda1, original_labda)

    print("Test 4 passed: Original multipliers unchanged")
    print("\nAll Part 5 tests passed!\n")


def test_part6():
    """Test the complete iterative Lagrangian heuristic."""

    instance = make_test_instance()
    heuristic = LagrangianHeuristic(instance)

    bestLB, bestUB, best_solution, history, stop_reason = (
        heuristic.runHeuristic()
    )

    assert len(history) > 0

    LB_history = np.array([
        row["best_LB"] for row in history
    ])

    UB_history = np.array([
        row["best_UB"] for row in history
    ])

    assert best_solution is not None
    assert best_solution.isFeasible()

    print("Test 1 passed: Feasible solution found")

    assert bestLB <= bestUB + 1e-9
    assert np.isclose(bestUB, 125)
    assert np.isclose(best_solution.getCosts(), bestUB)

    print("Test 2 passed: Valid bounds and best solution cost")

    assert np.all(np.diff(LB_history) >= -1e-9)
    assert np.all(np.diff(UB_history) <= 1e-9)

    print("Test 3 passed: Best bounds are monotonic")

    assert len(LB_history) == len(UB_history)
    assert 1 <= len(history) <= 1000
    assert history[-1]["iteration"] == len(history)

    print("Test 4 passed: Iteration histories recorded")

    assert np.isclose(bestLB, LB_history[-1])
    assert np.isclose(bestUB, UB_history[-1])
    assert isinstance(stop_reason, str)

    print("Test 5 passed: Final bounds and termination recorded")

    gap_percent = (
        (bestUB - bestLB)
        / max(1.0, abs(bestUB))
        * 100
    )

    print(f"\nBest lower bound: {bestLB:.4f}")
    print(f"Best upper bound: {bestUB:.4f}")
    print(f"Relative gap: {gap_percent:.4f}%")
    print(f"Iterations: {len(history)}")
    print(f"Termination: {stop_reason}")

    print("\nAll Part 6 tests passed!\n")


if __name__ == "__main__":

    test_part1()
    test_part2()
    test_part3()
    test_part4()
    test_part5()
    test_part6()
