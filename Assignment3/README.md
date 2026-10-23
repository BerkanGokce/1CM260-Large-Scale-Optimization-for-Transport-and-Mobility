# Lagrangian Heuristic for the Uncapacitated Facility Location Problem

This repository implements and evaluates a **Lagrangian heuristic for the Uncapacitated Facility Location Problem (UFL)**. The goal is to select facilities to open and assign customers to open facilities while minimizing facility opening and transportation costs.

The method relaxes the customer assignment constraints to obtain a **lower bound (LB)**, repairs relaxed solutions to obtain feasible **upper bounds (UB)**, and updates Lagrangian multipliers using a **Polyak-type subgradient method**. Experiments are provided for five benchmark instances, **MO1–MO5**.

For the mathematical formulation, implementation details, results, and discussion, see [the project report](1CM260_Large_Scale_Optimization_for_Transport___Mobility___Assignment_3_Report.pdf).

## Repository structure

```text
Assignment3/
├── UFL.py                 # UFL data structures and Lagrangian heuristic
├── main.py                # Run benchmark experiments and save diagnostics
├── test.py                # Run correctness checks on a small test instance
├── visualize.py           # Generate facility–customer network diagrams
├── Instances/
│   ├── MO1
│   ├── MO2
│   ├── MO3
│   ├── MO4
│   └── MO5                 # Benchmark input instances
├── Results/                # Generated figures and CSV files
│   ├── summary.csv
│   ├── MO1_convergence.png
│   ├── MO1_history.csv
│   ├── MO1_multipliers.csv
│   ├── MO1_multipliers.png
│   ├── MO1_subgradient.png
│   ├── MO1_network.png
│   └── ...                 # Corresponding outputs for MO2–MO5
└── 1CM260_Large_Scale_Optimization_for_Transport___Mobility___Assignment_3_Report.pdf
```

### Python files

| File | Purpose |
|---|---|
| `UFL.py` | Defines `UFL_Problem`, `UFL_Solution`, and `LagrangianHeuristic`. Handles instance loading, feasibility and cost calculations, the Lagrangian relaxation, feasible-solution construction, multiplier updates, and the iterative heuristic. |
| `main.py` | Runs the heuristic on MO1–MO5, prints a results summary, and saves convergence plots, multiplier plots, subgradient diagnostics, and CSV data. |
| `test.py` | Tests the main implementation components using a small, constructed UFL instance with a known optimum. |
| `visualize.py` | Runs the heuristic on MO1–MO5 and generates schematic diagrams of open/closed facilities, customers, and assignments. |

## How to run

Run the following command from the repository directory (the folder containing `main.py` and `UFL.py`):

```bash
python main.py
```

This runs the Lagrangian heuristic on **MO1–MO5** using the configured parameters (5,000 maximum iterations, initial step-size parameter `beta = 1.5`, and step-size reduction after 20 iterations without a best-lower-bound improvement).

The terminal displays each instance's best lower bound (LB), best upper bound (UB), relative gap, iteration count, number of open facilities, runtime, and stopping reason.

The script saves the following outputs in `Results/`:

| Output | Contents |
|---|---|
| `summary.csv` | One summary row per benchmark instance. |
| `MO*_convergence.png` | Best lower and upper bounds across iterations. |
| `MO*_history.csv` | Per-iteration bounds, gap, step-size parameter, subgradient norm, step size, and multiplier summary statistics. |
| `MO*_multipliers.csv` | Full multiplier vector recorded at every iteration. |
| `MO*_multipliers.png` | Evolution of the eight customer multipliers with the highest variation across iterations. |
| `MO*_subgradient.png` | Subgradient norm and Polyak step size across iterations. |

Here `MO*` represents `MO1`, `MO2`, `MO3`, `MO4`, or `MO5`. Existing outputs may be overwritten when the script is rerun.

Facility–customer network diagrams (`MO*_network.png`) are generated separately by `visualize.py`; they are not produced by `main.py`. These diagrams use illustrative coordinates rather than real geographic locations or distances.

## Configuration

The experiment settings are defined near the top of `main.py`:

```python
INSTANCE_NAMES = ["MO1", "MO2", "MO3", "MO4", "MO5"]
MAX_ITERATIONS = 5000
STAGNATION_LIMIT = 20
INITIAL_BETA = 1.5
```

Change these values to select benchmark instances or adjust the heuristic settings. `visualize.py` imports these settings from `main.py`, so the same configuration applies to its network runs.

The network drawing options (such as showing closed facilities or customer labels) are defined near the top of `visualize.py`.

## Understanding the results

- **Best LB:** Highest Lagrangian lower bound found so far.
- **Best UB:** Lowest cost of a feasible UFL solution found so far.
- **Gap (%):** `100 × (Best UB − Best LB) / max(1, |Best UB|)`; this is a bound-based optimality gap, not necessarily the actual distance from the optimum.
- **Convergence plot:** Shows whether the best bounds improve as iterations progress.
- **Multiplier plot:** Shows how selected customer assignment multipliers change during optimization.
- **Subgradient plot:** Helps assess remaining relaxed assignment violations and the size of multiplier updates.

For the benchmark results and a detailed interpretation of the convergence behavior, refer to the [report PDF](1CM260_Large_Scale_Optimization_for_Transport___Mobility___Assignment_3_Report.pdf).
