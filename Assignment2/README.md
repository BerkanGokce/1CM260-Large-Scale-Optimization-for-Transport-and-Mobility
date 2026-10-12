# Two-Echelon Capacitated Vehicle Routing Problem — Adaptive Large Neighborhood Search

This repository contains the implementation and computational experiments for **Assignment 2** of *Large-Scale Optimization for Transport & Mobility* (Yasmine Beck).

The **Two-Echelon Capacitated Vehicle Routing Problem (2E-CVRP)** concerns delivering goods from depots to customers through intermediate satellite facilities. First-echelon vehicles transport goods from depots to satellites, while second-echelon vehicles deliver goods from satellites to customers. The objective is to **minimize total travel distance, fixed vehicle costs, and satellite handling costs**, subject to vehicle-capacity constraints and the required customer deliveries.

The project builds on the Python ALNS template provided with the assignment. Its underlying problem, route, solution, and ALNS structures have largely been retained and extended, rather than replacing the original framework. The extensions include additional destroy and repair operators, adaptive operator selection, an optional **simulated annealing (SA)** acceptance criterion, an alternative constructive heuristic, feasibility checks, and benchmarking/visualization scripts.

See the accompanying assignment report for the methodology, design decisions, and detailed results.

## Getting started

Run the following commands **from the repository root** (`Assignment2/`). Make sure [Conda](https://docs.conda.io/) is installed.

**1. Create a Conda environment**

```bash
conda create -n env_name python=3.11 -y
```

**2. Activate the environment**

```bash
conda activate env_name
```

**3. Install the required dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Run the project**

For example, to solve a single 2E-CVRP instance using ALNS:

```bash
python main.py
```

The `instances/` directory must contain the supplied 2E-CVRP `.txt` instances. The code expects the assignment's instance naming convention, such as `Ca1-2,3,15.txt`.

See the **Which script should I run?** section below for the available experiments and their outputs.

## Which script should I run?

| Goal | Command | What it does / where to find results |
|---|---|---|
| **Solve one instance with ALNS** | `python main.py` | Runs ALNS on the instance configured in `main.py` and prints the best first- and second-echelon routes. |
| **Run and visualize the two constructive heuristics** | `python main_constructive_run.py` | Runs Random Insertion and Clarke–Wright Savings on the configured example instance and saves route visualizations in `constructive_results/`. |
| **Benchmark the constructive heuristics** | Edit `main_constructive_run.py` to enable `main_comparison(...)`, then run `python main_constructive_run.py` | Evaluates both initial-solution methods over the available instances; writes `constructive_results/constructive_results.csv` and `.txt`. The comparison plot can also be enabled in this script. |
| **Benchmark ALNS without SA** | Set `USE_SA = False` in `main_alns_results.py`, then run `python main_alns_results.py` | Evaluates both constructive starting methods; saves CSVs and optional plots in `alns_results/`. |
| **Benchmark ALNS with SA** | Set `USE_SA = True` in `main_alns_results.py`, then run `python main_alns_results.py` | Runs the same benchmark with simulated annealing enabled; saves CSVs and optional plots in `alns_sa_results/`. |
| **Compare ALNS with and without SA** | `python main_compare_alns_sa.py` | Reads the **existing** benchmark CSVs (does not rerun ALNS); writes comparison tables and PNG plots in `alns_comparison_results/`. |

**Recommended workflow:** First run the regular ALNS benchmark (`USE_SA = False`), then the SA benchmark (`USE_SA = True`), and finally `main_compare_alns_sa.py`. Keep the same instances, random seeds, iteration counts, and constructive methods for a fair comparison.

### Main configuration options

For a **single ALNS run**, edit `main.py`:

```python
path_to_file = "instances/Ca1-2,3,15.txt"
algo.constructive_method = "random"  # "random" or "savings"
algo.use_sa = True                  # False disables simulated annealing
```

For **batch ALNS experiments**, edit the configuration section at the top of `main_alns_results.py`:

```python
USE_SA = True
NUM_ITERATIONS = 200
BASE_SEED = 42
N_RUNS = 5
SAVE_PLOTS = True
SA_INITIAL_TEMPERATURE = None  # Automatically calibrated
SA_COOLING_RATE = 0.99
SA_END_TEMPERATURE = 1e-8
```

`N_RUNS` controls how many seeds are evaluated per instance and constructive method. With `SA_INITIAL_TEMPERATURE = None`, the starting temperature is calibrated from the initial solution cost.

**Note:** Rerunning a benchmark writes to its configured output directory. Preserve existing results before changing parameters.

## Algorithms at a glance

### Constructive heuristics (`constructive.py`)

- **Random Insertion:** Builds an initial feasible solution by inserting customers into vehicle routes using randomized choices and insertion logic.
- **Clarke–Wright Savings:** An alternative initialization method based on merging routes according to distance savings.

### Destroy operators (`destroy.py`)

- **Random Removal:** Randomly selects customers to remove from second-echelon routes.
- **Worst Removal:** Removes customers with the largest marginal travel-distance savings, recalculating after each removal.
- **Shaw Related Removal:** Removes groups of related customers based on geography, demand, and satellite assignment.

### Repair operators (`repair.py`)

- **Greedy Cheapest Insertion:** Repeatedly inserts the customer with the cheapest available insertion.
- **Regret-2 Insertion:** Prioritizes customers based on the difference between their best and second-best route alternatives.
- **Random Feasible Insertion:** Randomly chooses customers and feasible insertion options.

After customer-route repair, the first-echelon routes are rebuilt to supply the demand assigned to each satellite.

ALNS selects destroy and repair operators using **adaptive weights** that respond to search outcomes. When enabled, **simulated annealing** can accept a worse candidate solution with a temperature-dependent probability, helping the search explore beyond improving moves.

## Repository structure

```text
Assignment2/
├── instances/                  # Supplied 2E-CVRP input instances
├── constructive_results/       # Constructive-heuristic results and visualizations
├── alns_results/               # ALNS benchmark outputs (SA disabled)
├── alns_sa_results/            # ALNS benchmark outputs (SA enabled)
├── alns_comparison_results/    # ALNS vs. SA comparison tables and figures
│
├── main.py                     # Entry point: solve a single instance with ALNS
├── main_constructive_run.py    # Constructive examples, benchmark, and plots
├── main_alns_results.py             # Batch ALNS benchmark and per-run visualizations
├── main_compare_alns_sa.py          # Compare previously saved ALNS/SA results
│
├── alns.py                     # ALNS loop, adaptive weights, and SA acceptance
├── constructive.py             # Random Insertion and Clarke–Wright Savings
├── destroy.py                  # Three destroy operators
├── repair.py                   # Three repair operators; first-echelon rebuilding
├── solution.py                 # Solution representation, cost, and feasibility checks
├── route.py                    # Vehicle-route representation and route costs
├── problem.py                  # Instance parsing and distance calculations
├── visualize.py                # Route plotting and visualization utilities
│
├── requirements.txt            # Python dependencies
├── 1CM260_Large_Scale_Optimization_for_Transport_Mobility_Assignment_2_Report.pdf # Assignment report
├── assignment-2-description.pdf # Original assignment specification
└── README.md                   # Project guide
```

`__pycache__/` is generated automatically by Python and is not part of the implementation.

## Understanding the results

The batch ALNS script writes three principal CSV files in either `alns_results/` or `alns_sa_results/`:

| File | Contents |
|---|---|
| `alns_run_results.csv` | One row per run, including initial and best costs, improvements, runtime, route counts, and SA settings. |
| `alns_iteration_results.csv` | Iteration-level search history, including candidate/current/best costs, selected operators, weights, and acceptance information. |
| `alns_summary_results.csv` | Aggregated statistics across repeated runs for each instance and constructive method. |

When plot saving is enabled, per-run figures are organized by constructive heuristic, instance, and seed, including route visualizations, operator-weight evolution, and convergence.

`main_compare_alns_sa.py` creates paired comparison CSVs and figures; it requires the corresponding benchmark CSVs to have been generated first.

## Feasibility and scope

The solution validator checks customer coverage, delivered demand, vehicle capacities, route structure, and satellite supply–demand balance. Completed solutions are validated during the construction and ALNS workflow.

The implementation addresses the **capacitated** two-echelon VRP described in the assignment. Customer time windows and service times in the instance files are **not** enforced.

For further technical explanations, experiment settings, and interpretation of results, refer to the project report.

## Attribution

The original assignment and starter Python template were provided by **Yasmine Beck** for *Large-Scale Optimization for Transport & Mobility*. This repository extends that template for the assignment.

Supporting academic references and discussion of AI-assisted development are provided in the assignment report.