# Machine Scheduling Decision Support System

An interactive Python/Streamlit decision-support system for production scheduling. The application lets users define a scheduling problem, choose constraints and an objective, run multiple exact, heuristic, and metaheuristic solvers, and compare the resulting schedules through tables and Plotly visualizations.

## What the system supports

### Scheduling environments

- Single machine (`1`)
- Parallel machines (`P`)
- Flow shop (`F`)
- Job shop (`J`)
- Open shop (`O`)

### Constraints and objectives

The interface supports release dates, precedence, preemption-related options, and deadlines. Objectives include makespan (`Cmax`), total completion time (`sumCi`), weighted completion time (`sumwiCi`), maximum lateness (`Lmax`), and total tardiness (`sumTi`).

### Solvers

- Dispatching rules: SPT, LPT, EDD, ERD, WSPT, FIFO, Wrap-Around, and Johnson's rule for two-machine flow shops
- Genetic Algorithm
- Simulated Annealing
- Tabu Search
- Ant Colony Optimization
- LP-relaxation lower bounds

## Application workflow

1. Select the machine environment, constraints, and objective in the sidebar.
2. Configure the number of jobs and machines.
3. Edit processing times, release dates, due dates, and weights in the job table.
4. Select one or more solvers and tune their parameters.
5. Run the problem.
6. Compare objective values and computation times, inspect the best schedule, and review the Gantt and convergence charts.

The input columns are generated dynamically from the selected constraints and objectives. This keeps the problem definition aligned with the mathematical notation instead of exposing irrelevant fields.

## Installation and use

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501`.

## Verification results

The repository includes deterministic solver checks in `verify_solvers.py`. Running:

```bash
python verify_solvers.py
```

produced **7/7 passing tests**:

| Test | Result |
|---|---|
| Single machine, `Cmax` | LP bound = exhaustive optimum = SPT = 15 |
| Single machine, `sumCi` | LP bound = exhaustive optimum = SPT = 26 |
| Single machine, `Lmax` | EDD and exhaustive optimum = 2 |
| Single machine, `sumwiCi` | LP bound = exhaustive optimum = WSPT = 40 |
| Two-machine flow shop, `Cmax` | Exhaustive optimum = 12; LPT = 14; SPT = 15 |
| Two-machine parallel shop, `Cmax` | LP bound = LPT = 9 |
| Tabu Search convergence | Found the exhaustive optimum, `sumCi = 34` |

The separate job-shop check in `test_job_shop.py` produced makespans of 110.0 for forward order, 84.0 for reverse order, and eight distinct makespans across ten random orders in the documented run. This check is intentionally stochastic because the random-order portion is not seeded. These checks validate schedule construction, objective calculations, lower bounds, and the Tabu Search implementation on small instances.

## Project structure

```text
app.py                         Streamlit interface and solver orchestration
requirements.txt               Python dependencies
verify_solvers.py              Deterministic correctness and solver checks
test_job_shop.py               Job-shop schedule variance check
modules/
  scheduling_core.py           Jobs, machines, schedules, metrics, and validators
  dispatching_rules.py         Dispatching rules and schedule builders
  genetic_algorithm.py         DEAP-based genetic algorithm
  simulated_annealing.py       Simulated Annealing solver
  tabu_search.py               Tabu Search solver
  ant_colony.py                Ant Colony Optimization solver
  lp_solver.py                 LP-relaxation lower bounds
  visualization.py              Gantt, convergence, and comparison charts
```

## Scope

This is a scheduling DSS prototype for comparing scheduling formulations and solution strategies. LP results are lower bounds or model-specific approximations, while heuristic and metaheuristic results are feasible schedules whose quality depends on the selected instance and parameters.
