# SAMBO Sequential Model-Based Optimization

A compact Sequential Decision Analytics example using [SAMBO](https://sambo-optimization.github.io/) to optimize a stochastic inventory-control policy.

The decision rule is an `(s, S)` policy:

- if inventory is below `s`, order up to `S`;
- otherwise, place no order.

SAMBO treats the simulation as a black-box objective and searches for policy parameters that minimize average cost per period. The optimization uses SAMBO's sequential model-based optimization (`method="smbo"`).

## Why this is an SDA example

The inventory environment contains the sequential process:

- **State:** current inventory.
- **Decision:** order quantity.
- **Exogenous information:** stochastic Poisson demand.
- **Transition:** inventory after demand realization.
- **Objective:** holding + stockout + ordering cost.

SAMBO does not solve the state transition directly. Instead, it performs **direct policy search** over the parameters of the sequential decision policy.

## Reproducibility

Candidate policies are compared with **common random numbers**: every candidate is evaluated against the same deterministic seed schedule. This reduces simulation noise and makes both tests and comparisons reproducible.

## Project structure

```text
.
├── .github/workflows/tests.yml
├── pyproject.toml
├── src/sambo_inventory/
│   ├── __init__.py
│   ├── inventory.py
│   └── optimize.py
└── tests/
    ├── test_inventory.py
    └── test_optimize.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

SAMBO requires Python 3.10+.

## Run the optimization

```bash
python -m sambo_inventory.optimize
```

The program prints the best reorder point, order-up-to level, estimated average cost per period, and number of objective evaluations.

## Run tests

```bash
pytest
```

The test suite checks deterministic simulation, policy validation, cost evaluation, and the integration contract with `sambo.minimize(..., method="smbo")`.

## Continuous integration

GitHub Actions runs the tests on Python 3.10, 3.11, and 3.12 for every pull request and for pushes to `main`. Coverage must remain at least 85%.

## References

- SAMBO documentation: https://sambo-optimization.github.io/
- SAMBO source: https://github.com/sambo-optimization/sambo
