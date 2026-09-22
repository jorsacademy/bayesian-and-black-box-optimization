# SMAC3 Simulation-Based Optimization

A compact industrial-engineering example of simulation-based black-box optimization with **SMAC3 2.4.0**.

The project tunes a stochastic make-to-stock production policy. The two decisions are:

- `batch_size`: production lot size, searched over 5–40 units
- `safety_stock`: reorder trigger, searched over 0–30 units

The simulator models Poisson customer demand and Binomial production yield. The objective minimizes average per-period setup, production, inventory-holding, and lost-sales costs.

## Why SMAC3 here?

The objective is noisy, non-analytic, and only available through simulation. SMAC receives an explicit seed for every trial, which makes stochastic evaluations reproducible while still allowing different trials to see different random streams.

The repository also implements a budget-matched random-search baseline. Both incumbents are evaluated on the same independent validation seeds after search.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
python -m smac_sim.cli
# or
smac3-production-demo
```

## Test

```bash
pytest
```

CI enforces at least 90% coverage and runs on Python 3.10–3.13.

## Project structure

```text
src/smac_sim/simulation.py    # stochastic production simulator
src/smac_sim/optimization.py  # ConfigSpace, SMAC3 and random search
src/smac_sim/cli.py           # end-to-end comparison
tests/                        # unit and integration tests
.github/workflows/tests.yml   # multi-version CI
```

## Reproducibility

- NumPy uses `default_rng(seed)` inside the simulator.
- SMAC's `Scenario` is seeded.
- The SMAC intensifier evaluates each candidate with one explicit seed.
- Final validation uses a fixed seed set that is separate from search.

This keeps CI deterministic without pretending the underlying optimization problem is deterministic.
