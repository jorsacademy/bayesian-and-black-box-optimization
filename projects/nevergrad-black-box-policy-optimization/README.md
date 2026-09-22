# Nevergrad Black-Box Policy Optimization

Gradient-free optimization of a stochastic inventory policy with Nevergrad 1.0.12.

## What this repository demonstrates

- a stochastic lost-sales inventory simulator,
- an `(r, Q)` policy with integer decision variables,
- common-random-number evaluation across fixed demand seeds,
- Nevergrad `Instrumentation` with bounded integer `Scalar` parameters,
- optimizer comparison across `NGOpt`, `CMA`, `PSO`, and `RandomSearch`,
- a command-line demonstration,
- pytest coverage and GitHub Actions CI.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
nevergrad-policy-demo
```

The command prints the optimizer, recommended reorder point, order quantity, and mean simulated cost. Lower cost is better.

## Test

```bash
pytest
```

CI runs on Python 3.10 through 3.13 and enforces at least 90% package coverage.
