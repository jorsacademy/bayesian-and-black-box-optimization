# BoTorch Bayesian Policy Search

Bayesian optimization of a stochastic `(s, S)` inventory policy using **BoTorch 0.18.1**.

## Problem

The simulator models Poisson demand with immediate replenishment. A policy is defined by:

- `reorder_point` (`s`)
- `order_up_to` (`S`)

The cost combines ordering, holding, and lost-sales penalties. Policy evaluation uses a fixed set of demand seeds (common random numbers), making objective comparisons deterministic and reproducible.

## Bayesian optimization

The normalized two-dimensional policy vector is mapped to a feasible `(s, S)` policy. BoTorch then uses:

- `SingleTaskGP`
- `ExactMarginalLogLikelihood`
- `LogExpectedImprovement`
- `optimize_acqf`

A budget-matched scrambled Sobol search is included as the model-free baseline.

## Run

```bash
python -m pip install -e '.[dev]'
botorch-policy-demo
pytest
```

## Validation

The test suite checks simulator reproducibility and input validation, normalized-policy decoding, reproducible Sobol search, a real GP/acquisition optimization iteration, CLI behavior, and at least 90% coverage. GitHub Actions runs on Python 3.11, 3.12, and 3.13.
