# ProcessOptimizer Industrial Process Optimization

Bayesian optimization of a noisy industrial process using **ProcessOptimizer 1.1.2**.

## Problem

A synthetic batch process has three controllable continuous factors:

- temperature: 60–100 °C
- residence time: 1–8 h
- agitation: 200–800 rpm

The black-box loss combines yield/purity penalties, factor interactions, energy use, and experimental noise. The analytical objective is treated as unavailable to the optimizer.

## Optimization workflow

The example mirrors real experimental optimization:

1. define a `ProcessOptimizer.Space`,
2. create a Gaussian-process `Optimizer`,
3. call `ask()` for the next experiment,
4. evaluate the noisy process,
5. feed the result back with `tell()`,
6. repeat under a fixed experiment budget.

Search evaluations use common random numbers for fair comparison. The selected setting is then scored on independent validation seeds. A budget-matched random-search baseline is included.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
processoptimizer-demo
```

## Test

```bash
pytest
```

GitHub Actions verifies Python 3.10–3.12 and enforces at least 90% coverage.
