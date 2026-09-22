# Black-Box, Sequential Model-Based, and Simulation Optimization Series

This file maps repositories that optimize objectives which are expensive, noisy, simulation-based, or otherwise unavailable in convenient closed form. It is an index only: each repository remains independent because the optimization engines, assumptions, and evaluation contracts differ.

## Sequential model-based and Bayesian-style optimization

- `smac3-simulation-based-optimization` — SMAC3-based model-based optimization for simulation/expensive objectives.
- `sambo-sequential-model-based-optimization` — sequential model-based optimization through a different toolkit and modeling stack.
- `botorch-bayesian-policy-search` — Gaussian-process / acquisition-function based policy search using BoTorch.
- `constrained-bayesian-optimization-chemical-process-python` — Bayesian optimization with explicit process constraints.
- `processoptimizer-industrial-process-optimization` — process-tuning workflow using a specialized sequential optimizer.

## General black-box and policy search

- `nevergrad-black-box-policy-optimization` — derivative-free black-box policy optimization.
- `hyperopt-inventory-policy-optimization` — black-box tuning of inventory-policy parameters.
- `ray-tune-distributed-policy-search` — distributed experiment/policy search infrastructure.

## Simulation optimization and calibration

- `manufacturing-discrete-event-simulation-optimization-python` — optimization around a discrete-event simulation.
- `airport-checkin-simulation-optimization` — simulation optimization for airport service capacity.
- `simulation-model-calibration-uq` — calibration and uncertainty quantification for simulation models.
- `parallel-monte-carlo-stochastic-optimization-python` — Monte Carlo based stochastic evaluation/optimization.
- `cpp-accelerated-optimization-simulation-python` — performance-oriented optimization/simulation integration.

## Why these stay separate

Sharing a black-box objective does not make the methods equivalent. The repositories differ in surrogate models, acquisition/search rules, parallelism, constraints, stochastic noise, and whether the goal is parameter tuning, policy search, or simulation calibration. The useful organization is therefore a comparative tooling/methods series rather than one consolidated repository.
