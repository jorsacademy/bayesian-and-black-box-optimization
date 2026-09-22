# Repository Overlap Audit — Black-Box and Policy-Search Tooling

This document records portfolio overlap without merging, archiving, renaming, or deleting repositories.

## Status legend

- **Keep separate** — materially different method, orchestration model, or application.
- **Overlap but justified** — similar black-box problem, but a distinct library or algorithmic concept is the educational objective.
- **Potential consolidation** — unusually high duplication; review again before any future action.

## Inventory-policy tooling group

### `sambo-sequential-model-based-optimization`

**Overlap but justified.**

Focus: SAMBO sequential model-based optimization over an `(s,S)` policy with common random numbers.

### `hyperopt-inventory-policy-optimization`

**Overlap but justified.**

Focus: Hyperopt/TPE API, `Trials`, `fmin`, TPE versus random search, and valid integer policy parameterization.

### `botorch-bayesian-policy-search`

**Overlap but justified.**

Focus: explicit Gaussian-process Bayesian optimization with `SingleTaskGP`, marginal likelihood, acquisition functions, and acquisition optimization.

### `nevergrad-black-box-policy-optimization`

**Overlap but justified.**

Focus: derivative-free optimizer portfolio (`NGOpt`, CMA, PSO, RandomSearch) with integer instrumentation.

### `ray-tune-distributed-policy-search`

**Keep separate.**

Focus: distributed experiment orchestration, resource declarations, ASHA early stopping, incremental reporting, and result-grid handling. Ray Tune is serving a materially different systems/orchestration role from a local optimizer.

## Simulation/process tooling group

### `smac3-simulation-based-optimization`

**Keep separate.**

Focus: SMAC3 on a noisy make-to-stock simulator, explicit seeded trials, random-search control, and out-of-sample policy validation.

### `processoptimizer-industrial-process-optimization`

**Keep separate.**

Focus: ask/tell Gaussian-process experimental optimization for continuous industrial process factors. The domain, variable types, and experimental workflow differ from inventory-policy search.

## Audit conclusion

The SAMBO, Hyperopt, BoTorch, and Nevergrad repositories have substantial structural similarity because each optimizes a low-dimensional stochastic inventory policy. However, the repository-level learning objective is the optimizer/library itself rather than merely the inventory model.

Therefore they are **not current consolidation candidates**.

If portfolio maintenance ever becomes burdensome, the first non-destructive step should be to extract a shared inventory simulator/interface package or benchmark specification while keeping the library-specific repositories independent. A monorepo merge should be considered only if maintaining duplicated simulator code becomes more costly than the educational clarity gained from self-contained examples.
