# Bayesian and Black-Box Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`bayesian-optimization-endustri-muhendisligi`](projects/bayesian-optimization-endustri-muhendisligi/)
- [`botorch-bayesian-policy-search`](projects/botorch-bayesian-policy-search/)
- [`constrained-bayesian-optimization-chemical-process-python`](projects/constrained-bayesian-optimization-chemical-process-python/)
- [`nevergrad-black-box-policy-optimization`](projects/nevergrad-black-box-policy-optimization/)
- [`processoptimizer-industrial-process-optimization`](projects/processoptimizer-industrial-process-optimization/)
- [`sambo-sequential-model-based-optimization`](projects/sambo-sequential-model-based-optimization/)
- [`smac3-simulation-based-optimization`](projects/smac3-simulation-based-optimization/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

This repository is an educational introduction to Gaussian Process Regression (GPR) and Gaussian Process-based Bayesian Optimization in an industrial engineering context.

The central distinction is simple but important: `sklearn.gaussian_process.GaussianProcessRegressor` is **not** an optimizer by itself. It is a probabilistic regression model that can act as a **surrogate model** inside a Bayesian Optimization algorithm.

## What this repository covers

- Gaussian Process Regression fundamentals
- The difference between GPR and Bayesian Optimization
- Expected Improvement, Probability of Improvement, and Lower Confidence Bound
- A from-scratch Gaussian Process-based Bayesian Optimization loop for continuous decision variables
- Candidate-set Bayesian Optimization for discrete decision variables
- Simulation-based workforce and capacity optimization
- A synthetic manufacturing process-parameter optimization example
- Two-fidelity Gaussian Process surrogate modeling
- Cost-aware multi-fidelity optimization under an explicit evaluation budget
- Guidance on when Bayesian Optimization is and is not appropriate in industrial engineering

## Conceptual workflow

```text
Expensive objective / simulation / physical experiment
                    |
                    v
Observed data: X, y
                    |
                    v
Gaussian Process surrogate: mu(x), sigma(x)
                    |
                    v
Acquisition function
                    |
                    v
Choose the next experiment
                    |
                    v
Evaluate the expensive objective
                    |
                    v
Update the data and repeat
```

For multi-fidelity problems, the decision is extended to both **where** and **how accurately** to evaluate:

```text
candidate x
    |
    v
low-cost / low-fidelity evaluation
        or
high-cost / high-fidelity evaluation
    |
    v
cost-aware surrogate update
```

## Repository structure

```text
.
├── README.md
├── LICENSE.md
├── requirements.txt
├── src/
│   ├── gaussian_bo.py
│   ├── discrete_bo.py
│   ├── multi_fidelity.py
│   └── production_simulation.py
├── notebooks/
│   ├── 00_gaussian_process_regression_fundamentals.ipynb
│   ├── 01_bayesian_optimization_fundamentals.ipynb
│   ├── 02_shift_staffing_optimization.ipynb
│   └── 03_manufacturing_process_parameter_optimization.ipynb
└── docs/
    ├── code_review_notes.md
    ├── industrial_engineering_application_guide.md
    └── multi_fidelity_and_cost_aware_bo.md
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The notebooks can also be run in Google Colab after cloning the repository.

## Why implement the Bayesian Optimization loop directly?

Scikit-learn provides `GaussianProcessRegressor` and Gaussian Process kernels, but it does not provide a general-purpose `BayesianOptimization` estimator. The implementation in this repository keeps the sequential optimization mechanism visible: fit the GP, compute an acquisition function, choose a new point, evaluate the expensive objective, and update the model.

A higher-level alternative is `gp_minimize` from the separate `scikit-optimize` package. `scikit-optimize` is not a submodule of scikit-learn.

The multi-fidelity example follows the same teaching principle. It uses a transparent two-fidelity autoregressive surrogate

```text
f_high(x) = rho * f_low(x) + discrepancy(x)
```

and makes the evaluation-cost trade-off explicit rather than hiding it inside a framework.

See [Multi-Fidelity and Cost-Aware Bayesian Optimization](docs/multi_fidelity_and_cost_aware_bo.md).

Run the example with:

```bash
python src/multi_fidelity.py
```

## When Bayesian Optimization is a strong candidate

Bayesian Optimization becomes attractive when:

- each function evaluation is expensive,
- the objective has no convenient closed form,
- evaluation requires simulation, a physical experiment, or a costly software model,
- gradients are unavailable or unreliable,
- the number of decision variables is low to moderate,
- the evaluation budget is limited.

Industrial engineering examples include:

- CNC, welding, injection molding, furnace, or heat-treatment parameter tuning,
- simulation-based staffing and capacity planning,
- maintenance interval and alarm-threshold tuning,
- inventory-policy parameter tuning under stochastic demand,
- production-line simulation calibration,
- energy-quality trade-off tuning,
- hyperparameter tuning for expensive optimization or machine-learning pipelines.

Multi-fidelity optimization becomes especially useful when a cheaper approximation is available, such as a coarse simulation, reduced-order engineering model, shorter horizon, smaller Monte Carlo sample, or early-stopped training run.

## When it is usually not the first choice

Bayesian Optimization is not a general replacement for Operations Research methods. If a problem can be formulated directly and solved efficiently as LP, MILP, MINLP, CP-SAT, network optimization, or another structured model, those methods are usually more appropriate.

Bayesian Optimization is particularly useful when an expensive black-box evaluation layer sits outside the mathematical model.

Likewise, multi-fidelity methods are not automatically useful merely because two simulators exist. The cheaper fidelity must be materially cheaper and informative about the high-fidelity objective.

## Scope of the examples

The repository now provides a deliberate progression:

1. understand Gaussian Process prediction and uncertainty,
2. build a complete continuous Bayesian Optimization loop,
3. solve a stochastic discrete industrial-engineering problem and verify it against exhaustive search,
4. apply the same ideas to a manufacturing process experiment,
5. extend the surrogate view to a cost-aware two-fidelity optimization problem.

The multi-fidelity implementation is intentionally educational: its transparent fidelity-selection policy is useful for understanding the mechanics, but it is not presented as a universal or state-of-the-art acquisition strategy.

The repository is therefore intended as a compact but complete teaching sequence, not an exhaustive reference on modern Bayesian Optimization.

## License

This repository is intended for noncommercial use. See `LICENSE.md` for details.
