# Code Review Notes

This document records the main technical corrections made to earlier Gaussian Process / Bayesian Optimization examples.

## 1. What was conceptually correct

The original architecture was fundamentally sound:

1. use `GaussianProcessRegressor` as a surrogate model,
2. fit the GP to observed `X, y` data,
3. obtain `mu(x)` and `sigma(x)`,
4. compute an acquisition criterion such as EI, PI, or a confidence bound,
5. optimize that criterion to choose the next expensive evaluation,
6. evaluate the true objective,
7. add the new observation and repeat.

That is the standard sequential surrogate-optimization structure of Gaussian Process-based Bayesian Optimization.

## 2. Important corrections to the earlier custom optimizer

### 2.1 Bounds handling

The older code converted `bounds` to a NumPy array but then read `bounds.shape[0]` from the original object. A Python list has no `.shape` attribute. The corrected implementation reads the dimension from `self.bounds` after conversion and validates the complete `(n_dimensions, 2)` shape.

### 2.2 LCB direction/sign error

For minimization, the Lower Confidence Bound is commonly written as:

```text
LCB(x) = mu(x) - kappa * sigma(x)
```

and is minimized. The older implementation computed this value but passed it to a framework that **maximized** acquisition values. The corrected implementation uses `score = -LCB`, so every acquisition measure follows one consistent convention: larger score is better.

### 2.3 Matérn `nu=2.5` interpretation

The old comment described Matérn `nu=2.5` as corresponding to once-differentiable functions. The usual scikit-learn interpretation is approximately:

- `nu=1.5`: once differentiable,
- `nu=2.5`: twice differentiable.

### 2.4 Scalar objective outputs

In one-dimensional examples, an objective may accidentally return a NumPy array of shape `(1,)` rather than a scalar. This can produce inconsistent target arrays. The new implementation validates and converts every objective value to one finite `float`.

### 2.5 Reproducibility

The earlier code used several uncontrolled sources of randomness. The revised version uses `numpy.random.default_rng(random_state)` and also passes the seed to the Gaussian Process model.

### 2.6 Acquisition-function optimization

The original continuous implementation relied only on a small number of random L-BFGS-B starts. The revised implementation first evaluates thousands of random acquisition candidates, then locally refines several of the best candidates. This is more robust for low-dimensional teaching examples.

### 2.7 Decision-variable scaling

Industrial variables can have very different scales, for example spindle speed in thousands of rpm and depth of cut in millimeters. The continuous optimizer internally maps every decision variable to `[0, 1]` before fitting the GP.

## 3. Corrections to the staffing example

### 3.1 Minimizing waiting time alone

If the objective is only waiting time and staffing has no cost or budget penalty, the optimizer is naturally driven toward the maximum allowed staffing level. A result such as `[10, 10, 10]` is therefore a property of the objective formulation, not evidence that Bayesian Optimization discovered a subtle trade-off.

### 3.2 Mixing incompatible units

Adding `waiting minutes + monetary staffing cost + penalty` directly is not economically meaningful. The revised example converts waiting time into a monetary cost using a stated cost per customer-waiting-minute. The final objective is therefore expressed in one economic unit.

### 3.3 Randomness management

Calling `np.random.seed()` inside the simulation makes the experimental design hard to control. The revised simulation uses explicit seeds and `default_rng`.

### 3.4 Common Random Numbers

All candidate staffing policies are evaluated using the same replication seeds. This is a simple Common Random Numbers variance-reduction design and makes pairwise policy comparisons less noisy.

### 3.5 Shift start times were not actually optimized

An older version defined a `shift_starts` list but did not include start times in the decision variables or use them meaningfully in the queue model. The current staffing notebook is explicit: shift start times are fixed. Optimizing overlapping shift schedules requires a time-dependent capacity model and a different simulation structure.

## 4. `scikit-optimize` and `gp_minimize`

The basic use of `gp_minimize` was valid. Two clarifications matter:

1. `scikit-optimize` is a separate package, not a scikit-learn submodule.
2. Integer decisions are more clearly represented with explicit integer dimensions when using that library.

This repository implements the main examples directly on top of `GaussianProcessRegressor` so the relationship between the GP surrogate and the Bayesian Optimization loop remains visible.
