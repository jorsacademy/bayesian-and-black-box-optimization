# Multi-Fidelity and Cost-Aware Bayesian Optimization

Multi-fidelity optimization is useful when the same design can be evaluated at different levels of cost and accuracy.

Industrial examples include:

- coarse versus detailed simulation,
- short versus long simulation horizons,
- small versus large Monte Carlo sample sizes,
- low-resolution versus high-resolution engineering models,
- reduced-order versus full physics models,
- pilot-line versus production-scale experiments,
- early-stopped versus fully trained machine-learning models.

The central question is no longer only

> Which point should be evaluated next?

It becomes

> Which point should be evaluated next, and at what fidelity?

## 1. Two-fidelity model

Let

```text
f_L(x) = cheap, approximate evaluation
f_H(x) = expensive, decision-relevant evaluation
```

The implementation in `src/multi_fidelity.py` uses a compact autoregressive relationship

```text
f_H(x) = rho * f_L(x) + delta(x)
```

where:

- a Gaussian Process models the low-fidelity response,
- `rho` estimates linear cross-fidelity scaling,
- a second Gaussian Process models the high-fidelity discrepancy `delta(x)`.

This structure lets cheap evaluations inform beliefs about the expensive response without pretending that the low-fidelity model is exact.

## 2. Cost-aware acquisition

Ordinary Bayesian Optimization often chooses the point with the largest acquisition value.

With multiple fidelities, an evaluation also has a cost. A simple decision rule is therefore based on utility per unit cost.

For high fidelity, the educational implementation uses approximately

```text
high utility = Expected Improvement / evaluation cost
```

For low fidelity, it uses a transparent uncertainty-reduction proxy

```text
low utility = scaled low-fidelity uncertainty / low-fidelity cost
```

The algorithm then chooses the best currently affordable action.

This is intentionally a teaching policy. It is not presented as a universal or theoretically optimal multi-fidelity acquisition function.

## 3. Why high-fidelity evaluations also receive a low-fidelity value

When the implementation selects a high-fidelity point, it evaluates the cheap model at the same point before the expensive model.

That paired observation improves estimation of the relationship

```text
f_H(x) <-> f_L(x)
```

and keeps the discrepancy model interpretable.

If a real application already stores a low-fidelity value at that exact point, production code should reuse it rather than pay for it again.

## 4. Budget accounting

The optimizer tracks explicit evaluation costs:

```python
low_cost = 1.0
high_cost = 12.0
budget = 90.0
```

This is preferable to comparing algorithms only by iteration count. One high-fidelity evaluation may cost more than many low-fidelity evaluations.

For real experiments, replace the abstract cost with the resource that matters:

- wall-clock time,
- CPU/GPU hours,
- simulation replications,
- laboratory cost,
- energy,
- engineering effort.

## 5. Run the example

From the repository root:

```bash
python src/multi_fidelity.py
```

The script optimizes a synthetic one-dimensional high-fidelity function using a cheaper biased approximation.

It reports:

- selected fidelity,
- cumulative cost,
- best observed high-fidelity objective,
- final best point,
- number of low- and high-fidelity evaluations.

## 6. Experimental comparison to add in applications

A useful multi-fidelity experiment should compare at least:

1. random search using only high fidelity,
2. ordinary Bayesian Optimization using only high fidelity,
3. low-fidelity-only optimization followed by one high-fidelity validation,
4. cost-aware multi-fidelity optimization.

Use the **same total resource budget**.

Primary metrics should be decision-relevant:

- best validated high-fidelity objective,
- regret to a known reference if available,
- high-fidelity evaluation count,
- total cost,
- feasible-run rate for constrained problems.

## 7. Fidelity must represent a real approximation hierarchy

A fidelity variable should have operational meaning.

Good examples:

```text
simulation replications: 20 -> 100 -> 1000
mesh resolution: coarse -> medium -> fine
training epochs: 5 -> 25 -> 100
planning horizon: short -> medium -> full
physics model: reduced order -> full model
```

Do not manufacture a “low fidelity” by adding arbitrary noise unless the purpose is only to test software mechanics. The correlation and bias between fidelities determine whether cheap evaluations are informative.

## 8. When multi-fidelity optimization helps

It is attractive when:

- high-fidelity evaluations are expensive,
- cheaper approximations are materially cheaper,
- cheaper evaluations are correlated with the high-fidelity objective,
- the evaluation budget is limited,
- the optimizer can decide how to allocate resources adaptively.

It is less attractive when:

- the cheap model is weakly related to the true objective,
- the high-fidelity model is already inexpensive,
- fidelity cost differences are negligible,
- model bias changes unpredictably across the design space.

## 9. Relation to Hyperband and successive halving

Multi-fidelity optimization is broader than Gaussian-Process Bayesian Optimization.

Successive Halving and Hyperband allocate progressively larger budgets to promising configurations. BOHB combines budget allocation with model-based search.

Those methods are especially natural when fidelity is an ordered resource such as training epochs, sample size, or simulation length.

The implementation in this repository instead demonstrates a two-source surrogate model with explicit cross-fidelity correlation.

## 10. Extensions

Natural next steps include:

- more than two fidelity levels,
- fidelity-dependent noise,
- constrained multi-fidelity optimization,
- multi-objective cost-aware acquisition,
- learned evaluation cost,
- asynchronous parallel evaluations,
- BoTorch multi-fidelity acquisition functions,
- continuous fidelity variables,
- benchmark comparison against Hyperband/BOHB.

The important methodological principle is to evaluate the final recommendation at the high-fidelity level. Cheap-fidelity objective values should not be reported as if they were the true decision-quality metric.
