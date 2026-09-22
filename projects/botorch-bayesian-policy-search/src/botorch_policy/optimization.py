from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from .simulation import InventoryConfig, Policy, evaluate_policy

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE)


@dataclass(frozen=True)
class SearchResult:
    policy: Policy
    cost: float
    evaluations: int


def decode_policy(x: torch.Tensor) -> Policy:
    values = x.detach().cpu().double().flatten()
    if values.numel() != 2:
        raise ValueError("policy vector must have exactly two values")
    if torch.any(values < 0) or torch.any(values > 1):
        raise ValueError("policy vector must lie in [0, 1]^2")
    reorder_point = 2.0 + 18.0 * float(values[0])
    gap = 4.0 + 26.0 * float(values[1])
    return Policy(reorder_point=reorder_point, order_up_to=reorder_point + gap)


def policy_score(x: torch.Tensor, config: InventoryConfig = InventoryConfig()) -> float:
    """BoTorch maximizes score, so return negative expected cost."""
    return -evaluate_policy(decode_policy(x), config=config)


def _evaluate_tensor(xs: torch.Tensor, config: InventoryConfig) -> torch.Tensor:
    scores = [policy_score(x, config) for x in xs]
    return torch.tensor(scores, dtype=DTYPE).unsqueeze(-1)


def run_sobol_search(
    budget: int = 12,
    seed: int = 7,
    config: InventoryConfig = InventoryConfig(),
) -> SearchResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    engine = SobolEngine(dimension=2, scramble=True, seed=seed)
    xs = engine.draw(budget).to(dtype=DTYPE)
    ys = _evaluate_tensor(xs, config)
    idx = int(torch.argmax(ys).item())
    return SearchResult(decode_policy(xs[idx]), cost=float(-ys[idx].item()), evaluations=budget)


def run_botorch_search(
    n_initial: int = 6,
    n_iterations: int = 4,
    seed: int = 7,
    config: InventoryConfig = InventoryConfig(),
) -> SearchResult:
    if n_initial < 3:
        raise ValueError("n_initial must be at least 3")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")

    torch.manual_seed(seed)
    engine = SobolEngine(dimension=2, scramble=True, seed=seed)
    train_x = engine.draw(n_initial).to(dtype=DTYPE)
    train_y = _evaluate_tensor(train_x, config)

    for _ in range(n_iterations):
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        acquisition = LogExpectedImprovement(model=model, best_f=train_y.max())
        candidate, _ = optimize_acqf(
            acquisition,
            bounds=BOUNDS,
            q=1,
            num_restarts=3,
            raw_samples=24,
        )
        new_y = _evaluate_tensor(candidate, config)
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

    idx = int(torch.argmax(train_y).item())
    return SearchResult(
        policy=decode_policy(train_x[idx]),
        cost=float(-train_y[idx].item()),
        evaluations=len(train_x),
    )
