from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import nevergrad as ng

from .simulation import InventoryConfig, mean_policy_cost

DEFAULT_OPTIMIZERS = ("NGOpt", "CMA", "PSO", "RandomSearch")


@dataclass(frozen=True)
class OptimizationResult:
    optimizer: str
    reorder_point: int
    order_quantity: int
    cost: float
    budget: int


def make_parametrization(seed: int = 0) -> ng.p.Instrumentation:
    parametrization = ng.p.Instrumentation(
        reorder_point=ng.p.Scalar(lower=0, upper=30).set_integer_casting(),
        order_quantity=ng.p.Scalar(lower=5, upper=45).set_integer_casting(),
    )
    parametrization.random_state.seed(seed)
    return parametrization


def optimize_policy(
    optimizer_name: str = "NGOpt",
    *,
    budget: int = 40,
    seed: int = 0,
    evaluation_seeds: tuple[int, ...] = (11, 23, 37, 53),
    config: InventoryConfig | None = None,
) -> OptimizationResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if optimizer_name not in ng.optimizers.registry:
        raise ValueError(f"unknown Nevergrad optimizer: {optimizer_name}")

    parametrization = make_parametrization(seed)
    optimizer_cls = ng.optimizers.registry[optimizer_name]
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget, num_workers=1)

    def objective(reorder_point: float, order_quantity: float) -> float:
        return mean_policy_cost(
            int(reorder_point),
            int(order_quantity),
            seeds=evaluation_seeds,
            config=config,
        )

    recommendation = optimizer.minimize(objective)
    reorder_point = int(recommendation.kwargs["reorder_point"])
    order_quantity = int(recommendation.kwargs["order_quantity"])
    cost = objective(reorder_point, order_quantity)

    return OptimizationResult(
        optimizer=optimizer_name,
        reorder_point=reorder_point,
        order_quantity=order_quantity,
        cost=cost,
        budget=budget,
    )


def compare_optimizers(
    optimizers: Iterable[str] = DEFAULT_OPTIMIZERS,
    *,
    budget: int = 40,
    seed: int = 0,
    evaluation_seeds: tuple[int, ...] = (11, 23, 37, 53),
    config: InventoryConfig | None = None,
) -> list[OptimizationResult]:
    results = [
        optimize_policy(
            name,
            budget=budget,
            seed=seed,
            evaluation_seeds=evaluation_seeds,
            config=config,
        )
        for name in optimizers
    ]
    return sorted(results, key=lambda result: result.cost)
