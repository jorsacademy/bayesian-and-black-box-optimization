from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sambo

from .inventory import InventoryConfig, evaluate_policy


@dataclass(frozen=True)
class PolicyOptimizationResult:
    reorder_point: int
    order_upto: int
    average_cost: float
    evaluations: int


def optimize_inventory_policy(
    *,
    config: InventoryConfig | None = None,
    max_iter: int = 30,
    replications: int = 12,
    seed: int = 42,
) -> PolicyOptimizationResult:
    """Optimize an (s, S) policy with SAMBO's SMBO algorithm."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    cfg = config or InventoryConfig()

    def objective(x: np.ndarray) -> float:
        reorder_point = int(round(float(x[0])))
        order_upto = int(round(float(x[1])))
        return evaluate_policy(
            reorder_point,
            order_upto,
            config=cfg,
            replications=replications,
            seed=seed,
        )

    result = sambo.minimize(
        objective,
        bounds=[(0, cfg.max_inventory - 1), (1, cfg.max_inventory)],
        constraints=lambda x: x[0] < x[1],
        method="smbo",
        max_iter=max_iter,
    )

    return PolicyOptimizationResult(
        reorder_point=int(round(float(result.x[0]))),
        order_upto=int(round(float(result.x[1]))),
        average_cost=float(result.fun),
        evaluations=int(result.nfev),
    )


def main() -> None:
    result = optimize_inventory_policy()
    print("Best (s, S) policy")
    print(f"reorder_point={result.reorder_point}")
    print(f"order_upto={result.order_upto}")
    print(f"average_cost_per_period={result.average_cost:.3f}")
    print(f"objective_evaluations={result.evaluations}")


if __name__ == "__main__":
    main()
