from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    initial_inventory: int = 20
    max_inventory: int = 50
    holding_cost: float = 2.0
    stockout_cost: float = 10.0
    ordering_cost: float = 5.0
    demand_lambda: float = 15.0
    periods: int = 100


class InventoryEnvironment:
    """Simple periodic-review inventory environment with Poisson demand."""

    def __init__(self, config: InventoryConfig | None = None, seed: int | None = None):
        self.config = config or InventoryConfig()
        self.rng = np.random.default_rng(seed)
        self.state = self.config.initial_inventory
        self.period = 0

    def reset(self) -> int:
        self.state = self.config.initial_inventory
        self.period = 0
        return self.state

    def step(self, order_quantity: int) -> tuple[int, float, bool]:
        if order_quantity < 0:
            raise ValueError("order_quantity must be non-negative")

        demand = int(self.rng.poisson(self.config.demand_lambda))
        available = self.state + int(order_quantity)
        next_inventory = max(0, available - demand)

        holding = self.config.holding_cost * next_inventory
        stockout = self.config.stockout_cost * max(0, demand - available)
        ordering = self.config.ordering_cost * int(order_quantity)
        total_cost = holding + stockout + ordering

        self.state = min(next_inventory, self.config.max_inventory)
        self.period += 1
        done = self.period >= self.config.periods
        return self.state, float(total_cost), done


def evaluate_policy(
    reorder_point: int,
    order_upto: int,
    *,
    config: InventoryConfig | None = None,
    replications: int = 12,
    seed: int = 42,
) -> float:
    """Evaluate an (s, S) policy using common random numbers.

    A fixed deterministic seed schedule makes repeated objective evaluations
    reproducible and reduces noise when comparing candidate policies.
    """
    if reorder_point < 0:
        raise ValueError("reorder_point must be non-negative")
    if order_upto <= reorder_point:
        return float("inf")
    if replications <= 0:
        raise ValueError("replications must be positive")

    cfg = config or InventoryConfig()
    costs: list[float] = []

    for replication in range(replications):
        env = InventoryEnvironment(cfg, seed=seed + replication)
        state = env.reset()
        total_cost = 0.0

        while True:
            order = max(0, order_upto - state) if state < reorder_point else 0
            state, cost, done = env.step(order)
            total_cost += cost
            if done:
                break

        costs.append(total_cost / cfg.periods)

    return float(np.mean(costs))
