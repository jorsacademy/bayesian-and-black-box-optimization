from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 60
    demand_rate: float = 8.0
    initial_inventory: int = 16
    holding_cost: float = 1.0
    lost_sales_cost: float = 8.0
    order_cost: float = 2.0

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        if min(self.holding_cost, self.lost_sales_cost, self.order_cost) < 0:
            raise ValueError("cost parameters must be non-negative")


@dataclass(frozen=True)
class Policy:
    reorder_point: float
    order_up_to: float

    def validate(self) -> None:
        if self.reorder_point < 0:
            raise ValueError("reorder_point must be non-negative")
        if self.order_up_to <= self.reorder_point:
            raise ValueError("order_up_to must exceed reorder_point")


def simulate_policy(
    policy: Policy,
    config: InventoryConfig = InventoryConfig(),
    seed: int = 0,
) -> float:
    """Returns average per-period cost for an immediate-replenishment (s, S) policy."""
    config.validate()
    policy.validate()
    rng = np.random.default_rng(seed)
    inventory = float(config.initial_inventory)
    total_cost = 0.0

    for _ in range(config.horizon):
        if inventory <= policy.reorder_point:
            quantity = max(0.0, policy.order_up_to - inventory)
            inventory += quantity
            total_cost += config.order_cost * quantity

        demand = float(rng.poisson(config.demand_rate))
        sales = min(inventory, demand)
        lost_sales = demand - sales
        inventory -= sales
        total_cost += config.holding_cost * inventory
        total_cost += config.lost_sales_cost * lost_sales

    return total_cost / config.horizon


def evaluate_policy(
    policy: Policy,
    config: InventoryConfig = InventoryConfig(),
    seeds: tuple[int, ...] = (11, 23, 37, 53),
) -> float:
    """Evaluates a policy with common random numbers and returns mean cost."""
    if not seeds:
        raise ValueError("at least one seed is required")
    return float(np.mean([simulate_policy(policy, config, seed) for seed in seeds]))
