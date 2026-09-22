from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 90
    demand_rate: float = 8.0
    initial_inventory: int = 20
    holding_cost: float = 0.6
    shortage_cost: float = 5.0
    order_cost: float = 1.2

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        if min(self.holding_cost, self.shortage_cost, self.order_cost) < 0:
            raise ValueError("costs must be non-negative")


def simulate_policy(
    reorder_point: int,
    order_quantity: int,
    *,
    seed: int,
    config: InventoryConfig | None = None,
) -> float:
    """Returns average period cost for an (r, Q) lost-sales inventory policy."""
    cfg = config or InventoryConfig()
    cfg.validate()
    if reorder_point < 0:
        raise ValueError("reorder_point must be non-negative")
    if order_quantity <= 0:
        raise ValueError("order_quantity must be positive")

    rng = np.random.default_rng(seed)
    inventory = cfg.initial_inventory
    total_cost = 0.0

    for _ in range(cfg.horizon):
        if inventory <= reorder_point:
            inventory += order_quantity
            total_cost += cfg.order_cost * order_quantity

        demand = int(rng.poisson(cfg.demand_rate))
        sales = min(inventory, demand)
        lost_sales = demand - sales
        inventory -= sales

        total_cost += cfg.holding_cost * inventory
        total_cost += cfg.shortage_cost * lost_sales

    return total_cost / cfg.horizon


def mean_policy_cost(
    reorder_point: int,
    order_quantity: int,
    *,
    seeds: tuple[int, ...] = (11, 23, 37, 53),
    config: InventoryConfig | None = None,
) -> float:
    if not seeds:
        raise ValueError("at least one seed is required")
    costs = [
        simulate_policy(
            reorder_point,
            order_quantity,
            seed=seed,
            config=config,
        )
        for seed in seeds
    ]
    return float(np.mean(costs))
