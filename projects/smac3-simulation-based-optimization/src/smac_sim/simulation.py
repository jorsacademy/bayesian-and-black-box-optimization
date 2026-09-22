from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProductionConfig:
    horizon: int = 60
    demand_rate: float = 12.0
    yield_rate: float = 0.9
    holding_cost: float = 0.8
    shortage_cost: float = 4.0
    setup_cost: float = 8.0
    unit_cost: float = 0.5

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if not 0 < self.yield_rate <= 1:
            raise ValueError("yield_rate must be in (0, 1]")
        for name in ("holding_cost", "shortage_cost", "setup_cost", "unit_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def simulate_policy(
    batch_size: int,
    safety_stock: int,
    *,
    seed: int,
    config: ProductionConfig | None = None,
) -> float:
    """Returns average per-period cost of a stochastic make-to-stock policy."""
    cfg = config or ProductionConfig()
    cfg.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if safety_stock < 0:
        raise ValueError("safety_stock must be non-negative")

    rng = np.random.default_rng(seed)
    inventory = int(safety_stock)
    total_cost = 0.0

    for _ in range(cfg.horizon):
        if inventory <= safety_stock:
            good_units = int(rng.binomial(batch_size, cfg.yield_rate))
            inventory += good_units
            total_cost += cfg.setup_cost + cfg.unit_cost * batch_size

        demand = int(rng.poisson(cfg.demand_rate))
        fulfilled = min(inventory, demand)
        lost_sales = demand - fulfilled
        inventory -= fulfilled

        total_cost += cfg.holding_cost * inventory
        total_cost += cfg.shortage_cost * lost_sales

    return float(total_cost / cfg.horizon)
