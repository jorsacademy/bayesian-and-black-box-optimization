from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Integer
from smac import HyperparameterOptimizationFacade, Scenario

from .simulation import ProductionConfig, simulate_policy


@dataclass(frozen=True)
class OptimizationResult:
    batch_size: int
    safety_stock: int
    validation_cost: float


def make_configspace(seed: int = 0) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    cs.add(
        [
            Integer("batch_size", (5, 40), default=20),
            Integer("safety_stock", (0, 30), default=10),
        ]
    )
    return cs


def objective(
    config: Configuration,
    seed: int = 0,
    production_config: ProductionConfig | None = None,
) -> float:
    return simulate_policy(
        int(config["batch_size"]),
        int(config["safety_stock"]),
        seed=seed,
        config=production_config,
    )


def evaluate_configuration(
    config: Configuration,
    *,
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105),
    production_config: ProductionConfig | None = None,
) -> float:
    costs = [objective(config, seed=seed, production_config=production_config) for seed in seeds]
    return float(np.mean(costs))


def run_smac(
    *,
    n_trials: int = 20,
    seed: int = 0,
    production_config: ProductionConfig | None = None,
) -> OptimizationResult:
    if n_trials < 2:
        raise ValueError("n_trials must be at least 2")

    configspace = make_configspace(seed)
    with TemporaryDirectory(prefix="smac3-production-") as output_dir:
        scenario = Scenario(
            configspace,
            deterministic=False,
            n_trials=n_trials,
            seed=seed,
            output_directory=Path(output_dir),
        )
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1,
        )
        optimizer = HyperparameterOptimizationFacade(
            scenario,
            lambda config, seed=0: objective(
                config,
                seed=seed,
                production_config=production_config,
            ),
            intensifier=intensifier,
            overwrite=True,
            logging_level=False,
        )
        incumbent = optimizer.optimize()

    return OptimizationResult(
        batch_size=int(incumbent["batch_size"]),
        safety_stock=int(incumbent["safety_stock"]),
        validation_cost=evaluate_configuration(
            incumbent,
            production_config=production_config,
        ),
    )


def random_search(
    *,
    n_trials: int = 20,
    seed: int = 0,
    production_config: ProductionConfig | None = None,
) -> OptimizationResult:
    if n_trials < 1:
        raise ValueError("n_trials must be positive")

    configspace = make_configspace(seed)
    best_config: Configuration | None = None
    best_cost = float("inf")

    for _ in range(n_trials):
        config = configspace.sample_configuration()
        cost = objective(config, seed=seed, production_config=production_config)
        if cost < best_cost:
            best_config = config
            best_cost = cost

    assert best_config is not None
    return OptimizationResult(
        batch_size=int(best_config["batch_size"]),
        safety_stock=int(best_config["safety_stock"]),
        validation_cost=evaluate_configuration(
            best_config,
            production_config=production_config,
        ),
    )
