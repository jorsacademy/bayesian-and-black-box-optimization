from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import ProcessOptimizer as po

from .process import ProcessConfig, evaluate_setting

SEARCH_SEEDS = (101, 202, 303, 404)
VALIDATION_SEEDS = (1001, 1002, 1003, 1004, 1005, 1006)


@dataclass(frozen=True)
class OptimizationResult:
    setting: tuple[float, float, float]
    validation_loss: float
    evaluations: int
    method: str


def _space(config: ProcessConfig) -> po.Space:
    return po.Space([
        list(config.temperature_bounds),
        list(config.residence_bounds),
        list(config.agitation_bounds),
    ])


def run_processoptimizer(
    *,
    budget: int = 12,
    seed: int = 7,
    config: ProcessConfig | None = None,
) -> OptimizationResult:
    if budget < 4:
        raise ValueError("budget must be at least 4")
    cfg = config or ProcessConfig()
    optimizer = po.Optimizer(
        _space(cfg),
        base_estimator="GP",
        n_initial_points=4,
        random_state=seed,
    )
    evaluated: list[tuple[tuple[float, float, float], float]] = []
    for _ in range(budget):
        point = tuple(float(value) for value in optimizer.ask())
        loss = evaluate_setting(point, seeds=SEARCH_SEEDS, config=cfg)
        optimizer.tell(list(point), loss)
        evaluated.append((point, loss))

    best_setting = min(evaluated, key=lambda item: item[1])[0]
    validation = evaluate_setting(best_setting, seeds=VALIDATION_SEEDS, config=cfg)
    return OptimizationResult(best_setting, validation, budget, "processoptimizer-gp")


def random_search(
    *,
    budget: int = 12,
    seed: int = 7,
    config: ProcessConfig | None = None,
) -> OptimizationResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    cfg = config or ProcessConfig()
    rng = np.random.default_rng(seed)
    bounds = (cfg.temperature_bounds, cfg.residence_bounds, cfg.agitation_bounds)
    candidates: list[tuple[tuple[float, float, float], float]] = []
    for _ in range(budget):
        point = tuple(float(rng.uniform(low, high)) for low, high in bounds)
        loss = evaluate_setting(point, seeds=SEARCH_SEEDS, config=cfg)
        candidates.append((point, loss))
    best_setting = min(candidates, key=lambda item: item[1])[0]
    validation = evaluate_setting(best_setting, seeds=VALIDATION_SEEDS, config=cfg)
    return OptimizationResult(best_setting, validation, budget, "random-search")
