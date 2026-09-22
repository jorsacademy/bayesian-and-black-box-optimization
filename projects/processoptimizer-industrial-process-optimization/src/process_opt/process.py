from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProcessConfig:
    temperature_bounds: tuple[float, float] = (60.0, 100.0)
    residence_bounds: tuple[float, float] = (1.0, 8.0)
    agitation_bounds: tuple[float, float] = (200.0, 800.0)
    noise_std: float = 0.7

    def __post_init__(self) -> None:
        for low, high in (
            self.temperature_bounds,
            self.residence_bounds,
            self.agitation_bounds,
        ):
            if low >= high:
                raise ValueError("each process bound must satisfy low < high")
        if self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative")


def _validate_setting(setting: tuple[float, float, float], config: ProcessConfig) -> None:
    if len(setting) != 3:
        raise ValueError("setting must contain temperature, residence time, and agitation")
    for value, (low, high) in zip(
        setting,
        (config.temperature_bounds, config.residence_bounds, config.agitation_bounds),
    ):
        if not low <= value <= high:
            raise ValueError("process setting is outside configured bounds")


def simulate_batch(
    setting: tuple[float, float, float],
    *,
    seed: int,
    config: ProcessConfig | None = None,
) -> float:
    """Return a noisy scalar loss for one experimental batch."""
    cfg = config or ProcessConfig()
    _validate_setting(setting, cfg)
    temperature, residence, agitation = setting

    yield_penalty = ((temperature - 82.0) / 8.0) ** 2 + ((residence - 4.6) / 1.8) ** 2
    mixing_penalty = ((agitation - 520.0) / 180.0) ** 2
    interaction = 0.45 * ((temperature - 82.0) / 10.0) * ((residence - 4.6) / 2.0)
    energy = 0.015 * (temperature - 60.0) + 0.0007 * (agitation - 200.0)
    deterministic_loss = 4.0 + 2.2 * yield_penalty + 1.4 * mixing_penalty + interaction + energy

    rng = np.random.default_rng(seed)
    return float(deterministic_loss + rng.normal(0.0, cfg.noise_std))


def evaluate_setting(
    setting: tuple[float, float, float],
    *,
    seeds: tuple[int, ...] = (101, 202, 303, 404),
    config: ProcessConfig | None = None,
) -> float:
    if not seeds:
        raise ValueError("at least one seed is required")
    values = [simulate_batch(setting, seed=seed, config=config) for seed in seeds]
    return float(np.mean(values))
