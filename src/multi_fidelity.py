from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


Fidelity = Literal["low", "high"]


@dataclass
class MultiFidelityResult:
    """Summary of an educational two-fidelity optimization run."""

    best_x: np.ndarray
    best_y: float
    total_cost: float
    n_low_fidelity: int
    n_high_fidelity: int
    history: list[dict]


class TwoFidelitySurrogate:
    """Autoregressive two-fidelity Gaussian-process surrogate.

    The high-fidelity response is approximated as

        f_high(x) = rho * f_low(x) + delta(x)

    where both the low-fidelity process and the discrepancy delta are modeled
    with Gaussian processes. This is intentionally compact teaching code, not
    a replacement for a production multi-fidelity GP implementation.
    """

    def __init__(self, n_dimensions: int, random_state: int | None = 42) -> None:
        kernel_low = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(n_dimensions),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        kernel_delta = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(n_dimensions),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        self.gp_low = GaussianProcessRegressor(
            kernel=kernel_low,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=random_state,
        )
        self.gp_delta = GaussianProcessRegressor(
            kernel=kernel_delta,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=random_state,
        )
        self.rho = 1.0
        self._is_fit = False

    def fit(
        self,
        X_low: np.ndarray,
        y_low: np.ndarray,
        X_high: np.ndarray,
        y_high: np.ndarray,
    ) -> None:
        X_low = np.atleast_2d(np.asarray(X_low, dtype=float))
        y_low = np.asarray(y_low, dtype=float).reshape(-1)
        X_high = np.atleast_2d(np.asarray(X_high, dtype=float))
        y_high = np.asarray(y_high, dtype=float).reshape(-1)

        if len(X_low) < 2 or len(X_high) < 2:
            raise ValueError("At least two observations are required at each fidelity.")

        self.gp_low.fit(X_low, y_low)
        low_at_high = self.gp_low.predict(X_high)

        denominator = float(np.dot(low_at_high, low_at_high))
        if denominator > 1e-12:
            self.rho = float(np.dot(low_at_high, y_high) / denominator)
        else:
            self.rho = 1.0

        discrepancy = y_high - self.rho * low_at_high
        self.gp_delta.fit(X_high, discrepancy)
        self._is_fit = True

    def predict_low(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fit:
            raise RuntimeError("Fit the surrogate before prediction.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.gp_low.predict(X, return_std=True)

    def predict_high(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fit:
            raise RuntimeError("Fit the surrogate before prediction.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mu_low, sigma_low = self.gp_low.predict(X, return_std=True)
        mu_delta, sigma_delta = self.gp_delta.predict(X, return_std=True)

        mu_high = self.rho * mu_low + mu_delta
        sigma_high = np.sqrt(
            np.maximum((self.rho * sigma_low) ** 2 + sigma_delta**2, 1e-16)
        )
        return mu_high, sigma_high


class CostAwareTwoFidelityOptimizer:
    """Educational cost-aware optimizer for an expensive high-fidelity objective.

    The optimizer uses expected improvement per unit cost for high-fidelity
    evaluations and a low-fidelity uncertainty-reduction proxy per unit cost
    for cheap evaluations. The fidelity policy is deliberately transparent so
    students can inspect the exploration/cost trade-off.
    """

    def __init__(
        self,
        low_fidelity_function: Callable[[np.ndarray], float],
        high_fidelity_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        low_cost: float = 1.0,
        high_cost: float = 10.0,
        random_state: int | None = 42,
    ) -> None:
        self.low_fidelity_function = low_fidelity_function
        self.high_fidelity_function = high_fidelity_function
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2).")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Every lower bound must be smaller than its upper bound.")
        if low_cost <= 0 or high_cost <= 0:
            raise ValueError("Evaluation costs must be positive.")

        self.low_cost = float(low_cost)
        self.high_cost = float(high_cost)
        self.n_dimensions = self.bounds.shape[0]
        self.rng = np.random.default_rng(random_state)
        self.surrogate = TwoFidelitySurrogate(
            self.n_dimensions, random_state=random_state
        )

        self.X_low: list[np.ndarray] = []
        self.y_low: list[float] = []
        self.X_high: list[np.ndarray] = []
        self.y_high: list[float] = []
        self.total_cost = 0.0
        self.history: list[dict] = []

    def _sample_uniform(self, n: int) -> np.ndarray:
        unit = self.rng.random((n, self.n_dimensions))
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return lower + unit * (upper - lower)

    @staticmethod
    def _scalar(value: float | np.ndarray) -> float:
        result = float(np.asarray(value, dtype=float).reshape(-1)[0])
        if not np.isfinite(result):
            raise ValueError("Objective functions must return finite values.")
        return result

    def _evaluate(self, x: np.ndarray, fidelity: Fidelity) -> float:
        x = np.asarray(x, dtype=float).copy()
        if fidelity == "low":
            y = self._scalar(self.low_fidelity_function(x))
            self.X_low.append(x)
            self.y_low.append(y)
            self.total_cost += self.low_cost
            return y

        y = self._scalar(self.high_fidelity_function(x))
        self.X_high.append(x)
        self.y_high.append(y)
        self.total_cost += self.high_cost
        return y

    @staticmethod
    def _expected_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
        improvement = best_y - np.asarray(mu, dtype=float) - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)

    def initialize(self, n_low: int = 8, n_high: int = 3) -> None:
        if n_low < 2 or n_high < 2:
            raise ValueError("Use at least two initial observations per fidelity.")
        if n_high > n_low:
            raise ValueError("n_high cannot exceed n_low during initialization.")

        X = self._sample_uniform(n_low)
        for x in X:
            self._evaluate(x, "low")

        high_indices = np.linspace(0, n_low - 1, n_high, dtype=int)
        for index in high_indices:
            self._evaluate(X[index], "high")

    def _fit(self) -> None:
        self.surrogate.fit(
            np.asarray(self.X_low),
            np.asarray(self.y_low),
            np.asarray(self.X_high),
            np.asarray(self.y_high),
        )

    def optimize(
        self,
        budget: float = 100.0,
        n_candidates: int = 3000,
        low_exploration_weight: float = 0.25,
        verbose: bool = False,
    ) -> MultiFidelityResult:
        if not self.X_low:
            self.initialize()

        minimum_next_cost = min(self.low_cost, self.low_cost + self.high_cost)

        while self.total_cost + minimum_next_cost <= budget:
            self._fit()
            candidates = self._sample_uniform(n_candidates)

            mu_high, sigma_high = self.surrogate.predict_high(candidates)
            _, sigma_low = self.surrogate.predict_low(candidates)
            best_high = float(np.min(self.y_high))

            high_utility = (
                self._expected_improvement(mu_high, sigma_high, best_high)
                / (self.high_cost + self.low_cost)
            )
            low_utility = (
                low_exploration_weight
                * abs(self.surrogate.rho)
                * sigma_low
                / self.low_cost
            )

            best_high_index = int(np.argmax(high_utility))
            best_low_index = int(np.argmax(low_utility))

            can_afford_high = (
                self.total_cost + self.low_cost + self.high_cost <= budget
            )
            choose_high = can_afford_high and (
                high_utility[best_high_index] >= low_utility[best_low_index]
            )

            if choose_high:
                x = candidates[best_high_index]
                low_y = self._evaluate(x, "low")
                high_y = self._evaluate(x, "high")
                record = {
                    "fidelity": "high",
                    "x": x.copy(),
                    "low_y": low_y,
                    "high_y": high_y,
                    "total_cost": self.total_cost,
                }
            else:
                if self.total_cost + self.low_cost > budget:
                    break
                x = candidates[best_low_index]
                low_y = self._evaluate(x, "low")
                record = {
                    "fidelity": "low",
                    "x": x.copy(),
                    "low_y": low_y,
                    "total_cost": self.total_cost,
                }

            self.history.append(record)
            if verbose:
                best = float(np.min(self.y_high))
                print(
                    f"{record['fidelity']:>4} | "
                    f"cost={self.total_cost:6.1f} | "
                    f"best_high={best:.6f}"
                )

        best_index = int(np.argmin(self.y_high))
        return MultiFidelityResult(
            best_x=np.asarray(self.X_high[best_index]).copy(),
            best_y=float(self.y_high[best_index]),
            total_cost=float(self.total_cost),
            n_low_fidelity=len(self.y_low),
            n_high_fidelity=len(self.y_high),
            history=list(self.history),
        )


def high_fidelity_demo(x: np.ndarray) -> float:
    """Synthetic expensive target used by the runnable example."""
    x0 = float(x[0])
    return (x0 - 0.72) ** 2 + 0.08 * np.sin(12.0 * x0)


def low_fidelity_demo(x: np.ndarray) -> float:
    """Cheaper biased approximation of high_fidelity_demo."""
    x0 = float(x[0])
    return 0.85 * (x0 - 0.68) ** 2 + 0.06 * np.sin(11.0 * x0) + 0.025


if __name__ == "__main__":
    optimizer = CostAwareTwoFidelityOptimizer(
        low_fidelity_function=low_fidelity_demo,
        high_fidelity_function=high_fidelity_demo,
        bounds=np.array([[0.0, 1.0]]),
        low_cost=1.0,
        high_cost=12.0,
        random_state=42,
    )
    result = optimizer.optimize(budget=90.0, verbose=True)
    print("\nBest high-fidelity point:", result.best_x)
    print("Best high-fidelity value:", result.best_y)
    print("Total evaluation cost:", result.total_cost)
    print("Low/high evaluations:", result.n_low_fidelity, result.n_high_fidelity)
