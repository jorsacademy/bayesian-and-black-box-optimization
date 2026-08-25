from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern


@dataclass
class DiscreteOptimizationResult:
    """Store the main results of candidate-set Bayesian Optimization."""

    best_x: np.ndarray
    best_y: float
    observed_indices: list[int]
    y_observed: np.ndarray
    history: list[dict]


class DiscreteGaussianProcessBayesOptimizer:
    """Gaussian Process-based Bayesian Optimization over a finite candidate set.

    This formulation is useful for educational industrial-engineering examples
    involving integer staffing levels, batch sizes, or a finite set of process
    configurations. The optimizer does not evaluate every candidate: it uses a
    Gaussian Process surrogate and Expected Improvement to decide which
    unevaluated candidate should be tested next.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        candidate_points: np.ndarray,
        n_initial_points: int = 8,
        xi: float = 0.01,
        alpha: float = 1e-8,
        random_state: int | None = 42,
        kernel: Kernel | None = None,
        n_restarts_optimizer: int = 2,
    ) -> None:
        self.objective_function = objective_function
        self.candidate_points = np.asarray(candidate_points, dtype=float)

        if self.candidate_points.ndim != 2:
            raise ValueError("candidate_points must be a two-dimensional array.")

        self.n_candidates, self.n_dimensions = self.candidate_points.shape
        self.n_initial_points = int(n_initial_points)
        if not 2 <= self.n_initial_points < self.n_candidates:
            raise ValueError(
                "n_initial_points must be at least 2 and smaller than the candidate count."
            )

        self.xi = float(xi)
        self.rng = np.random.default_rng(random_state)

        self.lower = self.candidate_points.min(axis=0)
        self.upper = self.candidate_points.max(axis=0)
        self.range_ = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)

        if kernel is None:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.full(self.n_dimensions, 0.3),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=int(n_restarts_optimizer),
            random_state=random_state,
        )

        self.observed_indices: list[int] = []
        self.y_observed: list[float] = []
        self.history: list[dict] = []

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Scale candidate points to [0, 1]."""
        return (np.asarray(X, dtype=float) - self.lower) / self.range_

    @staticmethod
    def _to_scalar(value: float | np.ndarray) -> float:
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError("The objective function must return a single numeric value.")
        result = float(array.reshape(-1)[0])
        if not np.isfinite(result):
            raise ValueError("The objective function must return a finite value.")
        return result

    def _expected_improvement(self, candidates: np.ndarray) -> np.ndarray:
        """Compute Expected Improvement for minimization."""
        mu, sigma = self.gp.predict(self._scale(candidates), return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        best_y = float(np.min(self.y_observed))
        improvement = best_y - mu - self.xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)

    def optimize(
        self,
        n_iterations: int = 25,
        verbose: bool = False,
    ) -> DiscreteOptimizationResult:
        """Run candidate-set Bayesian Optimization."""
        if not self.observed_indices:
            initial = self.rng.choice(
                self.n_candidates,
                size=self.n_initial_points,
                replace=False,
            )
            for index in initial:
                index = int(index)
                y = self._to_scalar(
                    self.objective_function(self.candidate_points[index].copy())
                )
                self.observed_indices.append(index)
                self.y_observed.append(y)

        for iteration in range(int(n_iterations)):
            X_observed = self.candidate_points[self.observed_indices]
            y_observed = np.asarray(self.y_observed, dtype=float)
            self.gp.fit(self._scale(X_observed), y_observed)

            untested_mask = np.ones(self.n_candidates, dtype=bool)
            untested_mask[self.observed_indices] = False
            untested_indices = np.flatnonzero(untested_mask)
            if len(untested_indices) == 0:
                break

            untested_candidates = self.candidate_points[untested_indices]
            ei = self._expected_improvement(untested_candidates)

            local_index = int(np.argmax(ei))
            next_index = int(untested_indices[local_index])
            next_x = self.candidate_points[next_index].copy()
            next_y = self._to_scalar(self.objective_function(next_x))

            self.observed_indices.append(next_index)
            self.y_observed.append(next_y)

            best_local_index = int(np.argmin(self.y_observed))
            best_candidate_index = self.observed_indices[best_local_index]
            record = {
                "iteration": iteration + 1,
                "x": next_x.copy(),
                "y": float(next_y),
                "best_x": self.candidate_points[best_candidate_index].copy(),
                "best_y": float(self.y_observed[best_local_index]),
            }
            self.history.append(record)

            if verbose:
                print(
                    f"Iteration {iteration + 1:02d} | "
                    f"x = {next_x.astype(int).tolist()} | "
                    f"y = {next_y:.3f} | "
                    f"best = {record['best_y']:.3f}"
                )

        best_local_index = int(np.argmin(self.y_observed))
        best_candidate_index = self.observed_indices[best_local_index]
        return DiscreteOptimizationResult(
            best_x=self.candidate_points[best_candidate_index].copy(),
            best_y=float(self.y_observed[best_local_index]),
            observed_indices=list(self.observed_indices),
            y_observed=np.asarray(self.y_observed, dtype=float),
            history=list(self.history),
        )
