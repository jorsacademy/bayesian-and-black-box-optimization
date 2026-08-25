from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern


@dataclass
class OptimizationResult:
    """Store the main results of a Bayesian Optimization run."""

    best_x: np.ndarray
    best_y: float
    X_observed: np.ndarray
    y_observed: np.ndarray
    history: list[dict]


class GaussianProcessBayesOptimizer:
    """Gaussian Process-based Bayesian Optimization for bounded continuous variables.

    The class solves minimization problems. `GaussianProcessRegressor` is used
    only as the surrogate model; the acquisition function and sequential
    sampling loop implement the Bayesian Optimization logic.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Iterable[Iterable[float]],
        n_initial_points: int = 6,
        acquisition_function: str = "ei",
        xi: float = 0.01,
        kappa: float = 2.0,
        alpha: float = 1e-8,
        random_state: int | None = 42,
        kernel: Kernel | None = None,
        n_restarts_optimizer: int = 3,
    ) -> None:
        self.objective_function = objective_function
        self.bounds = np.asarray(bounds, dtype=float)

        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2).")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Every lower bound must be smaller than its upper bound.")

        self.n_dimensions = self.bounds.shape[0]
        self.n_initial_points = int(n_initial_points)
        if self.n_initial_points < 2:
            raise ValueError("n_initial_points must be at least 2.")

        self.acquisition_function = acquisition_function.lower()
        if self.acquisition_function not in {"ei", "pi", "lcb"}:
            raise ValueError("acquisition_function must be 'ei', 'pi', or 'lcb'.")

        self.xi = float(xi)
        self.kappa = float(kappa)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if kernel is None:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.full(self.n_dimensions, 0.2),
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

        self.X_observed: np.ndarray | None = None
        self.y_observed: np.ndarray | None = None
        self.history: list[dict] = []

    def _to_unit_space(self, X: np.ndarray) -> np.ndarray:
        """Scale decision variables to [0, 1]."""
        X = np.asarray(X, dtype=float)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return (X - lower) / (upper - lower)

    def _from_unit_space(self, U: np.ndarray) -> np.ndarray:
        """Map points from [0, 1] back to the original decision space."""
        U = np.asarray(U, dtype=float)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return lower + U * (upper - lower)

    @staticmethod
    def _to_scalar(value: float | np.ndarray) -> float:
        """Validate that the objective returns one finite numeric value."""
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError("The objective function must return a single numeric value.")
        result = float(array.reshape(-1)[0])
        if not np.isfinite(result):
            raise ValueError("The objective function must return a finite value.")
        return result

    def _acquisition_scores(self, U: np.ndarray) -> np.ndarray:
        """Return acquisition scores in a common 'larger is better' convention."""
        if self.y_observed is None:
            raise RuntimeError("At least one observation is required first.")

        U = np.atleast_2d(np.asarray(U, dtype=float))
        mu, sigma = self.gp.predict(U, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        best_y = float(np.min(self.y_observed))

        if self.acquisition_function == "ei":
            improvement = best_y - mu - self.xi
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0.0)

        if self.acquisition_function == "pi":
            z = (best_y - mu - self.xi) / sigma
            return norm.cdf(z)

        # LCB is naturally minimized. Negating it lets us keep a common
        # acquisition-score convention in which larger values are better.
        lcb = mu - self.kappa * sigma
        return -lcb

    def _select_next_point(
        self,
        n_candidates: int = 5000,
        n_local_starts: int = 12,
    ) -> np.ndarray:
        """Search the acquisition function globally, then refine good candidates."""
        candidates = self.rng.random((int(n_candidates), self.n_dimensions))
        scores = self._acquisition_scores(candidates)

        n_starts = min(int(n_local_starts), len(candidates))
        top_indices = np.argsort(scores)[-n_starts:]

        best_u = candidates[top_indices[-1]].copy()
        best_score = float(scores[top_indices[-1]])
        unit_bounds = [(0.0, 1.0)] * self.n_dimensions

        for index in top_indices:
            u0 = candidates[index]
            result = minimize(
                lambda u: -float(
                    self._acquisition_scores(np.asarray(u).reshape(1, -1))[0]
                ),
                u0,
                bounds=unit_bounds,
                method="L-BFGS-B",
            )

            if result.success:
                score = -float(result.fun)
                if score > best_score:
                    best_score = score
                    best_u = np.clip(result.x, 0.0, 1.0)

        # Avoid evaluating the exact same point again when possible.
        if self.X_observed is not None:
            observed_u = self._to_unit_space(self.X_observed)
            distances = np.linalg.norm(observed_u - best_u, axis=1)
            if np.min(distances) < 1e-8:
                for index in np.argsort(scores)[::-1]:
                    candidate = candidates[index]
                    distance = np.min(np.linalg.norm(observed_u - candidate, axis=1))
                    if distance >= 1e-8:
                        best_u = candidate
                        break

        return self._from_unit_space(best_u)

    def optimize(
        self,
        n_iterations: int = 20,
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run the sequential Bayesian Optimization loop."""
        if self.X_observed is None:
            initial_u = self.rng.random((self.n_initial_points, self.n_dimensions))
            self.X_observed = self._from_unit_space(initial_u)
            self.y_observed = np.array(
                [
                    self._to_scalar(self.objective_function(x.copy()))
                    for x in self.X_observed
                ],
                dtype=float,
            )

        for iteration in range(int(n_iterations)):
            self.gp.fit(self._to_unit_space(self.X_observed), self.y_observed)

            next_x = self._select_next_point()
            next_y = self._to_scalar(self.objective_function(next_x.copy()))

            self.X_observed = np.vstack([self.X_observed, next_x])
            self.y_observed = np.append(self.y_observed, next_y)

            best_index = int(np.argmin(self.y_observed))
            record = {
                "iteration": iteration + 1,
                "x": next_x.copy(),
                "y": float(next_y),
                "best_x": self.X_observed[best_index].copy(),
                "best_y": float(self.y_observed[best_index]),
            }
            self.history.append(record)

            if verbose:
                print(
                    f"Iteration {iteration + 1:02d} | "
                    f"y = {next_y:.6f} | "
                    f"best = {record['best_y']:.6f}"
                )

        best_index = int(np.argmin(self.y_observed))
        return OptimizationResult(
            best_x=self.X_observed[best_index].copy(),
            best_y=float(self.y_observed[best_index]),
            X_observed=self.X_observed.copy(),
            y_observed=self.y_observed.copy(),
            history=list(self.history),
        )
