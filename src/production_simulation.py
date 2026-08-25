from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import heapq

import numpy as np


@dataclass
class SimulationSummary:
    """Economic and service-level summary of a staffing policy."""

    total_cost: float
    staffing_cost: float
    waiting_cost: float
    service_level_penalty: float
    average_wait_minutes: float
    average_wait_by_shift: np.ndarray
    average_customer_count: float


def simulate_one_day(
    staffing: Iterable[int],
    seed: int,
    shift_duration_minutes: int = 240,
    arrival_rates: tuple[float, float, float] = (1.0, 1.6, 1.2),
    mean_service_time: float = 5.0,
    service_time_std: float = 1.0,
) -> dict:
    """Simulate three independent shifts of a multi-server queue for one day."""
    staffing = np.asarray(staffing, dtype=int)

    if staffing.shape != (3,):
        raise ValueError("staffing must contain exactly three values.")
    if np.any(staffing < 1):
        raise ValueError("Each shift must have at least one staff member.")
    if len(arrival_rates) != 3:
        raise ValueError("arrival_rates must contain one value for each of three shifts.")

    rng = np.random.default_rng(seed)

    total_wait = 0.0
    total_customers = 0
    shift_average_waits: list[float] = []

    for shift, staff_count in enumerate(staffing):
        arrivals_per_minute = rng.poisson(
            lam=arrival_rates[shift],
            size=shift_duration_minutes,
        )
        customer_count = int(arrivals_per_minute.sum())

        service_times = np.clip(
            rng.normal(
                loc=mean_service_time,
                scale=service_time_std,
                size=customer_count,
            ),
            1.0,
            12.0,
        )

        # Min-heap storing the next availability time of each server.
        server_available = [0.0] * int(staff_count)
        heapq.heapify(server_available)

        shift_wait = 0.0
        customer_index = 0

        for minute, n_arrivals in enumerate(arrivals_per_minute):
            for _ in range(int(n_arrivals)):
                earliest_available = heapq.heappop(server_available)
                service_start = max(float(minute), earliest_available)
                wait = service_start - float(minute)

                shift_wait += wait
                next_available = service_start + float(service_times[customer_index])
                heapq.heappush(server_available, next_available)
                customer_index += 1

        total_wait += shift_wait
        total_customers += customer_count
        shift_average_waits.append(
            shift_wait / customer_count if customer_count > 0 else 0.0
        )

    return {
        "total_wait": float(total_wait),
        "total_customers": int(total_customers),
        "average_wait_by_shift": np.asarray(shift_average_waits, dtype=float),
    }


def evaluate_staffing_policy(
    staffing: Iterable[int],
    replication_seeds: Iterable[int],
    hourly_staff_cost: float = 180.0,
    waiting_minute_cost: float = 2.0,
    target_shift_wait_minutes: float = 1.5,
    penalty_coefficient: float = 1500.0,
    shift_duration_minutes: int = 240,
    arrival_rates: tuple[float, float, float] = (1.0, 1.6, 1.2),
) -> SimulationSummary:
    """Evaluate a staffing policy over common-random-number replications.

    The same replication seeds should be used for all candidate policies. This
    is a simple Common Random Numbers variance-reduction design: competing
    staffing policies are compared under the same demand and service scenarios.
    """
    staffing = np.asarray(staffing, dtype=int)
    seeds = [int(seed) for seed in replication_seeds]

    if len(seeds) == 0:
        raise ValueError("At least one replication seed is required.")

    runs = [
        simulate_one_day(
            staffing=staffing,
            seed=seed,
            shift_duration_minutes=shift_duration_minutes,
            arrival_rates=arrival_rates,
        )
        for seed in seeds
    ]

    total_waits = np.array([run["total_wait"] for run in runs], dtype=float)
    total_customers = np.array([run["total_customers"] for run in runs], dtype=float)
    shift_waits = np.vstack([run["average_wait_by_shift"] for run in runs])

    shift_hours = shift_duration_minutes / 60.0
    staffing_cost = float(staffing.sum()) * shift_hours * hourly_staff_cost
    waiting_cost = float(total_waits.mean()) * waiting_minute_cost

    average_by_shift = shift_waits.mean(axis=0)
    excess = max(0.0, float(average_by_shift.max()) - target_shift_wait_minutes)
    service_level_penalty = penalty_coefficient * excess**2

    total_cost = staffing_cost + waiting_cost + service_level_penalty

    customer_sum = float(total_customers.sum())
    weighted_average_wait = (
        float(total_waits.sum()) / customer_sum if customer_sum > 0 else 0.0
    )

    return SimulationSummary(
        total_cost=float(total_cost),
        staffing_cost=float(staffing_cost),
        waiting_cost=float(waiting_cost),
        service_level_penalty=float(service_level_penalty),
        average_wait_minutes=float(weighted_average_wait),
        average_wait_by_shift=average_by_shift,
        average_customer_count=float(total_customers.mean()),
    )
