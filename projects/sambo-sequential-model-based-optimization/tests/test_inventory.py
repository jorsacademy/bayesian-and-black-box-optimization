import math

import pytest

from sambo_inventory.inventory import InventoryConfig, InventoryEnvironment, evaluate_policy


def test_environment_is_reproducible_with_same_seed():
    cfg = InventoryConfig(periods=3)
    a = InventoryEnvironment(cfg, seed=7)
    b = InventoryEnvironment(cfg, seed=7)

    trajectory_a = [a.step(5) for _ in range(3)]
    trajectory_b = [b.step(5) for _ in range(3)]

    assert trajectory_a == trajectory_b
    assert trajectory_a[-1][2] is True


def test_negative_order_is_rejected():
    env = InventoryEnvironment(seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        env.step(-1)


def test_policy_evaluation_is_deterministic():
    kwargs = dict(reorder_point=12, order_upto=35, replications=4, seed=123)
    first = evaluate_policy(**kwargs)
    second = evaluate_policy(**kwargs)

    assert math.isfinite(first)
    assert first == second


def test_invalid_policy_is_penalized():
    assert evaluate_policy(20, 20, replications=2) == float("inf")


def test_replications_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        evaluate_policy(10, 30, replications=0)
