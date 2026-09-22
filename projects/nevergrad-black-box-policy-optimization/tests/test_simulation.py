import pytest

from nevergrad_policy.simulation import InventoryConfig, mean_policy_cost, simulate_policy


def test_simulation_is_reproducible_for_fixed_seed():
    first = simulate_policy(8, 18, seed=123)
    second = simulate_policy(8, 18, seed=123)
    assert first == pytest.approx(second)
    assert first > 0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        simulate_policy(-1, 10, seed=0)
    with pytest.raises(ValueError):
        simulate_policy(1, 0, seed=0)
    with pytest.raises(ValueError):
        mean_policy_cost(1, 10, seeds=())
    with pytest.raises(ValueError):
        simulate_policy(1, 10, seed=0, config=InventoryConfig(horizon=0))


def test_mean_policy_cost_uses_multiple_seeds():
    cost = mean_policy_cost(6, 20, seeds=(1, 2, 3))
    assert isinstance(cost, float)
    assert cost > 0
