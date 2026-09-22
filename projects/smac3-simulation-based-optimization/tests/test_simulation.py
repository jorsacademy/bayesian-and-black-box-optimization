import pytest

from smac_sim.simulation import ProductionConfig, simulate_policy


def test_simulation_is_reproducible_for_same_seed():
    first = simulate_policy(20, 10, seed=123)
    second = simulate_policy(20, 10, seed=123)
    assert first == pytest.approx(second)


def test_simulation_returns_positive_finite_cost():
    cost = simulate_policy(18, 8, seed=5)
    assert cost > 0
    assert cost < 1_000


def test_invalid_policy_inputs_are_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        simulate_policy(0, 5, seed=0)
    with pytest.raises(ValueError, match="safety_stock"):
        simulate_policy(10, -1, seed=0)


def test_invalid_model_config_is_rejected():
    with pytest.raises(ValueError, match="horizon"):
        simulate_policy(10, 5, seed=0, config=ProductionConfig(horizon=0))
    with pytest.raises(ValueError, match="yield_rate"):
        simulate_policy(10, 5, seed=0, config=ProductionConfig(yield_rate=0))
