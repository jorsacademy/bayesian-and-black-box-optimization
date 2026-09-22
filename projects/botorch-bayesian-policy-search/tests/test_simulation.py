import pytest

from botorch_policy.simulation import InventoryConfig, Policy, evaluate_policy, simulate_policy


def test_simulation_is_reproducible_for_fixed_seed():
    policy = Policy(8.0, 24.0)
    first = simulate_policy(policy, seed=123)
    second = simulate_policy(policy, seed=123)
    assert first == pytest.approx(second)
    assert first > 0


def test_policy_and_config_validation():
    with pytest.raises(ValueError):
        simulate_policy(Policy(-1.0, 5.0))
    with pytest.raises(ValueError):
        simulate_policy(Policy(5.0, 5.0))
    with pytest.raises(ValueError):
        simulate_policy(Policy(1.0, 5.0), InventoryConfig(horizon=0))
    with pytest.raises(ValueError):
        evaluate_policy(Policy(1.0, 5.0), seeds=())


def test_common_random_number_evaluation_is_stable():
    policy = Policy(6.0, 20.0)
    assert evaluate_policy(policy) == pytest.approx(evaluate_policy(policy))
