import pytest

from smac_sim.optimization import (
    evaluate_configuration,
    make_configspace,
    objective,
    random_search,
    run_smac,
)


def test_configspace_and_objective_contract():
    cs = make_configspace(seed=3)
    config = cs.get_default_configuration()
    assert 5 <= int(config["batch_size"]) <= 40
    assert 0 <= int(config["safety_stock"]) <= 30
    assert objective(config, seed=11) > 0
    assert evaluate_configuration(config) > 0


def test_random_search_is_reproducible():
    first = random_search(n_trials=6, seed=4)
    second = random_search(n_trials=6, seed=4)
    assert first == second


def test_smac_runs_end_to_end():
    result = run_smac(n_trials=8, seed=2)
    assert 5 <= result.batch_size <= 40
    assert 0 <= result.safety_stock <= 30
    assert result.validation_cost > 0


def test_invalid_budgets_are_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        run_smac(n_trials=1)
    with pytest.raises(ValueError, match="positive"):
        random_search(n_trials=0)
