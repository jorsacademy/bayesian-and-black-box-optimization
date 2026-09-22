import pytest

from nevergrad_policy.optimization import compare_optimizers, make_parametrization, optimize_policy


def test_parametrization_contains_expected_integer_policy_variables():
    parametrization = make_parametrization(seed=7)
    assert set(parametrization.kwargs) == {"reorder_point", "order_quantity"}


def test_nevergrad_runs_end_to_end():
    result = optimize_policy("NGOpt", budget=8, seed=3, evaluation_seeds=(1, 2))
    assert result.optimizer == "NGOpt"
    assert 0 <= result.reorder_point <= 30
    assert 5 <= result.order_quantity <= 45
    assert result.cost > 0


def test_optimizer_comparison_is_sorted():
    results = compare_optimizers(
        optimizers=("NGOpt", "RandomSearch"),
        budget=6,
        seed=2,
        evaluation_seeds=(4, 5),
    )
    assert len(results) == 2
    assert results[0].cost <= results[1].cost


def test_invalid_optimizer_and_budget_are_rejected():
    with pytest.raises(ValueError, match="budget"):
        optimize_policy("NGOpt", budget=0)
    with pytest.raises(ValueError, match="unknown"):
        optimize_policy("definitely-not-an-optimizer", budget=2)
