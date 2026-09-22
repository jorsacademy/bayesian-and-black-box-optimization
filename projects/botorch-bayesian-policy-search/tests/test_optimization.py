import pytest
import torch

from botorch_policy.optimization import decode_policy, policy_score, run_botorch_search, run_sobol_search


def test_decode_policy_maps_unit_square_to_valid_policy():
    policy = decode_policy(torch.tensor([0.0, 1.0], dtype=torch.double))
    assert policy.reorder_point == pytest.approx(2.0)
    assert policy.order_up_to == pytest.approx(32.0)
    with pytest.raises(ValueError):
        decode_policy(torch.tensor([1.2, 0.5], dtype=torch.double))
    with pytest.raises(ValueError):
        decode_policy(torch.tensor([0.5], dtype=torch.double))


def test_policy_score_is_negative_cost():
    score = policy_score(torch.tensor([0.5, 0.5], dtype=torch.double))
    assert score < 0


def test_sobol_search_is_reproducible():
    first = run_sobol_search(budget=5, seed=5)
    second = run_sobol_search(budget=5, seed=5)
    assert first == second
    assert first.evaluations == 5
    with pytest.raises(ValueError):
        run_sobol_search(budget=0)


def test_botorch_runs_end_to_end():
    result = run_botorch_search(n_initial=4, n_iterations=1, seed=3)
    assert result.cost > 0
    assert result.evaluations == 5
    assert result.policy.order_up_to > result.policy.reorder_point
    with pytest.raises(ValueError):
        run_botorch_search(n_initial=2)
    with pytest.raises(ValueError):
        run_botorch_search(n_iterations=-1)
