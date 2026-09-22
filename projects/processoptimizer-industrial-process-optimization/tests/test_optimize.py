import math
import pytest

from process_opt.optimize import random_search, run_processoptimizer


def _assert_valid(result):
    t, r, a = result.setting
    assert 60.0 <= t <= 100.0
    assert 1.0 <= r <= 8.0
    assert 200.0 <= a <= 800.0
    assert math.isfinite(result.validation_loss)


def test_processoptimizer_runs_real_ask_tell_loop():
    result = run_processoptimizer(budget=6, seed=3)
    _assert_valid(result)
    assert result.evaluations == 6
    assert result.method == "processoptimizer-gp"


def test_processoptimizer_is_reproducible():
    a = run_processoptimizer(budget=5, seed=4)
    b = run_processoptimizer(budget=5, seed=4)
    assert a == pytest.approx(b)


def test_random_search_is_reproducible_and_budget_matched():
    a = random_search(budget=6, seed=5)
    b = random_search(budget=6, seed=5)
    _assert_valid(a)
    assert a == pytest.approx(b)
    assert a.evaluations == 6


def test_invalid_budgets_rejected():
    with pytest.raises(ValueError):
        run_processoptimizer(budget=3)
    with pytest.raises(ValueError):
        random_search(budget=0)
