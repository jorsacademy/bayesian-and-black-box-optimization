from types import SimpleNamespace

import numpy as np
import pytest

from sambo_inventory.inventory import InventoryConfig
from sambo_inventory import optimize as optimize_module


def test_optimize_inventory_policy_uses_sambo_smbo(monkeypatch):
    captured = {}

    def fake_minimize(fun, bounds, constraints, method, max_iter):
        captured.update(
            bounds=bounds,
            constraints=constraints,
            method=method,
            max_iter=max_iter,
        )
        candidate = np.array([10.0, 32.0])
        assert constraints(candidate)
        value = fun(candidate)
        assert value > 0
        return SimpleNamespace(x=candidate, fun=value, nfev=9)

    monkeypatch.setattr(optimize_module.sambo, "minimize", fake_minimize)

    result = optimize_module.optimize_inventory_policy(
        config=InventoryConfig(periods=5),
        max_iter=7,
        replications=2,
        seed=11,
    )

    assert captured["method"] == "smbo"
    assert captured["max_iter"] == 7
    assert result.reorder_point == 10
    assert result.order_upto == 32
    assert result.evaluations == 9


def test_optimize_rejects_nonpositive_iteration_budget():
    with pytest.raises(ValueError, match="positive"):
        optimize_module.optimize_inventory_policy(max_iter=0)
