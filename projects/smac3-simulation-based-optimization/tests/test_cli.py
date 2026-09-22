from smac_sim import cli
from smac_sim.optimization import OptimizationResult


def test_cli_prints_both_optimizers(monkeypatch, capsys):
    result = OptimizationResult(batch_size=20, safety_stock=10, validation_cost=12.345)
    monkeypatch.setattr(cli, "run_smac", lambda **_: result)
    monkeypatch.setattr(cli, "random_search", lambda **_: result)

    cli.main()
    output = capsys.readouterr().out
    assert "SMAC3" in output
    assert "Random search" in output
    assert "validation_cost=12.345" in output
