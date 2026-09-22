from process_opt import cli


def test_cli_output(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_processoptimizer", lambda: type("R", (), {"setting": (82.0, 4.5, 520.0), "validation_loss": 4.2})())
    monkeypatch.setattr(cli, "random_search", lambda: type("R", (), {"setting": (80.0, 4.0, 500.0), "validation_loss": 5.1})())
    cli.main()
    out = capsys.readouterr().out
    assert "ProcessOptimizer GP" in out
    assert "Random search" in out
    assert "validation_loss" in out
