from botorch_policy import cli


def test_cli_runs_end_to_end(capsys, monkeypatch):
    monkeypatch.setattr(cli, "run_botorch_search", lambda: cli.run_sobol_search(budget=4, seed=1))
    cli.main()
    output = capsys.readouterr().out
    assert "BoTorch:" in output
    assert "Sobol:" in output
    assert "evaluations=4" in output
