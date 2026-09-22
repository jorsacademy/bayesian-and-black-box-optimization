from nevergrad_policy import cli


def test_cli_prints_optimizer_table(monkeypatch, capsys):
    class Result:
        def __init__(self, optimizer, reorder_point, order_quantity, cost):
            self.optimizer = optimizer
            self.reorder_point = reorder_point
            self.order_quantity = order_quantity
            self.cost = cost

    monkeypatch.setattr(
        cli,
        "compare_optimizers",
        lambda budget: [Result("NGOpt", 7, 19, 12.345)],
    )
    cli.main()
    output = capsys.readouterr().out
    assert "optimizer,reorder_point,order_quantity,mean_cost" in output
    assert "NGOpt,7,19,12.345" in output
