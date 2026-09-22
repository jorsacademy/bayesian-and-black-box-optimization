from __future__ import annotations

from .optimization import compare_optimizers


def main() -> None:
    results = compare_optimizers(budget=24)
    print("optimizer,reorder_point,order_quantity,mean_cost")
    for result in results:
        print(
            f"{result.optimizer},{result.reorder_point},"
            f"{result.order_quantity},{result.cost:.3f}"
        )


if __name__ == "__main__":
    main()
