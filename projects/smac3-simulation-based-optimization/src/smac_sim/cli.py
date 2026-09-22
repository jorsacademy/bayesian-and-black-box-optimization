from __future__ import annotations

from .optimization import random_search, run_smac


def main() -> None:
    budget = 12
    smac_result = run_smac(n_trials=budget, seed=7)
    random_result = random_search(n_trials=budget, seed=7)

    print("SMAC3")
    print(
        f"batch_size={smac_result.batch_size} "
        f"safety_stock={smac_result.safety_stock} "
        f"validation_cost={smac_result.validation_cost:.3f}"
    )
    print("Random search")
    print(
        f"batch_size={random_result.batch_size} "
        f"safety_stock={random_result.safety_stock} "
        f"validation_cost={random_result.validation_cost:.3f}"
    )


if __name__ == "__main__":
    main()
