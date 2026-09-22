from __future__ import annotations

from .optimization import run_botorch_search, run_sobol_search


def _format(name: str, result) -> str:
    return (
        f"{name}: cost={result.cost:.3f}, "
        f"s={result.policy.reorder_point:.2f}, "
        f"S={result.policy.order_up_to:.2f}, "
        f"evaluations={result.evaluations}"
    )


def main() -> None:
    botorch_result = run_botorch_search()
    sobol_result = run_sobol_search(budget=botorch_result.evaluations)
    print(_format("BoTorch", botorch_result))
    print(_format("Sobol", sobol_result))


if __name__ == "__main__":
    main()
