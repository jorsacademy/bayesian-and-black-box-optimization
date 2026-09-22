from .optimize import random_search, run_processoptimizer


def main() -> None:
    gp = run_processoptimizer()
    baseline = random_search()
    print("ProcessOptimizer GP")
    print(f"setting={tuple(round(v, 3) for v in gp.setting)}")
    print(f"validation_loss={gp.validation_loss:.4f}")
    print("Random search")
    print(f"setting={tuple(round(v, 3) for v in baseline.setting)}")
    print(f"validation_loss={baseline.validation_loss:.4f}")


if __name__ == "__main__":
    main()
