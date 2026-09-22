import pytest

from process_opt.process import ProcessConfig, evaluate_setting, simulate_batch


def test_simulation_is_deterministic_for_seed():
    setting = (82.0, 4.6, 520.0)
    assert simulate_batch(setting, seed=11) == simulate_batch(setting, seed=11)


def test_evaluate_setting_is_deterministic():
    setting = (80.0, 5.0, 500.0)
    assert evaluate_setting(setting) == pytest.approx(evaluate_setting(setting))


def test_good_region_beats_extreme_setting_without_noise():
    cfg = ProcessConfig(noise_std=0.0)
    good = evaluate_setting((82.0, 4.6, 520.0), seeds=(1,), config=cfg)
    bad = evaluate_setting((60.0, 8.0, 800.0), seeds=(1,), config=cfg)
    assert good < bad


@pytest.mark.parametrize(
    "setting",
    [(59.0, 4.0, 500.0), (80.0, 9.0, 500.0), (80.0, 4.0, 900.0)],
)
def test_out_of_bounds_setting_rejected(setting):
    with pytest.raises(ValueError):
        simulate_batch(setting, seed=1)


def test_invalid_config_and_empty_seed_set_rejected():
    with pytest.raises(ValueError):
        ProcessConfig(temperature_bounds=(90.0, 80.0))
    with pytest.raises(ValueError):
        ProcessConfig(noise_std=-1.0)
    with pytest.raises(ValueError):
        evaluate_setting((80.0, 4.0, 500.0), seeds=())
