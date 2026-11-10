# Per-agent particle filter vs. its exact counterpart, the per-agent Kalman filter (code correctness),
# for every neighbour-input mode ("observed", "filtered", "filtered_var"). Both filters use the
# same per-agent model, so they must agree; the bias of that model is measured separately in
# checks/check_plugin_bias.py.
#
# Run from the repository root:  python -m checks.check_per_agent_filter

from itertools import combinations

import numpy as np

from checks.common import CHECK_RESULTS_DIR, load_base_config, with_model_params
from src import kalman, simulate
from src.config import Config, save_config
from src.neighbour_inputs import INPUT_MODES, InputMode
from src.particle_filter import per_agent_pf_loglik

OUTPUT_DIR = CHECK_RESULTS_DIR / "per_agent_filter"

N_PARTICLES = 500
N_REPLICATES = 50
TINY_MEASUREMENT_NOISE_SD = 1e-6


def check_pf_against_kalman(config: Config, dataset: dict, input_mode: InputMode) -> dict:
    """Per-agent PF against the exact per-agent Kalman filter for one input mode."""
    y, W = dataset["y"], dataset["W"]
    kalman_value, kalman_filtered = kalman.per_agent_kalman_loglik(
        y, W, config.model, config.simulation, return_filtered=True, input_mode=input_mode
    )
    differences = np.array([
        per_agent_pf_loglik(y, W, config.model, config.simulation, N_PARTICLES, np.random.default_rng(seed),
                            input_mode=input_mode)
        - kalman_value
        for seed in range(N_REPLICATES)
    ])
    likelihood_ratios = np.exp(differences)
    mean_ratio = likelihood_ratios.mean()
    ratio_se = likelihood_ratios.std(ddof=1) / np.sqrt(N_REPLICATES)

    _, pf_filtered = per_agent_pf_loglik(
        y, W, config.model, config.simulation, N_PARTICLES, np.random.default_rng(0),
        return_filtered=True, input_mode=input_mode,
    )
    min_ess_fraction = pf_filtered["ess"].min() / N_PARTICLES
    max_mean_gap = np.max(np.abs(pf_filtered["filtered_mean"] - kalman_filtered["filtered_mean"]))

    print(f"\nInput mode {input_mode!r}: per-agent PF vs per-agent Kalman, N = {N_PARTICLES}, "
          f"{N_REPLICATES} runs (seeds 0..{N_REPLICATES - 1})")
    print(f"    per-agent Kalman log-lik = {kalman_value:.4f}")
    print(f"    d = PF − Kalman: mean = {differences.mean():+.4f}, sd = {differences.std(ddof=1):.4f}")
    print(f"    mean(exp d) = {mean_ratio:.4f} ± {ratio_se:.4f} (s.e.)")
    print(f"    min ESS/N over agents and times (seed 0) = {min_ess_fraction:.3f}; "
          f"resamples per agent: mean {pf_filtered['n_resamples'].mean():.1f}")
    print(f"    filtered means: max_(i,t) |PF − Kalman| = {max_mean_gap:.4f}")

    assert abs(mean_ratio - 1) < 3 * ratio_se + 0.02, (
        f"[{input_mode}] mean(exp d) = {mean_ratio:.4f} not within 3 s.e. + 0.02 of 1 (s.e. = {ratio_se:.4f})"
    )
    assert max_mean_gap < 0.03, f"[{input_mode}] filtered means differ by {max_mean_gap:.4f} (limit 0.03)"
    return {"kalman_value": kalman_value, "mean_d": differences.mean(), "sd_d": differences.std(ddof=1),
            "mean_ratio": mean_ratio, "ratio_se": ratio_se, "min_ess_fraction": min_ess_fraction,
            "max_mean_gap": max_mean_gap}


def check_modes_agree_without_measurement_noise(config: Config) -> dict:
    """With σ_m → 0, κ → 1 and ŝ → Y, so all three Kalman input modes give the same log-likelihood."""
    tiny_noise_config = with_model_params(config, measurement_noise_sd=TINY_MEASUREMENT_NOISE_SD)
    dataset = simulate.simulate(tiny_noise_config)
    logliks = {
        input_mode: kalman.per_agent_kalman_loglik(
            dataset["y"], dataset["W"], tiny_noise_config.model, tiny_noise_config.simulation, input_mode=input_mode
        )
        for input_mode in INPUT_MODES
    }
    print(f"\nAll modes coincide when σ_m = {TINY_MEASUREMENT_NOISE_SD:g} (per-agent Kalman log-likelihood)")
    for input_mode, value in logliks.items():
        print(f"    {input_mode:<13} {value:.10f}")
    gaps = {}
    for first_mode, second_mode in combinations(INPUT_MODES, 2):
        gap = abs(logliks[first_mode] - logliks[second_mode])
        gaps[first_mode, second_mode] = gap
        print(f"    |{first_mode} − {second_mode}| = {gap:.2e}")
        assert gap < 1e-6, f"{first_mode} and {second_mode} differ by {gap:.2e} with σ_m = {TINY_MEASUREMENT_NOISE_SD:g}"
    return gaps


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_base_config()
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    print(f"Dataset: n = {config.network.n_agents}, T = {config.simulation.T}, "
          f"σ_e = {config.model.expression_noise_sd}, σ_m = {config.model.measurement_noise_sd}")

    results_by_mode = {input_mode: check_pf_against_kalman(config, dataset, input_mode) for input_mode in INPUT_MODES}
    mode_gaps = check_modes_agree_without_measurement_noise(config)

    print("\nSummary")
    print(f"  {'mode':<13} {'Kalman log-lik':>14} {'mean(exp d) ± s.e.':>20} {'sd(d)':>7} "
          f"{'max mean gap':>12} {'min ESS/N':>9}")
    for input_mode, row in results_by_mode.items():
        print(f"  {input_mode:<13} {row['kalman_value']:>14.4f} {row['mean_ratio']:>11.4f} ± {row['ratio_se']:.4f} "
              f"{row['sd_d']:>7.3f} {row['max_mean_gap']:>12.4f} {row['min_ess_fraction']:>9.3f}")
    print("  PF vs Kalman (mean(exp d) within 3 s.e. + 0.02 of 1; filtered-mean gap < 0.03): passed for all modes")
    print(f"  σ_m = {TINY_MEASUREMENT_NOISE_SD:g}: largest gap between modes = {max(mode_gaps.values()):.2e} < 1e-6  [passed]")


if __name__ == "__main__":
    main()
