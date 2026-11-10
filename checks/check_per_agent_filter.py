# Per-agent particle filter vs. its exact counterpart, the per-agent Kalman filter (code correctness).
# Both use the same plug-in model, so they must agree; the bias of that model is measured
# separately in checks/check_plugin_bias.py.
#
# Run from the repository root:  python -m checks.check_per_agent_filter

from pathlib import Path

import numpy as np

from src import kalman, simulate
from src.config import Config, load_config, save_config
from src.particle_filter import per_agent_pf_loglik

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "checks" / "per_agent_filter"

N_PARTICLES = 500
N_REPLICATES = 50


def check_pf_against_kalman(config: Config, dataset: dict) -> dict:
    """Per-agent PF against the exact per-agent Kalman filter (code correctness)."""
    y, W = dataset["y"], dataset["W"]
    kalman_value, kalman_filtered = kalman.per_agent_kalman_loglik(
        y, W, config.model, config.simulation, return_filtered=True
    )
    differences = np.array([
        per_agent_pf_loglik(y, W, config.model, config.simulation, N_PARTICLES, np.random.default_rng(seed))
        - kalman_value
        for seed in range(N_REPLICATES)
    ])
    likelihood_ratios = np.exp(differences)
    mean_ratio = likelihood_ratios.mean()
    ratio_se = likelihood_ratios.std(ddof=1) / np.sqrt(N_REPLICATES)

    _, pf_filtered = per_agent_pf_loglik(
        y, W, config.model, config.simulation, N_PARTICLES, np.random.default_rng(0), return_filtered=True
    )
    min_ess_fraction = pf_filtered["ess"].min() / N_PARTICLES
    max_mean_gap = np.max(np.abs(pf_filtered["filtered_mean"] - kalman_filtered["filtered_mean"]))

    print(f"\nPer-agent PF vs per-agent Kalman: N = {N_PARTICLES}, {N_REPLICATES} runs (seeds 0..{N_REPLICATES - 1})")
    print(f"    per-agent Kalman log-lik = {kalman_value:.4f}")
    print(f"    d = PF − Kalman: mean = {differences.mean():+.4f}, sd = {differences.std(ddof=1):.4f}")
    print(f"    mean(exp d) = {mean_ratio:.4f} ± {ratio_se:.4f} (s.e.)")
    print(f"    min ESS/N over agents and times (seed 0) = {min_ess_fraction:.3f}; "
          f"resamples per agent: mean {pf_filtered['n_resamples'].mean():.1f}")
    print(f"    filtered means: max_(i,t) |PF − Kalman| = {max_mean_gap:.4f}")

    assert abs(mean_ratio - 1) < 3 * ratio_se + 0.02, (
        f"mean(exp d) = {mean_ratio:.4f} not within 3 s.e. + 0.02 of 1 (s.e. = {ratio_se:.4f})"
    )
    assert max_mean_gap < 0.03, f"filtered means differ by {max_mean_gap:.4f} (limit 0.03)"
    return {"mean_d": differences.mean(), "sd_d": differences.std(ddof=1), "mean_ratio": mean_ratio,
            "ratio_se": ratio_se, "min_ess_fraction": min_ess_fraction, "max_mean_gap": max_mean_gap}



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(REPO_ROOT / "configs" / "degroot_base.yaml")
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    print(f"Dataset: n = {config.network.n_agents}, T = {config.simulation.T}, "
          f"σ_e = {config.model.expression_noise_sd}, σ_m = {config.model.measurement_noise_sd}")

    correctness = check_pf_against_kalman(config, dataset)

    print("\nSummary")
    print(f"  mean(exp d) = {correctness['mean_ratio']:.4f} ± {correctness['ratio_se']:.4f}, "
          f"sd(d) = {correctness['sd_d']:.3f}  [passed]")
    print(f"  max filtered-mean gap = {correctness['max_mean_gap']:.4f} < 0.03  [passed]")
    print(f"  min ESS/N = {correctness['min_ess_fraction']:.3f}")


if __name__ == "__main__":
    main()
