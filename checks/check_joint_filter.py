# Joint particle filter vs. the exact joint Kalman likelihood (linear-Gaussian DeGroot).
#
# Run from the repository root:  python -m checks.check_joint_filter

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from checks.common import CHECK_RESULTS_DIR, REPO_ROOT, load_base_config, self_weight_profile
from checks.plotting import (
    ESTIMATE_STYLE,
    EXACT_STYLE,
    POINTS_STYLE,
    ProfileCurve,
    dataset_title,
    plot_self_weight_profiles,
    save_figure,
    set_log_ticks,
    style_axes,
)
from src import kalman, simulate
from src.config import Config, save_config
from src.particle_filter import joint_pf_loglik

OUTPUT_DIR = CHECK_RESULTS_DIR / "joint_filter"

PARTICLE_COUNTS = [100, 1000, 10000]
N_REPLICATES = 100
N_PARTICLES_PROFILE = 2000
PROFILE_SEED = 0
DIMENSION_AGENT_COUNTS = [2, 5, 10, 20]
N_PARTICLES_DIMENSION = 1000
N_REPLICATES_DIMENSION = 30


def small_network_config(base_config: Config, n_agents: int) -> Config:
    """Baseline config on a small Watts–Strogatz network (k = 2, p = 0.1) with T = 50."""
    return replace(
        base_config,
        network=replace(base_config.network, n_agents=n_agents, k=2, p=0.1),
        simulation=replace(base_config.simulation, T=50),
    )


def pf_minus_kalman(config: Config, dataset: dict, kalman_value: float, n_particles: int,
                    seeds: range) -> tuple[np.ndarray, np.ndarray]:
    """PF − Kalman log-likelihood differences and mean ESS / N, one entry per seed."""
    differences, mean_ess_fractions = [], []
    for seed in seeds:
        pf_value, diagnostics = joint_pf_loglik(
            dataset["y"], dataset["W"], config.model, config.simulation, n_particles,
            np.random.default_rng(seed), return_filtered=True,
        )
        differences.append(pf_value - kalman_value)
        mean_ess_fractions.append(diagnostics["ess"].mean() / n_particles)
    return np.array(differences), np.array(mean_ess_fractions)


def check_convergence(config: Config, dataset: dict, kalman_value: float) -> dict:
    """(a) Bias and spread of the PF log-likelihood as the number of particles grows."""
    print("\n(a) Convergence: d = PF log-lik − Kalman log-lik, "
          f"{N_REPLICATES} runs per N (seeds 0..{N_REPLICATES - 1})")
    print(f"{'N':>7}  {'mean(d)':>9}  {'sd(d)':>8}  {'mean(exp d)':>11}  {'s.e.':>7}")
    convergence = {}
    for n_particles in PARTICLE_COUNTS:
        differences, _ = pf_minus_kalman(config, dataset, kalman_value, n_particles, range(N_REPLICATES))
        likelihood_ratios = np.exp(differences)
        convergence[n_particles] = {
            "mean_d": differences.mean(),
            "sd_d": differences.std(ddof=1),
            "mean_ratio": likelihood_ratios.mean(),
            "ratio_se": likelihood_ratios.std(ddof=1) / np.sqrt(N_REPLICATES),
        }
        row = convergence[n_particles]
        print(f"{n_particles:>7}  {row['mean_d']:>+9.4f}  {row['sd_d']:>8.4f}  "
              f"{row['mean_ratio']:>11.4f}  {row['ratio_se']:>7.4f}")

    largest = convergence[PARTICLE_COUNTS[-1]]
    assert abs(largest["mean_ratio"] - 1) < 3 * largest["ratio_se"] + 0.01, (
        f"mean(exp d) = {largest['mean_ratio']:.4f} at N = {PARTICLE_COUNTS[-1]} "
        f"is not within 3 s.e. + 0.01 of 1 (s.e. = {largest['ratio_se']:.4f})"
    )
    spreads = [convergence[n_particles]["sd_d"] for n_particles in PARTICLE_COUNTS]
    assert all(np.diff(spreads) < 0), f"sd(d) does not decrease with N: {spreads}"
    return convergence


def check_filtered_means(config: Config, dataset: dict) -> float:
    """(b) PF filtered means of x against the Kalman filtered means."""
    n_particles = PARTICLE_COUNTS[-1]
    _, kalman_means, _ = kalman.kalman_loglik(
        dataset["y"], dataset["W"], config.model, config.simulation, return_filtered=True
    )
    _, diagnostics = joint_pf_loglik(
        dataset["y"], dataset["W"], config.model, config.simulation, n_particles,
        np.random.default_rng(0), return_filtered=True,
    )
    n_agents = dataset["W"].shape[0]
    max_gap = np.max(np.abs(diagnostics["filtered_mean"] - kalman_means[:, :n_agents]))
    print(f"\n(b) Filtered means at N = {n_particles}: max_(i,t) |PF − Kalman| = {max_gap:.4f}")
    assert max_gap < 0.02, f"filtered means differ by {max_gap:.4f} (limit 0.02)"
    return max_gap


def check_profile(config: Config, dataset: dict) -> Path:
    """(c) λ profile: Kalman vs PF with common random numbers."""
    self_weight_grid = np.linspace(0.05, 0.95, 19)
    y, W = dataset["y"], dataset["W"]
    kalman_profile = self_weight_profile(
        lambda params: kalman.kalman_loglik(y, W, params, config.simulation), config.model, self_weight_grid
    )
    pf_profile = self_weight_profile(
        lambda params: joint_pf_loglik(y, W, params, config.simulation, N_PARTICLES_PROFILE,
                                       np.random.default_rng(PROFILE_SEED)),
        config.model, self_weight_grid,
    )
    print(f"\n(c) λ profile ({self_weight_grid.size} points, PF N = {N_PARTICLES_PROFILE}, seed {PROFILE_SEED} "
          f"at every point): argmax Kalman = {self_weight_grid[kalman_profile.argmax()]:.2f}, "
          f"PF = {self_weight_grid[pf_profile.argmax()]:.2f}; "
          f"max |PF − Kalman| = {np.max(np.abs(pf_profile - kalman_profile)):.3f}")

    return plot_self_weight_profiles(
        OUTPUT_DIR / "lambda_profile.png", self_weight_grid,
        [ProfileCurve(kalman_profile, "Kalman (exact)", EXACT_STYLE),
         ProfileCurve(pf_profile, rf"particle filter ($N = {N_PARTICLES_PROFILE}$, common seed)",
                      {**ESTIMATE_STYLE, "color": "C0"})],
        config.model.self_weight, rf"Profile over $\lambda$: Kalman vs. particle filter {dataset_title(config)}",
        relative=False, figsize=(7, 4),
    )


def check_dimension_effect(base_config: Config) -> tuple[dict, Path]:
    """(d) How the PF log-likelihood spread and ESS change with the number of agents (report only)."""
    print(f"\n(d) Dimension effect: N = {N_PARTICLES_DIMENSION}, {N_REPLICATES_DIMENSION} runs per n")
    print(f"{'n':>4}  {'sd(d)':>8}  {'mean(d)':>9}  {'mean ESS/N':>10}")
    dimension_results = {}
    for n_agents in DIMENSION_AGENT_COUNTS:
        config = small_network_config(base_config, n_agents)
        dataset = simulate.simulate(config)
        kalman_value = kalman.kalman_loglik(dataset["y"], dataset["W"], config.model, config.simulation)
        differences, ess_fractions = pf_minus_kalman(
            config, dataset, kalman_value, N_PARTICLES_DIMENSION, range(N_REPLICATES_DIMENSION)
        )
        dimension_results[n_agents] = {
            "sd_d": differences.std(ddof=1), "mean_d": differences.mean(), "ess_fraction": ess_fractions.mean(),
        }
        row = dimension_results[n_agents]
        print(f"{n_agents:>4}  {row['sd_d']:>8.4f}  {row['mean_d']:>+9.4f}  {row['ess_fraction']:>10.3f}")

    agent_counts = np.array(DIMENSION_AGENT_COUNTS)
    spreads = np.array([dimension_results[n_agents]["sd_d"] for n_agents in DIMENSION_AGENT_COUNTS])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(agent_counts, spreads, color="C0", **POINTS_STYLE)
    set_log_ticks(ax, agent_counts, base=2)
    set_log_ticks(ax, [0.125, 0.25, 0.5, 1, 2, 4, 8, 16], axis="y")
    ax.set_ylim(spreads.min() / 1.5, spreads.max() * 1.5)
    style_axes(ax, r"number of agents $n$", r"$\mathrm{sd}(\hat\ell_{\mathrm{PF}} - \ell_{\mathrm{Kalman}})$",
               rf"Joint PF spread vs. network size ($N = {N_PARTICLES_DIMENSION}$, $T = 50$)", legend=False)
    figure_path = save_figure(fig, OUTPUT_DIR / "dimension_effect.png")
    return dimension_results, figure_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_base_config()
    config = small_network_config(base_config, n_agents=5)
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    kalman_value = kalman.kalman_loglik(dataset["y"], dataset["W"], config.model, config.simulation)
    print(f"Dataset: n = {config.network.n_agents}, T = {config.simulation.T}, "
          f"Kalman log-likelihood = {kalman_value:.4f}")

    convergence = check_convergence(config, dataset, kalman_value)
    max_gap = check_filtered_means(config, dataset)
    profile_path = check_profile(config, dataset)
    dimension_results, dimension_path = check_dimension_effect(base_config)

    largest = convergence[PARTICLE_COUNTS[-1]]
    spreads_by_particles = " → ".join(
        f"{convergence[n_particles]['sd_d']:.3f}" for n_particles in PARTICLE_COUNTS
    )
    spreads_by_agents = ", ".join(
        f"{dimension_results[n_agents]['sd_d']:.3f}" for n_agents in DIMENSION_AGENT_COUNTS
    )
    print("\nSummary")
    print(f"  (a) N = {PARTICLE_COUNTS[-1]}: mean(exp d) = {largest['mean_ratio']:.4f} ± {largest['ratio_se']:.4f}; "
          f"sd(d) = {spreads_by_particles} for N = {PARTICLE_COUNTS}  [passed]")
    print(f"  (b) max filtered-mean gap = {max_gap:.4f} < 0.02  [passed]")
    print(f"  (c) profile figure: {profile_path.relative_to(REPO_ROOT)}")
    print(f"  (d) sd(d) for n = {DIMENSION_AGENT_COUNTS}: {spreads_by_agents}; "
          f"figure: {dimension_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
