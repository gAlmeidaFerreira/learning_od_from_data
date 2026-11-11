# Noise-driven per-agent particle filter, the likelihood estimator for correlated PMMH:
#   (a) unbiasedness: its likelihood estimate matches the per-agent Kalman filter on average;
#   (b) smoothness: with one fixed noise draw, the λ profile is smooth when particles are sorted
#       before resampling, and rough without sorting;
#   (c) correlation payoff (report only): sd of the log-likelihood ratio between the true ψ and a
#       nearby proposal ψ', with u' = ρ u + sqrt(1 − ρ²) ε, for several N and ρ.
#
# Run from the repository root:  python -m checks.check_correlated_pf

import os

# One BLAS thread per process: on these small matrices multithreaded BLAS is much slower, and the
# runs are parallelised across processes instead. Must be set before NumPy loads.
for blas_thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(blas_thread_variable, "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from checks.common import CHECK_RESULTS_DIR, REPO_ROOT, describe_dataset, load_base_config, self_weight_profile
from checks.plotting import (
    ESTIMATE_STYLE,
    EXACT_STYLE,
    POINTS_STYLE,
    ProfileCurve,
    plot_self_weight_profiles,
    save_figure,
    set_log_ticks,
    style_axes,
)
from src import kalman, simulate
from src.config import Config, ModelParams, save_config
from src.neighbour_inputs import InputMode
from src.particle_filter import crank_nicolson, draw_per_agent_noise, per_agent_pf_loglik_given_noise

OUTPUT_DIR = CHECK_RESULTS_DIR / "correlated_pf"

UNBIASEDNESS_MODES: tuple[InputMode, ...] = ("filtered", "filtered_var")
UNBIASEDNESS_PARTICLES = 500
UNBIASEDNESS_SEEDS = range(50)

PROFILE_PARTICLES = 500
PROFILE_NOISE_SEED = 0
PROFILE_GRID = np.linspace(0.05, 0.95, 37)
PROFILE_WINDOW = 20.0                # RMS over grid points within this many units of the Kalman maximum

PROPOSAL_STEP = 0.05                 # shift of logit λ, log σ_p and log σ_e from the true ψ to ψ'
PAYOFF_PARTICLE_COUNTS = [100, 500]
PAYOFF_CORRELATIONS = [0.0, 0.9, 0.99, 0.999]
PAYOFF_SEEDS = range(1000, 1050)     # same seeds for every (N, ρ), so the rows are paired


def proposal_params(params: ModelParams, shift: float = PROPOSAL_STEP) -> ModelParams:
    """ψ' with logit λ, log σ_p and log σ_e each shifted by `shift`."""
    logit_self_weight = np.log(params.self_weight / (1 - params.self_weight)) + shift
    return replace(
        params,
        self_weight=float(1 / (1 + np.exp(-logit_self_weight))),
        process_noise_sd=float(params.process_noise_sd * np.exp(shift)),
        expression_noise_sd=float(params.expression_noise_sd * np.exp(shift)),
    )


def unbiasedness_difference(task: tuple[Config, dict, InputMode, int]) -> tuple[InputMode, float]:
    """d = noise-driven PF − per-agent Kalman log-likelihood for one fresh noise draw."""
    config, dataset, input_mode, seed = task
    y, W = dataset["y"], dataset["W"]
    noise = draw_per_agent_noise(y.shape[0], UNBIASEDNESS_PARTICLES, y.shape[1], np.random.default_rng(seed))
    pf_value = per_agent_pf_loglik_given_noise(y, W, config.model, config.simulation, noise, input_mode=input_mode)
    kalman_value = kalman.per_agent_kalman_loglik(y, W, config.model, config.simulation, input_mode=input_mode)
    return input_mode, pf_value - kalman_value


def check_unbiasedness(config: Config, dataset: dict, pool: ProcessPoolExecutor) -> dict:
    """(a) mean(exp d) ≈ 1 for each corrected input mode."""
    tasks = [(config, dataset, input_mode, seed) for input_mode in UNBIASEDNESS_MODES for seed in UNBIASEDNESS_SEEDS]
    differences_by_mode: dict[str, list[float]] = {input_mode: [] for input_mode in UNBIASEDNESS_MODES}
    for input_mode, difference in pool.map(unbiasedness_difference, tasks):
        differences_by_mode[input_mode].append(difference)

    print(f"\n(a) Unbiasedness: d = noise-driven PF − per-agent Kalman, N = {UNBIASEDNESS_PARTICLES}, "
          f"{len(UNBIASEDNESS_SEEDS)} noise draws (seeds {UNBIASEDNESS_SEEDS.start}..{UNBIASEDNESS_SEEDS.stop - 1})")
    results = {}
    for input_mode, values in differences_by_mode.items():
        differences = np.array(values)
        likelihood_ratios = np.exp(differences)
        mean_ratio = likelihood_ratios.mean()
        ratio_se = likelihood_ratios.std(ddof=1) / np.sqrt(differences.size)
        results[input_mode] = {"mean_d": differences.mean(), "sd_d": differences.std(ddof=1),
                               "mean_ratio": mean_ratio, "ratio_se": ratio_se}
        print(f"    {input_mode:<13} mean(d) = {differences.mean():+.4f}, sd(d) = {differences.std(ddof=1):.4f}, "
              f"mean(exp d) = {mean_ratio:.4f} ± {ratio_se:.4f} (s.e.)")
    for input_mode, row in results.items():
        assert abs(row["mean_ratio"] - 1) < 3 * row["ratio_se"] + 0.02, (
            f"[{input_mode}] mean(exp d) = {row['mean_ratio']:.4f} not within 3 s.e. + 0.02 of 1 "
            f"(s.e. = {row['ratio_se']:.4f})"
        )
    return results


def check_smoothness(config: Config, dataset: dict) -> tuple[dict, Path]:
    """(b) λ profiles with one fixed noise draw: sorted vs unsorted resampling vs per-agent Kalman."""
    y, W = dataset["y"], dataset["W"]
    noise = draw_per_agent_noise(y.shape[0], PROFILE_PARTICLES, y.shape[1], np.random.default_rng(PROFILE_NOISE_SEED))
    kalman_profile = self_weight_profile(
        lambda params: kalman.per_agent_kalman_loglik(y, W, params, config.simulation, input_mode="filtered_var"),
        config.model, PROFILE_GRID,
    )
    pf_profiles = {
        sort_particles: self_weight_profile(
            lambda params: per_agent_pf_loglik_given_noise(
                y, W, params, config.simulation, noise, input_mode="filtered_var", sort_particles=sort_particles
            ),
            config.model, PROFILE_GRID,
        )
        for sort_particles in (True, False)
    }

    near_peak = kalman_profile >= kalman_profile.max() - PROFILE_WINDOW
    rms_deviations = {}
    for sort_particles, profile in pf_profiles.items():
        deviation = profile[near_peak] - kalman_profile[near_peak]
        rms_deviations[sort_particles] = float(np.sqrt(np.mean((deviation - deviation.mean()) ** 2)))

    print(f"\n(b) Smoothness: λ profile, \"filtered_var\", N = {PROFILE_PARTICLES}, one noise draw "
          f"(seed {PROFILE_NOISE_SEED}) at all {PROFILE_GRID.size} points")
    print(f"    RMS deviation from the per-agent Kalman profile over the {near_peak.sum()} points within "
          f"{PROFILE_WINDOW:g} units of its maximum (mean offset removed):")
    for sort_particles, rms in rms_deviations.items():
        print(f"    {'sorted' if sort_particles else 'unsorted':<9} {rms:.3f}")

    figure_path = plot_self_weight_profiles(
        OUTPUT_DIR / "profile_smoothness.png", PROFILE_GRID,
        [ProfileCurve(kalman_profile, "per-agent Kalman (exact for this model)", EXACT_STYLE),
         ProfileCurve(pf_profiles[True], f"noise-driven PF, sorted resampling (RMS {rms_deviations[True]:.2f})",
                      {**ESTIMATE_STYLE, "color": "C0", "linestyle": "-"}),
         ProfileCurve(pf_profiles[False], f"noise-driven PF, unsorted resampling (RMS {rms_deviations[False]:.2f})",
                      {**ESTIMATE_STYLE, "color": "C1", "marker": "s", "markersize": 3})],
        config.model.self_weight,
        rf"$\lambda$ profile with one fixed noise draw ($N = {PROFILE_PARTICLES}$, filtered inputs + variance)",
        legend_fontsize=8.5,
    )
    return rms_deviations, figure_path


def log_ratio_replicate(task: tuple[Config, dict, int, float, int]) -> tuple[int, float, float]:
    """D = log p̂(ψ', u') − log p̂(ψ, u) with u fresh and u' = crank_nicolson(u, ρ)."""
    config, dataset, n_particles, correlation, seed = task
    y, W = dataset["y"], dataset["W"]
    rng = np.random.default_rng(seed)
    current_noise = draw_per_agent_noise(y.shape[0], n_particles, y.shape[1], rng)
    proposed_noise = crank_nicolson(current_noise, correlation, rng)
    current_value = per_agent_pf_loglik_given_noise(y, W, config.model, config.simulation, current_noise)
    proposed_value = per_agent_pf_loglik_given_noise(
        y, W, proposal_params(config.model), config.simulation, proposed_noise
    )
    return n_particles, correlation, proposed_value - current_value


def check_correlation_payoff(config: Config, dataset: dict, pool: ProcessPoolExecutor) -> tuple[dict, tuple, Path]:
    """(c) sd and mean of D for each (N, ρ); report only."""
    y, W = dataset["y"], dataset["W"]
    proposal = proposal_params(config.model)
    true_difference = (
        kalman.per_agent_kalman_loglik(y, W, proposal, config.simulation, input_mode="filtered_var")
        - kalman.per_agent_kalman_loglik(y, W, config.model, config.simulation, input_mode="filtered_var")
    )

    tasks = [(config, dataset, n_particles, correlation, seed)
             for n_particles in PAYOFF_PARTICLE_COUNTS for correlation in PAYOFF_CORRELATIONS for seed in PAYOFF_SEEDS]
    ratios: dict[tuple[int, float], list[float]] = {}
    for n_particles, correlation, log_ratio in pool.map(log_ratio_replicate, tasks):
        ratios.setdefault((n_particles, correlation), []).append(log_ratio)
    summary = {key: {"sd": float(np.std(values, ddof=1)), "mean": float(np.mean(values))}
               for key, values in ratios.items()}

    # Cost grows with N only (ρ is free), so the cheapest setting is the smallest N reaching
    # sd(D) ≤ 1; among those, the smallest ρ, which lets u move most between iterations.
    reaching_target = [key for key in summary if summary[key]["sd"] <= 1.0]
    cheapest = min(reaching_target, key=lambda key: (key[0], key[1])) if reaching_target else None

    print(f"\n(c) Correlation payoff: D = log p̂(ψ', u') − log p̂(ψ, u), \"filtered_var\", ψ = truth, ψ' = logit λ, "
          f"log σ_p, log σ_e each +{PROPOSAL_STEP}; {len(PAYOFF_SEEDS)} replicates per row")
    print(f"    {'N':>5} {'ρ':>6} {'sd(D)':>8} {'mean(D)':>9}")
    for (n_particles, correlation), row in summary.items():
        marker = "  ← cheapest with sd(D) ≤ 1" if (n_particles, correlation) == cheapest else ""
        print(f"    {n_particles:>5} {correlation:>6g} {row['sd']:>8.3f} {row['mean']:>+9.3f}{marker}")
    print(f"    per-agent Kalman log p(Y | ψ') − log p(Y | ψ) = {true_difference:+.3f}")
    if cheapest is None:
        print("    no (N, ρ) reaches sd(D) ≤ 1")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    one_minus_correlation = [1 - correlation for correlation in PAYOFF_CORRELATIONS]
    for color, n_particles in zip(("C1", "C0"), PAYOFF_PARTICLE_COUNTS):
        ax.plot(one_minus_correlation, [summary[n_particles, correlation]["sd"] for correlation in PAYOFF_CORRELATIONS],
                color=color, label=rf"$N = {n_particles}$", **POINTS_STYLE)
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.2, label=r"target sd$(D) = 1$")
    ax.set_yscale("log")
    set_log_ticks(ax, one_minus_correlation, [rf"$\rho = {correlation:g}$" for correlation in PAYOFF_CORRELATIONS])
    ax.invert_xaxis()
    style_axes(ax, "", r"sd of $D = \log\hat p(Y \mid \psi', u') - \log\hat p(Y \mid \psi, u)$",
               "Correlated noise: spread of the log-likelihood ratio")
    figure_path = save_figure(fig, OUTPUT_DIR / "log_ratio_sd_vs_correlation.png")
    return summary, (cheapest, true_difference), figure_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_base_config()
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    dataset = {"y": dataset["y"], "W": dataset["W"]}      # only what the workers need
    print(describe_dataset(config))

    with ProcessPoolExecutor() as pool:
        unbiasedness = check_unbiasedness(config, dataset, pool)
        rms_deviations, profile_path = check_smoothness(config, dataset)
        payoff, (cheapest, true_difference), payoff_path = check_correlation_payoff(config, dataset, pool)

    print("\nSummary")
    print("  (a) mean(exp d) within 3 s.e. + 0.02 of 1: " + ", ".join(
        f"{input_mode} {row['mean_ratio']:.4f} ± {row['ratio_se']:.4f}" for input_mode, row in unbiasedness.items())
        + "  [passed]")
    print(f"  (b) RMS deviation from Kalman profile: sorted {rms_deviations[True]:.3f}, "
          f"unsorted {rms_deviations[False]:.3f}")
    print(f"  (c) sd(D) by (N, ρ): " + ", ".join(
        f"({n_particles}, {correlation:g}) {row['sd']:.3f}" for (n_particles, correlation), row in payoff.items()))
    print(f"      Kalman ΔlogL = {true_difference:+.3f}; cheapest with sd(D) ≤ 1: "
          + (f"N = {cheapest[0]}, ρ = {cheapest[1]:g}" if cheapest else "none"))
    print("  figures: " + ", ".join(str(path.relative_to(REPO_ROOT)) for path in (profile_path, payoff_path)))


if __name__ == "__main__":
    main()
