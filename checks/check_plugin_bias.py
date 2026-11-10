# Bias from the per-agent model's neighbour inputs. The per-agent filters replace the neighbours'
# hidden expressions s by observed posts Y ("observed", the plug-in), by filtered estimates ŝ
# ("filtered"), or by ŝ plus its uncertainty ("filtered_var"). Compares their estimates of λ, and
# the widths of their likelihood profiles, with the exact joint Kalman likelihood on one dataset
# and across measurement-noise levels (report only).
#
# Run from the repository root:  python -m checks.check_plugin_bias

import os

# One BLAS thread per process: on these small matrices (2n = 100) multithreaded BLAS is ~25x
# slower (measured ~1.1 s vs ~40 ms per joint Kalman evaluation). Must be set before NumPy loads.
for blas_thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(blas_thread_variable, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from checks.common import (
    CHECK_RESULTS_DIR,
    PROFILE_WIDTH_DROP,
    REPO_ROOT,
    load_base_config,
    profile_width,
    self_weight_profile,
    with_model_params,
)
from src import kalman, simulate
from src.config import Config, save_config
from src.neighbour_inputs import INPUT_MODES, InputMode
from src.particle_filter import per_agent_pf_loglik

OUTPUT_DIR = CHECK_RESULTS_DIR / "plugin_bias"

N_PARTICLES = 500
PROFILE_SEED = 0
PROFILE_GRID = np.linspace(0.05, 0.95, 37)
ESTIMATION_GRID = np.linspace(0.05, 0.95, 91)
MEASUREMENT_NOISE_LEVELS = [0.02, 0.05, 0.1, 0.2]
DATASET_SEEDS = range(100, 120)
WIDTH_DROP = PROFILE_WIDTH_DROP

# Methods compared: the exact joint Kalman and the per-agent Kalman in each input mode.
METHODS = ("joint", *INPUT_MODES)
METHOD_LABELS = {
    "joint": "joint Kalman (exact model)",
    "observed": "per-agent Kalman, observed inputs",
    "filtered": "per-agent Kalman, filtered inputs",
    "filtered_var": "per-agent Kalman, filtered inputs + variance",
}
METHOD_COLORS = {"joint": "0.15", "observed": "C1", "filtered": "C0", "filtered_var": "C4"}


def profile_by_method(y: np.ndarray, W: np.ndarray, config: Config, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Log-likelihood over a λ grid for the joint Kalman and each per-agent input mode."""
    profiles = {"joint": self_weight_profile(
        lambda params: kalman.kalman_loglik(y, W, params, config.simulation), config.model, grid
    )}
    for input_mode in INPUT_MODES:
        profiles[input_mode] = self_weight_profile(
            lambda params: kalman.per_agent_kalman_loglik(y, W, params, config.simulation, input_mode=input_mode),
            config.model, grid,
        )
    return profiles


def per_agent_pf_profile(y: np.ndarray, W: np.ndarray, config: Config, input_mode: InputMode) -> np.ndarray:
    """Per-agent PF log-likelihood over PROFILE_GRID with the same seed at every point."""
    return self_weight_profile(
        lambda params: per_agent_pf_loglik(y, W, params, config.simulation, N_PARTICLES,
                                           np.random.default_rng(PROFILE_SEED), input_mode=input_mode),
        config.model, PROFILE_GRID,
    )


def check_profiles(config: Config, dataset: dict) -> tuple[dict, dict, list[Path]]:
    """(a) λ profiles on one dataset: joint Kalman, per-agent Kalman in each mode, per-agent PF."""
    y, W = dataset["y"], dataset["W"]
    true_self_weight = config.model.self_weight
    kalman_profiles = profile_by_method(y, W, config, PROFILE_GRID)
    pf_profiles = {input_mode: per_agent_pf_profile(y, W, config, input_mode)
                   for input_mode in ("observed", "filtered_var")}

    curves = {**{f"{method} Kalman": kalman_profiles[method] for method in METHODS},
              **{f"{input_mode} PF": pf_profiles[input_mode] for input_mode in pf_profiles}}
    argmaxes = {name: float(PROFILE_GRID[profile.argmax()]) for name, profile in curves.items()}
    widths = {name: profile_width(PROFILE_GRID, profile) for name, profile in curves.items()}

    print(f"\n(a) λ profiles ({PROFILE_GRID.size} points; PF N = {N_PARTICLES}, seed {PROFILE_SEED} at every point)")
    print(f"    {'curve':<20} {'argmax λ':>9} {'width (Δℓ ≤ 2)':>15}")
    for name in curves:
        print(f"    {name:<20} {argmaxes[name]:>9.3f} {widths[name]:>15.3f}")
    print(f"    true λ = {true_self_weight}")

    figure_paths = []

    # Original figure: joint Kalman, per-agent Kalman and per-agent PF with observed (plug-in) inputs.
    # The per-agent PF estimates the per-agent Kalman curve, so it is shifted by the same constant.
    observed_peak = kalman_profiles["observed"].max()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(PROFILE_GRID, kalman_profiles["joint"] - kalman_profiles["joint"].max(), color="0.15", linewidth=1.8,
            label="joint Kalman (exact model)")
    ax.plot(PROFILE_GRID, kalman_profiles["observed"] - observed_peak, color="C1", linewidth=1.8,
            label="per-agent Kalman (plug-in model)")
    ax.plot(PROFILE_GRID, pf_profiles["observed"] - observed_peak, color="C0", linewidth=1.2, linestyle="--",
            marker="o", markersize=3.5, label=rf"per-agent PF ($N = {N_PARTICLES}$, common seed)")
    ax.axvline(true_self_weight, color="0.5", linestyle=":", linewidth=1.2, label=rf"true $\lambda = {true_self_weight}$")
    ax.set_ylim(-60, 5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"relative log-likelihood $\ell(\lambda) - \ell(\hat\lambda)$")
    ax.set_title(rf"Plug-in bias in the $\lambda$ profile ($n = {config.network.n_agents}$, "
                 rf"$T = {config.simulation.T}$, $\sigma_m = {config.model.measurement_noise_sd}$)")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figure_paths.append(OUTPUT_DIR / "lambda_profiles.png")
    fig.savefig(figure_paths[-1], dpi=150)
    plt.close(fig)

    # New figure: every input mode. The PF ("filtered_var") is shifted by its Kalman curve's peak.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in METHODS:
        profile = kalman_profiles[method]
        ax.plot(PROFILE_GRID, profile - profile.max(), color=METHOD_COLORS[method], linewidth=1.8,
                label=METHOD_LABELS[method])
    ax.plot(PROFILE_GRID, pf_profiles["filtered_var"] - kalman_profiles["filtered_var"].max(),
            color=METHOD_COLORS["filtered_var"], linewidth=1.0, linestyle="--", marker="o", markersize=3.5,
            label=rf"per-agent PF, filtered inputs + variance ($N = {N_PARTICLES}$, common seed)")
    ax.axhline(-WIDTH_DROP, color="0.6", linestyle="-.", linewidth=0.9,
               label=rf"$-{WIDTH_DROP:g}$ (profile width level)")
    ax.axvline(true_self_weight, color="0.5", linestyle=":", linewidth=1.2, label=rf"true $\lambda = {true_self_weight}$")
    ax.set_ylim(-60, 5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"relative log-likelihood $\ell(\lambda) - \ell(\hat\lambda)$")
    ax.set_title(rf"$\lambda$ profiles by neighbour-input mode ($n = {config.network.n_agents}$, "
                 rf"$T = {config.simulation.T}$, $\sigma_m = {config.model.measurement_noise_sd}$)")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figure_paths.append(OUTPUT_DIR / "lambda_profiles_by_input_mode.png")
    fig.savefig(figure_paths[-1], dpi=150)
    plt.close(fig)
    return argmaxes, widths, figure_paths


def estimate_self_weight(task: tuple[Config, np.ndarray, float, int]) -> tuple[float, dict[str, tuple[float, float]]]:
    """For one dataset: (λ̂, profile width) by grid argmax for the joint Kalman and each input mode."""
    config, W, measurement_noise_sd, dataset_seed = task
    noise_config = with_model_params(config, measurement_noise_sd=measurement_noise_sd)
    y = simulate.simulate(noise_config, W=W, rng=np.random.default_rng(dataset_seed))["y"]
    profiles = profile_by_method(y, W, noise_config, ESTIMATION_GRID)
    return measurement_noise_sd, {
        method: (float(ESTIMATION_GRID[profile.argmax()]), profile_width(ESTIMATION_GRID, profile))
        for method, profile in profiles.items()
    }


def check_bias_vs_measurement_noise(config: Config, W: np.ndarray) -> tuple[dict, list[Path]]:
    """(b) λ̂ and profile width for each method across measurement-noise levels (report only)."""
    tasks = [(config, W, noise_sd, seed) for noise_sd in MEASUREMENT_NOISE_LEVELS for seed in DATASET_SEEDS]
    with ProcessPoolExecutor() as pool:
        estimates = list(pool.map(estimate_self_weight, tasks))

    true_self_weight = config.model.self_weight
    summary: dict[float, dict[str, dict[str, float]]] = {}
    for noise_sd in MEASUREMENT_NOISE_LEVELS:
        summary[noise_sd] = {}
        for method in METHODS:
            self_weight_estimates = np.array([by_method[method][0] for level, by_method in estimates if level == noise_sd])
            widths = np.array([by_method[method][1] for level, by_method in estimates if level == noise_sd])
            summary[noise_sd][method] = {
                "mean": self_weight_estimates.mean(), "sd": self_weight_estimates.std(ddof=1), "width": widths.mean(),
            }

    print(f"\n(b) λ̂ vs measurement noise: {len(DATASET_SEEDS)} datasets per σ_m on the same network "
          f"(data seeds {DATASET_SEEDS.start}..{DATASET_SEEDS.stop - 1}), grid of {ESTIMATION_GRID.size} points")
    print(f"{'σ_m':>6}  {'joint λ̂ (mean ± sd)':>21}  {'per-agent λ̂ (mean ± sd)':>25}  "
          f"{'bias joint':>10}  {'bias per-agent':>14}")
    for noise_sd in MEASUREMENT_NOISE_LEVELS:
        joint_row, observed_row = summary[noise_sd]["joint"], summary[noise_sd]["observed"]
        print(f"{noise_sd:>6}  {joint_row['mean']:>12.3f} ± {joint_row['sd']:.3f}  "
              f"{observed_row['mean']:>16.3f} ± {observed_row['sd']:.3f}  "
              f"{joint_row['mean'] - true_self_weight:>+10.3f}  {observed_row['mean'] - true_self_weight:>+14.3f}")

    print(f"\n    By method (per-agent rows use the per-agent Kalman in that input mode); "
          f"width = mean range of λ within {WIDTH_DROP:g} of the profile maximum")
    print(f"{'σ_m':>6}  {'method':<13} {'λ̂ mean ± sd':>15}  {'bias':>7}  {'width':>6}")
    for noise_sd in MEASUREMENT_NOISE_LEVELS:
        for method in METHODS:
            row = summary[noise_sd][method]
            print(f"{noise_sd:>6}  {method:<13} {row['mean']:>7.3f} ± {row['sd']:.3f}  "
                  f"{row['mean'] - true_self_weight:>+7.3f}  {row['width']:>6.3f}")

    noise_levels = np.array(MEASUREMENT_NOISE_LEVELS)
    figure_paths = []

    # Original figure: joint vs per-agent Kalman with observed (plug-in) inputs.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, method, offset in [("joint Kalman", "joint", 0.96), ("per-agent Kalman", "observed", 1.04)]:
        ax.errorbar(noise_levels * offset, [summary[level][method]["mean"] for level in MEASUREMENT_NOISE_LEVELS],
                    yerr=[summary[level][method]["sd"] for level in MEASUREMENT_NOISE_LEVELS],
                    color=METHOD_COLORS[method], linewidth=1.6, marker="o", markersize=6, capsize=3, label=name)
    ax.axhline(true_self_weight, color="0.5", linestyle=":", linewidth=1.2, label=rf"true $\lambda = {true_self_weight}$")
    ax.set_xscale("log")
    ax.set_xticks(noise_levels, [f"{level:g}" for level in noise_levels])
    ax.minorticks_off()
    ax.set_xlabel(r"measurement noise $\sigma_m$")
    ax.set_ylabel(r"$\hat\lambda$ (mean $\pm$ sd over datasets)")
    ax.set_title(rf"Plug-in bias vs. measurement noise ($n = {config.network.n_agents}$, "
                 rf"$T = {config.simulation.T}$, {len(DATASET_SEEDS)} datasets per $\sigma_m$)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figure_paths.append(OUTPUT_DIR / "bias_vs_measurement_noise.png")
    fig.savefig(figure_paths[-1], dpi=150)
    plt.close(fig)

    # New figure: every method, λ̂ (left) and profile width (right).
    offsets = dict(zip(METHODS, (0.94, 0.98, 1.02, 1.06)))
    fig, (estimate_ax, width_ax) = plt.subplots(1, 2, figsize=(12, 4.6))
    for method in METHODS:
        shifted_levels = noise_levels * offsets[method]
        estimate_ax.errorbar(shifted_levels, [summary[level][method]["mean"] for level in MEASUREMENT_NOISE_LEVELS],
                             yerr=[summary[level][method]["sd"] for level in MEASUREMENT_NOISE_LEVELS],
                             color=METHOD_COLORS[method], linewidth=1.6, marker="o", markersize=6, capsize=3,
                             label=METHOD_LABELS[method])
        width_ax.plot(noise_levels, [summary[level][method]["width"] for level in MEASUREMENT_NOISE_LEVELS],
                      color=METHOD_COLORS[method], linewidth=1.6, marker="o", markersize=6, label=METHOD_LABELS[method])
    estimate_ax.axhline(true_self_weight, color="0.5", linestyle=":", linewidth=1.2,
                        label=rf"true $\lambda = {true_self_weight}$")
    for ax in (estimate_ax, width_ax):
        ax.set_xscale("log")
        ax.set_xticks(noise_levels, [f"{level:g}" for level in noise_levels])
        ax.minorticks_off()
        ax.set_xlabel(r"measurement noise $\sigma_m$")
        ax.grid(alpha=0.25)
    estimate_ax.set_ylabel(r"$\hat\lambda$ (mean $\pm$ sd over datasets)")
    estimate_ax.set_title(r"Estimate of $\lambda$")
    estimate_ax.legend(frameon=False, fontsize=8.5)
    width_ax.set_ylabel(rf"profile width: range of $\lambda$ with $\ell \geq \ell_{{\max}} - {WIDTH_DROP:g}$")
    width_ax.set_title("Profile width (smaller = more confident)")
    fig.suptitle(rf"Neighbour-input modes vs. joint Kalman ($n = {config.network.n_agents}$, "
                 rf"$T = {config.simulation.T}$, {len(DATASET_SEEDS)} datasets per $\sigma_m$)")
    fig.tight_layout()
    figure_paths.append(OUTPUT_DIR / "bias_vs_measurement_noise_by_input_mode.png")
    fig.savefig(figure_paths[-1], dpi=150)
    plt.close(fig)
    return summary, figure_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_base_config()
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    print(f"Dataset: n = {config.network.n_agents}, T = {config.simulation.T}, "
          f"σ_e = {config.model.expression_noise_sd}, σ_m = {config.model.measurement_noise_sd}")

    argmaxes, widths, profile_paths = check_profiles(config, dataset)
    bias_summary, bias_paths = check_bias_vs_measurement_noise(config, dataset["W"])

    true_self_weight = config.model.self_weight
    print("\nSummary")
    print(f"  (a) baseline dataset, argmax λ (width), true λ = {true_self_weight}: " + ", ".join(
        f"{name} {argmaxes[name]:.3f} ({widths[name]:.3f})" for name in argmaxes))
    print("  (b) mean bias of λ̂ [mean profile width] by σ_m:")
    for noise_sd in MEASUREMENT_NOISE_LEVELS:
        print(f"      σ_m={noise_sd:<5g} " + ", ".join(
            f"{method} {bias_summary[noise_sd][method]['mean'] - true_self_weight:+.3f} "
            f"[{bias_summary[noise_sd][method]['width']:.3f}]" for method in METHODS))
    print("  figures: " + ", ".join(str(path.relative_to(REPO_ROOT)) for path in profile_paths + bias_paths))


if __name__ == "__main__":
    main()
