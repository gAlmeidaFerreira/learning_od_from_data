# Bias from plug-in neighbour inputs: the per-agent model replaces the neighbours' hidden
# expressions s by their observed posts Y. Compares its estimate of λ with the exact joint
# Kalman likelihood, on one dataset and across measurement-noise levels (report only).
#
# Run from the repository root:  python -m checks.check_plugin_bias

import os

# One BLAS thread per process: on these small matrices (2n = 100) multithreaded BLAS is ~25x
# slower (measured ~1.1 s vs ~40 ms per joint Kalman evaluation). Must be set before NumPy loads.
for blas_thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(blas_thread_variable, "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import kalman, simulate
from src.config import Config, load_config, save_config
from src.particle_filter import per_agent_pf_loglik

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "checks" / "plugin_bias"

N_PARTICLES = 500
PROFILE_SEED = 0
PROFILE_GRID = np.linspace(0.05, 0.95, 37)
ESTIMATION_GRID = np.linspace(0.05, 0.95, 91)
MEASUREMENT_NOISE_LEVELS = [0.02, 0.05, 0.1, 0.2]
DATASET_SEEDS = range(100, 120)


def check_profiles(config: Config, dataset: dict) -> tuple[dict, Path]:
    """(a) λ profiles on one dataset: joint Kalman vs per-agent Kalman vs per-agent PF."""
    y, W = dataset["y"], dataset["W"]
    joint_profile, per_agent_profile, pf_profile = [], [], []
    for self_weight in PROFILE_GRID:
        params = replace(config.model, self_weight=self_weight)
        joint_profile.append(kalman.kalman_loglik(y, W, params, config.simulation))
        per_agent_profile.append(kalman.per_agent_kalman_loglik(y, W, params, config.simulation))
        pf_profile.append(per_agent_pf_loglik(
            y, W, params, config.simulation, N_PARTICLES, np.random.default_rng(PROFILE_SEED)
        ))
    joint_profile, per_agent_profile, pf_profile = map(np.array, (joint_profile, per_agent_profile, pf_profile))
    argmaxes = {
        "joint Kalman": PROFILE_GRID[joint_profile.argmax()],
        "per-agent Kalman": PROFILE_GRID[per_agent_profile.argmax()],
        "per-agent PF": PROFILE_GRID[pf_profile.argmax()],
    }
    print(f"\n(a) λ profiles ({PROFILE_GRID.size} points; PF N = {N_PARTICLES}, seed {PROFILE_SEED} at every point)")
    for name, value in argmaxes.items():
        print(f"    argmax {name:<17} = {value:.3f}   (true λ = {config.model.self_weight})")

    # The per-agent PF estimates the per-agent Kalman curve, so it is shifted by the same constant.
    per_agent_peak = per_agent_profile.max()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(PROFILE_GRID, joint_profile - joint_profile.max(), color="0.15", linewidth=1.8,
            label="joint Kalman (exact model)")
    ax.plot(PROFILE_GRID, per_agent_profile - per_agent_peak, color="C1", linewidth=1.8,
            label="per-agent Kalman (plug-in model)")
    ax.plot(PROFILE_GRID, pf_profile - per_agent_peak, color="C0", linewidth=1.2, linestyle="--",
            marker="o", markersize=3.5, label=rf"per-agent PF ($N = {N_PARTICLES}$, common seed)")
    ax.axvline(config.model.self_weight, color="0.5", linestyle=":", linewidth=1.2,
               label=rf"true $\lambda = {config.model.self_weight}$")
    ax.set_ylim(-60, 5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"relative log-likelihood $\ell(\lambda) - \ell(\hat\lambda)$")
    ax.set_title(rf"Plug-in bias in the $\lambda$ profile ($n = {config.network.n_agents}$, "
                 rf"$T = {config.simulation.T}$, $\sigma_m = {config.model.measurement_noise_sd}$)")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "lambda_profiles.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return argmaxes, figure_path



def estimate_self_weight(task: tuple[Config, np.ndarray, float, int]) -> tuple[float, float, float]:
    """λ̂ by grid argmax under the joint and the per-agent Kalman likelihood for one dataset."""
    config, W, measurement_noise_sd, dataset_seed = task
    noise_config = replace(config, model=replace(config.model, measurement_noise_sd=measurement_noise_sd))
    y = simulate.simulate(noise_config, W=W, rng=np.random.default_rng(dataset_seed))["y"]
    joint_profile, per_agent_profile = [], []
    for self_weight in ESTIMATION_GRID:
        params = replace(noise_config.model, self_weight=self_weight)
        joint_profile.append(kalman.kalman_loglik(y, W, params, noise_config.simulation))
        per_agent_profile.append(kalman.per_agent_kalman_loglik(y, W, params, noise_config.simulation))
    return (measurement_noise_sd,
            float(ESTIMATION_GRID[np.argmax(joint_profile)]),
            float(ESTIMATION_GRID[np.argmax(per_agent_profile)]))



def check_bias_vs_measurement_noise(config: Config, W: np.ndarray) -> tuple[dict, Path]:
    """(b) λ̂ from joint vs per-agent Kalman across measurement-noise levels (report only)."""
    tasks = [(config, W, noise_sd, seed) for noise_sd in MEASUREMENT_NOISE_LEVELS for seed in DATASET_SEEDS]
    with ProcessPoolExecutor() as pool:
        estimates = list(pool.map(estimate_self_weight, tasks))

    true_self_weight = config.model.self_weight
    summary = {}
    print(f"\n(b) λ̂ vs measurement noise: {len(DATASET_SEEDS)} datasets per σ_m on the same network "
          f"(data seeds {DATASET_SEEDS.start}..{DATASET_SEEDS.stop - 1}), grid of {ESTIMATION_GRID.size} points")
    print(f"{'σ_m':>6}  {'joint λ̂ (mean ± sd)':>21}  {'per-agent λ̂ (mean ± sd)':>25}  "
          f"{'bias joint':>10}  {'bias per-agent':>14}")
    for noise_sd in MEASUREMENT_NOISE_LEVELS:
        joint_estimates = np.array([joint for level, joint, _ in estimates if level == noise_sd])
        per_agent_estimates = np.array([per_agent for level, _, per_agent in estimates if level == noise_sd])
        summary[noise_sd] = {
            "joint_mean": joint_estimates.mean(), "joint_sd": joint_estimates.std(ddof=1),
            "per_agent_mean": per_agent_estimates.mean(), "per_agent_sd": per_agent_estimates.std(ddof=1),
        }
        row = summary[noise_sd]
        print(f"{noise_sd:>6}  {row['joint_mean']:>12.3f} ± {row['joint_sd']:.3f}  "
              f"{row['per_agent_mean']:>16.3f} ± {row['per_agent_sd']:.3f}  "
              f"{row['joint_mean'] - true_self_weight:>+10.3f}  {row['per_agent_mean'] - true_self_weight:>+14.3f}")

    noise_levels = np.array(MEASUREMENT_NOISE_LEVELS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, key, color, offset in [("joint Kalman", "joint", "0.15", 0.96), ("per-agent Kalman", "per_agent", "C1", 1.04)]:
        ax.errorbar(noise_levels * offset, [summary[level][f"{key}_mean"] for level in MEASUREMENT_NOISE_LEVELS],
                    yerr=[summary[level][f"{key}_sd"] for level in MEASUREMENT_NOISE_LEVELS],
                    color=color, linewidth=1.6, marker="o", markersize=6, capsize=3, label=name)
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
    figure_path = OUTPUT_DIR / "bias_vs_measurement_noise.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return summary, figure_path



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(REPO_ROOT / "configs" / "degroot_base.yaml")
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    print(f"Dataset: n = {config.network.n_agents}, T = {config.simulation.T}, "
          f"σ_e = {config.model.expression_noise_sd}, σ_m = {config.model.measurement_noise_sd}")

    argmaxes, profile_path = check_profiles(config, dataset)
    bias_summary, bias_path = check_bias_vs_measurement_noise(config, dataset["W"])

    true_self_weight = config.model.self_weight
    print("\nSummary")
    print("  (a) argmax λ: " + ", ".join(f"{name} {value:.3f}" for name, value in argmaxes.items())
          + f" (true {true_self_weight}); figure: {profile_path.relative_to(REPO_ROOT)}")
    print("  (b) mean bias of λ̂ (joint / per-agent): " + ", ".join(
        f"σ_m={level:g}: {bias_summary[level]['joint_mean'] - true_self_weight:+.3f} / "
        f"{bias_summary[level]['per_agent_mean'] - true_self_weight:+.3f}" for level in MEASUREMENT_NOISE_LEVELS
    ) + f"; figure: {bias_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
