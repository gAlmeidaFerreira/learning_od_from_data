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
    describe_dataset,
    load_base_config,
    profile_width,
    self_weight_profile,
    with_model_params,
)
from checks.plotting import (
    ESTIMATE_STYLE,
    EXACT_STYLE,
    POINTS_STYLE,
    ProfileCurve,
    dataset_title,
    mark_true_value,
    plot_self_weight_profiles,
    save_figure,
    set_log_ticks,
    style_axes,
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
    input_mode: InputMode               # declared so the lambda below keeps the Literal type
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

    # The per-agent PF estimates the per-agent Kalman curve, so it is shifted by that curve's peak.
    sigma_m_title = dataset_title(config, rf"$\sigma_m = {config.model.measurement_noise_sd}$")
    method_styles = {method: {"color": METHOD_COLORS[method], "linewidth": 1.8} for method in METHODS}
    figure_paths = [
        plot_self_weight_profiles(  # original figure: observed (plug-in) inputs only
            OUTPUT_DIR / "lambda_profiles.png", PROFILE_GRID,
            [ProfileCurve(kalman_profiles["joint"], METHOD_LABELS["joint"], EXACT_STYLE),
             ProfileCurve(kalman_profiles["observed"], "per-agent Kalman (plug-in model)", method_styles["observed"]),
             ProfileCurve(pf_profiles["observed"], rf"per-agent PF ($N = {N_PARTICLES}$, common seed)",
                          {**ESTIMATE_STYLE, "color": "C0"}, peak=kalman_profiles["observed"].max())],
            true_self_weight, rf"Plug-in bias in the $\lambda$ profile {sigma_m_title}",
        ),
        plot_self_weight_profiles(  # every input mode
            OUTPUT_DIR / "lambda_profiles_by_input_mode.png", PROFILE_GRID,
            [ProfileCurve(kalman_profiles[method], METHOD_LABELS[method], method_styles[method]) for method in METHODS]
            + [ProfileCurve(pf_profiles["filtered_var"],
                            rf"per-agent PF, filtered inputs + variance ($N = {N_PARTICLES}$, common seed)",
                            {**ESTIMATE_STYLE, "color": METHOD_COLORS["filtered_var"]},
                            peak=kalman_profiles["filtered_var"].max())],
            true_self_weight, rf"$\lambda$ profiles by neighbour-input mode {sigma_m_title}",
            width_level=WIDTH_DROP, figsize=(8, 4.8), legend_fontsize=8.5,
        ),
    ]
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

    def draw_estimates(ax, methods: dict[str, str], offsets: dict[str, float]) -> None:
        """λ̂ mean ± sd for each method, offset along the log x axis so the error bars do not overlap."""
        for method, label in methods.items():
            ax.errorbar(np.array(MEASUREMENT_NOISE_LEVELS) * offsets[method],
                        [summary[level][method]["mean"] for level in MEASUREMENT_NOISE_LEVELS],
                        yerr=[summary[level][method]["sd"] for level in MEASUREMENT_NOISE_LEVELS],
                        color=METHOD_COLORS[method], capsize=3, label=label, **POINTS_STYLE)
        mark_true_value(ax, true_self_weight, axis="y")
        set_log_ticks(ax, MEASUREMENT_NOISE_LEVELS)

    noise_xlabel = r"measurement noise $\sigma_m$"
    estimate_ylabel = r"$\hat\lambda$ (mean $\pm$ sd over datasets)"
    datasets_title = dataset_title(config, rf"{len(DATASET_SEEDS)} datasets per $\sigma_m$")

    # Original figure: joint vs per-agent Kalman with observed (plug-in) inputs.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_estimates(ax, {"joint": "joint Kalman", "observed": "per-agent Kalman"}, {"joint": 0.96, "observed": 1.04})
    style_axes(ax, noise_xlabel, estimate_ylabel, f"Plug-in bias vs. measurement noise {datasets_title}")
    figure_paths = [save_figure(fig, OUTPUT_DIR / "bias_vs_measurement_noise.png")]

    # Every method: λ̂ (left) and profile width (right).
    fig, (estimate_ax, width_ax) = plt.subplots(1, 2, figsize=(12, 4.6))
    draw_estimates(estimate_ax, METHOD_LABELS, dict(zip(METHODS, (0.94, 0.98, 1.02, 1.06))))
    for method in METHODS:
        width_ax.plot(MEASUREMENT_NOISE_LEVELS, [summary[level][method]["width"] for level in MEASUREMENT_NOISE_LEVELS],
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method], **POINTS_STYLE)
    set_log_ticks(width_ax, MEASUREMENT_NOISE_LEVELS)
    style_axes(estimate_ax, noise_xlabel, estimate_ylabel, r"Estimate of $\lambda$", legend_fontsize=8.5)
    style_axes(width_ax, noise_xlabel,
               rf"profile width: range of $\lambda$ with $\ell \geq \ell_{{\max}} - {WIDTH_DROP:g}$",
               "Profile width (smaller = more confident)", legend=False)
    fig.suptitle(f"Neighbour-input modes vs. joint Kalman {datasets_title}")
    figure_paths.append(save_figure(fig, OUTPUT_DIR / "bias_vs_measurement_noise_by_input_mode.png"))
    return summary, figure_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_base_config()
    save_config(config, OUTPUT_DIR / "config.yaml")

    dataset = simulate.simulate(config)
    print(describe_dataset(config))

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
