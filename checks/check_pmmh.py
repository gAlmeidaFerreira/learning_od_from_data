# Posterior of ψ = (λ, σ_p, σ_e) by exact MH on the Kalman likelihoods (references) and by
# pseudo-marginal MH on the per-agent particle filter (report only, no assertions yet).
#   A  exact MH, joint Kalman likelihood (exact model)
#   B  exact MH, per-agent Kalman, "filtered_var" inputs
#   C  exact MH, per-agent Kalman, "filtered" inputs
#   D  correlated PMMH, noise-driven per-agent PF, "filtered_var", N and ρ from the config
#   E  standard PMMH, as D with ρ = 0 (shorter run)
# D vs B is a sampler check (same likelihood up to particle noise); B, C vs A measure the cost of
# the per-agent approximation with σ_p and σ_e estimated jointly. All chains start at the true ψ.
#
# Run from the repository root:  python -m checks.check_pmmh

import os

# One BLAS thread per process: on these small matrices multithreaded BLAS is much slower, and the
# chains run in parallel processes instead. Must be set before NumPy loads.
for blas_thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(blas_thread_variable, "1")

import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from checks.common import CHECK_RESULTS_DIR, REPO_ROOT, describe_dataset, load_base_config
from checks.plotting import mark_true_value, save_figure, style_axes
from src import kalman, simulate
from src.config import Config, save_config
from src.inference import (
    PARAMETER_NAMES,
    InferenceConfig,
    autocorrelation,
    effective_sample_size,
    from_unconstrained,
    load_inference_config,
    pseudo_marginal_mh,
    to_unconstrained,
)
from src.particle_filter import crank_nicolson, draw_per_agent_noise, per_agent_pf_loglik_given_noise

OUTPUT_DIR = CHECK_RESULTS_DIR / "pmmh"
INFERENCE_CONFIG_PATH = REPO_ROOT / "configs" / "inference_base.yaml"

CHAINS = {
    "A": "exact MH, joint Kalman",
    "B": "exact MH, per-agent Kalman, filtered_var",
    "C": "exact MH, per-agent Kalman, filtered",
    "D": "correlated PMMH, per-agent PF, filtered_var",
    "E": "standard PMMH (ρ = 0), per-agent PF, filtered_var",
}
CHAIN_COLORS = {"A": "0.15", "B": "C0", "C": "C2", "D": "C3", "E": "C1"}
STANDARD_PMMH_ITERATIONS, STANDARD_PMMH_BURN_IN = 2000, 500
PARAMETER_SYMBOLS = {"self_weight": r"\lambda", "process_noise_sd": r"\sigma_p", "expression_noise_sd": r"\sigma_e"}
PARAMETER_LABELS = {"self_weight": "λ", "process_noise_sd": "σ_p", "expression_noise_sd": "σ_e"}
NOISE_CHECK_DRAWS = 50
NOISE_CHECK_SEED = 2024
AUTOCORRELATION_LAGS = (1, 10, 50, 100, 200)


def per_agent_pf_loglik_function(config: Config, dataset: dict, inference_config: InferenceConfig):
    """(loglik_fn(params, u), draw_noise(rng)) for the noise-driven per-agent PF."""
    y, W = dataset["y"], dataset["W"]

    def loglik_fn(params, noise):
        return per_agent_pf_loglik_given_noise(y, W, params, config.simulation, noise,
                                               input_mode=inference_config.pmmh.input_mode)

    def draw_noise(rng):
        return draw_per_agent_noise(y.shape[0], inference_config.pmmh.n_particles, y.shape[1], rng)

    return loglik_fn, draw_noise


def run_chain(task: tuple[str, Config, dict, InferenceConfig, np.random.SeedSequence]) -> tuple[str, dict]:
    """Run one chain (A–E) from the true ψ; returns (name, sampler output)."""
    name, config, dataset, inference_config, seed_sequence = task
    y, W = dataset["y"], dataset["W"]
    mcmc_config, pseudo_marginal = inference_config.mcmc, {}
    if name == "A":
        loglik_fn = lambda params, noise: kalman.kalman_loglik(y, W, params, config.simulation)
    elif name in ("B", "C"):
        input_mode = "filtered_var" if name == "B" else "filtered"
        loglik_fn = lambda params, noise: kalman.per_agent_kalman_loglik(
            y, W, params, config.simulation, input_mode=input_mode)
    else:
        loglik_fn, draw_noise = per_agent_pf_loglik_function(config, dataset, inference_config)
        rho = inference_config.pmmh.rho if name == "D" else 0.0
        pseudo_marginal = {"draw_noise": draw_noise, "rho": rho}
        if name == "E":
            mcmc_config = replace(mcmc_config, n_iterations=STANDARD_PMMH_ITERATIONS, burn_in=STANDARD_PMMH_BURN_IN)
    result = pseudo_marginal_mh(loglik_fn, to_unconstrained(config.model), config.model, inference_config.priors,
                                mcmc_config, np.random.default_rng(seed_sequence), **pseudo_marginal)
    result["burn_in"] = mcmc_config.burn_in
    return name, result


def summarise_chain(result: dict) -> dict:
    """Posterior summaries after burn-in: per parameter mean, 95% interval, ESS, ESS/min; correlations."""
    kept = result["params"][result["burn_in"]:]
    minutes = result["runtime_s"] / 60
    by_parameter = {}
    for column, name in enumerate(PARAMETER_NAMES):
        ess = effective_sample_size(kept[:, column])
        by_parameter[name] = {
            "mean": kept[:, column].mean(), "sd": kept[:, column].std(ddof=1),
            "low": np.quantile(kept[:, column], 0.025), "high": np.quantile(kept[:, column], 0.975),
            "ess": ess, "ess_per_minute": ess / minutes,
        }
    return {"parameters": by_parameter, "acceptance": result["accepted"][result["burn_in"]:].mean(),
            "correlation": np.corrcoef(kept.T), "runtime_s": result["runtime_s"], "n_kept": kept.shape[0]}


def print_chain_summaries(summaries: dict, true_values: dict) -> None:
    print("\nPosterior summaries after burn-in (true values: " + ", ".join(
        f"{PARAMETER_LABELS[name]} = {true_values[name]}" for name in PARAMETER_NAMES) + ")")
    print(f"  {'chain':<5} {'param':<5} {'mean':>8} {'2.5%':>8} {'97.5%':>8} {'ESS':>7} {'ESS/min':>8}")
    for chain, summary in summaries.items():
        for name in PARAMETER_NAMES:
            row = summary["parameters"][name]
            print(f"  {chain:<5} {PARAMETER_LABELS[name]:<5} {row['mean']:>8.4f} {row['low']:>8.4f} {row['high']:>8.4f} "
                  f"{row['ess']:>7.0f} {row['ess_per_minute']:>8.1f}")
        print(f"  {chain:<5} acceptance {summary['acceptance']:.3f}, {summary['n_kept']} kept draws, "
              f"runtime {summary['runtime_s']:.0f} s — {CHAINS[chain]}")
    print("\nPosterior correlations (λ, σ_p, σ_e)")
    for chain, summary in summaries.items():
        rows = summary["correlation"]
        print(f"  {chain}: ρ(λ,σ_p) = {rows[0, 1]:+.2f}, ρ(λ,σ_e) = {rows[0, 2]:+.2f}, ρ(σ_p,σ_e) = {rows[1, 2]:+.2f}")


def print_comparison(summaries: dict, reference: str, others: tuple[str, ...], heading: str) -> None:
    """Side-by-side posterior means and intervals; z = mean difference / combined Monte Carlo s.e."""
    print(f"\n{heading}")
    for name in PARAMETER_NAMES:
        reference_row = summaries[reference]["parameters"][name]
        cells = [f"{reference} {reference_row['mean']:.4f} [{reference_row['low']:.4f}, {reference_row['high']:.4f}]"]
        for other in others:
            row = summaries[other]["parameters"][name]
            monte_carlo_se = np.hypot(reference_row["sd"] / np.sqrt(reference_row["ess"]), row["sd"] / np.sqrt(row["ess"]))
            cells.append(f"{other} {row['mean']:.4f} [{row['low']:.4f}, {row['high']:.4f}] "
                         f"(z = {(row['mean'] - reference_row['mean']) / monte_carlo_se:+.1f})")
        print(f"  {PARAMETER_LABELS[name]:<4} " + "   ".join(cells))


def check_likelihood_ratio_noise(config: Config, dataset: dict, inference_config: InferenceConfig, chain_d: dict) -> float:
    """sd of (PF − Kalman) log-likelihood ratio for D's own proposals at D's posterior mean."""
    y, W = dataset["y"], dataset["W"]
    loglik_fn, draw_noise = per_agent_pf_loglik_function(config, dataset, inference_config)
    input_mode, rho = inference_config.pmmh.input_mode, inference_config.pmmh.rho
    theta = chain_d["theta"][chain_d["burn_in"]:].mean(axis=0)
    params = from_unconstrained(theta, config.model)
    kalman_current = kalman.per_agent_kalman_loglik(y, W, params, config.simulation, input_mode=input_mode)
    rng = np.random.default_rng(NOISE_CHECK_SEED)
    gaps = []
    for _ in range(NOISE_CHECK_DRAWS):
        proposed_params = from_unconstrained(rng.multivariate_normal(theta, chain_d["proposal_cov"]), config.model)
        noise = draw_noise(rng)
        proposed_noise = crank_nicolson(noise, rho, rng)
        pf_difference = loglik_fn(proposed_params, proposed_noise) - loglik_fn(params, noise)
        kalman_difference = kalman.per_agent_kalman_loglik(
            y, W, proposed_params, config.simulation, input_mode=input_mode) - kalman_current
        gaps.append(pf_difference - kalman_difference)
    gap_sd = float(np.std(gaps, ddof=1))
    print(f"\nNoise of D's log-likelihood ratio at its posterior mean ({NOISE_CHECK_DRAWS} proposals θ' ~ N(θ, Σ_D), "
          f"u' = crank_nicolson(u, ρ = {rho:g}), N = {inference_config.pmmh.n_particles}):")
    print(f"    sd(D_k − K_k) = {gap_sd:.3f} (target ≤ 1)"
          + ("" if gap_sd <= 1.2 else " — clearly above target: rerun D with n_particles = 200"))
    return gap_sd


def print_noise_mixing(results: dict, summaries: dict) -> None:
    kept_loglik = results["D"]["loglik"][results["D"]["burn_in"]:]
    correlations = autocorrelation(kept_loglik)
    print("\nMixing of u: autocorrelation of D's log-likelihood trace after burn-in")
    print("    " + ", ".join(f"lag {lag}: {correlations[lag]:+.2f}" for lag in AUTOCORRELATION_LAGS)
          + f"; ESS of the trace = {effective_sample_size(kept_loglik):.0f}")
    print("    ESS of D vs B (same likelihood, exact): " + ", ".join(
        f"{PARAMETER_LABELS[name]} {summaries['D']['parameters'][name]['ess']:.0f} vs "
        f"{summaries['B']['parameters'][name]['ess']:.0f}" for name in PARAMETER_NAMES))


def plot_results(results: dict, true_values: dict) -> list:
    figure_paths = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (column, name) in zip(axes, enumerate(PARAMETER_NAMES)):
        for chain in ("A", "B", "D", "E"):
            ax.plot(results[chain]["params"][:, column], color=CHAIN_COLORS[chain], linewidth=0.5, alpha=0.8,
                    label=f"{chain}: {CHAINS[chain]}")
        mark_true_value(ax, true_values[name], axis="y", symbol=PARAMETER_SYMBOLS[name])
        ax.axvline(results["A"]["burn_in"], color="0.6", linestyle="--", linewidth=0.8)
        style_axes(ax, "iteration" if name == PARAMETER_NAMES[-1] else "", rf"${PARAMETER_SYMBOLS[name]}$",
                   legend=name == PARAMETER_NAMES[0], legend_fontsize=7.5, legend_location="upper right")
    fig.suptitle("Trace plots (dashed: end of burn-in for A–D)")
    figure_paths.append(save_figure(fig, OUTPUT_DIR / "traces.png"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (column, name) in zip(axes, enumerate(PARAMETER_NAMES)):
        for chain in ("A", "B", "C", "D"):
            kept = results[chain]["params"][results[chain]["burn_in"]:, column]
            ax.hist(kept, bins=50, density=True, histtype="step", color=CHAIN_COLORS[chain], linewidth=1.4,
                    label=f"{chain}: {CHAINS[chain]}")
        mark_true_value(ax, true_values[name], symbol=PARAMETER_SYMBOLS[name])
        style_axes(ax, rf"${PARAMETER_SYMBOLS[name]}$", "posterior density" if column == 0 else "",
                   legend=column == 0, legend_fontsize=7.5)
    fig.suptitle("Marginal posteriors after burn-in")
    figure_paths.append(save_figure(fig, OUTPUT_DIR / "marginals.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for chain in ("A", "D"):
        kept = results[chain]["params"][results[chain]["burn_in"]:]
        ax.scatter(kept[:, 0], kept[:, 2], s=3, alpha=0.25, color=CHAIN_COLORS[chain], label=f"{chain}: {CHAINS[chain]}")
    ax.plot(true_values["self_weight"], true_values["expression_noise_sd"], marker="*", markersize=14, color="gold",
            markeredgecolor="black", linestyle="none", label="true values")
    style_axes(ax, r"$\lambda$", r"$\sigma_e$", r"Joint posterior of $\lambda$ and $\sigma_e$", legend_fontsize=8)
    figure_paths.append(save_figure(fig, OUTPUT_DIR / "self_weight_vs_expression_noise.png"))

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(results["D"]["loglik"], color=CHAIN_COLORS["D"], linewidth=0.6)
    ax.axvline(results["D"]["burn_in"], color="0.6", linestyle="--", linewidth=0.8)
    style_axes(ax, "iteration", r"$\log \hat p(Y \mid \psi, u)$",
               "Chain D: log-likelihood estimate of the current state (dashed: end of burn-in)", legend=False)
    figure_paths.append(save_figure(fig, OUTPUT_DIR / "pmmh_loglik_trace.png"))
    return figure_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_base_config()
    inference_config = load_inference_config(INFERENCE_CONFIG_PATH)
    save_config(config, OUTPUT_DIR / "config.yaml")
    shutil.copyfile(INFERENCE_CONFIG_PATH, OUTPUT_DIR / "inference_config.yaml")

    dataset = simulate.simulate(config)
    dataset = {"y": dataset["y"], "W": dataset["W"]}      # only what the workers need
    true_values = {name: getattr(config.model, name) for name in PARAMETER_NAMES}
    print(describe_dataset(config))
    print(f"MCMC: {inference_config.mcmc}; PMMH: {inference_config.pmmh}")

    chain_seeds = np.random.SeedSequence(inference_config.seed).spawn(len(CHAINS))
    tasks = [(name, config, dataset, inference_config, seed) for name, seed in zip(CHAINS, chain_seeds)]
    with ProcessPoolExecutor(max_workers=len(CHAINS)) as pool:
        results = dict(pool.map(run_chain, tasks))
    summaries = {name: summarise_chain(results[name]) for name in CHAINS}

    print_chain_summaries(summaries, true_values)
    print_comparison(summaries, "B", ("D", "E"),
                     "Sampler check: PMMH vs exact MH on the same per-agent likelihood (filtered_var)")
    print_comparison(summaries, "A", ("B", "C"),
                     "Cost of the per-agent approximation: per-agent Kalman vs joint Kalman (exact model)")
    gap_sd = check_likelihood_ratio_noise(config, dataset, inference_config, results["D"])
    print_noise_mixing(results, summaries)
    figure_paths = plot_results(results, true_values)

    print("\nSummary")
    print("  runtimes: " + ", ".join(f"{name} {summaries[name]['runtime_s']:.0f} s" for name in CHAINS))
    print("  acceptance: " + ", ".join(f"{name} {summaries[name]['acceptance']:.2f}" for name in CHAINS))
    print("  min ESS over parameters: " + ", ".join(
        f"{name} {min(row['ess'] for row in summaries[name]['parameters'].values()):.0f}" for name in CHAINS))
    print(f"  D's log-likelihood-ratio noise at its posterior mean: sd = {gap_sd:.3f} (target ≤ 1)")
    print("  figures: " + ", ".join(str(path.relative_to(REPO_ROOT)) for path in figure_paths))


if __name__ == "__main__":
    main()
