# Hidden opinions with parameter uncertainty (plan 1.2, "Hidden opinions"), on the baseline dataset
# of notebook 02 with "filtered_var" neighbour inputs:
#   (a) exact MH on the per-agent Kalman likelihood with an OpinionAccumulator, against
#       estimate_hidden_opinions with the per-agent Kalman filter re-run at every kept draw of the
#       same chain; both use the same states, so they must agree up to floating-point error;
#   (b) correlated PMMH on the noise-driven per-agent PF (N, ρ from configs/inference_base.yaml)
#       with an accumulator: RMSE against the true x, share of true x inside ±2 sd, and share of
#       the variance that comes from parameter uncertainty;
#   (c) one agent's true x, the PMMH estimate and its ±2 sd band.
# Both chains start at the true ψ, as in check_pmmh.py.
#
# Run from the repository root:  python -m checks.check_states

import os

# One BLAS thread per process: the two chains run in parallel processes (see check_pmmh.py).
for blas_thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(blas_thread_variable, "1")

import shutil
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from checks.common import (
    CHECK_RESULTS_DIR,
    INFERENCE_CONFIG_PATH,
    REPO_ROOT,
    describe_dataset,
    load_base_config,
    per_agent_pf_loglik_function,
)
from checks.plotting import EXACT_STYLE, dataset_title, save_figure, style_axes
from src import kalman, simulate
from src.config import Config, save_config
from src.inference import InferenceConfig, from_unconstrained, load_inference_config, pseudo_marginal_mh, to_unconstrained
from src.states import OpinionAccumulator, estimate_hidden_opinions

OUTPUT_DIR = CHECK_RESULTS_DIR / "check_states"
INPUT_MODE = "filtered_var"
SHOWN_AGENT = 3                      # also shown in notebook 02
BAND_SDS = 2.0
AGREEMENT_TOLERANCE = 1e-10          # (a): max |difference| allowed between the two estimates


def run_chain(task: tuple[str, Config, dict, InferenceConfig, np.random.SeedSequence]) -> tuple[str, dict]:
    """Run chain "exact" (per-agent Kalman) or "pmmh" (correlated PMMH) with an accumulator."""
    name, config, dataset, inference_config, seed_sequence = task
    y, W = dataset["y"], dataset["W"]
    pseudo_marginal = {}
    if name == "exact":
        def loglik_fn(params, noise):
            return kalman.per_agent_kalman_loglik(y, W, params, config.simulation, return_filtered=True,
                                                  input_mode=INPUT_MODE)
    else:
        loglik_fn, draw_noise = per_agent_pf_loglik_function(config, dataset, inference_config, return_filtered=True)
        pseudo_marginal = {"draw_noise": draw_noise, "rho": inference_config.pmmh.rho}
    accumulator = OpinionAccumulator(*y.shape)
    result = pseudo_marginal_mh(loglik_fn, to_unconstrained(config.model), config.model, inference_config.priors,
                                inference_config.mcmc, np.random.default_rng(seed_sequence),
                                accumulator=accumulator, **pseudo_marginal)
    result["accumulator"] = accumulator
    return name, result


def check_accumulator_agreement(config: Config, dataset: dict, chain: dict, burn_in: int) -> float:
    """(a) Re-run the per-agent Kalman filter at every kept draw; max |difference| from the accumulator."""
    y, W = dataset["y"], dataset["W"]
    kept_draws = [from_unconstrained(theta, config.model) for theta in chain["theta"][burn_in:]]
    rerun_mean, rerun_var = estimate_hidden_opinions(
        kept_draws,
        lambda params: kalman.per_agent_kalman_loglik(y, W, params, config.simulation, return_filtered=True,
                                                      input_mode=INPUT_MODE)[1],
    )
    accumulated_mean, accumulated_var = chain["accumulator"].result()
    max_difference = float(max(np.abs(rerun_mean - accumulated_mean).max(), np.abs(rerun_var - accumulated_var).max()))
    print(f"\n(a) Exact MH, per-agent Kalman ({INPUT_MODE}): accumulator vs estimate_hidden_opinions over "
          f"{len(kept_draws)} kept draws")
    print(f"    max |difference| in mean and variance = {max_difference:.2e} (tolerance {AGREEMENT_TOLERANCE:g}) — "
          + ("agree" if max_difference <= AGREEMENT_TOLERANCE else "DISAGREE"))
    return max_difference


def state_summary(accumulator: OpinionAccumulator, true_opinions: np.ndarray) -> dict:
    """RMSE against the true x, coverage of ±BAND_SDS sd, and the parameter share of the variance."""
    mean, var = accumulator.result()
    state_var = accumulator.sum_var / accumulator.count       # average filtered variance
    return {
        "rmse": float(np.sqrt(np.mean((mean - true_opinions) ** 2))),
        "coverage": float(np.mean(np.abs(true_opinions - mean) <= BAND_SDS * np.sqrt(var))),
        "parameter_share": float(1.0 - state_var.sum() / var.sum()),
        "median_sd": float(np.median(np.sqrt(var))),
    }


def plot_agent(path, config: Config, dataset: dict, accumulator: OpinionAccumulator, agent: int):
    """(c) One agent's true x, labels, estimated opinion and its ±BAND_SDS sd band."""
    mean, var = accumulator.result()
    time_steps = np.arange(mean.shape[0])
    band = BAND_SDS * np.sqrt(var[:, agent])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(time_steps, mean[:, agent] - band, mean[:, agent] + band, color="C3", alpha=0.25,
                    label=rf"$\pm {BAND_SDS:g}$ sd (state + parameter uncertainty)")
    ax.plot(time_steps, mean[:, agent], color="C3", linewidth=1.3, label="posterior mean (correlated PMMH)")
    ax.plot(time_steps, dataset["x"][:, agent], **EXACT_STYLE, label=r"true $x_{i,t}$")
    ax.plot(time_steps, dataset["y"][:, agent], color="0.45", linestyle="none", marker="x", markersize=3.5,
            label=r"label $Y_{i,t}$")
    style_axes(ax, r"time $t$", rf"opinion of agent $i = {agent}$",
               "Filtered opinion with the parameters integrated out " + dataset_title(config),
               legend_location="upper right", legend_fontsize=8)
    return save_figure(fig, path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_base_config()
    inference_config = load_inference_config(INFERENCE_CONFIG_PATH)
    if inference_config.pmmh.input_mode != INPUT_MODE:   # both chains must use the same neighbour inputs
        raise ValueError(f"check_states needs pmmh.input_mode = {INPUT_MODE!r}, got {inference_config.pmmh.input_mode!r}")
    save_config(config, OUTPUT_DIR / "config.yaml")
    shutil.copyfile(INFERENCE_CONFIG_PATH, OUTPUT_DIR / "inference_config.yaml")

    full_dataset = simulate.simulate(config)
    dataset = {"x": full_dataset["x"], "y": full_dataset["y"], "W": full_dataset["W"]}
    print(describe_dataset(config))
    print(f"MCMC: {inference_config.mcmc}; PMMH: {inference_config.pmmh}")

    chain_names = ("exact", "pmmh")
    chain_seeds = np.random.SeedSequence(inference_config.seed).spawn(len(chain_names))
    tasks = [(name, config, dataset, inference_config, seed) for name, seed in zip(chain_names, chain_seeds)]
    with ProcessPoolExecutor(max_workers=len(chain_names)) as pool:
        chains = dict(pool.map(run_chain, tasks))

    max_difference = check_accumulator_agreement(config, dataset, chains["exact"], inference_config.mcmc.burn_in)

    print(f"\n(b) Hidden opinions with ψ integrated out, {chains['pmmh']['accumulator'].count} kept draws per chain")
    print(f"    {'chain':<6} {'RMSE':>7} {f'inside ±{BAND_SDS:g} sd':>15} {'parameter share':>16} {'median sd':>10} "
          f"{'runtime':>8}")
    summaries = {}
    for name in chain_names:
        summaries[name] = state_summary(chains[name]["accumulator"], dataset["x"])
        row = summaries[name]
        print(f"    {name:<6} {row['rmse']:>7.4f} {row['coverage']:>15.1%} {row['parameter_share']:>16.1%} "
              f"{row['median_sd']:>10.4f} {chains[name]['runtime_s']:>7.0f}s")
    print("    expected for pmmh: RMSE about 0.053, about 95% inside, parameter share about 1%")

    figure_path = plot_agent(OUTPUT_DIR / "hidden_opinion_agent.png", config, dataset,
                             chains["pmmh"]["accumulator"], SHOWN_AGENT)
    print("\nSummary")
    print(f"  (a) accumulator vs re-run: max |difference| = {max_difference:.2e}")
    pmmh_row = summaries["pmmh"]
    print(f"  (b) PMMH: RMSE {pmmh_row['rmse']:.4f}, {pmmh_row['coverage']:.1%} inside ±{BAND_SDS:g} sd, "
          f"{pmmh_row['parameter_share']:.1%} of the variance from parameter uncertainty")
    print(f"  (c) figure: {figure_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
