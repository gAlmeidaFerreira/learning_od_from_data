# Particle-filter estimate of the log-likelihood.

from typing import Literal, cast, overload

import numpy as np
from scipy.special import logsumexp

from src import model
from src.config import ModelParams, SimulationConfig


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of resampled particles by systematic resampling (one uniform draw)."""
    n_particles = weights.size
    positions = rng.uniform(0.0, 1.0 / n_particles) + np.arange(n_particles) / n_particles
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0
    return np.searchsorted(cumulative_weights, positions)


# Joint filter: each particle carries every agent's x and s, so neighbours' expressions are
# tracked exactly. A per-agent filter with plug-in neighbour inputs (plan 1.2, version a) will
# sit next to it as a separate function with the same signature.


@overload
def joint_pf_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    rng: np.random.Generator,
    ess_threshold: float = 0.5,
    return_filtered: Literal[False] = False,
) -> float: ...


@overload
def joint_pf_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    rng: np.random.Generator,
    ess_threshold: float = 0.5,
    *,
    return_filtered: Literal[True],
) -> tuple[float, dict]: ...


def joint_pf_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    rng: np.random.Generator,
    ess_threshold: float = 0.5,
    return_filtered: bool = False,
) -> float | tuple[float, dict]:
    """Bootstrap particle-filter estimate of log p(Y | params); y is (T, n), NaN = no post.

    Particles are weighted by p(Y_t | x_t) with the expression integrated out, then s_t is
    drawn from p(s_t | x_t, Y_t). Resamples when ESS < ess_threshold * n_particles.
    If return_filtered, also returns {"filtered_mean": (T, n), "ess": (T,), "n_resamples": int},
    where filtered_mean is the weighted mean of x after weighting, before resampling.
    """
    n_steps, n_agents = y.shape
    log_weights = np.full(n_particles, -np.log(n_particles))
    total_loglik = 0.0
    filtered_mean = np.empty((n_steps, n_agents))
    ess_history = np.empty(n_steps)
    n_resamples = 0

    opinions = rng.normal(sim_params.x0_mean, sim_params.x0_sd, size=(n_particles, n_agents))
    for t in range(n_steps):
        if not np.isnan(y[t]).all():
            log_incremental_weights = model.obs_loglik_given_opinion(y[t], opinions, model_params)
            unnormalised = log_weights + log_incremental_weights
            step_loglik = cast(float, logsumexp(unnormalised))
            total_loglik += step_loglik
            log_weights = unnormalised - step_loglik

        expressions = model.sample_expression_given_obs(opinions, y[t], model_params, rng)

        weights = np.exp(log_weights)
        ess = 1.0 / np.sum(weights**2)
        ess_history[t] = ess
        filtered_mean[t] = weights @ opinions

        if ess < ess_threshold * n_particles:
            indices = systematic_resample(weights, rng)
            opinions, expressions = opinions[indices], expressions[indices]
            log_weights = np.full(n_particles, -np.log(n_particles))
            n_resamples += 1

        if t < n_steps - 1:            # propagate to the next step: x_{t+1} from x_t and s_t
            opinions = model.step(opinions, expressions, W, model_params, rng)

    if return_filtered:
        return float(total_loglik), {
            "filtered_mean": filtered_mean, "ess": ess_history, "n_resamples": n_resamples,
        }
    return float(total_loglik)


# Per-agent filter (plan 1.2, version a): agent i's filter tracks only x_i and replaces the
# neighbours' hidden expressions s_{j,t-1} by their observed posts Y_{j,t-1} (plug-in inputs).
# All n filters run at once in arrays of shape (N, n); column i is agent i's filter and
# columns never interact. Fast, but the plug-in inputs carry measurement noise (bias).


def _per_agent_obs_loglik(y_t: np.ndarray, x: np.ndarray, model_params: ModelParams) -> np.ndarray:
    """Elementwise log N(y_t[i]; x[:, i], σ_e² + σ_m²), shape (N, n) — not summed over agents."""
    if model_params.observation == "ordinal":
        raise NotImplementedError("Ordinal observation is not implemented yet")
    if model_params.observation != "continuous":
        raise ValueError(f"Unsupported observation: {model_params.observation!r}")
    total_var = model_params.expression_noise_sd**2 + model_params.measurement_noise_sd**2
    return -0.5 * ((y_t - x) ** 2 / total_var + np.log(2 * np.pi * total_var))


@overload
def per_agent_pf_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    rng: np.random.Generator,
    ess_threshold: float = 0.5,
    return_filtered: Literal[False] = False,
) -> float: ...


@overload
def per_agent_pf_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    rng: np.random.Generator,
    ess_threshold: float = 0.5,
    *,
    return_filtered: Literal[True],
) -> tuple[float, dict]: ...


def per_agent_pf_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    rng: np.random.Generator,
    ess_threshold: float = 0.5,
    return_filtered: bool = False,
) -> float | tuple[float, dict]:
    """Per-agent bootstrap PF with plug-in neighbour inputs; returns the summed log-likelihood.

    y is (T, n) with no missing posts. Each agent's column is weighted by its own post and
    resampled on its own when its ESS < ess_threshold * n_particles.
    If return_filtered, also returns {"per_agent_loglik": (n,), "filtered_mean": (T, n),
    "ess": (T, n), "n_resamples": (n,)}; filtered_mean is taken before resampling.
    """
    if np.isnan(y).any():
        raise NotImplementedError("missing posts are not supported by the per-agent filter yet")

    n_steps, n_agents = y.shape
    uniform_log_weight = -np.log(n_particles)
    log_weights = np.full((n_particles, n_agents), uniform_log_weight)
    per_agent_loglik = np.zeros(n_agents)
    filtered_mean = np.empty((n_steps, n_agents))
    ess_history = np.empty((n_steps, n_agents))
    n_resamples = np.zeros(n_agents, dtype=int)

    opinions = rng.normal(sim_params.x0_mean, sim_params.x0_sd, size=(n_particles, n_agents))
    for t in range(n_steps):
        if t > 0:
            # Plug-in: y[t-1] (shape (n,)) stands in for every particle's neighbour expressions.
            # Inside step, y[t-1] @ W.T = (W y_{t-1}) has shape (n,) and broadcasts over the
            # particle rows, so column i receives the input (W y_{t-1})_i.
            opinions = model.step(opinions, y[t - 1], W, model_params, rng)

        unnormalised = log_weights + _per_agent_obs_loglik(y[t], opinions, model_params)
        # SciPy's stubs allow a tuple (only with return_sign=True, unused here).
        step_loglik = cast(np.ndarray, logsumexp(unnormalised, axis=0))
        per_agent_loglik += step_loglik
        log_weights = unnormalised - step_loglik

        weights = np.exp(log_weights)
        ess = 1.0 / np.sum(weights**2, axis=0)
        ess_history[t] = ess
        filtered_mean[t] = np.sum(weights * opinions, axis=0)

        for agent in np.flatnonzero(ess < ess_threshold * n_particles):
            indices = systematic_resample(weights[:, agent], rng)
            opinions[:, agent] = opinions[indices, agent]
            log_weights[:, agent] = uniform_log_weight
            n_resamples[agent] += 1

    total_loglik = float(per_agent_loglik.sum())
    if return_filtered:
        return total_loglik, {
            "per_agent_loglik": per_agent_loglik, "filtered_mean": filtered_mean,
            "ess": ess_history, "n_resamples": n_resamples,
        }
    return total_loglik
