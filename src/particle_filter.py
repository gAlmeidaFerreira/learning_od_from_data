# Particle-filter estimate of the log-likelihood.

from typing import Literal, cast, overload

import numpy as np
from scipy.special import logsumexp

from src import model, observation
from src.config import ModelParams, SimulationConfig
from src.neighbour_inputs import InputMode, validate_per_agent_inputs
from src.observation import expression_estimate, gaussian_obs_loglik


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of resampled particles by systematic resampling (one uniform draw)."""
    n_particles = weights.size
    positions = rng.uniform(0.0, 1.0 / n_particles) + np.arange(n_particles) / n_particles
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0
    return np.searchsorted(cumulative_weights, positions)


# Joint filter: each particle carries every agent's x and s, so neighbours' expressions are
# tracked exactly (plan 1.2, version b).


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
            log_incremental_weights = observation.obs_loglik_given_opinion(y[t], opinions, model_params)
            unnormalised = log_weights + log_incremental_weights
            step_loglik = cast(float, logsumexp(unnormalised))
            total_loglik += step_loglik
            log_weights = unnormalised - step_loglik

        expressions = observation.sample_expression_given_obs(opinions, y[t], model_params, rng)

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
# neighbours' hidden expressions s_{j,t-1} by an input chosen with input_mode (see
# src/neighbour_inputs.py). In "filtered_var" each particle draws its own neighbour expressions
# ŝ + sqrt(v) η, η ~ N(0, I). All n filters run at once in arrays of shape (N, n); column i is
# agent i's filter and columns never interact.


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
    *,
    input_mode: InputMode = "observed",
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
    input_mode: InputMode = "observed",
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
    *,
    input_mode: InputMode = "observed",
) -> float | tuple[float, dict]:
    """Per-agent bootstrap PF; returns the log-likelihood summed over agents.

    y is (T, n) with no missing posts. Each agent's column is weighted by its own post and
    resampled on its own when its ESS < ess_threshold * n_particles. input_mode chooses the
    neighbour input (see the comment above).
    If return_filtered, also returns {"per_agent_loglik": (n,), "filtered_mean": (T, n),
    "ess": (T, n), "n_resamples": (n,)}; filtered_mean is taken before resampling.
    """
    validate_per_agent_inputs(y, input_mode)

    n_steps, n_agents = y.shape
    uniform_log_weight = -np.log(n_particles)
    log_weights = np.full((n_particles, n_agents), uniform_log_weight)
    per_agent_loglik = np.zeros(n_agents)
    filtered_mean = np.empty((n_steps, n_agents))
    ess_history = np.empty((n_steps, n_agents))
    n_resamples = np.zeros(n_agents, dtype=int)
    # Neighbours' expressions and their variance, set at the end of every step; first used at t = 1.
    neighbour_expressions = np.zeros(n_agents)
    neighbour_expression_var = np.zeros(n_agents)

    opinions = rng.normal(sim_params.x0_mean, sim_params.x0_sd, size=(n_particles, n_agents))
    for t in range(n_steps):
        if t > 0:
            if input_mode == "filtered_var":
                # Each particle row draws its own neighbour expressions. A row mixes particles of
                # different agents, but columns are independent filters, so sharing a row is fine.
                expressions_input = neighbour_expressions + np.sqrt(neighbour_expression_var) * rng.normal(
                    size=(n_particles, n_agents)
                )
            else:
                # Shape (n,): inside step, input @ W.T = W input has shape (n,) and broadcasts over
                # the particle rows, so column i receives the input (W input)_i.
                expressions_input = neighbour_expressions
            opinions = model.step(opinions, expressions_input, W, model_params, rng)

        unnormalised = log_weights + gaussian_obs_loglik(y[t], opinions, model_params)
        # SciPy's stubs allow a tuple (only with return_sign=True, unused here).
        step_loglik = cast(np.ndarray, logsumexp(unnormalised, axis=0))
        per_agent_loglik += step_loglik
        log_weights = unnormalised - step_loglik

        weights = np.exp(log_weights)
        ess = 1.0 / np.sum(weights**2, axis=0)
        ess_history[t] = ess
        opinion_mean = np.sum(weights * opinions, axis=0)
        filtered_mean[t] = opinion_mean

        if input_mode == "observed":
            neighbour_expressions = y[t]
        else:
            opinion_var = np.sum(weights * (opinions - opinion_mean) ** 2, axis=0)
            neighbour_expressions, neighbour_expression_var = expression_estimate(
                opinion_mean, opinion_var, y[t], model_params
            )

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
