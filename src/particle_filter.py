# Particle-filter estimate of the log-likelihood.

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal, cast, overload

import numpy as np
from scipy.special import logsumexp, ndtr

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


# The two parts that differ between the rng-based and the noise-driven per-agent filter.
DrawStandardNormal = Callable[[str, int], np.ndarray]
"""(kind, t) -> standard normals of shape (N, n); kind is "initial", "inputs" or "process"."""
ResampleColumns = Callable[[int, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""(t, opinions, weights, ess) -> (opinions after resampling, boolean mask (n,) of resampled columns)."""


def _run_per_agent_filter(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    n_particles: int,
    input_mode: InputMode,
    draw_standard_normal: DrawStandardNormal,
    resample_columns: ResampleColumns,
) -> tuple[float, dict]:
    """Loop shared by the per-agent filters; returns (log-likelihood, diagnostics dict)."""
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

    opinions = sim_params.x0_mean + sim_params.x0_sd * draw_standard_normal("initial", 0)
    for t in range(n_steps):
        if t > 0:
            if input_mode == "filtered_var":
                # Each particle row draws its own neighbour expressions. A row mixes particles of
                # different agents, but columns are independent filters, so sharing a row is fine.
                input_noise = draw_standard_normal("inputs", t)
                expressions_input = neighbour_expressions + np.sqrt(neighbour_expression_var) * input_noise
            else:
                # Shape (n,): inside step, input @ W.T = W input has shape (n,) and broadcasts over
                # the particle rows, so column i receives the input (W input)_i.
                expressions_input = neighbour_expressions
            opinions = model.step(opinions, expressions_input, W, model_params, None,
                                  standard_noise=draw_standard_normal("process", t))

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

        opinions, resampled = resample_columns(t, opinions, weights, ess)
        log_weights[:, resampled] = uniform_log_weight
        n_resamples += resampled

    return float(per_agent_loglik.sum()), {
        "per_agent_loglik": per_agent_loglik, "filtered_mean": filtered_mean,
        "ess": ess_history, "n_resamples": n_resamples,
    }


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
    shape = (n_particles, y.shape[1])

    def draw_standard_normal(kind: str, t: int) -> np.ndarray:
        return rng.standard_normal(shape)      # drawn in call order, the same stream as before

    def resample_low_ess_columns(t: int, opinions: np.ndarray, weights: np.ndarray, ess: np.ndarray):
        resampled = ess < ess_threshold * n_particles
        for agent in np.flatnonzero(resampled):
            opinions[:, agent] = opinions[systematic_resample(weights[:, agent], rng), agent]
        return opinions, resampled

    total_loglik, diagnostics = _run_per_agent_filter(
        y, W, model_params, sim_params, n_particles, input_mode, draw_standard_normal, resample_low_ess_columns
    )
    return (total_loglik, diagnostics) if return_filtered else total_loglik


# Noise-driven per-agent filter for correlated PMMH (plan 1.2). Every random number the filter
# uses is a standard normal in a PerAgentNoise u, so log p̂(Y | ψ, u) is a deterministic function
# of (ψ, u). Correlated PMMH moves u by a Crank–Nicolson step, u' = ρ u + sqrt(1 − ρ²) ε, so the
# estimates at the current and proposed points share most of their noise. That only works if the
# output changes smoothly with ψ and u, which rules out two things the rng-based filter does:
#   - an ESS threshold: whether a column resamples flips on or off as ψ or u change slightly,
#     and each flip makes the estimate jump; so this filter resamples every step;
#   - unsorted resampling: the comb picks particles by their position in the array, which is
#     arbitrary, so a tiny change in the weights can hand an offspring to a distant particle;
#     sorting each column by x first (states are scalar) makes nearby comb teeth pick nearby x.


@dataclass(frozen=True)
class PerAgentNoise:
    """All random numbers of the per-agent filter, every entry standard normal.

    initial (N, n): x_0 draws; process (T, N, n): process noise (index 0 unused);
    inputs (T, N, n): "filtered_var" input draws (index 0 unused); resampling (T, n): comb offsets.
    """

    initial: np.ndarray
    process: np.ndarray
    inputs: np.ndarray
    resampling: np.ndarray


def draw_per_agent_noise(
    n_steps: int, n_particles: int, n_agents: int, rng: np.random.Generator
) -> PerAgentNoise:
    """Fresh standard-normal PerAgentNoise for T = n_steps, N = n_particles, n = n_agents."""
    return PerAgentNoise(
        initial=rng.standard_normal((n_particles, n_agents)),
        process=rng.standard_normal((n_steps, n_particles, n_agents)),
        inputs=rng.standard_normal((n_steps, n_particles, n_agents)),
        resampling=rng.standard_normal((n_steps, n_agents)),
    )


def crank_nicolson(noise: PerAgentNoise, rho: float, rng: np.random.Generator) -> PerAgentNoise:
    """Correlated move u' = ρ u + sqrt(1 − ρ²) ε, ε ~ N(0, I), applied to every field.

    Keeps the N(0, I) distribution of u; ρ = 0 gives an independent draw, ρ → 1 keeps u.
    """
    innovation_scale = np.sqrt(1.0 - rho**2)
    return PerAgentNoise(**{
        field.name: rho * getattr(noise, field.name)
        + innovation_scale * rng.standard_normal(getattr(noise, field.name).shape)
        for field in fields(noise)
    })


def _systematic_resample_columns(
    values: np.ndarray, weights: np.ndarray, comb_noise: np.ndarray, sort_particles: bool = True
) -> np.ndarray:
    """Systematic resampling of every column at once; values, weights (N, n), comb_noise (n,).

    Column i is sorted by values[:, i] (unless sort_particles is False) and resampled with the
    comb (U_i + k) / N, k = 0..N-1, U_i = Φ(comb_noise[i]). Returns indices (N, n) into the
    original, unsorted rows of each column.
    """
    n_particles, n_columns = weights.shape
    if sort_particles:
        order = np.argsort(values, axis=0, kind="stable")
    else:
        order = np.broadcast_to(np.arange(n_particles)[:, None], weights.shape)
    cumulative_weights = np.cumsum(np.take_along_axis(weights, order, axis=0), axis=0)
    cumulative_weights[-1] = 1.0
    positions = (ndtr(comb_noise)[None, :] + np.arange(n_particles)[:, None]) / n_particles

    # One searchsorted for all columns: shift column i's cumulative weights and comb by i, so each
    # column occupies its own interval (i, i + 1] of one increasing array (column-major order).
    column_offsets = np.arange(n_columns)
    flat_cumulative = (cumulative_weights + column_offsets).T.ravel()
    flat_positions = (positions + column_offsets).T.ravel()
    flat_indices = np.searchsorted(flat_cumulative, flat_positions)
    sorted_indices = (flat_indices.reshape(n_columns, n_particles) - (column_offsets * n_particles)[:, None]).T
    return np.take_along_axis(order, sorted_indices, axis=0)


def sorted_systematic_resample(values: np.ndarray, weights: np.ndarray, z: float) -> np.ndarray:
    """Indices of particles resampled systematically after sorting them by values.

    Comb offset U = Φ(z) / N, so a standard-normal z drives it smoothly; returns indices into the
    original (unsorted) arrays. values and weights have shape (N,); weights sum to 1.
    """
    return _systematic_resample_columns(values[:, None], weights[:, None], np.array([z]))[:, 0]


@overload
def per_agent_pf_loglik_given_noise(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    noise: PerAgentNoise,
    return_filtered: Literal[False] = False,
    *,
    input_mode: InputMode = "filtered_var",
    sort_particles: bool = True,
) -> float: ...


@overload
def per_agent_pf_loglik_given_noise(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    noise: PerAgentNoise,
    return_filtered: Literal[True],
    *,
    input_mode: InputMode = "filtered_var",
    sort_particles: bool = True,
) -> tuple[float, dict]: ...


def per_agent_pf_loglik_given_noise(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    noise: PerAgentNoise,
    return_filtered: bool = False,
    *,
    input_mode: InputMode = "filtered_var",
    sort_particles: bool = True,
) -> float | tuple[float, dict]:
    """Per-agent bootstrap PF driven by pre-drawn noise; deterministic given its inputs.

    Same model and input modes as per_agent_pf_loglik, but every random number comes from noise
    (N = noise.initial.shape[0]) and every column is resampled at every step, sorted by x unless
    sort_particles is False (diagnostic only; see the comment above). Same return format as
    per_agent_pf_loglik.
    """
    n_steps, n_agents = y.shape
    n_particles = noise.initial.shape[0]
    for name, expected_shape in {"initial": (n_particles, n_agents), "process": (n_steps, n_particles, n_agents),
                                 "inputs": (n_steps, n_particles, n_agents), "resampling": (n_steps, n_agents)}.items():
        if getattr(noise, name).shape != expected_shape:
            raise ValueError(f"noise.{name} has shape {getattr(noise, name).shape}, expected {expected_shape}")

    def draw_standard_normal(kind: str, t: int) -> np.ndarray:
        return noise.initial if kind == "initial" else getattr(noise, kind)[t]

    def resample_every_column(t: int, opinions: np.ndarray, weights: np.ndarray, ess: np.ndarray):
        indices = _systematic_resample_columns(opinions, weights, noise.resampling[t], sort_particles)
        return np.take_along_axis(opinions, indices, axis=0), np.ones(n_agents, dtype=bool)

    total_loglik, diagnostics = _run_per_agent_filter(
        y, W, model_params, sim_params, n_particles, input_mode, draw_standard_normal, resample_every_column
    )
    return (total_loglik, diagnostics) if return_filtered else total_loglik
