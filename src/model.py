# Network, opinion dynamics and observation (noise) layers.

from collections.abc import Callable

import networkx as nx
import numpy as np

from src.config import ModelParams, NetworkConfig


def build_network(cfg: NetworkConfig, rng: np.random.Generator) -> tuple[nx.Graph, np.ndarray]:
    """Build the interaction graph and its row-stochastic influence matrix W.

    W[i, j] = 1/deg(i) for each neighbour j of i (no self-loops);
    an isolated node keeps its own opinion, W[i, i] = 1.
    """
    if cfg.type != "watts_strogatz":
        raise ValueError(f"Unsupported network type: {cfg.type!r}")
    graph = nx.watts_strogatz_graph(cfg.n_agents, cfg.k, cfg.p, seed=int(rng.integers(2**32)))

    adjacency = nx.to_numpy_array(graph, nodelist=range(cfg.n_agents))
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    isolated = degree == 0
    adjacency[isolated, isolated] = 1.0
    degree[isolated] = 1.0
    W = adjacency / degree[:, None]
    return graph, W


def degroot_step(
    x_prev: np.ndarray,
    s_prev: np.ndarray,
    W: np.ndarray,
    params: ModelParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """One DeGroot update from neighbours' expressions.

    x_t = λ x_{t-1} + (1-λ) W s_{t-1} + ε_t,  ε_t ~ N(0, σ_p² I).
    Accepts shape (n,) or batched (N, n).
    """
    self_weight = params.self_weight
    noise = rng.normal(0.0, params.process_noise_sd, size=x_prev.shape)
    return self_weight * x_prev + (1.0 - self_weight) * (s_prev @ W.T) + noise


DYNAMICS: dict[str, Callable[..., np.ndarray]] = {"degroot": degroot_step}


def step(
    x_prev: np.ndarray,
    s_prev: np.ndarray,
    W: np.ndarray,
    params: ModelParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance hidden opinions one step using the rule named by params.dynamics."""
    try:
        update = DYNAMICS[params.dynamics]
    except KeyError:
        raise ValueError(f"Unsupported dynamics: {params.dynamics!r}") from None
    return update(x_prev, s_prev, W, params, rng)


def express(x: np.ndarray, params: ModelParams, rng: np.random.Generator) -> np.ndarray:
    """Expressed signal s = x + ξ, ξ ~ N(0, σ_e² I)."""
    return x + rng.normal(0.0, params.expression_noise_sd, size=x.shape)


def observe(s: np.ndarray, params: ModelParams, rng: np.random.Generator) -> np.ndarray:
    """Researcher's observation of expressions: continuous Y = s + e, e ~ N(0, σ_m² I)."""
    if params.observation == "continuous":
        return s + rng.normal(0.0, params.measurement_noise_sd, size=s.shape)
    if params.observation == "ordinal":
        raise NotImplementedError("Ordinal observation is not implemented yet")
    raise ValueError(f"Unsupported observation: {params.observation!r}")


def obs_loglik_given_opinion(y: np.ndarray, x: np.ndarray, params: ModelParams) -> np.ndarray:
    """log p(Y_t | x_t) with the expression integrated out, one value per particle.

    Sum over observed agents (non-NaN y) of log N(y_i; x_i, σ_e² + σ_m²).
    y: (n,); x: (N, n). Returns (N,), zeros if no agent posted.
    """
    if params.observation == "ordinal":
        raise NotImplementedError("Ordinal observation is not implemented yet")
    if params.observation != "continuous":
        raise ValueError(f"Unsupported observation: {params.observation!r}")

    observed = ~np.isnan(y)
    if not observed.any():
        return np.zeros(x.shape[:-1])
    total_var = params.expression_noise_sd**2 + params.measurement_noise_sd**2
    residuals = y[observed] - x[..., observed]
    return -0.5 * np.sum(residuals**2 / total_var + np.log(2 * np.pi * total_var), axis=-1)


def sample_expression_given_obs(
    x: np.ndarray, y: np.ndarray, params: ModelParams, rng: np.random.Generator
) -> np.ndarray:
    """Draw s_t ~ p(s_t | x_t, Y_t) for every particle.

    With κ = σ_e² / (σ_e² + σ_m²): observed agents s ~ N(x + κ (y − x), (1 − κ) σ_e²);
    unobserved agents s ~ N(x, σ_e²). x: (N, n), y: (n,). Returns (N, n); s = x when σ_e = 0.
    """
    if params.observation == "ordinal":
        raise NotImplementedError("Ordinal observation is not implemented yet")
    if params.observation != "continuous":
        raise ValueError(f"Unsupported observation: {params.observation!r}")

    expression_var = params.expression_noise_sd**2
    total_var = expression_var + params.measurement_noise_sd**2
    shrinkage = expression_var / total_var if total_var > 0 else 0.0

    observed = ~np.isnan(y)
    conditional_mean = x.copy()
    conditional_mean[..., observed] += shrinkage * (y[observed] - x[..., observed])
    conditional_sd = np.where(observed, np.sqrt((1 - shrinkage) * expression_var), np.sqrt(expression_var))
    return conditional_mean + conditional_sd * rng.standard_normal(x.shape)
