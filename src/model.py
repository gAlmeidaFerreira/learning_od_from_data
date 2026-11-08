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
