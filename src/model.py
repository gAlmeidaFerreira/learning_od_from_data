# Network, opinion dynamics and expression layers. The observation layer is in src/observation.py.

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
    rng: np.random.Generator | None,
    *,
    standard_noise: np.ndarray | None = None,
) -> np.ndarray:
    """One DeGroot update from neighbours' expressions.

    x_t = λ x_{t-1} + (1-λ) W s_{t-1} + ε_t,  ε_t ~ N(0, σ_p² I).
    Accepts shape (n,) or batched (N, n). If standard_noise (same shape as x_prev, standard
    normal) is given, ε_t = σ_p · standard_noise and rng is not used (may be None).
    """
    self_weight = params.self_weight
    if standard_noise is not None:
        noise = params.process_noise_sd * standard_noise
    elif rng is not None:
        noise = rng.normal(0.0, params.process_noise_sd, size=x_prev.shape)
    else:
        raise ValueError("degroot_step needs rng or standard_noise")
    return self_weight * x_prev + (1.0 - self_weight) * (s_prev @ W.T) + noise


DYNAMICS: dict[str, Callable[..., np.ndarray]] = {"degroot": degroot_step}


def step(
    x_prev: np.ndarray,
    s_prev: np.ndarray,
    W: np.ndarray,
    params: ModelParams,
    rng: np.random.Generator | None,
    *,
    standard_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Advance hidden opinions one step using the rule named by params.dynamics.

    standard_noise: optional pre-drawn standard-normal process noise (see degroot_step).
    """
    try:
        update = DYNAMICS[params.dynamics]
    except KeyError:
        raise ValueError(f"Unsupported dynamics: {params.dynamics!r}") from None
    return update(x_prev, s_prev, W, params, rng, standard_noise=standard_noise)


def express(x: np.ndarray, params: ModelParams, rng: np.random.Generator) -> np.ndarray:
    """Expressed signal s = x + ξ, ξ ~ N(0, σ_e² I)."""
    return x + rng.normal(0.0, params.expression_noise_sd, size=x.shape)
