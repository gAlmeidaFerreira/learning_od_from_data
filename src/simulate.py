# Generates ground-truth hidden opinions and noisy observations.

from pathlib import Path

import networkx as nx
import numpy as np

from src import model, observation
from src.config import Config, load_config, save_config


def make_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent (network_rng, data_rng) streams spawned from one seed."""
    network_seq, data_seq = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(network_seq), np.random.default_rng(data_seq)


def simulate(
    config: Config,
    W: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate hidden opinions x, expressions s, observations y and posting mask P.

    If W is None the network is built from the network stream of config.seed;
    if rng is None the data stream of config.seed is used. Passing W and a fresh
    rng draws a new dataset on the same network. Arrays x, s, y, P have shape (T, n);
    y is NaN where P is False. "graph" is None when W is passed in.
    """
    network_rng, data_rng = make_rngs(config.seed)
    graph = None
    if W is None:
        graph, W = model.build_network(config.network, network_rng)
    if rng is None:
        rng = data_rng

    params = config.model
    T, n = config.simulation.T, W.shape[0]
    x = np.empty((T, n))
    s = np.empty((T, n))
    y = np.empty((T, n))

    x[0] = rng.normal(config.simulation.x0_mean, config.simulation.x0_sd, size=n)
    s[0] = model.express(x[0], params, rng)
    y[0] = observation.observe(s[0], params, rng)
    for t in range(1, T):
        x[t] = model.step(x[t - 1], s[t - 1], W, params, rng)
        s[t] = model.express(x[t], params, rng)
        y[t] = observation.observe(s[t], params, rng)

    P = np.ones((T, n), dtype=bool)
    y[~P] = np.nan
    return {"x": x, "s": s, "y": y, "P": P, "W": W, "graph": graph}


def save_simulation(result: dict, config: Config, path: str | Path) -> None:
    """Save arrays to <path>.npz and the config to <path>.yaml."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **{k: result[k] for k in ("x", "s", "y", "P", "W")})
    save_config(config, path.with_suffix(".yaml"))


def load_simulation(path: str | Path) -> tuple[dict, Config]:
    """Load a dataset written by save_simulation; the graph is rebuilt from W."""
    path = Path(path)
    with np.load(path.with_suffix(".npz")) as data:
        result = {k: data[k] for k in ("x", "s", "y", "P", "W")}
    adjacency = (result["W"] > 0).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    result["graph"] = nx.from_numpy_array(adjacency)
    return result, load_config(path.with_suffix(".yaml"))
