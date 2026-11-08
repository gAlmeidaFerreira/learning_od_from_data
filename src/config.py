# Loads YAML config files into parameter dataclasses.

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NetworkConfig:
    """Interaction network settings."""

    type: str
    n_agents: int
    k: int
    p: float


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation length."""

    T: int


@dataclass(frozen=True)
class ModelParams:
    """True parameters used to generate the data."""

    dynamics: str
    self_weight: float
    process_noise_sd: float
    expression_noise_sd: float
    measurement_noise_sd: float
    observation: str


@dataclass(frozen=True)
class FilterConfig:
    """Particle filter settings."""

    n_particles: int


@dataclass(frozen=True)
class Config:
    """Full experiment configuration."""

    seed: int
    network: NetworkConfig
    simulation: SimulationConfig
    model: ModelParams
    filter: FilterConfig


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a Config, rejecting missing or unknown keys."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    sections = {
        "network": NetworkConfig,
        "simulation": SimulationConfig,
        "model": ModelParams,
        "filter": FilterConfig,
    }
    to_check = [("top level", raw, {"seed", *sections})]
    to_check += [(name, raw.get(name) if isinstance(raw, dict) else None,
                  {f.name for f in fields(cls)}) for name, cls in sections.items()]

    for name, data, expected in to_check:
        if not isinstance(data, dict):
            raise ValueError(f"[{name}] must be a mapping of keys to values")
        problems = [f"{kind} key(s): {', '.join(sorted(keys))}"
                    for kind, keys in (("missing", expected - data.keys()),
                                       ("unknown", data.keys() - expected))
                    if keys]
        if problems:
            raise ValueError(f"[{name}] " + "; ".join(problems))

    return Config(
        seed=raw["seed"],
        **{name: cls(**raw[name]) for name, cls in sections.items()},
    )


def save_config(config: Config, path: str | Path) -> None:
    """Write a Config to a YAML file."""
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
