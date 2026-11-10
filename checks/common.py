# Helpers shared by the check scripts: paths, base config, output folders and λ profiles.

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from src.config import Config, ModelParams, load_config

# Check scripts only write figures to files. matplotlib switches backend even if pyplot was
# already imported, as long as no figure is open yet.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "degroot_base.yaml"
CHECK_RESULTS_DIR = REPO_ROOT / "results" / "checks"   # each check writes to CHECK_RESULTS_DIR / <name>
PROFILE_WIDTH_DROP = 2.0             # profile width = range of λ within this many log-units of the maximum


def load_base_config() -> Config:
    """The baseline config (configs/degroot_base.yaml); each check saves the config it uses next to its results."""
    return load_config(BASE_CONFIG_PATH)


def with_model_params(config: Config, **changes: float) -> Config:
    """Copy of config with some model parameters changed, e.g. with_model_params(config, measurement_noise_sd=0.2)."""
    return replace(config, model=replace(config.model, **changes))


def self_weight_profile(
    loglik_of_params: Callable[[ModelParams], float], model_params: ModelParams, grid: np.ndarray
) -> np.ndarray:
    """Log-likelihood over a grid of λ values, all other model parameters as in model_params.

    loglik_of_params maps model parameters to a log-likelihood; for a particle filter it should
    create its own rng from a fixed seed, so every grid point uses common random numbers.
    """
    return np.array([loglik_of_params(replace(model_params, self_weight=self_weight)) for self_weight in grid])


def profile_width(grid: np.ndarray, profile: np.ndarray, drop: float = PROFILE_WIDTH_DROP) -> float:
    """Range of λ where the profile is within `drop` of its maximum (linear interpolation at the edges).

    If the profile never falls that far on one side, the grid edge is used, so the width is a lower bound.
    """
    peak_index = int(np.argmax(profile))
    level = profile[peak_index] - drop

    left_index = peak_index
    while left_index > 0 and profile[left_index - 1] >= level:
        left_index -= 1
    if left_index == 0:
        left_edge = grid[0]
    else:
        fraction = (profile[left_index] - level) / (profile[left_index] - profile[left_index - 1])
        left_edge = grid[left_index] - fraction * (grid[left_index] - grid[left_index - 1])

    right_index = peak_index
    while right_index < grid.size - 1 and profile[right_index + 1] >= level:
        right_index += 1
    if right_index == grid.size - 1:
        right_edge = grid[-1]
    else:
        fraction = (profile[right_index] - level) / (profile[right_index] - profile[right_index + 1])
        right_edge = grid[right_index] + fraction * (grid[right_index + 1] - grid[right_index])

    return float(right_edge - left_edge)
