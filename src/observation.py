# Observation layer: how researchers see expressions (Y from s), and the formulas the filters
# derive from it. Continuous observation only for now: Y = s + e, e ~ N(0, σ_m² I).

import numpy as np

from src.config import ModelParams


def require_continuous_observation(params: ModelParams) -> None:
    """Raise unless params.observation is 'continuous' (ordinal is planned, anything else is an error)."""
    if params.observation == "ordinal":
        raise NotImplementedError("Ordinal observation is not implemented yet")
    if params.observation != "continuous":
        raise ValueError(f"Unsupported observation: {params.observation!r}")


def observation_variance(params: ModelParams) -> float:
    """Variance of Y given the opinion x, with the expression integrated out: σ_e² + σ_m²."""
    return params.expression_noise_sd**2 + params.measurement_noise_sd**2


def expression_shrinkage(params: ModelParams) -> float:
    """κ = σ_e² / (σ_e² + σ_m²): weight on Y when estimating the expression s from (x, Y); 0 if both are 0."""
    total_var = observation_variance(params)
    return params.expression_noise_sd**2 / total_var if total_var > 0 else 0.0


def observe(s: np.ndarray, params: ModelParams, rng: np.random.Generator) -> np.ndarray:
    """Researcher's observation of expressions: continuous Y = s + e, e ~ N(0, σ_m² I)."""
    require_continuous_observation(params)
    return s + rng.normal(0.0, params.measurement_noise_sd, size=s.shape)


def gaussian_obs_loglik(y: np.ndarray, x: np.ndarray, params: ModelParams) -> np.ndarray:
    """Elementwise log N(y; x, σ_e² + σ_m²), not summed; y and x broadcast against each other."""
    require_continuous_observation(params)
    total_var = observation_variance(params)
    return -0.5 * ((y - x) ** 2 / total_var + np.log(2 * np.pi * total_var))


def obs_loglik_given_opinion(y: np.ndarray, x: np.ndarray, params: ModelParams) -> np.ndarray:
    """log p(Y_t | x_t) with the expression integrated out, one value per particle.

    Sum over observed agents (non-NaN y) of log N(y_i; x_i, σ_e² + σ_m²).
    y: (n,); x: (N, n). Returns (N,), zeros if no agent posted.
    """
    require_continuous_observation(params)
    observed = ~np.isnan(y)
    if not observed.any():
        return np.zeros(x.shape[:-1])
    return np.sum(gaussian_obs_loglik(y[observed], x[..., observed], params), axis=-1)


def sample_expression_given_obs(
    x: np.ndarray, y: np.ndarray, params: ModelParams, rng: np.random.Generator
) -> np.ndarray:
    """Draw s_t ~ p(s_t | x_t, Y_t) for every particle.

    With κ = σ_e² / (σ_e² + σ_m²): observed agents s ~ N(x + κ (y − x), (1 − κ) σ_e²);
    unobserved agents s ~ N(x, σ_e²). x: (N, n), y: (n,). Returns (N, n); s = x when σ_e = 0.
    """
    require_continuous_observation(params)
    expression_var = params.expression_noise_sd**2
    shrinkage = expression_shrinkage(params)

    observed = ~np.isnan(y)
    conditional_mean = x.copy()
    conditional_mean[..., observed] += shrinkage * (y[observed] - x[..., observed])
    conditional_sd = np.where(observed, np.sqrt((1 - shrinkage) * expression_var), np.sqrt(expression_var))
    return conditional_mean + conditional_sd * rng.standard_normal(x.shape)


def expression_estimate(
    opinion_mean: np.ndarray, opinion_var: np.ndarray, y_t: np.ndarray, params: ModelParams
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of each agent's expression s_t, given x_t ~ N(m, P) and its post Y_t.

    With κ = σ_e² / (σ_e² + σ_m²):  ŝ = (1 − κ) m + κ Y,   v = (1 − κ)² P + (1 − κ) σ_e².
    Same conditional law as sample_expression_given_obs, averaged over x ~ N(m, P).
    """
    require_continuous_observation(params)
    expression_var = params.expression_noise_sd**2
    shrinkage = expression_shrinkage(params)
    estimate = (1 - shrinkage) * opinion_mean + shrinkage * y_t
    estimate_var = (1 - shrinkage) ** 2 * opinion_var + (1 - shrinkage) * expression_var
    return estimate, estimate_var
