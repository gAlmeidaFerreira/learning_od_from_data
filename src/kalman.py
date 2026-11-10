# Exact Kalman-filter likelihood for linear-Gaussian DeGroot (reference for the particle filter).

from typing import Literal, overload

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal

from src.config import ModelParams, SimulationConfig
from src.neighbour_inputs import InputMode, validate_per_agent_inputs
from src.observation import expression_estimate, observation_variance


def _require_linear_gaussian(model_params: ModelParams) -> None:
    """Raise unless the model is linear-Gaussian (DeGroot dynamics, continuous observation)."""
    if model_params.dynamics != "degroot":
        raise ValueError(f"Kalman filter needs dynamics 'degroot', got {model_params.dynamics!r}")
    if model_params.observation != "continuous":
        raise ValueError(f"Kalman filter needs continuous observation, got {model_params.observation!r}")


def state_space(
    W: np.ndarray, model_params: ModelParams, sim_params: SimulationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear-Gaussian state-space form (F, Q, H, R, m0, P0) of DeGroot with continuous Y.

    Augmented state z_t = [x_t; s_t] of size 2n:
      z_t = F z_{t-1} + w_t, w_t ~ N(0, Q);   Y_t = H z_t + e_t, e_t ~ N(0, R);   z_0 ~ N(m0, P0).
    """
    _require_linear_gaussian(model_params)

    n_agents = W.shape[0]
    identity = np.eye(n_agents)
    zeros = np.zeros((n_agents, n_agents))
    self_weight = model_params.self_weight
    process_var = model_params.process_noise_sd**2
    expression_var = model_params.expression_noise_sd**2
    prior_var = sim_params.x0_sd**2

    opinion_update = np.hstack([self_weight * identity, (1 - self_weight) * W])
    F = np.vstack([opinion_update, opinion_update])
    Q = np.block([
        [process_var * identity, process_var * identity],
        [process_var * identity, (process_var + expression_var) * identity],
    ])
    H = np.hstack([zeros, identity])
    R = model_params.measurement_noise_sd**2 * identity
    m0 = np.full(2 * n_agents, sim_params.x0_mean)
    P0 = np.block([
        [prior_var * identity, prior_var * identity],
        [prior_var * identity, (prior_var + expression_var) * identity],
    ])
    return F, Q, H, R, m0, P0


@overload
def kalman_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    return_filtered: Literal[False] = False,
) -> float: ...


@overload
def kalman_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    return_filtered: Literal[True],
) -> tuple[float, np.ndarray, np.ndarray]: ...


def kalman_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    return_filtered: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Exact log p(Y | params) by the Kalman filter; y has shape (T, n), NaN = missing post.

    If return_filtered, also returns filtered means (T, 2n) and covariances (T, 2n, 2n)
    of z_t = [x_t; s_t] given Y_0..Y_t.
    """
    F, Q, H, R, m0, P0 = state_space(W, model_params, sim_params)
    n_steps = y.shape[0]
    state_size = F.shape[0]
    identity = np.eye(state_size)

    total_loglik = 0.0
    filtered_means = np.empty((n_steps, state_size))
    filtered_covs = np.empty((n_steps, state_size, state_size))

    mean, cov = m0.copy(), P0.copy()
    for t in range(n_steps):
        if t > 0:                      # predict; at t = 0 the prior is already the prediction
            mean = F @ mean
            cov = F @ cov @ F.T + Q

        observed = ~np.isnan(y[t])
        if observed.any():
            H_obs = H[observed]
            R_obs = R[np.ix_(observed, observed)]
            innovation = y[t, observed] - H_obs @ mean
            innovation_cov = H_obs @ cov @ H_obs.T + R_obs
            cholesky = cho_factor(innovation_cov, lower=True)

            log_det = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
            mahalanobis = innovation @ cho_solve(cholesky, innovation)
            total_loglik += -0.5 * (observed.sum() * np.log(2 * np.pi) + log_det + mahalanobis)

            gain = cho_solve(cholesky, H_obs @ cov).T          # P H^T S^{-1}
            mean = mean + gain @ innovation
            correction = identity - gain @ H_obs
            cov = correction @ cov @ correction.T + gain @ R_obs @ gain.T   # Joseph form

        cov = 0.5 * (cov + cov.T)
        filtered_means[t] = mean
        filtered_covs[t] = cov

    if return_filtered:
        return float(total_loglik), filtered_means, filtered_covs
    return float(total_loglik)


def brute_force_loglik(
    y: np.ndarray, W: np.ndarray, model_params: ModelParams, sim_params: SimulationConfig
) -> float:
    """Test only: log p(Y) from the joint Gaussian of the stacked (Y_0, ..., Y_{T-1}).

    Builds the full Tn x Tn covariance, so use it only for tiny problems. NaN entries
    are marginalised out by dropping their rows and columns.
    """
    F, Q, H, R, m0, P0 = state_space(W, model_params, sim_params)
    n_steps, n_agents = y.shape

    # Marginal means and covariances of z_t, then cross-covariances Cov(z_t, z_u) = F^(t-u) Cov(z_u).
    state_means = [m0]
    state_covs = [P0]
    for _ in range(1, n_steps):
        state_means.append(F @ state_means[-1])
        state_covs.append(F @ state_covs[-1] @ F.T + Q)

    stacked_mean = np.concatenate([H @ m for m in state_means])
    stacked_cov = np.empty((n_steps * n_agents, n_steps * n_agents))
    for later in range(n_steps):
        for earlier in range(later + 1):
            cross_cov = np.linalg.matrix_power(F, later - earlier) @ state_covs[earlier]
            block = H @ cross_cov @ H.T + (R if later == earlier else 0.0)
            rows = slice(later * n_agents, (later + 1) * n_agents)
            cols = slice(earlier * n_agents, (earlier + 1) * n_agents)
            stacked_cov[rows, cols] = block
            stacked_cov[cols, rows] = block.T

    stacked_y = y.reshape(-1)
    observed = ~np.isnan(stacked_y)
    return float(multivariate_normal.logpdf(
        stacked_y[observed],
        mean=stacked_mean[observed],
        cov=stacked_cov[np.ix_(observed, observed)],  # type: ignore[arg-type]  # SciPy stub says int
    ))


# Per-agent Kalman filter: n independent scalar filters, vectorised over agents,
#   x_{i,t} = λ x_{i,t-1} + (1-λ) u_{i,t-1} + ε_{i,t},   Y_{i,t} = x_{i,t} + (ξ + e)_{i,t},
# where the neighbour input u_{t-1} = W (neighbours' expressions) depends on input_mode
# (see src/neighbour_inputs.py):
#   "observed":     u = W Y_{t-1}
#   "filtered":     u = W ŝ_{t-1}
#   "filtered_var": u = W ŝ_{t-1}, plus the input variance (1-λ)² (W∘W) v_{t-1} added to the prediction


@overload
def per_agent_kalman_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    return_filtered: Literal[False] = False,
    *,
    input_mode: InputMode = "observed",
) -> float: ...


@overload
def per_agent_kalman_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    return_filtered: Literal[True],
    *,
    input_mode: InputMode = "observed",
) -> tuple[float, dict]: ...


def per_agent_kalman_loglik(
    y: np.ndarray,
    W: np.ndarray,
    model_params: ModelParams,
    sim_params: SimulationConfig,
    return_filtered: bool = False,
    *,
    input_mode: InputMode = "observed",
) -> float | tuple[float, dict]:
    """Exact log-likelihood of the per-agent model (sum over agents); y is (T, n), no NaN.

    input_mode chooses the neighbour input (see src/neighbour_inputs.py). If return_filtered, also
    returns {"per_agent_loglik": (n,), "filtered_mean": (T, n), "filtered_var": (T, n)}.
    """
    _require_linear_gaussian(model_params)
    validate_per_agent_inputs(y, input_mode)

    n_steps, n_agents = y.shape
    self_weight = model_params.self_weight
    process_var = model_params.process_noise_sd**2
    observation_var = observation_variance(model_params)
    squared_weights = W**2

    mean = np.full(n_agents, sim_params.x0_mean)
    var = np.full(n_agents, sim_params.x0_sd**2)
    per_agent_loglik = np.zeros(n_agents)
    filtered_mean = np.empty((n_steps, n_agents))
    filtered_var = np.empty((n_steps, n_agents))
    # Neighbours' expressions and their variance, set at the end of every step; first used at t = 1.
    neighbour_expressions = np.zeros(n_agents)
    neighbour_expression_var = np.zeros(n_agents)

    for t in range(n_steps):
        if t > 0:
            mean = self_weight * mean + (1 - self_weight) * (W @ neighbour_expressions)
            var = self_weight**2 * var + process_var
            if input_mode == "filtered_var":
                var = var + (1 - self_weight) ** 2 * (squared_weights @ neighbour_expression_var)

        innovation_var = var + observation_var
        gain = var / innovation_var
        innovation = y[t] - mean
        per_agent_loglik += -0.5 * (np.log(2 * np.pi) + np.log(innovation_var) + innovation**2 / innovation_var)
        mean = mean + gain * innovation
        var = (1 - gain) * var

        filtered_mean[t] = mean
        filtered_var[t] = var

        if input_mode == "observed":
            neighbour_expressions = y[t]
        else:
            neighbour_expressions, neighbour_expression_var = expression_estimate(mean, var, y[t], model_params)

    total_loglik = float(per_agent_loglik.sum())
    if return_filtered:
        return total_loglik, {
            "per_agent_loglik": per_agent_loglik, "filtered_mean": filtered_mean, "filtered_var": filtered_var,
        }
    return total_loglik
