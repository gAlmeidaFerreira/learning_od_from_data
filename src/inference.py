# Parameter inference by (pseudo-marginal) Metropolis–Hastings.
#
# One engine serves two uses (plan 1.2):
#   - exact MH: the likelihood is deterministic (Kalman), u is always None;
#   - correlated PMMH: the likelihood is a particle-filter estimate p̂(Y | ψ, u) driven by the
#     random numbers u, which move with the parameters by a Crank–Nicolson step
#     u' = ρ u + sqrt(1 − ρ²) ε (ρ = 0 is standard PMMH, a fresh u at every iteration).
# The chain runs on the unconstrained θ = (logit λ, log σ_p, log σ_e) with a Gaussian random walk.

import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import ModelParams
from src.neighbour_inputs import INPUT_MODES, InputMode
from src.particle_filter import crank_nicolson

PARAMETER_NAMES = ("self_weight", "process_noise_sd", "expression_noise_sd")   # ψ = (λ, σ_p, σ_e), in θ order


# ---------------------------------------------------------------------------------------------
# Config

@dataclass(frozen=True)
class UniformPrior:
    """Uniform density on [low, high]."""

    low: float
    high: float


@dataclass(frozen=True)
class LognormalPrior:
    """log σ ~ N(log median, log_sd²)."""

    median: float
    log_sd: float


Prior = UniformPrior | LognormalPrior
PRIOR_TYPES: dict[str, type] = {"uniform": UniformPrior, "lognormal": LognormalPrior}


@dataclass(frozen=True)
class Priors:
    """One prior per inferred parameter."""

    self_weight: Prior
    process_noise_sd: Prior
    expression_noise_sd: Prior


@dataclass(frozen=True)
class MCMCConfig:
    """Random-walk MH settings."""

    n_iterations: int
    burn_in: int
    adapt_every: int
    initial_step_sd: float


@dataclass(frozen=True)
class PMMHConfig:
    """Pseudo-marginal settings for the particle-filter likelihood."""

    n_particles: int
    rho: float
    input_mode: InputMode


@dataclass(frozen=True)
class InferenceConfig:
    """Full inference configuration."""

    seed: int
    priors: Priors
    mcmc: MCMCConfig
    pmmh: PMMHConfig


def _check_keys(section: str, data: Any, expected: set[str]) -> None:
    """Raise ValueError unless data is a mapping with exactly the expected keys."""
    if not isinstance(data, dict):
        raise ValueError(f"[{section}] must be a mapping of keys to values")
    problems = [f"{kind} key(s): {', '.join(sorted(keys))}"
                for kind, keys in (("missing", expected - data.keys()), ("unknown", data.keys() - expected))
                if keys]
    if problems:
        raise ValueError(f"[{section}] " + "; ".join(problems))


def _field_names(cls: type) -> set[str]:
    return {field.name for field in fields(cls)}


def _parse_prior(section: str, data: Any) -> Prior:
    if not isinstance(data, dict) or data.get("dist") not in PRIOR_TYPES:
        raise ValueError(f"[{section}] needs 'dist', one of {sorted(PRIOR_TYPES)}")
    prior_type = PRIOR_TYPES[data["dist"]]
    _check_keys(section, data, {"dist", *_field_names(prior_type)})
    return prior_type(**{key: value for key, value in data.items() if key != "dist"})


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load an inference YAML file, rejecting missing or unknown keys (like src/config.py)."""
    with open(path) as file:
        raw = yaml.safe_load(file)
    _check_keys("top level", raw, {"seed", "priors", "mcmc", "pmmh"})
    _check_keys("priors", raw["priors"], set(PARAMETER_NAMES))
    _check_keys("mcmc", raw["mcmc"], _field_names(MCMCConfig))
    _check_keys("pmmh", raw["pmmh"], _field_names(PMMHConfig))
    if raw["pmmh"]["input_mode"] not in INPUT_MODES:
        raise ValueError(f"[pmmh] input_mode must be one of {INPUT_MODES}, got {raw['pmmh']['input_mode']!r}")
    return InferenceConfig(
        seed=raw["seed"],
        priors=Priors(**{name: _parse_prior(f"priors.{name}", raw["priors"][name]) for name in PARAMETER_NAMES}),
        mcmc=MCMCConfig(**raw["mcmc"]),
        pmmh=PMMHConfig(**raw["pmmh"]),
    )


# ---------------------------------------------------------------------------------------------
# Parameter transform and prior

def to_unconstrained(model_params: ModelParams) -> np.ndarray:
    """θ = (logit λ, log σ_p, log σ_e)."""
    self_weight = model_params.self_weight
    return np.array([np.log(self_weight / (1 - self_weight)),
                     np.log(model_params.process_noise_sd),
                     np.log(model_params.expression_noise_sd)])


def from_unconstrained(theta: np.ndarray, model_params: ModelParams) -> ModelParams:
    """ModelParams with (λ, σ_p, σ_e) from θ; every other field (σ_m included) copied from model_params."""
    return replace(model_params,
                   self_weight=float(1 / (1 + np.exp(-theta[0]))),
                   process_noise_sd=float(np.exp(theta[1])),
                   expression_noise_sd=float(np.exp(theta[2])))


def log_jacobian(theta: np.ndarray) -> float:
    """log |dψ/dθ| = log λ(1 − λ) + log σ_p + log σ_e, with ψ = (λ, σ_p, σ_e) the original parameters."""
    self_weight = 1 / (1 + np.exp(-theta[0]))
    return float(np.log(self_weight * (1 - self_weight)) + theta[1] + theta[2])


def _prior_logpdf(prior: Prior, value: float) -> float:
    if isinstance(prior, UniformPrior):
        return -np.log(prior.high - prior.low) if prior.low <= value <= prior.high else -np.inf
    if value <= 0:
        return -np.inf
    standardised = (np.log(value) - np.log(prior.median)) / prior.log_sd
    return float(-np.log(value) - np.log(prior.log_sd) - 0.5 * np.log(2 * np.pi) - 0.5 * standardised**2)


def log_prior(model_params: ModelParams, priors: Priors) -> float:
    """log p(ψ) on the original scale (λ, σ_p, σ_e); the parameters are a priori independent."""
    return sum(_prior_logpdf(getattr(priors, name), getattr(model_params, name)) for name in PARAMETER_NAMES)


# ---------------------------------------------------------------------------------------------
# Sampler

def pseudo_marginal_mh(
    loglik_fn: Callable[[ModelParams, Any], float],
    theta0: np.ndarray,
    model_params: ModelParams,
    priors: Priors,
    mcmc_cfg: MCMCConfig,
    rng: np.random.Generator,
    *,
    draw_noise: Callable[[np.random.Generator], Any] | None = None,
    rho: float = 0.0,
) -> dict:
    """Random-walk (pseudo-marginal) MH on θ = (logit λ, log σ_p, log σ_e).

    loglik_fn(params, u) returns log p(Y | params) (exact, u = None) or its estimate given the
    random numbers u. If draw_noise is None the likelihood is treated as exact; otherwise
    u starts at draw_noise(rng) and moves by u' = crank_nicolson(u, rho, rng) together with θ.
    model_params supplies the fields that are not inferred (σ_m, dynamics, ...).
    Returns {"theta", "params" (λ, σ_p, σ_e), "loglik", "proposed_loglik", "accepted",
    "proposal_cov", "runtime_s"}; arrays have one row per iteration, burn-in included.
    """
    start_time = time.perf_counter()
    n_iterations, dimension = mcmc_cfg.n_iterations, theta0.size
    adaptive_scale = 2.38**2 / dimension
    proposal_cov = mcmc_cfg.initial_step_sd**2 * np.eye(dimension)

    theta_trace = np.empty((n_iterations, dimension))
    params_trace = np.empty((n_iterations, dimension))
    loglik_trace = np.empty(n_iterations)
    proposed_loglik_trace = np.full(n_iterations, np.nan)
    accepted = np.zeros(n_iterations, dtype=bool)

    # Current state (θ, u, loglik). Its loglik is never recomputed: in PMMH the estimate is kept
    # with the state it belongs to, which is what makes the chain target the exact posterior.
    theta = np.asarray(theta0, dtype=float)
    params = from_unconstrained(theta, model_params)
    noise = draw_noise(rng) if draw_noise is not None else None
    loglik = loglik_fn(params, noise)
    # Target density in θ: p(Y | ψ(θ)) p(ψ(θ)) |dψ/dθ|. The chain moves on θ, but the prior is
    # stated on ψ, so the change of variables adds the log-Jacobian; forgetting it (or using
    # the inverse map's Jacobian) silently changes the prior and biases the posterior.
    log_prior_term = log_prior(params, priors) + log_jacobian(theta)

    for iteration in range(n_iterations):
        proposed_theta = theta + rng.multivariate_normal(np.zeros(dimension), proposal_cov)
        proposed_noise = crank_nicolson(noise, rho, rng) if noise is not None else None   # None ⇔ exact MH
        proposed_params = from_unconstrained(proposed_theta, model_params)
        proposed_log_prior_term = log_prior(proposed_params, priors) + log_jacobian(proposed_theta)

        if np.isfinite(proposed_log_prior_term):
            proposed_loglik = loglik_fn(proposed_params, proposed_noise)
            proposed_loglik_trace[iteration] = proposed_loglik
            # The random walk is symmetric in θ, so the proposal densities cancel.
            log_ratio = proposed_loglik - loglik + proposed_log_prior_term - log_prior_term
            if np.log(rng.uniform()) < log_ratio:   # accept (θ', u', loglik') together
                theta, noise, params = proposed_theta, proposed_noise, proposed_params
                loglik, log_prior_term = proposed_loglik, proposed_log_prior_term
                accepted[iteration] = True

        theta_trace[iteration] = theta
        params_trace[iteration] = [getattr(params, name) for name in PARAMETER_NAMES]
        loglik_trace[iteration] = loglik

        # Adaptive Metropolis (Haario et al.) during burn-in only; frozen afterwards, so the
        # kept samples come from a fixed, valid MH kernel.
        if iteration + 1 < mcmc_cfg.burn_in and (iteration + 1) % mcmc_cfg.adapt_every == 0:
            sample_cov = np.cov(theta_trace[: iteration + 1].T)
            proposal_cov = adaptive_scale * (sample_cov + 1e-6 * np.eye(dimension))

    return {
        "theta": theta_trace, "params": params_trace, "loglik": loglik_trace,
        "proposed_loglik": proposed_loglik_trace, "accepted": accepted,
        "proposal_cov": proposal_cov, "runtime_s": time.perf_counter() - start_time,
    }


# ---------------------------------------------------------------------------------------------
# Diagnostics

def autocorrelation(chain_1d: np.ndarray) -> np.ndarray:
    """Sample autocorrelations ρ_0 = 1, ρ_1, ..., ρ_{n-1} of a 1-D chain (via FFT)."""
    centred = np.asarray(chain_1d, dtype=float) - np.mean(chain_1d)
    n_samples = centred.size
    spectrum = np.fft.rfft(centred, n=2 * n_samples)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum))[:n_samples] / n_samples
    return autocovariance / autocovariance[0]


def effective_sample_size(chain_1d: np.ndarray) -> float:
    """ESS = n / τ by Geyer's initial monotone sequence estimator.

    τ = −1 + 2 Σ_k Γ_k with Γ_k = ρ_{2k} + ρ_{2k+1} (sums of adjacent autocorrelation pairs,
    which are positive and decreasing for a reversible chain). The sum stops at the first
    non-positive Γ_k (initial positive sequence), and each Γ_k is capped at the previous one
    (monotone), which removes the noise of the tail autocorrelations. Returns NaN for a
    constant chain.
    """
    chain = np.asarray(chain_1d, dtype=float)
    if np.ptp(chain) == 0:
        return float("nan")
    correlations = autocorrelation(chain)
    n_pairs = correlations.size // 2
    pair_sums = correlations[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)
    non_positive = np.flatnonzero(pair_sums <= 0)
    pair_sums = pair_sums[: non_positive[0] if non_positive.size else n_pairs]
    pair_sums = np.minimum.accumulate(pair_sums)
    integrated_time = max(-1.0 + 2.0 * pair_sums.sum(), 1e-12)
    return float(chain.size / integrated_time)
