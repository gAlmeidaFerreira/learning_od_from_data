# Hidden opinions with the parameters integrated out (plan 1.2, "Hidden opinions").
#
# Each posterior draw ψ^(k) gives a filter whose output is Gaussian per agent and step,
# x_{i,t} | Y_{0:t}, ψ^(k) ~ N(m^(k), P^(k)). Averaging over K draws gives a mixture whose mean and
# variance follow from the law of total variance:
#   E[x]   = mean_k m^(k)
#   Var[x] = mean_k P^(k)            (state uncertainty, average filtered variance)
#          + var_k m^(k)             (parameter uncertainty, spread of the filtered means)
# The draws can come from the MCMC chain itself (pseudo_marginal_mh with an OpinionAccumulator)
# or from re-running a filter at stored draws (estimate_hidden_opinions).

from collections.abc import Callable, Iterable

import numpy as np

from src.config import ModelParams


def mixture_moments(
    sum_m: np.ndarray, sum_m2: np.ndarray, sum_P: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of an equal-weight mixture of K = count Gaussians N(m^(k), P^(k)).

    sum_m, sum_m2 and sum_P are the sums over k of m^(k), (m^(k))² and P^(k). The variance is
    state plus parameter uncertainty: sum_P / K (average filtered variance) plus
    sum_m2 / K − mean² (spread of the filtered means across parameter draws).
    """
    if count < 1:
        raise ValueError("mixture_moments needs at least one draw")
    mean = sum_m / count
    var = sum_P / count + sum_m2 / count - mean**2
    return mean, var


class OpinionAccumulator:
    """Running sums of filtered means m, their squares and filtered variances P, shape (T, n).

    The sums are public so callers can split the variance: sum_var / count is the state part,
    and the rest of result()'s variance is the parameter part.
    """

    def __init__(self, n_steps: int, n_agents: int) -> None:
        self.sum_mean = np.zeros((n_steps, n_agents))
        self.sum_mean_squared = np.zeros((n_steps, n_agents))
        self.sum_var = np.zeros((n_steps, n_agents))
        self.count = 0

    def update(self, m: np.ndarray, P: np.ndarray) -> None:
        """Add one draw's filtered means m and variances P, each of shape (T, n)."""
        self.sum_mean += m
        self.sum_mean_squared += m**2
        self.sum_var += P
        self.count += 1

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean, var) of the hidden opinions over the draws added so far; see mixture_moments."""
        return mixture_moments(self.sum_mean, self.sum_mean_squared, self.sum_var, self.count)


def estimate_hidden_opinions(
    params_draws: Iterable[ModelParams], filter_fn: Callable[[ModelParams], dict]
) -> tuple[np.ndarray, np.ndarray]:
    """(mean, var), each (T, n), of the hidden opinions with the parameters integrated out.

    Runs filter_fn(params) at every posterior draw; it must return a dict with "filtered_mean" and
    "filtered_var" of shape (T, n), e.g. the diagnostics of per_agent_kalman_loglik or of a
    per-agent particle filter called with return_filtered=True.
    """
    accumulator = None
    for params in params_draws:
        filtered = filter_fn(params)
        if accumulator is None:
            accumulator = OpinionAccumulator(*filtered["filtered_mean"].shape)
        accumulator.update(filtered["filtered_mean"], filtered["filtered_var"])
    if accumulator is None:
        raise ValueError("estimate_hidden_opinions needs at least one parameter draw")
    return accumulator.result()
