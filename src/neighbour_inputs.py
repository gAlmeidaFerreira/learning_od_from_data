# Neighbour inputs of the per-agent filters (plan 1.2, version a). Agent i's filter tracks only x_i,
# so the neighbours' hidden expressions s_{j,t-1} in its update are replaced by an input chosen
# with input_mode:
#   "observed":     their posts Y_{j,t-1} (plug-in; biased, since Y carries measurement noise)
#   "filtered":     their filtered expression estimates ŝ_{j,t-1}, computed by the neighbours'
#                   own filters in the same pass (only data up to t-1)
#   "filtered_var": ŝ_{j,t-1} plus its uncertainty v_{j,t-1}
# ŝ and v come from observation.expression_estimate. Shared by the per-agent Kalman and particle filters.

from typing import Literal

import numpy as np

InputMode = Literal["observed", "filtered", "filtered_var"]
INPUT_MODES: tuple[InputMode, ...] = ("observed", "filtered", "filtered_var")


def validate_per_agent_inputs(y: np.ndarray, input_mode: str) -> None:
    """Raise if input_mode is unknown or y has missing posts (not supported by the per-agent filters yet)."""
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported input_mode: {input_mode!r} (expected one of {INPUT_MODES})")
    if np.isnan(y).any():
        raise NotImplementedError("missing posts are not supported by the per-agent filter yet")
