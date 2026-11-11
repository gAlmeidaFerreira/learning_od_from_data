# Plotting helpers shared by the check scripts: one figure style, the λ-profile figure, and saving.

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                # check scripts only write figures to files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import Config

# Line styles reused across figures; pass them as ax.plot(..., **STYLE).
EXACT_STYLE = {"color": "0.15", "linewidth": 1.8}                         # exact (Kalman) reference curves
ESTIMATE_STYLE = {"linewidth": 1.2, "linestyle": "--", "marker": "o", "markersize": 3.5}   # particle-filter curves
POINTS_STYLE = {"linewidth": 1.6, "marker": "o", "markersize": 6}         # one marker per setting
TRUE_VALUE_STYLE = {"color": "0.5", "linestyle": ":", "linewidth": 1.2}
RELATIVE_PROFILE_LIMITS = (-60, 5)


def dataset_title(config: Config, *extra: str) -> str:
    """Title suffix such as "($n = 50$, $T = 100$, $\\sigma_m = 0.1$)", with extra items appended."""
    items = [rf"$n = {config.network.n_agents}$", rf"$T = {config.simulation.T}$", *extra]
    return "(" + ", ".join(items) + ")"


def mark_true_value(ax: Axes, value: float, axis: str = "x", symbol: str = r"\lambda") -> None:
    """Dotted line at the true parameter value: vertical on the x axis, horizontal on the y axis."""
    draw_line = ax.axvline if axis == "x" else ax.axhline
    draw_line(value, **TRUE_VALUE_STYLE, label=rf"true ${symbol} = {value}$")


def set_log_ticks(ax: Axes, values, labels: list[str] | None = None, axis: str = "x", base: float = 10) -> None:
    """Log scale with a tick at each value (labelled with labels, or the value itself) and no minor ticks."""
    set_scale, set_ticks = (ax.set_xscale, ax.set_xticks) if axis == "x" else (ax.set_yscale, ax.set_yticks)
    set_scale("log", base=base)
    set_ticks(values, labels if labels is not None else [f"{value:g}" for value in values])
    ax.minorticks_off()


def style_axes(
    ax: Axes, xlabel: str, ylabel: str, title: str | None = None,
    legend_location: str = "best", legend_fontsize: float | None = None, legend: bool = True,
) -> None:
    """Axis labels, title, light grid and a frameless legend."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, loc=legend_location, fontsize=legend_fontsize)
    ax.grid(alpha=0.25)


def save_figure(fig: Figure, path: Path) -> Path:
    """Tight layout, save at 150 dpi, close the figure; returns path."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@dataclass
class ProfileCurve:
    """One curve of a λ-profile figure.

    In a relative figure the curve is shifted by `peak`, by default its own maximum. Pass another
    curve's maximum instead to keep two curves on the same level (e.g. a PF and the Kalman curve
    it estimates).
    """

    values: np.ndarray
    label: str
    style: dict = field(default_factory=dict)
    peak: float | None = None


def plot_self_weight_profiles(
    path: Path,
    grid: np.ndarray,
    curves: list[ProfileCurve],
    true_self_weight: float,
    title: str,
    relative: bool = True,
    width_level: float | None = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    legend_fontsize: float | None = None,
) -> Path:
    """Log-likelihood profiles over λ, with the true λ marked; saves the figure to path.

    relative: plot ℓ(λ) − peak on a fixed window (RELATIVE_PROFILE_LIMITS); otherwise plot ℓ(λ) itself.
    width_level: draw a horizontal line at −width_level (the profile-width threshold).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for curve in curves:
        shift = (curve.values.max() if curve.peak is None else curve.peak) if relative else 0.0
        ax.plot(grid, curve.values - shift, label=curve.label, **curve.style)
    if width_level is not None:
        ax.axhline(-width_level, color="0.6", linestyle="-.", linewidth=0.9,
                   label=rf"$-{width_level:g}$ (profile width level)")
    mark_true_value(ax, true_self_weight)
    if relative:
        ax.set_ylim(*RELATIVE_PROFILE_LIMITS)
        ylabel = r"relative log-likelihood $\ell(\lambda) - \max \ell$"
    else:
        ylabel = r"$\log p(Y \mid \lambda, \sigma_p, \sigma_e)$"
    style_axes(ax, r"$\lambda$", ylabel, title,
               legend_location="lower left" if relative else "best", legend_fontsize=legend_fontsize)
    return save_figure(fig, path)
