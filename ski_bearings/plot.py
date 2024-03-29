import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from osmnx.plot import _get_fig_ax


def plot_orientation(
    *,
    bin_counts: npt.NDArray[np.float64],
    bin_centers: npt.NDArray[np.float64],
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (5, 5),
    area: bool = True,
    color: str = "#D4A0A7",
    edgecolor: str = "black",
    linewidth: float = 0.5,
    alpha: float = 0.7,
    title: str | None = None,
    title_y: float = 1.05,
    title_font: dict[str, Any] | None = None,
    xtick_font: dict[str, Any] | None = None,
) -> tuple[Figure, PolarAxes]:
    """
    Plot a polar histogram of an edge bearing distribution.

    Modified from the `osmnx.plot_orientation` source code to support consuming
    distribution information directly rather than a graph.
    https://github.com/gboeing/osmnx/blob/d614c9608165a614f796c7ab8e57257e7f9f1a63/osmnx/plot.py#L666-L803

    Parameters
    ----------
    ax
        If not None, plot on this pre-existing axes instance (must have
        projection=polar).
    figsize
        If `ax` is None, create new figure with size `(width, height)`.
    area
        If True, set bar length so area is proportional to frequency.
        Otherwise, set bar length so height is proportional to frequency.
    color
        Color of the histogram bars.
    edgecolor
        Color of the histogram bar edges.
    linewidth
        Width of the histogram bar edges.
    alpha
        Opacity of the histogram bars.
    title
        The figure's title.
    title_y
        The y position to place `title`.
    title_font
        The title's `fontdict` to pass to matplotlib.
    xtick_font
        The xtick labels' `fontdict` to pass to matplotlib.

    Returns
    -------
    fig, ax
    """
    num_bins = len(bin_centers)
    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 24, "weight": "bold"}
    if xtick_font is None:
        xtick_font = {
            "family": "DejaVu Sans",
            "size": 10,
            "weight": "bold",
            "alpha": 1.0,
            "zorder": 3,
        }

    # positions: where to center each bar
    positions = np.radians(bin_centers)

    # width: make bars fill the circumference without gaps or overlaps
    width = 2 * np.pi / num_bins

    # radius: how long to make each bar. set bar length so either the bar area
    # (ie, via sqrt) or the bar height is proportional to the bin's frequency
    bin_frequency = bin_counts / bin_counts.sum()
    radius = np.sqrt(bin_frequency) if area else bin_frequency

    # create PolarAxes (if not passed-in) then set N at top and go clockwise
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, bgcolor=None, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=radius.max())

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, radius.max(), 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=2,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)
    fig.tight_layout()
    return fig, ax


def subplot_orientations(
    distribution_pl: pl.DataFrame, grouping_col: str, n_cols: int | None = None
) -> plt.Figure:
    """
    Plot orientations from multiple graphs in a grid.
    https://github.com/gboeing/osmnx-examples/blob/bb870c225906db5a7b02c4c87a28095cb9dceb30/notebooks/17-street-network-orientations.ipynb
    """
    # create figure and axes
    groupings = distribution_pl.partition_by(grouping_col, as_dict=True)
    n_groupings = len(groupings)
    if n_cols is None:
        n_cols = math.ceil(n_groupings**0.5)
    n_rows = math.ceil(n_groupings / n_cols)
    figsize = (n_cols * 5, n_rows * 5)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # plot each group's polar histogram
    for ax, (name, group_dist_pl) in zip(
        axes.flat, sorted(groupings.items()), strict=False
    ):
        fig, ax = plot_orientation(
            bin_counts=group_dist_pl.get_column("bin_count").to_numpy(),
            bin_centers=group_dist_pl.get_column("bin_center").to_numpy(),
            ax=ax,
            title=name,
            area=True,
            color="#D4A0A7",
        )
        ax.title.set_size(18)
        ax.yaxis.grid(False)
    # hide axes for unused subplots
    for ax in axes.flat[n_groupings:]:
        ax.axis("off")
    # add figure title and save image
    # suptitle_font = {
    #     "family": "DejaVu Sans",
    #     "fontsize": 60,
    #     "fontweight": "normal",
    #     "y": 1,
    # }
    # fig.suptitle("City Street Network Orientation", **suptitle_font)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    # fig.savefig("images/street-orientations.png", facecolor="w", dpi=100, bbox_inches="tight")
    plt.close()
    return fig
    # plt.close()
