import math
import textwrap
from enum import IntEnum
from functools import cache
from typing import Any

import lets_plot as lp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from lets_plot.plot.core import PlotSpec as LetsPlotSpec
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.text import Text as MplText
from osmnx.plot import _get_fig_ax

SUBPLOT_FIGSIZE = 4.0
"""Size in inches for each subplot in a grid."""


class MarginTextLocation(IntEnum):
    """Enum to map margin text locations to their degree x-values."""

    top_left = 315
    top_right = 45
    bottom_left = 225
    bottom_right = 135

    @property
    def radians(self) -> float:
        return math.radians(self.value)

    @property
    def vertical_alignment(self) -> str:
        return self.name.split("_")[0]

    @property
    def horizontal_alignment(self) -> str:
        return self.name.split("_")[1]


def plot_orientation(
    *,
    bin_counts: npt.NDArray[np.float64],
    bin_centers: npt.NDArray[np.float64],
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (5, 5),
    max_bin_count: float | None = None,
    area: bool = True,
    color: str = "#D4A0A7",
    edgecolor: str = "black",
    linewidth: float = 0.5,
    alpha: float = 0.7,
    title: str | None = None,
    title_y: float = 1.05,
    title_font: dict[str, Any] | None = None,
    xtick_font: dict[str, Any] | None = None,
    margin_text: dict[MarginTextLocation, str] | None = None,
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
    max_bin_count
        If not None, set the y-axis upper limit to this value.
        Useful for comparing multiple polar histograms on the same scale.
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
        title_font = {"family": "DejaVu Sans", "size": 18, "weight": "bold"}
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
    radius = np.sqrt(bin_counts) if area else bin_counts
    if max_bin_count is None:
        ylim = radius.max()
    else:
        ylim = np.sqrt(max_bin_count) if area else max_bin_count

    # create PolarAxes (if not passed-in) then set N at top and go clockwise
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, bgcolor=None, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=ylim)

    # configure the y-ticks and remove their labels
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

    if margin_text is None:
        margin_text = {
            MarginTextLocation.top_right: f"{bin_counts.sum():,.0f}m\nskiable\nvert",
        }
    for location, text in margin_text.items():
        if not text:
            continue
        _mpl_add_polar_margin_text(ax=ax, ylim=ylim, location=location, text=text)

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)
    fig.tight_layout()
    return fig, ax


def _mpl_add_polar_margin_text(
    ax: PolarAxes,
    ylim: float,
    location: MarginTextLocation,
    text: str,
    color: str = "#95A5A6",
) -> MplText:
    return ax.text(
        x=location.radians,
        y=ylim * 1.58,
        s=text,
        verticalalignment=location.vertical_alignment,
        horizontalalignment=location.horizontal_alignment,
        multialignment=location.horizontal_alignment,
        color=color,
    )


def subplot_orientations(
    groups_pl: pl.DataFrame,
    grouping_col: str,
    n_cols: int | None = None,
    free_y: bool = True,
    num_bins: int = 32,
    suptitle: str | None = None,
) -> plt.Figure:
    """
    Plot orientations from multiple graphs in a grid.
    https://github.com/gboeing/osmnx-examples/blob/bb870c225906db5a7b02c4c87a28095cb9dceb30/notebooks/17-street-network-orientations.ipynb

    Parameters
    ----------
    groups_pl
        A Polars DataFrame with one group per row and a bearings columns with the bearing distributions.
    grouping_col
        The column for naming each subplot/facet.
    n_cols
        The number of columns in the grid of subplots.
    free_y
        If True, each subplot's y-axis will be scaled independently.
        If False, all subplots will be scaled to the same maximum y-value.
    num_bins
        The number of bins in each polar histogram.
        This value must exist in the input groups_pl bearings.num_bins column.
    suptitle
        The figure's super title.
    """
    assert not groups_pl.select(grouping_col).is_duplicated().any()
    names = groups_pl.get_column(grouping_col).sort().to_list()
    # create figure and axes
    n_subplots = len(names)
    if n_cols is None:
        n_cols = math.ceil(n_subplots**0.5)
    n_rows = math.ceil(n_subplots / n_cols)
    figsize = (n_cols * SUBPLOT_FIGSIZE, n_rows * SUBPLOT_FIGSIZE)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, subplot_kw={"projection": "polar"}
    )
    dists_pl = (
        groups_pl.select(grouping_col, "bearings")
        .explode("bearings")
        .unnest("bearings")
        .filter(pl.col("num_bins") == num_bins)
    )
    # plot each group's polar histogram
    max_bin_count = None if free_y else dists_pl.get_column("bin_count").max()
    for ax, name in zip(axes.flat, names, strict=False):
        group_info = groups_pl.row(
            by_predicate=pl.col(grouping_col) == name, named=True
        )
        # ideas for margin text
        # top-left: ski areas, country, trail, lift count
        # top-right: total skiable vertical
        # bottom-left: poleward affinity/tendency, polar affinity
        # bottom-right: total trail length and average pitch or max elevation
        margin_text = {}
        if {"ski_areas_count", "run_count_filtered"}.issubset(group_info):
            margin_text[MarginTextLocation.top_left] = (
                f"{group_info["ski_areas_count"]:,} ski areas,\n{group_info["run_count_filtered"]:,} runs"
            )
        if "combined_vertical" in group_info:
            margin_text[MarginTextLocation.top_right] = (
                f"{group_info["combined_vertical"]:,.0f}m\nskiable\nvert"
            )
        group_dist_pl = dists_pl.filter(pl.col(grouping_col) == name)
        fig, ax = plot_orientation(
            bin_counts=group_dist_pl.get_column("bin_count").to_numpy(),
            bin_centers=group_dist_pl.get_column("bin_center").to_numpy(),
            ax=ax,
            max_bin_count=max_bin_count,
            title=name,
            area=True,
            color="#D4A0A7",
            margin_text=margin_text or None,
        )
        ax.title.set_size(18)
        ax.yaxis.grid(False)
        if free_y and "mean_bearing" in group_info:
            ax.scatter(
                x=np.radians(group_info["mean_bearing"]),
                y=group_info["mean_bearing_strength"] * ax.get_ylim()[1],
                color="blue",
                label="Mean Bearing",
                zorder=2,
                marker=get_snowflake_marker(),
                linewidths=0,
                s=1000,  # marker size
                alpha=0.7,
            )

    # hide axes for unused subplots
    for ax in axes.flat[n_subplots:]:
        ax.axis("off")
    if suptitle:
        # FIXME: spacing is inconsistent based on number of subplot rows
        fig.suptitle(t=suptitle, y=0.99, fontsize=13, verticalalignment="bottom")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.24)
    # fig.savefig("images/street-orientations.png", facecolor="w", dpi=100, bbox_inches="tight")
    plt.close()
    return fig


def plot_mean_bearing(
    ski_areas_pl: pl.DataFrame,
    *,
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (4.5, 4.5),
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
    Polar plot of mean bearings.
    """
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

    # bearings
    positions = (
        ski_areas_pl.select(pl.col("mean_bearing").radians().alias("mean_bearing_rad"))
        .get_column("mean_bearing_rad")
        .to_numpy()
    )
    magnitudes = ski_areas_pl.get_column("mean_bearing_strength").to_numpy()

    # create PolarAxes (if not passed-in) then set N at top and go clockwise
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, bgcolor=None, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=1.0)

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.scatter(
        x=positions,
        y=magnitudes,
        s=ski_areas_pl.get_column("combined_vertical").to_numpy() / 50,
        zorder=2,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )

    for i, ski_area_name in enumerate(ski_areas_pl.get_column("ski_area_name")):
        ski_area_name = textwrap.fill(ski_area_name, width=10)
        ax.annotate(
            text=ski_area_name,
            xy=(positions[i], magnitudes[i]),
            textcoords="offset points",
            xytext=(5, -5),
            ha="right",
            fontsize=4.5,
        )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)
    fig.tight_layout()
    return fig, ax


def subplot_orientations_lets_plot(
    distribution_pl: pl.DataFrame,
    grouping_col: str,
    n_cols: int | None = None,
    free_y: bool = True,
) -> LetsPlotSpec:
    """
    Plot orientations from multiple graphs in a grid
    using Let's Plot. This is exploratory with much broken.
    """
    return (
        lp.ggplot(
            data=(
                distribution_pl
                # .filter(pl.col("ski_area_name").is_in(["Sugarloaf", "Dartmouth Skiway"]))
                .with_columns(pl.col("bin_count").replace(0, 360))
            ),
            mapping=lp.aes(x="bin_center", y="bin_count"),
        )
        + lp.geom_bar(
            stat="identity",
            tooltips=lp.layer_tooltips()
            .format(field="^y", format="{,.0f}m")
            .format(field="^x", format="{}°"),
            width=0.7,
            size=0.5,
            color="black",
        )
        + lp.facet_wrap(
            [grouping_col],
            scales="free_y" if free_y else "fixed",
            ncol=n_cols,
        )
        + lp.scale_x_continuous(
            labels={
                0: "N",
                45: "NE",
                90: "E",
                135: "SE",
                180: "S",
                225: "SW",
                270: "W",
                315: "NW",
                360: "N",
            },
            # limits=(0, 360),
            name="Trial Orientation",
        )
        # + lp.xlim(0, 360)
        + lp.geom_text(x=45, y=1700, label="test")
        + lp.scale_y_continuous(trans="sqrt", breaks=[])
        + lp.guides(bin_count=None)
        + lp.coord_polar()  # start=np.radians(-5))
        + lp.theme(
            axis_title=lp.element_blank(),
            # axis_text_y=lp.element_blank(),
        )
    )


@cache
def get_snowflake_marker() -> MplPath:
    """
    Source: https://www.svgrepo.com/svg/490217/snowflake
    https://petercbsmith.github.io/marker-tutorial.html
    FIXME: Only six sided snowflakes are accurate.
    """
    return MplPath(
        vertices=[
            [368.0, 176.0],
            [353.888, 176.0],
            [358.312, 167.16],
            [362.264, 159.256],
            [359.064, 149.648],
            [351.152, 145.688],
            [343.232, 141.72],
            [333.64, 144.952],
            [329.68, 152.848],
            [318.112, 176.0],
            [230.624, 176.0],
            [292.488, 114.144],
            [317.048, 122.328],
            [318.728, 122.888],
            [320.44, 123.152],
            [322.112, 123.152],
            [328.816, 123.152],
            [335.056, 118.912],
            [337.288, 112.208],
            [340.088, 103.824],
            [335.552, 94.768],
            [327.168, 91.968],
            [317.784, 88.84],
            [327.76, 78.864],
            [334.008, 72.616],
            [334.008, 62.488],
            [327.76, 56.24],
            [321.512, 49.992],
            [311.384, 49.992],
            [305.136, 56.24],
            [295.152, 66.224],
            [292.024, 56.84],
            [289.232, 48.464],
            [280.2, 43.92],
            [271.784, 46.72],
            [263.4, 49.512],
            [258.872, 58.576],
            [261.664, 66.96],
            [269.856, 91.528],
            [208.0, 153.376],
            [208.0, 65.888],
            [231.16, 54.312],
            [239.064, 50.36],
            [242.272, 40.752],
            [238.32, 32.84],
            [234.368, 24.944],
            [224.776, 21.728],
            [216.848, 25.68],
            [208.0, 30.112],
            [208.0, 16.0],
            [208.0, 7.168],
            [200.832, 0.0],
            [192.0, 0.0],
            [183.168, 0.0],
            [176.0, 7.168],
            [176.0, 16.0],
            [176.0, 30.112],
            [167.16, 25.688],
            [159.272, 21.736],
            [149.648, 24.936],
            [145.688, 32.848],
            [141.736, 40.752],
            [144.944, 50.36],
            [152.848, 54.32],
            [176.0, 65.888],
            [176.0, 153.376],
            [114.136, 91.512],
            [122.328, 66.952],
            [125.12, 58.568],
            [120.592, 49.504],
            [112.208, 46.712],
            [103.848, 43.928],
            [94.76, 48.448],
            [91.968, 56.832],
            [88.84, 66.216],
            [78.864, 56.24],
            [72.624, 49.992],
            [62.48, 49.992],
            [56.24, 56.24],
            [49.992, 62.488],
            [49.992, 72.616],
            [56.24, 78.864],
            [66.216, 88.84],
            [56.832, 91.968],
            [48.448, 94.768],
            [43.92, 103.824],
            [46.712, 112.208],
            [48.952, 118.912],
            [55.192, 123.152],
            [61.888, 123.152],
            [63.568, 123.152],
            [65.272, 122.888],
            [66.952, 122.328],
            [91.512, 114.136],
            [153.376, 176.0],
            [65.888, 176.0],
            [54.312, 152.84],
            [50.352, 144.936],
            [40.728, 141.736],
            [32.84, 145.68],
            [24.936, 149.632],
            [21.736, 159.24],
            [25.68, 167.152],
            [30.112, 176.0],
            [16.0, 176.0],
            [7.168, 176.0],
            [0.0, 183.168],
            [0.0, 192.0],
            [0.0, 200.832],
            [7.168, 208.0],
            [16.0, 208.0],
            [30.112, 208.0],
            [25.688, 216.84],
            [21.736, 224.744],
            [24.944, 234.352],
            [32.848, 238.312],
            [35.144, 239.464],
            [37.584, 240.0],
            [39.992, 240.0],
            [45.856, 240.0],
            [51.512, 236.76],
            [54.32, 231.152],
            [65.888, 208.0],
            [153.376, 208.0],
            [91.512, 269.864],
            [66.952, 261.672],
            [58.592, 258.896],
            [49.504, 263.408],
            [46.712, 271.792],
            [43.92, 280.176],
            [48.448, 289.24],
            [56.832, 292.032],
            [66.216, 295.16],
            [56.24, 305.136],
            [49.992, 311.384],
            [49.992, 321.512],
            [56.24, 327.76],
            [59.36, 330.888],
            [63.456, 332.448],
            [67.552, 332.448],
            [71.648, 332.448],
            [75.744, 330.888],
            [78.864, 327.76],
            [88.84, 317.784],
            [91.968, 327.168],
            [94.208, 333.872],
            [100.448, 338.112],
            [107.144, 338.112],
            [108.824, 338.112],
            [110.528, 337.848],
            [112.208, 337.288],
            [120.592, 334.488],
            [125.12, 325.432],
            [122.328, 317.048],
            [114.136, 292.488],
            [176.0, 230.624],
            [176.0, 318.112],
            [152.84, 329.688],
            [144.936, 333.64],
            [141.736, 343.248],
            [145.68, 351.16],
            [148.496, 356.76],
            [154.144, 360.0],
            [160.008, 360.0],
            [162.416, 360.0],
            [164.856, 359.456],
            [167.152, 358.312],
            [176.0, 353.888],
            [176.0, 368.0],
            [176.0, 376.832],
            [183.168, 384.0],
            [192.0, 384.0],
            [200.832, 384.0],
            [208.0, 376.832],
            [208.0, 368.0],
            [208.0, 353.888],
            [216.84, 358.312],
            [219.144, 359.464],
            [221.584, 360.0],
            [223.984, 360.0],
            [229.856, 360.0],
            [235.504, 356.76],
            [238.312, 351.152],
            [242.264, 343.248],
            [239.064, 333.64],
            [231.152, 329.68],
            [208.0, 318.112],
            [208.0, 230.624],
            [269.864, 292.48],
            [261.672, 317.048],
            [258.872, 325.432],
            [263.408, 334.496],
            [271.792, 337.288],
            [273.472, 337.848],
            [275.184, 338.112],
            [276.856, 338.112],
            [283.56, 338.112],
            [289.8, 333.872],
            [292.032, 327.168],
            [295.16, 317.784],
            [305.144, 327.768],
            [308.272, 330.896],
            [312.36, 332.456],
            [316.456, 332.456],
            [320.552, 332.456],
            [324.64, 330.896],
            [327.768, 327.768],
            [334.016, 321.52],
            [334.016, 311.392],
            [327.768, 305.144],
            [317.792, 295.168],
            [327.176, 292.04],
            [335.56, 289.24],
            [340.088, 280.184],
            [337.296, 271.8],
            [334.504, 263.416],
            [325.472, 258.872],
            [317.056, 261.68],
            [292.496, 269.864],
            [230.624, 208.0],
            [318.104, 208.0],
            [329.68, 231.16],
            [332.496, 236.76],
            [338.144, 240.0],
            [344.016, 240.0],
            [346.416, 240.0],
            [348.856, 239.456],
            [351.16, 238.312],
            [359.064, 234.36],
            [362.272, 224.752],
            [358.32, 216.84],
            [353.888, 208.0],
            [368.0, 208.0],
            [376.832, 208.0],
            [384.0, 200.832],
            [384.0, 192.0],
            [384.0, 183.168],
            [376.832, 176.0],
            [368.0, 176.0],
            [368.0, 176.0],
        ],
        codes=[
            1,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            79,
        ],
    )
    from svgpath2mpl import parse_path

    return parse_path(
        """
        M368,176h-14.112l4.424-8.84c3.952-7.904,0.752-17.512-7.16-21.472c-7.92-3.968-17.512-0.736-21.472,7.16L318.112,176
        h-87.488l61.864-61.856l24.56,8.184c1.68,0.56,3.392,0.824,5.064,0.824c6.704,0,12.944-4.24,15.176-10.944
        c2.8-8.384-1.736-17.44-10.12-20.24l-9.384-3.128l9.976-9.976c6.248-6.248,6.248-16.376,0-22.624s-16.376-6.248-22.624,0
        l-9.984,9.984l-3.128-9.384c-2.792-8.376-11.824-12.92-20.24-10.12c-8.384,2.792-12.912,11.856-10.12,20.24l8.192,24.568
        L208,153.376V65.888l23.16-11.576c7.904-3.952,11.112-13.56,7.16-21.472c-3.952-7.896-13.544-11.112-21.472-7.16L208,30.112V16
        c0-8.832-7.168-16-16-16c-8.832,0-16,7.168-16,16v14.112l-8.84-4.424c-7.888-3.952-17.512-0.752-21.472,7.16
        c-3.952,7.904-0.744,17.512,7.16,21.472L176,65.888v87.488l-61.864-61.864l8.192-24.56c2.792-8.384-1.736-17.448-10.12-20.24
        c-8.36-2.784-17.448,1.736-20.24,10.12l-3.128,9.384l-9.976-9.976c-6.24-6.248-16.384-6.248-22.624,0
        c-6.248,6.248-6.248,16.376,0,22.624l9.976,9.976l-9.384,3.128c-8.384,2.8-12.912,11.856-10.12,20.24
        c2.24,6.704,8.48,10.944,15.176,10.944c1.68,0,3.384-0.264,5.064-0.824l24.56-8.192L153.376,176H65.888l-11.576-23.16
        c-3.96-7.904-13.584-11.104-21.472-7.16c-7.904,3.952-11.104,13.56-7.16,21.472L30.112,176H16c-8.832,0-16,7.168-16,16
        c0,8.832,7.168,16,16,16h14.112l-4.424,8.84c-3.952,7.904-0.744,17.512,7.16,21.472c2.296,1.152,4.736,1.688,7.144,1.688
        c5.864,0,11.52-3.24,14.328-8.848L65.888,208h87.488l-61.864,61.864l-24.56-8.192c-8.36-2.776-17.448,1.736-20.24,10.12
        c-2.792,8.384,1.736,17.448,10.12,20.24l9.384,3.128l-9.976,9.976c-6.248,6.248-6.248,16.376,0,22.624
        c3.12,3.128,7.216,4.688,11.312,4.688c4.096,0,8.192-1.56,11.312-4.688l9.976-9.976l3.128,9.384
        c2.24,6.704,8.48,10.944,15.176,10.944c1.68,0,3.384-0.264,5.064-0.824c8.384-2.8,12.912-11.856,10.12-20.24l-8.192-24.56
        L176,230.624v87.488l-23.16,11.576c-7.904,3.952-11.104,13.56-7.16,21.472c2.816,5.6,8.464,8.84,14.328,8.84
        c2.408,0,4.848-0.544,7.144-1.688l8.848-4.424V368c0,8.832,7.168,16,16,16c8.832,0,16-7.168,16-16v-14.112l8.84,4.424
        c2.304,1.152,4.744,1.688,7.144,1.688c5.872,0,11.52-3.24,14.328-8.848c3.952-7.904,0.752-17.512-7.16-21.472L208,318.112v-87.488
        l61.864,61.856l-8.192,24.568c-2.8,8.384,1.736,17.448,10.12,20.24c1.68,0.56,3.392,0.824,5.064,0.824
        c6.704,0,12.944-4.24,15.176-10.944l3.128-9.384l9.984,9.984c3.128,3.128,7.216,4.688,11.312,4.688s8.184-1.56,11.312-4.688
        c6.248-6.248,6.248-16.376,0-22.624l-9.976-9.976l9.384-3.128c8.384-2.8,12.912-11.856,10.12-20.24
        c-2.792-8.384-11.824-12.928-20.24-10.12l-24.56,8.184L230.624,208h87.48l11.576,23.16c2.816,5.6,8.464,8.84,14.336,8.84
        c2.4,0,4.84-0.544,7.144-1.688c7.904-3.952,11.112-13.56,7.16-21.472l-4.432-8.84H368c8.832,0,16-7.168,16-16
        C384,183.168,376.832,176,368,176z
        """
    )
