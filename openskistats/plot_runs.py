from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.colors import TwoSlopeNorm

from openskistats.analyze import load_runs_pl
from openskistats.utils import pl_hemisphere

LATITUDE_STEP = 3
BEARING_STEP = 4
LATITUDE_ABS_BREAKS = list(range(0, 90 + LATITUDE_STEP, LATITUDE_STEP))
BEARING_BREAKS = list(range(0, 360 + BEARING_STEP, BEARING_STEP))


def get_bearing_by_latitude_bin_metrics() -> pl.DataFrame:
    """
    Metrics for (latitude_abs_bin, bearing_bin) pairs.
    """
    bin_pattern = r"\[(.+), (.+)\)"
    return (
        load_runs_pl()
        .filter(pl.col("run_uses").list.contains("downhill"))
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .with_columns(
            # Flips a degree bearing across the east-west axis,
            # i.e. a latitudinal reflection or hemispherical flip of bearings in the southern hemisphere
            bearing_poleward=pl.when(pl.col("latitude").gt(0))
            .then(pl.col("bearing"))
            .otherwise(pl.lit(180).sub("bearing").mod(360)),
        )
        # FIXME: consider using get_bearing_histogram method to prevent bin-edge effects
        .select(
            "run_id",
            "segment_hash",
            # pl.col("latitude").round(0).cast(pl.Int32).alias("latitude"),
            pl.col("latitude")
            .abs()
            .cut(breaks=LATITUDE_ABS_BREAKS, left_closed=True)
            .alias("latitude_abs_bin"),
            pl.col("bearing_poleward")
            .cut(breaks=BEARING_BREAKS, left_closed=True)
            .alias("bearing_bin"),
            # pl.col("bearing").round(0).cast(pl.Int32).alias("bearing"),
            "distance_vertical_drop",
        )
        .group_by("latitude_abs_bin", "bearing_bin")
        .agg(
            pl.count("segment_hash").alias("segment_count"),
            pl.col("distance_vertical_drop").sum().alias("combined_vertical").round(5),
        )
        .filter(pl.col("combined_vertical") > 0)
        .with_columns(
            latitude_abs_bin_lower=pl.col("latitude_abs_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=1)
            .cast(pl.Int32),
            latitude_abs_bin_upper=pl.col("latitude_abs_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=2)
            .cast(pl.Int32),
            bearing_bin_lower=pl.col("bearing_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=1)
            .cast(pl.Int32),
            bearing_bin_upper=pl.col("bearing_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=2)
            .cast(pl.Int32),
        )
        .with_columns(
            latitude_abs_bin_center=pl.mean_horizontal(
                "latitude_abs_bin_lower", "latitude_abs_bin_upper"
            ),
            bearing_bin_center=pl.mean_horizontal(
                "bearing_bin_lower", "bearing_bin_upper"
            ),
        )
        # Proportion of combined_vertical within a latitude bin
        .with_columns(
            total_combined_vertical=pl.sum("combined_vertical").over(
                "latitude_abs_bin"
            ),
        )
        .with_columns(
            combined_vertical_prop=pl.col("combined_vertical").truediv(
                "total_combined_vertical"
            ),
        )
        .with_columns(
            combined_vertical_enrichment=pl.col("combined_vertical_prop").mul(
                len(BEARING_BREAKS) - 1
            ),
        )
        .with_columns(bearing_bin_center_radians=pl.col("bearing_bin_center").radians())
        .sort("latitude_abs_bin", "bearing_bin")
        .collect()
    )


@dataclass
class BearingByLatitudeBinMeshGrid:
    latitude_grid: npt.NDArray[np.float64]
    bearing_grid: npt.NDArray[np.float64]
    color_grid: npt.NDArray[np.float64]


def get_bearing_by_latitude_bin_mesh_grids() -> BearingByLatitudeBinMeshGrid:
    metrics_df = get_bearing_by_latitude_bin_metrics()
    grid_pl = (
        pl.DataFrame({"latitude_abs_bin_lower": LATITUDE_ABS_BREAKS[:-1]})
        .join(
            pl.DataFrame({"bearing_bin_lower": BEARING_BREAKS[:-1]}),
            how="cross",
        )
        .select(
            pl.all().cast(pl.Int32),
        )
        .join(
            metrics_df.select(
                "latitude_abs_bin_lower",
                "bearing_bin_lower",
                "combined_vertical",
                "combined_vertical_prop",
                "combined_vertical_enrichment",
            ),
            on=["latitude_abs_bin_lower", "bearing_bin_lower"],
            how="left",
        )
        .with_columns(
            total_combined_vertical=pl.sum("combined_vertical").over(
                "latitude_abs_bin_lower"
            ),
        )
        .with_columns(
            pl.col("combined_vertical").fill_null(0),
            pl.col("combined_vertical_prop").fill_null(0),
            combined_vertical_enrichment=pl.when(
                pl.col("total_combined_vertical") >= 10_000
            ).then(pl.col("combined_vertical_enrichment").fill_null(0)),
        )
        .pivot(
            index="latitude_abs_bin_lower",
            columns="bearing_bin_lower",
            # values="combined_vertical",
            values="combined_vertical_enrichment",
            sort_columns=False,  # order of discovery
        )
        .sort("latitude_abs_bin_lower")
        # .with_columns(pl.selectors.by_dtype(pl.Float64).fill_null(0.0))
        .drop("latitude_abs_bin_lower")
    )
    assert np.all(np.diff([int(x) for x in grid_pl.columns]) > 0)
    latitude_grid, bearing_grid = np.meshgrid(LATITUDE_ABS_BREAKS, BEARING_BREAKS)
    return BearingByLatitudeBinMeshGrid(
        latitude_grid=latitude_grid,
        bearing_grid=bearing_grid,
        color_grid=grid_pl.to_numpy(),
    )


def plot_bearing_by_latitude_bin() -> plt.Figure:
    grids = get_bearing_by_latitude_bin_mesh_grids()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    quad_mesh = ax.pcolormesh(
        (np.deg2rad(grids.bearing_grid)).transpose(),
        grids.latitude_grid.transpose(),
        grids.color_grid.clip(min=0, max=2.5),
        shading="flat",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.5),
    )
    plt.colorbar(quad_mesh, ax=ax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.grid(visible=False)
    # TODO: reuse code in plot
    ax.set_xticks(ax.get_xticks())
    xticklabels = ["Poleward", "", "E", "", "Equatorward", "", "W", ""]
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)
    # y-tick labeling
    latitude_ticks = np.arange(0, 91, 10)
    ax.set_yticks(latitude_ticks)
    ax.tick_params(axis="y", which="major", length=5, width=1)
    ax.set_yticklabels(
        [f"{r}°" if r in {0, 90} else "" for r in latitude_ticks],
        rotation=0,
        fontsize=7,
    )
    ax.set_rlabel_position(225)

    # Draw custom radial arcs for y-ticks
    for radius in latitude_ticks:
        theta_start = np.deg2rad(220)
        theta_end = np.deg2rad(230)
        theta = np.linspace(theta_start, theta_end, 100)
        ax.plot(
            theta,
            np.full_like(theta, radius),
            linewidth=1 if radius < 90 else 2,
            color="black",
        )

    return fig


def get_latitude_histogram() -> pl.DataFrame:
    # FIXME: hemisphere broken
    return (
        get_bearing_by_latitude_bin_metrics()
        .group_by("latitude_bin")
        .agg(
            pl.first("latitude_bin_lower").alias("latitude_bin_lower"),
            pl.first("latitude_bin_upper").alias("latitude_bin_upper"),
            pl.first("latitude_bin_center").alias("latitude_bin_center"),
            pl.sum("segment_count").alias("segment_count"),
            pl.sum("combined_vertical").round(5).alias("combined_vertical"),
        )
        .with_columns(
            hemisphere=pl_hemisphere("latitude_bin_center"),
            latitude_bin_center_abs=pl.col("latitude_bin_center").abs(),
        )
    )
