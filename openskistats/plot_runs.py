from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

from openskistats.analyze import load_runs_pl

LATITUDE_STEP = 6
BEARING_STEP = 4
LATITUDE_BREAKS = list(range(-90, 90 + LATITUDE_STEP, LATITUDE_STEP))
BEARING_BREAKS = list(range(0, 360 + BEARING_STEP, BEARING_STEP))


def get_bearing_by_latitude_bin_metrics() -> pl.DataFrame:
    """
    Metrics for (latitude_bin, bearing_bin) pairs.
    """
    bin_pattern = r"\[(.+), (.+)\)"
    return (
        load_runs_pl()
        .filter(pl.col("run_uses").list.contains("downhill"))
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .select(
            "run_id",
            "segment_hash",
            # pl.col("latitude").round(0).cast(pl.Int32).alias("latitude"),
            pl.col("latitude")
            .cut(breaks=LATITUDE_BREAKS, left_closed=True)
            .alias("latitude_bin"),
            pl.col("bearing")
            .cut(breaks=BEARING_BREAKS, left_closed=True)
            .alias("bearing_bin"),
            # pl.col("bearing").round(0).cast(pl.Int32).alias("bearing"),
            "distance_vertical_drop",
        )
        .group_by("latitude_bin", "bearing_bin")
        .agg(
            pl.count("segment_hash").alias("segment_count"),
            pl.col("distance_vertical_drop").sum().alias("combined_vertical").round(5),
        )
        .filter(pl.col("combined_vertical") > 0)
        .with_columns(
            latitude_bin_lower=pl.col("latitude_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=1)
            .cast(pl.Int32),
            latitude_bin_upper=pl.col("latitude_bin")
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
            latitude_bin_center=pl.mean_horizontal(
                "latitude_bin_lower", "latitude_bin_upper"
            ),
            bearing_bin_center=pl.mean_horizontal(
                "bearing_bin_lower", "bearing_bin_upper"
            ),
        )
        # Proportion of combined_vertical within a latitude bin
        .with_columns(
            combined_vertical_prop=pl.col("combined_vertical")
            / pl.sum("combined_vertical").over("latitude_bin")
        )
        .with_columns(bearing_bin_center_radians=pl.col("bearing_bin_center").radians())
        .sort("latitude_bin", "bearing_bin")
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
        pl.DataFrame({"latitude_bin_lower": LATITUDE_BREAKS[:-1]})
        .join(
            pl.DataFrame({"bearing_bin_lower": BEARING_BREAKS[:-1]}),
            how="cross",
        )
        .select(
            pl.all().cast(pl.Int32),
        )
        .join(
            metrics_df.select(
                "latitude_bin_lower",
                "bearing_bin_lower",
                "combined_vertical",
                "combined_vertical_prop",
            ),
            on=["latitude_bin_lower", "bearing_bin_lower"],
            how="left",
        )
        .with_columns(
            pl.col("combined_vertical").fill_null(0),
            pl.col("combined_vertical_prop").fill_null(0),
        )
        .pivot(
            index="latitude_bin_lower",
            columns="bearing_bin_lower",
            values="combined_vertical_prop",
            sort_columns=False,  # order of discovery
        )
        .sort("latitude_bin_lower")
        .with_columns(pl.selectors.by_dtype(pl.Float64).fill_null(0.0))
        .drop("latitude_bin_lower")
    )
    assert np.all(np.diff([int(x) for x in grid_pl.columns]) > 0)
    latitude_grid, bearing_grid = np.meshgrid(LATITUDE_BREAKS, BEARING_BREAKS)
    return BearingByLatitudeBinMeshGrid(
        latitude_grid=latitude_grid,
        bearing_grid=bearing_grid,
        color_grid=grid_pl.to_numpy(),
    )


def plot_bearing_by_latitude_bin() -> plt.Figure:
    # consider separate plots by hemisphere
    grids = get_bearing_by_latitude_bin_mesh_grids()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.pcolormesh(
        (np.deg2rad(grids.bearing_grid)).T,
        (grids.latitude_grid + 180).T,
        grids.color_grid.clip(min=0.0, max=0.04),
        shading="flat",
        cmap="Purples",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    # TODO: reuse code in plot
    ax.set_xticks(ax.get_xticks())
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)
    ax.set_yticks([-90 + 180, 0 + 180, 90 + 180])
    ax.set_yticklabels(labels=["SP", "Eq", "NP"])
    return fig


def get_latitude_histogram() -> pl.DataFrame:
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
            # FIXME: abstract
            hemisphere=pl.when(pl.col("latitude_bin_center").gt(0))
            .then(pl.lit("north"))
            .otherwise(pl.lit("south")),
            latitude_bin_center_abs=pl.col("latitude_bin_center").abs(),
        )
    )
