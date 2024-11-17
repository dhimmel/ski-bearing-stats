import logging
from pathlib import Path

import matplotlib.pyplot
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from patito.exceptions import DataFrameValidationError

from ski_bearings.bearing import (
    add_spatial_metric_columns,
    get_bearing_histograms,
    get_bearing_summary_stats,
)
from ski_bearings.models import BearingStatsModel, SkiAreaModel
from ski_bearings.openskimap_utils import (
    load_downhill_ski_areas_from_download_pl,
    load_runs_from_download_pl,
)
from ski_bearings.plot import plot_orientation, subplot_orientations
from ski_bearings.utils import get_data_directory


def get_ski_area_metrics_path(testing: bool = False) -> Path:
    return get_data_directory(testing=testing).joinpath("ski_area_metrics.parquet")


def get_runs_parquet_path(testing: bool = False) -> Path:
    return get_data_directory(testing=testing).joinpath("runs.parquet")


def process_and_export_runs() -> None:
    """
    Process and export runs from OpenSkiMap.
    """
    runs_df = load_runs_from_download_pl().lazy()
    coords_df = (
        runs_df.select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .pipe(add_spatial_metric_columns, partition_by="run_id")
        .group_by("run_id")
        .agg(run_coordinates_clean=pl.struct(pl.exclude("run_id")))
    )
    runs_df = (
        runs_df.drop("run_coordinates_clean").join(coords_df, on="run_id").collect()
    )
    runs_path = get_runs_parquet_path()
    logging.info(f"Writing {len(runs_df):,} runs to {runs_path}")
    runs_df.write_parquet(runs_path)


def analyze_all_ski_areas_polars(skip_runs: bool = False) -> None:
    """
    Analyze ski areas to create a table of ski areas and their metrics
    including bearing distributions.
    Keyed on ski_area_id.
    Write data as parquet.
    """
    if not skip_runs:
        process_and_export_runs()
    logging.info("Creating ski area metrics dataframe with bearing distributions.")
    ski_area_df = load_downhill_ski_areas_from_download_pl().lazy()
    ski_area_run_metrics_df = (
        load_runs_pl()
        .lazy()
        .filter(pl.col("run_uses").list.contains("downhill"))
        .explode("ski_area_ids")
        .rename({"ski_area_ids": "ski_area_id"})
        .filter(pl.col("ski_area_id").is_not_null())
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .with_columns(
            hemisphere=pl.when(pl.col("latitude").gt(0))
            .then(pl.lit("north"))
            .otherwise(pl.lit("south")),
        )
        .group_by("ski_area_id")
        .agg(
            run_count=pl.col("run_id").n_unique(),
            coordinate_count=pl.len(),
            segment_count=pl.count("segment_hash"),
            combined_vertical=pl.col("distance_vertical_drop").sum(),
            combined_distance=pl.col("distance_3d").sum(),
            latitude=pl.col("latitude").mean(),
            longitude=pl.col("longitude").mean(),
            min_elevation=pl.col("elevation").min(),
            max_elevation=pl.col("elevation").max(),
            hemisphere=pl.first("hemisphere"),
            _bearing_stats=pl.struct(
                "bearing",
                pl.col("distance_vertical_drop").alias("bearing_magnitude_net"),
                pl.col("distance_vertical_drop").alias("bearing_magnitude_cum"),
                "hemisphere",
            ).map_batches(_get_bearing_summary_stats_pl, returns_scalar=True),
            # do we need to filter nulls?
            bearings=pl.struct("bearing", "distance_vertical_drop").map_batches(
                lambda x: get_bearing_histograms(
                    bearings=x.struct.field("bearing"),
                    weights=x.struct.field("distance_vertical_drop"),
                ).to_struct(),
                returns_scalar=True,
            ),
        )
        .unnest("_bearing_stats")
    )
    ski_area_metrics_df = (
        ski_area_df.join(ski_area_run_metrics_df, on="ski_area_id", how="left")
        .with_columns(
            *[
                pl.col(col).fill_null(pl.lit(fill_value))
                for col, fill_value in SkiAreaModel.defaults.items()
            ]
        )
        .collect()
    )
    try:
        SkiAreaModel.validate(ski_area_metrics_df, allow_superfluous_columns=True)
    except DataFrameValidationError as exc:
        logging.error(f"SkiAreaModel.validate failed with {exc}")
    ski_area_metrics_path = get_ski_area_metrics_path()
    logging.info(f"Writing {ski_area_metrics_path}")
    ski_area_metrics_df.write_parquet(ski_area_metrics_path)


def load_runs_pl() -> pl.LazyFrame:
    path = get_runs_parquet_path()
    logging.info(f"Loading runs metrics from {path}")
    return pl.scan_parquet(source=path)


def load_ski_areas_pl(ski_area_filters: list[pl.Expr] | None = None) -> pl.DataFrame:
    path = get_ski_area_metrics_path()
    logging.info(f"Loading ski area metrics from {path}")
    return pl.read_parquet(source=path).filter(
        *_prepare_ski_area_filters(ski_area_filters)
    )


def load_bearing_distribution_pl(
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    """
    Table of ski area bearing distributions.
    Keyed on ski_area_id, num_bins, bin_center.
    """
    return (
        load_ski_areas_pl(ski_area_filters=ski_area_filters)
        .select("ski_area_id", "bearings")
        .explode("bearings")
        .filter(pl.col("bearings").is_not_null())
        .unnest("bearings")
    )


def _get_bearing_summary_stats_pl(struct_series: pl.Series) -> BearingStatsModel:
    df = (
        struct_series.alias("input_struct")
        .to_frame()
        .unnest("input_struct")
        .drop_nulls()
    )
    try:
        (hemisphere,) = df.get_column("hemisphere").unique()
    except ValueError:
        hemisphere = None
    return get_bearing_summary_stats(
        bearings=df.get_column("bearing").to_numpy(),
        net_magnitudes=df.get_column("bearing_magnitude_net").to_numpy(),
        cum_magnitudes=df.get_column("bearing_magnitude_cum").to_numpy(),
        hemisphere=hemisphere,
    )


def _prepare_ski_area_filters(
    ski_area_filters: list[pl.Expr] | None = None,
) -> list[pl.Expr | bool]:
    if not ski_area_filters:
        # pl.lit(True) had issues that True did not.
        # https://github.com/pola-rs/polars/issues/19771
        ski_area_filters = [True]
    return ski_area_filters


def aggregate_ski_areas_pl(
    group_by: list[str],
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    assert len(group_by) > 0
    bearings_pl = (
        aggregate_ski_area_bearing_dists_pl(
            group_by=group_by, ski_area_filters=ski_area_filters
        )
        .group_by(*group_by)
        .agg(bearings=pl.struct(pl.exclude(group_by)))
    )
    return (
        load_ski_areas_pl(ski_area_filters=ski_area_filters)
        .group_by(*group_by)
        .agg(
            ski_areas_count=pl.n_unique("ski_area_id"),
            country_count=pl.n_unique("country"),
            run_count=pl.sum("run_count"),
            coordinate_count=pl.sum("coordinate_count"),
            segment_count=pl.sum("segment_count"),
            lift_count=pl.sum("lift_count"),
            combined_vertical=pl.sum("combined_vertical"),
            combined_distance=pl.sum("combined_distance"),
            min_elevation=pl.min("min_elevation"),
            max_elevation=pl.max("max_elevation"),
            latitude=pl.mean("latitude"),
            longitude=pl.mean("longitude"),
            _bearing_stats=pl.struct(
                pl.col("bearing_mean").alias("bearing"),
                "bearing_magnitude_net",
                "bearing_magnitude_cum",
                "hemisphere",
            ).map_batches(_get_bearing_summary_stats_pl, returns_scalar=True),
        )
        .unnest("_bearing_stats")
        .join(bearings_pl, on=group_by)
        .sort(*group_by)
    )


def aggregate_ski_area_bearing_dists_pl(
    group_by: list[str],
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    return (
        load_ski_areas_pl(ski_area_filters=ski_area_filters)
        .explode("bearings")
        .filter(pl.col("bearings").is_not_null())
        .unnest("bearings")
        .group_by(*group_by, "num_bins", "bin_index")
        .agg(
            bin_center=pl.first("bin_center"),
            bin_count=pl.sum("bin_count"),
            bin_label=pl.first("bin_label"),
        )
        .with_columns(
            bin_count_total=pl.sum("bin_count").over(*group_by, "num_bins"),
        )
        # nan values are not helpful here when bin_count_total is 0
        # Instead, set bin_proportion to 0, although setting to null could also make sense
        .with_columns(
            bin_proportion=pl.when(pl.col("bin_count_total") > 0)
            .then(pl.col("bin_count").truediv("bin_count_total"))
            .otherwise(0.0)
        )
        .drop("bin_count_total")
        .sort(*group_by, "num_bins", "bin_center")
    )


def bearing_dists_by_us_state() -> pl.DataFrame:
    return aggregate_ski_areas_pl(
        group_by=["region"],
        ski_area_filters=[
            pl.col("country") == "United States",
            pl.col("region").is_not_null(),
            pl.col("osm_status") == "operating",
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def bearing_dists_by_hemisphere() -> pl.DataFrame:
    return aggregate_ski_areas_pl(
        group_by=["hemisphere"],
        ski_area_filters=[
            pl.col("hemisphere").is_not_null(),
            pl.col("osm_status") == "operating",
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def bearing_dists_by_status() -> pl.DataFrame:
    """
    Bearing distributions by ski area operating status.
    Only includes ski areas in the northern hemisphere because there were no abandoned ski areas in the southern hemisphere at ToW.
    """
    return aggregate_ski_areas_pl(
        group_by=["hemisphere", "osm_status"],
        ski_area_filters=[
            pl.col("hemisphere") == "north",
            pl.col("osm_status").is_in(["abandoned", "operating"]),
        ],
    ).with_columns(
        group_name=pl.format(
            "{} in {} Hem.",
            pl.col("osm_status").str.to_titlecase(),
            pl.col("hemisphere").str.to_titlecase(),
        )
    )


def bearing_dists_by_country() -> pl.DataFrame:
    return aggregate_ski_areas_pl(
        group_by=["country"],
        ski_area_filters=[
            pl.col("country").is_not_null(),
            pl.col("osm_status") == "operating",
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def ski_rose_the_world(min_combined_vertical: int = 10_000) -> pl.DataFrame:
    path = get_data_directory().joinpath("ski-roses.pdf")
    pdf_pages = PdfPages(
        filename=path,
        metadata={
            "Title": "Ski Roses of the World: Downhill Ski Trail Orientations",
            "Author": "https://github.com/dhimmel/ski-bearing-stats",
        },
    )
    grouping_col_to_stats = {
        "hemisphere": bearing_dists_by_hemisphere(),
        "group_name": bearing_dists_by_status(),
        "country": bearing_dists_by_country(),
        "region": bearing_dists_by_us_state(),
    }
    figures = []
    for grouping_col, groups_pl in grouping_col_to_stats.items():
        logging.info(f"Plotting ski roses by {grouping_col}")
        groups_pl = groups_pl.filter(
            pl.col("combined_vertical") >= min_combined_vertical
        )
        if groups_pl.is_empty():
            logging.info(
                f"Skipping {grouping_col} plot which returns no groups with combined_vertical >= {min_combined_vertical:,}m."
            )
            continue
        fig = subplot_orientations(
            groups_pl=groups_pl,
            grouping_col=grouping_col,
            n_cols=min(4, len(groups_pl)),
            free_y=True,
        )
        figures.append(fig)
    logging.info(f"Writing ski rose the world to {path}")
    with pdf_pages:
        for fig in figures:
            pdf_pages.savefig(fig, facecolor="#FFFFFF", bbox_inches="tight")
            matplotlib.pyplot.close(fig)


def get_display_ski_area_filters() -> list[pl.Expr]:
    """Ski area filters to produce a subset of ski areas for display."""
    return [
        pl.col("run_count") >= 3,
        pl.col("combined_vertical") >= 50,
        pl.col("ski_area_name").is_not_null(),
    ]


def create_ski_area_roses(overwrite: bool = False) -> None:
    """
    Export ski area roses to SVG for display.
    """
    directory = get_data_directory().joinpath("ski_areas")
    directory.mkdir(exist_ok=True)
    partitions = (
        load_bearing_distribution_pl(ski_area_filters=get_display_ski_area_filters())
        .filter(pl.col("num_bins") == 8)
        .partition_by("ski_area_id", as_dict=True)
    )
    logging.info(
        f"Creating ski area roses for {len(partitions)} ski areas in {directory}."
    )
    for (ski_area_id,), bearing_pl in partitions.items():
        path = directory.joinpath(f"{ski_area_id}.svg")
        if not overwrite and path.exists():
            continue
        fig, ax = plot_orientation(
            bin_counts=bearing_pl.get_column("bin_count").to_numpy(),
            bin_centers=bearing_pl.get_column("bin_center").to_numpy(),
            margin_text={},
            figsize=(1, 1),
            alpha=1.0,
            edgecolor="#6b6b6b",
            linewidth=0.4,
            disable_xticks=True,
        )
        # make the polar frame less prominent
        ax.spines["polar"].set_linewidth(0.4)
        ax.spines["polar"].set_color("#6b6b6b")
        # TODO: reduce margins around svg
        logging.info(f"Writing {path}")
        fig.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            metadata={
                "Title": "Ski Roses of the World: Downhill Ski Trail Orientations",
                "Creator": "https://github.com/dhimmel/ski-bearing-stats",
            },
        )
        matplotlib.pyplot.close(fig)
