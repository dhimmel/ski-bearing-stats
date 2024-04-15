import logging

import polars as pl

from ski_bearings.bearing import get_bearing_distributions_df
from ski_bearings.openskimap_utils import (
    get_ski_area_to_runs,
    load_downhill_ski_areas,
    load_runs,
)
from ski_bearings.osmnx_utils import (
    create_networkx_with_metadata,
)
from ski_bearings.utils import data_directory

bearing_distribution_path = data_directory.joinpath("bearing_distributions.parquet")
ski_area_metrics_path = data_directory.joinpath("ski_area_metrics.parquet")


def analyze_all_ski_areas() -> None:
    ski_area_df = load_downhill_ski_areas()
    ski_area_metadatas = {
        x["ski_area_id"]: x for x in ski_area_df.to_dict(orient="records")
    }
    runs = load_runs()
    ski_area_to_runs = get_ski_area_to_runs(runs)
    bearing_dist_dfs = []
    ski_area_metrics = []
    for ski_area_id, ski_area_metadata in ski_area_metadatas.items():
        logging.info(
            f"Analyzing {ski_area_id} named {ski_area_metadata['ski_area_name']}"
        )
        ski_area_runs = ski_area_to_runs.get(ski_area_id, [])
        ski_area_metadata = ski_area_metadatas[ski_area_id]
        graph = create_networkx_with_metadata(
            runs=ski_area_runs, ski_area_metadata=ski_area_metadata
        )
        ski_area_metrics.append(graph.graph)
        bearing_dist_df = get_bearing_distributions_df(graph).select(
            pl.lit(ski_area_id).alias("ski_area_id"),
            pl.all(),
        )
        bearing_dist_dfs.append(bearing_dist_df)
    bearing_dist_df = pl.concat(bearing_dist_dfs, how="vertical")
    ski_area_metrics_df = pl.DataFrame(data=ski_area_metrics)
    bearing_dist_df.write_parquet(bearing_distribution_path)
    ski_area_metrics_df.write_parquet(ski_area_metrics_path)


def load_bearing_distribution_pl() -> pl.DataFrame:
    return pl.read_parquet(bearing_distribution_path)


def load_ski_areas_pl() -> pl.DataFrame:
    return pl.read_parquet(ski_area_metrics_path)


def aggregate_ski_area_bearing_dists_pl(
    group_by: list[str],
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    return (
        load_ski_areas_pl()
        .filter(*ski_area_filters)
        .join(load_bearing_distribution_pl(), on="ski_area_id", how="inner")
        .group_by(*group_by, "num_bins", "bin_center")
        .agg(
            pl.first("bin_label").alias("bin_label"),
            pl.sum("bin_count").alias("bin_count"),
        )
        .with_columns(
            bin_count_total=pl.sum("bin_count").over(*group_by, "num_bins"),
        )
        .with_columns(
            bin_proportion=pl.col("bin_count") / pl.col("bin_count_total"),
        )
        .sort(*group_by, "num_bins", "bin_center")
    )


def bearing_dists_by_us_state() -> pl.DataFrame:
    return aggregate_ski_area_bearing_dists_pl(
        group_by=["location__localized__en__region"],
        ski_area_filters=[
            pl.col("location__localized__en__country") == "United States",
            pl.col("location__localized__en__region").is_not_null(),
            pl.col("status") == "operating",
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def bearing_dists_by_hemisphere() -> pl.DataFrame:
    return aggregate_ski_area_bearing_dists_pl(
        group_by=["hemisphere"],
        ski_area_filters=[
            pl.col("hemisphere").is_not_null(),
            pl.col("status") == "operating",
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def bearing_dists_by_country() -> pl.DataFrame:
    return aggregate_ski_area_bearing_dists_pl(
        group_by=["location__localized__en__country"],
        ski_area_filters=[
            pl.col("location__localized__en__country").is_not_null(),
            pl.col("status") == "operating",
            pl.col("ski_area_name").is_not_null(),
        ],
    )
