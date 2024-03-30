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
    bearing_dist_df.write_parquet(
        data_directory.joinpath("bearing_distributions.parquet")
    )
    ski_area_metrics_df.write_parquet(
        data_directory.joinpath("ski_area_metrics.parquet")
    )
