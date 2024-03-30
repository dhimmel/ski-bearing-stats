import itertools
import json
import logging
import lzma
import statistics
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import osmnx
import pandas as pd
import polars as pl
import requests
from osmnx.bearing import add_edge_bearings
from osmnx.distance import add_edge_lengths

from ski_bearings.bearing import get_mean_bearing
from ski_bearings.osmnx_utils import suppress_user_warning

data_directory = Path(__file__).parent.parent.joinpath("data")


bearing_labels = {
    0.0: "N",
    11.25: "NbE",
    22.5: "NNE",
    33.75: "NEbN",
    45.0: "NE",
    56.25: "NEbE",
    67.5: "ENE",
    78.75: "EbN",
    90.0: "E",
    101.25: "EbS",
    112.5: "ESE",
    123.75: "SEbE",
    135.0: "SE",
    146.25: "SEbS",
    157.5: "SSE",
    168.75: "SbE",
    180.0: "S",
    191.25: "SbW",
    202.5: "SSW",
    213.75: "SWbS",
    225.0: "SW",
    236.25: "SWbW",
    247.5: "WSW",
    258.75: "WbS",
    270.0: "W",
    281.25: "WbN",
    292.5: "WNW",
    303.75: "NWbW",
    315.0: "NW",
    326.25: "NWbN",
    337.5: "NNW",
    348.75: "NbW",
}
"""Bearing labels for 32-wind compass rose."""


def get_openskimap_path(name: Literal["runs", "ski_areas", "lifts"]) -> Path:
    return data_directory.joinpath(f"{name}.geojson.xz")


def download_openskimap_geojson(name: Literal["runs", "ski_areas", "lifts"]) -> None:
    """Download a single geojson file from OpenSkiMap and save it to disk with compression."""
    if not data_directory.exists():
        data_directory.mkdir()
    url = f"https://tiles.skimap.org/geojson/{name}.geojson"
    path = get_openskimap_path(name)
    logging.info(f"Downloading {url} to {path}")
    response = requests.get(url, allow_redirects=True)
    with lzma.open(path, "wb") as write_file:
        write_file.write(response.content)


def download_openskimap_geojsons() -> None:
    """Download all OpenSkiMap geojson files."""
    for name in ["runs", "ski_areas", "lifts"]:
        download_openskimap_geojson(name)  # type: ignore[arg-type]


@cache
def load_runs() -> Any:
    runs_path = get_openskimap_path("runs")
    if not runs_path.exists():
        download_openskimap_geojson(name="runs")
    with lzma.open(runs_path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    runs = data["features"]
    logging.info(f"Loaded {len(runs):,} runs.")
    return runs


@cache
def load_ski_areas() -> pd.DataFrame:
    ski_areas_path = get_openskimap_path("ski_areas")
    if not ski_areas_path.exists():
        download_openskimap_geojson(name="ski_areas")
    with lzma.open(ski_areas_path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    return pd.json_normalize([x["properties"] for x in data["features"]], sep="__")


class SkiRunDifficulty(StrEnum):
    novice = "novice"
    easy = "easy"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    extreme = "extreme"
    freeride = "freeride"
    other = "other"


@cache
def load_downhill_ski_areas() -> pd.DataFrame:
    ski_areas = load_ski_areas()
    return (
        ski_areas.rename(columns={"id": "ski_area_id", "name": "ski_area_name"})
        .query("type == 'skiArea'")
        .explode("activities")
        .query("activities == 'downhill'")[
            [
                "ski_area_id",
                "ski_area_name",
                "generated",
                "runConvention",
                "status",
                "location__iso3166_1Alpha2",
                "location__iso3166_2",
                "location__localized__en__country",
                "location__localized__en__region",
                "location__localized__en__locality",
                "websites",
                # "sources",  # inconsistently typed nested column 'id' as string or int
                "statistics__minElevation",
                "statistics__maxElevation",
                "statistics__runs__minElevation",
                "statistics__runs__maxElevation",
                *itertools.chain.from_iterable(
                    [
                        f"statistics__runs__byActivity__downhill__byDifficulty__{difficulty}__count",
                        f"statistics__runs__byActivity__downhill__byDifficulty__{difficulty}__lengthInKm",
                        f"statistics__runs__byActivity__downhill__byDifficulty__{difficulty}__combinedElevationChange",
                    ]
                    for difficulty in SkiRunDifficulty
                ),
                "statistics__lifts__minElevation",
                "statistics__lifts__maxElevation",
            ]
        ]
    )


def get_ski_area_to_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    # ski area names can be duplicated, like 'Black Mountain', so use the id instead.
    ski_area_to_runs: dict[str, Any] = {}
    for run in runs:
        if "downhill" not in run["properties"]["uses"]:
            continue
        if not (ski_areas := run["properties"]["skiAreas"]):
            continue
        for ski_area in ski_areas:
            if not (ski_area_id := ski_area["properties"]["id"]):
                continue
            ski_area_to_runs.setdefault(ski_area_id, []).append(run)
    return ski_area_to_runs


def create_networkx(runs: list[Any]) -> nx.MultiDiGraph:
    """
    Convert runs to an newtorkx MultiDiGraph compatible with OSMnx.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")  # https://epsg.io/4326
    graph.graph["run_count"] = len(runs)
    # filter out unsupported geometries like Polygons
    runs = [run for run in runs if run["geometry"]["type"] == "LineString"]
    graph.graph["run_count_filtered"] = len(runs)
    for run in runs:
        # NOTE: longitude comes before latitude in GeoJSON and osmnx, which is different than GPS coordinates
        for lon, lat, elevation in run["geometry"]["coordinates"]:
            graph.add_node((lon, lat), x=lon, y=lat, elevation=elevation)
    for run in runs:
        coordinates = run["geometry"]["coordinates"].copy()
        if coordinates[0][2] < coordinates[-1][2]:
            # Ensure the run is going downhill, such that starting elevation > ending elevation
            coordinates.reverse()
        lon_0, lat_0, elevation_0 = coordinates.pop(0)
        for lon_1, lat_1, elevation_1 in coordinates:
            graph.add_edge(
                (lon_0, lat_0),
                (lon_1, lat_1),
                vertical=max(0.0, elevation_0 - elevation_1),
            )
            lon_0, lat_0, elevation_0 = lon_1, lat_1, elevation_1
    if graph.number_of_edges() > 0:
        graph = add_edge_bearings(graph)
        graph = add_edge_lengths(graph)
    return graph


def create_networkx_with_metadata(
    runs: list[dict[str, Any]], ski_area_metadata: dict[str, Any]
) -> nx.MultiDiGraph:
    # ski_area_id = ski_area_metadata["ski_area_id"]
    # ski_area_name = ski_area_metadata["ski_area_name"]
    graph = create_networkx(runs)
    graph.graph = ski_area_metadata | graph.graph
    if graph.number_of_nodes() > 0:
        graph.graph["latitude"] = statistics.mean(
            lat for _, lat in graph.nodes(data="y")
        )
        graph.graph["hemisphere"] = "north" if graph.graph["latitude"] > 0 else "south"
    if graph.number_of_edges() > 0:
        with suppress_user_warning():
            bearings, weights = osmnx.bearing._extract_edge_bearings(
                graph, min_length=0, weight="vertical"
            )
        graph.graph["combined_vertical"] = sum(weights)
        mean_bearing, mean_bearing_strength = get_mean_bearing(bearings, weights)
        graph.graph["mean_bearing"] = mean_bearing
        graph.graph["mean_bearing_strength"] = mean_bearing_strength
    # graph.graph["orientation_entropy"] = osmnx.orientation_entropy(
    #     graph, num_bins=32, weight="vertical"
    # )
    # fig, ax = osmnx.plot_orientation(
    #     graph,
    #     num_bins=32,
    #     title=ski_area_name or ski_area_id,
    #     area=True,
    #     weight="vertical",
    #     color="#D4A0A7",
    # )
    return graph


def get_bearing_distributions_df(graph: nx.MultiDiGraph) -> pl.DataFrame:
    """
    Get the bearing distribution of a graph as a DataFrame.
    """
    bins = [2, 4, 8, 32]
    return pl.concat(
        [get_bearing_distribution_df(graph, num_bins=num_bins) for num_bins in bins],
        how="vertical",
    )


def get_bearing_distribution_df(graph: nx.MultiDiGraph, num_bins: int) -> pl.DataFrame:
    """
    Get the bearing distribution of a graph as a DataFrame.
    """
    with suppress_user_warning():
        bin_counts, bin_centers = osmnx.bearing._bearings_distribution(
            graph,
            num_bins=num_bins,
            min_length=0,
            weight="vertical",
        )
    # polars make dataframe from bin_counts, and bin_centers
    return (
        pl.DataFrame(
            {
                "bin_center": bin_centers,
                "bin_count": bin_counts,
            }
        )
        .with_columns(
            bin_proportion=pl.col("bin_count") / pl.sum("bin_count").over(pl.lit(True))
        )
        .with_columns(pl.lit(num_bins).alias("num_bins"))
        .with_columns(
            pl.col("bin_center")
            .replace(bearing_labels, default=None)
            .alias("bin_label")
        )
        .with_row_index(name="bin_index", offset=1)
    )


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
