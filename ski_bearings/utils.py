import itertools
import json
import logging
import lzma
import math
import statistics
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import osmnx
import pandas as pd
import requests
from osmnx.bearing import add_edge_bearings
from osmnx.distance import add_edge_lengths

data_directory = Path(__file__).parent.parent.joinpath("data")


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


def load_runs() -> Any:
    runs_path = get_openskimap_path("runs")
    if not runs_path.exists():
        download_openskimap_geojson(name="runs")
    with lzma.open(runs_path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    return data["features"]


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


def load_downhill_ski_areas() -> pd.DataFrame:
    ski_areas = load_ski_areas()
    return (
        ski_areas.query("type == 'skiArea'")
        .explode("activities")
        .query("activities == 'downhill'")[
            [
                "id",
                "name",
                "generated",
                "runConvention",
                "status",
                "location__iso3166_1Alpha2",
                "location__iso3166_2",
                "location__localized__en__country",
                "location__localized__en__region",
                "location__localized__en__locality",
                "websites",
                "sources",
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
    NOTE: Bearing distributions lack support for directed graphs
    https://github.com/gboeing/osmnx/issues/1137
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")  # https://epsg.io/4326
    # filter out unsupported geometries like Polygons
    runs = [run for run in runs if run["geometry"]["type"] == "LineString"]
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
                vertical=elevation_0 - elevation_1,
            )
            lon_0, lat_0, elevation_0 = lon_1, lat_1, elevation_1
    graph = add_edge_bearings(graph)
    graph = add_edge_lengths(graph)
    return graph


def analyze_ski_area(
    runs: list[dict[str, Any]], ski_area_metadata: dict[str, Any]
) -> nx.MultiDiGraph:
    ski_area_id = ski_area_metadata["id"]
    ski_area_name = ski_area_metadata["name"]
    graph = create_networkx(runs)
    graph.graph.update(ski_area_metadata)
    graph.graph["latitude"] = statistics.mean(lat for _, lat in graph.nodes(data="y"))
    graph.graph["hemisphere"] = "north" if graph.graph["latitude"] > 0 else "south"
    graph.graph["orientation_entropy"] = osmnx.orientation_entropy(
        graph, weight="vertical"
    )
    bin_counts, bin_edges = osmnx.bearing._bearings_distribution(
        graph,
        num_bins=2,
        min_length=0,
        weight="vertical",
    )
    graph.graph["orientation_2_north"], graph.graph["orientation_2_south"] = bin_counts
    bin_counts, bin_edges = osmnx.bearing._bearings_distribution(
        graph, num_bins=4, min_length=0, weight="vertical"
    )
    (
        graph.graph["orientation_4_north"],
        graph.graph["orientation_4_east"],
        graph.graph["orientation_4_south"],
        graph.graph["orientation_4_west"],
    ) = bin_counts
    fig, ax = osmnx.plot_orientation(
        graph,
        title=ski_area_name or ski_area_id,
        area=True,
        weight="vertical",
        color="#D4A0A7",
    )
    return graph


def subplot_orientations(groupings: dict[str, nx.MultiDiGraph]) -> plt.Figure:
    """
    Plot orientations from multiple graphs in a grid.
    https://github.com/gboeing/osmnx-examples/blob/bb870c225906db5a7b02c4c87a28095cb9dceb30/notebooks/17-street-network-orientations.ipynb
    """
    # create figure and axes
    n_groupings = len(groupings)
    n_cols = math.ceil(n_groupings**0.5)
    n_rows = math.ceil(n_groupings / n_cols)
    figsize = (n_cols * 5, n_rows * 5)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, subplot_kw={"projection": "polar"}
    )

    # plot each group's polar histogram
    for ax, (name, graph) in zip(axes.flat, sorted(groupings.items()), strict=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                # message="edge bearings will be directional",
                category=UserWarning,
            )
            fig, ax = osmnx.plot_orientation(
                graph,
                ax=ax,
                title=name,
                area=True,
                weight="vertical",
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
