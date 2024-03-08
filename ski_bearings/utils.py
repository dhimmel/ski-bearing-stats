import json
import logging
import lzma
import math
import warnings
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
    return pd.json_normalize([x["properties"] for x in data["features"]])


def get_ski_area_to_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    # using name is bad, duplicates like Black Mountain
    ski_area_to_runs: dict[str, Any] = {}
    for run in runs:
        if not (ski_areas := run["properties"]["skiAreas"]):
            continue
        for ski_area in ski_areas:
            if not (ski_area_name := ski_area["properties"]["name"]):
                continue
            ski_area_to_runs.setdefault(ski_area_name, []).append(run)
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
                message="edge bearings will be directional",
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
