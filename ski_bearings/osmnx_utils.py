import contextlib
import statistics
import warnings
from collections.abc import Generator
from typing import Any

import networkx as nx
import osmnx
from osmnx.bearing import add_edge_bearings
from osmnx.distance import add_edge_lengths

from ski_bearings.bearing import get_mean_bearing


@contextlib.contextmanager
def suppress_user_warning(
    category: type[Warning] = UserWarning, message: str = ""
) -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=message, category=category)
        yield


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
    return graph
