import itertools
import json
import logging
import lzma
from collections import Counter
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import polars as pl
import requests

from ski_bearings.utils import data_directory


class SkiRunDifficulty(StrEnum):
    novice = "novice"
    easy = "easy"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    extreme = "extreme"
    freeride = "freeride"
    other = "other"


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
    geometry_types = Counter(run["geometry"]["type"] for run in runs)
    logging.info(f"Loaded {len(runs):,} runs with geometry types {geometry_types}")
    return runs


def load_runs_pl() -> pl.DataFrame:
    runs = load_runs()
    rows = []
    for run in runs:
        if run["geometry"]["type"] != "LineString":
            continue
        row = {}
        run_properties = run["properties"]
        row["run_id"] = run_properties["id"]
        row["run_name"] = run_properties["name"]
        row["run_uses"] = run_properties["uses"]
        row["run_status"] = run_properties["status"]
        row["run_difficulty"] = run_properties["difficulty"]
        row["run_convention"] = run_properties["convention"]
        row["ski_area_ids"] = sorted(
            ski_area["properties"]["id"] for ski_area in run_properties["skiAreas"]
        )
        row["run_sources"] = sorted(
            "{type}:{id}".format(**source) for source in run_properties["sources"]
        )
        coordinate_rows = []
        for i, (lon, lat, ele) in enumerate(
            _clean_coordinates(run["geometry"]["coordinates"])
        ):
            coordinate_rows.append(
                {"index": i, "latitude": lat, "longitude": lon, "elevation": ele}
            )
        row["run_coordinates"] = coordinate_rows
        rows.append(row)
    return pl.DataFrame(rows, strict=False)


@cache
def load_ski_areas() -> pd.DataFrame:
    ski_areas_path = get_openskimap_path("ski_areas")
    if not ski_areas_path.exists():
        download_openskimap_geojson(name="ski_areas")
    with lzma.open(ski_areas_path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    return pd.json_normalize([x["properties"] for x in data["features"]], sep="__")


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


def _clean_coordinates(
    coordinates: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Sanitize run LineString coordinates to remove floating point errors and ensure downhill runs.
    NOTE: longitude comes before latitude in GeoJSON and osmnx, which is different than GPS coordinates.
    """
    # Round coordinates to undo floating point errors.
    # https://github.com/russellporter/openskimap.org/issues/137
    coordinates = [
        (round(lon, 7), round(lat, 7), round(ele, 2)) for lon, lat, ele in coordinates
    ]
    if coordinates[0][2] < coordinates[-1][2]:
        # Ensure the run is going downhill, such that starting elevation > ending elevation
        coordinates.reverse()
    return coordinates
