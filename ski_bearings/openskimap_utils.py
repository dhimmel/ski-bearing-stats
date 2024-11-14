import json
import logging
import lzma
import os
from collections import Counter
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal

import polars as pl
import requests

from ski_bearings.models import RunCoordinateModel
from ski_bearings.utils import data_directory, test_data_directory


class SkiRunDifficulty(StrEnum):
    novice = "novice"
    easy = "easy"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    extreme = "extreme"
    freeride = "freeride"
    other = "other"


def get_openskimap_path(
    name: Literal["runs", "ski_areas", "lifts"], testing: bool = False
) -> Path:
    if testing or "PYTEST_CURRENT_TEST" in os.environ:
        return test_data_directory.joinpath(f"{name}.geojson")
    return data_directory.joinpath(f"{name}.geojson.xz")


def download_openskimap_geojson(name: Literal["runs", "ski_areas", "lifts"]) -> None:
    """Download a single geojson file from OpenSkiMap and save it to disk with compression."""
    if not data_directory.exists():
        data_directory.mkdir()
    url = f"https://tiles.openskimap.org/geojson/{name}.geojson"
    path = get_openskimap_path(name)
    logging.info(f"Downloading {url} to {path}")
    headers = {
        "From": "https://github.com/dhimmel/ski-bearing-stats",
        "Referer": "https://openskimap.org/",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
    }
    response = requests.get(url, allow_redirects=True, headers=headers)
    response.raise_for_status()
    with lzma.open(path, "wb") as write_file:
        write_file.write(response.content)
    compressed_size_mb = path.stat().st_size / 1024**2
    logging.info(f"compressed size of {path.name} is {compressed_size_mb:.2f} MB")


def download_openskimap_geojsons() -> None:
    """Download all OpenSkiMap geojson files."""
    for name in ["runs", "ski_areas", "lifts"]:
        download_openskimap_geojson(name)  # type: ignore [arg-type]


@cache
def load_runs_from_download() -> list[Any]:
    runs_path = get_openskimap_path("runs")
    logging.info(f"Loading runs from {runs_path}")
    opener = lzma.open if runs_path.suffix == ".xz" else open
    with opener(runs_path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    runs = data["features"]
    assert isinstance(runs, list)
    geometry_types = Counter(run["geometry"]["type"] for run in runs)
    logging.info(f"Loaded {len(runs):,} runs with geometry types {geometry_types}")
    return runs


def _structure_coordinates(
    coordinates: list[tuple[float, float, float]],
) -> list[RunCoordinateModel]:
    return [
        RunCoordinateModel(
            index=i,
            latitude=lat,
            longitude=lon,
            elevation=ele,
        )
        for i, (lon, lat, ele) in enumerate(coordinates)
    ]


def load_runs_from_download_pl() -> pl.DataFrame:
    runs = load_runs_from_download()
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
        coordinates = run["geometry"]["coordinates"]
        row["run_coordinates_raw"] = [
            x.model_dump() for x in _structure_coordinates(coordinates)
        ]
        row["run_coordinates_clean"] = [
            x.model_dump()
            for x in _structure_coordinates(_clean_coordinates(coordinates))
        ]
        rows.append(row)
    return pl.DataFrame(rows, strict=False)


def load_ski_areas_from_download() -> list[Any]:
    ski_areas_path = get_openskimap_path("ski_areas")
    # polars cannot decompress xz: https://github.com/pola-rs/polars/pull/18536
    opener = lzma.open if ski_areas_path.suffix == ".xz" else open
    with opener(ski_areas_path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    ski_areas = data["features"]
    assert isinstance(ski_areas, list)
    logging.info(f"Loaded {len(ski_areas):,} ski areas.")
    return ski_areas


def load_ski_areas_from_download_pl() -> pl.DataFrame:
    return pl.json_normalize(
        data=[x["properties"] for x in load_ski_areas_from_download()],
        separator="__",
        strict=False,
    ).rename(mapping={"id": "ski_area_id", "name": "ski_area_name"})


@cache
def load_downhill_ski_areas_from_download_pl() -> pl.DataFrame:
    return (
        load_ski_areas_from_download_pl()
        .filter(pl.col("type") == "skiArea")
        .filter(pl.col("activities").list.contains("downhill"))
        .select(
            "ski_area_id",
            "ski_area_name",
            "generated",
            "runConvention",
            "status",
            pl.col("location__localized__en__country").alias("country"),
            pl.col("location__localized__en__region").alias("region"),
            pl.col("location__localized__en__locality").alias("locality"),
            pl.col("location__iso3166_1Alpha2").alias("country_code"),
            pl.col("location__iso3166_2").alias("country_subdiv_code"),
            "websites",
            # sources can have inconsistently typed nested column 'id' as string or int
            pl.col("sources")
            .list.eval(
                pl.concat_str(
                    pl.element().struct.field("type"),
                    pl.element().struct.field("id"),
                    separator=":",
                )
            )
            .alias("sources"),
            "statistics__minElevation",
            "statistics__maxElevation",
            "statistics__runs__minElevation",
            "statistics__runs__maxElevation",
            # *itertools.chain.from_iterable(
            #     [
            #         f"statistics__runs__byActivity__downhill__byDifficulty__{difficulty}__count",
            #         f"statistics__runs__byActivity__downhill__byDifficulty__{difficulty}__lengthInKm",
            #         f"statistics__runs__byActivity__downhill__byDifficulty__{difficulty}__combinedElevationChange",
            #     ]
            #     for difficulty in SkiRunDifficulty
            # ),
            "statistics__lifts__minElevation",
            "statistics__lifts__maxElevation",
        )
    )


def get_ski_area_to_runs(
    runs_pl: pl.DataFrame,
) -> dict[str, list[list[tuple[float, float, float]]]]:
    """
    For each ski area, get a list of runs, where each run is a list of (lon, lat, ele) coordinates.
    """
    # ski area names can be duplicated, like 'Black Mountain', so use the id instead.
    return dict(
        runs_pl.filter(pl.col("run_uses").list.contains("downhill"))
        .explode("ski_area_ids")
        .filter(pl.col("ski_area_ids").is_not_null())
        .select(
            pl.col("ski_area_ids").alias("ski_area_id"),
            pl.col("run_coordinates_clean")
            .list.eval(
                pl.concat_list(
                    pl.element().struct.field("longitude"),
                    pl.element().struct.field("latitude"),
                    pl.element().struct.field("elevation"),
                ).list.to_array(width=3)
            )
            .alias("run_coordinates_tuples"),
        )
        .group_by("ski_area_id")
        .agg(
            pl.col("run_coordinates_tuples").alias("run_coordinates"),
        )
        .iter_rows()
    )


def _clean_coordinates(
    coordinates: list[tuple[float, float, float]],
    min_elevation: float = -100.0,
    ensure_downhill: bool = True,
) -> list[tuple[float, float, float]]:
    """
    Sanitize run LineString coordinates to remove floating point errors and ensure downhill runs.
    NOTE: longitude comes before latitude in GeoJSON and osmnx, which is different than GPS coordinates.
    """
    # Round coordinates to undo floating point errors.
    # https://github.com/russellporter/openskimap.org/issues/137
    clean_coords = []
    for lon, lat, ele in coordinates:
        if ele < min_elevation:
            # remove extreme negative elevations
            # https://github.com/russellporter/openskimap.org/issues/141
            continue
        # TODO: consider extreme slope filtering
        clean_coords.append((round(lon, 7), round(lat, 7), round(ele, 2)))
    if not clean_coords:
        return clean_coords
    if ensure_downhill and (clean_coords[0][2] < clean_coords[-1][2]):
        # Ensure the run is going downhill, such that starting elevation > ending elevation
        clean_coords.reverse()
    return clean_coords


test_ski_area_ids = [
    "8896cde00150e73de1f1237320c88767c91ce099",  # Whaleback Mountain
    "dc24f332f3117625dc09479b5d10cbb31a592be4",  # Storrs Hill Ski Area
]


def generate_openskimap_test_data() -> None:
    test_run_features = []
    for run in load_runs_from_download():
        for ski_area in run["properties"]["skiAreas"]:
            if ski_area["properties"]["id"] in test_ski_area_ids:
                test_run_features.append(run)
    test_runs = {
        "type": "FeatureCollection",
        "features": test_run_features,
    }
    test_ski_areas = {
        "type": "FeatureCollection",
        "features": [
            x
            for x in load_ski_areas_from_download()
            if x["properties"]["id"] in test_ski_area_ids
        ],
    }
    assert len(test_runs["features"]) == len(test_ski_area_ids)
    get_openskimap_path("runs", testing=True).write_text(
        json.dumps(test_runs, indent=2, ensure_ascii=False)
    )
    get_openskimap_path("ski_areas", testing=True).write_text(
        json.dumps(test_ski_areas, indent=2, ensure_ascii=False)
    )
