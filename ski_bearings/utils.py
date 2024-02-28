import json
import logging
import lzma
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

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
    ski_area_to_runs: dict[str, Any] = {}
    for run in runs:
        if not (ski_areas := run["properties"]["skiAreas"]):
            continue
        for ski_area in ski_areas:
            if not (ski_area_name := ski_area["properties"]["name"]):
                continue
            ski_area_to_runs.setdefault(ski_area_name, []).append(run)
    return ski_area_to_runs
