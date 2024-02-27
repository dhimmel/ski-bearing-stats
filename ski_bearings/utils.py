import json
import lzma
import pathlib
from typing import Any

import requests

data_directory = pathlib.Path(__file__).parent.parent.joinpath("data")
runs_path = data_directory.joinpath("runs.geojson.xz")


def download_runs() -> None:
    if not data_directory.exists():
        data_directory.mkdir()
    url = "https://tiles.skimap.org/geojson/runs.geojson"
    response = requests.get(url, allow_redirects=True)
    with lzma.open(runs_path, "wb") as write_file:
        write_file.write(response.content)


def load_runs() -> Any:
    if not runs_path.exists():
        download_runs()
    with lzma.open(runs_path) as read_file:
        return json.load(read_file)


def get_ski_area_to_runs(runs: dict[str, Any]) -> dict[str, Any]:
    ski_area_to_runs: dict[str, Any] = {}
    for run in runs["features"]:
        if not (ski_areas := run["properties"]["skiAreas"]):
            continue
        for ski_area in ski_areas:
            if not (ski_area_name := ski_area["properties"]["name"]):
                continue
            ski_area_to_runs.setdefault(ski_area_name, []).append(run)
    return ski_area_to_runs
