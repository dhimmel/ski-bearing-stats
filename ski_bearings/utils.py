import json
import pathlib
from typing import Any
from urllib.request import urlretrieve

data_directory = pathlib.Path(__file__).parent.parent.joinpath("data")
runs_path = data_directory.joinpath("runs.geojson")


def download_runs() -> None:
    if not data_directory.exists():
        data_directory.mkdir()
    url = "https://tiles.skimap.org/geojson/runs.geojson"
    urlretrieve(url, runs_path)


def load_runs() -> Any:
    if not runs_path.exists():
        download_runs()
    with runs_path.open() as read_file:
        return json.load(read_file)
