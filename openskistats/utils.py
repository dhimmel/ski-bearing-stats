import os
from pathlib import Path

import polars as pl

repo_directory = Path(__file__).parent.parent
data_directory = repo_directory.joinpath("data")
test_data_directory = Path(__file__).parent.joinpath("tests", "data")


def get_data_directory(testing: bool = False) -> Path:
    directory = (
        test_data_directory
        if testing or "PYTEST_CURRENT_TEST" in os.environ
        else data_directory
    )
    directory.mkdir(exist_ok=True)
    return directory


def pl_hemisphere(latitude_col: str = "latitude") -> pl.Expr:
    return (
        pl.when(pl.col(latitude_col).gt(0))
        .then(pl.lit("north"))
        .when(pl.col(latitude_col).lt(0))
        .then(pl.lit("south"))
    )


def pl_flip_bearing(
    latitude_col: str = "latitude", bearing_col: str = "bearing"
) -> pl.Expr:
    """
    Flip bearings in the southern hemisphere across the hemisphere (east-west axis).
    Performs a latitudinal reflection or hemispherical flip of bearings in the southern hemisphere,
    while returning the original bearings in the northern hemisphere.
    """
    return (
        pl.when(pl.col(latitude_col).gt(0))
        .then(pl.col(bearing_col))
        .otherwise(pl.lit(180).sub(bearing_col).mod(360))
    )
