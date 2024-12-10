import polars as pl
from polars.testing import assert_frame_equal

from openskistats.utils import pl_flip_bearing


def test_pl_flip_bearing() -> None:
    expected_df = pl.DataFrame(
        [
            {"latitude": 10, "bearing": 0, "bearing_poleward": 0},
            {"latitude": 10, "bearing": 180, "bearing_poleward": 180},
            {"latitude": 10, "bearing": 45, "bearing_poleward": 45},
            {"latitude": -10, "bearing": 0, "bearing_poleward": 180},
            {"latitude": -10, "bearing": 5, "bearing_poleward": 175},
            {"latitude": -10, "bearing": 45, "bearing_poleward": 135},
            {"latitude": -10, "bearing": 90, "bearing_poleward": 90},
            {"latitude": -10, "bearing": 95, "bearing_poleward": 85},
            {"latitude": -10, "bearing": 180, "bearing_poleward": 0},
            {"latitude": -10, "bearing": 265, "bearing_poleward": 275},
            {"latitude": -10, "bearing": 270, "bearing_poleward": 270},
            {"latitude": -10, "bearing": 360, "bearing_poleward": 180},
        ]
    )
    output_df = expected_df.with_columns(bearing_poleward=pl_flip_bearing())
    assert_frame_equal(output_df, expected_df)
