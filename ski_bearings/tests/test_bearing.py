from dataclasses import dataclass

import numpy as np
import pytest

from ski_bearings.bearing import get_bearing_summary_stats


@dataclass
class BearingSummaryStatsPytestParam:
    """
    Dataclass for named pytest parameter definitions.
    https://github.com/pytest-dev/pytest/issues/9216
    """

    bearings: list[float]
    weights: list[float]
    expected_bearing: float
    expected_strength: float


@pytest.mark.parametrize(
    "param",
    [
        BearingSummaryStatsPytestParam(
            bearings=[0.0],
            weights=[2.0],
            expected_bearing=0.0,
            expected_strength=1.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=[1.0, 1.0],
            expected_bearing=45.0,
            expected_strength=0.7071068,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 360.0],
            weights=[1.0, 1.0],
            expected_bearing=360.0,
            expected_strength=1.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=[0.0, 1.0],
            expected_bearing=90.0,
            expected_strength=1.0,
        ),
    ],
)
def test_get_bearing_summary_stats(param: BearingSummaryStatsPytestParam) -> None:
    stats = get_bearing_summary_stats(np.array(param.bearings), np.array(param.weights))
    assert stats.mean_bearing_deg == pytest.approx(param.expected_bearing)
    assert stats.mean_bearing_strength == pytest.approx(param.expected_strength)
