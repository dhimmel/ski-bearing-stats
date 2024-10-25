from dataclasses import dataclass
from typing import Literal

import pytest

from ski_bearings.bearing import get_bearing_summary_stats


@dataclass
class BearingSummaryStatsPytestParam:
    """
    Dataclass for named pytest parameter definitions.
    https://github.com/pytest-dev/pytest/issues/9216
    """

    bearings: list[float] | None
    strengths: list[float] | None
    weights: list[float] | None
    hemisphere: Literal["north", "south"] | None
    expected_bearing: float
    expected_strength: float
    expected_poleward_affinity: float | None = None
    excepted_eastward_affinity: float | None = None


@pytest.mark.parametrize(
    "param",
    [
        BearingSummaryStatsPytestParam(
            bearings=[0.0],
            strengths=None,
            weights=[2.0],
            hemisphere="north",
            expected_bearing=0.0,
            expected_strength=1.0,
            expected_poleward_affinity=1.0,
            excepted_eastward_affinity=0.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            strengths=None,
            weights=[1.0, 1.0],
            hemisphere="south",
            expected_bearing=45.0,
            expected_strength=0.7071068,
            expected_poleward_affinity=-0.5,
            excepted_eastward_affinity=0.5,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            strengths=None,
            weights=[0.5, 0.5],
            hemisphere="north",
            expected_bearing=45.0,
            expected_strength=0.7071068,
            expected_poleward_affinity=0.5,
            excepted_eastward_affinity=0.5,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            strengths=[0.5, 0.5],
            weights=None,
            hemisphere="north",
            expected_bearing=45.0,
            expected_strength=0.7071068,
            expected_poleward_affinity=0.5,
            excepted_eastward_affinity=0.5,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 360.0],
            strengths=None,
            weights=[1.0, 1.0],
            hemisphere="north",
            expected_bearing=360.0,
            expected_strength=1.0,
            expected_poleward_affinity=1.0,
            excepted_eastward_affinity=0.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            strengths=None,
            weights=[0.0, 1.0],
            hemisphere="north",
            expected_bearing=90.0,
            expected_strength=1.0,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=1.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[90.0, 270.0],
            strengths=[0.1, 0.9],
            weights=None,
            hemisphere="north",
            expected_bearing=270.0,
            expected_strength=0.8,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=-0.8,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[90.0],
            strengths=[0.0],
            weights=None,
            hemisphere="north",
            expected_bearing=0.0,
            expected_strength=0.0,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=0.0,
        ),
        # weights and strengths
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            strengths=[0.2, 0.4],
            weights=[2, 4],
            hemisphere="north",
            expected_bearing=75.9637565,
            expected_strength=0.9162457,
            expected_poleward_affinity=0.2222222,
            excepted_eastward_affinity=0.8888889,
        ),
    ],
)
def test_get_bearing_summary_stats(param: BearingSummaryStatsPytestParam) -> None:
    stats = get_bearing_summary_stats(
        bearings=param.bearings,
        strengths=param.strengths,
        weights=param.weights,
        hemisphere=param.hemisphere,
    )
    assert stats.mean_bearing == pytest.approx(param.expected_bearing)
    assert stats.mean_bearing_strength == pytest.approx(param.expected_strength)
    assert stats.poleward_affinity == pytest.approx(param.expected_poleward_affinity)
    assert stats.eastward_affinity == pytest.approx(param.excepted_eastward_affinity)
