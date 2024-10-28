import itertools
from dataclasses import dataclass
from typing import Literal

import polars as pl
import pytest

from ski_bearings.analyze import (
    aggregate_ski_areas_pl,
    analyze_all_ski_areas,
)
from ski_bearings.bearing import get_bearing_summary_stats
from ski_bearings.openskimap_utils import get_ski_area_to_runs, load_runs
from ski_bearings.osmnx_utils import create_networkx_with_metadata


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
            expected_strength=0.3535534,
            expected_poleward_affinity=0.25,
            excepted_eastward_affinity=0.25,
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
            expected_bearing=0.0,
            expected_strength=0.0,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=0.0,
        ),  # should cancel each other out
        BearingSummaryStatsPytestParam(
            bearings=[90.0],
            strengths=[0.0],
            weights=None,
            hemisphere="north",
            expected_bearing=90.0,
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
            expected_bearing=63.4349488,
            expected_strength=0.2236068,
            expected_poleward_affinity=0.1,
            excepted_eastward_affinity=0.2,
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


def test_get_bearing_summary_stats_repeated_aggregation() -> None:
    """
    https://github.com/dhimmel/ski-bearing-stats/issues/1
    """
    # aggregate all runs at once
    all_runs = load_runs()
    # we cannot create networkx graph directly from all runs because get_ski_area_to_runs performs some filtering
    ski_area_to_runs = get_ski_area_to_runs(all_runs)
    all_runs_filtered = list(itertools.chain.from_iterable(ski_area_to_runs.values()))
    combined_graph = create_networkx_with_metadata(
        all_runs_filtered, ski_area_metadata={}
    )
    single_pass = combined_graph.graph
    # aggregate runs by ski area and then aggregate ski areas
    analyze_all_ski_areas()
    # group by hemisphere to avoid polars
    # ComputeError: at least one key is required in a group_by operation
    hemisphere_pl = aggregate_ski_areas_pl(
        group_by=["hemisphere"],
        # on test data, the default .filter(pl.lit(True)) gave a polars error:
        # ShapeError: filter's length: 1 differs from that of the series: 2
        ski_area_filters=[pl.col("hemisphere").is_not_null()],
    )
    double_pass = hemisphere_pl.row(by_predicate=pl.lit(True), named=True)
    for key in [
        "run_count",
        "run_count_filtered",
        "mean_bearing",
        "mean_bearing_strength",
        "poleward_affinity",
        "eastward_affinity",
        "vector_magnitude",
    ]:
        assert single_pass[key] == pytest.approx(
            double_pass[key]
        ), f"value mismatch for {key}"
