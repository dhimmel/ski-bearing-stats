from ski_bearings.analyze import (
    analyze_all_ski_areas_polars,
    load_ski_areas_pl,
    ski_rose_the_world,
)


def test_analysis_pipeline() -> None:
    analyze_all_ski_areas_polars()
    load_ski_areas_pl()
    ski_rose_the_world(min_combined_vertical=0)
