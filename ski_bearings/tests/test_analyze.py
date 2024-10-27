from ski_bearings.analyze import analyze_all_ski_areas, load_ski_areas_pl


def test_analyze_all_ski_areas() -> None:
    analyze_all_ski_areas()
    load_ski_areas_pl()
