from ski_bearings.openskimap_utils import load_runs


def test_load_runs() -> None:
    runs = load_runs()
    assert len(runs) > 5
