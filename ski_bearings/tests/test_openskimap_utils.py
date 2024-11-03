from ski_bearings.openskimap_utils import load_runs_from_download


def test_load_runs() -> None:
    runs = load_runs_from_download()
    assert len(runs) > 5
