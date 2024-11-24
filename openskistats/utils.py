import os
from pathlib import Path

repo_directory = Path(__file__).parent.parent.parent
data_directory = Path(__file__).parent.parent.joinpath("data")
test_data_directory = Path(__file__).parent.joinpath("tests", "data")


def get_data_directory(testing: bool = False) -> Path:
    directory = (
        test_data_directory
        if testing or "PYTEST_CURRENT_TEST" in os.environ
        else data_directory
    )
    directory.mkdir(exist_ok=True)
    return directory
