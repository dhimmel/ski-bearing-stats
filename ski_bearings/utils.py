import os
from pathlib import Path

data_directory = Path(__file__).parent.parent.joinpath("data")
test_data_directory = Path(__file__).parent.joinpath("tests", "data")


def get_data_directory(testing: bool = False) -> Path:
    if testing or "PYTEST_CURRENT_TEST" in os.environ:
        return test_data_directory
    return data_directory
