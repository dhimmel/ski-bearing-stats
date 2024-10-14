import logging

import typer

from ski_bearings.analyze import analyze_all_ski_areas
from ski_bearings.openskimap_utils import download_openskimap_geojsons

cli = typer.Typer(pretty_exceptions_show_locals=False)


class Commands:
    @staticmethod
    @cli.command(name="download")  # type: ignore [misc]
    def download() -> None:
        """Download latest OpenSkiMap source data."""
        download_openskimap_geojsons()

    @staticmethod
    @cli.command(name="analyze")  # type: ignore [misc]
    def analyze() -> None:
        """Extract ski area metadata and metrics."""
        analyze_all_ski_areas()

    @staticmethod
    def command() -> None:
        """
        Run like `poetry run ski_bearings`
        """
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        cli()
