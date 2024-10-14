import logging

import typer

from ski_bearings.analyze import analyze_all_ski_areas, load_ski_areas_pl
from ski_bearings.models import SkiAreaModel
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
    @cli.command(name="validate")  # type: ignore [misc]
    def validate() -> None:
        """Validate ski area metadata and metrics."""
        ski_areas = load_ski_areas_pl()
        SkiAreaModel.validate(ski_areas, allow_superfluous_columns=True)
        logging.info("SkiAreaModel.validate success.")

    @staticmethod
    def command() -> None:
        """
        Run like `poetry run ski_bearings`
        """
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        cli()
