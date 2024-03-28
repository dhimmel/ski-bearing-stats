import logging

import typer

import ski_bearings.utils

cli = typer.Typer()


class Commands:
    @staticmethod
    @cli.command(name="download")  # type: ignore [misc]
    def download() -> None:
        ski_bearings.utils.download_openskimap_geojsons()

    @staticmethod
    @cli.command(name="analyze")  # type: ignore [misc]
    def analyze() -> None:
        ski_bearings.utils.analyze_all_ski_areas()

    @staticmethod
    def command() -> None:
        """
        Run like `poetry run ski_bearings`
        """
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        cli()
