from typing import Literal

from patito import Field, Model


class SkiAreaBearingDistributionModel(Model):  # type: ignore [misc]
    num_bins: int = Field(
        description="Number of bins in the bearing distribution.",
        # multi-column primary key / uniqueness constraint
        # https://github.com/JakobGM/patito/issues/14
        # constraints=[pl.struct("num_bins", "bin_index").is_unique()],
    )
    bin_index: int = Field(description="Index of the bearing bin starting at 1.")
    bin_center: float = Field(description="Center of the bearing bin in degrees.")
    bin_count: int = Field(description="Count of bearings in the bin.")
    bin_proportion: float = Field(description="Proportion of bearings in the bin.")


class SkiAreaModel(Model):  # type: ignore [misc]
    ski_area_id: str = Field(
        unique=True,
        description="Unique OpenSkiMap identifier for a ski area.",
        examples=["fe8efce409aa78cfa20a1e6b5dd5e32369dbe687"],
    )
    ski_area_name: str | None = Field(
        description="Name of the ski area.",
        examples=["Black Mountain"],
    )
    generated: bool
    runConvention: Literal["japan", "europe", "north_america"]
    status: Literal["operating", "abandoned", "proposed", "disused"] | None = Field(
        description="Operating status of the ski area."
    )
    location__localized__en__country: str | None = Field(
        description="Country where the ski area is located.",
        examples=["United States"],
    )
    location__localized__en__region: str | None = Field(
        description="Region where the ski area is located.",
        examples=["New Hampshire"],
    )
    location__localized__en__locality: str | None = Field(
        description="Locality where the ski area is located.",
        examples=["Jackson"],
    )
    websites: list[str] | None = Field(
        description="List of URLs for the ski area.",
        examples=["https://www.blackmt.com/"],
    )
    run_count: int = Field(
        description="Total number of runs in the ski area.",
    )
    run_count_filtered: int = Field(
        description="Number of runs in the ski area with supported geometries.",
    )
    latitude: float | None = Field(
        description="Latitude of the ski area.",
    )
    hemisphere: Literal["north", "south"] | None = Field(
        description="Hemisphere of the ski area.",
    )
    combined_vertical: float | None = Field(
        description="Total vertical drop of the ski area in meters.",
    )
    mean_bearing: float | None = Field(
        description="Mean bearing of the ski area in degrees.",
    )
    mean_bearing_strength: float | None = Field(
        description="Mean bearing strength of the ski area.",
    )
    # bearings: list[SkiAreaBearingDistributionModel] | None = Field(
    #     description="Bearing distribution of the ski area.",
    # )
