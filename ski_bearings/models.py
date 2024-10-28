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
    bin_label: str | None = Field(
        description="Human readable short label of the bearing bin.",
        examples=["N", "NE", "NEbE", "ENE"],
    )


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
    country: str | None = Field(
        description="Country where the ski area is located.",
        examples=["United States"],
    )
    region: str | None = Field(
        description="Region/subdivision/province/state where the ski area is located.",
        examples=["New Hampshire"],
    )
    locality: str | None = Field(
        description="Locality/town/city where the ski area is located.",
        examples=["Jackson"],
    )
    country_code: str | None = Field(
        description="ISO 3166-1 alpha-2 two-letter country code.",
        examples=["US", "FR"],
    )
    country_subdiv_code: str | None = Field(
        description="ISO 3166-2 code for principal subdivision (e.g., province or state) of the country.",
        examples=["US-NH", "JP-01", "FR-ARA"],
    )
    # Validation fails on an empty list https://github.com/JakobGM/patito/issues/103
    websites: list[str | None] = Field(
        description="List of URLs for the ski area.",
        examples=["https://www.blackmt.com/"],
    )
    run_count: int = Field(
        description="Total number of runs in the ski area.",
        ge=0,
    )
    run_count_filtered: int = Field(
        description="Number of runs in the ski area with supported geometries.",
        ge=0,
    )
    latitude: float | None = Field(
        description="Latitude of the ski area in decimal degrees.",
        ge=-90,
        le=90,
    )
    longitude: float | None = Field(
        description="Longitude of the ski area in decimal degrees.",
        ge=-180,
        le=180,
    )
    hemisphere: Literal["north", "south"] | None = Field(
        description="Hemisphere of the ski area.",
    )
    combined_vertical: float | None = Field(
        description="Total vertical drop of the ski area in meters.",
    )
    mean_bearing: float | None = Field(
        description="Mean bearing of the ski area in degrees.",
        ge=0,
        lt=360,
    )
    mean_bearing_strength: float | None = Field(
        description="Mean bearing strength of the ski area.",
        ge=0,
        le=1,
    )
    poleward_affinity: float | None = Field(
        description="Poleward affinity of the ski area. "
        "Positive values indicate bearings cluster towards the pole of the hemisphere in which the ski area is located. "
        "Negative values indicate bearings cluster towards the equator.",
        ge=-1,
        le=1,
    )
    eastward_affinity: float | None = Field(
        description="Eastern affinity of the ski area. "
        "Positive values indicate bearings cluster towards the east. "
        "Negative values indicate bearings cluster towards the west.",
        ge=-1,
        le=1,
    )
    vector_magnitude: float | None = Field(
        description="Weighted vector summation of all segments of the ski area. "
        "Used to calculate the mean bearing and mean bearing strength.",
    )
    # https://github.com/JakobGM/patito/issues/104
    # bearings: list[SkiAreaBearingDistributionModel] = Field(
    #     description="Bearing distribution of the ski area.",
    # )
