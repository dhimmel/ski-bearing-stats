from typing import Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx
import polars as pl
from osmnx.bearing import calculate_bearing
from osmnx.distance import great_circle

from ski_bearings.models import BearingStatsModel


def add_spatial_metric_columns(
    df: pl.DataFrame | pl.LazyFrame, partition_by: str | list[str]
) -> pl.LazyFrame:
    """
    Add spatial metrics to a DataFrame of geographic coordinates.
    """
    for column in ["index", "latitude", "longitude", "elevation"]:
        assert column in df
    return (
        df.lazy()
        .with_columns(
            latitude_lag=pl.col("latitude")
            .shift(1)
            .over(partition_by, order_by="index"),
            longitude_lag=pl.col("longitude")
            .shift(1)
            .over(partition_by, order_by="index"),
            elevation_lag=pl.col("elevation")
            .shift(1)
            .over(partition_by, order_by="index"),
        )
        .with_columns(
            elevation_delta=pl.col("elevation_lag") - pl.col("elevation"),
            _coord_struct=pl.struct(
                "latitude_lag", "longitude_lag", "latitude", "longitude"
            ),
        )
        .with_columns(
            vertical=pl.col("elevation_delta").clip(lower_bound=0),
            distance=pl.col("_coord_struct").map_batches(
                lambda x: great_circle(
                    lat1=x.struct.field("latitude_lag"),
                    lon1=x.struct.field("longitude_lag"),
                    lat2=x.struct.field("latitude"),
                    lon2=x.struct.field("longitude"),
                )
            ),
            bearing=pl.col("_coord_struct").map_batches(
                lambda x: calculate_bearing(
                    lat1=x.struct.field("latitude_lag"),
                    lon1=x.struct.field("longitude_lag"),
                    lat2=x.struct.field("latitude"),
                    lon2=x.struct.field("longitude"),
                )
            ),
        )
        .drop("latitude_lag", "longitude_lag", "elevation_lag", "_coord_struct")
    )


bearing_labels = {
    0.0: "N",
    11.25: "NbE",
    22.5: "NNE",
    33.75: "NEbN",
    45.0: "NE",
    56.25: "NEbE",
    67.5: "ENE",
    78.75: "EbN",
    90.0: "E",
    101.25: "EbS",
    112.5: "ESE",
    123.75: "SEbE",
    135.0: "SE",
    146.25: "SEbS",
    157.5: "SSE",
    168.75: "SbE",
    180.0: "S",
    191.25: "SbW",
    202.5: "SSW",
    213.75: "SWbS",
    225.0: "SW",
    236.25: "SWbW",
    247.5: "WSW",
    258.75: "WbS",
    270.0: "W",
    281.25: "WbN",
    292.5: "WNW",
    303.75: "NWbW",
    315.0: "NW",
    326.25: "NWbN",
    337.5: "NNW",
    348.75: "NbW",
}
"""Bearing labels for 32-wind compass rose."""


def get_bearing_distributions_df(graph: nx.MultiDiGraph) -> pl.DataFrame:
    """
    Get the bearing distribution of a graph as a DataFrame.
    """
    bins = [2, 4, 8, 32]
    return pl.concat(
        [get_bearing_distribution_df(graph, num_bins=num_bins) for num_bins in bins],
        how="vertical",
    )


def get_bearing_distribution_df(graph: nx.MultiDiGraph, num_bins: int) -> pl.DataFrame:
    """
    Get the bearing distribution of a graph as a DataFrame.
    """
    from ski_bearings.osmnx_utils import suppress_user_warning

    with suppress_user_warning():
        bin_counts, bin_centers = osmnx.bearing._bearings_distribution(
            graph,
            num_bins=num_bins,
            min_length=0,
            weight="vertical",
        )
    # polars make dataframe from bin_counts, and bin_centers
    return (
        pl.DataFrame(
            {
                "bin_center": bin_centers,
                "bin_count": bin_counts,
            }
        )
        .with_columns(
            bin_proportion=pl.col("bin_count") / pl.sum("bin_count").over(pl.lit(True))
        )
        .with_columns(pl.lit(num_bins).alias("num_bins"))
        .with_columns(
            pl.col("bin_center")
            .replace_strict(bearing_labels, default=None)
            .alias("bin_label")
        )
        .with_row_index(name="bin_index", offset=1)
    )


def get_bearing_summary_stats(
    bearings: list[float] | npt.NDArray[np.float64],
    net_magnitudes: list[float] | npt.NDArray[np.float64] | None = None,
    cum_magnitudes: list[float] | npt.NDArray[np.float64] | None = None,
    hemisphere: Literal["north", "south"] | None = None,
) -> BearingStatsModel:
    """
    Compute the mean bearing (i.e. average direction, mean angle)
    and mean bearing strength (i.e. resultant vector length, concentration, magnitude) from a set of bearings,
    with optional strengths and weights.

    bearings:
        An array or list of bearing angles in degrees. These represent directions, headings, or orientations.
    net_magnitudes:
        An array or list of weights (importance factors, influence coefficients, scaling factors) applied to each bearing.
        If None, all weights are assumed to be 1.
        These represent external weighting factors, priorities, or significance levels assigned to each bearing.
    cum_magnitudes:
        An array or list of combined verticals of the each bearing.
        If None, all combined verticals are assumed to be 1.
        These represent the total verticals of all the original group of segments attributing to this bearing.
    hemisphere:
        The hemisphere in which the bearings are located used to calculate poleward affinity.
        If None, poleward affinity is not calculated.

    Notes:
    - The function computes the mean direction by converting bearings to unit vectors (directional cosines and sines),
      scaling them by their strengths (magnitudes), and applying weights during summation.
    - The mean bearing strength is calculated as the magnitude (length, norm) of the resultant vector
      divided by the sum of weighted strengths, providing a normalized measure (ranging from 0 to 1)
      of how tightly the bearings are clustered around the mean direction.
    - The function handles edge cases where the sum of weights is zero,
      returning a mean bearing strength of 0.0 in such scenarios.

    Development chats:
    - https://chatgpt.com/share/6718521f-6768-8011-aed4-db345efb68b7
    - https://chatgpt.com/share/a2648aee-194b-4744-8a81-648d124d17f2
    """
    if net_magnitudes is None:
        net_magnitudes = np.ones_like(bearings, dtype=np.float64)
    if cum_magnitudes is None:
        cum_magnitudes = np.ones_like(bearings, dtype=np.float64)
    bearings = np.array(bearings, dtype=np.float64)
    net_magnitudes = np.array(net_magnitudes, dtype=np.float64)
    cum_magnitudes = np.array(cum_magnitudes, dtype=np.float64)
    assert bearings.shape == net_magnitudes.shape == cum_magnitudes.shape

    # Sum all vectors in their complex number form using weights and bearings
    total_complex = sum(net_magnitudes * np.exp(1j * np.deg2rad(bearings)))
    # Convert the result back to polar coordinates
    cum_magnitude = sum(cum_magnitudes)
    net_magnitude = np.abs(total_complex)
    alignment = net_magnitude / cum_magnitude if cum_magnitude > 1e-10 else 0.0
    mean_bearing_rad = np.angle(total_complex) if net_magnitude > 1e-10 else 0.0
    mean_bearing_deg = np.rad2deg(np.round(mean_bearing_rad, 10)) % 360

    if hemisphere == "north":
        # Northern Hemisphere: poleward is 0 degrees
        poleward_affinity = alignment * np.cos(mean_bearing_rad)
    elif hemisphere == "south":
        # Southern Hemisphere: poleward is 180 degrees
        poleward_affinity = -alignment * np.cos(mean_bearing_rad)
    else:
        poleward_affinity = None
    eastward_affinity = alignment * np.sin(mean_bearing_rad)

    return BearingStatsModel(
        bearing_mean=round(mean_bearing_deg, 7),
        bearing_alignment=round(alignment, 7),
        # plus zero to avoid -0.0 <https://stackoverflow.com/a/74383961/4651668>
        poleward_affinity=(
            round(poleward_affinity + 0, 7) if poleward_affinity is not None else None
        ),
        eastward_affinity=round(eastward_affinity + 0, 7),
        bearing_magnitude_net=round(net_magnitude, 7),
        bearing_magnitude_cum=round(cum_magnitude, 7),
    )
