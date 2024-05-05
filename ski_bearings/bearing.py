from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx
import polars as pl

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
            .replace(bearing_labels, default=None)
            .alias("bin_label")
        )
        .with_row_index(name="bin_index", offset=1)
    )


@dataclass
class BearingSummaryStats:
    mean_bearing_deg: float
    mean_bearing_strength: float


def get_bearing_summary_stats(
    bearings: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> BearingSummaryStats:
    """
    https://chat.openai.com/share/a2648aee-194b-4744-8a81-648d124d17f2
    """
    # Convert bearings to radians
    bearings_rad = np.deg2rad(bearings)

    # Convert bearings to vectors and apply weights
    vectors = np.array([np.cos(bearings_rad), np.sin(bearings_rad)]) * weights

    # Sum the vectors
    vector_sum = np.sum(vectors, axis=1)

    # Calculate the strength/magnitude of the mean bearing
    mean_bearing_strength = np.linalg.norm(vector_sum) / np.sum(weights)

    # Convert the sum vector back to a bearing
    mean_bearing_rad = np.arctan2(vector_sum[1], vector_sum[0])
    mean_bearing_deg = np.rad2deg(mean_bearing_rad) % 360

    return BearingSummaryStats(mean_bearing_deg, mean_bearing_strength)
