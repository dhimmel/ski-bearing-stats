from dataclasses import dataclass
from typing import Literal

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
    mean_bearing: float
    """The mean bearing in degrees, calculated from the weighted and strength-scaled vectors."""
    mean_bearing_strength: float
    """The mean bearing strength (normalized magnitude, mean resultant length), representing the concentration, consistency, or dispersion of the bearings."""
    poleward_affinity: float | None
    """The poleward affinity, representing the tendency of bearings to cluster towards the neatest pole (1.0) or equator (-1.0)."""
    eastward_affinity: float
    """The eastern affinity, representing the tendency of bearings to cluster towards the east (1.0) or west (-1.0)."""
    vector_magnitude: float
    """The magnitude (length, norm) of the resultant vector, representing the overall sum of the bearings."""


def get_bearing_summary_stats(
    bearings: list[float] | npt.NDArray[np.float64],
    strengths: list[float] | npt.NDArray[np.float64] | None = None,
    weights: list[float] | npt.NDArray[np.float64] | None = None,
    hemisphere: Literal["north", "south"] | None = None,
) -> BearingSummaryStats:
    """
    Compute the mean bearing (i.e. average direction, mean angle)
    and mean bearing strength (i.e. resultant vector length, concentration, magnitude) from a set of bearings,
    with optional strengths and weights.

    bearings:
        An array or list of bearing angles in degrees. These represent directions, headings, or orientations.
    strengths:
        An array or list of strengths (magnitudes, amplitudes, reliabilities) corresponding to each bearing.
        If None, all strengths are assumed to be 1.
        These represent the inherent magnitude, confidence, or reliability of each bearing.
    weights:
        An array or list of weights (importance factors, influence coefficients, scaling factors) applied to each bearing.
        If None, all weights are assumed to be 1.
        These represent external weighting factors, priorities, or significance levels assigned to each bearing.
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
    if weights is None:
        weights = np.ones_like(bearings, dtype=np.float64)
    if strengths is None:
        strengths = np.ones_like(bearings, dtype=np.float64)

    bearings = np.array(bearings, dtype=np.float64)
    strengths = np.array(strengths, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    assert bearings.shape == strengths.shape == weights.shape

    # Scale vectors by strengths
    bearings_rad = np.deg2rad(bearings)
    # Sum all vectors in their complex number form using weights and bearings
    total_complex = sum(weights * np.exp(1j * bearings_rad))
    # Convert the result back to polar coordinates
    vector_magnitude = np.abs(total_complex)

    if vector_magnitude < 1e-10:
        mean_bearing_rad = 0.0
        mean_bearing_strength = 0.0
    else:
        mean_bearing_rad = np.angle(total_complex)
        mean_bearing_strength = vector_magnitude / sum(
            np.divide(
                weights, strengths, out=np.zeros_like(weights), where=strengths != 0
            )
        )

    mean_bearing_deg = np.rad2deg(mean_bearing_rad) % 360

    if hemisphere == "north":
        # Northern Hemisphere: poleward is 0 degrees
        poleward_affinity = mean_bearing_strength * np.cos(mean_bearing_rad)
    elif hemisphere == "south":
        # Southern Hemisphere: poleward is 180 degrees
        poleward_affinity = -mean_bearing_strength * np.cos(mean_bearing_rad)
    else:
        poleward_affinity = None
    eastward_affinity = mean_bearing_strength * np.sin(mean_bearing_rad)

    return BearingSummaryStats(
        mean_bearing=round(mean_bearing_deg, 7),
        mean_bearing_strength=round(mean_bearing_strength, 7),
        # plus zero to avoid -0.0 <https://stackoverflow.com/a/74383961/4651668>
        poleward_affinity=(
            round(poleward_affinity + 0, 7) if poleward_affinity is not None else None
        ),
        eastward_affinity=round(eastward_affinity + 0, 7),
        vector_magnitude=round(vector_magnitude, 7),
    )
