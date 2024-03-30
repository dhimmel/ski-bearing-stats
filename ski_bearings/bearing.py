import numpy as np
import numpy.typing as npt


def get_mean_bearing(
    bearings: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> tuple[float, float]:
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

    return mean_bearing_deg, mean_bearing_strength
