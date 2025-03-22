from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def compute_dtw_distance(signal1, signal2):
    """
    Computes Dynamic Time Warping (DTW) distance between two signals.

    :param signal1: First signal (list of floats)
    :param signal2: Second signal (list of floats)
    :return: DTW distance
    """
    distance, _ = fastdtw(signal1, signal2, dist=euclidean)
    return distance
