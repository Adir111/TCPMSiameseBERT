import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# Todo: use https://pypi.org/project/dtaidistance/
def compute_dtw_distance(signal1, signal2):
    """
    Computes Dynamic Time Warping (DTW) distance between two signals.

    :param signal1: First signal (list of floats)
    :param signal2: Second signal (list of floats)
    :return: DTW distance
    """
    signal1 = [np.array([v]) for v in signal1]
    signal2 = [np.array([v]) for v in signal2]
    distance, _ = fastdtw(signal1, signal2, dist=euclidean)
    return distance
