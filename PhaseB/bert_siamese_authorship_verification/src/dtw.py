from dtaidistance import dtw


def compute_dtw_distance(signal_1, signal_2):
    """
    Computes Dynamic Time Warping (DTW) distance between two signals.

    :param signal_1: First signal (list of floats)
    :param signal_2: Second signal (list of floats)
    :return: DTW distance
    """
    distance = dtw.distance(signal_1, signal_2)
    return distance
