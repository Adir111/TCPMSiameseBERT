import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dtw import compute_dtw_distance


def test_dtw_identical_signals():
    s = [[0.1], [0.2], [0.3], [0.4]]
    assert compute_dtw_distance(s, s) == 0.0, "Identical signals should have zero distance"


def test_dtw_different_length():
    s1 = [[0.1], [0.2], [0.3]]
    s2 = [[0.1], [0.2], [0.3], [0.4], [0.5]]
    dist = compute_dtw_distance(s1, s2)
    assert dist > 0, "DTW distance should be positive for different-length signals"


def test_dtw_symmetry():
    s1 = [[0.1], [0.2], [0.3]]
    s2 = [[0.3], [0.2], [0.1]]
    d1 = compute_dtw_distance(s1, s2)
    d2 = compute_dtw_distance(s2, s1)
    assert abs(d1 - d2) < 1e-6, "DTW should be symmetric"


def test_dtw_output_type():
    s1 = [[0.0], [1.0], [2.0]]
    s2 = [[1.0], [2.0], [3.0]]
    dist = compute_dtw_distance(s1, s2)
    assert isinstance(dist, float), "DTW result should be a float"
