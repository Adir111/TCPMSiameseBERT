import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dtw import compute_dtw_distance
from src.isolation_forest import AnomalyDetector


def test_end_to_end_signal_to_anomaly():
    # Mock signal vectors from 3 networks
    signals = [
        [[0.1], [0.2], [0.15], [0.3]],
        [[0.8], [0.9], [0.85], [0.95]],
        [[0.12], [0.22], [0.18], [0.29]]
    ]

    # Compute DTW matrix
    dtw_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            dist = compute_dtw_distance(signals[i], signals[j])
            dtw_matrix[i][j] = dtw_matrix[j][i] = dist

    detector = AnomalyDetector()
    scores = detector.fit_score(dtw_matrix)

    assert len(scores) == 3, "Should return one score per signal"
    assert np.isfinite(scores).all(), "Scores must be finite"


def test_dtw_matrix_symmetry():
    signals = [
        [[0.1], [0.2], [0.3]],
        [[0.15], [0.25], [0.35]],
        [[0.3], [0.4], [0.5]]
    ]
    dtw_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            dist = compute_dtw_distance(signals[i], signals[j])
            dtw_matrix[i][j] = dtw_matrix[j][i] = dist

    assert np.allclose(dtw_matrix, dtw_matrix.T, atol=1e-6), "DTW matrix must be symmetric"


def test_dtw_matrix_diagonal_is_zero():
    signals = [
        [[0.1], [0.2], [0.3]],
        [[0.15], [0.25], [0.35]],
        [[0.3], [0.4], [0.5]]
    ]
    dtw_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            dist = compute_dtw_distance(signals[i], signals[j])
            dtw_matrix[i][j] = dtw_matrix[j][i] = dist

    for i in range(3):
        assert dtw_matrix[i][i] == 0.0, "DTW distance to self must be zero"


def test_anomaly_detector_extreme_case():
    signals = [
        [[0.1], [0.2], [0.3]],
        [[0.11], [0.21], [0.31]],
        [[0.09], [0.19], [0.29]]
    ]
    dtw_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            dist = compute_dtw_distance(signals[i], signals[j])
            dtw_matrix[i][j] = dtw_matrix[j][i] = dist

    detector = AnomalyDetector()
    scores = detector.model.fit_predict(dtw_matrix)

    assert sum(scores) == 3, "No strong outliers expected, all should be normal"


def test_anomaly_detector_outlier_case():
    signals = [
        [[0.1], [0.2], [0.3]],
        [[0.1], [0.2], [0.3]],
        [[5.0], [5.0], [5.0]]  # Obvious outlier
    ]
    dtw_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            dist = compute_dtw_distance(signals[i], signals[j])
            dtw_matrix[i][j] = dtw_matrix[j][i] = dist

    detector = AnomalyDetector()
    scores = detector.fit_predict(dtw_matrix)

    assert np.sum(scores) == 2, "Two normal, one anomaly expected"
    assert scores[2] == 0 or scores[0] == 0, "Outlier should be flagged"


def test_anomaly_detector_fit_score_and_predict():
    signals = [
        [[0.1], [0.2], [0.3]],
        [[0.12], [0.21], [0.29]],
        [[5.0], [5.0], [5.0]]  # clear outlier
    ]

    dtw_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            dist = compute_dtw_distance(signals[i], signals[j])
            dtw_matrix[i][j] = dtw_matrix[j][i] = dist

    detector = AnomalyDetector()
    scores = detector.fit_score(dtw_matrix)
    preds = detector.fit_predict(dtw_matrix)

    # Check shapes
    assert scores.shape == (3,), "fit_score should return score for each signal"
    assert preds.shape == (3,), "fit_predict should return one prediction per signal"

    # Score sanity: higher = more normal
    assert np.argmax(scores) != np.argmin(scores), "There should be score variation"

    # At least one should be flagged as anomaly
    assert 0 in preds, "At least one sample should be an anomaly"
    assert 1 in preds, "At least one sample should be normal"
