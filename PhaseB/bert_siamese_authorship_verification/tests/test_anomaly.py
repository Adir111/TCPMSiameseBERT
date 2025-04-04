import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.isolation_forest import AnomalyDetector
import numpy as np


def test_anomaly_vector_shape():
    data = np.array([
        [0.1, 0.2, 0.3],
        [0.1, 0.25, 0.35],
        [0.8, 0.9, 0.95]
    ])
    detector = AnomalyDetector()
    scores = detector.fit_score(data)

    assert len(scores) == data.shape[0], "One score per sample expected"
    assert np.isfinite(scores).all(), "Anomaly scores contain NaNs or Infs"
