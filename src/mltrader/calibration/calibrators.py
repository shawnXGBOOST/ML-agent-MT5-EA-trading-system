"""
Shared calibrator classes.

IMPORTANT: this exists as its OWN module (not defined inline inside
train_final_model.py) specifically so that joblib/pickle can correctly
resolve the class when a DIFFERENT script (live_runner.py, realistic_backtest.py,
etc.) loads a saved model. If a class is defined inside a script's __main__
block, pickle records it as belonging to that script's __main__ — and any
other script trying to unpickle it will fail with
"Can't get attribute 'X' on <module '__main__'>", since ITS __main__ doesn't
have that class defined. Keeping it here means every script that does
`from calibrators import PlattCalibrator` (or even just has this file
importable on its path) can correctly load models calibrated this way.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt (sigmoid/logistic) scaling — a smooth, continuous, strictly
    monotonic calibration curve. Unlike isotonic regression, it CANNOT
    produce flat steps: fitting a 1-D logistic regression on raw score ->
    outcome is mathematically a sigmoid, which has no flat regions except
    asymptotically near 0/1. Use this when calibration_resolution_check.py
    finds a wide flat step in an isotonic calibrator — usually a sign the
    calibration sample was too small for isotonic's more flexible fit."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, raw_scores, y):
        X = np.asarray(raw_scores).reshape(-1, 1)
        self.model.fit(X, y)
        return self

    def predict(self, raw_scores):
        X = np.asarray(raw_scores).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]
