from math import expm1

import numpy as np
import pandas as pd

from app.data import FEATURES


def predict_quality(model, values):
    """Return the rounded score and unrounded quality estimate for one wine."""
    row = pd.DataFrame([values], columns=FEATURES)
    estimate = expm1(float(model.predict(row)[0]))
    score = int(np.clip(np.rint(estimate), 3, 8))
    return score, estimate
