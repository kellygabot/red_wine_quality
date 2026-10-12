import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

SCORES = tuple(range(3, 9))


def evaluate_models(models, x_test, y_test):
    actual = np.rint(np.expm1(y_test)).astype(int)
    rows = []
    for name, model in models.items():
        predicted = np.expm1(model.predict(x_test))
        rounded = np.clip(np.rint(predicted), SCORES[0], SCORES[-1]).astype(int)
        rows.append({
            "Model": name,
            "MAE (points)": float(mean_absolute_error(actual, predicted)),
            "Exact-score accuracy": float(accuracy_score(actual, rounded)),
            "Within ±1 point": float(np.mean(np.abs(actual - rounded) <= 1)),
        })
    return pd.DataFrame(rows)
