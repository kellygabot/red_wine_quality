import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def evaluate_models(models, x_train, y_train):
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(x_train))
    rows = []
    for name, model in models.items():
        mae, rmse, r2 = [], [], []
        for train_index, validation_index in folds:
            fold_model = clone(model).fit(x_train.iloc[train_index], y_train[train_index])
            actual = np.expm1(y_train[validation_index])
            predicted = np.expm1(fold_model.predict(x_train.iloc[validation_index]))
            mae.append(mean_absolute_error(actual, predicted))
            rmse.append(np.sqrt(mean_squared_error(actual, predicted)))
            r2.append(r2_score(actual, predicted))
        rows.append({
            "Model": name,
            "MAE (points)": float(np.mean(mae)),
            "MAE SD": float(np.std(mae, ddof=1)),
            "RMSE (points)": float(np.mean(rmse)),
            "RMSE SD": float(np.std(rmse, ddof=1)),
            "R²": float(np.mean(r2)),
            "R² SD": float(np.std(r2, ddof=1)),
        })
    return pd.DataFrame(rows)
