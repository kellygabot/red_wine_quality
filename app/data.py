from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = Path(__file__).resolve().parents[1]
DATA = PATH / "data" / "original" / "winequality-red.csv"
FEATURES = (
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
)


def load_original_data() -> pd.DataFrame:
    return pd.read_csv(DATA)


def clean_wine_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna().drop_duplicates().reset_index(drop=True)


def load_training_data():
    clean = clean_wine_data(load_original_data())
    x = clean.loc[:, FEATURES]
    y = np.log1p(clean["quality"].to_numpy())
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    return clean, x_train, x_test, y_train, y_test
