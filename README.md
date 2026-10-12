# Red Wine Quality

A small project for exploring the red wine quality dataset and comparing three prediction models: Random Forest, XGBoost, and Support Vector Regression. The two notebooks explain the analysis; a Streamlit app shows the plots, compares model results, and lets you estimate quality from 11 chemical measurements.

## Quick start

You need Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). Run these commands from the repository root:

```sh
uv sync --locked
uv run streamlit run app/app.py
```

Open the local address printed by Streamlit. The first page load trains all three models. Streamlit caches them in memory for later interactions, so no saved model file is required.

## Project layout

```text
data/
  original/winequality-red.csv       Source dataset
  processed/winequality-red-clean.csv  Cleaned reference copy
notebooks/
  01_eda.ipynb                      Exploratory plots of the original data
  02_model_training.ipynb           Cleaning, training, and model comparison
models/
  random_forest.py                  Random Forest training function
  xgboost.py                        XGBoost training function
  support_vector_regression.py      Support Vector Regression training function
app/
  app.py                            Streamlit interface
  data.py                           Shared data loading, cleaning, and split
  evaluation.py                     Shared model metrics
pyproject.toml                       Dependencies and Python requirement
uv.lock                             Locked dependency versions
```

## Notebooks

Open the notebooks in order, using the Python environment created by `uv sync` as the kernel. Run them from the repository root or the `notebooks/` directory; both locate the project files from either location.

1. `01_eda.ipynb` examines the original 1,599 wines with quality counts, feature distributions, correlations, box plots, and a scatter plot.
2. `02_model_training.ipynb` loads the same cleaned data and split used by the app, calls the three functions in `models/`, and reports five-fold cross-validation results on the training rows.

## Data and predictions

The shared loader drops rows with missing values and duplicate rows, leaving 1,359 wines. It uses 1,087 for training and reserves 272 as a held-out partition (`test_size=0.2`, `random_state=0`). The training rows are evaluated with the same five shuffled folds for all three models (`random_state=0`). Each fold starts with a fresh copy of the model, so SVR's scaler is fitted only on that fold's training rows. Each model also fits on the full training partition for app predictions. Each model learns `log1p(quality)`; validation predictions are converted back to quality points with `expm1` before scoring.

The app has three tabs: **Exploration** for the notebook's EDA plots, **Model Comparison** for cross-validation MAE, RMSE, and R² (mean and sample standard deviation), and **Prediction** for the 11 measurement sliders. All three metrics use unrounded, unclipped estimates in original quality points. Lower MAE and RMSE are better; higher R² is better. R² is not percentage accuracy. The rounded 3–8 score is only for prediction display. The MAE shown with a prediction describes average validation error, not the error of that particular wine. The reserved test partition is not scored in this workflow; earlier versions did inspect its results, so it should not be described as never used.

The models are trained on this dataset only, so their reported results do not establish accuracy for wines from other sources or measurement methods.
