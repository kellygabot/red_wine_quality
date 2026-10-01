# Red Wine Quality app

A local Streamlit app for exploring and predicting red wine quality on the dataset's **3–8** scale. It compares three notebook-trained models: **Random Forest, XGBoost, and LightGBM**. The app loads a saved model bundle; it does not train models when you open the page. No React build or API server is needed.

## Requirements and installation

- Python **3.14 or newer**
- [uv](https://docs.astral.sh/uv/) for dependency installation and commands
- A local copy of this repository, including `dataset/winequality-red.csv` and the exported model bundle

From the repository root:

```sh
cd app
uv sync --locked
```

`uv sync --locked` creates or updates `app/.venv` using the versions in `app/uv.lock`. Run all commands below from the `app` directory.

## Folder structure

```text
red_wine_quality/
├── dataset/
│   └── winequality-red.csv                 # Training data for the notebook
└── app/
    ├── app.py                              # Streamlit page and tabs
    ├── README.md
    ├── pyproject.toml                      # Direct dependencies and Python requirement
    ├── uv.lock                             # Resolved dependency versions
    ├── src/red_wine_quality_app/
    │   ├── model.ipynb                     # Training, EDA, evaluation, and export
    │   ├── model.py                        # Bundle loader and prediction interface
    │   ├── charts.py                       # Plotly diagnostic figures
    │   └── artifacts/model_bundle.joblib   # Exported models and evaluation data
    └── tests/                              # Model and Streamlit checks
```

## Dependencies

| Purpose | Packages |
| --- | --- |
| App and interactive charts | `streamlit`, `plotly` |
| Data handling | `pandas`, `numpy` |
| Models and evaluation | `scikit-learn`, `xgboost`, `lightgbm` |
| Saved model bundle | `joblib` |
| Notebook and exploratory plots | `ipykernel`, `matplotlib`, `seaborn` |

The exact resolved versions are in `uv.lock`. Load `model_bundle.joblib` only from this project: joblib uses pickle, which can execute code from an untrusted file.

## Run the app

```sh
uv run streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). To use another port:

```sh
uv run streamlit run app.py --server.port 8502
```

Run the checks with:

```sh
uv run python -m unittest discover -s tests -v
```

## Input format

The Predictions tab accepts **11 numeric chemical measurements**, one slider per feature. The values below are the defaults and allowed slider ranges exported from the training dataset; the app does not accept a CSV upload or free-form text input.

| Feature | Slider range | Default |
| --- | ---: | ---: |
| Fixed acidity | 4.6–15.9 | 7.9 |
| Volatile acidity | 0.12–1.58 | 0.52 |
| Citric acid | 0.00–1.00 | 0.26 |
| Residual sugar | 0.9–15.5 | 2.2 |
| Chlorides | 0.012–0.611 | 0.079 |
| Free sulfur dioxide | 1–72 | 14 |
| Total sulfur dioxide | 6–289 | 38 |
| Density | 0.99007–1.00369 | 0.99675 |
| pH | 2.74–4.01 | 3.31 |
| Sulphates | 0.33–2.00 | 0.62 |
| Alcohol | 8.4–14.9 | 10.2 |

For Python callers, `predict_wine(model, input_features)` takes a mapping with **exactly** these 11 feature names and finite numeric values. The order of keys does not matter. Example:

```python
from red_wine_quality_app.model import load_models, predict_wine

model = load_models()["Random Forest"]
measurements = {feature.name: feature.default for feature in model.features}
result = predict_wine(model, measurements)
print(result.score, result.estimated_quality)
```

## Application guide

1. Choose **Random Forest**, **XGBoost**, or **LightGBM** from the model selector above the tabs. The evaluation, charts, and predictions use that selection.
2. In **Model Evaluation**, compare exact-score accuracy, accuracy within ±1 point, MAE, and weighted F1. The heatmap and classification report break down results for scores 3–8. The benchmark table compares all three models on the same held-out split.
3. In **Visualizations**, inspect the selected model's feature importance, actual versus predicted score counts, a chemical-feature box plot by quality score, and the distribution of held-out residuals (`actual − unrounded prediction`). Choose a chemical feature above the box plot. The box plot uses all dataset rows; the other evaluation charts use held-out rows.
4. In **Predictions**, adjust the sliders and click **Predict quality**. The result shows a rounded score from 3–8 and an unrounded quality estimate. The displayed held-out MAE and exact-score error belong to the selected model's test set; they are not errors measured for that individual wine.

The notebook is the source of the fitted models. To refresh the bundle, open `src/red_wine_quality_app/model.ipynb` with the app environment as its kernel, run its data, training, and evaluation cells, then run the **final export cell**. The notebook expects its working directory to be `app/src/red_wine_quality_app` so its relative path to `dataset/winequality-red.csv` resolves. The export updates `artifacts/model_bundle.joblib`; restart or rerun Streamlit if an open page still shows old results.

## Model limitations

- The models were fit on **1,599 red wines** with a single **1,279/320 train/test split**. The displayed results are held-out estimates for this dataset, not evidence of performance on other wine sources or measurement methods.
- Rare quality scores have few examples, so per-class precision and recall can be unstable or zero. Weighted F1 gives more weight to common scores.
- Training uses `log1p(quality)`; predictions are converted back to quality points. The displayed integer score is rounded and limited to **3–8**, so different measurements can produce the same rounded score. Check the unrounded estimate for smaller changes.
- A wine's true quality is unknown at prediction time. The app cannot calculate its individual error or a reliable per-wine confidence percentage. Held-out MAE and exact-score error stay constant while you change sliders for the same model.
- Feature importance describes each model's fitted behavior. It does not establish a chemical property's causal effect, and native importance values are not directly comparable across the three algorithms.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv` or Python version error | Install uv and Python 3.14+, then rerun `uv sync --locked` from `app`. |
| Import error when starting the app | Run `uv sync --locked` and use `uv run streamlit run app.py` rather than a different Python environment. |
| “Model export not found” | Confirm `src/red_wine_quality_app/artifacts/model_bundle.joblib` exists; rerun the notebook's final export cell if needed. |
| “Unsupported version or feature schema” | The bundle and `model.py` disagree. Rerun the notebook export with the current code and confirm it contains the three permitted models. |
| Notebook cannot find the CSV | Check that `dataset/winequality-red.csv` exists and the notebook kernel's working directory is `app/src/red_wine_quality_app`. |
| Prediction score seems unchanged | The estimate may have changed without crossing an integer boundary; compare **Estimated quality**. Held-out errors are fixed for each selected model. |
| Port 8501 is occupied | Start Streamlit with `--server.port 8502` or another free port. |
