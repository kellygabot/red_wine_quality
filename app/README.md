# Red wine quality app

The Streamlit app displays four regressors trained in `src/red_wine_quality_app/model.ipynb`:
CatBoost, XGBoost, LightGBM, and Random Forest. The app loads their exported
bundle; it does not train models on startup. No React build or API server is needed.

From the `app` directory:

```sh
uv sync --locked
uv run streamlit run app.py
```

To update the models, run the training cells in `model.ipynb`, then run its final
export cell. Commit the resulting `src/red_wine_quality_app/artifacts/model_bundle.joblib`
alongside code and `uv.lock`. The notebook's target is `log1p(quality)`; the export
cell converts held out predictions back to the original quality scale before
calculating displayed metrics. Predictions are rounded and limited to quality
scores 3–8. Mean absolute error is in quality points and exact-score error is the
percentage of held out wines assigned the wrong integer score. These are model
evaluation metrics, not an observed error for an individual wine. The prediction
tab also shows the unrounded quality estimate, which can change even when two
inputs round to the same score. Held out metrics change only when the selected
model changes.

To run the tests:

```sh
uv run python -m unittest discover -s tests -v
```

`src/red_wine_quality_app/model.py` defines the app's loading and prediction
interface. If you change the notebook's feature schema or target transformation,
update that interface and rerun the export cell. Load the bundle only from this
project: joblib artifacts are pickle based and should not be loaded from untrusted sources.
