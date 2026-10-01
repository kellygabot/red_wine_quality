# Red wine quality app

This Streamlit app loads three regressors trained in `src/red_wine_quality_app/model.ipynb`:
Random Forest, XGBoost, and LightGBM. It does not train models on startup.

From the `app` directory:

```sh
uv sync --locked
uv run streamlit run app.py
```

To update the models, run the notebook's training cells, then its final export cell.
The resulting `src/red_wine_quality_app/artifacts/model_bundle.joblib` supplies the
models, held-out evaluation data, and data for the exploratory feature plot. The
notebook's target is `log1p(quality)`; the export converts predictions back to
quality points. Exact-score accuracy, weighted F1, and the classification report
use predictions rounded and clipped to scores 3–8. MAE and residuals use
unrounded predictions on the original quality scale. The three models share the
same held-out split.

The prediction tab shows both the rounded score and the unrounded estimate.
Held-out error metrics describe the selected model's test set; they do not
measure the error of a wine whose true quality is unknown.

To run the tests:

```sh
uv run python -m unittest discover -s tests -v
```

`src/red_wine_quality_app/model.py` loads the notebook export and serves
predictions; `charts.py` builds the Plotly figures. Only load this project's
bundle: joblib artifacts are pickle based and should not come from untrusted sources.
