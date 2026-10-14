# Red Wine Quality Prediction

A Streamlit application that explores red-wine measurements, compares Random Forest, XGBoost, and Support Vector Regression, and estimates quality from 11 chemical inputs.

[Open the application](https://kellygabot-red-wine-quality-appapp-a9o7rh.streamlit.app/) · [Technical documentation](https://docs.google.com/document/d/1ijbVBYwbfSq2W1cNObgz21-_Pt2_47U4JK8ExmLx1To/edit?tab=t.n7ty82ehhvvi)

## Contents

- [Installation and dependencies](#installation-and-dependencies)
- [Run the application](#run-the-application)
- [Project structure](#project-structure)
- [Data and input format](#data-and-input-format)
- [Application guide](#application-guide)
- [Reproduce the analysis](#reproduce-the-analysis)
- [Model evaluation](#model-evaluation)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Installation and dependencies

Install **Python 3.14 or newer** and [uv](https://docs.astral.sh/uv/). Download or clone the project and open a terminal in its root directory, containing `pyproject.toml` and `uv.lock`.

```sh
uv sync --locked
```

`pyproject.toml` declares the dependencies and Python requirement; `uv.lock` records the resolved environment. Initial installation requires internet access.

| Dependency | Purpose |
| --- | --- |
| Streamlit | Interface, forms, and caching |
| pandas and NumPy | Data preparation and numerical calculations |
| scikit-learn | Random Forest, SVR, scaling, splits, and metrics |
| XGBoost | Gradient-boosted regression |
| Plotly | Interactive application charts |
| Matplotlib and seaborn | Notebook plots |
| ipykernel | Notebook execution |

## Run the application

From the project root:

```sh
uv run streamlit run app/app.py
```

Open the local URL printed in the terminal. Initial startup trains all three models and performs five-fold validation. Later interactions reuse cached results in the running process. Restarting Streamlit can trigger training again. Press `Ctrl+C` in the terminal to stop.

## Project structure

```text
red_wine_quality/
├── app/
│   ├── app.py                     Streamlit interface
│   ├── data.py                    Loading, cleaning, and split
│   ├── evaluation.py              Five-fold evaluation
│   └── README.md                  Application notes
├── data/
│   ├── original/
│   │   └── winequality-red.csv
│   └── processed/
│       └── winequality-red-clean.csv
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
├── documentation/                 Technical guide and screenshots
├── paper/                         Final IMRaD DOCX and PDF
├── .python-version
├── pyproject.toml                 Dependency specification
├── uv.lock                        Locked environment
└── README.md
```

This overview omits model-training files, package markers, local environments, and editor settings. Training functions are imported from the existing `models` package.

## Data and input format

The runtime reads the comma-separated `data/original/winequality-red.csv`, which contains 11 numerical inputs and a `quality` target. Removing missing and exact duplicate rows leaves **1,359 of the original 1,599 records**. The runtime cleans the original CSV; the processed file is a reference copy.

The Prediction tab accepts **one wine through 11 sliders**:

| Input | Slider increment |
| --- | ---: |
| Fixed acidity | 0.1 |
| Volatile acidity | 0.01 |
| Citric acid | 0.01 |
| Residual sugar | 0.1 |
| Chlorides | 0.001 |
| Free sulfur dioxide | 1.0 |
| Total sulfur dioxide | 1.0 |
| Density | 0.00001 |
| pH | 0.01 |
| Sulphates | 0.01 |
| Alcohol | 0.1 |

Use the source dataset's measurement units. Slider limits come from cleaned data; defaults derive from feature medians. Quality is an output, not a form input. CSV upload and batch prediction are not implemented.

**Dataset documentation pending:** verified source URL, original publication, license, and feature-unit dictionary.

## Application guide

Choose a model using **Model for prediction**, then use the tabs:

| Tab | What to do |
| --- | --- |
| Exploration | Review original-data score counts, distributions, correlations, boxplots, and a scatter plot. |
| Model Comparison | Compare five-fold MAE, RMSE, and R², including mean and sample standard deviation. |
| Prediction | Adjust measurements and select **Predict quality**. |

The result shows a fractional **Estimated quality** and an integer **Predicted quality score**, rounded and clipped to 3–8. Submit again after changing measurements or switching models.

The displayed cross-validation MAE summarizes average model performance; it is not the measured error or confidence interval for the wine entered.

## Reproduce the analysis

Select the notebook kernel from the environment installed with `uv sync --locked`. Run all cells in this order:

1. `notebooks/eda.ipynb` — exploratory analysis of the original data.
2. `notebooks/model_training.ipynb` — preparation, training, and validation.

Confirm project imports resolve from the chosen working directory. Save executed outputs and record the code version and execution date. Rerun affected notebooks after data or code changes before updating paper tables and screenshots.

## Model evaluation

The fixed **80/20 split** uses `random_state=0`: **1,087 training rows** and **272 reserved test rows**.

- Training-only evaluation uses `KFold(n_splits=5, shuffle=True, random_state=0)`, with identical folds for all models.
- Every fold fits a fresh model. SVR's scaler learns only from that fold's fitting rows.
- Models learn `log1p(quality)`; `expm1` restores predictions before scoring.
- App prediction models fit separately on the full training partition.

| Metric | Meaning | Better direction |
| --- | --- | --- |
| MAE | Average absolute error in quality points; primary selection criterion | Lower |
| RMSE | Quality-point error with greater weight on large mistakes | Lower |
| R² | Squared-error performance relative to the evaluation-data mean | Higher |

Metrics use unrounded, unclipped estimates. The table reports fold means and sample SD (`ddof=1`). SD measures variation across folds; R² is not percentage accuracy.

The current workflow does not score the reserved test partition or automatically select one final model. Earlier versions inspected test results, so it should not be described as never used. Final selected-model test reporting remains separate work.

## Limitations

- Scores 5 and 6 dominate; rare-score performance may be less reliable.
- Quality is ordinal but modeled numerically. The log transform's benefit has not been established through a target-transform comparison.
- Configurations are fixed; no hyperparameter search is performed.
- Results do not establish performance on other wine sources or measurement procedures.
- The interface constrains slider inputs but provides no external-input API, batch prediction, or individual confidence intervals.

## Troubleshooting

| Problem | Action |
| --- | --- |
| uv is not recognized | Install uv and reopen the terminal. |
| Incompatible Python | Use Python 3.14 or newer and sync again. |
| Missing module | Run `uv sync --locked` and launch through `uv run`. |
| Dataset not found | Check `data/original/winequality-red.csv` exists. |
| Initial page takes time | Allow training and validation to finish; check terminal errors. |
| Page does not open | Confirm Streamlit is running and use its printed URL. |
| Prediction does not update | Select **Predict quality** again. |
| Results are stale after a data change | Restart Streamlit to rebuild cached results. |
| Notebook imports fail | Check the kernel, working directory, and project-root path setup. |

If the default port is occupied, stop the old instance or use:

```sh
uv run streamlit run app/app.py --server.port 8502
```
