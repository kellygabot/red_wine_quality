"""Run from the repository root: uv run streamlit run app/app.py."""

from decimal import Decimal
from math import expm1
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import FEATURES, load_original_data, load_training_data
from app.evaluation import evaluate_models
from models.random_forest import train_random_forest
from models.support_vector_regression import train_support_vector_regression
from models.xgboost import train_xgboost

STEPS = (0.1, 0.01, 0.01, 0.1, 0.001, 1.0, 1.0, 0.00001, 0.01, 0.01, 0.1)


@st.cache_resource(show_spinner="Training the three wine quality models…")
def get_results():
    clean, x_train, x_test, y_train, y_test = load_training_data()
    models = {
        "Random Forest": train_random_forest(x_train, y_train),
        "XGBoost": train_xgboost(x_train, y_train),
        "Support Vector Regression": train_support_vector_regression(x_train, y_train),
    }
    return clean, x_train, x_test, models, evaluate_models(models, x_train, y_train)


def render_eda(original):
    st.subheader("Explore the original data")
    st.caption(f"{len(original):,} wines · {len(FEATURES)} chemical measurements")
    counts = original["quality"].value_counts().sort_index().rename_axis("quality").reset_index(name="wines")
    st.plotly_chart(px.bar(counts, x="quality", y="wines", title="Wine quality score distribution"), width="stretch")
    distributions = original[list(FEATURES)].melt(var_name="feature", value_name="value")
    histogram = px.histogram(distributions, x="value", facet_col="feature", facet_col_wrap=3,
                             nbins=30, title="Input feature distributions", height=1050)
    histogram.update_xaxes(matches=None)
    st.plotly_chart(histogram, width="stretch")
    correlations = original[[*FEATURES, "quality"]].corr()
    st.plotly_chart(px.imshow(correlations, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                             title="Feature and quality correlations"), width="stretch")
    top = correlations["quality"].drop("quality").abs().nlargest(3).index
    box_data = original.melt(id_vars="quality", value_vars=top,
                             var_name="feature", value_name="value")
    boxes = px.box(box_data, x="quality", y="value", facet_col="feature",
                   title="Feature values by quality score")
    boxes.update_yaxes(matches=None)
    st.plotly_chart(boxes, width="stretch")
    sample = original.sample(n=min(800, len(original)), random_state=42)
    st.plotly_chart(px.scatter(sample, x="alcohol", y="volatile acidity", color="quality",
                               title="Alcohol and volatile acidity by quality score"), width="stretch")


def render_comparison(clean, x_train, x_test, comparison):
    st.subheader("Five-fold cross-validation")
    st.caption(f"{len(clean):,} cleaned rows · {len(x_train):,} training · {len(x_test):,} held out")
    st.dataframe(comparison.style.format({"MAE (points)": "{:.3f}",
                                          "MAE SD": "{:.3f}",
                                          "RMSE (points)": "{:.3f}",
                                          "RMSE SD": "{:.3f}",
                                          "R²": "{:.3f}",
                                          "R² SD": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("Mean and sample standard deviation over the same five training folds. "
               "All metrics use unrounded quality estimates. "
               "Lower MAE and RMSE are better; higher R² is better. R² is not percentage accuracy.")


def render_prediction(clean, selected, model, comparison):
    st.subheader("Predict a wine's quality score")
    values = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, (name, step) in enumerate(zip(FEATURES, STEPS)):
            decimals = max(0, -Decimal(str(step)).as_tuple().exponent)
            minimum = round(float(clean[name].min()), decimals)
            maximum = round(float(clean[name].max()), decimals)
            median = float(clean[name].median())
            default = round(minimum + round((median - minimum) / step) * step, decimals)
            with columns[index % 2]:
                values[name] = st.slider(name.title(), min_value=minimum, max_value=maximum,
                                         value=max(minimum, min(maximum, default)), step=step,
                                         format=f"%.{decimals}f")
        submitted = st.form_submit_button("Predict quality", type="primary")
    if submitted:
        row = pd.DataFrame([values], columns=FEATURES)
        estimate = expm1(float(model.predict(row)[0]))
        score = int(np.clip(np.rint(estimate), 3, 8))
        st.session_state["prediction"] = (selected, score, estimate)
    result = st.session_state.get("prediction")
    if result and result[0] == selected:
        st.success(f"Predicted quality score: {result[1]}")
        st.metric("Estimated quality", f"{result[2]:.3f} points")
        mae = comparison.loc[comparison["Model"] == selected, "MAE (points)"].iloc[0]
        st.caption(f"Mean cross-validation MAE for {selected}: {mae:.2f} points")


def main():
    st.set_page_config(page_title="Red Wine Quality", page_icon="🍷", layout="wide")
    st.title("Red Wine Quality")
    st.caption("Explore the original data, compare trained models, and predict a red wine's quality score.")
    clean, x_train, x_test, models, comparison = get_results()
    selected = st.selectbox("Model for prediction", tuple(models))
    eda_tab, comparison_tab, prediction_tab = st.tabs(["Exploration", "Model Comparison", "Prediction"])
    with eda_tab:
        render_eda(load_original_data())
    with comparison_tab:
        render_comparison(clean, x_train, x_test, comparison)
    with prediction_tab:
        render_prediction(clean, selected, models[selected], comparison)


if __name__ == "__main__":
    main()
