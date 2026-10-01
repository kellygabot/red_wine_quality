"""Run with `uv run streamlit run app.py` from the app directory."""

import pandas as pd
import streamlit as st

from red_wine_quality_app.charts import (
    confusion_heatmap,
    feature_importance,
    feature_quality_box,
    residual_distribution,
    score_distribution,
)
from red_wine_quality_app.model import ARTIFACT_PATH, MODEL_NAMES, ModelBundle, load_models, predict_wine

st.set_page_config(page_title="Red Wine Quality", page_icon="🍷", layout="wide")


@st.cache_resource(show_spinner="Loading the wine quality models…")
def get_models(export_timestamp: int) -> dict[str, ModelBundle]:
    return load_models()


def render_model_evaluation(
    model: ModelBundle, models: dict[str, ModelBundle], selected_model: str
) -> None:
    metrics = model.metrics
    st.subheader("Model evaluation")
    st.caption(f"{model.train_size:,} training wines · {model.test_size:,} held-out wines · scores 3–8")
    for column, (label, value) in zip(st.columns(4), (
        ("Exact-Score Accuracy", f"{metrics.accuracy:.1%}"),
        ("Within ±1 Point Accuracy", f"{metrics.within_one_accuracy:.1%}"),
        ("Mean Absolute Error (MAE)", f"{metrics.mean_absolute_error:.2f} points"),
        ("Weighted F1-Score", f"{metrics.weighted_f1:.3f}"),
    )):
        column.metric(label, value)

    st.plotly_chart(confusion_heatmap(metrics), width="stretch")
    report_column, benchmark_column = st.columns(2)
    with report_column:
        st.write("Classification report")
        st.dataframe(pd.DataFrame(metrics.classification_report), hide_index=True, width="stretch")
    with benchmark_column:
        st.write("Held-out model benchmark")
        benchmark = pd.DataFrame([
            {"Model": name, "MAE (points)": round(bundle.metrics.mean_absolute_error, 3),
             "Accuracy (%)": round(100 * bundle.metrics.accuracy, 1),
             "Selected": name == selected_model}
            for name, bundle in models.items()
        ])
        st.dataframe(benchmark, hide_index=True, width="stretch")
    st.caption("MAE uses unrounded predictions; accuracy and F1 use scores rounded to 3–8. "
               "All models use the same held-out split.")


def render_visualizations(model: ModelBundle) -> None:
    st.subheader("Model diagnostics")
    st.plotly_chart(feature_importance(model), width="stretch")
    st.caption("Feature importance is native to the selected model and is not directly comparable across algorithms.")
    st.plotly_chart(score_distribution(model.metrics), width="stretch")
    feature = st.selectbox("Chemical feature", model.feature_names,
                           index=model.feature_names.index("alcohol"))
    st.plotly_chart(feature_quality_box(model, feature), width="stretch")
    st.caption("The feature plot uses all wines in the original dataset.")
    st.plotly_chart(residual_distribution(model), width="stretch")
    st.caption("Residuals are actual minus unrounded predicted quality on held-out wines. "
               "The plot helps inspect bias and spread; it does not establish normality.")


def render_predictions(model: ModelBundle, selected_model: str) -> None:
    st.subheader("Predict a wine's quality score")
    st.caption("Adjust the measurements, then select Predict quality.")
    with st.form("prediction_form"):
        columns = st.columns(2)
        input_features = {}
        for index, feature in enumerate(model.features):
            with columns[index % 2]:
                input_features[feature.name] = st.slider(
                    feature.name.title(), min_value=feature.minimum, max_value=feature.maximum,
                    value=feature.default, step=feature.step, format=feature.display_format,
                    key=f"feature_{index}",
                )
        submitted = st.form_submit_button("Predict quality", type="primary")

    if submitted:
        try:
            st.session_state["prediction_result"] = (
                selected_model, predict_wine(model, input_features)
            )
        except ValueError as exc:
            st.error(f"Prediction could not be made: {exc}")
    if "prediction_result" in st.session_state:
        result_model, result = st.session_state["prediction_result"]
        if result_model == selected_model:
            st.success(f"Predicted quality score: {result.score}")
            estimate, mae, error = st.columns(3)
            estimate.metric("Estimated quality", f"{result.estimated_quality:.3f} points")
            mae.metric("Held out mean absolute error", f"{result.mean_absolute_error:.2f} points")
            error.metric("Held out exact-score error", f"{result.error_rate:.1%}")
            st.caption("The estimate uses the last submitted values; the score is rounded. "
                       "Held-out errors describe the selected model's test set and stay the same between wines.")


try:
    models = get_models(ARTIFACT_PATH.stat().st_mtime_ns)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"The model could not start: {exc}")
    st.stop()

st.title("Red Wine Quality")
st.caption("Explore model performance and predict a red wine's quality score.")
selected_model = st.sidebar.selectbox("Model", MODEL_NAMES)
model = models[selected_model]
evaluation_tab, visualizations_tab, predictions_tab = st.tabs(
    ["Model Evaluation", "Visualizations", "Predictions"]
)
with evaluation_tab:
    render_model_evaluation(model, models, selected_model)
with visualizations_tab:
    render_visualizations(model)
with predictions_tab:
    render_predictions(model, selected_model)
