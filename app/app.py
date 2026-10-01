"""Run with `uv run streamlit run app.py` from the app directory."""

import pandas as pd
import streamlit as st

from red_wine_quality_app.model import ARTIFACT_PATH, load_models, predict_wine


# Page setup and cached model loading
st.set_page_config(page_title="Red Wine Quality", page_icon="🍷", layout="wide")


@st.cache_resource(show_spinner="Loading the wine quality models…")
def get_models(export_timestamp: int):
    return load_models()


try:
    models = get_models(ARTIFACT_PATH.stat().st_mtime_ns)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"The model could not start: {exc}")
    st.stop()

# Page title and tabs
st.title("Red Wine Quality")
st.caption("Explore the model and predict a red wine's quality score.")
selected_model = st.selectbox("Model", list(models))
model = models[selected_model]

calculations_tab, visualizations_tab, predictions_tab = st.tabs(
    ["Calculations", "Visualizations", "Predictions"]
)

with calculations_tab:
    # Evaluation metrics and confusion matrix
    st.subheader("Model evaluation")
    st.caption(
        f"{model.train_size:,} training rows · {model.test_size:,} test rows · "
        f"quality scores {model.metrics.quality_scores[0]}–{model.metrics.quality_scores[-1]}"
    )
    metrics = model.metrics
    for column, (label, value) in zip(
        st.columns(4),
        (
            ("Exact-score accuracy", f"{metrics.accuracy:.1%}"),
            ("Exact-score error", f"{metrics.error_rate:.1%}"),
            ("Within one point", f"{metrics.within_one_accuracy:.1%}"),
            ("Mean absolute error", f"{metrics.mean_absolute_error:.2f} points"),
        ),
    ):
        column.metric(label, value)

    st.write("Confusion matrix (rows: actual; columns: predicted)")
    st.dataframe(
        pd.DataFrame(
            metrics.confusion_matrix,
            index=[f"Actual {score}" for score in metrics.quality_scores],
            columns=[f"Predicted {score}" for score in metrics.quality_scores],
        )
    )
    st.caption("Exact-score error is 100% minus exact-score accuracy on held out wines.")

with visualizations_tab:
    st.subheader("Model behavior")

    # Feature importance chart
    importance = pd.DataFrame(
        {
            "Feature": model.feature_names,
            "Importance": model.feature_importances,
        }
    ).sort_values("Importance", ascending=False)
    st.write("Feature importance")
    st.bar_chart(importance, x="Feature", y="Importance", horizontal=True)

    # Actual and predicted quality score counts
    score_counts = pd.DataFrame(
        {
            "Quality score": metrics.quality_scores,
            "Actual": metrics.actual_counts,
            "Predicted": metrics.predicted_counts,
        }
    )
    st.write("Actual and predicted scores in the held out set")
    st.bar_chart(score_counts, x="Quality score", y=["Actual", "Predicted"])

with predictions_tab:
    st.subheader("Predict a wine's quality score")
    st.caption("Adjust the measurements, then select Predict quality.")

    # Feature sliders and prediction button
    with st.form("prediction_form"):
        columns = st.columns(2)
        input_features = {}
        for index, feature in enumerate(model.features):
            with columns[index % 2]:
                input_features[feature.name] = st.slider(
                    feature.name.title(),
                    min_value=feature.minimum,
                    max_value=feature.maximum,
                    value=feature.default,
                    step=feature.step,
                    format=feature.display_format,
                    key=f"feature_{index}",
                )
        submitted = st.form_submit_button("Predict quality", type="primary")

    # Run prediction when the form is submitted
    if submitted:
        try:
            st.session_state["prediction_result"] = (
                selected_model, predict_wine(model, input_features)
            )
        except ValueError as exc:
            st.error(f"Prediction could not be made: {exc}")

    # Display the prediction and model error rate
    if "prediction_result" in st.session_state:
        result_model, result = st.session_state["prediction_result"]
        if result_model == selected_model:
            st.success(f"Predicted quality score: {result.score}")
            estimate, mae, error = st.columns(3)
            estimate.metric("Estimated quality", f"{result.estimated_quality:.3f} points")
            mae.metric("Held out mean absolute error", f"{result.mean_absolute_error:.2f} points")
            error.metric("Held out exact-score error", f"{result.error_rate:.1%}")
            st.caption(
                "The estimate uses the last submitted slider values; the score is rounded. "
                "Held out errors describe the selected model's test set, so they do not change "
                "between wines. This wine's true quality is unknown."
            )
