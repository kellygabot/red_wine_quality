"""Plotly figures for the notebook-exported model diagnostics."""

import plotly.graph_objects as go

from .model import EvaluationMetrics, ModelBundle


def confusion_heatmap(metrics: EvaluationMetrics) -> go.Figure:
    labels = [str(score) for score in metrics.quality_scores]
    figure = go.Figure(go.Heatmap(
        z=metrics.confusion_matrix, x=labels, y=labels,
        colorscale="Reds", texttemplate="%{z}", hovertemplate="Actual %{y}<br>Predicted %{x}<br>Wines %{z}<extra></extra>",
    ))
    figure.update_layout(title="Actual vs. predicted quality", xaxis_title="Predicted score",
                         yaxis_title="Actual score", yaxis_autorange="reversed")
    return figure


def feature_importance(model: ModelBundle) -> go.Figure:
    ranked = sorted(zip(model.feature_names, model.feature_importances), key=lambda item: item[1])
    figure = go.Figure(go.Bar(x=[value for _, value in ranked],
                             y=[name for name, _ in ranked], orientation="h",
                             marker_color="#8b2635"))
    figure.update_layout(title="Feature importance", xaxis_title="Model-native importance",
                         yaxis_title="Chemical property", height=450)
    return figure


def score_distribution(metrics: EvaluationMetrics) -> go.Figure:
    labels = [str(score) for score in metrics.quality_scores]
    figure = go.Figure()
    figure.add_bar(name="Actual", x=labels, y=metrics.actual_counts, marker_color="#8b2635")
    figure.add_bar(name="Predicted", x=labels, y=metrics.predicted_counts, marker_color="#d49a61")
    figure.update_layout(title="Actual vs. predicted score counts", barmode="group",
                         xaxis_title="Quality score", yaxis_title="Number of wines")
    return figure


def feature_quality_box(model: ModelBundle, feature: str) -> go.Figure:
    if feature not in model.feature_names:
        raise ValueError(f"Unknown feature: {feature}")
    figure = go.Figure()
    for score in model.metrics.quality_scores:
        values = model.eda_data.loc[model.eda_data["quality"] == score, feature]
        figure.add_trace(go.Box(y=values, name=str(score), boxpoints="outliers",
                                marker_color="#8b2635", showlegend=False))
    figure.update_layout(title=f"{feature.title()} by quality score", xaxis_title="Quality score",
                         yaxis_title=feature.title())
    return figure


def residual_distribution(model: ModelBundle) -> go.Figure:
    residuals = [actual - predicted for actual, predicted in
                 zip(model.test_actual, model.test_predictions)]
    figure = go.Figure(go.Histogram(x=residuals, nbinsx=30, marker_color="#8b2635"))
    figure.add_vline(x=0, line_dash="dash", line_color="#333333")
    figure.update_layout(title="Held-out residuals", xaxis_title="Actual − predicted quality points",
                         yaxis_title="Number of wines")
    return figure
