"""Visualization helpers for peptide descriptor and QSAR outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_TEMPLATE = "plotly_white"


def _top_variance_features(df: pd.DataFrame, columns: list[str], max_features: int = 12) -> list[str]:
    if not columns:
        return []
    variance = df[columns].var(numeric_only=True).sort_values(ascending=False)
    return variance.head(max_features).index.tolist()


def plot_descriptor_distribution(df: pd.DataFrame, descriptor_cols: list[str], max_features: int = 12) -> go.Figure:
    top_features = _top_variance_features(df, descriptor_cols, max_features=max_features)
    if not top_features:
        return go.Figure()
    melted = df[top_features].melt(var_name="Descriptor", value_name="Value")
    fig = px.violin(
        melted,
        x="Descriptor",
        y="Value",
        color="Descriptor",
        box=True,
        points=False,
        title="Descriptor Distributions (Top-Variance Features)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, descriptor_cols: list[str], max_features: int = 80) -> go.Figure:
    use_cols = descriptor_cols[:max_features]
    if not use_cols:
        return go.Figure()
    corr = df[use_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Descriptor Correlation Heatmap",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return fig


def plot_embedding(
    embedding_df: pd.DataFrame,
    labels: Iterable | None = None,
    names: Iterable | None = None,
    title: str = "Descriptor Embedding",
) -> go.Figure:
    if embedding_df.shape[1] < 2:
        return go.Figure()
    cols = list(embedding_df.columns[:2])
    plot_df = embedding_df.copy()
    if labels is not None:
        plot_df["Label"] = pd.Series(labels, index=embedding_df.index).astype(str)
        color_arg = "Label"
    else:
        color_arg = None
    if names is not None:
        plot_df["Name"] = pd.Series(names, index=embedding_df.index).astype(str)
        hover_name = "Name"
    else:
        hover_name = None

    fig = px.scatter(
        plot_df,
        x=cols[0],
        y=cols[1],
        color=color_arg,
        hover_name=hover_name,
        title=title,
        template=PLOTLY_TEMPLATE,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#0B1F3A")))
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, primary_metric: str) -> go.Figure:
    if comparison_df.empty or primary_metric not in comparison_df.columns:
        return go.Figure()
    fig = px.bar(
        comparison_df,
        x="Model",
        y=primary_metric,
        color="Model",
        title=f"Model Comparison ({primary_metric})",
        template=PLOTLY_TEMPLATE,
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, title: str = "Feature Importance") -> go.Figure:
    if importance_df.empty:
        return go.Figure()
    value_col = "Importance" if "Importance" in importance_df.columns else "MeanAbsSHAP"
    top = importance_df.head(top_n).iloc[::-1]
    fig = px.bar(
        top,
        x=value_col,
        y="Feature",
        orientation="h",
        title=title,
        template=PLOTLY_TEMPLATE,
        color=value_col,
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(yaxis_title=None)
    return fig


def plot_prediction_ranking(prediction_df: pd.DataFrame, score_col: str = "RankingScore", top_n: int = 30) -> go.Figure:
    if prediction_df.empty or score_col not in prediction_df.columns:
        return go.Figure()
    top = prediction_df.head(top_n).copy()
    top["Rank"] = np.arange(1, len(top) + 1)
    fig = px.bar(
        top,
        x="Rank",
        y=score_col,
        hover_data=["Sequence", "Prediction"],
        title=f"Top {len(top)} Predicted Peptides",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Rank (Best to Worst)")
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return go.Figure()

    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title="Actual vs Predicted",
        template=PLOTLY_TEMPLATE,
        opacity=0.8,
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Ideal",
            line=dict(color="#0B1F3A", dash="dash"),
        )
    )
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={"x": "Predicted", "y": "Residual (Actual - Predicted)"},
        title="Residual Plot",
        template=PLOTLY_TEMPLATE,
        opacity=0.8,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#0B1F3A")
    return fig


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    if matrix.size == 0:
        return go.Figure()
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def plot_pca_explained_variance(explained_df: pd.DataFrame) -> go.Figure:
    if explained_df is None or explained_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=explained_df["Component"],
            y=explained_df["ExplainedVarianceRatio"],
            name="Explained Variance",
            marker_color="#3B82F6",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=explained_df["Component"],
            y=explained_df["CumulativeVarianceRatio"],
            mode="lines+markers",
            name="Cumulative Variance",
            yaxis="y2",
            line=dict(color="#0B1F3A", width=2),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="PCA Explained Variance",
        yaxis=dict(title="Variance Ratio"),
        yaxis2=dict(
            title="Cumulative",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        xaxis_title="Principal Components",
    )
    return fig


def plot_pca_loading_bar(loadings_df: pd.DataFrame, component: str = "PC1", top_n: int = 20) -> go.Figure:
    if loadings_df is None or loadings_df.empty or component not in loadings_df.columns:
        return go.Figure()
    ranked = (
        loadings_df[["Feature", component]]
        .assign(AbsLoading=lambda d: d[component].abs())
        .sort_values("AbsLoading", ascending=False)
        .head(top_n)
        .sort_values(component)
    )
    fig = px.bar(
        ranked,
        x=component,
        y="Feature",
        orientation="h",
        title=f"Top PCA Loadings ({component})",
        template=PLOTLY_TEMPLATE,
        color=component,
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(yaxis_title=None)
    return fig


def plot_structure_quality_distribution(criteria_df: pd.DataFrame) -> go.Figure:
    if criteria_df is None or criteria_df.empty or "StructureQualityScore" not in criteria_df.columns:
        return go.Figure()
    fig = px.histogram(
        criteria_df,
        x="StructureQualityScore",
        color="StructureAssessment" if "StructureAssessment" in criteria_df.columns else None,
        nbins=20,
        title="Peptide Structure Quality Score Distribution",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Structure Quality Score (%)")
    return fig


def plot_structure_criteria_radar(criteria_row: pd.Series, feature_cols: list[str]) -> go.Figure:
    if criteria_row is None or not feature_cols:
        return go.Figure()
    values = [float(criteria_row.get(col, 0.0)) for col in feature_cols]
    labels = [col.replace("Pass_", "") for col in feature_cols]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Criteria Pass (1=Pass, 0=Fail)",
            line=dict(color="#0B6EA8"),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Structure Criteria Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
    )
    return fig


def plot_position_residue_heatmap(effects_df: pd.DataFrame, value_col: str = "DesignScore") -> go.Figure:
    if effects_df is None or effects_df.empty or value_col not in effects_df.columns:
        return go.Figure()
    pivot = effects_df.pivot_table(
        index="Residue",
        columns="Position",
        values=value_col,
        aggfunc="mean",
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdBu_r",
        title=f"Position-Residue {value_col} Heatmap",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Peptide Position", yaxis_title="Residue")
    return fig


def plot_position_recommendation_scores(recommendations_df: pd.DataFrame) -> go.Figure:
    if recommendations_df is None or recommendations_df.empty:
        return go.Figure()
    fig = px.bar(
        recommendations_df,
        x="Position",
        y="RecommendedMeanTarget",
        color="RecommendedResidue",
        hover_data=["AvoidResidue", "EvidenceCount", "RecommendedCorrelation"],
        title="Recommended Residue by Position",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Peptide Position")
    return fig
