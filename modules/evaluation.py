"""Evaluation helpers for regression and classification QSAR tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics, confusion matrix, and optional ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    unique_classes = np.unique(y_true)
    binary_task = unique_classes.shape[0] == 2
    if binary_task and y_prob is not None:
        try:
            y_prob = np.asarray(y_prob)
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                prob_positive = y_prob[:, 1]
            else:
                prob_positive = y_prob.ravel()
            metrics["ROC_AUC"] = float(roc_auc_score(y_true, prob_positive))
        except Exception:
            # ROC-AUC is optional and may fail for unsupported probability outputs.
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan

    matrix_labels = labels if labels is not None else unique_classes.tolist()
    metrics["ConfusionMatrix"] = confusion_matrix(y_true, y_pred, labels=matrix_labels)
    metrics["ClassLabels"] = [str(x) for x in matrix_labels]
    metrics["ClassificationReport"] = classification_report(
        y_true,
        y_pred,
        labels=matrix_labels,
        output_dict=True,
        zero_division=0,
    )

    return metrics

