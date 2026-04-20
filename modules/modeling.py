"""Model training, persistence, prediction, and explainability utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR

from .descriptor_engine import DescriptorConfig, calculate_descriptors_dataframe, get_descriptor_columns
from .evaluation import evaluate_classification, evaluate_regression
from .preprocessing import FeaturePreprocessor, PreprocessingConfig, SplitConfig, split_dataset

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None

try:  # pragma: no cover - optional dependency
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


def _normalize_task_type(task_type: str) -> str:
    task = (task_type or "regression").strip().lower().replace("-", "_")
    if task in {"binary", "binary_classification", "classification_binary"}:
        return "binary_classification"
    if task in {"multiclass", "multi_class", "multiclass_classification", "classification_multiclass"}:
        return "multiclass_classification"
    if "class" in task and "multi" in task:
        return "multiclass_classification"
    if "class" in task:
        return "binary_classification"
    return "regression"


def _is_classification(task_type: str) -> bool:
    return "classification" in _normalize_task_type(task_type)


def get_model_family(model_name: str) -> str:
    lower = model_name.lower()
    if "linear" in lower or "ridge" in lower or "lasso" in lower or "elastic" in lower or "logistic" in lower:
        return "Linear"
    if "svr" in lower or "svm" in lower:
        return "Nonlinear Kernel"
    if "knn" in lower:
        return "Nonlinear Distance"
    if "forest" in lower or "boost" in lower or "xgboost" in lower or "lightgbm" in lower or "tree" in lower:
        return "Tree Ensemble"
    if "mlp" in lower:
        return "Neural"
    if "bayes" in lower:
        return "Probabilistic"
    return "Other"


def available_models_for_task(task_type: str, n_classes: int = 2) -> dict[str, Any]:
    """Return configured model objects supported for a task type."""
    task = _normalize_task_type(task_type)
    models: dict[str, Any] = {}

    if task == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001, max_iter=10000, random_state=42),
            "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=1
            ),
            "SVR (RBF)": SVR(C=10.0, gamma="scale"),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            ),
            "kNN": KNeighborsRegressor(n_neighbors=5),
            "MLP Regressor": MLPRegressor(
                hidden_layer_sizes=(128, 64),
                random_state=42,
                max_iter=3000,
                early_stopping=True,
            ),
        }
        if XGBRegressor is not None:
            models["XGBoost"] = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        if LGBMRegressor is not None:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
            )
    else:
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            ),
            "SVM (RBF)": SVC(C=5.0, gamma="scale", probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            ),
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "MLP Classifier": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                random_state=42,
                max_iter=3000,
                early_stopping=True,
            ),
        }
        if XGBClassifier is not None:
            if n_classes <= 2:
                models["XGBoost"] = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
            else:
                models["XGBoost"] = XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                )
        if LGBMClassifier is not None:
            if n_classes <= 2:
                models["LightGBM"] = LGBMClassifier(
                    objective="binary",
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                )
            else:
                models["LightGBM"] = LGBMClassifier(
                    objective="multiclass",
                    num_class=n_classes,
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                )
    return models


@dataclass
class ModelBundle:
    """Serializable object containing everything needed for future predictions."""

    task_type: str
    model_name: str
    estimator: Any
    preprocessor: FeaturePreprocessor
    descriptor_config: DescriptorConfig
    descriptor_columns: list[str]
    target_column: str
    feature_names_transformed: list[str]
    label_encoder: LabelEncoder | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    cv_summary: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _prepare_target(y: pd.Series, task_type: str) -> tuple[pd.Series, LabelEncoder | None]:
    task = _normalize_task_type(task_type)
    if task == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        return y_num, None

    y_text = y.astype(str).str.strip()
    y_text = y_text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(y_text), index=y_text.index, name=y.name)
    return encoded, encoder


def _pick_cv(task_type: str, y: pd.Series, requested_folds: int) -> Any | None:
    folds = max(2, int(requested_folds))
    if _is_classification(task_type):
        class_counts = y.value_counts()
        if class_counts.empty:
            return None
        max_folds = int(class_counts.min())
        folds = min(folds, max_folds)
        if folds < 2:
            return None
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    folds = min(folds, len(y))
    if folds < 2:
        return None
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def _run_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    cv_folds: int,
) -> dict[str, float]:
    cv = _pick_cv(task_type, y, cv_folds)
    if cv is None:
        return {}

    if _is_classification(task_type):
        scoring = ["accuracy", "f1_weighted"]
    else:
        scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

    try:
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
    except Exception:
        return {}

    summary: dict[str, float] = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        clean_name = key.replace("test_", "CV_")
        values = np.asarray(values, dtype=float)
        if "neg_" in clean_name:
            values = -values
            clean_name = clean_name.replace("neg_", "")
        summary[f"{clean_name}_mean"] = float(np.mean(values))
        summary[f"{clean_name}_std"] = float(np.std(values))
    return summary


def _safe_predict_proba(estimator: Any, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        return estimator.predict_proba(X)
    except Exception:
        return None


def _decode_if_needed(values: np.ndarray, encoder: LabelEncoder | None) -> np.ndarray:
    if encoder is None:
        return values
    try:
        int_values = np.asarray(values, dtype=int)
        return encoder.inverse_transform(int_values)
    except Exception:
        return values


def train_and_compare_models(
    descriptor_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    selected_models: list[str],
    descriptor_config: DescriptorConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    split_config: SplitConfig | None = None,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train selected models, evaluate, and return a comparison package."""
    descriptor_config = descriptor_config or DescriptorConfig()
    preprocessing_config = preprocessing_config or PreprocessingConfig()
    split_config = split_config or SplitConfig()
    task = _normalize_task_type(task_type)

    if target_column not in descriptor_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    descriptor_columns = [col for col in get_descriptor_columns(descriptor_df) if col != target_column]
    if not descriptor_columns:
        raise ValueError("No descriptor columns were found. Please calculate descriptors first.")

    working = descriptor_df.copy()
    working = working.dropna(subset=[target_column]).copy()
    if working.empty:
        raise ValueError("No rows available after removing missing target values.")

    y_raw = working[target_column]
    y_encoded, label_encoder = _prepare_target(y_raw, task)
    if task == "regression":
        valid_mask = y_encoded.notna()
        working = working.loc[valid_mask].copy()
        y_encoded = y_encoded.loc[valid_mask]

    if working.empty:
        raise ValueError("No valid rows remain for model training.")

    if task == "binary_classification":
        if y_encoded.nunique() != 2:
            raise ValueError(
                "Binary classification requires exactly two classes in the selected target column."
            )

    X = working[descriptor_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.Series(y_encoded.values, index=working.index, name=target_column)

    split = split_dataset(X, y, task, split_config)
    preprocessor = FeaturePreprocessor(preprocessing_config)
    preprocessor.fit(split["X_train"])

    X_train = preprocessor.transform_dataframe(split["X_train"])
    X_val = preprocessor.transform_dataframe(split["X_val"]) if not split["X_val"].empty else split["X_val"]
    X_test = preprocessor.transform_dataframe(split["X_test"])

    n_classes = int(y.nunique()) if _is_classification(task) else 1
    catalog = available_models_for_task(task, n_classes=n_classes)
    chosen = [name for name in selected_models if name in catalog]
    if not chosen:
        raise ValueError("Please select at least one valid model.")

    model_records: list[dict[str, Any]] = []
    model_outputs: dict[str, Any] = {}

    for model_name in chosen:
        estimator = clone(catalog[model_name])
        estimator.fit(X_train, split["y_train"])

        y_pred_test = estimator.predict(X_test)
        y_prob_test = _safe_predict_proba(estimator, X_test) if _is_classification(task) else None

        if _is_classification(task):
            y_true_for_eval = _decode_if_needed(split["y_test"].to_numpy(), label_encoder)
            y_pred_for_eval = _decode_if_needed(y_pred_test, label_encoder)
            labels_for_eval = (
                [str(label) for label in label_encoder.classes_]
                if label_encoder is not None
                else [str(v) for v in np.unique(y.to_numpy())]
            )
            metrics = evaluate_classification(
                y_true_for_eval,
                y_pred_for_eval,
                y_prob=y_prob_test,
                labels=labels_for_eval,
            )
            record = {
                "Model": model_name,
                "ModelFamily": get_model_family(model_name),
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC_AUC": metrics["ROC_AUC"],
            }
            primary_score = metrics["F1"]
        else:
            metrics = evaluate_regression(split["y_test"].to_numpy(), y_pred_test)
            record = {
                "Model": model_name,
                "ModelFamily": get_model_family(model_name),
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
            }
            primary_score = metrics["R2"]

        cv_summary = _run_cv(estimator, X_train, split["y_train"], task, cv_folds=cv_folds)
        record.update(cv_summary)
        record["PrimaryScore"] = primary_score
        model_records.append(record)

        y_pred_test_decoded = _decode_if_needed(y_pred_test, label_encoder)
        y_true_test_decoded = _decode_if_needed(split["y_test"].to_numpy(), label_encoder)
        y_pred_val_decoded = None
        y_true_val_decoded = None
        y_prob_val = None
        if isinstance(X_val, pd.DataFrame) and not X_val.empty:
            y_pred_val = estimator.predict(X_val)
            y_pred_val_decoded = _decode_if_needed(y_pred_val, label_encoder)
            y_true_val_decoded = _decode_if_needed(split["y_val"].to_numpy(), label_encoder)
            if _is_classification(task):
                y_prob_val = _safe_predict_proba(estimator, X_val)

        bundle = ModelBundle(
            task_type=task,
            model_name=model_name,
            estimator=estimator,
            preprocessor=preprocessor,
            descriptor_config=descriptor_config,
            descriptor_columns=descriptor_columns,
            target_column=target_column,
            feature_names_transformed=preprocessor.output_feature_names.copy(),
            label_encoder=label_encoder,
            metrics=metrics,
            cv_summary=cv_summary,
        )

        model_outputs[model_name] = {
            "bundle": bundle,
            "test": {
                "y_true": y_true_test_decoded,
                "y_pred": y_pred_test_decoded,
                "y_prob": y_prob_test,
                "metrics": metrics,
            },
            "validation": {
                "y_true": y_true_val_decoded,
                "y_pred": y_pred_val_decoded,
                "y_prob": y_prob_val,
            },
        }

    comparison_df = pd.DataFrame(model_records).sort_values("PrimaryScore", ascending=False).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["Model"])

    return {
        "task_type": task,
        "comparison_table": comparison_df,
        "model_outputs": model_outputs,
        "best_model_name": best_model_name,
        "preprocessing_config": preprocessing_config,
        "split_config": split_config,
        "dataset_size": int(len(working)),
    }


def summarize_model_bundle(bundle: ModelBundle) -> dict[str, Any]:
    estimator = bundle.estimator
    model_params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    compact_params = {}
    for key in sorted(model_params.keys()):
        value = model_params[key]
        if isinstance(value, (int, float, str, bool, type(None))):
            compact_params[key] = value
    return {
        "ModelName": bundle.model_name,
        "ModelFamily": get_model_family(bundle.model_name),
        "TaskType": bundle.task_type,
        "TargetColumn": bundle.target_column,
        "DescriptorCount": len(bundle.descriptor_columns),
        "TransformedFeatureCount": len(bundle.feature_names_transformed),
        "CreatedAt": bundle.created_at,
        "EstimatorClass": estimator.__class__.__name__,
        "EstimatorParams": compact_params,
    }


def save_model_bundle(bundle: ModelBundle, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(model_path: str | Path) -> ModelBundle:
    model = joblib.load(model_path)
    if not isinstance(model, ModelBundle):
        raise TypeError("Loaded file is not a valid ModelBundle.")
    return model


def predict_with_bundle(bundle: ModelBundle, sequences_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate descriptors and model predictions for incoming peptide sequences."""
    descriptor_df = calculate_descriptors_dataframe(
        sequences_df,
        sequence_col="Sequence",
        config=bundle.descriptor_config,
    )
    if descriptor_df.empty:
        raise ValueError("No valid peptide sequences were available for prediction.")

    missing = [col for col in bundle.descriptor_columns if col not in descriptor_df.columns]
    for col in missing:
        descriptor_df[col] = 0.0

    X_descriptors = descriptor_df[bundle.descriptor_columns].apply(pd.to_numeric, errors="coerce")
    X_model = bundle.preprocessor.transform_dataframe(X_descriptors)

    y_pred = bundle.estimator.predict(X_model)
    y_prob = _safe_predict_proba(bundle.estimator, X_model) if _is_classification(bundle.task_type) else None

    if bundle.label_encoder is not None:
        y_pred_decoded = _decode_if_needed(y_pred, bundle.label_encoder)
    else:
        y_pred_decoded = y_pred

    result_df = descriptor_df[["Sequence"]].copy()
    if "Name" in descriptor_df.columns:
        result_df.insert(0, "Name", descriptor_df["Name"])

    result_df["Prediction"] = y_pred_decoded

    if y_prob is not None:
        class_labels = (
            [str(v) for v in bundle.label_encoder.classes_]
            if bundle.label_encoder is not None
            else [str(i) for i in range(y_prob.shape[1])]
        )
        prob_df = pd.DataFrame(y_prob, columns=[f"Prob_{c}" for c in class_labels], index=result_df.index)
        result_df = pd.concat([result_df, prob_df], axis=1)

        if y_prob.shape[1] == 2:
            result_df["RankingScore"] = y_prob[:, 1]
        else:
            result_df["RankingScore"] = y_prob.max(axis=1)
    else:
        result_df["RankingScore"] = pd.to_numeric(result_df["Prediction"], errors="coerce")

    result_df = result_df.sort_values("RankingScore", ascending=False).reset_index(drop=True)
    result_df.insert(0, "Rank", np.arange(1, len(result_df) + 1))
    return result_df, descriptor_df


def compute_model_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    """Return model-native feature importance if the estimator exposes it."""
    estimator = bundle.estimator
    features = bundle.feature_names_transformed

    values: np.ndarray | None = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)

    if values is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    n = min(len(features), len(values))
    out = pd.DataFrame({"Feature": features[:n], "Importance": values[:n]})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def compute_shap_importance(
    bundle: ModelBundle,
    descriptor_df: pd.DataFrame,
    max_samples: int = 300,
) -> pd.DataFrame:
    """Compute SHAP mean absolute contribution per feature when supported."""
    if shap is None:
        raise ImportError("SHAP is not installed.")

    if descriptor_df.empty:
        raise ValueError("Descriptor table is empty.")

    missing = [col for col in bundle.descriptor_columns if col not in descriptor_df.columns]
    if missing:
        raise ValueError("Descriptor table does not match the selected model features.")

    X_input = descriptor_df[bundle.descriptor_columns].apply(pd.to_numeric, errors="coerce")
    X_model = bundle.preprocessor.transform_dataframe(X_input)
    if X_model.empty:
        raise ValueError("No data left after preprocessing.")

    sample_n = min(max_samples, len(X_model))
    X_sample = X_model.sample(n=sample_n, random_state=42) if len(X_model) > sample_n else X_model

    explainer = shap.Explainer(bundle.estimator, X_sample)
    shap_values = explainer(X_sample)

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = np.mean(np.abs(values), axis=2)
    abs_mean = np.mean(np.abs(values), axis=0)

    importance_df = pd.DataFrame(
        {
            "Feature": X_sample.columns,
            "MeanAbsSHAP": abs_mean,
        }
    ).sort_values("MeanAbsSHAP", ascending=False)

    return importance_df.reset_index(drop=True)
