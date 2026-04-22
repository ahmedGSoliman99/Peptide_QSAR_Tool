"""Data preprocessing helpers for QSAR modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None


@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    impute_strategy: str = "median"
    scaler: str = "standard"  # one of: standard, minmax, robust, none
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    use_pca: bool = False
    pca_components: int = 20


def _get_scaler(scaler_name: str):
    scaler_name = (scaler_name or "none").lower()
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    if scaler_name == "robust":
        return RobustScaler()
    return None


class FeaturePreprocessor:
    """Stateful preprocessor used for training and later prediction."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.input_columns: list[str] = []
        self.output_feature_names: list[str] = []
        self.correlation_drop_columns: list[str] = []

        self.imputer = SimpleImputer(strategy=self.config.impute_strategy)
        self.variance_selector = VarianceThreshold(threshold=max(self.config.variance_threshold, 0.0))
        self.scaler = _get_scaler(self.config.scaler)
        self.pca = None

        self._variance_columns: list[str] = []
        self._corr_kept_columns: list[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        numeric = X.select_dtypes(include=["number"]).copy()
        if numeric.empty:
            raise ValueError("No numeric descriptor columns available after preprocessing.")

        self.input_columns = list(numeric.columns)

        imputed = self.imputer.fit_transform(numeric)
        imputed_df = pd.DataFrame(imputed, columns=self.input_columns, index=numeric.index)

        var_matrix = self.variance_selector.fit_transform(imputed_df)
        var_mask = self.variance_selector.get_support()
        self._variance_columns = [col for col, keep in zip(self.input_columns, var_mask) if keep]

        var_df = pd.DataFrame(var_matrix, columns=self._variance_columns, index=numeric.index)
        self.correlation_drop_columns = self._compute_corr_drop_columns(
            var_df, self.config.correlation_threshold
        )
        self._corr_kept_columns = [c for c in self._variance_columns if c not in self.correlation_drop_columns]
        corr_df = var_df[self._corr_kept_columns]

        if self.scaler is not None:
            scaled = self.scaler.fit_transform(corr_df)
            scaled_df = pd.DataFrame(scaled, columns=self._corr_kept_columns, index=numeric.index)
        else:
            scaled_df = corr_df.copy()

        if self.config.use_pca:
            n_components = max(2, min(self.config.pca_components, scaled_df.shape[1]))
            self.pca = PCA(n_components=n_components, random_state=42)
            transformed = self.pca.fit_transform(scaled_df)
            self.output_feature_names = [f"PC{i + 1}" for i in range(transformed.shape[1])]
        else:
            self.output_feature_names = self._corr_kept_columns.copy()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.input_columns:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        aligned = X.reindex(columns=self.input_columns).copy()
        imputed = self.imputer.transform(aligned)
        imputed_df = pd.DataFrame(imputed, columns=self.input_columns, index=aligned.index)
        var_matrix = self.variance_selector.transform(imputed_df)

        var_df = pd.DataFrame(var_matrix, columns=self._variance_columns, index=aligned.index)
        corr_df = var_df[self._corr_kept_columns]

        if self.scaler is not None:
            scaled = self.scaler.transform(corr_df)
            scaled_df = pd.DataFrame(scaled, columns=self._corr_kept_columns, index=aligned.index)
        else:
            scaled_df = corr_df

        if self.pca is not None:
            return self.pca.transform(scaled_df)
        return scaled_df.to_numpy(dtype=float)

    def transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        matrix = self.transform(X)
        return pd.DataFrame(matrix, columns=self.output_feature_names, index=X.index)

    def fit_transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform_dataframe(X)

    @staticmethod
    def _compute_corr_drop_columns(df: pd.DataFrame, threshold: float) -> list[str]:
        if df.shape[1] <= 1 or threshold >= 1.0:
            return []
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return [column for column in upper.columns if any(upper[column] > threshold)]


def _regression_stratify_bins(y: pd.Series, max_bins: int = 5) -> pd.Series | None:
    """Create target-quantile bins so regression splits keep activity ranges balanced."""
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().sum() < 12 or y_num.nunique(dropna=True) < 4:
        return None

    n_bins = min(max_bins, max(2, int(np.sqrt(y_num.notna().sum()))))
    try:
        bins = pd.qcut(y_num, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        return None

    bins = pd.Series(bins, index=y.index)
    counts = bins.value_counts(dropna=True)
    if counts.empty or len(counts) < 2 or counts.min() < 2:
        return None
    return bins.astype("Int64")


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    config: SplitConfig | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    config = config or SplitConfig()
    task_type = task_type.lower().strip()
    is_classification = "classification" in task_type
    stratify_values = y if is_classification else _regression_stratify_bins(y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_values,
    )

    if config.val_size > 0:
        relative_val = config.val_size / (1.0 - config.test_size)
        relative_val = min(max(relative_val, 0.05), 0.5)
        stratify_train_val = y_train_val if is_classification else _regression_stratify_bins(y_train_val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val,
            random_state=config.random_state,
            stratify=stratify_train_val,
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = X_train.iloc[0:0].copy()
        y_val = y_train.iloc[0:0].copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def compute_embedding(
    X: pd.DataFrame,
    method: str = "PCA",
    random_state: int = 42,
    n_components: int = 2,
) -> pd.DataFrame:
    method = method.upper()
    numeric = X.select_dtypes(include=["number"])
    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    scaled = StandardScaler().fit_transform(imputed)

    if method == "PCA":
        embedder = PCA(n_components=n_components, random_state=random_state)
        arr = embedder.fit_transform(scaled)
    elif method in ("TSNE", "T-SNE"):
        if scaled.shape[0] < 3:
            raise ValueError("t-SNE requires at least 3 samples.")
        perplexity = min(30, max(2, scaled.shape[0] // 3))
        perplexity = min(perplexity, scaled.shape[0] - 1)
        embedder = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        arr = embedder.fit_transform(scaled)
    elif method == "UMAP":
        if umap is None:
            raise ImportError("UMAP is not installed. Please install umap-learn.")
        embedder = umap.UMAP(n_components=n_components, random_state=random_state)
        arr = embedder.fit_transform(scaled)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

    columns = [f"{method}_1", f"{method}_2"]
    return pd.DataFrame(arr, columns=columns, index=X.index)


def compute_pca_analysis(
    X: pd.DataFrame,
    random_state: int = 42,
    n_components: int = 10,
) -> dict[str, pd.DataFrame]:
    """Run PCA and return scores, explained variance, and loadings."""
    numeric = X.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise ValueError("No numeric features available for PCA.")

    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    scaled = StandardScaler().fit_transform(imputed)

    max_comp = min(numeric.shape[0], numeric.shape[1])
    n_comp = max(2, min(int(n_components), max_comp))
    pca = PCA(n_components=n_comp, random_state=random_state)
    scores = pca.fit_transform(scaled)

    pc_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    score_df = pd.DataFrame(scores, columns=pc_cols, index=numeric.index)

    explained = pd.DataFrame(
        {
            "Component": pc_cols,
            "ExplainedVarianceRatio": pca.explained_variance_ratio_,
            "CumulativeVarianceRatio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric.columns,
        columns=pc_cols,
    ).reset_index(names="Feature")

    return {
        "scores": score_df,
        "explained_variance": explained,
        "loadings": loadings,
    }
