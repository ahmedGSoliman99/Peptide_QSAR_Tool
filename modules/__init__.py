"""Peptide QSAR tool modules package."""

from .descriptor_engine import DescriptorConfig, calculate_descriptors_dataframe
from .evaluation import evaluate_classification, evaluate_regression
from .modeling import (
    ModelBundle,
    available_models_for_task,
    get_model_family,
    load_model_bundle,
    predict_with_bundle,
    save_model_bundle,
    summarize_model_bundle,
    train_and_compare_models,
)
from .preprocessing import FeaturePreprocessor, PreprocessingConfig, SplitConfig, split_dataset
from .structure_criteria import evaluate_structure_criteria, summarize_structure_quality

__all__ = [
    "DescriptorConfig",
    "calculate_descriptors_dataframe",
    "evaluate_classification",
    "evaluate_regression",
    "FeaturePreprocessor",
    "PreprocessingConfig",
    "SplitConfig",
    "split_dataset",
    "ModelBundle",
    "available_models_for_task",
    "get_model_family",
    "save_model_bundle",
    "load_model_bundle",
    "predict_with_bundle",
    "summarize_model_bundle",
    "train_and_compare_models",
    "evaluate_structure_criteria",
    "summarize_structure_quality",
]
