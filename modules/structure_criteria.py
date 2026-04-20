"""Heuristic peptide structure-quality criteria for screening new candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


# These screening windows are practical heuristics for early peptide triage.
# They are not universal pharmacological rules and should be adjusted per project.
DEFAULT_CRITERIA = {
    "Length": (6, 40),
    "NetCharge_pH7": (-1.0, 10.0),
    "Gravy": (-2.5, 1.5),
    "InstabilityIndex": (0.0, 40.0),
    "BomanIndex": (-3.0, 3.0),
    "Aromaticity": (0.0, 0.40),
    "HydrophobicMoment100": (0.10, 3.50),
    "HelixFraction": (0.10, 1.00),
}


def evaluate_structure_criteria(
    descriptor_df: pd.DataFrame,
    criteria: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate each peptide against configurable screening criteria."""
    criteria = criteria or DEFAULT_CRITERIA
    if descriptor_df is None or descriptor_df.empty:
        return pd.DataFrame()

    out = descriptor_df.copy()
    pass_cols: list[str] = []

    for feature, (min_v, max_v) in criteria.items():
        if feature not in out.columns:
            continue
        numeric = pd.to_numeric(out[feature], errors="coerce")
        col_name = f"Pass_{feature}"
        out[col_name] = ((numeric >= min_v) & (numeric <= max_v)).astype(int)
        pass_cols.append(col_name)

    if not pass_cols:
        out["StructureQualityScore"] = np.nan
        out["StructureAssessment"] = "Insufficient descriptor coverage"
        return out

    out["CriteriaPassCount"] = out[pass_cols].sum(axis=1)
    out["CriteriaTotal"] = len(pass_cols)
    out["StructureQualityScore"] = (out["CriteriaPassCount"] / max(len(pass_cols), 1)) * 100.0

    def _label(score: float) -> str:
        if pd.isna(score):
            return "Unknown"
        if score >= 85:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 50:
            return "Moderate"
        return "Needs Optimization"

    out["StructureAssessment"] = out["StructureQualityScore"].map(_label)
    return out


def summarize_structure_quality(criteria_df: pd.DataFrame) -> dict[str, float]:
    if criteria_df is None or criteria_df.empty or "StructureQualityScore" not in criteria_df.columns:
        return {}
    score = pd.to_numeric(criteria_df["StructureQualityScore"], errors="coerce")
    return {
        "MeanStructureQualityScore": float(score.mean()),
        "MedianStructureQualityScore": float(score.median()),
        "HighQualityCount(score>=70)": int((score >= 70).sum()),
    }

