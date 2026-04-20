"""Position-level peptide design suggestions from activity correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import AA_CLASSES, STANDARD_AMINO_ACIDS
from .io_utils import clean_sequence


LOWER_IS_BETTER_HINTS = ("ic50", "mic", "ec50", "ki", "kd", "docking", "vina", "affinity")


def infer_optimization_direction(target_column: str) -> str:
    """Return 'minimize' for common potency/docking targets, otherwise 'maximize'."""
    target = (target_column or "").lower()
    if any(hint in target for hint in LOWER_IS_BETTER_HINTS):
        return "minimize"
    return "maximize"


def _pearson_binary(values: np.ndarray, target: np.ndarray) -> float:
    if values.std() == 0 or target.std() == 0:
        return np.nan
    return float(np.corrcoef(values, target)[0, 1])


def positional_activity_analysis(
    df: pd.DataFrame,
    sequence_col: str,
    target_col: str,
    direction: str = "auto",
    min_count: int = 2,
    max_positions: int = 60,
) -> dict[str, pd.DataFrame]:
    """Analyze which residues at each position correlate with the target."""
    if sequence_col not in df.columns or target_col not in df.columns:
        raise ValueError("Sequence and target columns are required for positional analysis.")

    working = df[[sequence_col, target_col]].copy()
    working[sequence_col] = working[sequence_col].map(clean_sequence)
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working.dropna(subset=[target_col])
    working = working[working[sequence_col].astype(str).str.len() > 0].copy()
    if working.empty:
        raise ValueError("No numeric target values were available for positional analysis.")

    direction = infer_optimization_direction(target_col) if direction == "auto" else direction
    sign = 1.0 if direction == "maximize" else -1.0
    sequences = working[sequence_col].tolist()
    y = working[target_col].to_numpy(dtype=float)
    global_mean = float(np.mean(y))
    max_len = min(max(len(seq) for seq in sequences), int(max_positions))

    rows: list[dict] = []
    for pos in range(max_len):
        residues_at_pos = np.array([seq[pos] if len(seq) > pos else "" for seq in sequences], dtype=object)
        for aa in STANDARD_AMINO_ACIDS:
            mask = residues_at_pos == aa
            count = int(mask.sum())
            if count < int(min_count):
                continue
            mean_target = float(np.mean(y[mask]))
            delta = mean_target - global_mean
            corr = _pearson_binary(mask.astype(float), y)
            rows.append(
                {
                    "Position": pos + 1,
                    "Residue": aa,
                    "Count": count,
                    "MeanTarget": mean_target,
                    "DeltaFromMean": delta,
                    "Correlation": corr,
                    "DesignScore": sign * delta,
                    "Direction": direction,
                }
            )

    analysis_df = pd.DataFrame(rows)
    if analysis_df.empty:
        return {
            "position_residue_effects": analysis_df,
            "position_recommendations": pd.DataFrame(),
            "class_recommendations": pd.DataFrame(),
        }

    rec_rows: list[dict] = []
    for pos, group in analysis_df.groupby("Position"):
        best = group.sort_values(["DesignScore", "Count"], ascending=[False, False]).iloc[0]
        worst = group.sort_values(["DesignScore", "Count"], ascending=[True, False]).iloc[0]
        rec_rows.append(
            {
                "Position": int(pos),
                "RecommendedResidue": best["Residue"],
                "AvoidResidue": worst["Residue"],
                "RecommendedMeanTarget": float(best["MeanTarget"]),
                "AvoidMeanTarget": float(worst["MeanTarget"]),
                "RecommendedCorrelation": float(best["Correlation"]) if pd.notna(best["Correlation"]) else np.nan,
                "EvidenceCount": int(best["Count"]),
                "Direction": direction,
                "Suggestion": (
                    f"Position {int(pos)} favors {best['Residue']} "
                    f"({target_col} mean {best['MeanTarget']:.3g}); avoid {worst['Residue']} "
                    f"when evidence is consistent."
                ),
            }
        )
    recommendations_df = pd.DataFrame(rec_rows)

    class_rows: list[dict] = []
    for pos in range(1, max_len + 1):
        pos_effects = analysis_df[analysis_df["Position"] == pos]
        if pos_effects.empty:
            continue
        for class_name, residues in AA_CLASSES.items():
            subset = pos_effects[pos_effects["Residue"].isin(residues)]
            if subset.empty:
                continue
            weighted_score = np.average(subset["DesignScore"], weights=subset["Count"])
            class_rows.append(
                {
                    "Position": pos,
                    "ResidueClass": class_name,
                    "WeightedDesignScore": float(weighted_score),
                    "TotalEvidenceCount": int(subset["Count"].sum()),
                    "Direction": direction,
                }
            )

    return {
        "position_residue_effects": analysis_df.sort_values(
            ["Position", "DesignScore"], ascending=[True, False]
        ).reset_index(drop=True),
        "position_recommendations": recommendations_df,
        "class_recommendations": pd.DataFrame(class_rows).sort_values(
            ["Position", "WeightedDesignScore"], ascending=[True, False]
        ).reset_index(drop=True)
        if class_rows
        else pd.DataFrame(),
    }


def suggest_mutations_for_sequence(
    sequence: str,
    recommendations_df: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    """Create position-specific mutation suggestions for one new peptide."""
    sequence = clean_sequence(sequence)
    if not sequence or recommendations_df is None or recommendations_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in recommendations_df.iterrows():
        pos = int(row["Position"])
        if pos > len(sequence):
            continue
        current = sequence[pos - 1]
        recommended = str(row["RecommendedResidue"])
        if current == recommended:
            action = "Keep"
            suggested_sequence = sequence
        else:
            action = "Consider mutation"
            suggested_sequence = sequence[: pos - 1] + recommended + sequence[pos:]
        rows.append(
            {
                "Position": pos,
                "CurrentResidue": current,
                "RecommendedResidue": recommended,
                "Action": action,
                "SuggestedSequence": suggested_sequence,
                "EvidenceCount": int(row.get("EvidenceCount", 0)),
                "RecommendedCorrelation": row.get("RecommendedCorrelation", np.nan),
                "Suggestion": row.get("Suggestion", ""),
            }
        )
    return pd.DataFrame(rows).head(int(top_n))

