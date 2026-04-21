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


def _edit_distance(a: str, b: str) -> int:
    """Small Levenshtein implementation to avoid an extra dependency."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _mutation_summary(reference: str, sequence: str, limit: int = 12) -> str:
    changes: list[str] = []
    max_shared = min(len(reference), len(sequence))
    for idx in range(max_shared):
        if reference[idx] != sequence[idx]:
            changes.append(f"{reference[idx]}{idx + 1}{sequence[idx]}")
    if len(sequence) > len(reference):
        changes.append(f"+{len(sequence) - len(reference)} residue(s)")
    elif len(reference) > len(sequence):
        changes.append(f"-{len(reference) - len(sequence)} residue(s)")
    if not changes:
        return "Reference"
    suffix = " ..." if len(changes) > limit else ""
    return ", ".join(changes[:limit]) + suffix


def best_reference_sequence(
    df: pd.DataFrame,
    sequence_col: str,
    target_col: str,
    direction: str = "auto",
) -> pd.Series:
    """Return the best observed peptide by numeric target."""
    if sequence_col not in df.columns or target_col not in df.columns:
        raise ValueError("Sequence and target columns are required.")
    direction = infer_optimization_direction(target_col) if direction == "auto" else direction
    working = df[[sequence_col, target_col] + ([c for c in ["Name"] if c in df.columns])].copy()
    working[sequence_col] = working[sequence_col].map(clean_sequence)
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working.dropna(subset=[target_col])
    working = working[working[sequence_col].astype(str).str.len() > 0]
    if working.empty:
        raise ValueError("No numeric target values are available.")
    idx = working[target_col].idxmax() if direction == "maximize" else working[target_col].idxmin()
    return working.loc[idx]


def sequence_alignment_to_reference(
    df: pd.DataFrame,
    sequence_col: str,
    target_col: str,
    direction: str = "auto",
) -> pd.DataFrame:
    """Compare every peptide with the best observed peptide sequence."""
    reference = best_reference_sequence(df, sequence_col, target_col, direction=direction)
    ref_seq = clean_sequence(reference[sequence_col])
    ref_target = float(reference[target_col])
    rows: list[dict] = []
    for _, row in df.iterrows():
        sequence = clean_sequence(row.get(sequence_col, ""))
        if not sequence:
            continue
        edit_distance = _edit_distance(ref_seq, sequence)
        max_len = max(len(ref_seq), len(sequence), 1)
        shared = sum(1 for a, b in zip(ref_seq, sequence) if a == b)
        rows.append(
            {
                "Name": row.get("Name", ""),
                "Sequence": sequence,
                "Target": pd.to_numeric(row.get(target_col), errors="coerce"),
                "ReferenceName": reference.get("Name", "Best reference"),
                "ReferenceSequence": ref_seq,
                "ReferenceTarget": ref_target,
                "SequenceIdentityToReference": round(shared / max_len, 3),
                "EditSimilarityToReference": round(1.0 - edit_distance / max_len, 3),
                "EditDistanceToReference": int(edit_distance),
                "LengthDifference": int(len(sequence) - len(ref_seq)),
                "MutationSummaryVsReference": _mutation_summary(ref_seq, sequence),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["EditSimilarityToReference", "SequenceIdentityToReference"],
        ascending=False,
    ).reset_index(drop=True)


def peptide_activity_design_profile(
    df: pd.DataFrame,
    target_col: str,
    descriptor_cols: list[str],
    direction: str = "auto",
) -> pd.DataFrame:
    """Summarize descriptor ranges observed in the best-activity peptides."""
    direction = infer_optimization_direction(target_col) if direction == "auto" else direction
    target = pd.to_numeric(df[target_col], errors="coerce")
    threshold = target.quantile(0.75 if direction == "maximize" else 0.25)
    mask = target >= threshold if direction == "maximize" else target <= threshold
    top = df[mask].copy()
    if top.empty:
        return pd.DataFrame()
    preferred = [
        "Length",
        "MolecularWeight",
        "Gravy",
        "Hydrophobicity_KD",
        "NetCharge_pH7",
        "ChargeDensity",
        "IsoelectricPoint",
        "AliphaticIndex",
        "BomanIndex",
        "HydrophobicMoment100",
        "HelixFraction",
        "SheetFraction",
        "InstabilityIndex",
    ]
    rows: list[dict] = []
    for descriptor in preferred:
        if descriptor not in descriptor_cols or descriptor not in top.columns:
            continue
        values = pd.to_numeric(top[descriptor], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Criterion": descriptor,
                "RecommendedRange": f"{values.quantile(0.10):.3g} to {values.quantile(0.90):.3g}",
                "MedianInBestPeptides": float(values.median()),
                "Direction": direction,
                "Reason": "Observed range among the best-activity peptides in this dataset.",
            }
        )
    return pd.DataFrame(rows)
