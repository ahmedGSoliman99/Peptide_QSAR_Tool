"""Peptide descriptor calculation engine."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:  # pragma: no cover - optional at import time, required for 3D descriptors
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    rdMolDescriptors = None

from .constants import (
    AA_CLASSES,
    BOMAN_SCALE,
    CHOU_FASMAN_HELIX,
    CHOU_FASMAN_SHEET,
    HYDROPHOBICITY_KD_SCALE,
    MOTIF_FINGERPRINTS,
    RESIDUE_ATOMS,
    STANDARD_AMINO_ACIDS,
)
from .io_utils import clean_sequence


@dataclass
class DescriptorConfig:
    include_dipeptide: bool = False
    include_fingerprints: bool = True
    include_elemental: bool = True
    include_3d: bool = False


def _safe_protein_analysis(sequence: str) -> ProteinAnalysis:
    return ProteinAnalysis(sequence)


def _aa_composition(sequence: str) -> dict[str, float]:
    length = len(sequence)
    counts = {aa: sequence.count(aa) for aa in STANDARD_AMINO_ACIDS}
    if length == 0:
        return {f"AAC_{aa}": 0.0 for aa in STANDARD_AMINO_ACIDS}
    return {f"AAC_{aa}": counts[aa] / length for aa in STANDARD_AMINO_ACIDS}


def _dipeptide_composition(sequence: str) -> dict[str, float]:
    dipeptides = [a + b for a, b in itertools.product(STANDARD_AMINO_ACIDS, repeat=2)]
    if len(sequence) < 2:
        return {f"DPC_{di}": 0.0 for di in dipeptides}

    total = len(sequence) - 1
    counts = {di: 0 for di in dipeptides}
    for idx in range(total):
        pair = sequence[idx : idx + 2]
        if pair in counts:
            counts[pair] += 1
    return {f"DPC_{di}": counts[di] / total for di in dipeptides}


def _sequence_entropy(sequence: str) -> float:
    if not sequence:
        return 0.0
    freqs = np.array([sequence.count(aa) / len(sequence) for aa in STANDARD_AMINO_ACIDS], dtype=float)
    freqs = freqs[freqs > 0]
    return float(-np.sum(freqs * np.log2(freqs)))


def _aliphatic_index(sequence: str) -> float:
    if not sequence:
        return 0.0
    length = len(sequence)
    ala = sequence.count("A") / length * 100
    val = sequence.count("V") / length * 100
    ile_leu = (sequence.count("I") + sequence.count("L")) / length * 100
    return float(ala + 2.9 * val + 3.9 * ile_leu)


def _boman_index(sequence: str) -> float:
    if not sequence:
        return 0.0
    return float(sum(BOMAN_SCALE.get(aa, 0.0) for aa in sequence) / len(sequence))


def _hydrophobic_moment(sequence: str, angle_degrees: float = 100.0) -> float:
    """Approximate hydrophobic moment for an alpha-helical projection."""
    if not sequence:
        return 0.0
    angle_radians = math.radians(angle_degrees)
    x_sum = 0.0
    y_sum = 0.0
    for idx, aa in enumerate(sequence):
        h_value = HYDROPHOBICITY_KD_SCALE.get(aa, 0.0)
        theta = idx * angle_radians
        x_sum += h_value * math.cos(theta)
        y_sum += h_value * math.sin(theta)
    return float(math.sqrt(x_sum * x_sum + y_sum * y_sum) / len(sequence))


def _average_propensity(sequence: str, scale: dict[str, float]) -> float:
    if not sequence:
        return 0.0
    return float(sum(scale.get(aa, 0.0) for aa in sequence) / len(sequence))


def _residue_class_frequencies(sequence: str) -> dict[str, float]:
    length = len(sequence) if sequence else 1
    return {
        class_name: sum(1 for aa in sequence if aa in aa_set) / length
        for class_name, aa_set in AA_CLASSES.items()
    }


def _elemental_composition(sequence: str) -> dict[str, float]:
    totals = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0}
    for aa in sequence:
        atom_map = RESIDUE_ATOMS.get(aa)
        if atom_map is None:
            continue
        for element, value in atom_map.items():
            totals[element] += value

    if sequence:
        # Add H2O for complete peptide termini.
        totals["H"] += 2
        totals["O"] += 1

    total_atoms = max(sum(totals.values()), 1)
    return {
        "Elem_C": float(totals["C"]),
        "Elem_H": float(totals["H"]),
        "Elem_N": float(totals["N"]),
        "Elem_O": float(totals["O"]),
        "Elem_S": float(totals["S"]),
        "Elem_C_fraction": totals["C"] / total_atoms,
        "Elem_N_fraction": totals["N"] / total_atoms,
        "Elem_O_fraction": totals["O"] / total_atoms,
        "Elem_S_fraction": totals["S"] / total_atoms,
    }


def _motif_fingerprints(sequence: str) -> dict[str, float]:
    if len(sequence) < 2:
        return {f"Motif_{motif}": 0.0 for motif in MOTIF_FINGERPRINTS}
    denom = len(sequence) - 1
    return {f"Motif_{motif}": sequence.count(motif) / denom for motif in MOTIF_FINGERPRINTS}


def _zero_3d_descriptors(success: float = 0.0) -> dict[str, float]:
    return {
        "3D_EmbedSuccess": float(success),
        "3D_Asphericity": 0.0,
        "3D_Eccentricity": 0.0,
        "3D_InertialShapeFactor": 0.0,
        "3D_NPR1": 0.0,
        "3D_NPR2": 0.0,
        "3D_PMI1": 0.0,
        "3D_PMI2": 0.0,
        "3D_PMI3": 0.0,
        "3D_RadiusOfGyration": 0.0,
        "3D_SpherocityIndex": 0.0,
    }


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def generate_peptide_3d_mol(sequence: str):
    """Generate an approximate 3D peptide conformer from a one-letter sequence."""
    if Chem is None or AllChem is None:
        return None, "RDKit is not installed"
    sequence = clean_sequence(sequence)
    if not sequence:
        return None, "Empty sequence"
    if len(sequence) > 45:
        return None, "3D embedding skipped for peptides longer than 45 residues"
    mol = Chem.MolFromFASTA(sequence)
    if mol is None:
        return None, "RDKit could not build peptide from sequence"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if status != 0:
        return None, "3D embedding failed"
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=250)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)
    except Exception:
        pass
    return mol, "ETKDG 3D conformer generated"


def _peptide_3d_descriptors(sequence: str) -> dict[str, float]:
    out = _zero_3d_descriptors(success=0.0)
    if rdMolDescriptors is None:
        return out
    mol3d, _ = generate_peptide_3d_mol(sequence)
    if mol3d is None:
        return out
    out["3D_EmbedSuccess"] = 1.0
    descriptor_funcs = {
        "3D_Asphericity": getattr(rdMolDescriptors, "CalcAsphericity", None),
        "3D_Eccentricity": getattr(rdMolDescriptors, "CalcEccentricity", None),
        "3D_InertialShapeFactor": getattr(rdMolDescriptors, "CalcInertialShapeFactor", None),
        "3D_NPR1": getattr(rdMolDescriptors, "CalcNPR1", None),
        "3D_NPR2": getattr(rdMolDescriptors, "CalcNPR2", None),
        "3D_PMI1": getattr(rdMolDescriptors, "CalcPMI1", None),
        "3D_PMI2": getattr(rdMolDescriptors, "CalcPMI2", None),
        "3D_PMI3": getattr(rdMolDescriptors, "CalcPMI3", None),
        "3D_RadiusOfGyration": getattr(rdMolDescriptors, "CalcRadiusOfGyration", None),
        "3D_SpherocityIndex": getattr(rdMolDescriptors, "CalcSpherocityIndex", None),
    }
    for name, func in descriptor_funcs.items():
        if func is None:
            continue
        try:
            value = _safe_float(func(mol3d))
        except Exception:
            value = None
        out[name] = value if value is not None else 0.0
    return out


def calculate_sequence_descriptors(sequence: str, config: DescriptorConfig) -> dict[str, float]:
    sequence = clean_sequence(sequence)
    if not sequence:
        raise ValueError("Empty sequence is not allowed.")

    analysis = _safe_protein_analysis(sequence)
    helix_frac, turn_frac, sheet_frac = analysis.secondary_structure_fraction()
    ext_reduced, ext_oxidized = analysis.molar_extinction_coefficient()
    length = len(sequence)
    net_charge = float(analysis.charge_at_pH(7.0))

    base = {
        "Length": length,
        "MolecularWeight": float(analysis.molecular_weight()),
        "Aromaticity": float(analysis.aromaticity()),
        "Hydrophobicity_KD": float(analysis.gravy()),
        "NetCharge_pH7": net_charge,
        "IsoelectricPoint": float(analysis.isoelectric_point()),
        "InstabilityIndex": float(analysis.instability_index()),
        "Gravy": float(analysis.gravy()),
        "AliphaticIndex": _aliphatic_index(sequence),
        "BomanIndex": _boman_index(sequence),
        "ShannonEntropy": _sequence_entropy(sequence),
        "ChargeDensity": net_charge / max(length, 1),
        "HydrophobicMoment100": _hydrophobic_moment(sequence, angle_degrees=100.0),
        "HelixFraction": float(helix_frac),
        "TurnFraction": float(turn_frac),
        "SheetFraction": float(sheet_frac),
        "HelixPropensityAvg": _average_propensity(sequence, CHOU_FASMAN_HELIX),
        "SheetPropensityAvg": _average_propensity(sequence, CHOU_FASMAN_SHEET),
        "ExtCoeffReduced": float(ext_reduced),
        "ExtCoeffOxidized": float(ext_oxidized),
    }

    base.update(_aa_composition(sequence))
    base.update(_residue_class_frequencies(sequence))

    if config.include_elemental:
        base.update(_elemental_composition(sequence))
    if config.include_3d:
        base.update(_peptide_3d_descriptors(sequence))
    if config.include_fingerprints:
        base.update(_motif_fingerprints(sequence))
    if config.include_dipeptide:
        base.update(_dipeptide_composition(sequence))
    return base


def calculate_descriptors_dataframe(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    config: DescriptorConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Calculate descriptors for a dataframe containing peptide sequences."""
    config = config or DescriptorConfig()
    rows: list[dict] = []
    total = len(df)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        data = row._asdict()
        sequence = clean_sequence(data.get(sequence_col, ""))
        if not sequence:
            continue
        descriptors = calculate_sequence_descriptors(sequence, config)
        combined = {k: v for k, v in data.items() if k != sequence_col}
        combined["Sequence"] = sequence
        combined.update(descriptors)
        rows.append(combined)
        if progress_callback is not None:
            progress_callback(idx, total)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_descriptor_columns(df: pd.DataFrame) -> list[str]:
    descriptor_base = {
        "Length",
        "MolecularWeight",
        "Aromaticity",
        "Hydrophobicity_KD",
        "NetCharge_pH7",
        "IsoelectricPoint",
        "InstabilityIndex",
        "Gravy",
        "AliphaticIndex",
        "BomanIndex",
        "ShannonEntropy",
        "ChargeDensity",
        "HydrophobicMoment100",
        "HelixFraction",
        "TurnFraction",
        "SheetFraction",
        "HelixPropensityAvg",
        "SheetPropensityAvg",
        "ExtCoeffReduced",
        "ExtCoeffOxidized",
    }
    descriptor_base.update(AA_CLASSES.keys())
    descriptor_prefixes = ("AAC_", "DPC_", "Motif_", "Elem_", "3D_")

    out: list[str] = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col in descriptor_base or col.startswith(descriptor_prefixes):
            out.append(col)
    return out


def describe_descriptor_set(config: DescriptorConfig) -> str:
    parts = [
        "AAC, core physicochemical descriptors, and structural propensity descriptors",
        "dipeptide composition" if config.include_dipeptide else "no dipeptide composition",
        "motif fingerprints" if config.include_fingerprints else "no motif fingerprints",
        "elemental composition" if config.include_elemental else "no elemental composition",
        "approximate 3D conformer descriptors" if config.include_3d else "no 3D conformer descriptors",
    ]
    return ", ".join(parts)
