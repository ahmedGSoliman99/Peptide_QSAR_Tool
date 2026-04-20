"""Domain constants for peptide descriptor calculations."""

from __future__ import annotations

STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AMINO_ACIDS)

# Residue elemental formulas inside a peptide chain (after condensation).
# One H2O is added for complete peptide termini in calculations.
RESIDUE_ATOMS = {
    "A": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 0},
    "C": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1},
    "D": {"C": 4, "H": 5, "N": 1, "O": 3, "S": 0},
    "E": {"C": 5, "H": 7, "N": 1, "O": 3, "S": 0},
    "F": {"C": 9, "H": 9, "N": 1, "O": 1, "S": 0},
    "G": {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0},
    "H": {"C": 6, "H": 7, "N": 3, "O": 1, "S": 0},
    "I": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},
    "K": {"C": 6, "H": 12, "N": 2, "O": 1, "S": 0},
    "L": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},
    "M": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 1},
    "N": {"C": 4, "H": 6, "N": 2, "O": 2, "S": 0},
    "P": {"C": 5, "H": 7, "N": 1, "O": 1, "S": 0},
    "Q": {"C": 5, "H": 8, "N": 2, "O": 2, "S": 0},
    "R": {"C": 6, "H": 12, "N": 4, "O": 1, "S": 0},
    "S": {"C": 3, "H": 5, "N": 1, "O": 2, "S": 0},
    "T": {"C": 4, "H": 7, "N": 1, "O": 2, "S": 0},
    "V": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 0},
    "W": {"C": 11, "H": 10, "N": 2, "O": 1, "S": 0},
    "Y": {"C": 9, "H": 9, "N": 1, "O": 2, "S": 0},
}

AA_CLASSES = {
    "positive_frac": set("KRH"),
    "negative_frac": set("DE"),
    "polar_uncharged_frac": set("STNQCY"),
    "hydrophobic_frac": set("AILMFWVPG"),
    "aromatic_frac": set("FYW"),
    "small_frac": set("AGSTP"),
}

# Solubility-like residue values frequently used in peptide design workflows.
# Boman index approximation = average of residue potentials.
BOMAN_SCALE = {
    "A": 0.17,
    "C": 0.24,
    "D": -1.23,
    "E": -2.02,
    "F": 1.13,
    "G": 0.01,
    "H": -0.96,
    "I": 0.31,
    "K": -0.99,
    "L": 0.56,
    "M": 0.23,
    "N": -0.42,
    "P": -0.45,
    "Q": -0.58,
    "R": -0.81,
    "S": -0.13,
    "T": -0.14,
    "V": 0.07,
    "W": 1.85,
    "Y": 0.94,
}

MOTIF_FINGERPRINTS = [
    "KK",
    "RR",
    "KR",
    "RK",
    "GG",
    "PP",
    "FW",
    "WF",
    "HP",
    "PH",
]

# Kyte-Doolittle hydrophobicity scale.
HYDROPHOBICITY_KD_SCALE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Chou-Fasman propensities (normalized values used for sequence averages).
CHOU_FASMAN_HELIX = {
    "A": 1.45,
    "C": 0.77,
    "D": 1.01,
    "E": 1.51,
    "F": 1.13,
    "G": 0.53,
    "H": 1.00,
    "I": 1.08,
    "K": 1.16,
    "L": 1.21,
    "M": 1.45,
    "N": 0.67,
    "P": 0.59,
    "Q": 1.11,
    "R": 0.98,
    "S": 0.79,
    "T": 0.82,
    "V": 1.06,
    "W": 1.08,
    "Y": 0.69,
}

CHOU_FASMAN_SHEET = {
    "A": 0.97,
    "C": 1.30,
    "D": 0.54,
    "E": 0.37,
    "F": 1.38,
    "G": 0.81,
    "H": 0.87,
    "I": 1.60,
    "K": 0.74,
    "L": 1.30,
    "M": 1.05,
    "N": 0.89,
    "P": 0.62,
    "Q": 1.10,
    "R": 0.90,
    "S": 0.72,
    "T": 1.20,
    "V": 1.70,
    "W": 1.37,
    "Y": 1.47,
}
