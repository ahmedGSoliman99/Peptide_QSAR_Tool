"""Input parsing and peptide validation helpers."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import STANDARD_AA_SET


SEQUENCE_COLUMN_CANDIDATES = [
    "sequence",
    "peptide",
    "peptide_sequence",
    "peptidesequence",
    "seq",
    "aa_sequence",
    "aasequence",
    "aminoacidsequence",
    "amino_acid_sequence",
]

TARGET_COLUMN_HINTS = [
    "activity",
    "ic50",
    "mic",
    "toxicity",
    "solubility",
    "bindingscore",
    "docking",
    "docking_score",
    "dockingscore",
    "vina",
    "affinity",
    "property",
    "target",
]


@dataclass
class ValidationResult:
    valid_df: pd.DataFrame
    invalid_df: pd.DataFrame
    duplicate_df: pd.DataFrame


def clean_sequence(sequence: str) -> str:
    """Normalize peptide sequence text."""
    if sequence is None:
        return ""
    return re.sub(r"\s+", "", str(sequence).strip().upper())


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower().strip())


def parse_fasta_text(text: str) -> pd.DataFrame:
    """Parse FASTA text into a dataframe with Name and Sequence."""
    records: list[dict] = []
    current_name = None
    current_seq: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None:
                records.append({"Name": current_name, "Sequence": "".join(current_seq)})
            current_name = line[1:].strip() or f"Sequence_{len(records) + 1}"
            current_seq = []
            continue
        current_seq.append(clean_sequence(line))

    if current_name is not None:
        records.append({"Name": current_name, "Sequence": "".join(current_seq)})

    return pd.DataFrame(records)


def parse_sequence_block(text: str) -> pd.DataFrame:
    """Parse manual sequence input (plain lines, comma-separated, or FASTA)."""
    if not text or not text.strip():
        return pd.DataFrame(columns=["Name", "Sequence"])

    if ">" in text:
        fasta_df = parse_fasta_text(text)
        if not fasta_df.empty:
            return fasta_df

    chunks = re.split(r"[\n,;\t ]+", text.strip())
    sequences = [clean_sequence(chunk) for chunk in chunks if clean_sequence(chunk)]
    records = [{"Name": f"Manual_{idx + 1}", "Sequence": seq} for idx, seq in enumerate(sequences)]
    return pd.DataFrame(records)


def parse_sdf_text(text: str) -> pd.DataFrame:
    """Parse SDF text and extract molecule properties including sequence fields."""
    records: list[dict] = []
    sequence_keys = {_normalize_header(x) for x in SEQUENCE_COLUMN_CANDIDATES}

    blocks = text.split("$$$$")
    for idx, block in enumerate(blocks, start=1):
        if not block.strip():
            continue
        raw_lines = [line.rstrip("\r") for line in block.splitlines()]
        first_non_empty = next((i for i, line in enumerate(raw_lines) if line.strip()), None)
        if first_non_empty is None:
            continue
        lines = raw_lines[first_non_empty:]

        record: dict[str, str] = {"Name": lines[0].strip() or f"SDF_{idx}"}

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            prop_match = re.match(r"^>\s*<([^>]+)>\s*$", line)
            if prop_match:
                prop_name = prop_match.group(1).strip()
                i += 1
                values: list[str] = []
                while i < len(lines) and lines[i].strip() != "":
                    values.append(lines[i].strip())
                    i += 1
                record[prop_name] = " ".join(values).strip()
            i += 1

        for key, value in list(record.items()):
            norm = _normalize_header(key)
            if norm in sequence_keys and "Sequence" not in record:
                record["Sequence"] = clean_sequence(value)
                break

        if "Sequence" not in record:
            # Fallback: infer sequence-like property from uppercase amino-acid text.
            for key, value in list(record.items()):
                if key == "Name":
                    continue
                seq_candidate = clean_sequence(value)
                if len(seq_candidate) >= 3 and all(ch in STANDARD_AA_SET for ch in seq_candidate):
                    record["Sequence"] = seq_candidate
                    break

        if "Sequence" not in record:
            title_candidate = clean_sequence(record.get("Name", ""))
            if len(title_candidate) >= 3 and all(ch in STANDARD_AA_SET for ch in title_candidate):
                record["Sequence"] = title_candidate

        records.append(record)

    if not records:
        return pd.DataFrame(columns=["Name", "Sequence"])

    df = pd.DataFrame(records)
    for col in df.columns:
        if col in {"Name", "Sequence"}:
            continue
        as_num = pd.to_numeric(df[col], errors="coerce")
        if float(as_num.notna().mean()) >= 0.8:
            df[col] = as_num
    return df


def _decode_bytes(file_bytes: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def guess_sequence_column(df: pd.DataFrame) -> str | None:
    lowered = {_normalize_header(col): col for col in df.columns}
    for candidate in SEQUENCE_COLUMN_CANDIDATES:
        candidate_norm = _normalize_header(candidate)
        if candidate_norm in lowered:
            return lowered[candidate_norm]
    for col in df.columns:
        col_l = _normalize_header(col)
        if "seq" in col_l or "peptide" in col_l:
            return col

    # Value-based fallback for messy files (including many SDF exports).
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        sample = series.astype(str).head(200)
        if sample.empty:
            continue
        peptide_like = sample.map(
            lambda v: (
                lambda s: len(s) >= 3 and all(ch in STANDARD_AA_SET for ch in s)
            )(clean_sequence(v))
        )
        if float(peptide_like.mean()) >= 0.6:
            return col
    return None


def guess_target_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        cl = col.lower().strip()
        if cl in TARGET_COLUMN_HINTS or any(hint in cl for hint in TARGET_COLUMN_HINTS):
            candidates.append(col)
    numeric_candidates = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_candidates:
        if col not in candidates and col.lower().strip() != "id":
            candidates.append(col)
    return candidates


def read_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    ext = Path(file_name).suffix.lower()

    if ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    if ext in (".fasta", ".fa", ".faa"):
        return parse_fasta_text(_decode_bytes(file_bytes))
    if ext == ".sdf":
        return parse_sdf_text(_decode_bytes(file_bytes))
    if ext == ".txt":
        raw_text = _decode_bytes(file_bytes)
        # TXT can be FASTA, delimited table, or plain line-separated sequences.
        if ">" in raw_text:
            fasta_df = parse_fasta_text(raw_text)
            if not fasta_df.empty:
                return fasta_df
        # Try parsing as tab/comma-separated text table first.
        for sep in ("\t", ",", ";", "|"):
            try:
                table = pd.read_csv(io.StringIO(raw_text), sep=sep)
                if table.shape[1] > 1:
                    return table
            except Exception:
                continue
        return parse_sequence_block(raw_text)

    raise ValueError(f"Unsupported file format: {ext}")


def validate_sequences(df: pd.DataFrame, sequence_col: str = "Sequence") -> ValidationResult:
    """Validate peptide sequences and split valid/invalid/duplicate rows."""
    working = df.copy()
    working[sequence_col] = working[sequence_col].map(clean_sequence)
    working = working[working[sequence_col].astype(str).str.len() > 0].copy()

    invalid_mask = working[sequence_col].map(
        lambda seq: any(char not in STANDARD_AA_SET for char in seq)
    )
    invalid_df = working[invalid_mask].copy()
    invalid_df["InvalidCharacters"] = invalid_df[sequence_col].map(
        lambda seq: "".join(sorted(set(char for char in seq if char not in STANDARD_AA_SET)))
    )
    valid_df = working[~invalid_mask].copy()

    duplicate_mask = valid_df.duplicated(subset=[sequence_col], keep=False)
    duplicate_df = valid_df[duplicate_mask].sort_values(sequence_col).copy()
    valid_df = valid_df.drop_duplicates(subset=[sequence_col], keep="first").copy()

    return ValidationResult(valid_df=valid_df, invalid_df=invalid_df, duplicate_df=duplicate_df)


def merge_manual_and_uploaded(
    uploaded_df: pd.DataFrame | None,
    manual_df: pd.DataFrame | None,
    sequence_col_uploaded: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if uploaded_df is not None and not uploaded_df.empty:
        df_u = uploaded_df.copy()
        if sequence_col_uploaded and sequence_col_uploaded != "Sequence":
            df_u = df_u.rename(columns={sequence_col_uploaded: "Sequence"})
        frames.append(df_u)
    if manual_df is not None and not manual_df.empty:
        frames.append(manual_df.copy())
    if not frames:
        return pd.DataFrame(columns=["Name", "Sequence"])
    return pd.concat(frames, ignore_index=True)


def to_fasta(records: Iterable[tuple[str, str]]) -> str:
    lines: list[str] = []
    for name, sequence in records:
        lines.append(f">{name}")
        lines.append(clean_sequence(sequence))
    return "\n".join(lines)
