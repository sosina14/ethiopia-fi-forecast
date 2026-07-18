"""
data_loader.py

Data loading and validation utilities for the Ethiopia Financial Inclusion
forecasting project (10 Academy Week 11 Challenge).

Author: Sosina Ayele
"""
from pathlib import Path
import pandas as pd


def load_unified_data(path: str) -> pd.DataFrame:
    """Load the unified financial-inclusion dataset and parse dates.

    Args:
        path: Path to ethiopia_fi_unified_data.csv.

    Returns:
        DataFrame with observation_date parsed as datetime.

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
        ValueError: If required columns are missing from the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(p)

    required_cols = {"record_id", "record_type", "observation_date", "indicator_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    n_bad = df["observation_date"].isna().sum()
    if n_bad:
        print(f"Warning: {n_bad} row(s) had an unparseable observation_date")

    return df


def load_impact_links(path: str) -> pd.DataFrame:
    """Load impact_link records and validate that parent_id is present.

    Args:
        path: Path to Impact_sheet.csv.

    Returns:
        DataFrame of impact_link records.

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
        ValueError: If the parent_id column is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Impact sheet not found at {path}")

    df = pd.read_csv(p)
    if "parent_id" not in df.columns:
        raise ValueError("Impact sheet is missing the 'parent_id' column")

    return df


def load_reference_codes(path: str) -> pd.DataFrame:
    """Load the reference_codes.csv lookup of valid categorical values.

    Args:
        path: Path to reference_codes.csv.

    Returns:
        DataFrame of valid field/code/description rows.

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference codes file not found at {path}")

    return pd.read_csv(p)


def split_by_record_type(df: pd.DataFrame) -> dict:
    """Split the unified dataset into observations, events, and targets.

    Args:
        df: Unified dataset as returned by load_unified_data().

    Returns:
        Dict with keys 'observations', 'events', 'targets', each a DataFrame.

    Raises:
        ValueError: If 'record_type' column is not present in df.
    """
    if "record_type" not in df.columns:
        raise ValueError("DataFrame has no 'record_type' column to split on")

    return {
        "observations": df[df["record_type"] == "observation"].copy(),
        "events": df[df["record_type"] == "event"].copy(),
        "targets": df[df["record_type"] == "target"].copy(),
    }


def validate_against_reference(df: pd.DataFrame, ref: pd.DataFrame, field: str) -> pd.DataFrame:
    """Return rows in df whose value in `field` is not a valid code per ref.

    Args:
        df: Dataset to validate (must contain the given field column).
        ref: Reference codes DataFrame with 'field' and 'code' columns.
        field: Name of the column in df to validate (e.g. 'pillar').

    Returns:
        Subset of df with invalid / unrecognized values in `field`
        (NaN values are treated as valid and excluded from the result).

    Raises:
        ValueError: If `field` is not a column in df, or ref lacks
            the expected 'field'/'code' columns.
    """
    if field not in df.columns:
        raise ValueError(f"'{field}' is not a column in the provided DataFrame")
    if not {"field", "code"}.issubset(ref.columns):
        raise ValueError("Reference DataFrame must have 'field' and 'code' columns")

    valid_codes = set(ref.loc[ref["field"] == field, "code"])
    mask = df[field].notna() & ~df[field].isin(valid_codes)
    return df[mask]
