"""
Basic unit tests for src/data_loader.py.

Run with: pytest tests/test_data_loader.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import (
    load_unified_data,
    load_impact_links,
    split_by_record_type,
    validate_against_reference,
)


def test_load_unified_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_unified_data("does_not_exist.csv")


def test_load_unified_data_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_unified_data(str(bad_csv))


def test_load_unified_data_parses_dates(tmp_path):
    good_csv = tmp_path / "good.csv"
    pd.DataFrame({
        "record_id": ["R1"],
        "record_type": ["observation"],
        "observation_date": ["2024-01-01"],
        "indicator_code": ["ACC_OWNERSHIP"],
    }).to_csv(good_csv, index=False)

    df = load_unified_data(str(good_csv))
    assert pd.api.types.is_datetime64_any_dtype(df["observation_date"])


def test_load_impact_links_missing_parent_id(tmp_path):
    bad_csv = tmp_path / "bad_impact.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_impact_links(str(bad_csv))


def test_split_by_record_type():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "record_type": ["observation", "event", "target"],
    })
    parts = split_by_record_type(df)
    assert len(parts["observations"]) == 1
    assert len(parts["events"]) == 1
    assert len(parts["targets"]) == 1


def test_split_by_record_type_missing_column():
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError):
        split_by_record_type(df)


def test_validate_against_reference():
    df = pd.DataFrame({"pillar": ["ACCESS", "GENDER", None]})
    ref = pd.DataFrame({
        "field": ["pillar", "pillar", "pillar"],
        "code": ["ACCESS", "USAGE", "QUALITY"],
    })
    invalid = validate_against_reference(df, ref, "pillar")
    assert list(invalid["pillar"]) == ["GENDER"]
