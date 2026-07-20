"""
Basic unit tests for src/impact_model.py.

Run with: pytest tests/test_impact_model.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from impact_model import (
    encode_impact_score,
    build_association_matrix,
    ramp_effect,
    combine_effects,
    validate_against_observed,
)


def test_encode_impact_score_basic():
    df = pd.DataFrame({
        "impact_direction": ["increase", "decrease", "increase"],
        "impact_magnitude": ["high", "medium", "low"],
    })
    out = encode_impact_score(df)
    assert list(out["impact_score"]) == [3, -2, 1]


def test_encode_impact_score_unrecognized_warns(capsys):
    df = pd.DataFrame({
        "impact_direction": ["sideways"],
        "impact_magnitude": ["high"],
    })
    out = encode_impact_score(df)
    assert out["impact_score"].isna().all()
    assert "Warning" in capsys.readouterr().out


def test_build_association_matrix():
    df = pd.DataFrame({
        "parent_id": ["EVT_0001", "EVT_0001", "EVT_0002"],
        "related_indicator": ["ACC_OWNERSHIP", "USG_P2P_COUNT", "ACC_OWNERSHIP"],
        "impact_score": [3, 3, -2],
    })
    matrix = build_association_matrix(df)
    assert matrix.loc["EVT_0001", "ACC_OWNERSHIP"] == 3
    assert matrix.loc["EVT_0002", "ACC_OWNERSHIP"] == -2
    assert pd.isna(matrix.loc["EVT_0002", "USG_P2P_COUNT"])


def test_build_association_matrix_missing_columns():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(ValueError):
        build_association_matrix(df)


def test_ramp_effect_zero_before_event():
    dates = pd.to_datetime(["2021-01-01", "2021-06-01"])
    effect = ramp_effect("2021-05-17", lag_months=12, magnitude=3, as_of_dates=dates)
    assert effect[0] == 0.0


def test_ramp_effect_full_after_lag():
    dates = pd.to_datetime(["2023-06-01"])
    effect = ramp_effect("2021-05-17", lag_months=12, magnitude=3, as_of_dates=dates)
    assert effect[0] == pytest.approx(3.0, abs=0.01)


def test_ramp_effect_negative_lag_raises():
    dates = pd.to_datetime(["2021-06-01"])
    with pytest.raises(ValueError):
        ramp_effect("2021-05-17", lag_months=-1, magnitude=3, as_of_dates=dates)


def test_combine_effects_sums():
    e1 = np.array([1.0, 2.0])
    e2 = np.array([0.5, 0.5])
    combined = combine_effects([e1, e2])
    assert list(combined) == [1.5, 2.5]


def test_combine_effects_empty():
    assert list(combine_effects([])) == []


def test_validate_against_observed_aligned():
    result = validate_against_observed(predicted_change=4.0, observed_change=3.0, tolerance_pp=5.0)
    assert result["aligned"] is True
    assert result["gap_pp"] == 1.0


def test_validate_against_observed_diverges():
    result = validate_against_observed(predicted_change=15.0, observed_change=3.0, tolerance_pp=5.0)
    assert result["aligned"] is False
