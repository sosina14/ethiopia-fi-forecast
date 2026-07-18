"""
Basic unit tests for src/analysis.py.

Run with: pytest tests/test_analysis.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from analysis import (
    get_indicator_series,
    get_disaggregated_series,
    compute_growth_rates,
    merge_impact_links_with_events,
    find_unlinked_events,
)


def _sample_obs():
    return pd.DataFrame({
        "indicator_code": ["ACC_OWNERSHIP", "ACC_OWNERSHIP", "ACC_OWNERSHIP", "ACC_OWNERSHIP"],
        "gender": [None, None, "male", "female"],
        "value_numeric": [35, 49, 56, 36],
        "observation_date": pd.to_datetime(["2017-01-01", "2024-01-01", "2022-01-01", "2022-01-01"]),
    })


def test_get_indicator_series_excludes_gender_rows():
    obs = _sample_obs()
    series = get_indicator_series(obs, "ACC_OWNERSHIP")
    assert len(series) == 2
    assert series["gender"].isna().all()


def test_get_indicator_series_empty_warns(capsys):
    obs = _sample_obs()
    series = get_indicator_series(obs, "NOT_A_REAL_CODE")
    assert series.empty
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_get_disaggregated_series():
    obs = _sample_obs()
    series = get_disaggregated_series(obs, "ACC_OWNERSHIP")
    assert len(series) == 2
    assert set(series["gender"]) == {"male", "female"}


def test_compute_growth_rates():
    obs = _sample_obs()
    series = get_indicator_series(obs, "ACC_OWNERSHIP")
    out = compute_growth_rates(series)
    assert "pp_change" in out.columns
    assert out["pp_change"].iloc[0] != out["pp_change"].iloc[0]  # first row is NaN


def test_merge_impact_links_with_events():
    impact_df = pd.DataFrame({"parent_id": ["EVT_0001", "EVT_0002"]})
    events = pd.DataFrame({
        "record_id": ["EVT_0001"],
        "observation_date": pd.to_datetime(["2021-05-17"]),
        "category": ["product_launch"],
        "source_name": ["Ethio Telecom"],
    })
    merged = merge_impact_links_with_events(impact_df, events)
    assert merged.loc[merged["parent_id"] == "EVT_0001", "event_category"].iloc[0] == "product_launch"
    assert merged.loc[merged["parent_id"] == "EVT_0002", "event_id"].isna().iloc[0]


def test_find_unlinked_events():
    impact_df = pd.DataFrame({"parent_id": ["EVT_0001"]})
    events = pd.DataFrame({"record_id": ["EVT_0001", "EVT_0006", "EVT_0009"]})
    unlinked = find_unlinked_events(impact_df, events)
    assert set(unlinked["record_id"]) == {"EVT_0006", "EVT_0009"}
