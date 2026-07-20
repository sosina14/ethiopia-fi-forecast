"""
Basic unit tests for src/forecasting.py.

Run with: pytest tests/test_forecasting.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from forecasting import (
    fit_trend_model,
    forecast_baseline,
    apply_event_augmentation,
    build_scenarios,
)


def _sample_series():
    return pd.DataFrame({
        "observation_date": pd.to_datetime(["2014-01-01", "2017-01-01", "2021-01-01", "2024-01-01"]),
        "value_numeric": [22, 35, 46, 49],
    })


def test_fit_trend_model_too_few_points():
    series = pd.DataFrame({
        "observation_date": pd.to_datetime(["2024-01-01"]),
        "value_numeric": [49],
    })
    with pytest.raises(ValueError):
        fit_trend_model(series)


def test_fit_trend_model_fits():
    fit = fit_trend_model(_sample_series())
    assert "model" in fit
    assert fit["model"].params["years_since_start"] > 0  # upward trend


def test_forecast_baseline_returns_expected_years():
    fit = fit_trend_model(_sample_series())
    forecast = forecast_baseline(fit, [2025, 2026, 2027])
    assert list(forecast["year"]) == [2025, 2026, 2027]
    assert (forecast["ci_lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["ci_upper"]).all()


def test_forecast_baseline_rejects_predate():
    fit = fit_trend_model(_sample_series())
    with pytest.raises(ValueError):
        forecast_baseline(fit, [2010])


def test_apply_event_augmentation():
    baseline = pd.DataFrame({"year": [2025, 2026], "forecast": [50.0, 51.0],
                              "ci_lower": [48, 49], "ci_upper": [52, 53]})
    out = apply_event_augmentation(baseline, {2025: 1.0, 2026: 2.0})
    assert list(out["forecast_with_events"]) == [51.0, 53.0]


def test_apply_event_augmentation_missing_year_warns(capsys):
    baseline = pd.DataFrame({"year": [2025], "forecast": [50.0], "ci_lower": [48], "ci_upper": [52]})
    out = apply_event_augmentation(baseline, {})
    assert out["event_effect_pp"].iloc[0] == 0.0
    assert "Warning" in capsys.readouterr().out


def test_build_scenarios_ordering():
    aug = pd.DataFrame({
        "year": [2025], "forecast": [50.0], "ci_lower": [48], "ci_upper": [52],
        "event_effect_pp": [2.0], "forecast_with_events": [52.0],
    })
    scenarios = build_scenarios(aug)
    row = scenarios.iloc[0]
    assert row["pessimistic"] <= row["base"] <= row["optimistic"]


def test_build_scenarios_invalid_multipliers():
    aug = pd.DataFrame({
        "year": [2025], "forecast": [50.0], "ci_lower": [48], "ci_upper": [52],
        "event_effect_pp": [2.0], "forecast_with_events": [52.0],
    })
    with pytest.raises(ValueError):
        build_scenarios(aug, optimistic_multiplier=0.8)
    with pytest.raises(ValueError):
        build_scenarios(aug, pessimistic_multiplier=1.2)
