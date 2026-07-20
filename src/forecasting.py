"""
forecasting.py

Trend regression, event-augmented forecasting, and scenario generation for
the Ethiopia Financial Inclusion forecasting project (10 Academy Week 11
Challenge, Task 4).

Author: Sosina Ayele
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_trend_model(series: pd.DataFrame, date_col: str = "observation_date",
                     value_col: str = "value_numeric"):
    """Fit an OLS trend model of value on years-since-first-observation.

    Args:
        series: Indicator time series (e.g. from get_indicator_series()),
            sorted or unsorted, must have at least 2 rows.
        date_col: Name of the date column.
        value_col: Name of the numeric value column.

    Returns:
        Dict with the fitted statsmodels OLSResults ('model'), the
        reference start date ('start_date'), and the training frame used
        ('data') with an added 'years_since_start' column.

    Raises:
        ValueError: If series has fewer than 2 observations (cannot fit a
            trend line).
    """
    if len(series) < 2:
        raise ValueError(
            f"Need at least 2 observations to fit a trend, got {len(series)}"
        )

    data = series.sort_values(date_col).copy()
    start_date = data[date_col].iloc[0]
    data["years_since_start"] = (data[date_col] - start_date).dt.days / 365.25

    X = sm.add_constant(data["years_since_start"])
    y = data[value_col]
    model = sm.OLS(y, X).fit()

    return {"model": model, "start_date": start_date, "data": data}


def forecast_baseline(fit_result: dict, target_years: list, alpha: float = 0.05) -> pd.DataFrame:
    """Generate baseline (trend-only) forecasts with confidence intervals.

    Args:
        fit_result: Output of fit_trend_model().
        target_years: List of calendar years to forecast (e.g. [2025, 2026, 2027]).
        alpha: Significance level for the confidence interval (default 95% CI).

    Returns:
        DataFrame with columns: year, forecast, ci_lower, ci_upper.

    Raises:
        ValueError: If any target_year predates the training start date.
    """
    model = fit_result["model"]
    start_date = fit_result["start_date"]

    rows = []
    for year in target_years:
        target_date = pd.Timestamp(f"{year}-01-01")
        years_since_start = (target_date - start_date).days / 365.25
        if years_since_start < 0:
            raise ValueError(f"{year} predates the training start date {start_date.date()}")

        X_new = sm.add_constant(pd.DataFrame({"years_since_start": [years_since_start]}),
                                 has_constant="add")
        pred = model.get_prediction(X_new)
        summary = pred.summary_frame(alpha=alpha)

        rows.append({
            "year": year,
            "forecast": summary["mean"].iloc[0],
            "ci_lower": summary["mean_ci_lower"].iloc[0],
            "ci_upper": summary["mean_ci_upper"].iloc[0],
        })

    return pd.DataFrame(rows)


def apply_event_augmentation(baseline_forecast: pd.DataFrame,
                              event_effects_pp: dict) -> pd.DataFrame:
    """Add estimated cumulative event effects (in percentage points) on top
    of a baseline trend forecast.

    Args:
        baseline_forecast: Output of forecast_baseline().
        event_effects_pp: Dict mapping year -> cumulative pp effect to add
            (e.g. {2025: 1.5, 2026: 2.0, 2027: 2.2}).

    Returns:
        Copy of baseline_forecast with an added 'event_effect_pp' and
        'forecast_with_events' column. Years missing from event_effects_pp
        get a 0 effect (with a warning).
    """
    out = baseline_forecast.copy()
    effects = []
    for year in out["year"]:
        if year not in event_effects_pp:
            print(f"Warning: no event effect provided for {year}; assuming 0")
        effects.append(event_effects_pp.get(year, 0.0))

    out["event_effect_pp"] = effects
    out["forecast_with_events"] = out["forecast"] + out["event_effect_pp"]
    return out


def build_scenarios(event_augmented_forecast: pd.DataFrame,
                     optimistic_multiplier: float = 1.5,
                     pessimistic_multiplier: float = 0.5) -> pd.DataFrame:
    """Build optimistic / base / pessimistic scenario forecasts by scaling
    the event-effect component (not the underlying trend, which is held
    fixed across scenarios).

    Args:
        event_augmented_forecast: Output of apply_event_augmentation().
        optimistic_multiplier: Scale applied to event_effect_pp for the
            optimistic scenario (events land faster / larger than estimated).
        pessimistic_multiplier: Scale applied to event_effect_pp for the
            pessimistic scenario (events land slower / smaller).

    Returns:
        DataFrame with year, trend-only baseline, and optimistic/base/
        pessimistic forecast columns.

    Raises:
        ValueError: If optimistic_multiplier < 1 or pessimistic_multiplier > 1
            (scenarios would be inverted / non-sensical).
    """
    if optimistic_multiplier < 1:
        raise ValueError("optimistic_multiplier must be >= 1")
    if pessimistic_multiplier > 1:
        raise ValueError("pessimistic_multiplier must be <= 1")

    out = event_augmented_forecast.copy()
    out["optimistic"] = out["forecast"] + out["event_effect_pp"] * optimistic_multiplier
    out["base"] = out["forecast_with_events"]
    out["pessimistic"] = out["forecast"] + out["event_effect_pp"] * pessimistic_multiplier
    return out[["year", "forecast", "ci_lower", "ci_upper", "optimistic", "base", "pessimistic"]]
