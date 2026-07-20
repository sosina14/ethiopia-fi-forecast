"""
impact_model.py

Event-impact modeling utilities for the Ethiopia Financial Inclusion
forecasting project (10 Academy Week 11 Challenge, Task 3).

Author: Sosina Ayele
"""
from datetime import datetime

import numpy as np
import pandas as pd

# Ordinal encoding for qualitative magnitude/direction labels used in the
# starter dataset's impact_link records.
MAGNITUDE_SCALE = {"low": 1, "medium": 2, "high": 3}
DIRECTION_SIGN = {"increase": 1, "decrease": -1}


def encode_impact_score(impact_df: pd.DataFrame) -> pd.DataFrame:
    """Convert qualitative impact_direction/impact_magnitude into a signed
    numeric score usable for aggregation and visualization.

    Args:
        impact_df: impact_link records with 'impact_direction' and
            'impact_magnitude' columns.

    Returns:
        Copy of impact_df with an added 'impact_score' column in
        [-3, -2, -1, 1, 2, 3]. Rows with unrecognized labels get NaN
        and a warning is printed (rather than silently dropped).
    """
    out = impact_df.copy()

    mag = out["impact_magnitude"].map(MAGNITUDE_SCALE)
    sign = out["impact_direction"].map(DIRECTION_SIGN)
    out["impact_score"] = mag * sign

    n_bad = out["impact_score"].isna().sum()
    if n_bad:
        print(f"Warning: {n_bad} impact_link row(s) had unrecognized "
              f"impact_direction/impact_magnitude values and were scored NaN")

    return out


def build_association_matrix(impact_df: pd.DataFrame) -> pd.DataFrame:
    """Build the event x indicator association matrix.

    Args:
        impact_df: impact_link records, must already have 'impact_score'
            (see encode_impact_score) and 'parent_id', 'related_indicator'.

    Returns:
        DataFrame with events (parent_id) as rows, indicator_codes
        (related_indicator) as columns, and impact_score as values.
        Missing event-indicator pairs are left as NaN (no claimed effect),
        not zero, to distinguish "no evidence" from "evidence of no effect".
    """
    required = {"parent_id", "related_indicator", "impact_score"}
    missing = required - set(impact_df.columns)
    if missing:
        raise ValueError(f"impact_df missing required columns: {missing}. "
                          f"Run encode_impact_score() first.")

    matrix = impact_df.pivot_table(
        index="parent_id", columns="related_indicator",
        values="impact_score", aggfunc="mean"
    )
    return matrix


def ramp_effect(event_date, lag_months: float, magnitude: float,
                 as_of_dates: pd.DatetimeIndex) -> np.ndarray:
    """Model an event's effect as a linear ramp from 0 to full magnitude
    over `lag_months`, then held constant.

    Args:
        event_date: The event's date (str or Timestamp).
        lag_months: Months to reach full effect.
        magnitude: Full-strength signed effect (e.g. from impact_score).
        as_of_dates: Dates at which to evaluate the effect.

    Returns:
        Array of effect values aligned to as_of_dates, 0 before the event
        and ramping linearly to `magnitude` over `lag_months`.

    Raises:
        ValueError: If lag_months is negative.
    """
    if lag_months < 0:
        raise ValueError("lag_months must be non-negative")

    event_date = pd.Timestamp(event_date)
    months_since = (as_of_dates - event_date).days / 30.44

    effect = np.where(
        months_since <= 0, 0.0,
        np.where(
            lag_months == 0, magnitude,
            np.clip(months_since / max(lag_months, 1e-9), 0, 1) * magnitude
        )
    )
    return effect


def combine_effects(effects: list) -> np.ndarray:
    """Combine multiple event effect arrays (same length, same dates) into
    a single net effect by simple summation.

    Args:
        effects: List of numpy arrays (one per event) of equal length.

    Returns:
        Element-wise sum. Returns an empty array if `effects` is empty.
    """
    if not effects:
        return np.array([])
    return np.sum(np.vstack(effects), axis=0)


def validate_against_observed(predicted_change: float, observed_change: float,
                               tolerance_pp: float = 5.0) -> dict:
    """Compare a modeled cumulative effect against an observed pp change.

    Args:
        predicted_change: Model's estimated cumulative effect (pp).
        observed_change: Actual observed change in the indicator (pp).
        tolerance_pp: Acceptable absolute gap before flagging a divergence.

    Returns:
        Dict with predicted, observed, gap, and a boolean 'aligned' flag.
    """
    gap = predicted_change - observed_change
    return {
        "predicted_pp": predicted_change,
        "observed_pp": observed_change,
        "gap_pp": gap,
        "aligned": abs(gap) <= tolerance_pp,
    }
