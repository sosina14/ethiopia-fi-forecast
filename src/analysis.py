"""
analysis.py

Reusable exploratory-analysis and event-impact-linking functions for the
Ethiopia Financial Inclusion forecasting project (10 Academy Week 11 Challenge).

Author: Sosina Ayele
"""
import pandas as pd


def get_indicator_series(obs: pd.DataFrame, indicator_code: str) -> pd.DataFrame:
    """Return a clean, sorted single-indicator national time series.

    Gender-disaggregated rows (where the 'gender' field is populated) are
    excluded so that male/female breakouts of the same survey wave do not
    get plotted alongside the national total and distort the trend line.

    Args:
        obs: Observations DataFrame (record_type == 'observation').
        indicator_code: Indicator code to filter on, e.g. 'ACC_OWNERSHIP'.

    Returns:
        DataFrame sorted by observation_date, national-level rows only.
    """
    if "indicator_code" not in obs.columns:
        raise ValueError("obs DataFrame has no 'indicator_code' column")

    series = obs[
        (obs["indicator_code"] == indicator_code) & (obs["gender"].isna())
    ].sort_values("observation_date")

    if series.empty:
        print(f"Warning: no national-level observations found for {indicator_code}")

    return series


def get_disaggregated_series(obs: pd.DataFrame, indicator_code: str) -> pd.DataFrame:
    """Return gender-disaggregated rows only for a given indicator.

    Args:
        obs: Observations DataFrame.
        indicator_code: Indicator code to filter on.

    Returns:
        DataFrame of rows where 'gender' is populated, sorted by date.
    """
    series = obs[
        (obs["indicator_code"] == indicator_code) & (obs["gender"].notna())
    ].sort_values("observation_date")
    return series


def compute_growth_rates(series: pd.DataFrame, value_col: str = "value_numeric") -> pd.DataFrame:
    """Add period-over-period percentage-point change to an indicator series.

    Args:
        series: Output of get_indicator_series(), sorted by date.
        value_col: Column holding the numeric value.

    Returns:
        Copy of series with an added 'pp_change' column.
    """
    out = series.copy()
    out["pp_change"] = out[value_col].diff()
    return out


def merge_impact_links_with_events(impact_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join impact_link records to their parent event details via parent_id.

    Args:
        impact_df: Impact-link records (must contain 'parent_id').
        events: Event records (record_type == 'event'), must contain
            'record_id', 'observation_date', 'category', 'source_name'.

    Returns:
        Merged DataFrame with event_id / event_date / event_category columns
        added to each impact_link row.
    """
    required = {"record_id", "observation_date", "category", "source_name"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events DataFrame missing columns: {missing}")

    events_slim = events[list(required)].rename(
        columns={
            "record_id": "event_id",
            "observation_date": "event_date",
            "category": "event_category",
        }
    )
    merged = impact_df.merge(events_slim, left_on="parent_id", right_on="event_id", how="left")

    unmatched = merged[merged["event_id"].isna()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} impact_link row(s) have no matching event")

    return merged


def find_unlinked_events(impact_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return events that have no corresponding impact_link record.

    Args:
        impact_df: Impact-link records (must contain 'parent_id').
        events: Event records (must contain 'record_id').

    Returns:
        Subset of events whose record_id does not appear as a parent_id
        in impact_df.
    """
    linked_ids = set(impact_df["parent_id"].dropna())
    return events[~events["record_id"].isin(linked_ids)]


def registered_vs_active_gap(obs: pd.DataFrame,
                              registered_codes: list,
                              active_codes: list) -> dict:
    """Summarize the gap between registered accounts and active usage.

    Args:
        obs: Observations DataFrame.
        registered_codes: Indicator codes representing registered accounts,
            e.g. ['USG_MPESA_USERS', 'USG_TELEBIRR_USERS'].
        active_codes: Indicator codes representing active usage,
            e.g. ['USG_MPESA_ACTIVE', 'USG_ACTIVE_RATE'].

    Returns:
        Dict with 'registered' and 'active' DataFrames for further inspection.
    """
    registered = obs[obs["indicator_code"].isin(registered_codes)].sort_values("observation_date")
    active = obs[obs["indicator_code"].isin(active_codes)].sort_values("observation_date")

    if registered.empty or active.empty:
        print("Warning: registered or active indicator set returned no rows; "
              "check indicator_code spelling against reference_codes.csv")

    return {"registered": registered, "active": active}
