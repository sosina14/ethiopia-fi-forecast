"""
dashboard/app.py

Interactive Streamlit dashboard for the Ethiopia Financial Inclusion
Forecasting project (10 Academy Week 11 Challenge, Task 5).

Author: Sosina Ayele

Run locally with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import load_unified_data, split_by_record_type
from analysis import get_indicator_series, get_disaggregated_series, merge_impact_links_with_events

st.set_page_config(page_title="Ethiopia Financial Inclusion Dashboard",
                    layout="wide", page_icon="🇪🇹")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

NAVY = "#0B2545"
TEAL = "#1B9AAA"


# ---------------------------------------------------------------------
# Data loading (cached, with graceful error handling)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the enriched dataset and split into observations/events/targets.

    Returns:
        Tuple of (df, obs, events, targets, impact_df) or (None, ...) on
        failure, with the error surfaced via st.error rather than a crash.
    """
    try:
        df = load_unified_data(str(DATA_DIR / "ethiopia_fi_enriched.csv"))
        parts = split_by_record_type(df)
        impact_path = DATA_DIR / "impact_sheet_enriched.csv"
        impact_df = pd.read_csv(impact_path) if impact_path.exists() else pd.DataFrame()
        return df, parts["observations"], parts["events"], parts["targets"], impact_df
    except FileNotFoundError as e:
        st.error(f"Could not load data: {e}. Run notebooks/02_eda.ipynb first "
                  f"to generate data/processed/ files.")
        return None, None, None, None, None
    except ValueError as e:
        st.error(f"Data validation error: {e}")
        return None, None, None, None, None


@st.cache_data
def load_forecast():
    """Load the Task 4 forecast summary, if available.

    Returns:
        DataFrame or None if the file doesn't exist yet.
    """
    path = REPORTS_DIR / "forecast_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


df, obs, events, targets, impact_df = load_data()
forecast_df = load_forecast()

if df is None:
    st.stop()

obs = obs.copy()
obs["year"] = obs["observation_date"].dt.year

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("🇪🇹 Ethiopia FI Dashboard")
page = st.sidebar.radio("Navigate to:",
                         ["Overview", "Trends", "Forecasts", "Inclusion Projections"])
st.sidebar.markdown("---")
st.sidebar.caption("Data: unified Findex-based dataset, enriched by Sosina Ayele. "
                    "10 Academy Week 11 Challenge.")


# ---------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------
def render_overview():
    st.title("Overview")
    st.caption("Key metrics summary for Ethiopia's financial inclusion trajectory")

    acc = get_indicator_series(obs, "ACC_OWNERSHIP")
    mm = get_indicator_series(obs, "ACC_MM_ACCOUNT")
    gap = get_indicator_series(obs, "GEN_GAP_ACC")
    p2p = obs[obs["indicator_code"] == "USG_P2P_COUNT"].sort_values("observation_date")
    atm = obs[obs["indicator_code"] == "USG_ATM_COUNT"].sort_values("observation_date")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest = acc.iloc[-1] if not acc.empty else None
        prev = acc.iloc[-2] if len(acc) >= 2 else None
        delta = f"{latest['value_numeric'] - prev['value_numeric']:.1f}pp" if latest is not None and prev is not None else None
        st.metric("Account Ownership", f"{latest['value_numeric']:.0f}%" if latest is not None else "N/A", delta)
    with col2:
        latest = mm.iloc[-1] if not mm.empty else None
        st.metric("Mobile Money Accounts", f"{latest['value_numeric']:.1f}%" if latest is not None else "N/A")
    with col3:
        latest = gap.iloc[-1] if not gap.empty else None
        st.metric("Gender Gap (Access)", f"{latest['value_numeric']:.0f}pp" if latest is not None else "N/A",
                   help="NFIS-II target: 10pp by 2025")
    with col4:
        st.metric("Events Cataloged", len(events))

    st.markdown("### P2P / ATM Crossover")
    if not p2p.empty and not atm.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p2p["observation_date"], y=p2p["value_numeric"],
                                  name="P2P Transactions", line=dict(color=TEAL)))
        fig.add_trace(go.Scatter(x=atm["observation_date"], y=atm["value_numeric"],
                                  name="ATM Cash Withdrawals", line=dict(color=NAVY)))
        fig.update_layout(title="P2P vs. ATM Transaction Volume", height=350)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("P2P/ATM crossover data not available in the current dataset.")

    st.markdown("### Growth Rate Highlights")
    if len(acc) >= 2:
        acc_growth = acc.copy()
        acc_growth["pp_change"] = acc_growth["value_numeric"].diff()
        fig = px.bar(acc_growth.dropna(subset=["pp_change"]), x="year", y="pp_change",
                     title="Account Ownership: pp Change Between Survey Waves",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------
# Trends page
# ---------------------------------------------------------------------
def render_trends():
    st.title("Trends")
    st.caption("Explore indicator time series with event overlays")

    indicator_options = sorted(obs["indicator_code"].dropna().unique())
    default_idx = indicator_options.index("ACC_OWNERSHIP") if "ACC_OWNERSHIP" in indicator_options else 0
    selected = st.selectbox("Choose an indicator:", indicator_options, index=default_idx)

    min_year, max_year = int(obs["year"].min()), int(obs["year"].max())
    year_range = st.slider("Date range", min_year, max_year, (min_year, max_year))

    channel = st.radio("View:", ["National", "By gender (if available)"], horizontal=True)

    if channel == "National":
        series = get_indicator_series(obs, selected)
    else:
        series = get_disaggregated_series(obs, selected)

    series = series[(series["year"] >= year_range[0]) & (series["year"] <= year_range[1])]

    if series.empty:
        st.warning(f"No data for {selected} in the selected range/view.")
        return

    if channel == "By gender (if available)" and "gender" in series.columns:
        fig = px.line(series, x="observation_date", y="value_numeric", color="gender",
                       markers=True, title=f"{selected} — by gender")
    else:
        fig = px.line(series, x="observation_date", y="value_numeric", markers=True,
                       title=selected, color_discrete_sequence=[TEAL])

    for _, e in events.iterrows():
        if year_range[0] <= e["observation_date"].year <= year_range[1]:
            fig.add_vline(x=e["observation_date"].timestamp() * 1000, line_dash="dash",
                           line_color="gray", opacity=0.5)

    fig.update_layout(height=450)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Data Table")
    st.dataframe(series[["observation_date", "indicator_code", "value_numeric", "source_name", "confidence"]])
    st.download_button("Download this data as CSV", series.to_csv(index=False),
                        file_name=f"{selected}_{year_range[0]}_{year_range[1]}.csv")


# ---------------------------------------------------------------------
# Forecasts page
# ---------------------------------------------------------------------
def render_forecasts():
    st.title("Forecasts")
    st.caption("2025-2027 projections with confidence intervals (Task 4)")

    if forecast_df is None:
        st.warning("No forecast data found. Run notebooks/04_forecasting.ipynb first to generate "
                    "reports/forecast_summary.csv.")
        return

    available_indicators = forecast_df["indicator"].unique()
    selected_indicator = st.selectbox("Indicator:", available_indicators)
    model_view = st.radio("Model:", ["Trend-only (baseline)", "Event-augmented"], horizontal=True)

    sub = forecast_df[forecast_df["indicator"] == selected_indicator]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["year"], y=sub["ci_upper"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=sub["year"], y=sub["ci_lower"], fill="tonexty", line=dict(width=0),
                              name="95% CI", fillcolor="rgba(150,150,150,0.25)"))

    y_col = "forecast" if model_view == "Trend-only (baseline)" else "base"
    fig.add_trace(go.Scatter(x=sub["year"], y=sub[y_col], mode="lines+markers",
                              name=model_view, line=dict(color=TEAL, width=3)))

    fig.update_layout(title=f"{selected_indicator} Forecast, 2025-2027", height=450,
                       yaxis_title="% of adults")
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Key Projected Milestones")
    for _, row in sub.iterrows():
        st.write(f"**{int(row['year'])}:** {row[y_col]:.1f}% "
                 f"(95% CI: {row['ci_lower']:.1f}%–{row['ci_upper']:.1f}%)")

    st.dataframe(sub.round(1))


# ---------------------------------------------------------------------
# Inclusion Projections page
# ---------------------------------------------------------------------
def render_projections():
    st.title("Inclusion Projections")
    st.caption("Progress toward NFIS-II policy targets")

    if forecast_df is None:
        st.warning("No forecast data found. Run notebooks/04_forecasting.ipynb first.")
        return

    scenario = st.select_slider("Scenario:", options=["Pessimistic", "Base", "Optimistic"], value="Base")
    scenario_col = scenario.lower()

    acc_forecast = forecast_df[forecast_df["indicator"] == "ACC_OWNERSHIP"]

    if not acc_forecast.empty:
        fig = go.Figure()
        acc_hist = get_indicator_series(obs, "ACC_OWNERSHIP")
        fig.add_trace(go.Scatter(x=acc_hist["year"], y=acc_hist["value_numeric"],
                                  mode="lines+markers", name="Observed", line=dict(color=NAVY)))
        fig.add_trace(go.Scatter(x=acc_forecast["year"], y=acc_forecast[scenario_col],
                                  mode="lines+markers", name=f"{scenario} forecast", line=dict(color=TEAL, dash="dash")))
        fig.add_hline(y=60, line_dash="dot", line_color="darkred",
                      annotation_text="60% inclusion goal")
        fig.update_layout(title="Progress Toward 60% Account Ownership Goal", height=450)
        st.plotly_chart(fig, width='stretch')

        final_year_val = acc_forecast[acc_forecast["year"] == acc_forecast["year"].max()][scenario_col].iloc[0]
        gap_to_goal = 60 - final_year_val
        st.metric(f"Gap to 60% goal by {int(acc_forecast['year'].max())}", f"{gap_to_goal:.1f}pp remaining")

    st.markdown("### Answers to the Consortium's Key Questions")
    with st.expander("What drives financial inclusion in Ethiopia?", expanded=True):
        st.write("Product launches (Telebirr, M-Pesa) drive Usage indicators quickly (3-6 month lags); "
                 "policy and infrastructure (NFIS-II, Fayda digital ID) drive Access and Gender outcomes "
                 "slowly (24-48 month lags). See the Event-Impact Narrative in the Task 2/3 notebooks.")
    with st.expander("How do events affect inclusion outcomes?"):
        if not impact_df.empty:
            merged = merge_impact_links_with_events(impact_df, events)
            st.dataframe(merged[["parent_id", "event_category", "related_indicator",
                                  "impact_direction", "impact_magnitude", "lag_months"]])
    with st.expander("What does 2025-2027 look like?"):
        st.write("See the Forecasts page. Base-case Access growth remains below the 60% goal through 2027; "
                 "the gender gap (15pp in 2024) is not on pace to hit the NFIS-II 10pp target by 2025.")


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
if page == "Overview":
    render_overview()
elif page == "Trends":
    render_trends()
elif page == "Forecasts":
    render_forecasts()
elif page == "Inclusion Projections":
    render_projections()
