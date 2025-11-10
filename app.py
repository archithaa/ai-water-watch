import json
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from analysis import load_data, compute_state_summary, compute_monthly_trends, get_sensitivity_data
from google import genai
from agent_execute import llm_analyze


import plotly.io as pio

pio.templates.default = "simple_white"

# Consistent color palette
NEUTRAL_PALETTE = [
    "#1f77b4",  # Policy Blue
    "#4C72B0",
    "#6BAED6",
    "#9ECAE1",
    "#C6DBEF"
]


# ---------- Helpers for NL -> operations ---------
import json, re, plotly.express as px

def llm_analyze(q: str, df: pd.DataFrame):
    client = genai.Client(vertexai=True, project="ai-water-watch-477615", location="us-central1")

    # Send compact context to stay within token limits
    sample = df.sample(min(len(df), 30), random_state=42).to_dict(orient="records")
    cols = list(df.columns)

    prompt = f"""
You are a data analysis assistant. The user asked: "{q}"

Columns: {cols}
Sample rows (JSON list): {sample}

Return ONLY JSON (no prose) describing the plot to build, like:
{{
  "chart": "bar|line|scatter",
  "groupby": "state" | null,
  "aggregate": {{"total_water_use_mld": "sum"}} | null,
  "x": "state|month|data_centers|...",
  "y": "total_water_use_mld|water_per_center|...",
  "title": "Human-readable title"
}}
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role":"user","parts":[{"text": prompt}]}]
    )

    # --- Parse JSON safely
    text = (getattr(resp, "output_text", None) or getattr(resp, "text", "") or "").strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        plan = json.loads(text)
    except Exception:
        raise ValueError(f"LLM did not return valid JSON:\n{text}")

    if not plan:
        raise Exception("LLM did not return valid plan.")

    chart_type = plan.get("chart", "bar")
    groupby = plan.get("groupby")
    agg = plan.get("aggregate") or {}
    title = plan.get("title", "Result")

    # derive x/y with sane defaults
    x = plan.get("x")
    y = plan.get("y")

    # compute table `g`
    if agg and isinstance(agg, dict):
        agg_col, agg_func = list(agg.items())[0]
        if groupby:
            g = df.groupby(groupby, as_index=False)[agg_col].agg(agg_func)
        else:
            g = df[[agg_col]].agg(agg_func).reset_index().rename(columns={"index": groupby or "metric", agg_col: agg_col})
            if not x: x = groupby or "metric"
            if not y: y = agg_col
    else:
        # no aggregation → try to use original df (limit rows)
        g = df.copy()
        if not x or not y:
            # fallback to a sensible pair if missing
            x = x or ("month" if "month" in g.columns else "state")
            y = y or ("total_water_use_mld" if "total_water_use_mld" in g.columns else "data_centers")

    # plot
    if chart_type == "bar":
        fig = px.bar(g, x=x, y=y, title=title)
    elif chart_type == "line":
        fig = px.line(g, x=x, y=y, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(g, x=x, y=y, title=title, trendline=None)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    return title, fig


def llm_insight_response(q: str, df: pd.DataFrame) -> str:
    """Ask Gemini for a plain-language executive reasoning answer using the full dataset."""
    client = genai.Client(vertexai=True,
                          project="ai-water-watch-477615",
                          location="us-central1")

    # Condensed dataset so Gemini can reason safely
    summary = df.groupby("state", as_index=False).agg({
        "population_million": "mean",
        "data_centers": "mean",
        "total_water_use_mld": "mean",
        "stress_score": "mean",
        "water_per_center": "mean"
    }).round(3)

    prompt = f"""
You are an environmental & AI infrastructure analyst.

User question:
"{q}"

Use ONLY this data summary:

{summary.to_string(index=False)}

Provide a concise, decision-focused 4–6 sentence insight.
No charts. No filler. No repeating the question.
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role":"user", "parts":[{"text": prompt}]}]
    )

    return resp.text



def ensure_stress(df0: pd.DataFrame) -> pd.DataFrame:
    """Guarantee we have a stress_score column, matching your analysis."""
    df = df0.copy()
    if "stress_score" not in df.columns:
        # Derive stress_score using same logic as analysis.py
        if {"drought_index", "cooling_efficiency", "precipitation_mm"}.issubset(df.columns):
            import numpy as np
            df["stress_score"] = (df["drought_index"] * df["cooling_efficiency"]) / np.log1p(df["precipitation_mm"])
        else:
            df["stress_score"] = pd.NA
    return df

CAUSAL_TRIGGERS = {"why", "cause", "causing", "impact of", "due to", "because", "driver", "drivers", "explain", "reason", "lead to", "leads to"}
PREDICT_TRIGGERS = {"predict", "forecast", "will", "likely", "project", "projection"}
POLICY_TRIGGERS  = {"should", "recommend", "policy", "regulate", "zoning", "mandate"}

def classify_query_for_flags(q: str) -> dict:
    ql = q.lower()
    flags = {
        "causal": any(w in ql for w in CAUSAL_TRIGGERS),
        "predictive": any(w in ql for w in PREDICT_TRIGGERS),
        "policy": any(w in ql for w in POLICY_TRIGGERS),
    }
    flags["needs_guardrail"] = any(flags.values())
    return flags

def explanatory_stub(df, state_col="state"):
    # Simple evidence for “why” questions
    by_state = (df.groupby(state_col, as_index=False)
                  .agg(drought_index=("drought_index","mean"),
                       precip=("precipitation_mm","mean"),
                       cooling=("cooling_efficiency","mean"),
                       stress=("stress_score","mean"))
                  .sort_values("stress", ascending=False))
    msg = (
        "This is an explanatory *hypothesis*, not causal proof. "
        "Evidence shown: higher average drought index and/or lower precipitation can co-occur with higher stress. "
        "Cooling efficiency may moderate stress."
    )
    return msg, by_state

def answer_query_nocode(q: str, df_in: pd.DataFrame):
    import plotly.express as px
    df = ensure_stress(df_in)
    q_low = q.lower().strip()
    steps = []

    flags = classify_query_for_flags(q)

    # --- 0) Causal guardrail (FIXED: now returns)
    if flags["causal"]:
        steps.append("Flagged as causal/explanatory → show evidence with hedged language.")
        msg, table = explanatory_stub(df)
        fig = px.scatter(
            table, x="precip", y="stress", size="drought_index",
            hover_name="state", title="Evidence View: Precipitation vs Stress (Bubble = Drought)"
        )
        return msg, table, fig, "\n".join(steps)

    # 1) Row count
    if ("how many" in q_low and ("rows" in q_low or "entries" in q_low)) or q_low in {"count rows", "number of rows"}:
        steps.append("Counted total rows in the dataset.")
        return f"Total rows: **{len(df)}**", None, None, "\n".join(steps)

    # 1b) TOTAL DATA CENTERS (FIXED SPELLINGS)
    if any(phrase in q_low for phrase in [
        "data center", "data centers", "datacenter",
        "data centre", "data centres"
    ]):
        steps.append("Summed data_centers across dataset.")
        total = df["data_centers"].sum()
        return f"Total data centers across all states: **{int(total):,}**", None, None, "\n".join(steps)

    # 2) Distinct states
    if ("how many" in q_low and "states" in q_low) or q_low == "states?":
        steps.append("Computed distinct states.")
        return f"Distinct states: **{df['state'].nunique()}**", df[["state"]].drop_duplicates().sort_values("state"), None, "\n".join(steps)

    if "list states" in q_low or "which states" in q_low:
        steps.append("Listed unique states.")
        return "States present:", df[["state"]].drop_duplicates().sort_values("state"), None, "\n".join(steps)

    # 3) Stress comparisons
    if ("which state" in q_low and ("highest" in q_low or "max" in q_low) and "stress" in q_low):
        steps.append("Aggregated mean stress_score by state and sorted descending.")
        g = df.groupby("state", as_index=False)["stress_score"].mean().dropna().sort_values("stress_score", ascending=False)
        return "Top stressed states (mean stress_score):", g, px.bar(g, x="state", y="stress_score"), "\n".join(steps)

    if ("lowest" in q_low and "stress" in q_low):
        steps.append("Aggregated mean stress_score by state and sorted ascending.")
        g = df.groupby("state", as_index=False)["stress_score"].mean().dropna().sort_values("stress_score", ascending=True)
        return "Lowest stressed states:", g, px.bar(g, x="state", y="stress_score"), "\n".join(steps)

    # 4) Trends
    if "trend" in q_low or "over time" in q_low or "monthly" in q_low:
        steps.append("Summed total_water_use_mld by month to plot trend.")
        t = df.copy()
        t["month"] = pd.to_datetime(t["month"], errors="coerce")
        g = t.groupby("month", as_index=False)["total_water_use_mld"].sum().dropna()
        return "Monthly total water use:", g, px.line(g, x="month", y="total_water_use_mld", markers=True), "\n".join(steps)

    # 5) Reasoning mode
    if flags["predictive"] or flags["policy"]:
        steps.append("Routed to reasoning mode (causal/policy/predictive detected).")
        return llm_insight_response(q, df), None, None, "\n".join(steps)

    # 6) Gemini Chart Planner
    steps.append("No rule matched → attempting Gemini chart planner")
    try:
        answer, fig = llm_analyze(q, df)
        return answer, None, fig, "\n".join(steps)
    except:
        pass

    # 7) Final text fallback
    cols = ", ".join(df.columns)
    return (
        f"I couldn’t interpret that. Try:\n"
        "- *Which state has highest water stress?*\n"
        "- *Show monthly trend of total water use.*\n\n"
        f"Columns available: {cols}",
        None, None, "\n".join(steps)
    )



# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(page_title="AI Water Watch", layout="wide")

st.title("💧 AI Water Watch — Prototype Dashboard")
st.markdown("""
This prototype explores how **AI data-center expansion intersects with water stress**
in five highly invested U.S. states.  
Upload new data or explore the default dataset.
""")

# ==============================
# UPLOAD / LOAD
# ==============================
uploaded = st.file_uploader("📂 Upload new CSV (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_data()

summary = compute_state_summary(df)
monthly = compute_monthly_trends(df)
sensitivity = get_sensitivity_data(df)

if "water_per_center" not in df.columns:
    df["water_per_center"] = df["total_water_use_mld"] / df["data_centers"]

# ==============================
# ADD LAT/LON IF MISSING
# ==============================
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    coords = {
        "Arizona": [34.0489, -111.0937],
        "California": [36.7783, -119.4179],
        "Nevada": [38.8026, -116.4194],
        "Texas": [31.9686, -99.9018],
        "Utah": [39.3200, -111.0937],
    }
    df["latitude"] = df["state"].map(lambda s: coords.get(s, [37.8, -96])[0])
    df["longitude"] = df["state"].map(lambda s: coords.get(s, [37.8, -96])[1])

# ==============================
# TABS
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "State Overview",
    "Trends",
    "Map View",
    "Policy Insights",
    "Ask Your Data"
])

# ==============================
# TAB 1 — STATE OVERVIEW
# ==============================
with tab1:
    st.subheader("State-Level Overview")

    # KPI Summary
    colA, colB, colC = st.columns(3)
    colA.metric("Total Water Use (MLD)", f"{summary['total_water_use_mld'].sum():,.0f}")
    colB.metric("Avg Stress Score", f"{summary['stress_score'].mean():.2f}")
    colC.metric("Total Population (M)", f"{summary['population_million'].sum():,.1f}")

    st.dataframe(summary, width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("####  Average Water Use per State")
        fig1 = px.bar(
            summary,
            x="state",
            y="total_water_use_mld",
            color="stress_score",
            color_continuous_scale="Blues",
            labels={"total_water_use_mld": "Total Water Use (MLD)", "state": "State"}
        )
        st.plotly_chart(fig1, width='stretch')
        st.caption("Shows total water withdrawal relative to stress levels in each state.")

    with col2:
        st.markdown("#### Water Efficiency vs Stress")
        fig2 = px.scatter(
            summary,
            x="water_per_center",
            y="stress_score",
            size="population_million",
            color="state",
            color_discrete_sequence=NEUTRAL_PALETTE,
            hover_name="state",
            title="Water Efficiency vs Stress"
        )
        st.plotly_chart(fig2, width='stretch')
        st.caption("Larger bubbles indicate states with higher population; ideal zone: lower-right (low stress, high efficiency).")

# ==============================
# TAB 2 — TEMPORAL TRENDS
# ==============================
with tab2:
    st.subheader(" Monthly Trends & Seasonality")

    subtab1, subtab2 = st.tabs(["Trendline", "Water per Center"])

    with subtab1:
        fig3 = px.line(
            monthly,
            x="month",
            y="total_water_use_mld",
            color_discrete_sequence=["#1f77b4"],
            title="Total Water Use Over Time",
            markers=True,
        )
        st.plotly_chart(fig3, width='stretch')
        st.caption("Tracks total water consumption across months to detect seasonal peaks.")

    with subtab2:
        fig4 = px.line(
            monthly,
            x="month",
            y="water_per_center",
            color_discrete_sequence=["#4C72B0"],
            title="Water Use per Center (Trend)"
        )
        st.plotly_chart(fig4, width='stretch')
        st.caption("Shows operational water use per data center over time.")

    # --- Sensitivity Chart ---
    st.markdown("#### Water Stress per Million People")
    sensitivity["stress_per_million"] = sensitivity["stress_score"] / sensitivity["population_million"]
    
    fig5 = px.bar(
        sensitivity,
        x="state",
        y="stress_per_million",
        color="state",
        color_discrete_sequence=NEUTRAL_PALETTE,
        labels={"stress_per_million": "Water Stress per Million People", "state": "State"},
        title="Comparative Water Stress Index (Population-Adjusted)"
    )
    fig5.update_yaxes(range=[0, sensitivity["stress_per_million"].max() * 1.2])
    st.plotly_chart(fig5, width='stretch')
    st.caption("Normalizes stress relative to population, making cross-state comparison clearer.")

# ==============================
# TAB 3 — MAP VIEW
# ==============================
with tab3:
    st.subheader("Interactive Water Stress Map")

    m = folium.Map(location=[37.8, -96], zoom_start=4)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=float(row['stress_score'] * 100),
            popup=f"{row['state']}: Stress {row['stress_score']:.2f}",
            color='red' if row['stress_score'] > 0.1 else 'green',
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    st_folium(m, width=700)
    st.caption("Circle size reflects relative water stress intensity per state.")


# ==============================
# TAB 4 — POLICY INSIGHTS
# ==============================
with tab4:
    st.header("Executive Summary — AI & Water Sustainability")
    st.write("Generate decision-oriented insights summarizing the dataset.")

    # ✅ Instantiate Gemini client here
    client = genai.Client(
        vertexai=True,
        project="ai-water-watch-477615",
        location="us-central1"
    )

    if st.button("Generate Policy Insight", type="primary"):

        summary = df.groupby("state", as_index=False).agg({
            "population_million": "mean",
            "data_centers": "mean",
            "total_water_use_mld": "mean",
            "stress_score": "mean",
            "water_per_center": "mean"
        }).round(3)

        context = summary.to_string(index=False)

        insight_prompt = f"""
        You are an environmental policy analyst.
        Produce a concise decision-grade summary (<=120 words).
        Avoid hype. Avoid speculation. Do not invent numbers.

        Dataset Summary:
        {context}
        """

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": insight_prompt}]}]
            )

            text = getattr(resp, "output_text", None) or getattr(resp, "text", None)
            
            if not text and hasattr(resp, "candidates") and resp.candidates:
                parts = getattrs(resp.candidates[0].content, "parts", [])
                text = "".join([getattr(p, "text", "") for p in parts if hasattr(p, "text")])
            
            text = text.strip() if text else "No insight returned."

            st.markdown("### Executive Summary")
            st.write(text.strip() if text else "No insight returned.")

        except Exception as e:
            st.error(f"⚠️ Policy insight failed: {e}")

    else:
        st.info("Click **Generate Policy Insight** to analyze the data.")




# ==============================
# TAB 5 — ASK YOUR DATA (Conversational analytics)
# ==============================
with tab5:
    st.subheader("Ask Your Data")
    st.caption("Type a question. I’ll run the operation and show a chart/table when possible.")

    q = st.text_input("Your question", placeholder="e.g., Which state has highest water stress?")
    if st.button("Run", type="primary"):
        if not df.empty:
            msg, table, fig, steps = answer_query_nocode(q, df)
            st.markdown(f"**Answer:** {msg}")
            if table is not None and not table.empty:
                st.dataframe(table, width='stretch')
            if fig is not None:
                st.plotly_chart(fig, width='stretch', key=f"ask_chart_{hash(q)}")
            with st.expander("Show steps (how I interpreted your question)"):
                st.code(steps)
        else:
            st.info("Please upload or load data first.")


# ==============================
# FOOTER
# ==============================
st.caption("Built for: Sustainable Water–Energy Management | © 2025 Architha")
