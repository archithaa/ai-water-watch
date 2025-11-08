import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from analysis import load_data, compute_state_summary, compute_monthly_trends, get_sensitivity_data
from google import genai

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
tab1, tab2, tab3, tab4 = st.tabs([
    "State Overview",
    "Trends",
    "Map View",
    "Policy Insights"
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
        st.markdown("#### ⚖️ Water Efficiency vs Stress")
        fig2 = px.scatter(
            summary,
            x="water_per_center",
            y="stress_score",
            size="population_million",
            color="state",
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
            color_discrete_sequence=["green"],
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
        labels={"stress_per_million": "Water Stress per Million People", "state": "State"},
        title="Comparative Water Stress Index (Population-Adjusted)"
    )
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
# GEMINI — POLICY INSIGHTS
# ==============================
def generate_policy_insights(prompt_text):
    try:
        client = genai.Client(
            vertexai=True,
            project="ai-water-watch-477615",
            location="us-central1"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "role": "user",
                "parts": [{"text": prompt_text}]
            }],
            config={"max_output_tokens": 250, "temperature": 0.4}
        )

        return response.text
    except Exception as e:
        return f"⚠️ Gemini API call failed: {e}"

# ==============================
# TAB 4 — POLICY INSIGHTS
# ==============================
with tab4:
    st.header("Executive Summary — AI & Water Sustainability")
    st.write("Generating AI-driven policy insights using Gemini...")

    if not summary.empty:
        top_state = summary.loc[summary['stress_score'].idxmax(), 'state']
        avg_stress = round(summary['stress_score'].mean(), 3)
        avg_water = round(summary['total_water_use_mld'].mean(), 2)
        summary_stats = summary.describe().to_string()

        prompt_text = (
            f"Analyze this dataset on AI data centers and water stress across U.S. states. "
            f"The most stressed state is {top_state}, with an average stress score of {avg_stress}. "
            f"Average total water use across states is {avg_water} MLD.\n\n"
            f"Here are summary statistics:\n{summary_stats}\n\n"
            "Write a concise McKinsey-style executive summary "
            "linking AI infrastructure growth, population pressure, and water sustainability."
        )

        insights = generate_policy_insights(prompt_text)

        if not insights or len(insights.strip()) < 50:
            insights = (
                f"Among the analyzed states, **{top_state}** faces the highest water stress. "
                f"Future AI infrastructure expansion should prioritize water-efficient cooling systems "
                f"and integrate regional hydrological planning. "
                f"Average water use across states ({avg_water} MLD) underscores the need "
                f"for sustainability metrics in digital infrastructure development."
            )

        st.markdown("###  Executive Summary")
        st.write(insights)
    else:
        st.info("Please upload or load data first to generate policy insights.")

# ==============================
# FOOTER
# ==============================
st.caption("Built for: Sustainable Water–Energy Management | © 2025 Architha")
