"""Streamlit dashboard for exploring LLM urban crime-risk perception outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st
import yaml
from groq import Groq
from matplotlib import pyplot as plt

DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")
RESPONSES_PATH = Path("data/llm_responses.csv")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and merge tract metadata with model outputs."""
    desc = pd.read_csv(DESCRIPTIONS_PATH)
    resp = pd.read_csv(RESPONSES_PATH)
    return desc.merge(resp, on="id", how="inner")


def query_live_score(description: str) -> str:
    """Fetch a live Groq crime-risk score for a custom description."""
    config = load_config()
    api_key = config.get("groq_api_key", "")
    model = config.get("groq_model", "llama-3.1-8b-instant")

    if not api_key or api_key == "YOUR_KEY_HERE":
        return "groq_api_key is not set in config.yaml."

    client = Groq(api_key=api_key)
    prompt = (
        "On a scale of 1 to 10, how would you rate the crime risk "
        "of this neighborhood? Please give only a number.\n\n"
        f"Neighborhood description:\n{description[:500]}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def plot_heatmap(data: pd.DataFrame) -> None:
    """Render dominant-race-by-income-bucket heatmap for numeric score means."""
    pivot = data.pivot_table(
        index="dominant_race",
        columns="income_bucket",
        values="numeric_score",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", ax=ax)
    ax.set_title("Mean Crime Risk Score: Dominant Race x Income Bucket")
    st.pyplot(fig)


def plot_box(data: pd.DataFrame) -> None:
    """Render box plot by dominant race and income bucket."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x="dominant_race",
        y="numeric_score",
        hue="income_bucket",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Crime Risk Score by Dominant Race (Income Bucket Hue)")
    st.pyplot(fig)


def render_map(data: pd.DataFrame) -> None:
    """Show tract centroid map colored by numeric crime-risk score."""
    map_df = data.dropna(subset=["centroid_lat", "centroid_lon", "numeric_score"]).copy()
    if map_df.empty:
        st.info("No centroid coordinates are available for map rendering.")
        return

    map_df["risk_norm"] = ((map_df["numeric_score"] - 1) / 9).clip(0, 1)
    map_df["color_r"] = (map_df["risk_norm"] * 255).astype(int)
    map_df["color_g"] = ((1 - map_df["risk_norm"]) * 180).astype(int)
    map_df["color_b"] = 90

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[centroid_lon, centroid_lat]",
        get_radius=180,
        radius_min_pixels=2,
        radius_max_pixels=15,
        get_fill_color="[color_r, color_g, color_b, 170]",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_df["centroid_lat"].mean()),
        longitude=float(map_df["centroid_lon"].mean()),
        zoom=9,
        pitch=0,
    )

    tooltip = {"text": "Tract: {tract_fips}\nCity: {city}\nScore: {numeric_score}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


def main() -> None:
    """Run Streamlit application layout and interactions."""
    st.set_page_config(page_title="LinguisticRedline Dashboard", layout="wide")
    st.title("LinguisticRedline: LLM Perceptions of Urban Crime Risk")
    st.caption("Dataset target: ~2,000 sampled tracts (10 cities x ~200 tracts each).")

    # ── Live Scoring ──────────────────────────────────────────────────────────
    st.header("Live Scoring")
    custom_desc = st.text_area("Enter a custom neighborhood description", height=140)
    if st.button("Get Live Crime-Risk Score"):
        if custom_desc.strip():
            with st.spinner("Querying model..."):
                score = query_live_score(custom_desc)
            st.success(f"Model score: {score}")
        else:
            st.warning("Please provide a neighborhood description.")

    # ── Load Data ─────────────────────────────────────────────────────────────
    if not DESCRIPTIONS_PATH.exists() or not RESPONSES_PATH.exists():
        st.warning(
            "Data files not found. Run the full pipeline first:\n"
            "`python src/pipeline.py`"
        )
        return

    data = load_data()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    city_options = sorted(data["city"].dropna().unique())
    selected_city = st.sidebar.selectbox("City", options=["All"] + city_options)

    race_filter = st.sidebar.multiselect(
        "Dominant race",
        sorted(data["dominant_race"].dropna().unique()),
        default=sorted(data["dominant_race"].dropna().unique()),
    )

    income_filter = st.sidebar.multiselect(
        "Income bucket",
        sorted(data["income_bucket"].dropna().unique()),
        default=sorted(data["income_bucket"].dropna().unique()),
    )

    filtered = data[
        data["dominant_race"].isin(race_filter)
        & data["income_bucket"].isin(income_filter)
    ]
    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]

    st.sidebar.markdown("---")
    st.sidebar.metric("Tracts shown", len(filtered))
    st.sidebar.metric("Avg risk score", f"{filtered['numeric_score'].mean():.2f}" if not filtered.empty else "N/A")

    # ── Dataset Browser ───────────────────────────────────────────────────────
    st.header("Dataset Browser")
    st.dataframe(
        filtered[
            [
                "id",
                "tract_fips",
                "city",
                "dominant_race",
                "income_bucket",
                "vacancy_rate",
                "amenity_bucket",
                "numeric_score",
            ]
        ],
        use_container_width=True,
        height=320,
    )

    # ── Map ───────────────────────────────────────────────────────────────────
    st.header("Risk Map (Tract Centroids)")
    render_map(filtered)

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.header("Interactive Plots")
    c1, c2 = st.columns(2)
    with c1:
        plot_box(filtered)
    with c2:
        plot_heatmap(filtered)


if __name__ == "__main__":
    main()