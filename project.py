import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List

# --------------------------------------------------
# 🚀 CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="🌍 Travel Cities Explorer (Advanced)",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 🎨 GLOBAL THEME / CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;600;800&display=swap');

    html, body, [class*="st"]  {
        font-family: 'Nunito', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #e1f5fe 100%);
    }

    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div:first-child {
        background-color: #f1f8e9;
        border-right: 1px solid #c5e1a5;
    }

    h1, h2, h3, h4 {
        color: #004d40;
    }

    .stDataFrame thead tr th {
        background-color: #00695c;
        color: white;
    }

    button[kind="secondary"], div[data-testid="stDownloadButton"] > button {
        background-color: #00897b !important;
        color: white !important;
        border-radius: 8px;
    }

    .stDataFrame tbody tr:hover {
        background-color: rgba(0, 105, 92, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 🗂️ DATA LOADING & PREP
# --------------------------------------------------
DATA_PATH = "Worldwide Travel Cities Dataset (Ratings and Climate).csv"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and compute derived fields."""
    df = pd.read_csv(path)

    # Category columns contributing to the overall score
    category_cols: List[str] = [
        "culture", "adventure", "nature", "beaches", "nightlife",
        "cuisine", "wellness", "urban", "seclusion"
    ]

    # Ensure numeric & handle missing values for category columns
    df[category_cols] = df[category_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ----- Compute overall score (0‑10 scale assumed) -----
    df["overall_score"] = df[category_cols].mean(axis=1).round(2)

    # ----- Parse avg_temp_monthly JSON and compute yearly average -----
    def extract_yearly_avg(temp_json_str: str):
        if pd.isna(temp_json_str):
            return np.nan
        try:
            monthly = json.loads(temp_json_str)
            avg_vals = [m.get("avg", np.nan) for m in monthly.values()]
            avg_vals = [v for v in avg_vals if pd.notnull(v)]
            return round(float(np.mean(avg_vals)), 2) if avg_vals else np.nan
        except (ValueError, TypeError):
            return np.nan

    df["AvgTempC"] = df["avg_temp_monthly"].apply(extract_yearly_avg)

    # Clean column names for UI
    df.rename(columns={
        "city": "City",
        "country": "Country",
        "region": "Region",
    }, inplace=True)

    return df

# Load & prepare dataset
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("CSV file not found. Please ensure it's in the same folder as the app.")
    st.stop()

# --------------------------------------------------
# 🖼️ HEADER
# --------------------------------------------------
st.title("🧳 Travel Cities Explorer")
st.markdown(
    "<span style='color:#00695c;'>Discover global destinations using experience scores, climate data, and more.</span>",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 🔍 SIDEBAR FILTERS
# --------------------------------------------------
with st.sidebar:
    st.header("🔍 Filters")

    search_term = st.text_input("Search City or Country")

    regions = ["All"] + sorted(df["Region"].dropna().unique())
    region_filter = st.selectbox("Region", regions, index=0)

    # Score slider
    score_min, score_max = float(df["overall_score"].min()), float(df["overall_score"].max())
    score_range = st.slider("Overall Score", score_min, score_max, (score_min, score_max))

    # Temperature slider (handle NaNs safely)
    temp_col = df["AvgTempC"].dropna()
    if temp_col.empty:
        temp_min, temp_max = 0.0, 0.0
    else:
        temp_min, temp_max = float(temp_col.min()), float(temp_col.max())
    temp_range = st.slider("Average Temperature (°C)", temp_min, temp_max, (temp_min, temp_max))

    top_n = st.slider("Max rows to display", 10, 1000, 100, step=10)

# --------------------------------------------------
# 🔎 DATA FILTERING FUNCTION
# --------------------------------------------------

def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    df_f = data.copy()

    if search_term:
        mask = (
            df_f["City"].str.contains(search_term, case=False, na=False) |
            df_f["Country"].str.contains(search_term, case=False, na=False)
        )
        df_f = df_f[mask]

    if region_filter != "All":
        df_f = df_f[df_f["Region"] == region_filter]

    df_f = df_f[
        (df_f["overall_score"].between(score_range[0], score_range[1])) &
        (df_f["AvgTempC"].between(temp_range[0], temp_range[1]))
    ]

    return df_f.head(top_n)

filtered_df = apply_filters(df)

# --------------------------------------------------
# 📑 MAIN TABS
# --------------------------------------------------
explorer_tab, insights_tab = st.tabs(["🌆 Explorer", "📊 Insights"])

# --------------------------------------------------
# 🌆 EXPLORER TAB
# --------------------------------------------------
with explorer_tab:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"🏙️ {len(filtered_df):,} city/cities shown (max {top_n})")
    with col2:
        @st.cache_data
        def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
            return df_in.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            data=to_csv_bytes(filtered_df),
            file_name="filtered_cities.csv",
            mime="text/csv",
        )

    st.dataframe(filtered_df, use_container_width=True, height=420)

    # Detailed City View
    if not filtered_df.empty:
        st.markdown("---")
        st.subheader("🔎 City Details")
        selected_city = st.selectbox("Select a city for details", filtered_df["City"].unique())
        detail = filtered_df[filtered_df["City"] == selected_city].iloc[0]

        cat_cols = [
            "culture", "adventure", "nature", "beaches", "nightlife",
            "cuisine", "wellness", "urban", "seclusion"
        ]

        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"### 🏙️ {detail['City']}, {detail['Country']}")
            st.markdown(f"**🌍 Region:** {detail['Region']}")
            st.markdown(f"**⭐ Overall Score:** {detail['overall_score']}")
            st.markdown(f"**🌡️ Avg Temp (°C):** {detail['AvgTempC'] if pd.notnull(detail['AvgTempC']) else 'N/A'}")
            st.markdown(f"**📝 Description:** {detail['short_description'] if pd.notnull(detail['short_description']) else 'N/A'}")

        with right:
            # Radar-like plot (polar)
            angles = np.linspace(0, 2 * np.pi, len(cat_cols), endpoint=False).tolist()
            scores = detail[cat_cols].tolist()
            scores += scores[:1]
            angles += angles[:1]

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, scores)
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(cat_cols, fontsize=8)
            ax.set_title("Experience Profile", pad=15)
            ax.set_yticklabels([])
            st.pyplot(fig)
    else:
        st.info("No cities match your filters.")

# --------------------------------------------------
# 📊 INSIGHTS TAB
# --------------------------------------------------
with insights_tab:
    st.subheader("📊 Dataset Insights")

    tot = len(df)
    avg_score = df["overall_score"].mean()
    avg_temp = df["AvgTempC"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Cities", f"{tot}")
    m2.metric("Avg Score", f"{avg_score:.2f}")
    m3.metric("Avg Temp (°C)", f"{avg_temp:.1f}")

    st.markdown("---")

    # Score distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(df["overall_score"], bins=20, edgecolor="black")
    ax1.set_xlabel("Overall Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution")
    st.pyplot(fig1)

    # Score vs temperature scatter
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(df["AvgTempC"], df["overall_score"], alpha=0.6)
    ax2.set_xlabel("Avg Temp (°C)")
    ax2.set_ylabel("Overall Score")
    ax2.set_title("Score vs Avg Temp")
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### 🏆 Top 10 Cities by Overall Score")
    st.table(df.sort_values("overall_score", ascending=False).head(10)[["City", "Country", "Region", "overall_score"]])

# --------------------------------------------------
# 📢 FOOTER
# --------------------------------------------------
st.markdown(
    """
    ---
    <div style='text-align:center;'>
        Crafted with ❤️ by <strong>Shubhankar Pal</strong> • Travel Cities Explorer 🌐
    </div>
    """,
    unsafe_allow_html=True,
)
