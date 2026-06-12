import streamlit as st
import pandas as pd
import plotly.express as px
import json

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Candidate Ranking Dashboard",
    layout="wide"
)

st.title("AI-Powered Candidate Ranking Dashboard")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

df = pd.read_csv(
    "data/semantic_submission_full.csv"
)

with open(
    "data/semantic_submission_full_summaries.json",
    "r",
    encoding="utf-8"
) as f:
    summaries = json.load(f)

with open(
    "data/job_requirements.json",
    "r",
    encoding="utf-8"
) as f:
    requirements = json.load(f)

# ---------------------------------------------------
# Top Metrics
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Candidates Processed",
        "100,000"
    )

with col2:
    st.metric(
        "Top Ranked Candidates",
        len(df)
    )

with col3:
    st.metric(
        "Best Score",
        round(df["score"].max(), 4)
    )

# ---------------------------------------------------
# Top Candidate Card
# ---------------------------------------------------

top = df.iloc[0]

st.success(
    f"""
🏆 Top Recommended Candidate

Candidate ID: {top['candidate_id']}

Rank: {top['rank']}

Score: {top['score']}

Reason:
{top['reasoning']}
"""
)

# ---------------------------------------------------
# Download Button
# ---------------------------------------------------

st.download_button(
    label="📥 Download Ranked CSV",
    data=open(
        "data/semantic_submission_full.csv",
        "rb"
    ),
    file_name="semantic_submission.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("AI Talent Intelligence")

st.sidebar.metric(
    "Candidates Processed",
    "100,000"
)

st.sidebar.metric(
    "Top Recommendations",
    len(df)
)

# ---------------------------------------------------
# Job Requirements
# ---------------------------------------------------

st.subheader("Job Summary")

st.info("""
Role: Senior AI Engineer

Focus Areas:
• Machine Learning
• NLP
• Recommendation Systems
• Search & Ranking
• Python
• Deep Learning
""")

# ---------------------------------------------------
# Search Candidate
# ---------------------------------------------------

st.subheader("Search Candidate")

search = st.text_input(
    "Enter Candidate ID"
)

if search:
    result = df[
        df["candidate_id"]
        .astype(str)
        .str.contains(
            search,
            case=False
        )
    ]

    st.dataframe(result)

# ---------------------------------------------------
# Filter Candidates
# ---------------------------------------------------

st.subheader("Filter Candidates")

score = st.slider(
    "Minimum Score",
    float(df["score"].min()),
    float(df["score"].max()),
    float(df["score"].min())
)

filtered = df[
    df["score"] >= score
]

st.dataframe(filtered)

# ---------------------------------------------------
# Candidate Explorer
# ---------------------------------------------------

st.subheader("Candidate Explorer")

selected = st.selectbox(
    "Select Candidate",
    df["candidate_id"]
)

candidate_data = df[
    df["candidate_id"] == selected
]

# ---------------------------------------------------
# Candidate Details
# ---------------------------------------------------

st.subheader("Candidate Details")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Rank",
        int(candidate_data.iloc[0]["rank"])
    )

with col2:
    st.metric(
        "Score",
        round(
            candidate_data.iloc[0]["score"],
            4
        )
    )

st.write(
    candidate_data.iloc[0]["reasoning"]
)

# ---------------------------------------------------
# AI Explanation
# ---------------------------------------------------

st.subheader("AI Explanation")

if selected in summaries:
    st.write(
        summaries[selected]
    )

# ---------------------------------------------------
# Score Distribution
# ---------------------------------------------------

st.subheader("Score Distribution")

histogram = px.histogram(
    df,
    x="score",
    nbins=20,
    title="Candidate Score Distribution"
)

st.plotly_chart(
    histogram,
    use_container_width=True
)

# ---------------------------------------------------
# Recommendation Distribution
# ---------------------------------------------------

if "recommendation" in df.columns:

    st.subheader(
        "Recommendation Distribution"
    )

    pie = px.pie(
        df,
        names="recommendation",
        title="Recommendation Breakdown"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

# ---------------------------------------------------
# Top Ranked Candidates
# ---------------------------------------------------

st.subheader(
    "Top Ranked Candidates"
)

st.dataframe(
    df.head(100),
    use_container_width=True
)