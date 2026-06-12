import streamlit as st
import pandas as pd

st.title("AI Candidate Ranking Dashboard")

df = pd.read_csv("output/semantic_submission_full.csv")

st.dataframe(df.head(100))

st.sidebar.title("AI Talent Intelligence")

st.sidebar.metric(
    "Candidates Processed",
    "100,000"
)

st.sidebar.metric(
    "Top Ranked",
    "100"
)

score = st.slider(
    "Minimum Score",
    float(df.score.min()),
    float(df.score.max()),
    float(df.score.min())
)

filtered = df[df.score >= score]

st.dataframe(filtered)

candidate = st.selectbox(
    "Candidate",
    df["candidate_id"]
)

st.write(
    df[df["candidate_id"] == candidate]
)