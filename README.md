# AI-Powered Semantic Candidate Ranking System

## Overview

An end-to-end AI recruitment intelligence platform built for the Redrob India Runs Data & AI Challenge.

The system analyzes a job description, semantically evaluates 100,000 candidate profiles, ranks the most relevant candidates, generates explainable recommendations, and provides an interactive recruiter dashboard for exploration and decision-making.

---

## Live Demo

🚀 Dashboard: https://redrob-thedataandaichallenge.streamlit.app/

📂 GitHub Repository: https://github.com/linga-1221/the_data_and_ai_challenge

---

## Problem Statement

Traditional ATS systems rely heavily on keyword matching, often missing highly relevant candidates whose profiles use different terminology.

This solution leverages semantic search, embeddings, behavioral signals, trust scoring, and explainable AI to identify the best candidates beyond simple keyword overlap.

---

## Key Features

* Semantic candidate matching using embeddings
* Job description parsing and requirement extraction
* Multi-factor ranking engine
* Trust and behavioral scoring
* Explainable AI candidate summaries
* Top-100 candidate recommendation generation
* Interactive recruiter dashboard
* CSV submission generation
* Scalable evaluation for 100,000 candidate profiles

---

## System Architecture

Job Description

↓

Job Parser

↓

Requirement Extraction

↓

Embedding Manager

↓

Semantic Matcher

↓

Trust & Behavioral Scoring

↓

Ranking Engine

↓

Top 100 Candidates

↓

CSV Submission + Recruiter Dashboard

---

## Ranking Methodology

The final ranking score combines multiple candidate signals:

| Component          | Description                                           |
| ------------------ | ----------------------------------------------------- |
| Semantic Fit       | Embedding similarity between candidate profile and JD |
| Experience Score   | Relevance of work experience                          |
| Skill Depth        | Coverage and strength of required skills              |
| Behavioral Score   | Candidate engagement and profile quality signals      |
| Trust Score        | Reliability and consistency indicators                |
| Concept Similarity | Matching of related concepts beyond keywords          |

Final ranking is generated using weighted score aggregation.

---

## Recruiter Dashboard

The Streamlit dashboard provides:

* Candidate search
* Score-based filtering
* Candidate exploration
* AI-generated explanations
* Score distribution analytics
* Recommendation distribution analytics
* Downloadable ranked CSV

---

## Technologies Used

### Programming

* Python

### AI / NLP

* Sentence Transformers
* Semantic Search
* Embedding-Based Retrieval
* Explainable AI

### Data Processing

* Pandas
* NumPy
* Scikit-Learn

### Dashboard

* Streamlit
* Plotly

### Infrastructure

* Git
* GitHub

---

## Results

* Processed 100,000 candidate profiles
* Generated ranked Top-100 recommendations
* Achieved strong semantic candidate matching
* Produced explainable ranking decisions
* Built recruiter-facing analytics dashboard

---

## Project Structure

```text
app.py
rank.py
semantic_ranker.py
semantic_matcher.py
embedding_manager.py
job_parser.py
trust_scorer.py
summarizer.py

data/
├── semantic_submission_full.csv
├── semantic_submission_full_summaries.json
└── job_requirements.json
```

---

## Future Improvements

* Hybrid vector + keyword retrieval
* Cross-encoder re-ranking
* Recruiter feedback learning loop
* Skill-gap analysis
* Real-time dashboard analytics
* Multi-job recommendation support

---

