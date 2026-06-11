# Redrob AI - Semantic Candidate Ranking System

## Overview

This project implements a semantic candidate ranking pipeline for the Redrob India Data & AI Challenge.

The objective is to rank candidates against a job description using semantic understanding rather than simple keyword matching.

The system combines:

* Semantic skill matching
* Experience relevance scoring
* Behavioral scoring
* Trust scoring
* Explainable ranking reasoning

to identify candidates most relevant to AI-focused hiring requirements.

---

## Architecture

### Job Parsing

The job description is parsed to extract:

* Required skills
* Preferred skills
* Responsibilities
* Semantic concepts

### Semantic Matching

Candidate profiles are compared against the job requirements using sentence embeddings.

The matching pipeline evaluates:

* Skill similarity
* Experience similarity
* Title similarity
* Summary similarity

### Skill Depth Analysis

Relevant skills are scored using:

* Semantic match quality
* Proficiency level
* Duration of experience

### Behavioral Scoring

Behavioral indicators include:

* Activity recency
* Recruiter response rate
* Interview completion rate
* Open-to-work signals

### Trust Scoring

Trust signals are used to assess profile reliability and consistency.

### Final Ranking

Final candidate score combines:

* Semantic Fit
* Experience Relevance
* Skill Depth
* Behavioral Score
* Trust Score
* Location Match

---

## Results

Validation was performed on:

| Candidates |
| ---------- |
| 1,000      |
| 5,000      |
| 10,000     |
| 100,000    |

Final full-dataset run:

* Candidates processed: 100,000
* Top score: 0.7591
* Top-100 score range: 0.6615 - 0.7591

Top-ranked candidates included:

* Senior Machine Learning Engineers
* Senior NLP Engineers
* Lead AI Engineers
* Search Engineers
* Applied ML Engineers
* AI Research Engineers

---

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ranking:

```bash
python semantic_ranker.py \
  --candidates candidates.jsonl \
  --job job_description.docx \
  --out semantic_submission.csv
```

Run on a subset:

```bash
python semantic_ranker.py \
  --candidates candidates.jsonl \
  --job job_description.docx \
  --out semantic_submission.csv \
  --max 10000
```

---

## Key Improvements

* Fixed semantic ranking pipeline initialization issues
* Improved responsibility extraction
* Improved skill similarity scoring
* Improved ranking explainability
* Calibrated recommendation thresholds
* Validated ranking quality on 100k candidate profiles

---

## Repository Structure

```text
semantic_ranker.py
semantic_matcher.py
embedding_manager.py
job_parser.py
summarizer.py
trust_scorer.py
evaluator.py
config_semantic.json
requirements.txt
README.md
```
