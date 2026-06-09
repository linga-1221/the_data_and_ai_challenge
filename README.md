# AI-Powered Candidate Ranking System

## 🎯 Problem Statement

Recruiters miss great candidates because keyword filters can't understand who actually fits a role. This system ranks 100,000+ candidate profiles the way a great recruiter would — by understanding the full picture, not just matching keywords.

## 📋 Requirements Met

| Requirement | Status |
|-------------|--------|
| Process 100,000 candidates | ✅ 99,954 scored in 95 seconds |
| Output top 100 as CSV | ✅ `submission.csv` |
| Columns: candidate_id, rank, score, reasoning | ✅ |
| Scores monotonic non-increasing | ✅ Verified |
| < 5 minutes runtime | ✅ 1.6 minutes |
| No external APIs / LLMs | ✅ Pure Python |
| CPU only | ✅ No GPU required |
| Honeypot detection | ✅ 46 filtered |

## 🏗️ Architecture

### Multi-Signal Scoring Model

The system uses a **weighted linear model** with 4 components:

```
final_score = (0.35 × skill + 0.30 × career + 0.20 × behavioral + 0.15 × location) × disqualifier_multiplier
```

#### 1. Skill Score (35%)

Searches both **explicit skills list** and **career description text** using normalized substring matching.

**Must-have skills** (65% weight): embeddings, vector database, pinecone, weaviate, qdrant, milvus, faiss, opensearch, elasticsearch, pgvector, hybrid search, semantic search, ranking/re-ranking, python, retrieval

**Nice-to-have** (25% weight): LoRA/QLoRA, learning-to-rank, NDCG/MRR, A/B testing, evaluation frameworks, RAG, LLM, BERT, BM25, recommendation systems

**Proficiency bonus** (10% max): +0.05 for advanced, +0.08 for expert, +0.04 for duration ≥24 months

#### 2. Career Score (30%)

| Component | Weight | Details |
|-----------|--------|---------|
| YoE band | 40% | <3yr=0.10, 3-5yr=0.55, 5-9yr=1.00, 9-12yr=0.80, 12+yr=0.60 |
| Company type | 25% | Product companies score 0.40-1.00; services-only = 0.20 |
| Title relevance | 20% | AI/ML/NLP/Search/Recommendation titles = 1.0 |
| Evidence phrases | 10% | "hybrid retrieval", "dense retrieval", "ranking pipeline", etc. |
| Production bonus | +0.15 | Keywords: deployed, production, scale, queries per, monitoring |
| Prestige bonus | +0.35 | Google, Meta, Razorpay, Flipkart, etc. |

#### 3. Behavioral Score (20%)

Based on `redrob_signals`:
- Last active recency (25%)
- Open to work flag (15%)
- Recruiter response rate (20%)
- Interview completion rate (15%)
- Notice period (15%)
- GitHub activity (10%)

#### 4. Location Score (15%)

| Category | Score |
|----------|-------|
| Preferred Indian cities (Pune, Noida, Hyderabad, etc.) | 1.0 |
| Other India | 0.65 (0.80 if willing to relocate) |
| International | 0.10 (0.30 if willing to relocate) |

## 🚫 Disqualifiers & Honeypots

**Disqualifier Multipliers** (applied to final score):
- Entire career at services companies only: ×0.30
- CV/speech/robotics primary with no NLP/IR: ×0.20
- Unrelated titles (Marketing, Sales, HR, etc.): ×0.05

**Honeypot Detection** (excluded from results):
- 5+ advanced/expert skills with 0 duration
- Negative career tenure (end date before start)
- Stated YoE exceeds sum of career history by >6 years
- YoE > 35

## 🎯 Why This Works

1. **No keyword spamming** — Skills are matched across both explicit list AND free-text descriptions, detecting evidence in context.
2. **Production experience matters** — Description scanning for deployment/production keywords filters out academic-only candidates.
3. **Company quality recognized** — Product company experience is rewarded; pure services-only careers are heavily penalized.
4. **Behavioral signals matter** — Active candidates open to work with good response rates rank higher.
5. **Location intelligence** — Preferred hiring locations score higher, but relocatable international candidates aren't entirely excluded.
6. **Title relevance** — Only candidates with AI/ML/Search/IR titles are considered top contenders.

The top candidates consistently show:
- AI/ML/Search/NLP titles (5-9 years experience)
- Vector database & retrieval skills (FAISS, Pinecone, Qdrant, Elasticsearch, pgvector)
- Product company backgrounds (Razorpay, Paytm, Flipkart, Google, Amazon, etc.)
- Production evidence in descriptions ("serving XM+ queries", "hybrid retrieval", "deployed", "scale")
- Preferred Indian locations

## 🚀 Performance

- **Runtime**: 95 seconds for 100,000 candidates on standard CPU
- **Memory**: ~300MB peak (fits easily in 16GB RAM)
- **Throughput**: ~1,050 candidates/second

## 📦 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run ranking (adjust paths as needed)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Optional: limit for testing
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --max 1000
```

## 📁 File Structure

```
/
├── rank.py              # Main ranking system
├── requirements.txt     # Dependencies (numpy, pandas)
├── submission.csv       # Output file (top 100 ranked candidates)
└── README.md            # This file
```

## 🔍 Inspecting Results

```bash
# View top 10
head -11 submission.csv

# Check score range
python -c "import pandas as pd; df=pd.read_csv('submission.csv'); print(df['score'].describe())"

# Count preferred city placements
python -c "import pandas as pd; df=pd.read_csv('submission.csv'); print(df['reasoning'].str.contains('Loc: (Pune|Noida|Hyderabad|Bangalore|Mumbai|Delhi|Jaipur|Kochi|Kolkata|Chennai|Ahmedabad|Indore)').sum())"
```

## ✅ Results Summary

- **Total candidates processed**: 100,000
- **Scored**: 99,954
- **Honeypots filtered**: 46
- **Top score**: 0.7548
- **Score range**: 0.6700 - 0.7548
- **Top titles**: Lead AI Engineer, Staff ML Engineer, Senior NLP Engineer, Senior ML Engineer, Search Engineer, Recommendation Systems Engineer
- **Locations**: Premium Indian tech hubs (Jaipur, Kochi, Coimbatore, Noida, Pune, Kolkata, Trivandrum, Vizag, Chandigarh, Bangalore, Delhi)

## 📝 License

MIT License — Feel free to use and adapt for your hiring workflows.

---

**Built for The Data & AI Challenge** — Finding signal in the noise.
