# Migration Guide: v1 (Keyword) → v2 (Semantic)

This guide helps you migrate from the original keyword-based ranking system (`rank.py`) to the new semantic ranking system (`semantic_ranker.py`).

---

## Table of Contents

1. [What Changed](#what-changed)
2. [Step-by-Step Migration](#step-by-step-migration)
3. [Configuration Migration](#configuration-migration)
4. [Performance Considerations](#performance-considerations)
5. [Troubleshooting](#troubleshooting)

---

## What Changed

### Architecture

| Aspect | v1 (rank.py) | v2 (semantic_ranker.py) |
|--------|--------------|------------------------|
| Job parsing | Hardcoded AI/ML skills | LLM or rule-based parsing |
| Matching | Regex & substring | Sentence embeddings (384-d) |
| Speed | O(n) linear scan | O(log n) with FAISS index |
| Weights | Fixed (35/30/20/15) | Configurable (40/25/15/10/5/5) |
| Explainability | Basic bullet list | Detailed reasoning + summaries |
| Trust | Honeypot only | Comprehensive detection |
| Generalization | AI/ML roles only | Any job description |

### Output Format

| v1 Field | v2 Equivalent | Notes |
|----------|---------------|-------|
| `score` | `score` | Similar 0-1 range, different calculation |
| `reasoning` | `reasoning` + `recommendation` + `trust_level` | More detailed |
| N/A | `semantic_fit`, `experience`, `skill_depth`, etc. | Component breakdown |

---

## Step-by-Step Migration

### 1. Install New Dependencies

```bash
# Remove old environment (optional)
pip uninstall -y numpy pandas  # Don't fully remove if other projects need them

# Install new requirements
pip install -r requirements.txt

# This will install:
# - sentence-transformers (embeddings)
# - faiss-cpu (vector search)
# - anthropic (optional LLM)
# - scikit-learn (metrics)
```

### 2. Prepare Job Description

The new system requires a job description, not just config.

Create `job_description.txt`:

```text
Senior Machine Learning Engineer

We are looking for a Senior ML Engineer...

Required Skills:
- Python
- PyTorch/TensorFlow
- AWS
...
```

Or create `job_requirements.json` (pre-parsed format):

```json
{
  "required_skills": ["Python", "PyTorch", "AWS"],
  "preferred_skills": ["Kubernetes", "Docker"],
  "seniority": "senior",
  "experience_years_min": 5,
  ...
}
```

### 3. Run the New System

```bash
# Using job description file
python semantic_ranker.py \
  --candidates candidates.jsonl \
  --job job_description.txt \
  --out submission_v2.csv

# Using pre-parsed requirements
python semantic_ranker.py \
  --candidates candidates.jsonl \
  --job-requirements job_requirements.json \
  --out submission_v2.csv

# For faster testing (subset)
python semantic_ranker.py \
  --candidates candidates.jsonl \
  --job job_description.txt \
  --out submission_v2.csv \
  --max 1000 \
  --top-n 100
```

### 4. Compare Results

Run both systems and compare:

```bash
# Run v1
python rank.py --candidates candidates.jsonl --out submission_v1.csv

# Run v2
python semantic_ranker.py --candidates candidates.jsonl --job job_description.txt --out submission_v2.csv

# Compare rankings
python -c "
import pandas as pd
v1 = pd.read_csv('submission_v1.csv')
v2 = pd.read_csv('submission_v2.csv')
print('v1 top 10:', v1.head(10)['candidate_id'].tolist())
print('v2 top 10:', v2.head(10)['candidate_id'].tolist())
print('Overlap:', len(set(v1['candidate_id']).intersection(set(v2['candidate_id']))))
"
```

---

## Configuration Migration

### v1 `config.json`

```json
{
  "component_weights": {
    "skill": 0.35,
    "career": 0.30,
    "behavioral": 0.20,
    "location": 0.15
  },
  "must_have_skills": [...],
  "preferred_cities": [...],
  ...
}
```

### v2 `config_semantic.json`

```json
{
  "component_weights": {
    "semantic_fit": 0.40,
    "experience": 0.25,
    "skill_depth": 0.15,
    "behavioral": 0.10,
    "trust": 0.05,
    "location": 0.05
  },
  "location_scoring": {
    "preferred_cities": [...]  // Reuse from config.json
  },
  ...
}
```

### Sharing Settings

The v2 system can read `config.json` for certain settings (services_companies, preferred_cities). Copy your v1 config:

```bash
cp config.json ./config_semantic.json  # Then customize weights
```

---

## Performance Considerations

### First Run - Model Download

The first run downloads the embedding model (~400MB):

```
Downloading: 100%|█████████████████████| 400/400 MB
```

**Tip**: Pre-download by running a quick test first:

```bash
python test_semantic.py
```

### RAM Usage

| Operation | v1 | v2 |
|-----------|----|----|
| Peak RAM (100k) | ~300MB | ~700MB |
| Peak RAM (500k) | ~1.5GB | ~3GB |

**Solution**: Use smaller batches or reduce dataset size.

### Runtime

| Dataset | v1 | v2 (single core) |
|---------|----|------------------|
| 10k | ~10s | ~20s |
| 100k | ~95s | ~180s |
| 500k | ~8m | ~15m |

**Tips to speed up**:
1. Use `--num-workers 4` (your CPU cores)
2. Enable FAISS (default) - rebuild once with `--no-faiss` NOT set
3. Use cached index (first run builds it)
4. Reduce dataset with `--max` for testing

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Fix**: Install dependencies:

```bash
pip install sentence-transformers
# Also: pip install faiss-cpu
```

### FAISS Not Available

```
ImportError: libfaiss.so: cannot open shared object file
```

**Fix**: Use CPU version:

```bash
pip uninstall faiss faiss-gpu
pip install faiss-cpu
```

### LLM Parsing Fails

If using `--use-llm` and it fails:

1. Check API key: `echo $ANTHROPIC_API_KEY`
2. Without key, falls back to rule-based
3. Or disable LLM: remove `--use-llm`

### Out of Memory

```
RuntimeError: CUDA out of memory
```

The system works on CPU by default. If still OOM:

1. Reduce `--batch-size` (default: 32 → try 16)
2. Process fewer candidates: `--max 50000`
3. Close other applications

### Scores All Similar

If all candidates have very close scores:

- Check job description is descriptive enough
- Try a different embedding model
- Check config weights are not all equal

---

## Rollback

To revert to v1:

```bash
# Rename current files
mv semantic_ranker.py semantic_ranker.py.v2
mv config_semantic.json config_semantic.json.v2

# Restore v1 from git (or rename backups)
git checkout HEAD -- rank.py config.json

# Run v1
python rank.py --candidates data.jsonl --out v1_results.csv
```

---

## Support

- Check `test_semantic.py` for validation
- See `README.md` for full documentation
- Review `config_semantic.json` to tweak weights
- Open an issue for bugs

---

## Summary

| Task | v1 Command | v2 Command |
|------|-----------|------------|
| Rank candidates | `python rank.py --candidates X --out Y` | `python semantic_ranker.py --candidates X --job jd.txt --out Y` |
| Test installation | N/A | `python test_semantic.py` |
| Change weights | Edit `config.json` | Edit `config_semantic.json` |
| Add job type | Hardcode new skills | Provide new job description |

**Key difference**: v1 needs config changes for new jobs; v2 just needs a new JD.

---

End of Migration Guide
