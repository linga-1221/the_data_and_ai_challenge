"""
Semantic Ranking System - Main Entry Point

Recruiter-grade semantic candidate ranking with:
- LLM-based job description parsing
- Sentence transformer embeddings
- FAISS for scalable search
- Hybrid multi-component scoring:
  * 40% Semantic Fit
  * 25% Experience Relevance
  * 15% Skill Depth
  * 10% Behavioral Signals
  * 5% Trustworthiness
  * 5% Location
- Explainable AI reasoning
- Profile trust scoring
- Recruiter summaries
- Evaluation framework
"""

import os
import json
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Import our modules
from job_parser import JobRequirements, extract_job_requirements
from embedding_manager import create_embedding_manager, EmbeddingManager
from semantic_matcher import create_semantic_matcher, SemanticMatcher, MatchScores
from trust_scorer import create_trust_scorer, TrustScorer, TrustScore
from summarizer import create_summarizer, RecruiterSummarizer, RecruiterSummary
from evaluator import RankingEvaluator

logger = logging.getLogger(__name__)


@dataclass
class RankerConfig:
    """Main configuration for semantic ranker"""
    input_path: str
    output_path: str
    job_description: str = ""
    job_requirements_path: Optional[str] = None
    use_llm_for_job_parsing: bool = False
    llm_api_key: Optional[str] = None
    top_n: int = 100
    max_candidates: Optional[int] = None
    num_workers: int = 0
    batch_size: int = 32
    cache_dir: str = "./cache"
    faiss_enabled: bool = True
    semantic_config_path: str = "./config_semantic.json"

    # Component weights (must sum to 1.0)
    semantic_weight: float = 0.40
    experience_weight: float = 0.25
    skill_weight: float = 0.15
    behavioral_weight: float = 0.10
    trust_weight: float = 0.05
    location_weight: float = 0.05

    # Semantic matcher weights
    semantic_skill_weight: float = 0.25
    semantic_experience_weight: float = 0.40
    semantic_title_weight: float = 0.10
    semantic_summary_weight: float = 0.25

    # Trust thresholds
    trust_min_score: float = 0.3

    # Runtime state (not from CLI)
    preferred_cities: List[str] = field(default_factory=list)

    def validate(self):
        """Validate configuration"""
        total = (self.semantic_weight + self.experience_weight + self.skill_weight +
                self.behavioral_weight + self.trust_weight + self.location_weight)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Component weights must sum to 1.0, got {total}")
        return True


@dataclass
class RankedCandidate:
    """Final ranked candidate output"""
    candidate_id: str
    rank: int
    final_score: float
    semantic_fit_score: float
    experience_score: float
    skill_depth_score: float
    behavioral_score: float
    trust_score: float
    location_score: float
    concept_similarity: float
    reasoning: List[str]
    summary: Optional[RecruiterSummary] = None
    trust_details: Optional[TrustScore] = None

    def to_output_dict(self) -> Dict:
        """Convert to output dictionary (CSV format)"""
        return {
            'candidate_id': self.candidate_id,
            'rank': self.rank,
            'score': round(self.final_score, 4),
            'semantic_fit': round(self.semantic_fit_score, 4),
            'experience': round(self.experience_score, 4),
            'skill_depth': round(self.skill_depth_score, 4),
            'behavioral': round(self.behavioral_score, 4),
            'trust': round(self.trust_score, 4),
            'location': round(self.location_score, 4),
            'concept_similarity': round(self.concept_similarity, 4),
            'reasoning': '; '.join(self.reasoning[:5])[:500],
            'recommendation': self.summary.recommendation if self.summary else "",
            'trust_level': self.trust_details.trust_level if self.trust_details else "unknown"
        }


class SemanticRanker:
    """
    Main semantic ranking system.

    Process:
    1. Parse job description into structured requirements
    2. Load and index candidates with embeddings
    3. For each candidate:
       - Compute semantic fit (40%)
       - Compute experience relevance (25%)
       - Compute skill depth (15%)
       - Compute behavioral signals (10%)
       - Compute trust score (5%)
       - Compute location score (5%)
       - Apply trust multiplier
    4. Generate explainable reasoning
    5. Generate recruiter summaries
    6. Rank and output top N
    """

    # Existing component weights from original system for behavioral/location
    ORIGINAL_BEHAVIORAL_WEIGHTS = {
        'last_active': 0.25,
        'open_to_work': 0.15,
        'recruiter_response_rate': 0.20,
        'interview_completion_rate': 0.15,
        'notice_period': 0.15,
        'github_activity': 0.10
    }

    ORIGINAL_LOCATION_SCORES = {
        'preferred': 1.0,
        'india_relocatable': 0.80,
        'india_non_preferred': 0.65,
        'international_relocatable': 0.30,
        'international': 0.10
    }

    def __init__(self, config: RankerConfig):
        self.config = config
        self.config.validate()

        # Initialize components
        self._init_components()

        # State
        self.job_requirements: Optional[JobRequirements] = None
        self.evaluator = RankingEvaluator()

        # Cache for candidate data
        self._candidate_cache: Dict[str, Dict] = {}

    def _init_components(self) -> None:
        """Initialize all system components"""
        logger.info("Initializing Semantic Ranker components...")

        # Load semantic config for location cities and other settings
        try:
            with open(self.config.semantic_config_path, 'r') as f:
                semantic_cfg = json.load(f)
            # Populate config from semantic config
            self.config.preferred_cities = semantic_cfg.get('location_scoring', {}).get('preferred_cities', [])
            # Could load more settings if needed
            logger.info(f"Loaded semantic config with {len(self.config.preferred_cities)} preferred cities")
        except Exception as e:
            logger.warning(f"Could not load semantic config: {e}. Using empty preferred cities.")
            self.config.preferred_cities = []

        # Embedding manager
        emb_config = {
            'cache_dir': self.config.cache_dir,
            'faiss_index_path': os.path.join(self.config.cache_dir, 'faiss_index.bin'),
            'use_faiss': self.config.faiss_enabled,
            'batch_size': self.config.batch_size
        }
        self.embedding_manager = create_embedding_manager(emb_config)

        # Semantic matcher
        self.semantic_matcher = create_semantic_matcher(
            embedding_config=emb_config,
            skill_weight=self.config.semantic_skill_weight,
            experience_weight=self.config.semantic_experience_weight,
            title_weight=self.config.semantic_title_weight,
            summary_weight=self.config.semantic_summary_weight
        )

        # Trust scorer (need original config for services companies etc.)
        try:
            with open('./config.json', 'r') as f:
                original_config = json.load(f)
        except:
            original_config = {}

        self.trust_scorer = create_trust_scorer(original_config)

        # Summarizer (optional LLM)
        self.summarizer = create_summarizer(
            use_llm=self.config.use_llm_for_job_parsing,
            llm_api_key=self.config.llm_api_key
        )

        logger.info("All components initialized")

    def parse_job_description(self) -> JobRequirements:
        """
        Parse job description into structured requirements.

        Returns:
            JobRequirements object
        """
        logger.info("Parsing job description...")

        if self.config.job_requirements_path and os.path.exists(self.config.job_requirements_path):
            logger.info(f"Loading job requirements from {self.config.job_requirements_path}")
            req = JobRequirements.load(self.config.job_requirements_path)
        elif self.config.job_description:
            req = extract_job_requirements(
                self.config.job_description,
                use_llm=self.config.use_llm_for_job_parsing,
                llm_api_key=self.config.llm_api_key
            )
            # Save extracted requirements for reference
            output_dir = os.path.dirname(self.config.output_path)
            os.makedirs(output_dir, exist_ok=True)
            req_path = os.path.join(output_dir, 'job_requirements.json')
            req.save(req_path)
            logger.info(f"Saved job requirements to {req_path}")
        else:
            raise ValueError("Either job_description or job_requirements_path must be provided")

        self.job_requirements = req
        logger.info(f"Job requirements: {len(req.required_skills)} required skills, "
                   f"{len(req.preferred_skills)} preferred skills")
        return req

    def build_candidate_index(
        self,
        candidate_profiles: List[Dict],
        rebuild: bool = False
    ) -> None:
        """
        Build FAISS index for candidate embeddings.

        Args:
            candidate_profiles: List of candidate data dictionaries
            rebuild: Force rebuild even if cache exists
        """
        logger.info(f"Building candidate index for {len(candidate_profiles)} profiles...")

        # Check if index already exists
        index_path = os.path.join(self.config.cache_dir, 'faiss_index.bin')
        mapping_path = os.path.join(self.config.cache_dir, 'id_mappings.pkl')
        if not rebuild and os.path.exists(index_path) and os.path.exists(mapping_path):
            logger.info("Found existing FAISS index, loading...")
            try:
                import pickle
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self.embedding_manager._id_to_idx = mappings['id_to_idx']
                    self.embedding_manager._idx_to_id = mappings['idx_to_id']
                self.embedding_manager._load_cache()
                logger.info(f"Loaded {len(self.embedding_manager._id_to_idx)} candidates from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}, rebuilding...")

        # Generate embeddings for all candidates
        candidate_ids = []
        embedding_texts = []

        for candidate in candidate_profiles:
            cid = candidate.get('candidate_id')
            if not cid:
                continue

            # Build combined text for embedding
            texts = []

            # Add current title and summary
            profile = candidate.get('profile', {})
            if profile.get('current_title'):
                texts.append(profile['current_title'])
            if profile.get('headline'):
                texts.append(profile['headline'])
            if profile.get('summary'):
                texts.append(profile['summary'])

            # Add skills
            skills = candidate.get('skills', [])
            skill_texts = [s.get('name', '') for s in skills[:20]]
            if skill_texts:
                texts.append(' '.join(skill_texts))

            # Add career descriptions
            for job in candidate.get('career_history', [])[:5]:
                if job.get('description'):
                    texts.append(job['description'][:200])

            embedding_texts.append(' '.join(texts)[:1500])
            candidate_ids.append(cid)
            self._candidate_cache[cid] = candidate

        logger.info(f"Generating embeddings for {len(candidate_ids)} candidates...")
        embeddings = self.embedding_manager.embed_texts(embedding_texts)

        # Build FAISS index
        logger.info("Building FAISS index...")
        self.embedding_manager.build_faiss_index(candidate_ids, embeddings)

        logger.info(f"Index built: {len(candidate_ids)} candidates")

    def score_candidate(
        self,
        candidate_id: str,
        candidate_profile: Dict,
        job_requirements: JobRequirements
    ) -> RankedCandidate:
        """
        Compute comprehensive score for a single candidate.

        Args:
            candidate_id: Candidate identifier
            candidate_profile: Full candidate data
            job_requirements: Structured job requirements

        Returns:
            RankedCandidate object with all scores and reasoning
        """
        start_time = time.time()

        # 1. Semantic Fit (40%)
        semantic_scores = self.semantic_matcher.match(job_requirements, candidate_profile)
        semantic_fit = semantic_scores.semantic_fit_score

        # 2. Experience Relevance (25%)
        # Reuse semantic experience score, but can add additional factors
        experience_score = semantic_scores.experience_similarity

        # Bonus for years of experience
        profile = candidate_profile.get('profile', {})
        yoe = profile.get('years_of_experience', 0)
        exp_yoe_bonus = 0.0
        if yoe >= 5:
            exp_yoe_bonus = 0.1
        elif yoe >= 3:
            exp_yoe_bonus = 0.05
        experience_score = min(experience_score + exp_yoe_bonus, 1.0)

        # 3. Skill Depth (15%)
        skill_depth = self._compute_skill_depth(
            candidate_profile.get('skills', []),
            job_requirements.required_skills + job_requirements.preferred_skills,
            semantic_scores.detailed_skill_matches
        )

        # 4. Behavioral Signals (10%)
        behavioral_score = self._compute_behavioral_score(candidate_profile)

        # 5. Trustworthiness (5%)
        trust_score_obj = self.trust_scorer.calculate_trust_score(candidate_profile, job_requirements)
        trust_score = trust_score_obj.overall_score

        # 6. Location (5%)
        location_score = self._compute_location_score(candidate_profile)

        # Combine weighted scores
        final_score = (
            self.config.semantic_weight * semantic_fit +
            self.config.experience_weight * experience_score +
            self.config.skill_weight * skill_depth +
            self.config.behavioral_weight * behavioral_score +
            self.config.trust_weight * trust_score +
            self.config.location_weight * location_score
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            candidate_profile,
            job_requirements,
            semantic_scores,
            skill_depth,
            experience_score,
            behavioral_score,
            trust_score_obj,
            location_score
        )

        # Generate summary
        summary = self.summarizer.generate_summary(
            candidate_profile,
            job_requirements,
            semantic_scores,
            trust_score_obj,
            final_score,
            semantic_scores.detailed_skill_matches
        )

        elapsed = time.time() - start_time
        logger.debug(f"Scored {candidate_id} in {elapsed:.3f}s")

        return RankedCandidate(
            candidate_id=candidate_id,
            rank=0,  # Will be set later
            final_score=final_score,
            semantic_fit_score=semantic_fit,
            experience_score=experience_score,
            skill_depth_score=skill_depth,
            behavioral_score=behavioral_score,
            trust_score=trust_score,
            location_score=location_score,
            concept_similarity=semantic_scores.concept_similarity,
            reasoning=reasoning,
            summary=summary,
            trust_details=trust_score_obj
        )

    def _compute_skill_depth(
        self,
        candidate_skills: List[Dict],
        job_skills: List[str],
        skill_matches: List[Dict]
    ) -> float:
        """
        Compute skill depth score based on proficiency and duration.
        Based on original skill scoring logic.
        """
        if not candidate_skills:
            return 0.0

        total_score = 0.0

        # Base from semantic skill matches
        if skill_matches:
            match_qualities = [m['similarity'] for m in skill_matches]
            coverage = min(len(skill_matches) / max(len(job_skills), 1), 1.0)
            base_score = (
                0.6 * np.mean(match_qualities[:10]) +
                0.4 * coverage
            )
        else:
            base_score = 0.0

        # Proficiency bonus
        proficiency_bonus = 0.0
        for skill in candidate_skills[:15]:
            prof = skill.get('proficiency', '').lower()
            duration = skill.get('duration_months', 0)

            # Only bonus for skills relevant to job (simplified check)
            skill_name = skill.get('name', '').lower()
            if any(js.lower() in skill_name or skill_name in js.lower() for js in job_skills[:20]):
                if prof == 'expert':
                    proficiency_bonus += 0.08
                elif prof == 'advanced':
                    proficiency_bonus += 0.05
                if duration >= 24:
                    proficiency_bonus += 0.04

        proficiency_bonus = min(proficiency_bonus, 0.10)

        # Duration bonus for relevant skill longevity
        relevant_durations = [
            s.get('duration_months', 0) for s in candidate_skills[:15]
            if any(js.lower() in s.get('name', '').lower() for js in job_skills[:20])
        ]
        if relevant_durations:
            avg_relevant_duration = np.mean(relevant_durations)
            duration_score = min(avg_relevant_duration / 36.0, 0.1)  # Max bonus at 36 months
        else:
            duration_score = 0.0

        final_score = base_score + proficiency_bonus + duration_score
        return min(final_score, 1.0)

    def _compute_behavioral_score(self, candidate_profile: Dict) -> float:
        """
        Compute behavioral score from redrob signals and profile activity.
        Reuses logic from original BehavioralScorer.
        """
        redrob = candidate_profile.get('redrob_signals', {})
        profile = candidate_profile.get('profile', {})

        weights = self.ORIGINAL_BEHAVIORAL_WEIGHTS
        scores = {}
        evidence = []

        # Last active recency
        last_active = profile.get('last_active_date') or redrob.get('last_active_date')
        active_score = self._score_last_active(last_active)
        scores['last_active'] = active_score

        # Open to work
        open_to_work = redrob.get('open_to_work_flag', False)
        scores['open_to_work'] = 1.0 if open_to_work else 0.45
        evidence.append(f"Open: {open_to_work}")

        # Recruiter response rate
        response_rate = redrob.get('recruiter_response_rate', 0.5)
        if isinstance(response_rate, (int, float)):
            scores['recruiter_response_rate'] = min(max(response_rate, 0.0), 1.0)
        else:
            scores['recruiter_response_rate'] = 0.5

        # Interview completion rate
        interview_rate = redrob.get('interview_completion_rate', 0.5)
        if isinstance(interview_rate, (int, float)):
            scores['interview_completion_rate'] = min(max(interview_rate, 0.0), 1.0)
        else:
            scores['interview_completion_rate'] = 0.5

        # Notice period (shorter is better)
        notice_days = redrob.get('notice_period_days', 999)
        notice_score = self._score_notice_period(notice_days)
        scores['notice_period'] = notice_score

        # GitHub activity
        github_score = redrob.get('github_activity_score', -1)
        if github_score == -1 or github_score is None:
            scores['github_activity'] = 0.30
        else:
            scores['github_activity'] = min(max(github_score, 0.0), 100) / 100.0

        # Weighted sum
        behavioral_score = sum(
            scores[key] * weights[key] for key in weights if key in scores
        )

        return min(behavioral_score, 1.0)

    def _score_last_active(self, last_active: Any) -> float:
        """Score based on last active recency"""
        if not last_active:
            return 0.20

        try:
            from datetime import datetime
            date_str = str(last_active).strip()
            if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
                return 1.0

            # Try to parse
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']:
                try:
                    dt = datetime.strptime(date_str[:10], fmt)
                    days_ago = (datetime.now() - dt).days
                    if days_ago < 7:
                        return 1.0
                    elif days_ago < 14:
                        return 0.95
                    elif days_ago < 30:
                        return 0.85
                    elif days_ago < 60:
                        return 0.65
                    elif days_ago < 90:
                        return 0.45
                    elif days_ago < 180:
                        return 0.20
                    else:
                        return 0.05
                except:
                    pass
        except:
            pass

        return 0.20

    def _score_notice_period(self, notice_days: int) -> float:
        """Score based on notice period (shorter is better)"""
        thresholds = [
            (15, 1.0),
            (30, 0.90),
            (60, 0.70),
            (90, 0.45),
            (float('inf'), 0.20)
        ]
        for max_days, score in thresholds:
            if notice_days <= max_days:
                return score
        return 0.20

    def _compute_location_score(self, candidate_profile: Dict) -> float:
        """
        Compute location score based on preferences.
        Uses config_semantic preferred cities.
        """
        profile = candidate_profile.get('profile', {})
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        willing_to_relocate = candidate_profile.get('redrob_signals', {}).get('willing_to_relocate', False)

        preferred_cities = self.config.preferred_cities

        if not location:
            return 0.5

        # Check preferred cities
        if any(city in location for city in preferred_cities):
            return 1.0

        # Check other Indian cities
        if 'india' in country or 'india' in location:
            if willing_to_relocate:
                return 0.80
            return 0.65

        # International
        if willing_to_relocate:
            return 0.30
        return 0.10

    def _generate_reasoning(
        self,
        candidate_profile: Dict,
        job_requirements: JobRequirements,
        semantic_scores: MatchScores,
        skill_depth: float,
        experience_score: float,
        behavioral_score: float,
        trust_score_obj: TrustScore,
        location_score: float
    ) -> List[str]:
        """Generate clear, concise reasoning for the ranking"""
        reasoning = []
        profile = candidate_profile.get('profile', {})

        # Core matches
        logger.debug(
            "%s skill_sim=%.3f matched=%d",
            candidate_profile.get("candidate_id"),
            semantic_scores.skill_similarity,
            len(semantic_scores.detailed_skill_matches)
        )
        if semantic_scores.skill_similarity > 0.25:
            reasoning.append(f"Skills: {len(semantic_scores.detailed_skill_matches)} matched")
        else:
            reasoning.append("Limited skill overlap")

        # Experience and title
        title = profile.get('current_title', '')
        yoe = profile.get('years_of_experience', 0)
        if experience_score > 0.6:
            reasoning.append(f"Experience relevant ({int(yoe)}yr as {title})")
        else:
            reasoning.append(f"Experience relevance: {experience_score:.2f}")

        # Trust
        if trust_score_obj.trust_level == "critical":
            reasoning.append("Trust concerns detected - review required")
        elif trust_score_obj.trust_level == "low":
            reasoning.append("Minor profile inconsistencies")

        # Key skill depth
        if skill_depth > 0.7:
            reasoning.append("Strong skill depth")
        elif skill_depth < 0.4:
            reasoning.append("Limited skill depth")

        # Behavioral
        if behavioral_score > 0.8:
            reasoning.append("Strong engagement signals")
        elif behavioral_score < 0.5:
            reasoning.append("Low platform activity")

        # Location
        if location_score < 0.5:
            reasoning.append("Location may be a constraint")

        return reasoning[:5]

    def rank_candidates(
        self,
        candidate_profiles: List[Dict],
        output_summaries: bool = True
    ) -> Tuple[List[RankedCandidate], pd.DataFrame]:
        """
        Main ranking function.

        Args:
            candidate_profiles: List of candidate dictionaries
            output_summaries: Generate recruiter summaries (slower)

        Returns:
            (list_of_ranked_candidates, output_dataframe)
        """
        logger.info("Starting candidate ranking...")

        if not self.job_requirements:
            raise ValueError("Job requirements not parsed. Call parse_job_description() first.")

        # Build/load FAISS index for efficient retrieval (optional for now)
        # For prefetching semantic candidates
        self.build_candidate_index(candidate_profiles)

        # Score all candidates (can be parallelized)
        num_workers = self.config.num_workers if self.config.num_workers > 0 else 1
        scored_candidates = []

        logger.info(f"Scoring {len(candidate_profiles)} candidates...")
        start_time = time.time()

        for i, candidate in enumerate(candidate_profiles):
            cid = candidate.get('candidate_id')
            if not cid:
                continue

            try:
                ranked = self.score_candidate(cid, candidate, self.job_requirements)
                scored_candidates.append(ranked)

                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"  Progress: {i+1}/{len(candidate_profiles)} candidates ({rate:.0f}/s)")

            except Exception as e:
                logger.error(
                    f"Error scoring candidate {cid}: {e}",
                    exc_info=True
                )

        elapsed = time.time() - start_time
        logger.info(f"Scoring completed in {elapsed:.2f}s ({len(scored_candidates)/elapsed:.0f} candidates/s)")

        # Sort by final score descending
        scored_candidates.sort(key=lambda c: (-c.final_score, c.candidate_id))

        # Apply ranks
        for i, cand in enumerate(scored_candidates, 1):
            cand.rank = i

        # Get top N
        top_candidates = scored_candidates[:self.config.top_n]

        # Create output DataFrame
        output_data = [c.to_output_dict() for c in top_candidates]
        df = pd.DataFrame(output_data)

        # Verify monotonicity
        for i in range(1, len(df)):
            if df.iloc[i]['score'] > df.iloc[i-1]['score'] + 1e-6:
                logger.warning(f"Score not monotonic at rank {i+1}: {df.iloc[i]['score']} > {df.iloc[i-1]['score']}")
                df.at[i, 'score'] = df.iloc[i-1]['score']

        logger.info(f"Top {len(df)} candidates prepared")
        logger.info(f"Score range: {df['score'].min():.4f} - {df['score'].max():.4f}")

        return top_candidates, df

    def run(self) -> pd.DataFrame:
        """
        Run the complete ranking pipeline.

        Returns:
            Output DataFrame with rankings
        """
        # Parse job requirements
        self.parse_job_description()

        # Load candidates
        logger.info(f"Loading candidates from {self.config.input_path}")
        candidates = []
        with open(self.config.input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.config.max_candidates and i >= self.config.max_candidates:
                    break
                try:
                    candidate = json.loads(line.strip())
                    candidates.append(candidate)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {i+1}: {e}")

        logger.info(f"Loaded {len(candidates)} candidates")

        # Rank candidates
        top_candidates, df = self.rank_candidates(candidates, output_summaries=True)

        # Save output
        output_dir = os.path.dirname(self.config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(self.config.output_path, index=False)
        logger.info(f"Saved {len(df)} ranked candidates to {self.config.output_path}")

        # Save detailed summaries
        summaries_path = self.config.output_path.replace('.csv', '_summaries.json')
        summaries_data = []
        for cand in top_candidates[:50]:  # Top 50 detailed summaries
            summaries_data.append(cand.summary.to_dict())
        with open(summaries_path, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed summaries to {summaries_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SEMANTIC RANKING COMPLETE")
        print("=" * 60)
        print(f"Processed: {len(candidates)} candidates")
        print(f"Top candidates: {len(top_candidates)}")
        print(f"Score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
        print(f"Output: {self.config.output_path}")
        print("\nTop 10:")
        print(df[['rank', 'candidate_id', 'score', 'recommendation']].head(10).to_string(index=False))
        print("\nRecommendation distribution:")
        print(df['recommendation'].value_counts())
        print("=" * 60)

        return df


def create_ranker(
    input_path: str,
    output_path: str,
    job_description: str = "",
    job_requirements_path: Optional[str] = None,
    **kwargs
) -> SemanticRanker:
    """
    Factory function to create and configure SemanticRanker.

    Args:
        input_path: Path to candidates.jsonl
        output_path: Path for output CSV
        job_description: Raw job description (if not using requirements_path)
        job_requirements_path: Path to pre-parsed job requirements JSON
        **kwargs: Additional RankerConfig options

    Returns:
        Configured SemanticRanker instance
    """
    config = RankerConfig(
        input_path=input_path,
        output_path=output_path,
        job_description=job_description,
        job_requirements_path=job_requirements_path,
        **kwargs
    )
    return SemanticRanker(config)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Candidate Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --candidates data/candidates.jsonl --job "jd.txt" --out results.csv
  %(prog)s --candidates data/candidates.jsonl --job-requirements jobreq.json --out results.csv --top-n 100
  %(prog)s --candidates data/candidates.jsonl --job "jd.txt" --out results.csv --num-workers 4 --use-llm
        """
    )

    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl file')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--job', help='Path to job description text file OR raw JD string')
    parser.add_argument('--job-requirements', help='Path to pre-parsed job requirements JSON')
    parser.add_argument('--top-n', type=int, default=100, help='Number of top candidates to output')
    parser.add_argument('--max', type=int, help='Maximum candidates to process (for testing)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of parallel workers (0=auto)')
    parser.add_argument('--batch-size', type=int, default=32, help='Embedding batch size')
    parser.add_argument('--cache-dir', default='./cache', help='Cache directory')
    parser.add_argument('--no-faiss', action='store_true', help='Disable FAISS indexing')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for job parsing')
    parser.add_argument('--llm-api-key', help='API key for LLM service')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get job description
    job_description = ""
    if args.job:
        if os.path.isfile(args.job):
            if args.job.lower().endswith('.docx'):
                try:
                    from docx import Document
                    doc = Document(args.job)
                    job_description = "\n".join(p.text for p in doc.paragraphs)
                except ImportError:
                    raise ImportError("python-docx is required to read .docx files. Install with: pip install python-docx")
            else:
                with open(args.job, 'r', encoding='utf-8') as f:
                    job_description = f.read()
        else:
            job_description = args.job

    # Limit candidates if testing
    max_candidates = args.max

    # Create ranker
    ranker = create_ranker(
        input_path=args.candidates,
        output_path=args.out,
        job_description=job_description,
        job_requirements_path=args.job_requirements,
        top_n=args.top_n,
        max_candidates=max_candidates,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        faiss_enabled=not args.no_faiss,
        use_llm_for_job_parsing=args.use_llm,
        llm_api_key=args.llm_api_key
    )

    # Run ranking
    try:
        df = ranker.run()
        return 0
    except Exception as e:
        logger.error(f"Ranking failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())