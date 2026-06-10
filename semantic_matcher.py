"""
Semantic Matcher

Computes semantic similarity between job requirements and candidate profiles.
Uses multi-strategy matching: skill similarity, experience relevance, responsibility alignment.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from embedding_manager import EmbeddingManager, create_embedding_manager
import job_parser

logger = logging.getLogger(__name__)

# Concept descriptions for embedding-based matching
DEFAULT_CONCEPT_DESCRIPTIONS = {
    'ranking_systems': 'ranking models, learning to rank, LTR, GBDT, XGBoost, LightGBM, scoring functions, catalog ranking, position-based models, listwise ranking',
    'recommendation_systems': 'recommendation engines, collaborative filtering, content-based filtering, personalized recommendations, matrix factorization, implicit feedback, recommendation algorithms',
    'retrieval_systems': 'information retrieval, search systems, document retrieval, query processing, indexing, semantic search, dense retrieval, sparse retrieval, vector search',
    'production_ml': 'production machine learning, deploying models, model serving, inference pipelines, MLOps, CI/CD ML, monitoring models, scaling ML, latency optimization, model lifecycle',
    'evaluation_frameworks': 'A/B testing, experimentation platforms, offline evaluation, online metrics, NDCG, MRR, MAP, precision recall, holdout validation, cross validation',
    'search_relevance': 'search relevance tuning, query understanding, result ranking, BM25, TF-IDF, semantic similarity, search quality, relevance feedback',
    'product_company_experience': 'product company background, shipping products to users, working in product teams, user-focused development, end-to-end ownership, fast-paced product environment',
    'startup_builder_mindset': 'startup experience, early-stage company, builder mindset, founder background, autonomous, hands-on, rapid iteration, MVP development, entrepreneurial',
    'recruiter_workflows': 'recruiter tools, applicant tracking system ATS, candidate management platforms, interview scheduling, job board integration, talent acquisition technology',
    'candidate_matching': 'candidate job matching, resume parsing, profile matching algorithms, fit scoring systems, talent matching, job recommendation',
    'hybrid_retrieval': 'hybrid search systems, dense and sparse retrieval, multi-vector search, reciprocal rank fusion RRF, fusion methods, combining retrieval signals',
    'llm_reranking': 'LLM-based reranking, AI reranking, cross-encoders, listwise reranking, step API, GPT reranking, Claude reranking, transformer rerankers'
}

# Known product companies (non-services, product-focused)
DEFAULT_PRODUCT_COMPANIES = {
    'google', 'meta', 'netflix', 'microsoft', 'amazon', 'apple',
    'linkedin', 'openai', 'anthropic', 'deepmind', 'mistral ai',
    'cohere', 'huggingface', 'databricks', 'snowflake',
    'flipkart', 'zomato', 'swiggy', 'razorpay', 'paytm', 'phonepe',
    'cred', 'meesho', 'ola', 'freshworks', 'zoho', 'shopee',
    'byju', 'unacademy', 'airbnb', 'uber', 'doordash', 'instacart',
    'stripe', 'twilio', 'slack', 'notion', 'figma', 'adobe',
    'spotify', 'dropbox', 'salesforce', 'atlassian', 'github',
    'mongodb', 'hashicorp', 'reddit', 'pinterest', 'snap'
}

# Startup indicators in company names or job descriptions
DEFAULT_STARTUP_INDICATORS = [
    'startup', 'early stage', 'stealth', 'seed', 'series a', 'series b',
    'co-founder', 'founder', 'building from ground', '0 to 1',
    'small team', 'rapid iteration', 'agile startup', 'high growth',
    'pre-launch', 'garage', 'bootstrapped'
]

# Production indicator keywords (for heuristic boosts only)
DEFAULT_PRODUCTION_INDICATORS = {
    'deployed', 'deployment', 'production', 'serve', 'serving',
    'scaled', 'scaling', 'latency', 'throughput', 'qps',
    'monitoring', 'alerting', 'incident', 'uptime', 'sla',
    'ci/cd', 'docker', 'kubernetes', 'k8s', 'microservices',
    'api', 'endpoint', 'model serving', 'inference service'
}


@dataclass
class MatchScores:
    """Container for semantic match scores"""
    semantic_fit_score: float = 0.0  # Overall semantic similarity
    skill_similarity: float = 0.0  # Skill-to-skill similarity
    experience_similarity: float = 0.0  # Career description to responsibilities
    title_similarity: float = 0.0  # Job title match
    summary_similarity: float = 0.0  # Profile summary match
    concept_similarity: float = 0.0  # Semantic concept alignment
    concept_matches: List[Dict[str, Any]] = field(default_factory=list)
    detailed_skill_matches: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'semantic_fit_score': round(self.semantic_fit_score, 4),
            'skill_similarity': round(self.skill_similarity, 4),
            'experience_similarity': round(self.experience_similarity, 4),
            'title_similarity': round(self.title_similarity, 4),
            'summary_similarity': round(self.summary_similarity, 4),
            'detailed_skill_matches': self.detailed_skill_matches[:10],
            'reasoning': self.reasoning[:5]
        }


class SemanticMatcher:
    """
    Performs semantic matching between job requirements and candidate profiles.

    Uses multiple embedding-based strategies:
    1. Skill similarity: Job required skills vs candidate skills
    2. Experience similarity: Job responsibilities vs candidate experience descriptions
    3. Title similarity: Job title vs candidate current title
    4. Summary similarity: Job description vs candidate profile summary
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        skill_weight: float = 0.35,
        experience_weight: float = 0.25,
        title_weight: float = 0.10,
        summary_weight: float = 0.10,
        concept_weight: float = 0.20,
        concepts_config: Optional[Dict] = None
    ):
        self.embedding_manager = embedding_manager
        self.weights = {
            'skill': skill_weight,
            'experience': experience_weight,
            'title': title_weight,
            'summary': summary_weight,
            'concept': concept_weight
        }
        self._skill_cache: Dict[Tuple[str, str], float] = {}
        # Concept matching configuration
        if concepts_config is None:
            concepts_config = {}
        self.concept_descriptions = concepts_config.get('concept_descriptions', DEFAULT_CONCEPT_DESCRIPTIONS)
        self.product_companies = set(concepts_config.get('product_companies', DEFAULT_PRODUCT_COMPANIES))
        self.startup_indicators = concepts_config.get('startup_indicators', DEFAULT_STARTUP_INDICATORS)
        self.production_indicators = set(concepts_config.get('production_indicators', DEFAULT_PRODUCTION_INDICATORS))
        self._concept_embeddings: Dict[str, np.ndarray] = {}  # cache

    def match(
        self,
        job_requirements: job_parser.JobRequirements,
        candidate_profile: Dict[str, Any]
    ) -> MatchScores:
        """
        Compute semantic match scores for a candidate.

        Args:
            job_requirements: Structured job requirements
            candidate_profile: Candidate data dictionary with profile, career_history, skills

        Returns:
            MatchScores object with detailed scores
        """
        scores = MatchScores()
        candidate_id = candidate_profile.get('candidate_id', 'unknown')

        try:
            # 1. Skill similarity
            all_job_skills = list(set(
                job_requirements.required_skills +
                job_requirements.preferred_skills
            ))

            skill_score, skill_matches = self._compute_skill_similarity(
                all_job_skills,
                candidate_profile.get('skills', [])
            )
            scores.skill_similarity = skill_score
            scores.detailed_skill_matches = skill_matches[:10]

            # 2. Experience similarity
            exp_score = self._compute_experience_similarity(
                job_requirements.responsibilities,
                candidate_profile.get('career_history', [])
            )
            scores.experience_similarity = exp_score

            # 3. Title similarity
            title_score = self._compute_title_similarity(
                job_requirements.job_title,
                candidate_profile.get('profile', {}).get('current_title', '')
            )
            scores.title_similarity = title_score

            # 4. Summary/Profile similarity
            summary_score = self._compute_summary_similarity(
                job_requirements.raw_text,
                candidate_profile.get('profile', {}).get('summary', ''),
                candidate_profile.get('profile', {}).get('headline', '')
            )
            scores.summary_similarity = summary_score

            # 5. Concept similarity (semantic hiring concepts)
            concept_score, concept_matches = self._compute_concept_similarity(
                job_requirements, candidate_profile
            )
            scores.concept_similarity = concept_score
            scores.concept_matches = concept_matches

            # Weighted overall score
            scores.semantic_fit_score = (
                self.weights['skill'] * skill_score +
                self.weights['experience'] * exp_score +
                self.weights['title'] * title_score +
                self.weights['summary'] * summary_score +
                self.weights['concept'] * concept_score
            )

            # Build reasoning
            scores.reasoning = self._build_reasoning(scores, skill_matches)

            logger.debug(f"Candidate {candidate_id}: semantic_fit={scores.semantic_fit_score:.3f}")

        except Exception as e:
            logger.error(f"Error in semantic matching for {candidate_id}: {e}")
            scores.semantic_fit_score = 0.0

        return scores

    def _compute_skill_similarity(
        self,
        job_skills: List[str],
        candidate_skills: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute similarity between job required skills and candidate skills using embeddings.

        Returns:
            (similarity_score, match_details)
        """
        if not job_skills or not candidate_skills:
            return 0.0, []

        # Extract candidate skill names and prepare texts
        candidate_skill_names = [s.get('name', '') for s in candidate_skills]
        candidate_skill_texts = []
        for s in candidate_skills:
            name = s.get('name', '')
            prof = s.get('proficiency', '')
            duration = s.get('duration_months', 0)
            text = f"{name}"
            if prof:
                text += f" ({prof} proficiency, {duration} months)" if duration else f" ({prof})"
            candidate_skill_texts.append(text)

        job_skill_texts = [f"{skill}" for skill in job_skills]

        try:
            # Generate embeddings
            job_embeddings = self.embedding_manager.embed_texts(job_skill_texts)
            candidate_embeddings = self.embedding_manager.embed_texts(candidate_skill_texts)

            # Compute similarity matrix
            similarity_matrix = self.embedding_manager.compute_similarity_matrix(
                job_embeddings, candidate_embeddings
            )

            # For each job skill, find best matching candidate skill
            matches = []
            matched_job_indices = set()

            for i, job_skill in enumerate(job_skills):
                best_cand_idx = np.argmax(similarity_matrix[i])
                best_score = similarity_matrix[i][best_cand_idx]

                if best_score > 0.35:  # Threshold for meaningful match
                    matched_job_indices.add(i)
                    cand_skill = candidate_skills[best_cand_idx]
                    matches.append({
                        'job_skill': job_skill,
                        'candidate_skill': cand_skill.get('name', ''),
                        'similarity': float(best_score),
                        'candidate_proficiency': cand_skill.get('proficiency', ''),
                        'candidate_duration_months': cand_skill.get('duration_months', 0)
                    })

            # Score = percentage of job skills matched above threshold (weighted by match quality)
            if len(job_skills) > 0:
                base_score = len(matched_job_indices) / len(job_skills)
                avg_match_quality = np.mean([m['similarity'] for m in matches]) if matches else 0.0
                final_score = (
    0.6 * base_score +
    0.4 * avg_match_quality
)
            else:
                final_score = 0.0

            # Add cross-skill bonus: match unique candidate skills to multiple job skills
            unique_cand_skills = set(m['candidate_skill'] for m in matches)
            cross_bonus = min(len(unique_cand_skills) * 0.05, 0.15)

            final_score = min(final_score + cross_bonus, 1.0)

            # Sort matches by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)

            return final_score, matches

        except Exception as e:
            logger.error(f"Skill similarity computation error: {e}")
            return 0.0, []

    def _compute_experience_similarity(
        self,
        responsibilities: List[str],
        career_history: List[Dict[str, Any]]
    ) -> float:
        """Compute similarity between job responsibilities and candidate experience"""
        if not responsibilities or not career_history:
            return 0.0

        # Prepare candidate experience texts
        exp_texts = []
        for job in career_history:
            title = job.get('title', '')
            company = job.get('company', '')
            desc = job.get('description', '')
            exp_text = f"{title} at {company}. {desc}"
            exp_texts.append(exp_text[:500])  # Limit length

        # Prepare responsibility texts
        resp_texts = [resp[:500] for resp in responsibilities[:5]]

        try:
            # Generate embeddings
            resp_embeddings = self.embedding_manager.embed_texts(resp_texts)
            exp_embeddings = self.embedding_manager.embed_texts(exp_texts)

            # Compute similarity matrix
            similarity_matrix = self.embedding_manager.compute_similarity_matrix(
                resp_embeddings, exp_embeddings
            )

            # Best match for each responsibility
            similarities = []
            for i in range(len(resp_embeddings)):
                best_match = np.max(similarity_matrix[i])
                if best_match > 0.3:
                    similarities.append(best_match)

            if not similarities:
                return 0.0

            # Score = mean of top matches with quality bonus
            mean_sim = np.mean(similarities)

            # Bonus for coverage - percentage of responsibilities with good match
            coverage = len([s for s in similarities if s > 0.35]) / len(resp_embeddings)
            coverage_bonus = coverage * 0.2

            final_score = min(mean_sim + coverage_bonus, 1.0)
            return float(final_score)

        except Exception as e:
            logger.error(f"Experience similarity error: {e}")
            return 0.0

    def _compute_title_similarity(
        self,
        job_title: str,
        candidate_title: str
    ) -> float:
        """Compute similarity between job title and candidate title"""
        if not job_title or not candidate_title:
            return 0.0

        try:
            # Direct string matching for exact and partial
            job_norm = job_title.lower().strip()
            cand_norm = candidate_title.lower().strip()

            # Exact match
            if job_norm == cand_norm:
                return 1.0

            # Substring match
            if job_norm in cand_norm or cand_norm in job_norm:
                return 0.9

            # Embedding similarity
            embeddings = self.embedding_manager.embed_texts([job_title, candidate_title])
            similarity = float(self.embedding_manager.compute_similarity_matrix(
                embeddings[:1], embeddings[1:]
            )[0][0])

            # Boost for matching key terms
            job_tokens = set(job_norm.split())
            cand_tokens = set(cand_norm.split())
            overlap = len(job_tokens & cand_tokens) / max(len(job_tokens), 1)
            boosted = similarity + overlap * 0.3

            return min(boosted, 1.0)

        except Exception as e:
            logger.error(f"Title similarity error: {e}")
            return 0.0

    def _compute_summary_similarity(
        self,
        job_description: str,
        candidate_summary: str,
        candidate_headline: str
    ) -> float:
        """Compute similarity between full job description and candidate summary"""
        if not job_description:
            return 0.0

        # Combine summary and headline
        candidate_text = f"{candidate_headline}. {candidate_summary}"
        if not candidate_text.strip():
            return 0.0

        try:
            # Truncate for embedding
            job_text = job_description[:2000]
            cand_text = candidate_text[:1000]

            embeddings = self.embedding_manager.embed_texts([job_text, cand_text])
            similarity = float(self.embedding_manager.compute_similarity_matrix(
                embeddings[:1], embeddings[1:]
            )[0][0])

            return similarity

        except Exception as e:
            logger.error(f"Summary similarity error: {e}")
            return 0.0

    def _build_candidate_corpus(self, candidate_profile: Dict[str, Any]) -> str:
        """Build a comprehensive text corpus from candidate profile for concept matching."""
        parts = []
        profile = candidate_profile.get('profile', {})

        # Summary and headline
        summary = profile.get('summary', '')
        headline = profile.get('headline', '')
        if summary:
            parts.append(summary)
        if headline:
            parts.append(headline)

        # Current position
        current_title = profile.get('current_title', '')
        current_company = profile.get('current_company', '')
        if current_title:
            parts.append(f"Current role: {current_title}")
        if current_company:
            parts.append(f"at {current_company}")

        # Career history
        for job in candidate_profile.get('career_history', []):
            title = job.get('title', '')
            company = job.get('company', '')
            desc = job.get('description', '')
            job_text = f"{title} at {company}. {desc}"
            parts.append(job_text)

        # Skills
        skills = candidate_profile.get('skills', [])
        skill_names = [s.get('name', '') for s in skills if s.get('name')]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

        # Projects and achievements (if present)
        projects = candidate_profile.get('projects', [])
        for proj in projects:
            name = proj.get('name', '')
            desc = proj.get('description', '')
            if name or desc:
                parts.append(f"Project: {name}. {desc}")

        achievements = candidate_profile.get('achievements', [])
        for ach in achievements:
            parts.append(str(ach))

        return " ".join(parts).strip()

    def _compute_concept_similarity(
        self,
        job_requirements: job_parser.JobRequirements,
        candidate_profile: Dict[str, Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute semantic concept match between job requirements and candidate profile.

        Returns:
            (concept_score, concept_details)
        """
        concepts = job_requirements.semantic_concepts
        if not concepts:
            return 0.0, []

        candidate_text = self._build_candidate_corpus(candidate_profile)
        if not candidate_text:
            return 0.0, []

        # Generate candidate embedding (once)
        candidate_emb = self.embedding_manager.embed_texts([candidate_text])[0]

        similarities = []
        concept_details = []
        candidate_text_lower = candidate_text.lower()

        for concept in concepts:
            # Get concept description embedding with caching
            if concept not in self._concept_embeddings:
                desc = self.concept_descriptions.get(concept, concept)
                emb = self.embedding_manager.embed_texts([desc])[0]
                self._concept_embeddings[concept] = emb
            else:
                emb = self._concept_embeddings[concept]

            # Compute cosine similarity
            sim = self.embedding_manager.compute_similarity_matrix(
                [candidate_emb], [emb]
            )[0][0]
            sim = max(0.0, min(1.0, float(sim)))

            # Heuristic boosts (minor, max 0.1 each)
            boost = 0.0

            if concept == 'production_ml':
                # Count production indicators in candidate's career text
                count = sum(1 for kw in self.production_indicators if kw in candidate_text_lower)
                boost = min(count * 0.015, 0.10)
            elif concept == 'product_company_experience':
                # Check if candidate worked at any known product company
                for comp in self.product_companies:
                    if comp in candidate_text_lower:
                        boost = 0.12
                        break
            elif concept == 'startup_builder_mindset':
                # Check for startup indicators
                for indicator in self.startup_indicators:
                    if indicator in candidate_text_lower:
                        boost = 0.12
                        break

            final_sim = min(sim + boost, 1.0)
            similarities.append(final_sim)
            concept_details.append({
                'concept': concept,
                'base_similarity': float(sim),
                'boost': boost,
                'final_similarity': final_sim
            })

        avg_score = float(np.mean(similarities)) if similarities else 0.0
        return avg_score, concept_details

    def _build_reasoning(self, scores: MatchScores, skill_matches: List[Dict]) -> List[str]:
        """Build human-readable reasoning for the match"""
        reasoning = []

        # Skill reasoning
        if skill_matches:
            top_matches = skill_matches[:3]
            skill_names = [m['candidate_skill'] for m in top_matches]
            reasoning.append(f"Skill alignment: {len(skill_matches)} matched including {', '.join(skill_names)}")
        else:
            reasoning.append("Limited skill overlap with job requirements")

        # Title reasoning
        if scores.title_similarity > 0.7:
            reasoning.append("Title closely matches the role")
        elif scores.title_similarity > 0.4:
            reasoning.append("Title partially relevant")
        else:
            reasoning.append("Title not closely aligned")

        # Experience reasoning
        if scores.experience_similarity > 0.6:
            reasoning.append("Experience demonstrates relevant background")
        elif scores.experience_similarity > 0.3:
            reasoning.append("Some relevant experience found")
        else:
            reasoning.append("Limited experience relevance")

        # Overall
        if scores.semantic_fit_score > 0.7:
            reasoning.append("Strong overall semantic fit")
        elif scores.semantic_fit_score > 0.4:
            reasoning.append("Moderate overall fit")
        else:
            reasoning.append("Weak semantic alignment")

        return reasoning


def create_semantic_matcher(
    embedding_config: Optional[Dict] = None,
    skill_weight: float = 0.40,
    experience_weight: float = 0.30,
    title_weight: float = 0.15,
    summary_weight: float = 0.15
) -> SemanticMatcher:
    """
    Factory function to create SemanticMatcher.

    Args:
        embedding_config: Configuration for embedding manager
        skill_weight: Weight for skill similarity (default 0.40)
        experience_weight: Weight for experience similarity (default 0.30)
        title_weight: Weight for title similarity (default 0.15)
        summary_weight: Weight for summary similarity (default 0.15)

    Returns:
        SemanticMatcher instance
    """
    emb_manager = create_embedding_manager(embedding_config)
    return SemanticMatcher(
        emb_manager,
        skill_weight=skill_weight,
        experience_weight=experience_weight,
        title_weight=title_weight,
        summary_weight=summary_weight
    )


if __name__ == '__main__':
    import json
    import job_parser

    # Test
    jd = """
    Senior Machine Learning Engineer

    We are looking for a Senior ML Engineer with strong experience in PyTorch,
    TensorFlow, and production ML systems. You will build recommendation systems
    and semantic search features. Must have Python, AWS, and experience with
    vector databases like FAISS or Pinecone.
    """

    requirements = job_parser.extract_job_requirements(jd, use_llm=False)
    print("Extracted requirements:")
    print(f"  Required: {requirements.required_skills[:5]}")
    print(f"  Preferred: {requirements.preferred_skills[:5]}")

    # Test candidate
    candidate = {
        'candidate_id': 'CAND_001',
        'profile': {
            'current_title': 'ML Engineer',
            'summary': 'Experienced ML engineer with 5 years in Python, PyTorch, and building recommendation systems.',
            'headline': 'Senior ML Engineer'
        },
        'skills': [
            {'name': 'Python', 'proficiency': 'expert', 'duration_months': 60},
            {'name': 'PyTorch', 'proficiency': 'advanced', 'duration_months': 36},
            {'name': 'TensorFlow', 'proficiency': 'intermediate', 'duration_months': 24},
            {'name': 'AWS', 'proficiency': 'advanced', 'duration_months': 30},
        ],
        'career_history': [
            {
                'title': 'ML Engineer',
                'company': 'TechCorp',
                'description': 'Built recommendation systems using PyTorch and deployed to AWS SageMaker.'
            }
        ]
    }

    matcher = create_semantic_matcher()
    scores = matcher.match(requirements, candidate)
    print("\nMatch Scores:")
    print(json.dumps(scores.to_dict(), indent=2))