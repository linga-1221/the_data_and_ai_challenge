"""
Job Understanding Engine

Extracts structured requirements from raw job descriptions using LLM or rule-based fallback.

Output schema:
{
    "required_skills": [...],
    "preferred_skills": [...],
    "responsibilities": [...],
    "seniority": "",
    "industry": "",
    "behavioral_traits": [...],
    "job_title": "",
    "location_requirements": {...}
}
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

# Semantic hiring concept categories
SEMANTIC_CONCEPTS = [
    'ranking_systems',
    'recommendation_systems',
    'retrieval_systems',
    'production_ml',
    'evaluation_frameworks',
    'search_relevance',
    'product_company_experience',
    'startup_builder_mindset',
    'recruiter_workflows',
    'candidate_matching',
    'hybrid_retrieval',
    'llm_reranking'
]

# Concept keyword patterns for rule-based extraction
CONCEPT_PATTERNS = {
    'ranking_systems': [
        r'\b(ranking|rank|learning.rank|ltr|learner.rank|gbdt|xgboost|lightgbm|catalog ranking)\b',
        r'\b(rank[ing]*\s+(model|system|pipeline|algorithm))\b'
    ],
    'recommendation_systems': [
        r'\b(recommendation|recommend|personalization|collaborative.filtering|content.based)\b',
        r'\b(users?.*like|similar.*items?|suggest|recommends?)\b'
    ],
    'retrieval_systems': [
        r'\b(retrieval|retrieve|search|finder|lookup)\b',
        r'\b(dense.retrieval|sparse.retrieval|vector.retrieval)\b',
        r'\b(information.retrieval|ir)\b'
    ],
    'production_ml': [
        r'\b(production|deploy|deployment|serving|model.serving|inference|online.*inference)\b',
        r'\b(mlops|continuous.integration|ci/cd|pipeline|airflow|dagster)\b',
        r'\b(monitoring|scaling|latency|throughput|qps)\b'
    ],
    'evaluation_frameworks': [
        r'\b(evaluation|metric|benchmark|measure|score|ndcg|mrr|map|precision|recall)\b',
        r'\b(a/b.test|ab.test|experiment|holdout|cross.validation)\b'
    ],
    'search_relevance': [
        r'\b(search.relevance|relevance|ranking|bm25|tf.idf|semantic.search)\b',
        r'\b(queries?|query|keyword.search|full.text)\b'
    ],
    'product_company_experience': [
        r'\b(product.company|product.team|product.builder|startup.experience)\b',
        r'\b(fast.paced|agile|ship.products|user.facing)\b',
        r'\b(end.to.end|full.cycle|own.products)\b'
    ],
    'startup_builder_mindset': [
        r'\b(startup|builder|founder|early.stage|lean|mvp|iterate)\b',
        r'\b(autonomous|self.starter|ownership|bias.for.action)\b'
    ],
    'recruiter_workflows': [
        r'\b(recruiter|recruiting|talent.acquisition|hiring|sourcing|candidate.sourcing)\b',
        r'\b(ats|applicant.tracking|job.board|recruitment.platform)\b'
    ],
    'candidate_matching': [
        r'\b(candidate.matching|candidate.job|job.match|fit.score)\b',
        r'\b(matching.algorithm|alignment|compatibility)\b'
    ],
    'hybrid_retrieval': [
        r'\b(hybrid|dense.sparse|multi.vector|rerank|re-rank)\b',
        r'\b(keyword.+semantic|bm25.+embedding|fusion|reciprocal.rank)\b'
    ],
    'llm_reranking': [
        r'\b(llm|large.language.model|gpt|claude|reranking|rerank)\b',
        r'\b(ai.rank|intelligent.rerank|llm.judge|cross.encoder)\b'
    ]
}

# Semantic hiring concept categories
SEMANTIC_CONCEPTS = [
    'ranking_systems',
    'recommendation_systems',
    'retrieval_systems',
    'production_ml',
    'evaluation_frameworks',
    'search_relevance',
    'product_company_experience',
    'startup_builder_mindset',
    'recruiter_workflows',
    'candidate_matching',
    'hybrid_retrieval',
    'llm_reranking'
]

# Concept keyword patterns for rule-based extraction
CONCEPT_PATTERNS = {
    'ranking_systems': [
        r'\b(ranking|rank|learning.rank|ltr|learner.rank|gbdt|xgboost|lightgbm|catalog ranking)\b',
        r'\b(rank[ing]*\s+(model|system|pipeline|algorithm))\b',
        r'\b(positioning|ordre|scoring)\b'
    ],
    'recommendation_systems': [
        r'\b(recommendation|recommend|personalization|collaborative.filtering|content.based)\b',
        r'\b(users?.*like|similar.*items?|suggest|recommends?)\b',
        r'\b(als|surprise|implicit)\b'
    ],
    'retrieval_systems': [
        r'\b(retrieval|retrieve|search|finder|lookup)\b',
        r'\b(dense.retrieval|sparse.retrieval|vector.retrieval)\b',
        r'\b(information.retrieval|ir)\b'
    ],
    'production_ml': [
        r'\b(production|deploy|deployment|serving|model.serving|inference|online.*inference)\b',
        r'\b(mlops|continuous.integration|ci/cd|pipeline|airflow|dagster)\b',
        r'\b(monitoring|scaling|latency|throughput|qps)\b'
    ],
    'evaluation_frameworks': [
        r'\b(evaluation|metric|benchmark|measure|score|ndcg|mrr|map|precision|recall)\b',
        r'\b(a/b.test|ab.test|experiment|holdout|cross.validation)\b',
        r'\b(offline.evaluation|online.evaluation|human.eval)\b'
    ],
    'search_relevance': [
        r'\b(search.relevance|relevance|ranking|bm25|tf.idf|semantic.search)\b',
        r'\b(queries?|query|keyword.search|full.text)\b',
        r'\b(ranking|candidate.jd.matching)\b'
    ],
    'product_company_experience': [
        r'\b(product.company|product.team|product.builder|startup.experience)\b',
        r'\b(fast.paced|agile|ship.products|user.facing)\b',
        r'\b(end.to.end|full.cycle|own.products)\b'
    ],
    'startup_builder_mindset': [
        r'\b(startup|builder|founder|early.stage|lean|mvp|iterate)\b',
        r'\b(autonomous|self.starter|ownership|bias.for.action)\b',
        r'\b(impact|move.fast|break.things|experiment)\b'
    ],
    'recruiter_workflows': [
        r'\b(recruiter|recruiting|talent.acquisition|hiring|sourcing|candidate.sourcing)\b',
        r'\b(ats|applicant.tracking|job.board|recruitment.platform)\b',
        r'\b(scheduler|interview|assessment|screening)\b'
    ],
    'candidate_matching': [
        r'\b(candidate.matching|candidate.job|job.match|fit.score)\b',
        r'\b(matching.algorithm|alignment|compatibility)\b',
        r'\b(resume.parsing|profile.match|skill.match)\b'
    ],
    'hybrid_retrieval': [
        r'\b(hybrid|dense.sparse|multi.vector|rerank|re-rank)\b',
        r'\b(keyword.+semantic|bm25.+embedding|fusion|reciprocal.rank)\b',
        r'\b(rrf|hybrid.search)\b'
    ],
    'llm_reranking': [
        r'\b(llm|large.language.model|gpt|claude|reranking|rerank)\b',
        r'\b(ai.rank|intelligent.rerank|llm.judge|cross.encoder)\b',
        r'\b(step.api|cross.encoder|listwise|pairwise)\b'
    ]
}


@dataclass
class JobRequirements:
    """Structured job requirements extracted from job description"""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    seniority: str = ""
    industry: str = ""
    behavioral_traits: List[str] = field(default_factory=list)
    job_title: str = ""
    location_requirements: Dict[str, Any] = field(default_factory=dict)
    experience_years_min: Optional[float] = None
    experience_years_max: Optional[float] = None
    education_requirements: List[str] = field(default_factory=list)
    raw_text: str = ""
    semantic_concepts: List[str] = field(default_factory=list)
    concept_evidence: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'JobRequirements':
        """Load from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class JobParser:
    """Parses job descriptions into structured requirements"""

    # Skill categories commonly found in tech jobs
    TECH_SKILL_PATTERNS = {
        'programming_languages': r'\b(python|java|javascript|typescript|c\+\+|go|rust|scala|ruby|php|swift|kotlin)\b',
        'ml_frameworks': r'\b(tensorflow|pytorch|keras|jax|mxnet|caffe)\b',
        'ml_libs': r'\b(scikit-learn|numpy|pandas|scipy|matplotlib|seaborn)\b',
        'cloud': r'\b(aws|azure|gcp|google cloud|amazon web services|cloudformation|terraform)\b',
        'containers': r'\b(docker|kubernetes|k8s|helm|istio)\b',
        'databases': r'\b(sql|postgresql|mysql|mongodb|redis|cassandra|elasticsearch|opensearch)\b',
        'vector_dbs': r'\b(pinecone|weaviate|qdrant|milvus|faiss|pgvector|chroma)\b',
        'streaming': r'\b(kafka|kinesis|pubsub|pulsar|spark|flink)\b',
        'orchestration': r'\b(airflow|dagster|prefect|luigi)\b',
        'retrieval': r'\b(elasticsearch|solr|search|retrieval|reranking|re-ranking)\b',
    }

    SENIORITY_PATTERNS = {
        'junior': r'\b(junior|entry.level|early.career|graduate|intern|associate)\b',
        'mid': r'\b(mid|intermediate|2.?5|3.?5|5.?7)\b',
        'senior': r'\b(senior|lead|principal|staff|architect|7.?10|8.?12)\b',
        'manager': r'\b(manager|director|vp|head.of)\b',
    }

    def __init__(self, use_llm: bool = True, llm_api_key: Optional[str] = None):
        """
        Initialize job parser.

        Args:
            use_llm: Use LLM for extraction if True, otherwise rule-based only
            llm_api_key: API key for LLM service (Anthropic/OpenAI)
        """
        self.use_llm = use_llm and llm_api_key is not None
        self.llm_api_key = llm_api_key
        self.llm_client = None

        if self.use_llm:
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=llm_api_key)
                logger.info("Initialized Anthropic client for job parsing")
            except ImportError:
                logger.warning("Anthropic package not installed. Install with: pip install anthropic")
                self.use_llm = False

    def _extract_semantic_concepts(self, text: str) -> List[str]:
        """
        Extract semantic concepts from job description text using pattern matching.
        Returns list of concept names that are present.
        """
        concepts_found = set()
        for concept, patterns in CONCEPT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.I):
                    concepts_found.add(concept)
                    break  # Found one pattern for this concept, move on
        return sorted(concepts_found)

    def parse(self, job_description: str) -> JobRequirements:
        """
        Parse job description into structured requirements.

        Args:
            job_description: Raw job description text

        Returns:
            JobRequirements object with extracted structured data
        """
        if self.use_llm and self.llm_client:
            try:
                return self._parse_with_llm(job_description)
            except Exception as e:
                logger.error(f"LLM parsing failed: {e}. Falling back to rule-based.")
                return self._parse_rule_based(job_description)
        else:
            return self._parse_rule_based(job_description)

    def _parse_with_llm(self, job_description: str) -> JobRequirements:
        """Use Claude API to extract structured requirements"""
        prompt = """You are an expert job description analyzer. Extract structured information from the job description.

Output valid JSON with these fields:
{
    "required_skills": ["skill1", "skill2", ...],  // Must-have skills (specific technologies: Python, TensorFlow, AWS, etc.)
    "preferred_skills": ["skill1", "skill2", ...], // Nice-to-have skills
    "responsibilities": ["resp1", ...],  // Key responsibilities
    "seniority": "junior|mid|senior|lead|manager|executive",
    "industry": "industry name or 'any'",
    "behavioral_traits": ["trait1", ...],  // Desired behaviors/cultural traits
    "job_title": "extracted job title",
    "location_requirements": {"type": "remote|onsite|hybrid", "locations": []},
    "experience_years_min": number or null,
    "experience_years_max": number or null,
    "education_requirements": ["degree1", ...],
    "semantic_concepts": ["concept1", "concept2", ...]  // ABSTRACT CONCEPTS from this list:
        // ranking_systems, recommendation_systems, retrieval_systems, production_ml,
        // evaluation_frameworks, search_relevance, product_company_experience,
        // startup_builder_mindset, recruiter_workflows, candidate_matching,
        // hybrid_retrieval, llm_reranking
}

SEMANTIC CONCEPTS - SELECT ALL THAT APPLY:

- ranking_systems: The role involves building/optimizing ranking models (LTR, GBDTs, scoring algorithms, catalog ranking)
- recommendation_systems: Building/managing recommendation engines, personalization, collaborative filtering
- retrieval_systems: Working on search/retrieval infrastructure (dense/sparse retrieval, information retrieval)
- production_ml: Deploying ML models to production, MLOps, serving infrastructure, monitoring, scaling
- evaluation_frameworks: A/B testing, metrics (NDCG, MRR), experimentation, offline evaluation
- search_relevance: Focus on improving search result quality, relevance tuning, query understanding
- product_company_experience: JD emphasizes product company background, startup experience, shipping to users
- startup_builder_mindset: Seeks entrepreneurial, hands-on, autonomous builders comfortable with ambiguity
- recruiter_workflows: Role involves recruiter tools, talent acquisition systems, ATS, candidate management
- candidate_matching: Building systems that match candidates to jobs/resumes to positions
- hybrid_retrieval: Combining multiple retrieval methods (dense+sparse, multi-vector, fusion approaches)
- llm_reranking: Using LLMs for reranking, AI-based relevance, cross-encoders, listwise reranking

IMPORTANT: Choose concepts based on WHAT THE JOB IS ABOUT, not just keyword presence. A job that discusses "building recommendation systems" should get "recommendation_systems" even if the word "ranking" is not present.

Guidelines for concepts:
- Look at responsibilities, required skills context, and overall focus
- "production_ml" = emphasis on deployment, serving, monitoring, not just "I use PyTorch"
- "product_company_experience" = JD says "product company experience", "startup experience", "shipping products"
- "startup_builder_mindset" = JD seeks "founder mindset", "early stage", "hands-on", "autonomous"
- Include multiple concepts if applicable (typically 2-5)

Job Description:
{job_description}

Return ONLY valid JSON, no other text.
""".format(job_description=job_description[:15000])  # Limit length

        try:
            response = self.llm_client.messages.create(
                model="claude-opus-4-8-20250515",  # Latest Claude model
                max_tokens=2000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            # Find JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(content)

            # Create JobRequirements
            req = JobRequirements(
                required_skills=data.get('required_skills', []),
                preferred_skills=data.get('preferred_skills', []),
                responsibilities=data.get('responsibilities', []),
                seniority=data.get('seniority', ''),
                industry=data.get('industry', ''),
                behavioral_traits=data.get('behavioral_traits', []),
                job_title=data.get('job_title', ''),
                location_requirements=data.get('location_requirements', {}),
                experience_years_min=data.get('experience_years_min'),
                experience_years_max=data.get('experience_years_max'),
                education_requirements=data.get('education_requirements', []),
                raw_text=job_description,
                semantic_concepts=data.get('semantic_concepts', []),
                concept_evidence=data.get('concept_evidence', {})
            )
            logger.info(f"LLM extraction: {len(req.required_skills)} required, {len(req.preferred_skills)} preferred skills")
            return req

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            raise

    def _parse_rule_based(self, job_description: str) -> JobRequirements:
        """Rule-based extraction fallback"""
        text = job_description.lower()
        req = JobRequirements(raw_text=job_description)

        # Extract job title (usually first few lines)
        lines = job_description.split('\n')[:5]
        for line in lines:
            if any(title in line.lower() for title in ['engineer', 'developer', 'scientist', 'analyst', 'designer', 'manager']):
                req.job_title = line.strip()[:100]
                break

        # Extract skills using pattern matching
        all_skills = set()
        required_skills = set()
        preferred_skills = set()

        # Look for required/preferred sections
        has_required_section = bool(re.search(r'(required|must have|essential|key skills|technologies)', text, re.I))
        has_preferred_section = bool(re.search(r'(preferred|nice to have|desirable|plus|bonus)', text, re.I))

        # Sets to track explicit categorizations
        explicit_required = set()
        explicit_preferred = set()

        # Parse explicit "Must have:" / "Required:" lines
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith('must have') or line_lower.startswith('required'):
                if ':' in line:
                    skills_str = line.split(':', 1)[1].strip()
                    # Split on commas and 'and'
                    parts = [p.strip() for p in skills_str.replace(' and', ',').split(',') if p.strip()]
                    for skill in parts:
                        skill_clean = skill.strip()
                        if skill_clean:
                            # Use same normalization as pattern matching for consistency
                            skill_norm = skill_clean.upper() if len(skill_clean) <= 3 else skill_clean.title()
                            explicit_required.add(skill_norm)
                            required_skills.add(skill_norm)
                            all_skills.add(skill_norm)
            elif line_lower.startswith('nice to have') or line_lower.startswith('preferred'):
                if ':' in line:
                    skills_str = line.split(':', 1)[1].strip()
                    parts = [p.strip() for p in skills_str.replace(' and', ',').split(',') if p.strip()]
                    for skill in parts:
                        skill_clean = skill.strip()
                        if skill_clean:
                            skill_norm = skill_clean.upper() if len(skill_clean) <= 3 else skill_clean.title()
                            explicit_preferred.add(skill_norm)
                            preferred_skills.add(skill_norm)
                            all_skills.add(skill_norm)

        # Extract from patterns
        for category, pattern in self.TECH_SKILL_PATTERNS.items():
            matches = re.findall(pattern, text, re.I)
            for match in matches:
                skill = match.upper() if len(match) <= 3 else match.title()
                all_skills.add(skill)
                if skill in explicit_required:
                    required_skills.add(skill)
                elif skill in explicit_preferred:
                    preferred_skills.add(skill)
                elif not has_preferred_section:
                    required_skills.add(skill)
                else:
                    preferred_skills.add(skill)

        req.required_skills = sorted(required_skills) if required_skills else sorted(all_skills)[:20]
        req.preferred_skills = sorted(preferred_skills)

        # Extract responsibilities (bullet points, numbered lists)
        responsibilities = []
        for line in job_description.split('\n'):
            line = line.strip()
            if re.match(r'^[\d\-\*\••]\s+', line) or (len(line) > 20 and ':' in line[:50]):
                if not any(skip in line.lower() for skip in [
    'salary',
    'benefits',
    'qualification',
    'required:',
    'preferred:',
    'job description:',
    'company:',
    'location:',
    'employment type:',
    'notice period:'
]):
                    responsibilities.append(line[:200])
        # If insufficient responsibilities, supplement with sentences containing action verbs
        if len(responsibilities) < 5:
            sentences = re.split(r'[.!?]+', job_description)
            action_verbs = ['build', 'design', 'develop', 'implement', 'create', 'maintain', 'optimize', 'manage', 'lead', 'collaborate', 'work on', 'deploy', 'improve', 'engineer', 'architect']
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 20:
                    continue
                if any(verb in sent_clean.lower() for verb in action_verbs):
                    if sent_clean not in responsibilities:
                        responsibilities.append(sent_clean[:200])
                        if len(responsibilities) >= 10:
                            break
        req.responsibilities = responsibilities[:10]

        # Seniority detection
        for level, pattern in self.SENIORITY_PATTERNS.items():
            if re.search(pattern, text, re.I):
                req.seniority = level
                break

        # Experience extraction
        exp_match = re.search(r'(\d+)\+?\s*(?:to|-|–)?\s*(\d+)?\s*years?\s+(?:of)?\s+experience', text, re.I)
        if exp_match:
            req.experience_years_min = float(exp_match.group(1))
            if exp_match.group(2):
                req.experience_years_max = float(exp_match.group(2))
        else:
            # Look for single number
            single_exp = re.search(r'(\d+)\+?\s*years?\s+experience', text, re.I)
            if single_exp:
                req.experience_years_min = float(single_exp.group(1))

        # Industry detection
        industries = ['finance', 'healthcare', 'e-commerce', 'retail', 'gaming', 'edtech', 'fintech']
        for ind in industries:
            if ind in text or ind.replace('-', ' ') in text:
                req.industry = ind
                break

        # Behavioral traits
        behavioral_keywords = ['collaborative', 'innovative', 'fast-paced', 'agile', 'customer-focused',
                              'mission-driven', 'data-driven', 'outcomes-focused', 'team player']
        req.behavioral_traits = [kw for kw in behavioral_keywords if kw in text]

        # Extract semantic concepts using pattern matching
        req.semantic_concepts = self._extract_semantic_concepts(text)

        logger.info(f"Rule-based extraction: {len(req.required_skills)} required, {len(req.preferred_skills)} preferred")
        return req


def extract_job_requirements(job_description: str, use_llm: bool = False, llm_api_key: Optional[str] = None) -> JobRequirements:
    """
    Convenience function to extract job requirements.

    Args:
        job_description: Raw job description text
        use_llm: Whether to use LLM extraction
        llm_api_key: API key for LLM service

    Returns:
        JobRequirements object
    """
    parser = JobParser(use_llm=use_llm, llm_api_key=llm_api_key)
    return parser.parse(job_description)


if __name__ == '__main__':
    # Example usage
    sample_jd = """
    Senior AI Engineer

    We are looking for a Senior AI Engineer to join our team. You will be responsible for building
    production-grade machine learning systems, including semantic search and recommendation systems.

    Required Skills:
    - 5+ years of experience in ML engineering
    - Python, PyTorch, TensorFlow
    - Experience with vector databases (FAISS, Pinecone, or similar)
    - Strong background in information retrieval and semantic search
    - Experience deploying models to production

    Preferred Skills:
    - Experience with LLMs and RAG systems
    - Knowledge of AWS or GCP
    - Familiarity with Docker and Kubernetes

    Responsibilities:
    - Design and implement AI/ML systems
    - Optimize model performance and latency
    - Collaborate with product and engineering teams
    - Mentor junior engineers

    We are a fast-growing fintech company looking for collaborative individuals who thrive in
    agile environments.
    """

    requirements = extract_job_requirements(sample_jd, use_llm=False)
    print(json.dumps(requirements.to_dict(), indent=2, ensure_ascii=False))
