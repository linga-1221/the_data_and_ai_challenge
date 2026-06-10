"""
Recruiter Summary Generator

Generates human-readable summaries for recruiters:
- Top strengths (bullet points)
- Missing skills (vs job requirements)
- Risk factors (concerns)
- Hiring recommendation (Strong/Moderate/Weak match)
- Talent insights
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from job_parser import JobRequirements
import trust_scorer

logger = logging.getLogger(__name__)


@dataclass
class RecruiterSummary:
    """Comprehensive recruiter-facing summary"""
    candidate_id: str = ""
    candidate_name: str = ""
    current_title: str = ""
    overall_score: float = 0.0  # 0-1
    recommendation: str = ""  # Strong/Moderate/Weak/Reject
    summary_text: str = ""
    top_strengths: List[str] = field(default_factory=list)
    skill_gaps: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    talent_insights: List[str] = field(default_factory=list)
    semantic_fit_score: float = 0.0
    experience_score: float = 0.0
    trust_level: str = "high"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output"""
        return {
            'candidate_id': self.candidate_id,
            'candidate_name': self.candidate_name,
            'current_title': self.current_title,
            'overall_score': round(self.overall_score, 4),
            'recommendation': self.recommendation,
            'summary': self.summary_text[:500],
            'top_strengths': self.top_strengths[:5],
            'skill_gaps': self.skill_gaps[:8],
            'risk_factors': self.risk_factors[:5],
            'talent_insights': self.talent_insights[:5],
            'semantic_fit_score': round(self.semantic_fit_score, 4),
            'experience_score': round(self.experience_score, 4),
            'trust_level': self.trust_level
        }


class RecruiterSummarizer:
    """
    Generates recruiter-friendly summaries with actionable insights.

    Uses template-based generation (no LLM required for speed and reliability),
    but can optionally integrate LLM for richer narratives.
    """

    # Score thresholds for recommendations
    SCORE_THRESHOLDS = {
        'strong': 0.70,
        'moderate': 0.50,
        'weak': 0.30,
        'reject': 0.0
    }

    def __init__(self, use_llm: bool = False, llm_api_key: Optional[str] = None):
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key
        self.llm_client = None

        if self.use_llm and llm_api_key:
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=llm_api_key)
                logger.info("Initialized Anthropic client for summaries")
            except ImportError:
                self.use_llm = False

    def generate_summary(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: JobRequirements,
        semantic_scores: Any,  # MatchScores from semantic_matcher
        trust_score: trust_scorer.TrustScore,
        final_score: float,
        skill_matches: List[Dict[str, Any]]
    ) -> RecruiterSummary:
        """
        Generate comprehensive recruiter summary.

        Args:
            candidate_profile: Candidate data dictionary
            job_requirements: Structured job requirements
            semantic_scores: Semantic match scores object
            trust_score: Trust assessment
            final_score: Overall ranking score (0-1)
            skill_matches: List of matched skills with details

        Returns:
            RecruiterSummary object
        """
        summary = RecruiterSummary(
            candidate_id=candidate_profile.get('candidate_id', ''),
            candidate_name=candidate_profile.get('profile', {}).get('anonymized_name', ''),
            current_title=candidate_profile.get('profile', {}).get('current_title', ''),
            overall_score=final_score,
            semantic_fit_score=semantic_scores.semantic_fit_score,
            experience_score=semantic_scores.experience_similarity,
            trust_level=trust_score.trust_level
        )

        # Determine recommendation based on score and trust
        summary.recommendation = self._determine_recommendation(
            final_score,
            trust_score.overall_score,
            semantic_scores.semantic_fit_score
        )

        # Build summary sections
        summary.top_strengths = self._extract_strengths(
            candidate_profile,
            job_requirements,
            semantic_scores,
            skill_matches
        )

        summary.skill_gaps = self._identify_skill_gaps(
            job_requirements.required_skills,
            job_requirements.preferred_skills,
            candidate_profile.get('skills', []),
            skill_matches
        )

        summary.risk_factors = self._extract_risk_factors(
            trust_score,
            candidate_profile,
            job_requirements
        )

        summary.talent_insights = self._generate_talent_insights(
            candidate_profile,
            semantic_scores,
            trust_score
        )

        # Generate narrative summary
        summary.summary_text = self._compose_summary_text(summary)

        return summary

    def _determine_recommendation(
        self,
        overall_score: float,
        trust_score: float,
        semantic_score: float
    ) -> str:
        """Determine hiring recommendation based on scores"""
        # Adjust for trust
        effective_score = overall_score * trust_score

        if effective_score >= self.SCORE_THRESHOLDS['strong'] and semantic_score >= 0.6:
            return "Strong Match"
        elif effective_score >= self.SCORE_THRESHOLDS['moderate']:
            return "Consider"
        elif effective_score >= self.SCORE_THRESHOLDS['weak']:
            return "Borderline"
        else:
            return "Not Recommended"

    def _extract_strengths(
        self,
        candidate_profile: Dict,
        job_requirements: JobRequirements,
        semantic_scores: Any,
        skill_matches: List[Dict]
    ) -> List[str]:
        """Extract candidate's top strengths relative to job"""
        strengths = []
        profile = candidate_profile.get('profile', {})
        career_history = candidate_profile.get('career_history', [])
        skills = candidate_profile.get('skills', [])

        # Experience-based strengths
        yoe = profile.get('years_of_experience', 0)
        if yoe >= 5:
            strengths.append(f"{int(yoe)} years of professional experience")
        elif yoe >= 3:
            strengths.append(f"{int(yoe)} years of hands-on experience")

        # Title alignment
        current_title = profile.get('current_title', '').lower()
        if any(term in current_title for term in ['senior', 'lead', 'principal', 'staff']):
            strengths.append("Senior-level professional experience")

        # Company pedigree
        companies = [job.get('company', '').lower() for job in career_history]
        prestige_keywords = ['google', 'meta', 'amazon', 'microsoft', 'apple', 'netflix',
                           ' flipkart', 'razorpay', 'paytm', 'zomato', 'swiggy', 'ola']
        has_prestige = any(any(p in c for p in prestige_keywords) for c in companies)
        if has_prestige:
            strengths.append("Experience at leading tech companies")

        # Skill matches (top 3)
        if skill_matches:
            top_skills = [m['candidate_skill'] for m in skill_matches[:3]]
            strengths.append(f"Strong skills: {', '.join(top_skills)}")

        # Production experience
        all_descriptions = ' '.join(job.get('description', '') for job in career_history).lower()
        prod_keywords = ['deployed', 'production', 'scale', 'users', 'serving']
        if sum(1 for kw in prod_keywords if kw in all_descriptions) >= 2:
            strengths.append("Proven production deployment experience")

        # Behavioral signals
        redrob = candidate_profile.get('redrob_signals', {})
        if redrob.get('open_to_work_flag'):
            strengths.append("Actively looking for new opportunities")
        if redrob.get('recruiter_response_rate', 0) > 0.7:
            strengths.append("High recruiter response rate")

        return strengths[:5] if strengths else ["Relevant experience profile"]

    def _identify_skill_gaps(
        self,
        required_skills: List[str],
        preferred_skills: List[str],
        candidate_skills: List[Dict],
        skill_matches: List[Dict]
    ) -> List[str]:
        """Identify skills from JD that candidate lacks"""
        matched_skill_names = set(m['candidate_skill'].lower() for m in skill_matches)
        candidate_skill_names = set(s.get('name', '').lower() for s in candidate_skills)

        # Find required skills not matched
        gaps = []
        for skill in required_skills:
            skill_lower = skill.lower()
            # Check if any variation of the skill is matched
            is_matched = any(
                skill_lower in matched.lower() or matched.lower() in skill_lower
                for matched in matched_skill_names | candidate_skill_names
            )
            if not is_matched:
                gaps.append(skill)

        # Also note preferred skills gaps if score is low
        if len(matched_skill_names) < len(required_skills) * 0.5:
            for skill in preferred_skills[:3]:
                if skill.lower() not in candidate_skill_names:
                    gaps.append(f"{skill} (preferred)")

        return gaps[:8]

    def _extract_risk_factors(
        self,
        trust_score: trust_scorer.TrustScore,
        candidate_profile: Dict,
        job_requirements: JobRequirements
    ) -> List[str]:
        """Extract risk factors from trust assessment"""
        risks = []

        if trust_score.trust_level == "critical":
            risks.append("Critical trust issues detected - requires manual verification")

        for flag in getattr(trust_score, 'flags', []):
            if flag['severity'] in ['high', 'critical']:
                risks.append(flag['message'])

        # Additional risks based on profile
        profile = candidate_profile.get('profile', {})
        yoe = profile.get('years_of_experience', 0)
        if job_requirements.experience_years_min is not None and yoe < job_requirements.experience_years_min:
            risks.append(f"Below required experience: {yoe} years vs {job_requirements.experience_years_min}+ required")

        # Check location
        location = profile.get('location', '').lower()
        if not any(city in location for city in ['india', 'pune', 'hyderabad', 'bangalore', 'mumbai', 'delhi', 'noida']):
            risks.append("International location - may have relocation constraints")

        return risks[:5]

    def _generate_talent_insights(
        self,
        candidate_profile: Dict,
        semantic_scores: Any,
        trust_score: trust_scorer.TrustScore
    ) -> List[str]:
        """Generate deeper talent insights"""
        insights = []
        profile = candidate_profile.get('profile', {})

        # Career stability
        career_history = candidate_profile.get('career_history', [])
        if len(career_history) > 0:
            avg_tenure = sum(j.get('duration_months', 0) for j in career_history) / len(career_history)
            if avg_tenure > 36:
                insights.append("Shows long-term commitment to employers")
            elif avg_tenure < 18 and len(career_history) > 3:
                insights.append("History of shorter engagements - verify stability")

        # Profile signals from redrob
        redrob = candidate_profile.get('redrob_signals', {})
        if redrob.get('search_appearance_30d', 0) > 100:
            insights.append("High recruiter interest (frequently appeared in searches)")
        if redrob.get('saved_by_recruiters_30d', 0) > 10:
            insights.append("Often saved by recruiters")

        # Trust implications
        if trust_score.trust_level == "high":
            insights.append("Clean profile with strong trust indicators")
        elif trust_score.trust_level == "medium":
            insights.append("Profile passes basic validation - review flagged items")

        # Semantic fit observation
        if semantic_scores.semantic_fit_score > 0.8:
            insights.append("Excellent semantic alignment with role requirements")
        elif semantic_scores.semantic_fit_score > 0.6:
            insights.append("Good semantic fit with some alignment gaps")

        return insights[:5]

    def _compose_summary_text(self, summary: RecruiterSummary) -> str:
        """Compose narrative summary for the candidate"""
        parts = []

        # Header
        parts.append(f"Candidate: {summary.candidate_name} ({summary.current_title})")
        parts.append(f"Overall Score: {summary.overall_score:.2f} | Recommendation: {summary.recommendation}")
        parts.append(f"Trust Level: {summary.trust_level.title()}")
        parts.append("")

        # Strengths
        if summary.top_strengths:
            parts.append("Key Strengths:")
            for strength in summary.top_strengths[:3]:
                parts.append(f"  - {strength}")
            parts.append("")

        # Skill gaps
        if summary.skill_gaps:
            parts.append("Skill Gaps (Critical):")
            for gap in summary.skill_gaps[:4]:
                parts.append(f"  - {gap}")
            parts.append("")

        # Risk factors
        if summary.risk_factors:
            parts.append("Risk Factors:")
            for risk in summary.risk_factors[:3]:
                parts.append(f"  - {risk}")
            parts.append("")

        # Insights
        if summary.talent_insights:
            parts.append("Talent Insights:")
            for insight in summary.talent_insights[:2]:
                parts.append(f"  - {insight}")

        return "\n".join(parts)


def create_summarizer(use_llm: bool = False, llm_api_key: Optional[str] = None) -> RecruiterSummarizer:
    """Factory function to create RecruiterSummarizer"""
    return RecruiterSummarizer(use_llm=use_llm, llm_api_key=llm_api_key)


if __name__ == '__main__':
    # Test
    from semantic_matcher import MatchScores

    job_req = JobRequirements(
        required_skills=['Python', 'TensorFlow', 'AWS'],
        preferred_skills=['Kubernetes', 'Docker'],
        responsibilities=['Build ML systems', 'Deploy to production'],
        job_title='ML Engineer'
    )

    candidate = {
        'candidate_id': 'CAND_001',
        'profile': {
            'anonymized_name': 'John Doe',
            'current_title': 'ML Engineer',
            'years_of_experience': 5
        },
        'skills': [
            {'name': 'Python', 'proficiency': 'expert', 'duration_months': 60},
            {'name': 'TensorFlow', 'proficiency': 'advanced', 'duration_months': 36}
        ],
        'career_history': [
            {'company': 'TechCorp', 'title': 'ML Engineer', 'duration_months': 36,
             'description': 'Built ML systems in production with AWS'}
        ],
        'redrob_signals': {
            'open_to_work_flag': True,
            'recruiter_response_rate': 0.8
        }
    }

    semantic_scores = MatchScores(
        semantic_fit_score=0.72,
        skill_similarity=0.85,
        experience_similarity=0.65,
        title_similarity=0.90,
        summary_similarity=0.60
    )

    trust = trust_scorer.TrustScore(overall_score=0.88, trust_level='high')

    summarizer = create_summarizer()
    summary = summarizer.generate_summary(candidate, job_req, semantic_scores, trust, 0.82, [])
    print("Recruiter Summary:")
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
