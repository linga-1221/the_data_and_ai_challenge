"""
Trust Scorer

Advanced trustworthiness detection for candidate profiles.

Detects:
- Skill stuffing (too many claimed skills)
- Unrealistic timelines (impossible dates, negative tenure)
- Inflated experience (YoE mismatch)
- Contradictory claims (skills not mentioned in experience)
- Inconsistent career progression
- Suspicious activity patterns
- Profile completeness anomalies

Returns a trust score between 0.3 and 1.0.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrustScore:
    """Container for trust score and breakdown"""
    overall_score: float = 1.0  # 0.3-1.0 multiplier
    flag_count: int = 0
    flags: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trust_level: str = "high"  # high, medium, low, critical

    def to_dict(self) -> Dict:
        return {
            'overall_score': round(self.overall_score, 4),
            'flag_count': self.flag_count,
            'flags': self.flags[:10],
            'warnings': self.warnings[:5],
            'trust_level': self.trust_level
        }


class TrustScorer:
    """
    Comprehensive trust scoring for candidate profiles.

    Trust score is a multiplier (0.3-1.0) applied to final ranking score.
    Lower scores indicate more suspicious profiles.
    """

    # Thresholds for various checks
    MAX_ADVANCED_ZERO_DURATION_DEFAULT = 5
    MAX_YOE_DEFAULT = 40
    MAX_TITLE_CHANGES_PER_YEAR = 0.5  # Too many title changes may indicate job hopping
    MIN_SKILL_DURATION_FOR_PROFICIENCY = {
        'advanced': 12,
        'expert': 24
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_advanced_zero_duration = self.config.get(
            'honeypot', {}
        ).get('max_advanced_zero_duration', self.MAX_ADVANCED_ZERO_DURATION_DEFAULT)
        self.max_yoe = self.config.get('honeypot', {}).get('max_yoe', self.MAX_YOE_DEFAULT)
        self.services_companies = set(self.config.get('services_companies', []))

    def calculate_trust_score(
        self,
        candidate_profile: Dict[str, Any],
        job_requirements: Optional[Any] = None
    ) -> TrustScore:
        """
        Calculate overall trust score for a candidate.

        Args:
            candidate_profile: Full candidate data dictionary
            job_requirements: Optional job requirements for relevance checks

        Returns:
            TrustScore object with overall score and flags
        """
        trust = TrustScore()
        candidate_id = candidate_profile.get('candidate_id', 'unknown')

        try:
            profile = candidate_profile.get('profile', {})
            skills = candidate_profile.get('skills', [])
            career_history = candidate_profile.get('career_history', [])
            education = candidate_profile.get('education', [])
            redrob_signals = candidate_profile.get('redrob_signals', {})

            # Run all trust checks
            self._check_skill_consistency(skills, trust)
            self._check_timeline_validity(career_history, trust)
            self._check_yoe_consistency(profile, career_history, trust)
            self._check_career_progression(career_history, profile, trust)
            self._check_skill_experience_alignment(skills, career_history, trust)
            self._check_profile_completeness(profile, skills, education, career_history, trust)
            self._check_redrob_anomalies(redrob_signals, trust)
            self._check_education_consistency(education, profile, trust)
            self._check_title_skill_correlation(career_history, skills, trust)
            self._check_company_duration_consistency(career_history, trust)

            # Calculate final score based on flags
            trust.overall_score = self._compute_score(trust)
            trust.trust_level = self._determine_trust_level(trust.overall_score)

            logger.debug(f"Trust score for {candidate_id}: {trust.overall_score:.3f} ({trust.trust_level})")

        except Exception as e:
            logger.error(f"Trust scoring error for {candidate_id}: {e}")
            trust.overall_score = 0.5  # Penalize unknown errors
            trust.flags.append({'check': 'system_error', 'message': str(e), 'severity': 'high'})

        return trust

    def _check_skill_consistency(self, skills: List[Dict], trust: TrustScore) -> None:
        """Check for skill stuffing and unrealistic proficiency claims"""
        if not skills:
            return

        # Count advanced/expert skills with zero duration
        advanced_zero = sum(
            1 for s in skills
            if s.get('proficiency') in ['advanced', 'expert'] and
            s.get('duration_months', 0) == 0
        )

        if advanced_zero > self.max_advanced_zero_duration:
            trust.flags.append({
                'check': 'skill_stuffing',
                'message': f"{advanced_zero} advanced/expert skills with 0 duration",
                'severity': 'high',
                'score_impact': 0.4
            })
            trust.flag_count += 1

        # Check for suspicious skill name patterns
        skill_names = [s.get('name', '').lower() for s in skills]
        skill_set = set(skill_names)

        # Too many similar skills (e.g., "Python", "python", "Python3")
        normalized_names = [re.sub(r'[^a-z0-9]', '', n) for n in skill_names if n]
        duplicates = len(normalized_names) - len(set(normalized_names))
        if duplicates > 3:
            trust.warnings.append(f"Possible duplicate skills: {duplicates} similar names")
            trust.flags.append({
                'check': 'duplicate_skills',
                'message': f"{duplicates} potentially duplicate skill entries",
                'severity': 'low',
                'score_impact': 0.05
            })
            trust.flag_count += 1

        # Check for too many skills total (may indicate breadth over depth)
        if len(skills) > 50:
            trust.warnings.append(f"Very high skill count: {len(skills)} skills")
            trust.flags.append({
                'check': 'excessive_skills',
                'message': f"{len(skills)} skills claimed",
                'severity': 'medium',
                'score_impact': 0.1
            })
            trust.flag_count += 1

        # Check for very new skills with high proficiency
        current_year = datetime.now().year
        for s in skills:
            if s.get('proficiency') in ['expert', 'advanced']:
                # Check if skill was recently acquired but claimed as expert
                # We don't have start dates, but duration can indicate
                duration = s.get('duration_months', 0)
                if 0 < duration < 6:
                    trust.warnings.append(f"High proficiency with very short duration: {s.get('name')}")

    def _check_timeline_validity(self, career_history: List[Dict], trust: TrustScore) -> None:
        """Check for negative tenure, overlapping jobs, impossible dates"""
        if not career_history:
            return

        current_year = datetime.now().year
        max_plausible_start = current_year - 60  # Can't start work before 60 years ago
        min_plausible_end = current_year - 10  # Job can't end more than 10 years ago unless it's current

        for job in career_history:
            start_date = job.get('start_date', '')
            end_date = job.get('end_date')
            is_current = job.get('is_current', False)

            # Parse years
            start_year_match = re.search(r'(\d{4})', str(start_date))
            start_year = int(start_year_match.group(1)) if start_year_match else None

            if is_current and end_date:
                end_year_match = re.search(r'(\d{4})', str(end_date))
                end_year = int(end_year_match.group(1)) if end_year_match else None
            else:
                end_year = None

            # Check for negative tenure (end before start)
            if start_year and end_year and end_year < start_year:
                trust.flags.append({
                    'check': 'negative_tenure',
                    'message': f"{job.get('company')}: {start_year}-{end_year}",
                    'severity': 'critical',
                    'score_impact': 0.6
                })
                trust.flag_count += 1

            # Check for impossibly early start
            if start_year and start_year < max_plausible_start:
                trust.warnings.append(f"Unusually early start year: {start_year} at {job.get('company')}")

            # Check for current job ending in past
            if is_current and end_year and end_year < current_year - 1:
                trust.warnings.append(f"Current job marked but ended {end_year}")

        # Check for overlapping jobs
        self._check_overlapping_jobs(career_history, trust)

    def _check_overlapping_jobs(self, career_history: List[Dict], trust: TrustScore) -> None:
        """Detect overlapping job periods"""
        intervals = []
        for job in career_history:
            start = job.get('start_date')
            end = job.get('end_date')
            is_current = job.get('is_current', False)

            start_dt = self._parse_date(start)
            if not start_dt:
                continue

            end_dt = None
            if is_current:
                end_dt = datetime.now()
            elif end:
                end_dt = self._parse_date(end)

            if end_dt and end_dt >= start_dt:
                intervals.append({
                    'start': start_dt,
                    'end': end_dt,
                    'company': job.get('company'),
                    'title': job.get('title')
                })

        # Check pairwise overlaps
        overlaps = 0
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                int1, int2 = intervals[i], intervals[j]
                if (int1['start'] < int2['end'] and int2['start'] < int1['end']):
                    overlaps += 1
                    trust.warnings.append(
                        f"Overlapping jobs: {int1['company']} & {int2['company']}"
                    )

        if overlaps > 2:
            trust.flags.append({
                'check': 'overlapping_jobs',
                'message': f"{overlaps} overlapping job periods detected",
                'severity': 'medium',
                'score_impact': 0.15
            })
            trust.flag_count += 1

    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        date_str = str(date_str).strip()
        if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
            return datetime.now()

        try:
            # Try ISO format
            return datetime.strptime(date_str[:10], '%Y-%m-%d')
        except:
            try:
                # Try other format
                return datetime.strptime(date_str, '%d-%m-%Y')
            except:
                # Try extracting year
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    year = int(year_match.group(1))
                    return datetime(year, 1, 1)
        return None

    def _check_yoe_consistency(
        self,
        profile: Dict,
        career_history: List[Dict],
        trust: TrustScore
    ) -> None:
        """Check if stated years of experience matches career history"""
        stated_yoe = profile.get('years_of_experience', 0)

        # Calculate total duration from career
        total_months = 0
        for job in career_history:
            duration = job.get('duration_months', 0)
            if duration > 0:
                total_months += duration

        calculated_yoe = total_months / 12

        # Allow some tolerance
        if stated_yoe > 0 and calculated_yoe > 0:
            diff = abs(stated_yoe - calculated_yoe)
            if diff > 3:  # More than 3 years difference is suspicious
                trust.flags.append({
                    'check': 'yoe_mismatch',
                    'message': f"Stated: {stated_yoe}yr, Career sum: {calculated_yoe:.1f}yr",
                    'severity': 'high',
                    'score_impact': 0.25
                })
                trust.flag_count += 1

        # Check for impossible YoE
        if stated_yoe > self.max_yoe:
            trust.flags.append({
                'check': 'impossible_yoe',
                'message': f"Stated YoE: {stated_yoe} years (exceeds max {self.max_yoe})",
                'severity': 'critical',
                'score_impact': 0.5
            })
            trust.flag_count += 1

    def _check_career_progression(
        self,
        career_history: List[Dict],
        profile: Dict,
        trust: TrustScore
    ) -> None:
        """Check for logical career progression"""
        if len(career_history) < 2:
            return

        current_title = profile.get('current_title', '').lower()
        titles = [job.get('title', '').lower() for job in career_history]

        # Check for title demotion
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager']
        junior_keywords = ['junior', 'associate', 'intern', 'entry']

        def get_title_level(title: str) -> int:
            if any(kw in title for kw in junior_keywords):
                return 0
            if any(kw in title for kw in senior_keywords):
                return 2
            return 1

        # Compare first and last titles
        first_level = get_title_level(titles[0]) if titles else 1
        last_level = get_title_level(current_title)

        if last_level < first_level - 1:
            trust.warnings.append("Possible title demotion in career progression")

    def _check_skill_experience_alignment(
        self,
        skills: List[Dict],
        career_history: List[Dict],
        trust: TrustScore
    ) -> None:
        """Check if claimed skills appear in experience descriptions"""
        if not skills or not career_history:
            return

        skill_names = set(s.get('name', '').lower() for s in skills)

        # Get all text from career descriptions and titles
        exp_text = ' '.join(
            job.get('description', '') + ' ' + job.get('title', '')
            for job in career_history
        ).lower()

        # Check for skills that don't appear anywhere in experience
        unverified_skills = []
        for skill in skill_names:
            # Normalize skill name for searching
            skill_norm = re.sub(r'[^a-z0-9]', '', skill.lower())
            exp_norm = re.sub(r'[^a-z0-9]', '', exp_text)

            # Check for skill name as substring
            if skill_norm not in exp_norm:
                # Also check individual words (e.g., "PyTorch" vs "PyTorch")
                words = skill_norm.split() if ' ' in skill_norm else [skill_norm]
                if not all(w in exp_norm for w in words if len(w) > 2):
                    unverified_skills.append(skill)

        if len(unverified_skills) > len(skill_names) * 0.3:  # More than 30% unverified
            trust.flags.append({
                'check': 'unverified_skills',
                'message': f"{len(unverified_skills)} skills not found in experience",
                'severity': 'medium',
                'score_impact': 0.15
            })
            trust.flag_count += 1
            trust.warnings.append(f"Unverified skills: {', '.join(unverified_skills[:5])}")

    def _check_profile_completeness(
        self,
        profile: Dict,
        skills: List[Any],
        education: List[Any],
        career_history: List[Any],
        trust: TrustScore
    ) -> None:
        """Check profile completeness and consistency"""
        completeness_score = 0.0
        max_score = 0.0

        # Required fields
        required_fields = ['current_title', 'current_company', 'years_of_experience', 'location']
        for field in required_fields:
            max_score += 1
            if profile.get(field):
                completeness_score += 1

        # Skills
        max_score += 1
        if len(skills) >= 5:
            completeness_score += 1
        elif len(skills) > 0:
            completeness_score += 0.5
            trust.warnings.append(f"Low skill count: {len(skills)}")

        # Career history
        max_score += 1
        if len(career_history) >= 2:
            completeness_score += 1
        elif len(career_history) == 1:
            completeness_score += 0.5
        else:
            trust.flags.append({
                'check': 'no_career_history',
                'message': "No career history provided",
                'severity': 'medium',
                'score_impact': 0.2
            })
            trust.flag_count += 1

        # Education
        max_score += 1
        if len(education) >= 1:
            completeness_score += 1

        completeness_pct = completeness_score / max_score if max_score > 0 else 0

        if completeness_pct < 0.5:
            trust.flags.append({
                'check': 'low_completeness',
                'message': f"Profile completeness: {completeness_pct:.0%}",
                'severity': 'medium',
                'score_impact': 0.15
            })
            trust.flag_count += 1

    def _check_redrob_anomalies(self, redrob_signals: Dict, trust: TrustScore) -> None:
        """Check for anomalies in redrob signals"""
        if not redrob_signals:
            trust.warnings.append("No redrob signals available")
            return

        # Check for perfect scores that might be fabricated
        perfect_metrics = []
        for key, value in redrob_signals.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                if abs(value - 1.0) < 0.01:
                    perfect_metrics.append(key)

        if len(perfect_metrics) > 5:
            trust.flags.append({
                'check': 'suspiciously_perfect',
                'message': f"{len(perfect_metrics)} metrics are exactly 1.0",
                'severity': 'low',
                'score_impact': 0.05
            })
            trust.flag_count += 1

        # Check for zero activity with high experience
        activity_score = redrob_signals.get('search_appearance_30d', 0)
        if activity_score == 0 and redrob_signals.get('profile_completeness_score', 0) > 80:
            trust.warnings.append("High completeness but zero recent activity")

    def _check_education_consistency(self, education: List[Dict], profile: Dict, trust: TrustScore) -> None:
        """Check if education timeline aligns with career"""
        if not education:
            return

        first_job_year = None
        for job in profile.get('career_history', []):
            start = self._parse_date(job.get('start_date'))
            if start:
                if first_job_year is None or start.year < first_job_year:
                    first_job_year = start.year

        if first_job_year:
            for edu in education:
                end_year = edu.get('end_year')
                if end_year and end_year > first_job_year + 2:
                    trust.warnings.append(
                        f"Education end year ({end_year}) after first job ({first_job_year})"
                    )

    def _check_title_skill_correlation(
        self,
        career_history: List[Dict],
        skills: List[Dict],
        trust: TrustScore
    ) -> None:
        """Check if skills align with job titles throughout career"""
        if not career_history or not skills:
            return

        skill_names = set(s.get('name', '').lower() for s in skills)
        tech_titles = ['engineer', 'developer', 'scientist', 'analyst', 'architect']

        # Count how many job titles are tech-related
        tech_title_count = sum(
            1 for job in career_history
            if any(tt in job.get('title', '').lower() for tt in tech_titles)
        )

        # If no tech titles but many tech skills, that's suspicious
        if tech_title_count == 0 and len(skill_names) > 10:
            trust.flags.append({
                'check': 'title_skill_mismatch',
                'message': "Non-technical titles but many technical skills",
                'severity': 'medium',
                'score_impact': 0.15
            })
            trust.flag_count += 1

    def _check_company_duration_consistency(
        self,
        career_history: List[Dict],
        trust: TrustScore
    ) -> None:
        """Check if durations between consecutive jobs make sense"""
        if len(career_history) < 2:
            return

        prev_end = None
        for job in career_history:
            start = self._parse_date(job.get('start_date'))
            end = self._parse_date(job.get('end_date'))

            if prev_end and start:
                gap_days = (start - prev_end).days
                if gap_days < -30:  # Overlap already checked separately
                    # Already handled
                    pass
                elif 0 <= gap_days < 30:
                    trust.warnings.append(f"Very short gap between jobs: {gap_days} days")

            if end:
                prev_end = end

    def _compute_score(self, trust: TrustScore) -> float:
        """Compute overall trust score from flags"""
        score = 1.0

        for flag in trust.flags:
            impact = flag.get('score_impact', 0.1)
            severity_multiplier = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5,
                'critical': 2.0
            }.get(flag.get('severity', 'medium'), 1.0)
            score -= impact * severity_multiplier

        # Warnings don't directly reduce score but indicate issues
        if len(trust.warnings) > 5:
            score *= 0.95

        return max(0.3, min(1.0, score))

    def _determine_trust_level(self, score: float) -> str:
        """Determine trust level from score"""
        if score >= 0.9:
            return "high"
        elif score >= 0.7:
            return "medium"
        elif score >= 0.5:
            return "low"
        else:
            return "critical"

    def get_trust_multiplier(self, trust_score: TrustScore) -> float:
        """Get final trust multiplier (alias for overall_score)"""
        return trust_score.overall_score


def create_trust_scorer(config: Optional[Dict] = None) -> TrustScorer:
    """Factory function to create TrustScorer"""
    return TrustScorer(config)


if __name__ == '__main__':
    # Test with sample candidate
    candidate = {
        'candidate_id': 'CAND_TEST',
        'profile': {
            'years_of_experience': 8.5,
            'current_title': 'Senior ML Engineer',
            'current_company': 'TechCorp'
        },
        'skills': [
            {'name': 'Python', 'proficiency': 'expert', 'duration_months': 72},
            {'name': 'PyTorch', 'proficiency': 'expert', 'duration_months': 48},
            {'name': 'TensorFlow', 'proficiency': 'advanced', 'duration_months': 36},
            {'name': 'AWS', 'proficiency': 'advanced', 'duration_months': 24},
            {'name': 'Docker', 'proficiency': 'expert', 'duration_months': 0},  # Suspicious!
        ],
        'career_history': [
            {
                'company': 'TechCorp',
                'title': 'ML Engineer',
                'start_date': '2020-01-15',
                'end_date': 'present',
                'is_current': True,
                'duration_months': 54
            },
            {
                'company': 'StartupXYZ',
                'title': 'Software Engineer',
                'start_date': '2018-06-01',
                'end_date': '2019-12-15',
                'is_current': False,
                'duration_months': 18
            }
        ],
        'education': [
            {'institution': 'University', 'degree': 'BS', 'field_of_study': 'CS', 'start_year': 2014, 'end_year': 2018}
        ],
        'redrob_signals': {
            'profile_completeness_score': 95,
            'open_to_work_flag': True,
            'recruiter_response_rate': 0.85,
            'last_active_date': '2024-01-15',
            'search_appearance_30d': 150
        }
    }

    scorer = create_trust_scorer()
    trust = scorer.calculate_trust_score(candidate)
    print("Trust Score:")
    print(json.dumps(trust.to_dict(), indent=2))
