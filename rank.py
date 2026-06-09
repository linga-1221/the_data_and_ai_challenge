#!/usr/bin/env python3
"""
Rank 100,000 candidate profiles against a job description.
Outputs top 100 candidates to CSV with scores and reasoning.

NO external APIs, NO GPU, NO torch - pure Python + pandas/numpy only.
Target: < 5 minutes on 16GB RAM CPU.
"""

import json
import re
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ============ CONSTANTS ============

# Services companies - disqualifying if entire career
SERVICES_COMPANIES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'hcl', 'tech mahindra', 'ibm', 'dell technologies', 'dell',
    'larsen & toubro', 'l&t', 'lti', 'tataconsultancyservices',
    'persistent systems', 'virtusa', 'mphasis', 'cyient', 'birlasoft',
    'hexaware', 'zensar', 'niit', 'adp', 'exl', 'genpact', 'concentrix',
    'first american', 'fis', 'fisglobal'
}

# Prestigious product companies
PRESTIGE_COMPANIES = {
    'google', 'meta', 'netflix', 'microsoft', 'amazon', 'apple',
    'linkedin', 'openai', 'anthropic', 'deepmind', 'mistral ai',
    'cohere', ' huggingface', 'databricks', 'snowflake',
    'flipkart', 'zomato', 'swiggy', 'razorpay', 'paytm', 'phonepe',
    'cred', 'meesho', 'ola', 'freshworks', 'zoho', 'shopee',
    'byju', 'unacademy', 'airbnb', 'uber', 'doordash', 'instacart',
    'stripe', 'twilio', 'slack', 'notion', 'figma', 'adobe'
}

# Preferred Indian cities
PREFERRED_CITIES = {
    'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'bengaluru',
    'bangalore', 'gurgaon', 'gurugram', 'ncr', 'vizag', 'visakhapatnam',
    'chennai', 'kolkata', 'jaipur', 'ahmedabad', 'indore',
    'bhubaneswar', 'chandigarh', 'coimbatore', 'trivandrum', 'kochi',
    'thiruvananthapuram'
}

# Must-have skills (base weight 65%)
MUST_HAVE_SKILLS = {
    'embeddings', 'vector database', 'pinecone', 'weaviate', 'qdrant',
    'milvus', 'faiss', 'opensearch', 'elasticsearch', 'pgvector',
    'hybrid search', 'hybrid retrieval', 'dense retrieval', 'retrieval',
    'ranking', 're-ranking', 'reranking', 'information retrieval',
    'semantic search', 'python', 'vector search', 'vector db',
    'vector similarity', 'cosine similarity', 'ann', 'approximate nearest neighbor'
}

# Must-have skill variations (normalized)
MUST_HAVE_NORMALIZED = {
    'embedding': 'embeddings',
    'faiss': 'faiss',
    'pinecone': 'pinecone',
    'weaviate': 'weaviate',
    'qdrant': 'qdrant',
    'milvus': 'milvus',
    'elastic search': 'elasticsearch',
    'opensearch': 'opensearch',
    'pgvector': 'pgvector',
    'hybrid': 'hybrid search',
    'dense retrieval': 'dense retrieval',
    'retrieval': 'retrieval',
    'ranking': 'ranking',
    'reranking': 're-ranking',
    'semantic search': 'semantic search',
    'vector': 'vector database',
    'python': 'python',
    'information retrieval': 'information retrieval'
}

# Nice-to-have skills (base weight 25%)
NICE_TO_HAVE = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning',
    'learning to rank', 'ltr', 'xgboost', 'lightgbm',
    'ndcg', 'mrr', 'a/b testing', 'ab testing', 'a/b test',
    'evaluation framework', 'metric', 'map', 'recall',
    'rag', 'retrieval augmented generation', 'llm', 'large language model',
    'bert', 'transformer', 'nlp', 'natural language processing',
    'bm25', 'recommendation systems', 'recommendation system',
    'personalization', 'search ranking', 'learning to rank'
}

# Evidence phrases in career descriptions
EVIDENCE_PHRASES = {
    'hybrid retrieval', 'learning-to-rank', 'semantic search',
    'ranking pipeline', 'embedding generation', 'a/b test',
    'candidate-jd matching', 're-ranking', 'dense retrieval',
    'vector search', 'recommendation systems', 'sentence-transformer',
    'bge', 'fine-tuned', 'ndcg', 'deployed', 'production',
    'scale', 'real users', 'serving', 'queries per', 'monitoring'
}

# Production keywords
PRODUCTION_KEYWORDS = {
    'deployed', 'production', 'scale', 'real users', 'serving',
    'queries per', 'qps', 'latency', 'throughput', 'monitoring',
    'alerting', 'incident', 'uptime', 'sla', 'ci/cd', 'docker',
    'kubernetes', 'microservices', 'api', 'endpoint'
}

# Relevant job titles (partial match)
RELEVANT_TITLES = {
    'ai engineer', 'ml engineer', 'machine learning', 'nlp engineer',
    'applied scientist', 'applied ml', 'search engineer', 'ranking engineer',
    'recommendation', 'search engineer', 'information retrieval',
    'data scientist ml', 'ml developer', 'ai developer', 'research engineer',
    'senior ml engineer', 'staff ml engineer', 'principal ml engineer'
}

# Disqualifying titles
DISQUALIFY_TITLES = {
    'marketing', 'sales', 'hr', 'recruiter', 'accountant', 'finance',
    'sales manager', 'marketing manager', 'hr manager', 'recruiter manager',
    'business development', 'bd', 'project manager', 'program manager',
    'product manager', 'ux', 'ui', 'designer', 'graphic', 'video',
    'photographer', 'content', 'copywriter', 'technical writer',
    'support', 'customer success', 'salesforce', 'admin', 'consultant'
}

# Career date parsing patterns
DATE_PATTERNS = [
    r'(\d{4})-(\d{1,2})-(\d{1,2})',
    r'(\d{1,2})-(\d{1,2})-(\d{4})',
    r'(\w+)\s+(\d{4})',
    r'(\d{4})'
]


@dataclass
class Candidate:
    """Candidate data structure"""
    candidate_id: str
    profile: Dict[str, Any]
    career_history: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    redrob_signals: Dict[str, Any]

    # Computed scores
    skill_score: float = 0.0
    career_score: float = 0.0
    behavioral_score: float = 0.0
    location_score: float = 0.0
    disqualifier_multiplier: float = 1.0
    final_score: float = 0.0
    reasoning_parts: List[str] = field(default_factory=list)
    is_honeypot: bool = False
    honeypot_reason: str = ""


class TextNormalizer:
    """Normalize text for matching"""

    @staticmethod
    def normalize(text: str) -> str:
        """Lowercase, remove extra whitespace, normalize punctuation"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Normalize skill name"""
        s = skill.lower().strip()
        # Common variations
        s = s.replace('fine-tuning', 'fine tuning').replace('fine-tuned', 'fine tuned')
        s = s.replace('a/b testing', 'ab testing').replace('a/b test', 'ab test')
        s = s.replace('learning-to-rank', 'learning to rank')
        s = s.replace('information retrieval', 'information retrieval')
        s = s.replace('recommendation systems', 'recommendation systems')
        return s


class SkillMatcher:
    """Match skills against candidate profile"""

    def __init__(self):
        self.normalizer = TextNormalizer()

    def extract_text_from_career(self, career_history: List[Dict]) -> str:
        """Extract all text from career descriptions"""
        texts = []
        for job in career_history:
            desc = job.get('description', '')
            title = job.get('title', '')
            company = job.get('company', '')
            if desc:
                texts.append(desc)
            if title:
                texts.append(title)
            if company:
                texts.append(company)
        return ' '.join(texts)

    def calculate_skill_score(
        self,
        skills: List[Dict],
        career_history: List[Dict]
    ) -> Tuple[float, List[str]]:
        """
        Calculate skill score from skills list and career descriptions.
        Returns (score, evidence_list)
        """
        # Build normalized skill name set
        skill_names = set()
        skill_details = {}  # name -> {proficiency, duration}

        for skill in skills:
            name = self.normalizer.normalize_skill(skill.get('name', ''))
            skill_names.add(name)
            skill_details[name] = {
                'proficiency': skill.get('proficiency', 'beginner'),
                'duration': skill.get('duration_months', 0),
                'endorsements': skill.get('endorsements', 0)
            }

        # Get career descriptions text
        career_text = self.extract_text_from_career(career_history)
        career_text_norm = self.normalizer.normalize(career_text)

        # Check must-have skills
        must_have_hits = set()
        for skill in MUST_HAVE_SKILLS:
            skill_norm = self.normalizer.normalize(skill)
            # Exact phrase match in skills list
            if any(skill_norm in s for s in skill_names):
                must_have_hits.add(skill_norm)
            # Phrase match in career text
            if skill_norm in career_text_norm:
                must_have_hits.add(skill_norm)
            # Word-level match for single-word skills
            if ' ' not in skill_norm and skill_norm in career_text_norm.split():
                must_have_hits.add(skill_norm)

        must_have_score = min(len(must_have_hits) / len(MUST_HAVE_SKILLS), 1.0)

        # Check nice-to-have skills
        nice_hits = set()
        for skill in NICE_TO_HAVE:
            skill_norm = self.normalizer.normalize(skill)
            if any(skill_norm in s for s in skill_names):
                nice_hits.add(skill_norm)
            if skill_norm in career_text_norm:
                nice_hits.add(skill_norm)
            if ' ' not in skill_norm and skill_norm in career_text_norm.split():
                nice_hits.add(skill_norm)

        nice_score = min(len(nice_hits) / len(NICE_TO_HAVE), 1.0)

        # Proficiency bonus
        proficiency_bonus = 0.0
        for name, details in skill_details.items():
            if name in must_have_hits or name in nice_hits:
                prof = details['proficiency']
                if prof == 'advanced':
                    proficiency_bonus += 0.05
                elif prof == 'expert':
                    proficiency_bonus += 0.08
                # Duration bonus (24+ months)
                if details['duration'] >= 24:
                    proficiency_bonus += 0.04

        proficiency_bonus = min(proficiency_bonus, 0.10)

        # Combined skill score: 65% must-have + 25% nice-to-have + 10% proficiency
        skill_score = (0.65 * must_have_score) + (0.25 * nice_score) + (0.10 * proficiency_bonus)
        skill_score = min(skill_score, 1.0)

        # Build evidence list
        evidence = []
        if must_have_hits:
            evidence.append(f"Must-have skills: {', '.join(sorted(must_have_hits)[:5])}")
        if nice_hits:
            evidence.append(f"Nice skills: {', '.join(sorted(nice_hits)[:3])}")
        if proficiency_bonus > 0:
            evidence.append(f"Proficiency bonus: +{proficiency_bonus:.2f}")

        return skill_score, evidence


class CareerScorer:
    """Score based on career history"""

    def __init__(self):
        self.services_companies = SERVICES_COMPANIES
        self.prestige_companies = PRESTIGE_COMPANIES
        self.relevant_titles = RELEVANT_TITLES

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None

        date_str = str(date_str).strip()
        if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
            return datetime.now()

        # Try various formats
        for pattern in DATE_PATTERNS:
            try:
                if re.match(pattern, date_str, re.I):
                    parts = re.findall(r'\d+', date_str)
                    if len(parts) == 4:  # MM-DD-YYYY
                        return datetime(int(parts[2]), int(parts[0]), int(parts[1]))
                    elif len(parts) == 3:
                        # Assume YYYY-MM-DD
                        return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 2:
                        # Month Year
                        return datetime(int(parts[1]), int(parts[0]), 1)
                    elif len(parts) == 1:
                        # Year only
                        return datetime(int(parts[0]), 12, 31)
            except:
                pass

        return None

    def calculate_career_score(
        self,
        career_history: List[Dict],
        years_of_experience: float,
        current_title: str,
        current_company: str
    ) -> Tuple[float, List[str]]:
        """
        Calculate career score from career history.
        Returns (score, evidence_list)
        """
        evidence = []
        score_components = []
        total_months = 0
        services_months = 0
        prestige_months = 0
        product_months = 0

        # Analyze each job
        for job in career_history:
            start = self._parse_date(job.get('start_date'))
            end = self._parse_date(job.get('end_date'))
            company = job.get('company', '').lower()
            desc = job.get('description', '').lower()
            title = job.get('title', '').lower()

            if start and end and end > start:
                duration_months = (end - start).days / 30.44
            else:
                # Fallback to provided duration
                duration_months = job.get('duration_months', 0)
                if duration_months == 0 and job.get('is_current'):
                    duration_months = (datetime.now() - start).days / 30.44 if start else 0

            total_months = max(total_months, duration_months)

            # Check if services company
            is_services = any(svc in company for svc in self.services_companies)
            if is_services:
                services_months += duration_months
            else:
                product_months += duration_months

            # Check if prestige company
            if any(pre in company for pre in self.prestige_companies):
                prestige_months += duration_months

        # Normalize to years
        total_years = total_months / 12
        if total_years < years_of_experience - 2:
            total_years = years_of_experience

        # 1. YoE scoring
        if years_of_experience < 3:
            yoe_score = 0.10
        elif years_of_experience < 5:
            yoe_score = 0.55
        elif years_of_experience < 9:
            yoe_score = 1.00
        elif years_of_experience < 12:
            yoe_score = 0.80
        else:
            yoe_score = 0.60

        score_components.append(('YoE', 0.40, yoe_score))
        evidence.append(f"YoE: {years_of_experience:.1f} years (score: {yoe_score:.2f})")

        # 2. Company type scoring
        if total_months > 0:
            product_ratio = product_months / total_months
        else:
            product_ratio = 0.0

        if product_months > 0:
            company_score = 0.40 + (0.60 * product_ratio)  # 0.40 to 1.00
        else:
            company_score = 0.20  # All services

        score_components.append(('Company type', 0.25, company_score))
        evidence.append(f"Company ratio: {product_ratio:.1%} product (score: {company_score:.2f})")

        # 3. Title relevance
        title_lower = current_title.lower() if current_title else ''
        title_match = any(rel in title_lower for rel in self.relevant_titles)
        title_score = 1.0 if title_match else 0.0

        score_components.append(('Title', 0.20, title_score))
        evidence.append(f"Title match: {title_match} ({current_title})")

        # 4. Evidence phrases from descriptions
        all_descriptions = ' '.join([
            job.get('description', '') for job in career_history
        ]).lower()

        evidence_count = sum(1 for phrase in EVIDENCE_PHRASES if phrase in all_descriptions)
        evidence_score = min(evidence_count / 5.0, 1.0)  # 5 phrases = 1.0

        score_components.append(('Evidence', 0.10, evidence_score))
        if evidence_count > 0:
            evidence.append(f"Evidence phrases: {evidence_count} found")

        # 5. Production evidence
        production_count = sum(1 for word in PRODUCTION_KEYWORDS if word in all_descriptions)
        production_bonus = min(production_count * 0.1, 0.15)  # Up to +0.15
        if production_count > 0:
            evidence.append(f"Production keywords: {production_count}")

        # 6. Prestige bonus
        prestige_ratio = prestige_months / max(total_months, 1)
        prestige_bonus = prestige_ratio * 0.35  # Up to +0.35
        if prestige_ratio > 0:
            evidence.append(f"Prestige bonus: +{prestige_bonus:.2f}")

        # Calculate weighted score
        total_weight = sum(w for _, w, _ in score_components)
        weighted_score = sum(score * weight for _, weight, score in score_components) / total_weight

        # Add bonuses
        weighted_score = min(weighted_score + production_bonus + prestige_bonus, 1.0)

        return weighted_score, evidence

    def is_primarily_services(self, career_history: List[Dict]) -> bool:
        """Check if career is entirely at services companies"""
        if not career_history:
            return False

        for job in career_history:
            company = job.get('company', '').lower()
            if not any(svc in company for svc in self.services_companies):
                return False
        return True

    def is_pure_research(self, career_history: List[Dict], titles: List[str]) -> bool:
        """Check if career is pure research/academic with no production"""
        has_prod_company = False
        has_prod_keywords = False

        for job in career_history:
            company = job.get('company', '').lower()
            desc = job.get('description', '').lower()

            # Check for companies that are research-oriented only
            if any(word in company for word in ['university', 'research lab', 'institute', 'college']):
                continue

            # Check for production keywords
            if any(kw in desc for kw in PRODUCTION_KEYWORDS):
                has_prod_keywords = True

            if not any(svc in company for svc in self.services_companies):
                has_prod_company = True

        # Also check titles
        prod_titles = any(any(word in t.lower() for word in ['engineer', 'developer', 'applied'])
                         for t in titles)

        return not (has_prod_keywords or has_prod_company) and not prod_titles


class BehavioralScorer:
    """Score based on redrob signals and recency"""

    WEIGHTS = {
        'last_active': 0.25,
        'open_to_work': 0.15,
        'recruiter_response_rate': 0.20,
        'interview_completion_rate': 0.15,
        'notice_period': 0.15,
        'github_activity': 0.10
    }

    def score_last_active(self, last_active: Optional[str]) -> Tuple[float, str]:
        """Score based on last active date"""
        if not last_active:
            return 0.20, "No last active date"

        try:
            # Try to parse
            for pattern in DATE_PATTERNS:
                if re.match(pattern, last_active, re.I):
                    parsed = self._parse_date(last_active)
                    if parsed:
                        days_ago = (datetime.now() - parsed).days
                        if days_ago < 7:
                            return 1.0, f"Active {days_ago}d ago"
                        elif days_ago < 14:
                            return 0.95, f"Active {days_ago}d ago"
                        elif days_ago < 30:
                            return 0.85, f"Active {days_ago}d ago"
                        elif days_ago < 60:
                            return 0.65, f"Active {days_ago}d ago"
                        elif days_ago < 90:
                            return 0.45, f"Active {days_ago}d ago"
                        elif days_ago < 180:
                            return 0.20, f"Active {days_ago}d ago"
                        else:
                            return 0.05, f"Active {days_ago}d ago"
        except:
            pass

        return 0.20, "Unclear last active"

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string"""
        try:
            date_str = str(date_str).strip()
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    pass
            # Try just year
            if re.match(r'^\d{4}$', date_str):
                return datetime(int(date_str), 12, 31)
        except:
            pass
        return None

    def calculate_behavioral_score(
        self,
        redrob_signals: Dict[str, Any],
        last_active_date: Optional[str] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate behavioral score from redrob signals.
        Returns (score, evidence_list)
        """
        evidence = []
        scores = {}

        # 1. Last active recency
        active_score, active_evidence = self.score_last_active(
            last_active_date or redrob_signals.get('last_active_date')
        )
        scores['last_active'] = active_score
        evidence.append(active_evidence)

        # 2. Open to work
        open_to_work = redrob_signals.get('open_to_work_flag', False)
        scores['open_to_work'] = 1.0 if open_to_work else 0.45
        evidence.append(f"Open to work: {open_to_work}")

        # 3. Recruiter response rate
        response_rate = redrob_signals.get('recruiter_response_rate', 0.0)
        if isinstance(response_rate, (int, float)):
            scores['recruiter_response_rate'] = min(max(response_rate, 0.0), 1.0)
            evidence.append(f"Response rate: {response_rate:.1%}")
        else:
            scores['recruiter_response_rate'] = 0.5
            evidence.append("Response rate: N/A")

        # 4. Interview completion rate
        interview_rate = redrob_signals.get('interview_completion_rate', 0.0)
        if isinstance(interview_rate, (int, float)):
            scores['interview_completion_rate'] = min(max(interview_rate, 0.0), 1.0)
            evidence.append(f"Interview rate: {interview_rate:.1%}")
        else:
            scores['interview_completion_rate'] = 0.5
            evidence.append("Interview rate: N/A")

        # 5. Notice period
        notice_days = redrob_signals.get('notice_period_days', 999)
        if notice_days <= 15:
            notice_score = 1.0
        elif notice_days <= 30:
            notice_score = 0.90
        elif notice_days <= 60:
            notice_score = 0.70
        elif notice_days <= 90:
            notice_score = 0.45
        else:
            notice_score = 0.20
        scores['notice_period'] = notice_score
        evidence.append(f"Notice: {notice_days} days")

        # 6. GitHub activity
        github_score = redrob_signals.get('github_activity_score', 0)
        if github_score == -1 or github_score is None:
            github_norm = 0.30  # Neutral for no account
        else:
            github_norm = min(max(github_score, 0.0), 100) / 100.0
        scores['github_activity'] = github_norm
        evidence.append(f"GitHub: {github_norm:.1%}")

        # Weighted score
        weighted_score = sum(
            scores[key] * self.WEIGHTS[key]
            for key in self.WEIGHTS
        )

        return weighted_score, evidence


class LocationScorer:
    """Score based on location"""

    def __init__(self):
        self.preferred = PREFERRED_CITIES
        self.indian_cities = self.preferred | {
            'ahmedabad', 'bhopal', 'bikaner', 'coimbatore', 'dehradun',
            'faridabad', 'ghaziabad', 'hubli', 'jabalpur', 'jamshedpur',
            'jodhpur', 'kannur', 'kolhapur', 'kottayam', 'ludhiana',
            'mangalore', 'mysore', 'nagpur', 'patna', 'pune', 'rajkot',
            'surat', 'vadodara', 'vijayawada', 'visakhapatnam'
        }

    def normalize_location(self, location: str) -> Tuple[str, Optional[str]]:
        """Extract city and country from location string"""
        if not location:
            return "", None

        loc = location.lower().strip()
        parts = re.split(r'[,\|]', loc)

        city = parts[0].strip() if parts else loc
        country = parts[-1].strip() if len(parts) > 1 else None

        # Clean city name
        city = re.sub(r'[^a-z\s]', '', city).strip()

        return city, country

    def calculate_location_score(
        self,
        location: str,
        country: Optional[str],
        willing_to_relocate: bool = False
    ) -> Tuple[float, List[str]]:
        """
        Calculate location score.
        Returns (score, evidence_list)
        """
        evidence = []
        city, detected_country = self.normalize_location(location)

        if not city:
            return 0.10, ["Unknown location"]

        # Check preferred cities
        if city in self.preferred:
            evidence.append(f"City {city} is preferred")
            return 1.0, evidence

        # Check if in India
        is_india = (
            country and 'india' in country.lower() or
            city in self.indian_cities or
            (detected_country and 'india' in detected_country.lower())
        )

        if is_india:
            if willing_to_relocate:
                evidence.append(f"City {city} (relocatable) - 0.80")
                return 0.80, evidence
            else:
                evidence.append(f"City {city} - non-preferred Indian city")
                return 0.65, evidence
        else:
            if willing_to_relocate:
                evidence.append(f"International (relocatable) - 0.30")
                return 0.30, evidence
            else:
                evidence.append(f"International, not relocatable")
                return 0.10, evidence


class HoneypotDetector:
    """Detect fake or problematic profiles"""

    def __init__(self):
        self.services_companies = SERVICES_COMPANIES

    def check_honeypot(
        self,
        candidate: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if candidate is a honeypot/spam.
        Returns (is_honeypot, reason)
        """
        # 1. Excessive advanced skills with zero duration
        skills = candidate.get('skills', [])
        advanced_with_no_duration = sum(
            1 for s in skills
            if s.get('proficiency') in ['advanced', 'expert'] and
            s.get('duration_months', 0) == 0
        )
        if advanced_with_no_duration >= 5:
            return True, f"5+ advanced skills with 0 duration ({advanced_with_no_duration})"

        # 2. Negative tenure from career dates
        career = candidate.get('career_history', [])
        for job in career:
            start = job.get('start_date')
            end = job.get('end_date')
            if start and end and job.get('is_current') != True:
                try:
                    # Simple date comparison
                    if self._dates_negative(start, end):
                        return True, f"Negative tenure: {job.get('company')} {start}-{end}"
                except:
                    pass

        # 3. Stated YoE exceeds career history
        years_of_exp = candidate.get('profile', {}).get('years_of_experience', 0)
        total_career_months = 0
        for job in career:
            dur = job.get('duration_months', 0)
            if dur > 0:
                total_career_months += dur

        if total_career_months > 0:
            diff_years = abs(years_of_exp - (total_career_months / 12))
            if diff_years > 6:
                return True, f"YoE mismatch: stated {years_of_exp}, career sum {total_career_months/12:.1f}"

        # 4. Excessive YoE
        if years_of_exp > 35:
            return True, f"Impossible YoE: {years_of_exp} years"

        # 5. Only services companies check (handled as disqualifier, not honeypot)
        # 6. Unrelated title (handled as disqualifier)

        return False, ""

    def _dates_negative(self, start: str, end: str) -> bool:
        """Check if end is before start"""
        # Simple year extraction
        start_year = re.search(r'(\d{4})', str(start))
        end_year = re.search(r'(\d{4})', str(end))
        if start_year and end_year:
            start_y = int(start_year.group(1))
            end_y = int(end_year.group(1))
            if end_y < start_y:
                return True
            if end_y == start_y:
                # Check months
                start_m = re.search(r'-(\d{1,2})', str(start))
                end_m = re.search(r'-(\d{1,2})', str(end))
                if start_m and end_m:
                    if int(end_m.group(1)) < int(start_m.group(1)):
                        return True
        return False


class CandidateRanker:
    """Main ranking system"""

    def __init__(self):
        self.skill_matcher = SkillMatcher()
        self.career_scorer = CareerScorer()
        self.behavioral_scorer = BehavioralScorer()
        self.location_scorer = LocationScorer()
        self.honeypot_detector = HoneypotDetector()

        self.normalizer = TextNormalizer()

    def parse_profiles(
        self,
        input_path: str,
        max_candidates: int = None
    ) -> List[Candidate]:
        """Load and parse candidate profiles"""
        candidates = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_candidates and i >= max_candidates:
                    break

                try:
                    data = json.loads(line.strip())

                    # Extract fields
                    profile = data.get('profile', {})
                    career_history = data.get('career_history', [])
                    education = data.get('education', [])
                    skills = data.get('skills', [])
                    redrob_signals = data.get('redrob_signals', {})

                    candidate_id = data.get('candidate_id', f"cand_{i}")

                    candidate = Candidate(
                        candidate_id=candidate_id,
                        profile=profile,
                        career_history=career_history,
                        education=education,
                        skills=skills,
                        redrob_signals=redrob_signals
                    )

                    candidates.append(candidate)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {i+1}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Error parsing line {i+1}: {e}", file=sys.stderr)

        print(f"Loaded {len(candidates)} candidates")
        return candidates

    def check_disqualifiers(
        self,
        candidate: Candidate
    ) -> float:
        """
        Check disqualifier conditions and return multiplier.
        Returns 1.0 if no disqualifiers, lower values otherwise.
        """
        multiplier = 1.0
        reasons = []

        # 1. Entire career at services companies
        if self.career_scorer.is_primarily_services(candidate.career_history):
            multiplier *= 0.30
            reasons.append("entirely services companies (x0.30)")

        # 2. CV/speech/robotics primary skills (no NLP/IR)
        all_skills = [s.get('name', '').lower() for s in candidate.skills]
        all_text = ' '.join([s.get('name', '') for s in candidate.skills] +
                           [j.get('description', '') for j in candidate.career_history])

        has_cv = any(word in all_text for word in ['computer vision', 'cv', 'image', 'video'])
        has_speech = any(word in all_text for word in ['speech', 'audio', 'voice'])
        has_robotics = any(word in all_text for word in ['robotics', 'robot', 'autonomous'])

        has_nlp_ir = any(word in all_text for word in [
            'nlp', 'natural language', 'text', 'language model',
            'retrieval', 'search', 'ranking', 'embedding'
        ])

        if (has_cv or has_speech or has_robotics) and not has_nlp_ir:
            multiplier *= 0.20
            reasons.append("CV/speech/robotics only, no NLP/IR (x0.20)")

        # 3. Unrelated title
        current_title = candidate.profile.get('current_title', '').lower()
        if any(disq in current_title for disq in DISQUALIFY_TITLES):
            multiplier *= 0.05
            reasons.append(f"unrelated title: {current_title} (x0.05)")

        if reasons:
            candidate.reasoning_parts.append(f"Disqualifiers: {', '.join(reasons)}")

        return multiplier

    def calculate_score(self, candidate: Candidate) -> None:
        """Calculate all scores for a candidate"""

        # Get basic fields
        profile = candidate.profile
        years_of_exp = profile.get('years_of_experience', 0.0)
        current_title = profile.get('current_title', '')
        current_company = profile.get('current_company', '')
        location = profile.get('location', '')

        # 1. Skill Score (35%)
        skill_score, skill_evidence = self.skill_matcher.calculate_skill_score(
            candidate.skills, candidate.career_history
        )
        candidate.skill_score = skill_score
        candidate.reasoning_parts.extend(skill_evidence)

        # 2. Career Score (30%)
        career_score, career_evidence = self.career_scorer.calculate_career_score(
            candidate.career_history, years_of_exp, current_title, current_company
        )
        candidate.career_score = career_score
        candidate.reasoning_parts.extend(career_evidence)

        # Research penalty
        if self.career_scorer.is_pure_research(
            candidate.career_history,
            [current_title] + [j.get('title', '') for j in candidate.career_history]
        ):
            candidate.career_score *= 0.6
            candidate.reasoning_parts.append("Pure research background (x0.6)")

        # 3. Behavioral Score (20%)
        behavioral_score, behavioral_evidence = self.behavioral_scorer.calculate_behavioral_score(
            candidate.redrob_signals,
            profile.get('last_active_date')
        )
        candidate.behavioral_score = behavioral_score
        candidate.reasoning_parts.extend(behavioral_evidence)

        # 4. Location Score (15%)
        willing_to_relocate = candidate.redrob_signals.get('willing_to_relocate', False)
        location_score, location_evidence = self.location_scorer.calculate_location_score(
            location,
            profile.get('country'),
            willing_to_relocate
        )
        candidate.location_score = location_score
        candidate.reasoning_parts.extend(location_evidence)

        # 5. Disqualifier multiplier
        disqualifier_mult = self.check_disqualifiers(candidate)
        candidate.disqualifier_multiplier = disqualifier_mult

        # 6. Final score
        raw_score = (0.35 * skill_score +
                    0.30 * career_score +
                    0.20 * behavioral_score +
                    0.15 * location_score)

        candidate.final_score = min(raw_score * disqualifier_mult, 1.0)
        candidate.final_score = max(candidate.final_score, 0.0)

        # Format final reasoning (1-2 sentences)
        key_facts = []
        key_facts.append(f"Title: {current_title}, YoE: {years_of_exp:.0f}")

        # Key strengths from evidence
        if skill_score > 0.5:
            key_facts.append(f"Skills: {skill_score:.2f}")
        if career_score > 0.7:
            key_facts.append(f"Career: {career_score:.2f}")

        # Behavioral highlights
        if behavioral_score < 0.5:
            notice = candidate.redrob_signals.get('notice_period_days', 0)
            key_facts.append(f"Notice: {notice}d")
        response = candidate.redrob_signals.get('recruiter_response_rate', 0)
        if isinstance(response, (int, float)) and response < 0.5:
            key_facts.append(f"Response: {response:.0%}")

        # Location
        loc_city = location.split(',')[0] if location else 'Unknown'
        key_facts.append(f"Loc: {loc_city}")

        candidate.reasoning_parts = ['. '.join(key_facts[:3])]

    def rank_candidates(
        self,
        candidates: List[Candidate],
        output_path: str,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Rank all candidates and output top N to CSV.
        """
        print(f"Scoring {len(candidates)} candidates...")

        scored_candidates = []
        skipped_honeypots = 0

        for i, candidate in enumerate(candidates):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{len(candidates)}")

            # Check honeypot
            is_honeypot, reason = self.honeypot_detector.check_honeypot(
                candidate.__dict__
            )
            if is_honeypot:
                skipped_honeypots += 1
                candidate.is_honeypot = True
                candidate.honeypot_reason = reason
                continue

            # Calculate score
            self.calculate_score(candidate)
            scored_candidates.append(candidate)

        print(f"  Scored: {len(scored_candidates)}")
        print(f"  Skipped (honeypot): {skipped_honeypots}")

        # Sort by score (desc), then candidate_id (asc)
        scored_candidates.sort(key=lambda c: (-c.final_score, c.candidate_id))

        # Take top N
        top_candidates = scored_candidates[:top_n]

        # Create DataFrame
        data = []
        for rank, candidate in enumerate(top_candidates, 1):
            data.append({
                'candidate_id': candidate.candidate_id,
                'rank': rank,
                'score': round(candidate.final_score, 4),
                'reasoning': '. '.join(candidate.reasoning_parts)[:300]
            })

        df = pd.DataFrame(data)

        # Verify monotonicity
        for i in range(1, len(df)):
            if df.iloc[i]['score'] > df.iloc[i-1]['score'] + 0.0001:  # floating point tolerance
                print(f"Warning: Score not monotonic at rank {i+1} (score {df.iloc[i]['score']} > {df.iloc[i-1]['score']})")
                # Fix by adjusting
                df.at[i, 'score'] = df.iloc[i-1]['score']

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved top {len(df)} candidates to {output_path}")
        print(f"\nTop 10:")
        print(df.head(10).to_string(index=False))

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidate profiles against job description"
    )
    parser.add_argument(
        '--candidates',
        required=True,
        help='Path to candidates.jsonl file'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--max',
        type=int,
        help='Maximum candidates to process (for testing)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CANDIDATE RANKING SYSTEM")
    print("=" * 60)
    print(f"Input: {args.candidates}")
    print(f"Output: {args.out}")
    print("=" * 60)

    # Initialize ranker
    ranker = CandidateRanker()

    # Load candidates
    candidates = ranker.parse_profiles(
        args.candidates,
        max_candidates=args.max
    )

    if not candidates:
        print("ERROR: No candidates loaded")
        return 1

    # Rank and output
    start_time = datetime.now()
    df = ranker.rank_candidates(candidates, args.out, top_n=100)
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 60)
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Final score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
