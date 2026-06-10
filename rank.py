 import json
  import os
  import re
  import sys
  import argparse
  from datetime import datetime, timedelta
  from typing import Dict, List, Tuple, Set, Optional, Any
  from dataclasses import dataclass, field
  from collections import defaultdict
  import warnings
  from concurrent.futures import ProcessPoolExecutor
  import numpy as np
  import pandas as pd

  warnings.filterwarnings('ignore')

  # ============ CONSTANTS (for default config) ============

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
      'cohere', 'huggingface', 'databricks', 'snowflake',
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

  # Other Indian cities (non-preferred)
  OTHER_INDIAN_CITIES = [
      'ahmedabad', 'bhopal', 'bikaner', 'dehradun', 'faridabad',
      'ghaziabad', 'hubli', 'jabalpur', 'jamshedpur', 'jodhpur',
      'kannur', 'kolhapur', 'kottayam', 'ludhiana', 'mangalore',
      'mysore', 'nagpur', 'patna', 'rajkot', 'surat', 'vadodara',
      'vijayawada'
  ]

  # Must-have skills (base weight 65%)
  MUST_HAVE_SKILLS = {
      'embeddings', 'vector database', 'pinecone', 'weaviate', 'qdrant',
      'milvus', 'faiss', 'opensearch', 'elasticsearch', 'pgvector',
      'hybrid search', 'hybrid retrieval', 'dense retrieval', 'retrieval',
      'ranking', 're-ranking', 'reranking', 'information retrieval',
      'semantic search', 'python', 'vector search', 'vector db',
      'vector similarity', 'cosine similarity', 'ann', 'approximate nearest neighbor'
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

  # Default behavioral weights
  DEFAULT_BEHAVIORAL_WEIGHTS = {
      'last_active': 0.25,
      'open_to_work': 0.15,
      'recruiter_response_rate': 0.20,
      'interview_completion_rate': 0.15,
      'notice_period': 0.15,
      'github_activity': 0.10
  }

  DEFAULT_CAREER_WEIGHTS = {
      'yoe': 0.40,
      'company_type': 0.25,
      'title_relevance': 0.20,
      'evidence_phrases': 0.10
  }

  # Default configuration (used if no config file exists)
  DEFAULT_CONFIG = {
      "component_weights": {
          "skill": 0.35,
          "career": 0.30,
          "behavioral": 0.20,
          "location": 0.15
      },
      "services_companies": list(SERVICES_COMPANIES),
      "prestige_companies": list(PRESTIGE_COMPANIES),
      "preferred_cities": list(PREFERRED_CITIES),
      "other_indian_cities": OTHER_INDIAN_CITIES,
      "must_have_skills": list(MUST_HAVE_SKILLS),
      "nice_to_have": list(NICE_TO_HAVE),
      "evidence_phrases": list(EVIDENCE_PHRASES),
      "production_keywords": list(PRODUCTION_KEYWORDS),
      "relevant_titles": list(RELEVANT_TITLES),
      "disqualify_titles": list(DISQUALIFY_TITLES),
      "disqualifier_multipliers": {
          "services_only": 0.30,
          "cv_speech_robotics_only": 0.20,
          "unrelated_title": 0.05
      },
      "honeypot": {
          "max_advanced_zero_duration": 5,
          "max_yoe": 35
      },
      "pure_research_penalty": 0.6,
      "top_n": 100,
      "behavioral_weights": DEFAULT_BEHAVIORAL_WEIGHTS,
      "career_weights": DEFAULT_CAREER_WEIGHTS,
      "skill_weights": {
          "must_have": 0.65,
          "nice_to_have": 0.25,
          "proficiency": 0.10
      },
      "skill_proficiency_bonus": {
          "advanced": 0.05,
          "expert": 0.08,
          "duration_24_plus_months": 0.04
      },
      "max_proficiency_bonus": 0.10,
      "production_bonus_max": 0.15,
      "production_bonus_per_keyword": 0.1,
      "prestige_bonus_ratio": 0.35,
      "title_relevance_match_score": 1.0
  }

  def load_config(config_path: str) -> Dict:
      """Load configuration from JSON file. If not found, creates default."""
      if not os.path.exists(config_path):
          print(f"Config file {config_path} not found. Creating default config.")
          with open(config_path, 'w', encoding='utf-8') as f:
              json.dump(DEFAULT_CONFIG, f, indent=2)
          return DEFAULT_CONFIG
      try:
          with open(config_path, 'r', encoding='utf-8') as f:
              config = json.load(f)
          return config
      except Exception as e:
          print(f"Error loading config: {e}. Using default configuration.")
          return DEFAULT_CONFIG

  @dataclass
  class Candidate:
      candidate_id: str
      profile: Dict[str, Any]
      career_history: List[Dict[str, Any]]
      education: List[Dict[str, Any]]
      skills: List[Dict[str, Any]]
      redrob_signals: Dict[str, Any]
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
          if not text: return ""
          text = text.lower()
          text = re.sub(r'[^\w\s-]', ' ', text)
          text = re.sub(r'\s+', ' ', text)
          return text.strip()
      @staticmethod
      def normalize_skill(skill: str) -> str:
          s = skill.lower().strip()
          s = s.replace('fine-tuning', 'fine tuning').replace('fine-tuned', 'fine tuned')
          s = s.replace('a/b testing', 'ab testing').replace('a/b test', 'ab test')
          s = s.replace('learning-to-rank', 'learning to rank')
          s = re.sub(r'[^\w\s-]', ' ', s)
          s = re.sub(r'\s+', ' ', s).strip()
          return s

  class SkillMatcher:
      """Match skills against candidate profile"""
      def __init__(self, config: Dict):
          self.config = config
          self._prepare_skills(config)
      def _prepare_skills(self, config: Dict):
          self.must_have_phrases = [self._normalize(s) for s in config['must_have_skills'] if s]
          self.nice_to_have_phrases = [self._normalize(s) for s in config['nice_to_have'] if s]
          self._compile_patterns()
          self.must_have_set = set(self.must_have_phrases)
          self.nice_to_have_set = set(self.nice_to_have_phrases)
      def _normalize(self, text: str) -> str:
          if not text: return ""
          t = text.lower().strip()
          t = t.replace('fine-tuning', 'fine tuning').replace('fine-tuned', 'fine tuned')
          t = t.replace('a/b testing', 'ab testing').replace('a/b test', 'ab test')
          t = t.replace('learning-to-rank', 'learning to rank')
          t = re.sub(r'[^a-z0-9\s]', ' ', t)
          t = re.sub(r'\s+', ' ', t).strip()
          return t
      def _compile_patterns(self):
          if self.must_have_phrases:
              escaped = [re.escape(p) for p in self.must_have_phrases]
              pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
              self.must_have_regex = re.compile(pattern)
          else:
              self.must_have_regex = None
          if self.nice_to_have_phrases:
              escaped = [re.escape(p) for p in self.nice_to_have_phrases]
              pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
              self.nice_to_have_regex = re.compile(pattern)
          else:
              self.nice_to_have_regex = None
      def extract_text_from_career(self, career_history: List[Dict]) -> str:
          texts = []
          for job in career_history:
              desc = job.get('description', '')
              title = job.get('title', '')
              company = job.get('company', '')
              if desc: texts.append(desc)
              if title: texts.append(title)
              if company: texts.append(company)
          return ' '.join(texts)
      def calculate_skill_score(self, skills: List[Dict], career_history: List[Dict]) -> Tuple[float, List[str]]:
          skill_names = set()
          skill_details = {}
          for skill in skills:
              name = self._normalize(skill.get('name', ''))
              if name:
                  skill_names.add(name)
                  skill_details[name] = {
                      'proficiency': skill.get('proficiency', 'beginner'),
                      'duration': skill.get('duration_months', 0),
                      'endorsements': skill.get('endorsements', 0)
                  }
          career_text = self.extract_text_from_career(career_history)
          career_norm = self._normalize(career_text)
          must_have_hits = set()
          for mh in self.must_have_phrases:
              if any(mh in sn for sn in skill_names):
                  must_have_hits.add(mh)
          if self.must_have_regex:
              matches = set(self.must_have_regex.findall(career_norm))
              must_have_hits.update(matches)
          total_must = len(self.must_have_phrases) if self.must_have_phrases else 1
          must_have_score = min(len(must_have_hits) / total_must, 1.0)
          nice_hits = set()
          for nh in self.nice_to_have_phrases:
              if any(nh in sn for sn in skill_names):
                  nice_hits.add(nh)
          if self.nice_to_have_regex:
              matches = set(self.nice_to_have_regex.findall(career_norm))
              nice_hits.update(matches)
          total_nice = len(self.nice_to_have_phrases) if self.nice_to_have_phrases else 1
          nice_score = min(len(nice_hits) / total_nice, 1.0)
          proficiency_bonus = 0.0
          for name, details in skill_details.items():
              if name in must_have_hits or name in nice_hits:
                  prof = details['proficiency']
                  if prof == 'advanced':
                      proficiency_bonus += 0.05
                  elif prof == 'expert':
                      proficiency_bonus += 0.08
                  if details['duration'] >= 24:
                      proficiency_bonus += 0.04
          proficiency_bonus = min(proficiency_bonus, 0.10)
          skill_score = (0.65 * must_have_score) + (0.25 * nice_score) + (0.10 * proficiency_bonus)
          skill_score = min(skill_score, 1.0)
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
      def __init__(self, config: Dict):
          self.services_companies = set(config['services_companies'])
          self.prestige_companies = set(config['prestige_companies'])
          self.relevant_titles = set(config['relevant_titles'])
          self.evidence_phrases = set(config['evidence_phrases'])
          self.production_keywords = set(config['production_keywords'])
          self.production_bonus_per = config.get('production_bonus_per_keyword', 0.1)
          self.production_bonus_max = config.get('production_bonus_max', 0.15)
          self.prestige_bonus_ratio = config.get('prestige_bonus_ratio', 0.35)
          self.career_weights = config.get('career_weights', DEFAULT_CAREER_WEIGHTS)
          self.yoe_bands = [
              (3, 0.10),
              (5, 0.55),
              (9, 1.00),
              (12, 0.80),
              (float('inf'), 0.60)
          ]
      def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
          if not date_str: return None
          date_str = str(date_str).strip()
          if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
              return datetime.now()
          for pattern in DATE_PATTERNS:
              try:
                  if re.match(pattern, date_str, re.I):
                      parts = re.findall(r'\d+', date_str)
                      if len(parts) == 4:
                          return datetime(int(parts[2]), int(parts[0]), int(parts[1]))
                      elif len(parts) == 3:
                          return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                      elif len(parts) == 2:
                          return datetime(int(parts[1]), int(parts[0]), 1)
                      elif len(parts) == 1:
                          return datetime(int(parts[0]), 12, 31)
              except:
                  pass
          return None
      def _get_yoe_score(self, years_of_experience: float) -> float:
          for max_yoe, score in self.yoe_bands:
              if years_of_experience < max_yoe:
                  return score
          return self.yoe_bands[-1][1]
      def calculate_career_score(self, career_history: List[Dict], years_of_experience: float, current_title: str,
  current_company: str) -> Tuple[float, List[str]]:
          evidence = []
          score_components = []
          total_months = 0
          prestige_months = 0
          product_months = 0
          for job in career_history:
              start = self._parse_date(job.get('start_date'))
              end = self._parse_date(job.get('end_date'))
              company = job.get('company', '').lower()
              desc = job.get('description', '').lower()
              title = job.get('title', '').lower()
              if start and end and end > start:
                  duration_months = (end - start).days / 30.44
              else:
                  duration_months = job.get('duration_months', 0)
                  if duration_months == 0 and job.get('is_current'):
                      duration_months = (datetime.now() - start).days / 30.44 if start else 0
              total_months = max(total_months, duration_months)
              is_services = any(svc in company for svc in self.services_companies)
              if not is_services:
                  product_months += duration_months
              if any(pre in company for pre in self.prestige_companies):
                  prestige_months += duration_months
          yoe_score = self._get_yoe_score(years_of_experience)
          score_components.append(('YoE', self.career_weights['yoe'], yoe_score))
          evidence.append(f"YoE: {years_of_experience:.1f} years (score: {yoe_score:.2f})")
          if total_months > 0:
              product_ratio = product_months / total_months
          else:
              product_ratio = 0.0
          if product_months > 0:
              company_score = 0.40 + (0.60 * product_ratio)
          else:
              company_score = 0.20
          score_components.append(('Company type', self.career_weights['company_type'], company_score))
          evidence.append(f"Company ratio: {product_ratio:.1%} product (score: {company_score:.2f})")
          title_lower = current_title.lower() if current_title else ''
          title_match = any(rel in title_lower for rel in self.relevant_titles)
          title_score = 1.0 if title_match else 0.0
          score_components.append(('Title', self.career_weights['title_relevance'], title_score))
          evidence.append(f"Title match: {title_match} ({current_title})")
          all_descriptions = ' '.join(job.get('description', '') for job in career_history).lower()
          evidence_count = sum(1 for phrase in self.evidence_phrases if phrase in all_descriptions)
          evidence_score = min(evidence_count / 5.0, 1.0)
          score_components.append(('Evidence', self.career_weights['evidence_phrases'], evidence_score))
          if evidence_count > 0:
              evidence.append(f"Evidence phrases: {evidence_count} found")
          production_count = sum(1 for kw in self.production_keywords if kw in all_descriptions)
          production_bonus = min(production_count * self.production_bonus_per, self.production_bonus_max)
          if production_count > 0:
              evidence.append(f"Production keywords: {production_count}")
          prestige_ratio = prestige_months / max(total_months, 1)
          prestige_bonus = prestige_ratio * self.prestige_bonus_ratio
          if prestige_ratio > 0:
              evidence.append(f"Prestige bonus: +{prestige_bonus:.2f}")
          total_weight = sum(w for _, w, _ in score_components)
          weighted_score = sum(score * weight for _, weight, score in score_components) / total_weight
          weighted_score = min(weighted_score + production_bonus + prestige_bonus, 1.0)
          return weighted_score, evidence
      def is_primarily_services(self, career_history: List[Dict]) -> bool:
          if not career_history: return False
          for job in career_history:
              company = job.get('company', '').lower()
              if not any(svc in company for svc in self.services_companies):
                  return False
          return True
      def is_pure_research(self, career_history: List[Dict], titles: List[str]) -> bool:
          has_prod_company = False
          has_prod_keywords = False
          for job in career_history:
              company = job.get('company', '').lower()
              desc = job.get('description', '').lower()
              if any(word in company for word in ['university', 'research lab', 'institute', 'college']):
                  continue
              if any(kw in desc for kw in self.production_keywords):
                  has_prod_keywords = True
              if not any(svc in company for svc in self.services_companies):
                  has_prod_company = True
          prod_titles = any(any(word in t.lower() for word in ['engineer', 'developer', 'applied']) for t in titles)
          return not (has_prod_keywords or has_prod_company) and not prod_titles

  class BehavioralScorer:
      """Score based on redrob signals and recency"""
      def __init__(self, config: Dict):
          self.weights = config.get('behavioral_weights', DEFAULT_BEHAVIORAL_WEIGHTS)
          self.notice_thresholds = [
              (15, 1.0),
              (30, 0.90),
              (60, 0.70),
              (90, 0.45),
              (float('inf'), 0.20)
          ]
          self.active_thresholds = [
              (7, 1.0),
              (14, 0.95),
              (30, 0.85),
              (60, 0.65),
              (90, 0.45),
              (180, 0.20),
              (float('inf'), 0.05)
          ]
      def score_last_active(self, last_active: Optional[str]) -> Tuple[float, str]:
          if not last_active:
              return 0.20, "No last active date"
          try:
              for pattern in DATE_PATTERNS:
                  if re.match(pattern, last_active, re.I):
                      parsed = self._parse_date(last_active)
                      if parsed:
                          days_ago = (datetime.now() - parsed).days
                          for max_days, score in self.active_thresholds:
                              if days_ago < max_days:
                                  return score, f"Active {days_ago}d ago"
          except:
              pass
          return 0.20, "Unclear last active"
      def _parse_date(self, date_str: str) -> Optional[datetime]:
          try:
              date_str = str(date_str).strip()
              for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']:
                  try:
                      return datetime.strptime(date_str, fmt)
                  except:
                      pass
              if re.match(r'^\d{4}$', date_str):
                  return datetime(int(date_str), 12, 31)
          except:
              pass
          return None
      def calculate_behavioral_score(self, redrob_signals: Dict[str, Any], last_active_date: Optional[str] = None) ->
  Tuple[float, List[str]]:
          evidence = []
          scores = {}
          active_score, active_evidence = self.score_last_active(last_active_date or
  redrob_signals.get('last_active_date'))
          scores['last_active'] = active_score
          evidence.append(active_evidence)
          open_to_work = redrob_signals.get('open_to_work_flag', False)
          scores['open_to_work'] = 1.0 if open_to_work else 0.45
          evidence.append(f"Open to work: {open_to_work}")
          response_rate = redrob_signals.get('recruiter_response_rate', 0.0)
          if isinstance(response_rate, (int, float)):
              scores['recruiter_response_rate'] = min(max(response_rate, 0.0), 1.0)
              evidence.append(f"Response rate: {response_rate:.1%}")
          else:
              scores['recruiter_response_rate'] = 0.5
              evidence.append("Response rate: N/A")
          interview_rate = redrob_signals.get('interview_completion_rate', 0.0)
          if isinstance(interview_rate, (int, float)):
              scores['interview_completion_rate'] = min(max(interview_rate, 0.0), 1.0)
              evidence.append(f"Interview rate: {interview_rate:.1%}")
          else:
              scores['interview_completion_rate'] = 0.5
              evidence.append("Interview rate: N/A")
          notice_days = redrob_signals.get('notice_period_days', 999)
          for max_days, score in self.notice_thresholds:
              if notice_days <= max_days:
                  notice_score = score
                  break
          scores['notice_period'] = notice_score
          evidence.append(f"Notice: {notice_days} days")
          github_score = redrob_signals.get('github_activity_score', 0)
          if github_score == -1 or github_score is None:
              github_norm = 0.30
          else:
              github_norm = min(max(github_score, 0.0), 100) / 100.0
          scores['github_activity'] = github_norm
          evidence.append(f"GitHub: {github_norm:.1%}")
          weighted_score = sum(scores[key] * self.weights[key] for key in self.weights if key in scores)
          return weighted_score, evidence

  class LocationScorer:
      """Score based on location"""
      def __init__(self, config: Dict):
          self.preferred = set(config['preferred_cities'])
          other = set(config.get('other_indian_cities', []))
          self.indian_cities = self.preferred | other
      def normalize_location(self, location: str) -> Tuple[str, Optional[str]]:
          if not location: return "", None
          loc = location.lower().strip()
          parts = re.split(r'[,\|]', loc)
          city = parts[0].strip() if parts else loc
          country = parts[-1].strip() if len(parts) > 1 else None
          city = re.sub(r'[^a-z\s]', '', city).strip()
          return city, country
      def calculate_location_score(self, location: str, country: Optional[str], willing_to_relocate: bool = False) ->
  Tuple[float, List[str]]:
          evidence = []
          city, detected_country = self.normalize_location(location)
          if not city:
              return 0.10, ["Unknown location"]
          if city in self.preferred:
              evidence.append(f"City {city} is preferred")
              return 1.0, evidence
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
      def __init__(self, config: Dict):
          honeypot_cfg = config.get('honeypot', {})
          self.max_advanced_zero_duration = honeypot_cfg.get('max_advanced_zero_duration', 5)
          self.max_yoe = honeypot_cfg.get('max_yoe', 35)
          self.services_companies = set(config['services_companies'])
      def check_honeypot(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
          skills = candidate.get('skills', [])
          advanced_with_no_duration = sum(
              1 for s in skills
              if s.get('proficiency') in ['advanced', 'expert'] and s.get('duration_months', 0) == 0
          )
          if advanced_with_no_duration >= self.max_advanced_zero_duration:
              return True, f"{self.max_advanced_zero_duration}+ advanced skills with 0 duration
  ({advanced_with_no_duration})"
          career = candidate.get('career_history', [])
          for job in career:
              start = job.get('start_date')
              end = job.get('end_date')
              if start and end and job.get('is_current') != True:
                  try:
                      if self._dates_negative(start, end):
                          return True, f"Negative tenure: {job.get('company')} {start}-{end}"
                  except:
                      pass
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
          if years_of_exp > self.max_yoe:
              return True, f"Impossible YoE: {years_of_exp} years"
          return False, ""
      def _dates_negative(self, start: str, end: str) -> bool:
          start_year = re.search(r'(\d{4})', str(start))
          end_year = re.search(r'(\d{4})', str(end))
          if start_year and end_year:
              start_y = int(start_year.group(1))
              end_y = int(end_year.group(1))
              if end_y < start_y:
                  return True
              if end_y == start_y:
                  start_m = re.search(r'-(\d{1,2})', str(start))
                  end_m = re.search(r'-(\d{1,2})', str(end))
                  if start_m and end_m:
                      if int(end_m.group(1)) < int(start_m.group(1)):
                          return True
          return False

  class CandidateRanker:
      """Main ranking system"""
      def __init__(self, config: Dict):
          self.config = config
          self.skill_matcher = SkillMatcher(config)
          self.career_scorer = CareerScorer(config)
          self.behavioral_scorer = BehavioralScorer(config)
          self.location_scorer = LocationScorer(config)
          self.honeypot_detector = HoneypotDetector(config)
          self.pure_research_penalty = config.get('pure_research_penalty', 0.6)
          self.disqualifier_multipliers = config['disqualifier_multipliers']
          self.disqualify_titles = set(config['disqualify_titles'])
      def parse_profiles(self, input_path: str, max_candidates: int = None) -> List[Candidate]:
          candidates = []
          with open(input_path, 'r', encoding='utf-8') as f:
              for i, line in enumerate(f):
                  if max_candidates and i >= max_candidates:
                      break
                  try:
                      data = json.loads(line.strip())
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
      def check_disqualifiers(self, candidate: Candidate) -> float:
          multiplier = 1.0
          reasons = []
          if self.career_scorer.is_primarily_services(candidate.career_history):
              multiplier *= self.disqualifier_multipliers.get('services_only', 0.30)
              reasons.append("entirely services companies (x0.30)")
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
              multiplier *= self.disqualifier_multipliers.get('cv_speech_robotics_only', 0.20)
              reasons.append("CV/speech/robotics only, no NLP/IR (x0.20)")
          current_title = candidate.profile.get('current_title', '').lower()
          if any(disq in current_title for disq in self.disqualify_titles):
              multiplier *= self.disqualifier_multipliers.get('unrelated_title', 0.05)
              reasons.append(f"unrelated title: {current_title} (x0.05)")
          if reasons:
              candidate.reasoning_parts.append(f"Disqualifiers: {', '.join(reasons)}")
          return multiplier
      def calculate_score(self, candidate: Candidate) -> None:
          profile = candidate.profile
          years_of_exp = profile.get('years_of_experience', 0.0)
          current_title = profile.get('current_title', '')
          current_company = profile.get('current_company', '')
          location = profile.get('location', '')
          skill_score, skill_evidence = self.skill_matcher.calculate_skill_score(candidate.skills,
  candidate.career_history)
          candidate.skill_score = skill_score
          candidate.reasoning_parts.extend(skill_evidence)
          career_score, career_evidence = self.career_scorer.calculate_career_score(
              candidate.career_history, years_of_exp, current_title, current_company
          )
          candidate.career_score = career_score
          candidate.reasoning_parts.extend(career_evidence)
          if self.career_scorer.is_pure_research(
              candidate.career_history,
              [current_title] + [j.get('title', '') for j in candidate.career_history]
          ):
              candidate.career_score *= self.pure_research_penalty
              candidate.reasoning_parts.append(f"Pure research background (x{self.pure_research_penalty})")
          behavioral_score, behavioral_evidence = self.behavioral_scorer.calculate_behavioral_score(
              candidate.redrob_signals,
              profile.get('last_active_date')
          )
          candidate.behavioral_score = behavioral_score
          candidate.reasoning_parts.extend(behavioral_evidence)
          willing_to_relocate = candidate.redrob_signals.get('willing_to_relocate', False)
          location_score, location_evidence = self.location_scorer.calculate_location_score(
              location,
              profile.get('country'),
              willing_to_relocate
          )
          candidate.location_score = location_score
          candidate.reasoning_parts.extend(location_evidence)
          disqualifier_mult = self.check_disqualifiers(candidate)
          candidate.disqualifier_multiplier = disqualifier_mult
          raw_score = (
              self.config['component_weights']['skill'] * skill_score +
              self.config['component_weights']['career'] * career_score +
              self.config['component_weights']['behavioral'] * behavioral_score +
              self.config['component_weights']['location'] * location_score
          )
          candidate.final_score = min(raw_score * disqualifier_mult, 1.0)
          candidate.final_score = max(candidate.final_score, 0.0)
          key_facts = []
          key_facts.append(f"Title: {current_title}, YoE: {years_of_exp:.0f}")
          key_facts.append(f"Career: {career_score:.2f}")
          loc_city = location.split(',')[0] if location else 'Unknown'
          key_facts.append(f"Loc: {loc_city}")
          candidate.reasoning_parts = ['. '.join(key_facts)]

  def parse_candidates(input_path: str, max_candidates: int = None) -> List[Candidate]:
      candidates = []
      with open(input_path, 'r', encoding='utf-8') as f:
          for i, line in enumerate(f):
              if max_candidates and i >= max_candidates:
                  break
              try:
                  data = json.loads(line.strip())
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

  _GLOBAL_RANKER = None
  def init_worker(config_dict: Dict):
      global _GLOBAL_RANKER
      _GLOBAL_RANKER = CandidateRanker(config_dict)
  def score_candidate(candidate: Candidate) -> Candidate:
      _GLOBAL_RANKER.calculate_score(candidate)
      return candidate

  def main():
      parser = argparse.ArgumentParser(description="Rank candidate profiles against job description")
      parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl file')
      parser.add_argument('--out', required=True, help='Output CSV file path')
      parser.add_argument('--max', type=int, help='Maximum candidates to process (for testing)')
      parser.add_argument('--config', default='config.json', help='Path to config JSON file')
      parser.add_argument('--jobs', type=int, default=1, help='Number of parallel workers (default: 1, use 0 for all
  CPUs)')
      args = parser.parse_args()
      config = load_config(args.config)
      top_n = config.get('top_n', 100)
      num_jobs = args.jobs if args.jobs > 0 else os.cpu_count() or 1
      print("=" * 60)
      print("CANDIDATE RANKING SYSTEM (IMPROVED)")
      print("=" * 60)
      print(f"Input: {args.candidates}")
      print(f"Output: {args.out}")
      print(f"Config: {args.config}")
      print(f"Workers: {num_jobs}")
      print("=" * 60)
      candidates = parse_candidates(args.candidates, args.max)
      if not candidates:
          print("ERROR: No candidates loaded")
          return 1
      honeypot_detector = HoneypotDetector(config)
      candidates_to_score = []
      skipped_honeypots = 0
      for c in candidates:
          is_honeypot, reason = honeypot_detector.check_honeypot(c.__dict__)
          if is_honeypot:
              skipped_honeypots += 1
              c.is_honeypot = True
              c.honeypot_reason = reason
          else:
              candidates_to_score.append(c)
      print(f"After honeypot filter: {len(candidates_to_score)} candidates to score (skipped {skipped_honeypots})")
      if not candidates_to_score:
          print("ERROR: No candidates to score after filtering")
          return 1
      start_time = datetime.now()
      if num_jobs > 1:
          print(f"Using {num_jobs} parallel workers")
          scored_candidates = []
          with ProcessPoolExecutor(max_workers=num_jobs, initializer=init_worker, initargs=(config,)) as executor:
              for i, candidate in enumerate(executor.map(score_candidate, candidates_to_score, chunksize=1000)):
                  scored_candidates.append(candidate)
                  if (i+1) % 10000 == 0:
                      print(f"  Progress: {i+1}/{len(candidates_to_score)}")
      else:
          print("Running sequentially")
          ranker = CandidateRanker(config)
          scored_candidates = []
          for i, candidate in enumerate(candidates_to_score):
              if (i+1) % 10000 == 0:
                  print(f"  Progress: {i+1}/{len(candidates_to_score)}")
              ranker.calculate_score(candidate)
              scored_candidates.append(candidate)
      elapsed = (datetime.now() - start_time).total_seconds()
      print(f"Scoring completed in {elapsed:.2f} seconds")
      scored_candidates.sort(key=lambda c: (-c.final_score, c.candidate_id))
      top_candidates = scored_candidates[:top_n]
      data = []
      for rank, candidate in enumerate(top_candidates, 1):
          data.append({
              'candidate_id': candidate.candidate_id,
              'rank': rank,
              'score': round(candidate.final_score, 4),
              'reasoning': '. '.join(candidate.reasoning_parts)[:300]
          })
      df = pd.DataFrame(data)
      for i in range(1, len(df)):
          if df.iloc[i]['score'] > df.iloc[i-1]['score'] + 0.0001:
              print(f"Warning: Score not monotonic at rank {i+1} (score {df.iloc[i]['score']} >
  {df.iloc[i-1]['score']})")
              df.at[i, 'score'] = df.iloc[i-1]['score']
      df.to_csv(args.out, index=False)
      print(f"Saved top {len(df)} candidates to {args.out}")
      print(f"\nTop 10:")
      print(df.head(10).to_string(index=False))
      print("\n" + "=" * 60)
      print(f"Completed in {elapsed:.2f} seconds")
      print(f"Final score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
      print("=" * 60)
      return 0

  if __name__ == '__main__':
      sys.exit(main())