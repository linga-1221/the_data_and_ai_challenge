"""
Evaluation Framework

Computes ranking quality metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)

Supports both automatic evaluation (if ground truth available) and
statistical analysis of ranking distribution.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics at various k values"""
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    precision_at_50: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    recall_at_50: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    ndcg_at_50: float = 0.0
    map_at_10: float = 0.0
    map_at_20: float = 0.0
    map_at_50: float = 0.0
    num_queries: int = 0
    num_relevant: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'precision_at_5': round(self.precision_at_5, 4),
            'precision_at_10': round(self.precision_at_10, 4),
            'precision_at_20': round(self.precision_at_20, 4),
            'precision_at_50': round(self.precision_at_50, 4),
            'recall_at_10': round(self.recall_at_10, 4),
            'recall_at_20': round(self.recall_at_20, 4),
            'recall_at_50': round(self.recall_at_50, 4),
            'ndcg_at_5': round(self.ndcg_at_5, 4),
            'ndcg_at_10': round(self.ndcg_at_10, 4),
            'ndcg_at_20': round(self.ndcg_at_20, 4),
            'ndcg_at_50': round(self.ndcg_at_50, 4),
            'map_at_10': round(self.map_at_10, 4),
            'map_at_20': round(self.map_at_20, 4),
            'map_at_50': round(self.map_at_50, 4),
            'num_queries': self.num_queries,
            'num_relevant': self.num_relevant
        }


class RankingEvaluator:
    """
    Evaluates ranking quality using standard IR metrics.

    Requires ground truth: a mapping of query -> set of relevant candidate IDs.
    For this competition, you'll need a gold standard set of "correct" hires or
    recruiter-labeled relevance.
    """

    def __init__(self):
        self._dcg_cache: Dict[Tuple[Tuple[int, ...], int], float] = {}

    def evaluate(
        self,
        rankings: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [5, 10, 20, 50],
        relevance_scores: Optional[Dict[str, Dict[str, int]]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate rankings against ground truth.

        Args:
            rankings: Dict mapping query_id to list of (candidate_id, score) sorted by score
            ground_truth: Dict mapping query_id to list of relevant candidate_ids
            k_values: List of k values for metrics
            relevance_scores: Optional dict of query->{candidate: relevance_score}. If None, binary relevance.

        Returns:
            EvaluationMetrics object with computed metrics
        """
        metrics = EvaluationMetrics()
        metrics.num_queries = len(rankings)

        # Ensure we have ground truth for all queries
        queries = list(rankings.keys())
        if not queries:
            logger.warning("No rankings provided for evaluation")
            return metrics

        # Accumulate metrics across queries
        all_precision = {k: [] for k in k_values}
        all_recall = {k: [] for k in k_values}
        all_ndcg = {k: [] for k in k_values}
        all_ap = {k: [] for k in k_values}

        total_relevant = 0

        for query in queries:
            ranking = rankings[query]
            relevant_set = set(ground_truth.get(query, []))
            total_relevant += len(relevant_set)

            # Get relevance scores if available
            query_rel_scores = relevance_scores.get(query, {}) if relevance_scores else {}

            for k in k_values:
                if k > len(ranking):
                    continue

                # Precision@k
                top_k = [cid for cid, _ in ranking[:k]]
                num_relevant_at_k = len([cid for cid in top_k if cid in relevant_set])
                precision = num_relevant_at_k / k if k > 0 else 0
                all_precision[k].append(precision)

                # Recall@k
                recall = num_relevant_at_k / len(relevant_set) if relevant_set else 0
                all_recall[k].append(recall)

                # NDCG@k
                ndcg = self._ndcg_at_k(ranking, relevant_set, k, query_rel_scores)
                all_ndcg[k].append(ndcg)

                # Average Precision (for MAP)
                ap = self._average_precision(ranking, relevant_set)
                all_ap[k].append(ap)

        # Compute averages
        if metrics.num_queries > 0:
            for k in k_values:
                if all_precision[k]:
                    setattr(metrics, f'precision_at_{k}', np.mean(all_precision[k]))
                if all_recall[k]:
                    setattr(metrics, f'recall_at_{k}', np.mean(all_recall[k]))
                if all_ndcg[k]:
                    setattr(metrics, f'ndcg_at_{k}', np.mean(all_ndcg[k]))
                if all_ap[k]:
                    setattr(metrics, f'map_at_{k}', np.mean(all_ap[k]))

            metrics.num_relevant = total_relevant // metrics.num_queries if metrics.num_queries else 0

        return metrics

    def _dcg_at_k(
        self,
        ranking: List[Tuple[str, float]],
        relevance: Dict[str, int],
        k: int
    ) -> float:
        """Compute Discounted Cumulative Gain at K"""
        dcg = 0.0
        for i, (cid, _) in enumerate(ranking[:k], 1):
            rel = relevance.get(cid, 0)
            dcg += rel / np.log2(i + 1)
        return dcg

    def _ndcg_at_k(
        self,
        ranking: List[Tuple[str, float]],
        relevant_set: set,
        k: int,
        relevance_scores: Dict[str, int]
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain at K"""
        # Build relevance dict for this ranking
        if relevance_scores:
            rel_dict = {cid: relevance_scores.get(cid, 0) for cid, _ in ranking[:k]}
        else:
            # Binary relevance
            rel_dict = {cid: 1 if cid in relevant_set else 0 for cid, _ in ranking[:k]}

        # DCG
        dcg = self._dcg_at_k(ranking, rel_dict, k)

        # Ideal DCG - sort by relevance descending
        ideal_ranking = sorted(
            [(cid, rel_dict.get(cid, 0)) for cid in relevant_set],
            key=lambda x: x[1],
            reverse=True
        )[:k]
        idcg = self._dcg_at_k(ideal_ranking, {cid: score for cid, score in ideal_ranking}, k)

        if idcg > 0:
            return dcg / idcg
        return 0.0

    def _average_precision(
        self,
        ranking: List[Tuple[str, float]],
        relevant_set: set
    ) -> float:
        """Compute Average Precision for a single query"""
        if not relevant_set:
            return 0.0

        ap = 0.0
        num_relevant_seen = 0

        for i, (cid, _) in enumerate(ranking, 1):
            if cid in relevant_set:
                num_relevant_seen += 1
                precision_at_i = num_relevant_seen / i
                ap += precision_at_i

        # Normalize by number of relevant documents
        return ap / len(relevant_set) if relevant_set else 0.0

    def compute_ranking_statistics(
        self,
        rankings: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Any]:
        """
        Compute statistical analysis of ranking distribution.

        Args:
            rankings: Dict mapping query_id to sorted (candidate_id, score) list

        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_queries': len(rankings),
            'avg_candidates_per_query': np.mean([len(r) for r in rankings.values()]),
            'score_ranges': [],
            'score_gaps': [],
            'monotonicity_violations': 0
        }

        all_gaps = []
        for query, ranking in rankings.items():
            if len(ranking) < 2:
                continue

            # Check monotonicity (scores should be non-increasing)
            scores = [score for _, score in ranking]
            for i in range(1, len(scores)):
                if scores[i] > scores[i-1] + 1e-6:
                    stats['monotonicity_violations'] += 1

            # Compute gaps between consecutive scores
            gaps = [scores[i-1] - scores[i] for i in range(1, len(scores))]
            all_gaps.extend(gaps)

            # Score range for this query
            if scores:
                stats['score_ranges'].append({
                    'query': query,
                    'min': float(min(scores)),
                    'max': float(max(scores)),
                    'range': float(max(scores) - min(scores))
                })

        if all_gaps:
            stats['avg_score_gap'] = float(np.mean(all_gaps))
            stats['median_score_gap'] = float(np.median(all_gaps))
            stats['min_score_gap'] = float(min(all_gaps))
            stats['max_score_gap'] = float(max(all_gaps))
        else:
            stats.update({
                'avg_score_gap': 0.0,
                'median_score_gap': 0.0,
                'min_score_gap': 0.0,
                'max_score_gap': 0.0
            })

        # Gap distribution analysis
        if all_gaps:
            percentile_gaps = [10, 50, 90]
            stats['gap_percentiles'] = {
                f'p{p}': float(np.percentile(all_gaps, p))
                for p in percentile_gaps
            }

        return stats


class CandidateRankingEvaluator:
    """
    Convenience wrapper for evaluating candidate rankings with ground truth.

    Can load ground truth from various formats:
    - List of positive candidate IDs per job
    - Recruiter-labeled relevance scores
    - Historical hire data
    """

    def __init__(self):
        self.evaluator = RankingEvaluator()

    def evaluate_from_file(
        self,
        rankings_csv_path: str,
        ground_truth_csv_path: str,
        query_id: str = "default",
        candidate_id_col: str = "candidate_id",
        score_col: str = "score",
        relevant_col: str = "is_relevant"
    ) -> EvaluationMetrics:
        """
        Load rankings and ground truth from CSV files and evaluate.

        Args:
            rankings_csv_path: Path to rankings CSV (candidate_id, score, etc.)
            ground_truth_csv_path: Path to ground truth CSV
            query_id: Single query ID for single-job evaluation
            candidate_id_col: Column name for candidate ID
            score_col: Column name for ranking score
            relevant_col: Column name for relevance (binary or graded)

        Returns:
            EvaluationMetrics
        """
        import pandas as pd

        # Load rankings
        rankings_df = pd.read_csv(rankings_csv_path)
        ranking = list(zip(
            rankings_df[candidate_id_col].astype(str),
            rankings_df[score_col].astype(float)
        ))

        # Load ground truth
        gt_df = pd.read_csv(ground_truth_csv_path)
        relevant_cids = set(gt_df[gt_df[relevant_col] > 0][candidate_id_col].astype(str))

        # Build structures
        rankings_dict = {query_id: ranking}
        ground_truth_dict = {query_id: list(relevant_cids)}

        return self.evaluator.evaluate(rankings_dict, ground_truth_dict)

    def evaluate_multi_query_from_file(
        self,
        rankings_csv_path: str,
        ground_truth_csv_path: str,
        query_col: str = "query_id",
        candidate_id_col: str = "candidate_id",
        score_col: str = "score",
        relevance_col: str = "relevance"
    ) -> EvaluationMetrics:
        """
        Evaluate multiple queries (jobs) from CSV.

        CSV format:
        query_id,candidate_id,score,relevance
        """
        import pandas as pd

        rankings_df = pd.read_csv(rankings_csv_path)
        gt_df = pd.read_csv(ground_truth_csv_path)

        # Build rankings dict
        rankings_dict = {}
        for query_id, group in rankings_df.groupby(query_col):
            ranking = list(zip(
                group[candidate_id_col].astype(str),
                group[score_col].astype(float)
            ))
            rankings_dict[str(query_id)] = ranking

        # Build ground truth dict with graded relevance if available
        ground_truth_dict = {}
        relevance_scores = {}
        for query_id, group in gt_df.groupby(query_col):
            ground_truth_dict[str(query_id)] = list(
                group[group[relevance_col] > 0][candidate_id_col].astype(str)
            )
            # Grade relevance scores
            relevance_scores[str(query_id)] = dict(
                zip(
                    group[candidate_id_col].astype(str),
                    group[relevance_col].astype(int)
                )
            )

        return self.evaluator.evaluate(
            rankings_dict,
            ground_truth_dict,
            relevance_scores=relevance_scores
        )


def create_evaluator() -> RankingEvaluator:
    """Factory function"""
    return RankingEvaluator()


if __name__ == '__main__':
    # Test evaluation
    evaluator = RankingEvaluator()

    # Simulated ground truth: these are the "correct" top candidates for this job
    ground_truth = {
        'job_001': ['CAND_0001', 'CAND_0003', 'CAND_0005', 'CAND_0007', 'CAND_0010']
    }

    # Rankings to evaluate: list of (candidate_id, score) sorted by score
    rankings = {
        'job_001': [
            ('CAND_0001', 0.95),
            ('CAND_0003', 0.92),
            ('CAND_0002', 0.88),  # Not in ground truth
            ('CAND_0005', 0.85),
            ('CAND_0004', 0.82),  # Not in ground truth
            ('CAND_0007', 0.78),
            ('CAND_0008', 0.75),  # Not in ground truth
            ('CAND_0009', 0.72),  # Not in ground truth
            ('CAND_0010', 0.68),
            ('CAND_0011', 0.65),  # Not in ground truth
        ]
    }

    metrics = evaluator.evaluate(rankings, ground_truth, k_values=[5, 10, 20, 50])

    print("Evaluation Metrics:")
    for field, value in metrics.to_dict().items():
        print(f"  {field}: {value}")

    # Statistics
    stats = evaluator.compute_ranking_statistics(rankings)
    print("\nRanking Statistics:")
    for key, value in stats.items():
        if key != 'score_ranges':
            print(f"  {key}: {value}")
