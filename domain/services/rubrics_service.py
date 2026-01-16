# domain/services/rubrics_service.py
from typing import Dict, List, Any, Optional
import re

from domain.models.evaluation import RubricCategory, RubricScore


class RubricsService:
    def evaluate(self, signals: Dict[str, float]) -> Dict[RubricCategory, RubricScore]:
        results: Dict[RubricCategory, RubricScore] = {}

        for category in RubricCategory:
            raw_value: float = signals.get(category.value, 50.0)

            bounded_score: int = min(100, max(0, int(raw_value)))

            results[category] = RubricScore(
                score=bounded_score,
                rationale=f"Score derived from {category.value}",
            )

        return results


class RubricEvaluator:
    """Domain-level rubric evaluator with specific evaluation methods."""

    @staticmethod
    def evaluate_factual_accuracy(
        analysis: str,
        expected_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate factual accuracy of an analysis."""
        score = 1.0
        issues = []

        # Check for presence of numbers (basic heuristic)
        has_numbers = bool(re.search(r'\d+', analysis))
        if has_numbers:
            score += 0.5

        # Check if expected values are present
        if expected_values:
            for key, expected in expected_values.items():
                if str(expected) in analysis:
                    score += 0.25
                else:
                    issues.append(f"Expected value for {key} not found")

        return {
            "score": min(2.0, score),
            "passed": score >= 1.0,
            "issues": issues,
            "rationale": "Factual accuracy evaluated based on numeric content and expected values"
        }

    @staticmethod
    def evaluate_source_fidelity(
        analysis: str,
        source_documents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate source fidelity of an analysis."""
        score = 0.5
        issues = []

        if source_documents:
            score += 0.5
            # Check if source document content is referenced
            for doc in source_documents:
                if doc.get('content') and any(
                    word.lower() in analysis.lower()
                    for word in doc.get('content', '').split()[:10]
                ):
                    score += 0.25

        return {
            "score": min(2.0, score),
            "passed": score >= 1.0,
            "issues": issues,
            "rationale": "Source fidelity evaluated based on document references"
        }

    @staticmethod
    def evaluate_regulatory_compliance(analysis: str) -> Dict[str, Any]:
        """Evaluate regulatory compliance awareness."""
        score = 1.0
        compliance_keywords = [
            "sec", "gaap", "ifrs", "regulation", "compliance",
            "disclosure", "filing", "audit", "sox"
        ]

        mentions = sum(1 for kw in compliance_keywords if kw in analysis.lower())
        score += min(1.0, mentions * 0.25)

        return {
            "score": score,
            "passed": score >= 1.0,
            "issues": [],
            "rationale": f"Found {mentions} compliance-related mentions"
        }

    @staticmethod
    def evaluate_completeness(analysis: str) -> Dict[str, Any]:
        """Evaluate analysis completeness."""
        word_count = len(analysis.split())
        sections = ["financial", "risk", "operational", "strategic", "recommendation"]
        sections_found = sum(1 for s in sections if s in analysis.lower())

        score = min(2.0, (word_count / 200) + (sections_found * 0.2))

        return {
            "score": score,
            "passed": score >= 1.0,
            "issues": [],
            "rationale": f"Analysis has {word_count} words and covers {sections_found}/5 key sections"
        }

    @staticmethod
    def evaluate_clarity(analysis: str) -> Dict[str, Any]:
        """Evaluate clarity of the analysis."""
        sentences = analysis.count('.') + analysis.count('!') + analysis.count('?')
        words = len(analysis.split())
        avg_sentence_len = words / max(sentences, 1)

        if avg_sentence_len < 20:
            score = 2.0
        elif avg_sentence_len < 30:
            score = 1.5
        else:
            score = 1.0

        return {
            "score": score,
            "passed": score >= 1.0,
            "issues": [],
            "rationale": f"Average sentence length: {avg_sentence_len:.1f} words"
        }

    @staticmethod
    def evaluate_uncertainty(analysis: str) -> Dict[str, Any]:
        """Evaluate handling of uncertainty in the analysis."""
        uncertainty_markers = [
            "may", "might", "could", "approximately", "estimated",
            "uncertain", "risk", "potentially", "likely"
        ]

        mentions = sum(1 for m in uncertainty_markers if m in analysis.lower())
        score = 1.0 + min(1.0, mentions * 0.2)

        return {
            "score": score,
            "passed": True,
            "issues": [],
            "rationale": f"Found {mentions} uncertainty markers"
        }

    @staticmethod
    def evaluate_consistency(analysis: str) -> Dict[str, Any]:
        """Evaluate internal consistency of the analysis."""
        # Basic heuristic: look for contradictory statements
        contradictions = [
            ("increase", "decrease"),
            ("growth", "decline"),
            ("positive", "negative"),
            ("strong", "weak")
        ]

        issues = []
        for word1, word2 in contradictions:
            if word1 in analysis.lower() and word2 in analysis.lower():
                issues.append(f"Potential contradiction: both '{word1}' and '{word2}' used")

        score = 2.0 - len(issues) * 0.5

        return {
            "score": max(0, score),
            "passed": score >= 1.0,
            "issues": issues,
            "rationale": "Consistency check based on contradictory terms"
        }

    @staticmethod
    def evaluate_all_rubrics(
        analysis: str,
        expected_values: Optional[Dict[str, Any]] = None,
        source_documents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all rubrics for an analysis."""
        return {
            "factual_accuracy": RubricEvaluator.evaluate_factual_accuracy(analysis, expected_values),
            "source_fidelity": RubricEvaluator.evaluate_source_fidelity(analysis, source_documents),
            "regulatory_compliance": RubricEvaluator.evaluate_regulatory_compliance(analysis),
            "completeness": RubricEvaluator.evaluate_completeness(analysis),
            "clarity": RubricEvaluator.evaluate_clarity(analysis),
            "uncertainty": RubricEvaluator.evaluate_uncertainty(analysis),
            "consistency": RubricEvaluator.evaluate_consistency(analysis),
        }
