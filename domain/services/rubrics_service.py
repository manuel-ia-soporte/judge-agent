# domain/services/rubrics_service.py
from typing import Dict, List, Any, Optional
import re
from datetime import datetime, timedelta

from domain.models.evaluation import (
    RubricCategory,
    RubricScore,
    RubricEvaluation,
)
from domain.models.finance import FinancialAnalysis, SECDocument


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
    def _build_evaluation(
        rubric_name: str,
        score: float,
        passed: bool,
        feedback: str,
        evidence: Optional[List[str]] = None,
        confidence: float = 0.9,
    ) -> RubricEvaluation:
        return RubricEvaluation(
            rubric_name=rubric_name,
            score=min(2.0, max(0.0, score)),
            is_passed=passed,
            feedback=feedback,
            evidence=evidence or [],
            confidence_score=max(0.0, min(1.0, confidence)),
        )

    @staticmethod
    def evaluate_factual_accuracy(
        analysis: FinancialAnalysis,
        expected_values: Optional[Dict[str, Any]] = None,
    ) -> RubricEvaluation:
        content = analysis.content
        score = 1.0
        evidence: List[str] = []
        issues: List[str] = []

        if re.search(r"\d+", content):
            score += 0.5
            evidence.append("Contains quantitative references")

        if expected_values:
            for key, expected in expected_values.items():
                expected_str = f"{expected:,}" if isinstance(expected, (int, float)) else str(expected)
                if expected_str.replace(",", "") in content.replace(",", ""):
                    score += 0.2
                    evidence.append(f"Matches expected {key}")
                else:
                    issues.append(f"Missing explicit reference to {key}")

        passed = score >= 1.0 and not issues
        feedback = (
            "All referenced figures align with expectations"
            if not issues
            else ", ".join(issues)
        )
        return RubricEvaluator._build_evaluation(
            "factual_accuracy", score, passed, feedback, evidence
        )

    @staticmethod
    def evaluate_source_fidelity(
        analysis: FinancialAnalysis,
        source_documents: Optional[List[SECDocument]] = None,
    ) -> RubricEvaluation:
        content = analysis.content.lower()
        score = 0.8 if analysis.source_documents else 0.5
        evidence: List[str] = []

        docs = source_documents or analysis.source_documents
        for doc in docs or []:
            doc_ref = doc.filing_type.value.lower() if isinstance(doc, SECDocument) else str(doc)
            if doc_ref in content:
                score += 0.2
                evidence.append(f"References {doc_ref.upper()}")

        passed = score >= 1.0
        feedback = (
            "Sources explicitly cited"
            if passed
            else "Provide explicit citations to SEC filings"
        )
        return RubricEvaluator._build_evaluation(
            "source_fidelity", score, passed, feedback, evidence
        )

    @staticmethod
    def evaluate_regulatory_compliance(
        analysis: FinancialAnalysis,
    ) -> RubricEvaluation:
        content = analysis.content.lower()
        compliance_keywords = [
            "sec",
            "gaap",
            "ifrs",
            "regulation",
            "compliance",
            "disclosure",
            "filing",
            "audit",
            "sox",
        ]
        promotional_phrases = ["buy", "sell", "guaranteed", "must"]
        mentions = sum(1 for kw in compliance_keywords if kw in content)
        penalties = sum(1 for phrase in promotional_phrases if phrase in content)

        score = 1.0 + min(1.0, mentions * 0.2) - penalties * 0.2
        passed = score >= 1.0
        feedback = (
            f"References {mentions} compliance terms"
            if passed
            else "Avoid promotional language; cite compliance context"
        )
        return RubricEvaluator._build_evaluation(
            "regulatory_compliance", score, passed, feedback
        )

    @staticmethod
    def evaluate_financial_reasoning(
        analysis: FinancialAnalysis,
    ) -> RubricEvaluation:
        metrics = analysis.metrics_used or []
        score = 1.0 + min(0.5, len(metrics) * 0.1)
        if re.search(r"margin|ratio|guidance|forecast", analysis.content.lower()):
            score += 0.3

        passed = score >= 1.0
        feedback = (
            "Uses multiple financial metrics"
            if passed
            else "Incorporate metrics and reasoning"
        )
        evidence = [f"Metrics referenced: {len(metrics)}"] if metrics else []
        return RubricEvaluator._build_evaluation(
            "financial_reasoning", score, passed, feedback, evidence
        )

    @staticmethod
    def evaluate_materiality_relevance(
        analysis: FinancialAnalysis,
    ) -> RubricEvaluation:
        content = analysis.content.lower()
        material_terms = ["material", "significant", "major", "impact"]
        mentions = sum(1 for term in material_terms if term in content)
        score = 0.8 + min(0.8, mentions * 0.2)

        passed = score >= 1.0
        feedback = (
            "Clearly distinguishes material impacts"
            if passed
            else "Highlight material impacts and relevance"
        )
        return RubricEvaluator._build_evaluation(
            "materiality_relevance", score, passed, feedback
        )

    @staticmethod
    def evaluate_completeness(analysis: FinancialAnalysis) -> RubricEvaluation:
        content = analysis.content.lower()
        sections = ["financial", "risk", "operational", "strategic", "recommend"]
        covered_sections = [section for section in sections if section in content]
        sections_found = len(covered_sections)
        word_count = len(analysis.content.split())
        score = min(2.0, (word_count / 200.0) + sections_found * 0.2)
        passed = score >= 1.0
        missing_sections = [section for section in sections if section not in covered_sections]
        evidence = [f"Includes {section} section" for section in covered_sections]
        if not evidence:
            evidence.append("Includes 0 key sections")
        evidence.append(f"Word count: {word_count}")
        feedback = (
            "Analysis covers core sections"
            if passed
            else f"Missing sections: {', '.join(missing_sections) or 'key areas'}"
        )
        return RubricEvaluator._build_evaluation(
            "completeness", score, passed, feedback, evidence
        )

    @staticmethod
    def evaluate_consistency(analysis: FinancialAnalysis) -> RubricEvaluation:
        content = analysis.content.lower()
        contradictions = [
            ("increase", "decrease"),
            ("growth", "decline"),
            ("positive", "negative"),
            ("strong", "weak"),
        ]
        issues = [
            f"Both '{a}' and '{b}' used"
            for a, b in contradictions
            if a in content and b in content
        ]
        score = max(0.0, 2.0 - len(issues) * 0.5)
        passed = score >= 1.0
        feedback = (
            "No contradictory statements detected"
            if not issues
            else "; ".join(issues)
        )
        return RubricEvaluator._build_evaluation(
            "consistency", score, passed, feedback, issues
        )

    @staticmethod
    def evaluate_temporal_validity(
        analysis: FinancialAnalysis,
    ) -> RubricEvaluation:
        recency = datetime.utcnow() - analysis.analysis_date
        score = 2.0 if recency <= timedelta(days=180) else 1.0
        if "202" in analysis.content:
            score += 0.2
        passed = score >= 1.0
        feedback = (
            "Analysis references recent filings"
            if passed
            else "Update analysis with recent filings"
        )
        return RubricEvaluator._build_evaluation(
            "temporal_validity", score, passed, feedback
        )

    @staticmethod
    def evaluate_risk_awareness(analysis: FinancialAnalysis) -> RubricEvaluation:
        risk_count = len(analysis.risks_identified or [])
        risk_keywords = [
            "risk",
            "uncertain",
            "volatility",
            "exposure",
            "threat",
            "challenge",
        ]
        content = analysis.content.lower()
        mentions = sum(1 for kw in risk_keywords if kw in content)
        score = min(2.0, 0.8 + (risk_count * 0.2) + mentions * 0.1)
        passed = score >= 1.0
        evidence = [f"Risks listed: {risk_count}"] if risk_count else []
        feedback = (
            "Risks enumerated with context"
            if passed
            else "Discuss concrete risk factors"
        )
        return RubricEvaluator._build_evaluation(
            "risk_awareness", score, passed, feedback, evidence
        )

    @staticmethod
    def evaluate_clarity_interpretability(
        analysis: FinancialAnalysis,
    ) -> RubricEvaluation:
        sentences = analysis.content.count(".") + analysis.content.count("!") + analysis.content.count("?")
        words = len(analysis.content.split())
        avg_sentence_len = words / max(1, sentences)
        if avg_sentence_len <= 20:
            score = 2.0
        elif avg_sentence_len <= 30:
            score = 1.5
        else:
            score = 1.0
        jargon_penalty = len(re.findall(r"\b(?:EBITDA|derivative|notional)\b", analysis.content)) * 0.1
        score -= jargon_penalty
        passed = score >= 1.0
        feedback = (
            f"Average sentence length {avg_sentence_len:.1f}"
            if passed
            else "Shorten sentences and explain jargon"
        )
        return RubricEvaluator._build_evaluation(
            "clarity_interpretability", score, passed, feedback
        )

    @staticmethod
    def evaluate_uncertainty_handling(
        analysis: FinancialAnalysis,
    ) -> RubricEvaluation:
        content = analysis.content.lower()
        uncertainty_markers = [
            "may",
            "might",
            "could",
            "possibly",
            "approximately",
            "estimated",
            "assuming",
        ]
        mentions = sum(1 for marker in uncertainty_markers if marker in content)
        overconfident_phrases = ["will definitely", "guaranteed", "certain"]
        penalties = sum(1 for phrase in overconfident_phrases if phrase in content)
        score = 1.0 + min(1.0, mentions * 0.2) - penalties * 0.3
        passed = score >= 1.0
        feedback = (
            "Appropriately communicates uncertainty"
            if penalties == 0
            else "Avoid absolute statements"
        )
        return RubricEvaluator._build_evaluation(
            "uncertainty_handling", score, passed, feedback
        )

    @staticmethod
    def evaluate_actionability(analysis: FinancialAnalysis) -> RubricEvaluation:
        action_verbs = ["recommend", "should", "prioritize", "reduce", "increase"]
        conclusions = analysis.conclusions or []
        verb_mentions = sum(
            1
            for verb in action_verbs
            if verb in analysis.content.lower()
        )
        score = 0.8 + min(1.0, (len(conclusions) * 0.2) + verb_mentions * 0.1)
        passed = score >= 1.0
        feedback = (
            "Provides actionable recommendations"
            if passed
            else "Add concrete next steps"
        )
        evidence = conclusions[:2]
        return RubricEvaluator._build_evaluation(
            "actionability", score, passed, feedback, evidence
        )

    @staticmethod
    def evaluate_all_rubrics(
        analysis: FinancialAnalysis,
        source_documents: Optional[List[SECDocument]] = None,
        expected_values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, RubricEvaluation]:
        return {
            "factual_accuracy": RubricEvaluator.evaluate_factual_accuracy(analysis, expected_values),
            "source_fidelity": RubricEvaluator.evaluate_source_fidelity(analysis, source_documents),
            "regulatory_compliance": RubricEvaluator.evaluate_regulatory_compliance(analysis),
            "financial_reasoning": RubricEvaluator.evaluate_financial_reasoning(analysis),
            "materiality_relevance": RubricEvaluator.evaluate_materiality_relevance(analysis),
            "completeness": RubricEvaluator.evaluate_completeness(analysis),
            "consistency": RubricEvaluator.evaluate_consistency(analysis),
            "temporal_validity": RubricEvaluator.evaluate_temporal_validity(analysis),
            "risk_awareness": RubricEvaluator.evaluate_risk_awareness(analysis),
            "clarity_interpretability": RubricEvaluator.evaluate_clarity_interpretability(analysis),
            "uncertainty_handling": RubricEvaluator.evaluate_uncertainty_handling(analysis),
            "actionability": RubricEvaluator.evaluate_actionability(analysis),
        }
