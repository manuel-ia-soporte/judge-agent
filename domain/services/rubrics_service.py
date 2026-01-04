# domain/services/rubrics_service.py
"""Rubrics domain service."""
from typing import List, Dict
from dataclasses import dataclass
from domain.models.evaluation import RubricEvaluation
from domain.models.finance import FinancialAnalysis, SECDocument
import re


@dataclass
class RubricEvaluator:
    """Domain service for evaluating specific rubrics"""

    @staticmethod
    def evaluate_factual_accuracy(
            analysis: FinancialAnalysis,
            expected_values: Dict[str, float]
    ) -> RubricEvaluation:
        """Evaluate factual accuracy rubric"""
        score = 2.0
        feedback = []
        evidence = []

        # Check numerical accuracy
        for metric, expected_value in expected_values.items():
            # Extract metric from analysis (simplified)
            pattern = rf"{metric}.*?(\d+(?:\.\d+)?)"
            matches = re.findall(pattern, analysis.content, re.IGNORECASE)

            if matches:
                try:
                    actual_value = float(matches[0])
                    if abs(actual_value - expected_value) / expected_value < 0.01:  # 1% tolerance
                        evidence.append(f"{metric} matches: {actual_value}")
                    else:
                        score -= 0.5
                        feedback.append(f"{metric} mismatch: expected {expected_value}, found {actual_value}")
                except ValueError:
                    score -= 0.5
                    feedback.append(f"Could not parse {metric} value")

        return RubricEvaluation(
            rubric_name="factual_accuracy",
            score=max(0, score),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "All facts accurate",
            evidence=evidence
        )

    @staticmethod
    def evaluate_source_fidelity(
            analysis: FinancialAnalysis,
            sec_documents: List[SECDocument]
    ) -> RubricEvaluation:
        """Evaluate source fidelity rubric"""
        score = 2.0
        feedback = []
        evidence = []

        # Check citations
        cited_sources = analysis.get_cited_sources()
        if not cited_sources:
            score = 0.0
            feedback.append("No sources cited")
        else:
            evidence.append(f"Cited {len(cited_sources)} sources")

        # Check for unsupported claims
        unsupported_keywords = ["definitely", "will", "guarantee", "certain"]
        for keyword in unsupported_keywords:
            if keyword in analysis.content.lower():
                score -= 0.3
                feedback.append(f"Unsupported claim with '{keyword}'")

        return RubricEvaluation(
            rubric_name="source_fidelity",
            score=max(0, score),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Well-grounded in sources",
            evidence=evidence
        )

    @staticmethod
    def evaluate_regulatory_compliance(
            analysis: FinancialAnalysis
    ) -> RubricEvaluation:
        """Evaluate regulatory compliance rubric"""
        score = 2.0
        feedback = []

        # Check for compliance violations
        violations = [
            (r"buy\s+(?:this\s+)?stock", "Investment advice"),
            (r"you should", "Directive language"),
            (r"guarantee|promise", "Guarantees"),
            (r"will happen", "Predictions")
        ]

        content_lower = analysis.content.lower()
        for pattern, violation_type in violations:
            if re.search(pattern, content_lower):
                score -= 0.5
                feedback.append(f"Potential {violation_type} violation")

        # Check neutral language
        if len(feedback) == 0:
            feedback.append("Compliant with SEC regulations")

        return RubricEvaluation(
            rubric_name="regulatory_compliance",
            score=max(0, score),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback),
            evidence=[]
        )

    @staticmethod
    def evaluate_financial_reasoning(
            analysis: FinancialAnalysis
    ) -> RubricEvaluation:
        """Evaluate financial reasoning quality"""
        score = 1.0  # Start with neutral score
        feedback = []
        evidence = []

        # Check for logical fallacies
        fallacies = [
            (r"debt.*increased.*therefore.*improved", "Debt improvement fallacy"),
            (r"revenue.*down.*but.*healthy", "Revenue health fallacy")
        ]

        for pattern, fallacy in fallacies:
            if re.search(pattern, analysis.content, re.IGNORECASE):
                score -= 0.5
                feedback.append(f"Logical fallacy: {fallacy}")

        # Check for proper ratio interpretation
        ratios = ["debt/equity", "current ratio", "profit margin", "roa", "roe"]
        ratio_count = sum(1 for ratio in ratios if ratio in analysis.content.lower())

        if ratio_count >= 2:
            score += 0.5
            evidence.append(f"Uses {ratio_count} financial ratios")

        return RubricEvaluation(
            rubric_name="financial_reasoning",
            score=max(0, min(2, score)),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Sound financial reasoning",
            evidence=evidence
        )

    @staticmethod
    def evaluate_materiality(
            analysis: FinancialAnalysis,
            material_items: List[str]
    ) -> RubricEvaluation:
        """Evaluate materiality and relevance"""
        score = 1.0
        feedback = []

        # Check if material items are addressed
        material_covered = 0
        for item in material_items:
            if item.lower() in analysis.content.lower():
                material_covered += 1
                feedback.append(f"Covers {item}")

        coverage_rate = material_covered / len(material_items) if material_items else 1
        score = 1.0 + coverage_rate  # Scale to 1-2

        return RubricEvaluation(
            rubric_name="materiality_relevance",
            score=min(2, score),
            is_passed=score >= 1.0,
            feedback=f"Covers {material_covered}/{len(material_items)} material items",
            evidence=feedback
        )