# agents/judge_agent/rubrics_evaluator.py
from typing import Dict, List, Optional
import re
from dataclasses import dataclass
from domain.models.finance import FinancialAnalysis, SECDocument
from domain.models.evaluation import RubricEvaluation


@dataclass
class RubricEvaluator:
    """Comprehensive rubric evaluator for financial analysis"""

    @staticmethod
    def evaluate_all_rubrics(
            analysis: FinancialAnalysis,
            sec_documents: List[SECDocument],
            expected_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, RubricEvaluation]:
        """Evaluate all 12 rubrics"""
        evaluations = {}

        # 1. Factual Accuracy
        evaluations["factual_accuracy"] = RubricEvaluator.evaluate_factual_accuracy(
            analysis, expected_values or {}
        )

        # 2. Source Fidelity
        evaluations["source_fidelity"] = RubricEvaluator.evaluate_source_fidelity(
            analysis, sec_documents
        )

        # 3. Regulatory Compliance
        evaluations["regulatory_compliance"] = RubricEvaluator.evaluate_regulatory_compliance(
            analysis
        )

        # 4. Financial Reasoning
        evaluations["financial_reasoning"] = RubricEvaluator.evaluate_financial_reasoning(
            analysis
        )

        # 5. Materiality & Relevance
        material_items = RubricEvaluator._extract_material_items(sec_documents)
        evaluations["materiality_relevance"] = RubricEvaluator.evaluate_materiality(
            analysis, material_items
        )

        # 6. Completeness
        evaluations["completeness"] = RubricEvaluator.evaluate_completeness(analysis)

        # 7. Consistency
        evaluations["consistency"] = RubricEvaluator.evaluate_consistency(analysis)

        # 8. Temporal Validity
        evaluations["temporal_validity"] = RubricEvaluator.evaluate_temporal_validity(
            analysis, sec_documents
        )

        # 9. Risk Awareness
        evaluations["risk_awareness"] = RubricEvaluator.evaluate_risk_awareness(
            analysis, sec_documents
        )

        # 10. Clarity & Interpretability
        evaluations["clarity_interpretability"] = RubricEvaluator.evaluate_clarity(
            analysis
        )

        # 11. Uncertainty Handling
        evaluations["uncertainty_handling"] = RubricEvaluator.evaluate_uncertainty(
            analysis
        )

        # 12. Actionability
        evaluations["actionability"] = RubricEvaluator.evaluate_actionability(analysis)

        return evaluations

    @staticmethod
    def evaluate_completeness(analysis: FinancialAnalysis) -> RubricEvaluation:
        """Evaluate completeness rubric"""
        score = 1.0
        feedback = []
        evidence = []

        # Check if analysis has key components
        components = {
            "introduction": bool(analysis.content[:100].strip()),
            "metrics": len(analysis.metrics_used) > 0,
            "conclusions": len(analysis.conclusions) > 0,
            "risks": len(analysis.risks_identified) > 0,
            "assumptions": len(analysis.assumptions) > 0
        }

        present_components = sum(1 for present in components.values() if present)
        component_score = present_components / len(components)

        score = 0.5 + component_score * 1.5  # Scale to 0.5-2.0

        for component, present in components.items():
            if present:
                evidence.append(f"Includes {component}")
            else:
                feedback.append(f"Missing {component}")

        return RubricEvaluation(
            rubric_name="completeness",
            score=min(2.0, max(0.0, score)),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Complete analysis",
            evidence=evidence
        )

    @staticmethod
    def evaluate_consistency(analysis: FinancialAnalysis) -> RubricEvaluation:
        """Evaluate consistency rubric"""
        score = 2.0
        feedback = []

        # Check for internal contradictions
        contradictions = RubricEvaluator._find_contradictions(analysis.content)

        for contradiction in contradictions:
            score -= 0.5
            feedback.append(f"Contradiction: {contradiction}")

        # Check numeric consistency
        numbers = re.findall(r'\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|M|B)?', analysis.content)
        if numbers:
            # Check if same numbers are used consistently
            pass

        return RubricEvaluation(
            rubric_name="consistency",
            score=max(0, score),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Internally consistent",
            evidence=[]
        )

    @staticmethod
    def evaluate_temporal_validity(
            analysis: FinancialAnalysis,
            sec_documents: List[SECDocument]
    ) -> RubricEvaluation:
        """Evaluate temporal validity rubric"""
        score = 2.0
        feedback = []
        evidence = []

        if not sec_documents:
            return RubricEvaluation(
                rubric_name="temporal_validity",
                score=1.0,
                is_passed=True,
                feedback="No documents to validate against",
                evidence=[]
            )

        # Get the latest document date
        latest_doc = max(sec_documents, key=lambda d: d.filing_date)
        latest_date = latest_doc.filing_date

        # Check for outdated references
        outdated_terms = [
            "last year", "previous year", "prior year",
            "last quarter", "previous quarter"
        ]

        for term in outdated_terms:
            if term in analysis.content.lower():
                evidence.append(f"References {term}")

        # Check if analysis mentions specific timeframes
        timeframe_patterns = [
            r"in \d{4}", r"Q\d \d{4}", r"\d{4} annual",
            r"as of \w+ \d{1,2}, \d{4}"
        ]

        timeframe_matches = 0
        for pattern in timeframe_patterns:
            if re.search(pattern, analysis.content, re.IGNORECASE):
                timeframe_matches += 1

        if timeframe_matches == 0:
            score -= 0.5
            feedback.append("No specific timeframes mentioned")

        return RubricEvaluation(
            rubric_name="temporal_validity",
            score=score,
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Temporally valid",
            evidence=evidence
        )

    @staticmethod
    def evaluate_risk_awareness(
            analysis: FinancialAnalysis,
            sec_documents: List[SECDocument]
    ) -> RubricEvaluation:
        """Evaluate risk awareness rubric"""
        score = 1.0
        feedback = []
        evidence = []

        # Count risk mentions
        risk_keywords = ["risk", "uncertainty", "volatility", "exposure", "liability"]
        risk_mentions = sum(
            1 for keyword in risk_keywords
            if keyword in analysis.content.lower()
        )

        # Check if specific risks are mentioned
        specific_risks = [
            "market risk", "credit risk", "liquidity risk",
            "operational risk", "regulatory risk", "cybersecurity risk"
        ]

        mentioned_risks = [
            risk for risk in specific_risks
            if risk in analysis.content.lower()
        ]

        # Score based on risk coverage
        if risk_mentions >= 3:
            score += 0.5
            evidence.append(f"Mentions {risk_mentions} risk-related terms")

        if mentioned_risks:
            score += len(mentioned_risks) * 0.2
            evidence.append(f"Identifies specific risks: {', '.join(mentioned_risks)}")

        # Check if mitigation is discussed
        if "mitigat" in analysis.content.lower():
            score += 0.3
            evidence.append("Discusses risk mitigation")

        # Check against actual risks from documents
        if sec_documents:
            actual_risks = RubricEvaluator._extract_actual_risks(sec_documents)
            covered_risks = [
                risk for risk in actual_risks[:5]  # Check top 5
                if any(risk_term in analysis.content.lower() for risk_term in risk.split()[:3])
            ]

            if covered_risks:
                coverage_rate = len(covered_risks) / min(5, len(actual_risks))
                score += coverage_rate * 0.5
                evidence.append(f"Covers {len(covered_risks)} actual risks from filings")

        return RubricEvaluation(
            rubric_name="risk_awareness",
            score=min(2.0, score),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Aware of risks",
            evidence=evidence
        )

    @staticmethod
    def evaluate_clarity(analysis: FinancialAnalysis) -> RubricEvaluation:
        """Evaluate clarity and interpretability rubric"""
        score = 1.0
        feedback = []
        evidence = []

        # Calculate readability metrics
        sentences = re.split(r'[.!?]+', analysis.content)
        words = analysis.content.split()

        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)

            if avg_sentence_length < 25:
                score += 0.3
                evidence.append("Concise sentences")
            elif avg_sentence_length > 40:
                score -= 0.2
                feedback.append("Sentences too long")

        # Check for jargon
        jargon_terms = ["EBITDA", "amortization", "accrual", "derivative", "hedging"]
        jargon_count = sum(
            1 for term in jargon_terms
            if term.lower() in analysis.content.lower()
        )

        if jargon_count > 0:
            # Check if jargon is explained
            explanations = sum(
                1 for term in jargon_terms
                if f"{term} (" in analysis.content or f"{term.lower()} (" in analysis.content
            )

            if explanations == jargon_count:
                score += 0.3
                evidence.append("Jargon properly explained")
            else:
                score -= 0.2
                feedback.append("Unexplained jargon")

        # Check structure
        structure_indicators = ["first", "second", "third", "in conclusion", "summary"]
        structure_count = sum(
            1 for indicator in structure_indicators
            if indicator in analysis.content.lower()
        )

        if structure_count >= 2:
            score += 0.3
            evidence.append("Well-structured analysis")

        return RubricEvaluation(
            rubric_name="clarity_interpretability",
            score=min(2.0, max(0.0, score)),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Clear and interpretable",
            evidence=evidence
        )

    @staticmethod
    def evaluate_uncertainty(analysis: FinancialAnalysis) -> RubricEvaluation:
        """Evaluate uncertainty handling rubric"""
        score = 1.0
        feedback = []
        evidence = []

        # Check for uncertainty qualifiers
        qualifiers = [
            "may", "could", "might", "possible", "potential",
            "estimate", "approximately", "roughly", "about"
        ]

        qualifier_count = sum(
            1 for qualifier in qualifiers
            if qualifier in analysis.content.lower()
        )

        if qualifier_count >= 2:
            score += 0.5
            evidence.append(f"Uses {qualifier_count} uncertainty qualifiers")

        # Check for explicit assumptions
        assumption_keywords = ["assume", "assuming", "presume", "presuming"]
        assumption_count = sum(
            1 for keyword in assumption_keywords
            if keyword in analysis.content.lower()
        )

        if assumption_count > 0:
            score += 0.3
            evidence.append("Explicitly states assumptions")

        # Check for overconfident statements
        overconfident_patterns = [
            r"will definitely", r"certain to", r"guarantee",
            r"no doubt", r"absolutely"
        ]

        overconfident_count = 0
        for pattern in overconfident_patterns:
            if re.search(pattern, analysis.content, re.IGNORECASE):
                overconfident_count += 1

        if overconfident_count > 0:
            score -= overconfident_count * 0.3
            feedback.append(f"Contains {overconfident_count} overconfident statements")

        # Check for data limitations
        limitation_keywords = ["limitation", "constraint", "caveat", "warning"]
        if any(keyword in analysis.content.lower() for keyword in limitation_keywords):
            score += 0.2
            evidence.append("Acknowledges limitations")

        return RubricEvaluation(
            rubric_name="uncertainty_handling",
            score=min(2.0, max(0.0, score)),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Appropriately handles uncertainty",
            evidence=evidence
        )

    @staticmethod
    def evaluate_actionability(analysis: FinancialAnalysis) -> RubricEvaluation:
        """Evaluate actionability rubric"""
        score = 1.0
        feedback = []
        evidence = []

        # Check for actionable insights
        actionable_phrases = [
            "suggest", "recommend", "consider", "monitor",
            "review", "evaluate", "assess", "investigate"
        ]

        actionable_count = sum(
            1 for phrase in actionable_phrases
            if phrase in analysis.content.lower()
        )

        if actionable_count >= 2:
            score += 0.5
            evidence.append(f"Contains {actionable_count} actionable suggestions")

        # Check for specific next steps
        next_step_patterns = [
            r"next steps?", r"going forward", r"in the future",
            r"should consider", r"need to"
        ]

        next_step_count = 0
        for pattern in next_step_patterns:
            if re.search(pattern, analysis.content, re.IGNORECASE):
                next_step_count += 1

        if next_step_count > 0:
            score += 0.3
            evidence.append(f"Suggests {next_step_count} next steps")

        # Check for compliance-safe language
        compliance_violations = [
            r"buy now", r"sell immediately", r"invest in",
            r"guaranteed return", r"definitely will"
        ]

        violations = 0
        for pattern in compliance_violations:
            if re.search(pattern, analysis.content, re.IGNORECASE):
                violations += 1

        if violations > 0:
            score -= violations * 0.5
            feedback.append(f"Contains {violations} compliance violations")

        return RubricEvaluation(
            rubric_name="actionability",
            score=min(2.0, max(0.0, score)),
            is_passed=score >= 1.0,
            feedback="; ".join(feedback) if feedback else "Provides actionable insights",
            evidence=evidence
        )

    @staticmethod
    def _find_contradictions(text: str) -> List[str]:
        """Find contradictions in text"""
        contradictions = []

        # Look for contradictory statements
        contradiction_pairs = [
            (r"increased", r"decreased"),
            (r"improved", r"worsened"),
            (r"profitable", r"losing money"),
            (r"growing", r"declining")
        ]

        for pos, neg in contradiction_pairs:
            if re.search(pos, text, re.IGNORECASE) and re.search(neg, text, re.IGNORECASE):
                # Check if they're close together (might be comparing different things)
                contradictions.append(f"Both {pos} and {neg} mentioned")

        return contradictions

    @staticmethod
    def _extract_material_items(sec_documents: List[SECDocument]) -> List[str]:
        """Extract material items from SEC documents"""
        material_items = [
            "revenue", "net income", "assets", "liabilities",
            "cash flow", "debt", "equity", "risk factors",
            "management discussion", "legal proceedings"
        ]

        # Add specific items from documents
        for doc in sec_documents:
            if "Item 1A" in doc.items:
                material_items.append("risk factors detailed")
            if "Item 7" in doc.items:
                material_items.append("management analysis")
            if "Item 8" in doc.items:
                material_items.append("financial statements")

        return list(set(material_items))

    @staticmethod
    def _extract_actual_risks(sec_documents: List[SECDocument]) -> List[str]:
        """Extract actual risks from SEC documents"""
        risks = []

        for doc in sec_documents:
            if "Item 1A" in doc.items:
                # Parse risk factors
                risk_text = doc.items["Item 1A"]
                # Simplified extraction
                risk_sentences = re.split(r'[.!?]+', risk_text)
                risks.extend([s.strip()[:100] for s in risk_sentences[:10] if s.strip()])

        return risks[:20]  # Limit to 20 risks