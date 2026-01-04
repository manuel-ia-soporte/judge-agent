# domain/services/risk_assessment_service.py
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from ..models.entities import SECDocument
from ..models.value_objects import RiskFactor


@dataclass
class RiskAssessmentService:
    """Domain Service for Risk Assessment"""

    def extract_risk_factors(self, documents: List[SECDocument]) -> List[RiskFactor]:
        """Extract risk factors from SEC documents"""
        risk_factors = []

        for doc in documents:
            if "Item 1A" in doc.items:
                risk_text = doc.items["Item 1A"]
                risk_items = self._parse_risk_items(risk_text)

                for risk_item in risk_items:
                    risk_factor = self._analyze_risk_item(risk_item)
                    if risk_factor:
                        risk_factors.append(risk_factor)

        return self._deduplicate_risk_factors(risk_factors)

    def categorize_risks(self, risk_factors: List[RiskFactor]) -> Dict[str, List[RiskFactor]]:
        """Categorize risks by type"""
        categories = {
            "market": [],
            "financial": [],
            "operational": [],
            "regulatory": [],
            "strategic": [],
            "reputational": []
        }

        category_keywords = {
            "market": ["market", "competition", "demand", "price", "economic"],
            "financial": ["financial", "liquidity", "debt", "credit", "interest"],
            "operational": ["operational", "supply chain", "production", "quality"],
            "regulatory": ["regulatory", "compliance", "legal", "government"],
            "strategic": ["strategic", "acquisition", "expansion", "innovation"],
            "reputational": ["reputational", "brand", "public relations", "image"]
        }

        for risk in risk_factors:
            risk_lower = risk.description.lower()
            for category, keywords in category_keywords.items():
                if any(keyword in risk_lower for keyword in keywords):
                    categories[category].append(risk)
                    break

        return categories

    def assess_overall_risk(self, risk_factors: List[RiskFactor]) -> Tuple[str, float]:
        """Assess overall risk level and score"""
        if not risk_factors:
            return "unknown", 0.0

        total_risk_score = sum(r.risk_score() for r in risk_factors)
        avg_risk_score = total_risk_score / len(risk_factors)

        if avg_risk_score > 0.7:
            return "high", avg_risk_score
        elif avg_risk_score > 0.4:
            return "medium", avg_risk_score
        else:
            return "low", avg_risk_score

    def identify_mitigations(self, risk_factors: List[RiskFactor]) -> List[str]:
        """Identify risk mitigations"""
        mitigations = []

        for risk in risk_factors:
            if risk.mitigation:
                mitigations.append(risk.mitigation)

        return list(set(mitigations))[:5]  # Deduplicate and limit

    # Private helper methods
    @staticmethod
    def _parse_risk_items(risk_text: str) -> List[str]:
        """Parse individual risk items from risk text"""
        risk_items = []

        # Split by numbered items
        pattern = r'(\d+\.\s*)(.*?)(?=\d+\.\s*|\Z)'
        matches = re.findall(pattern, risk_text, re.DOTALL)

        for match in matches:
            risk_text = match[1].strip()
            risk_text = re.sub(r'\s+', ' ', risk_text)
            if len(risk_text) > 20:  # Meaningful length
                risk_items.append(risk_text[:500])  # Limit length

        return risk_items

    def _analyze_risk_item(self, risk_text: str) -> Optional[RiskFactor]:
        """Analyze individual risk item"""
        if not risk_text:
            return None

        # Determine severity
        severity = self._assess_severity(risk_text)

        # Determine category
        category = self._classify_category(risk_text)

        # Extract mitigation if mentioned
        mitigation = self._extract_mitigation(risk_text)

        return RiskFactor(
            description=risk_text[:200],  # Limit length
            category=category,
            severity=severity,
            probability=0.5,  # Default. Could be refined
            mitigation=mitigation
        )

    @staticmethod
    def _assess_severity(risk_text: str) -> str:
        """Assess risk severity"""
        risk_lower = risk_text.lower()

        high_keywords = ["materially", "significantly", "substantially", "severely", "critical"]
        medium_keywords = ["moderately", "could", "may", "might", "potential"]

        if any(keyword in risk_lower for keyword in high_keywords):
            return "high"
        elif any(keyword in risk_lower for keyword in medium_keywords):
            return "medium"
        else:
            return "low"

    @staticmethod
    def _classify_category(risk_text: str) -> str:
        """Classify risk category"""
        risk_lower = risk_text.lower()

        categories = {
            "market": ["market", "competition", "demand", "price"],
            "financial": ["financial", "liquidity", "debt", "credit"],
            "operational": ["operational", "supply chain", "production"],
            "regulatory": ["regulatory", "compliance", "legal"],
            "strategic": ["strategic", "acquisition", "expansion"],
            "reputational": ["reputational", "brand", "public relations"]
        }

        for category, keywords in categories.items():
            if any(keyword in risk_lower for keyword in keywords):
                return category

        return "other"

    @staticmethod
    def _extract_mitigation(risk_text: str) -> Optional[str]:
        """Extract mitigation from risk text"""
        mitigation_keywords = ["mitigate", "manage", "control", "address", "reduce"]

        sentences = risk_text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in mitigation_keywords):
                return sentence.strip()[:100]

        return None

    @staticmethod
    def _deduplicate_risk_factors(risk_factors: List[RiskFactor]) -> List[RiskFactor]:
        """Deduplicate similar risk factors"""
        unique_risks = []
        seen_descriptions = set()

        for risk in risk_factors:
            # Create a normalized description for comparison
            normalized = risk.description.lower().replace('the ', '').replace('a ', '')
            if normalized not in seen_descriptions:
                seen_descriptions.add(normalized)
                unique_risks.append(risk)

        return unique_risks