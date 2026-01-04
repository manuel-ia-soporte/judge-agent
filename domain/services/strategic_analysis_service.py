# domain/services/strategic_analysis_service.py
"""Strategic analysis domain service"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..models.entities import SECDocument
from ..models.value_objects import RiskFactor


@dataclass
class StrategicPosition:
    """Strategic position analysis"""
    competitive_advantage: Optional[str] = None
    market_position: Optional[str] = None
    growth_strategy: Optional[str] = None
    innovation_capability: Optional[float] = None
    strategic_agility: Optional[float] = None


class StrategicAnalysisService:
    """Domain service for strategic analysis"""

    @staticmethod
    def analyze_strategic_position(
            documents: List[SECDocument]
    ) -> StrategicPosition:
        """Analyze company's strategic position"""
        position = StrategicPosition()

        for doc in documents:
            # Analyze business description
            if "Item 1" in doc.items:
                business_desc = doc.items["Item 1"]
                position = StrategicAnalysisService._analyze_business_description(
                    business_desc, position
                )

            # Analyze MD&A for strategy discussion
            if "Item 7" in doc.items:
                mdna_text = doc.items["Item 7"]
                position = StrategicAnalysisService._analyze_strategy_in_mdna(
                    mdna_text, position
                )

        return position

    @staticmethod
    def assess_competitive_advantage(
            documents: List[SECDocument]
    ) -> Dict[str, Any]:
        """Assess company's competitive advantage"""
        assessment = {
            "score": 0.0,
            "sources": [],
            "sustainability": "unknown",
            "moat_strength": "unknown"
        }

        advantage_indicators = {
            "brand": 0,
            "patent": 0,
            "technology": 0,
            "scale": 0,
            "network": 0,
            "switching_costs": 0,
            "regulation": 0
        }

        for doc in documents:
            text = doc.raw_text.lower() if doc.raw_text else ""

            # Look for competitive advantage indicators
            if "brand" in text or "trademark" in text:
                advantage_indicators["brand"] += 1

            if "patent" in text or "intellectual property" in text:
                advantage_indicators["patent"] += 1

            if "technology" in text or "proprietary" in text or "unique" in text:
                advantage_indicators["technology"] += 1

            if "economies of scale" in text or "scale" in text:
                advantage_indicators["scale"] += 1

            if "network effect" in text or "ecosystem" in text:
                advantage_indicators["network"] += 1

            if "switching cost" in text or "lock-in" in text:
                advantage_indicators["switching_costs"] += 1

            if "regulated" in text or "license" in text or "permit" in text:
                advantage_indicators["regulation"] += 1

        # Calculate score
        total_indicators = sum(1 for v in advantage_indicators.values() if v > 0)
        assessment["score"] = total_indicators / len(advantage_indicators)

        # Identify sources
        for source, count in advantage_indicators.items():
            if count > 0:
                assessment["sources"].append(source)

        # Assess sustainability
        if assessment["score"] >= 0.7:
            assessment["sustainability"] = "high"
            assessment["moat_strength"] = "wide"
        elif assessment["score"] >= 0.4:
            assessment["sustainability"] = "medium"
            assessment["moat_strength"] = "narrow"
        else:
            assessment["sustainability"] = "low"
            assessment["moat_strength"] = "none"

        return assessment

    @staticmethod
    def analyze_growth_strategies(
            documents: List[SECDocument]
    ) -> Dict[str, Any]:
        """Analyze growth strategies mentioned in filings"""
        strategies = {
            "organic": {"mentions": 0, "details": []},
            "acquisition": {"mentions": 0, "details": []},
            "expansion": {"mentions": 0, "details": []},
            "innovation": {"mentions": 0, "details": []},
            "partnership": {"mentions": 0, "details": []}
        }

        for doc in documents:
            # Check MD&A for growth discussion
            if "Item 7" in doc.items:
                mdna_text = doc.items["Item 7"].lower()

                if "organic growth" in mdna_text or "internal growth" in mdna_text:
                    strategies["organic"]["mentions"] += 1

                if "acquisition" in mdna_text or "merge" in mdna_text:
                    strategies["acquisition"]["mentions"] += 1

                if "expansion" in mdna_text or "new market" in mdna_text:
                    strategies["expansion"]["mentions"] += 1

                if "innovation" in mdna_text or "r&d" in mdna_text:
                    strategies["innovation"]["mentions"] += 1

                if "partnership" in mdna_text or "joint venture" in mdna_text:
                    strategies["partnership"]["mentions"] += 1

            # Check for acquisitions in recent events
            if doc.filing_type == "8-K":
                text = doc.raw_text.lower() if doc.raw_text else ""
                if "acquire" in text or "acquisition" in text:
                    strategies["acquisition"]["details"].append({
                        "date": doc.filing_date,
                        "description": "Acquisition mentioned in 8-K"
                    })

        # Calculate primary strategy
        total_mentions = sum(s["mentions"] for s in strategies.values())
        if total_mentions > 0:
            primary_strategy = max(
                strategies.items(),
                key=lambda x: x[1]["mentions"]
            )[0]
            strategies["primary_strategy"] = primary_strategy

        return strategies

    @staticmethod
    def identify_strategic_risks(
            documents: List[SECDocument]
    ) -> List[RiskFactor]:
        """Identify strategic risks from filings"""
        risks = []

        for doc in documents:
            # Look in the risk factors section
            if "Item 1A" in doc.items:
                risk_text = doc.items["Item 1A"]
                strategic_risks = (
                    StrategicAnalysisService._extract_strategic_risks(risk_text)
                )
                risks.extend(strategic_risks)

            # Look in MD&A for strategic challenges
            if "Item 7" in doc.items:
                mdna_text = doc.items["Item 7"]
                mdna_risks = (
                    StrategicAnalysisService._extract_mdna_strategic_risks(mdna_text)
                )
                risks.extend(mdna_risks)

        return risks[:10]  # Return top 10

    @staticmethod
    def assess_innovation_capability(
            documents: List[SECDocument]
    ) -> Dict[str, Any]:
        """Assess company's innovation capability"""
        assessment = {
            "score": 0.0,
            "r&d_intensity": 0.0,
            "patent_count": 0,
            "innovation_mentions": 0,
            "new_product_mentions": 0
        }

        for doc in documents:
            text = doc.raw_text.lower() if doc.raw_text else ""

            # Count innovation-related terms
            innovation_keywords = [
                "innovation", "innovative", "research and development",
                "r&d", "new product", "technology", "patent", "intellectual property"
            ]

            for keyword in innovation_keywords:
                if keyword in text:
                    assessment["innovation_mentions"] += 1

            # Look for R&D expenditure
            if "research and development" in text:
                # Try to extract R&D amount
                import re
                rd_match = re.search(
                    r'research.*?\$?(\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE
                )
                if rd_match:
                    # This would need more sophisticated parsing
                    pass

        # Calculate score based on mentions
        if assessment["innovation_mentions"] > 10:
            assessment["score"] = 0.8
        elif assessment["innovation_mentions"] > 5:
            assessment["score"] = 0.6
        elif assessment["innovation_mentions"] > 2:
            assessment["score"] = 0.4
        else:
            assessment["score"] = 0.2

        return assessment

    @staticmethod
    def analyze_market_position(
            documents: List[SECDocument]
    ) -> Dict[str, Any]:
        """Analyze company's market position"""
        analysis = {
            "market_share_mentions": 0,
            "competition_mentions": 0,
            "market_position": "unknown",
            "competitive_pressures": []
        }

        for doc in documents:
            text = doc.raw_text.lower() if doc.raw_text else ""

            # Look for market share discussions
            if "market share" in text:
                analysis["market_share_mentions"] += 1

            # Look for competition discussions
            if "competition" in text or "competitive" in text:
                analysis["competition_mentions"] += 1

                # Extract competitive pressures
                lines = text.split('.')
                for line in lines:
                    if "competition" in line or "competitive" in line:
                        line_clean = line.strip()[:100]
                        if line_clean and len(line_clean) > 20:
                            analysis["competitive_pressures"].append(line_clean)

        # Determine market position based on mentions
        if analysis["market_share_mentions"] >= 3:
            analysis["market_position"] = "leader"
        elif analysis["competition_mentions"] >= 5:
            analysis["market_position"] = "challenger"
        elif analysis["competition_mentions"] >= 2:
            analysis["market_position"] = "follower"
        else:
            analysis["market_position"] = "niche"

        return analysis

    # Private helper methods
    @staticmethod
    def _analyze_business_description(
            business_desc: str,
            position: StrategicPosition
    ) -> StrategicPosition:
        """Analyze business description for strategic position"""
        desc_lower = business_desc.lower()

        # Look for competitive advantage mentions
        advantage_keywords = [
            "competitive advantage", "unique", "differentiated",
            "proprietary", "leading", "dominant"
        ]

        for keyword in advantage_keywords:
            if keyword in desc_lower:
                position.competitive_advantage = "mentioned"
                break

        # Look for market position
        if "leader" in desc_lower or "leading" in desc_lower:
            position.market_position = "leader"
        elif "niche" in desc_lower:
            position.market_position = "niche"

        return position

    @staticmethod
    def _analyze_strategy_in_mdna(
            mdna_text: str,
            position: StrategicPosition
    ) -> StrategicPosition:
        """Analyze MD&A for strategy discussion"""
        mdna_lower = mdna_text.lower()

        # Look for growth strategy mentions
        growth_keywords = [
            "growth strategy", "expansion", "investment",
            "acquisition", "market development"
        ]

        for keyword in growth_keywords:
            if keyword in mdna_lower:
                position.growth_strategy = "discussed"
                break

        return position

    @staticmethod
    def _extract_strategic_risks(risk_text: str) -> List[RiskFactor]:
        """Extract strategic risks from risk factor text"""
        from ..models.value_objects import RiskFactor
        from ..models.enums import SeverityLevel

        risks = []
        strategic_keywords = [
            "strategic", "competition", "market share", "obsolescence",
            "disruption", "innovation", "technology", "acquisition",
            "merger", "partnership", "joint venture"
        ]

        lines = risk_text.split('.')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in strategic_keywords):
                # Determine severity
                severity = SeverityLevel.MEDIUM
                if "significant" in line_lower or "major" in line_lower:
                    severity = SeverityLevel.HIGH
                elif "potential" in line_lower or "may" in line_lower:
                    severity = SeverityLevel.LOW

                risk = RiskFactor(
                    description=line.strip()[:200],
                    category="strategic",
                    severity=severity,
                    probability=0.4,
                    impact="strategic"
                )
                risks.append(risk)

        return risks

    @staticmethod
    def _extract_mdna_strategic_risks(mdna_text: str) -> List[RiskFactor]:
        """Extract strategic risks from MD&A"""
        from ..models.value_objects import RiskFactor
        from ..models.enums import SeverityLevel

        risks = []
        challenge_keywords = [
            "challenge", "risk", "uncertainty", "threat",
            "headwind", "pressure", "volatility"
        ]

        strategic_contexts = [
            "competition", "market", "strategy", "growth",
            "acquisition", "partnership", "innovation"
        ]

        lines = mdna_text.split('.')
        for line in lines:
            line_lower = line.lower()
            has_challenge = any(keyword in line_lower for keyword in challenge_keywords)
            has_strategic_context = any(ctx in line_lower for ctx in strategic_contexts)

            if has_challenge and has_strategic_context:
                risk = RiskFactor(
                    description=line.strip()[:200],
                    category="strategic",
                    severity=SeverityLevel.MEDIUM,
                    probability=0.3,
                    impact="strategic"
                )
                risks.append(risk)

        return risks