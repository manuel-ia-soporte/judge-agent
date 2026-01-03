# agents/finance_agent/sec_analyzer.py
from typing import Dict, List, Any, Optional, Tuple
import re
import statistics
from datetime import datetime
import pandas as pd
from infrastructure.sec_edgar.edgar_parser import EdgarParser


class SECAnalyzer:
    """Specialized analyzer for SEC documents"""

    def __init__(self):
        self.parser = EdgarParser()
        self.financial_ratios = {
            "liquidity": ["current_ratio", "quick_ratio", "cash_ratio"],
            "solvency": ["debt_to_equity", "debt_ratio", "equity_ratio"],
            "profitability": ["gross_margin", "operating_margin", "net_margin", "roe", "roa"],
            "efficiency": ["asset_turnover", "inventory_turnover", "receivables_turnover"],
            "market": ["pe_ratio", "pb_ratio", "dividend_yield"]
        }

    async def analyze_filing_structure(self, filing_text: str) -> Dict[str, Any]:
        """Analyze the structure and completeness of SEC filing"""
        structure_analysis = {
            "has_executive_summary": False,
            "has_risk_factors": False,
            "has_mda": False,
            "has_financial_statements": False,
            "has_notes": False,
            "has_exhibits": False,
            "word_count": len(filing_text.split()),
            "section_count": 0
        }

        # Check for key sections
        sections = [
            ("EXECUTIVE SUMMARY", "has_executive_summary"),
            ("RISK FACTORS", "has_risk_factors"),
            ("MANAGEMENT'S DISCUSSION", "has_mda"),
            ("FINANCIAL STATEMENTS", "has_financial_statements"),
            ("NOTES TO FINANCIAL STATEMENTS", "has_notes"),
            ("EXHIBITS", "has_exhibits")
        ]

        for section_name, flag in sections:
            if re.search(section_name, filing_text, re.IGNORECASE):
                structure_analysis[flag] = True
                structure_analysis["section_count"] += 1

        # Check for required items (for 10-K)
        required_items = ["ITEM 1", "ITEM 1A", "ITEM 7", "ITEM 8"]
        present_items = []

        for item in required_items:
            if re.search(f"{item}\\.", filing_text):
                present_items.append(item)

        structure_analysis["required_items_present"] = len(present_items)
        structure_analysis["missing_items"] = [item for item in required_items if item not in present_items]

        # Calculate completeness score
        completeness_score = (
                                     (structure_analysis["section_count"] / len(sections)) * 0.4 +
                                     (structure_analysis["required_items_present"] / len(required_items)) * 0.6
                             ) * 2  # Scale to 0-2

        structure_analysis["completeness_score"] = round(completeness_score, 2)
        structure_analysis["completeness_rating"] = self._get_rating(completeness_score)

        return structure_analysis

    def _get_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 1.8:
            return "excellent"
        elif score >= 1.5:
            return "good"
        elif score >= 1.0:
            return "adequate"
        else:
            return "poor"

    async def extract_financial_metrics(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and categorize financial metrics"""
        metrics = {
            "balance_sheet": {},
            "income_statement": {},
            "cash_flow": {},
            "key_ratios": {}
        }

        # Parse financial statements
        statements = self.parser.parse_financial_statements(filing_data)

        # Categorize metrics
        for statement_type, statement_data in statements.items():
            if statement_type == "balance_sheet":
                metrics["balance_sheet"] = statement_data
            elif statement_type == "income_statement":
                metrics["income_statement"] = statement_data
            elif statement_type == "cash_flow":
                metrics["cash_flow"] = statement_data

        # Calculate ratios if we have enough data
        metrics["key_ratios"] = self._calculate_ratios_from_statements(statements)

        return metrics

    def _calculate_ratios_from_statements(self, statements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial ratios from statements"""
        ratios = {}

        # Extract values
        balance_sheet = statements.get("balance_sheet", {})
        income_statement = statements.get("income_statement", {})

        # Current Ratio = Current Assets / Current Liabilities
        current_assets = balance_sheet.get("AssetsCurrent", {}).get("value", 0)
        current_liabilities = balance_sheet.get("LiabilitiesCurrent", {}).get("value", 0)

        if current_liabilities > 0:
            ratios["current_ratio"] = current_assets / current_liabilities

        # Debt to Equity = Total Debt / Total Equity
        total_debt = balance_sheet.get("LongTermDebt", {}).get("value", 0)
        total_equity = balance_sheet.get("StockholdersEquity", {}).get("value", 0)

        if total_equity > 0:
            ratios["debt_to_equity"] = total_debt / total_equity

        # Profit Margin = Net Income / Revenue
        net_income = income_statement.get("NetIncomeLoss", {}).get("value", 0)
        revenue = income_statement.get("RevenueFromContractWithCustomerExcludingAssessedTax", {}).get("value", 0)

        if revenue > 0:
            ratios["net_margin"] = net_income / revenue

        # ROE = Net Income / Total Equity
        if total_equity > 0:
            ratios["roe"] = net_income / total_equity

        return ratios

    async def analyze_risk_factors(self, filing_text: str) -> Dict[str, Any]:
        """Analyze risk factors in filing"""
        risk_analysis = {
            "risk_count": 0,
            "risk_categories": {},
            "risk_severity": {},
            "mitigation_mentioned": False,
            "risk_trend": "stable"
        }

        # Extract risk section
        risk_section = self.parser._extract_section(filing_text, "item_1a")

        if not risk_section:
            return risk_analysis

        # Count risks
        risk_items = self.parser._parse_risk_items(risk_section)
        risk_analysis["risk_count"] = len(risk_items)

        # Categorize risks
        risk_categories = {
            "market": ["market", "competition", "demand", "price"],
            "financial": ["financial", "liquidity", "debt", "credit"],
            "operational": ["operational", "supply chain", "production", "quality"],
            "regulatory": ["regulatory", "compliance", "legal", "government"],
            "strategic": ["strategic", "acquisition", "expansion", "innovation"],
            "reputational": ["reputational", "brand", "public relations", "image"]
        }

        for risk in risk_items:
            for category, keywords in risk_categories.items():
                if any(keyword in risk.lower() for keyword in keywords):
                    risk_analysis["risk_categories"][category] = risk_analysis["risk_categories"].get(category, 0) + 1
                    break

        # Assess severity
        severity_keywords = {
            "high": ["materially", "significantly", "substantially", "severely", "critical"],
            "medium": ["moderately", "could", "may", "might", "potential"],
            "low": ["minor", "slight", "limited", "manageable"]
        }

        for risk in risk_items:
            for severity, keywords in severity_keywords.items():
                if any(keyword in risk.lower() for keyword in keywords):
                    risk_analysis["risk_severity"][severity] = risk_analysis["risk_severity"].get(severity, 0) + 1
                    break

        # Check for mitigation
        mitigation_keywords = ["mitigate", "manage", "control", "address", "reduce"]
        for keyword in mitigation_keywords:
            if keyword in risk_section.lower():
                risk_analysis["mitigation_mentioned"] = True
                break

        # Determine risk trend
        trend_indicators = {
            "increasing": ["increasing", "growing", "rising", "expanding"],
            "decreasing": ["decreasing", "declining", "reducing", "diminishing"],
            "stable": ["stable", "consistent", "unchanged", "steady"]
        }

        for trend, indicators in trend_indicators.items():
            if any(indicator in risk_section.lower() for indicator in indicators):
                risk_analysis["risk_trend"] = trend
                break

        return risk_analysis

    async def compare_with_peers(
            self,
            company_metrics: Dict[str, Any],
            peer_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare company metrics with peers"""
        comparison = {
            "company_position": {},
            "peer_average": {},
            "peer_median": {},
            "percentile_rank": {},
            "strengths": [],
            "weaknesses": []
        }

        if not peer_metrics:
            return comparison

        # Calculate peer statistics for each metric
        for metric_category, metrics in company_metrics.items():
            if not isinstance(metrics, dict):
                continue

            for metric_name, company_value in metrics.items():
                if isinstance(company_value, dict) and "value" in company_value:
                    company_val = company_value["value"]

                    # Collect peer values
                    peer_values = []
                    for peer in peer_metrics:
                        if (metric_category in peer and
                                metric_name in peer[metric_category] and
                                "value" in peer[metric_category][metric_name]):
                            peer_values.append(peer[metric_category][metric_name]["value"])

                    if peer_values:
                        # Calculate statistics
                        comparison["peer_average"][metric_name] = statistics.mean(peer_values)
                        comparison["peer_median"][metric_name] = statistics.median(peer_values)

                        # Calculate percentile rank
                        better_than = sum(1 for v in peer_values if company_val > v)
                        comparison["percentile_rank"][metric_name] = (better_than / len(peer_values)) * 100

                        # Determine if strength or weakness
                        if comparison["percentile_rank"][metric_name] >= 75:
                            comparison["strengths"].append(metric_name)
                        elif comparison["percentile_rank"][metric_name] <= 25:
                            comparison["weaknesses"].append(metric_name)

        return comparison

    async def generate_investment_considerations(
            self,
            analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Generate investment considerations from analysis"""
        considerations = []

        # Financial considerations
        if "financial_metrics" in analysis_results:
            metrics = analysis_results["financial_metrics"]

            # Check liquidity
            if "current_ratio" in metrics.get("key_ratios", {}):
                cr = metrics["key_ratios"]["current_ratio"]
                if cr < 1:
                    considerations.append("Low current ratio indicates potential liquidity concerns")
                elif cr > 2:
                    considerations.append("Strong liquidity position with high current ratio")

            # Check leverage
            if "debt_to_equity" in metrics.get("key_ratios", {}):
                de = metrics["key_ratios"]["debt_to_equity"]
                if de > 2:
                    considerations.append("High debt-to-equity ratio suggests significant financial leverage")
                elif de < 0.5:
                    considerations.append("Conservative capital structure with low debt levels")

            # Check profitability
            if "net_margin" in metrics.get("key_ratios", {}):
                margin = metrics["key_ratios"]["net_margin"]
                if margin > 0.15:
                    considerations.append("Strong profitability with high net margin")
                elif margin < 0.05:
                    considerations.append("Low net margin may indicate pricing pressure or high costs")

        # Risk considerations
        if "risk_analysis" in analysis_results:
            risk = analysis_results["risk_analysis"]

            if risk.get("risk_count", 0) > 20:
                considerations.append("High number of disclosed risk factors indicates complex risk profile")

            if risk.get("risk_severity", {}).get("high", 0) > 5:
                considerations.append("Multiple high-severity risks identified")

            if not risk.get("mitigation_mentioned", False):
                considerations.append("Limited discussion of risk mitigation strategies")

        # Structure considerations
        if "structure_analysis" in analysis_results:
            structure = analysis_results["structure_analysis"]

            if structure.get("completeness_score", 0) < 1.0:
                considerations.append("Filing structure may be incomplete or poorly organized")

            if structure.get("missing_items"):
                considerations.append(f"Missing required items: {', '.join(structure['missing_items'])}")

        return considerations

    async def create_analysis_report(
            self,
            filing_data: Dict[str, Any],
            analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        report = {
            "metadata": {
                "analysis_date": datetime.utcnow().isoformat(),
                "filing_date": filing_data.get("filingDate", ""),
                "company": filing_data.get("companyName", ""),
                "cik": filing_data.get("cik", ""),
                "filing_type": filing_data.get("form", "")
            },
            "executive_summary": {
                "overall_assessment": self._generate_overall_assessment(analysis_results),
                "key_findings": self._extract_key_findings(analysis_results),
                "recommendations": await self.generate_investment_considerations(analysis_results)
            },
            "detailed_analysis": analysis_results,
            "appendix": {
                "methodology": "Analysis based on SEC EDGAR data using natural language processing and financial ratio analysis",
                "limitations": "Analysis is based on disclosed information and may not capture all material factors",
                "data_sources": ["SEC EDGAR", "Company filings"]
            }
        }

        return report

    def _generate_overall_assessment(self, analysis_results: Dict[str, Any]) -> str:
        """Generate overall assessment"""
        # Simple heuristic for overall assessment
        scores = []

        if "structure_analysis" in analysis_results:
            scores.append(analysis_results["structure_analysis"].get("completeness_score", 1.0))

        if "financial_metrics" in analysis_results and "key_ratios" in analysis_results["financial_metrics"]:
            ratios = analysis_results["financial_metrics"]["key_ratios"]

            # Score based on key ratios
            ratio_scores = []
            if "current_ratio" in ratios:
                ratio_scores.append(1.0 if ratios["current_ratio"] > 1.5 else 0.5)

            if "debt_to_equity" in ratios:
                ratio_scores.append(1.0 if ratios["debt_to_equity"] < 1.0 else 0.5)

            if ratio_scores:
                scores.append(statistics.mean(ratio_scores))

        if scores:
            avg_score = statistics.mean(scores)
            if avg_score >= 1.5:
                return "Strong financial position with comprehensive disclosures"
            elif avg_score >= 1.0:
                return "Adequate financial position with standard disclosures"
            else:
                return "Areas for improvement in financial position and/or disclosures"

        return "Limited information available for assessment"

    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis"""
        findings = []

        # Financial findings
        if "financial_metrics" in analysis_results and "key_ratios" in analysis_results["financial_metrics"]:
            ratios = analysis_results["financial_metrics"]["key_ratios"]

            if "current_ratio" in ratios:
                findings.append(f"Current ratio: {ratios['current_ratio']:.2f}")

            if "debt_to_equity" in ratios:
                findings.append(f"Debt-to-equity: {ratios['debt_to_equity']:.2f}")

        # Risk findings
        if "risk_analysis" in analysis_results:
            risk = analysis_results["risk_analysis"]
            findings.append(f"Risk factors identified: {risk.get('risk_count', 0)}")

            if risk.get("risk_severity", {}).get("high", 0) > 0:
                findings.append(f"High severity risks: {risk['risk_severity']['high']}")

        # Structure findings
        if "structure_analysis" in analysis_results:
            structure = analysis_results["structure_analysis"]
            findings.append(f"Filing completeness: {structure.get('completeness_rating', 'unknown')}")

        return findings