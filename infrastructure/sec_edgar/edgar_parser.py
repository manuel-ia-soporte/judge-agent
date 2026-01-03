# infrastructure/sec_edgar/edgar_parser.py
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import pandas as pd


class EdgarParser:
    """Parser for SEC EDGAR filings"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.us_gaap_patterns = {
            "revenue": r"Revenues?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "net_income": r"Net\s+Income\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "total_assets": r"Total\s+Assets\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "total_liabilities": r"Total\s+Liabilities\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "cash": r"Cash\s+and\s+cash\s+equivalents\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "debt": r"Total\s+Debt\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "equity": r"Total\s+Equity\s*\$?\s*([\d,]+(?:\.\d+)?)",
        }

        self.section_patterns = {
            "item_1a": r"(ITEM\s+1A\.?\s*RISK\s+FACTORS)(.*?)(?=ITEM\s+1B|\nITEM|\Z)",
            "item_7": r"(ITEM\s+7\.?\s*MANAGEMENT'S\s+DISCUSSION)(.*?)(?=ITEM\s+7A|\nITEM|\Z)",
            "item_8": r"(ITEM\s+8\.?\s*FINANCIAL\s+STATEMENTS)(.*?)(?=ITEM\s+9|\nITEM|\Z)",
        }

    def parse_financial_statements(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse financial statements from SEC filing"""
        try:
            if "facts" not in filing_data or "us-gaap" not in filing_data["facts"]:
                return self._parse_from_text(filing_data)

            return self._parse_from_structured_data(filing_data)

        except Exception as e:
            self.logger.error(f"Failed to parse financial statements: {e}")
            return {"error": str(e)}

    def _parse_from_structured_data(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse from structured XBRL data"""
        us_gaap = filing_data["facts"]["us-gaap"]
        statements = {}

        # Balance Sheet
        balance_sheet = self._extract_statement(us_gaap, "balance_sheet")
        if balance_sheet:
            statements["balance_sheet"] = balance_sheet

        # Income Statement
        income_statement = self._extract_statement(us_gaap, "income_statement")
        if income_statement:
            statements["income_statement"] = income_statement

        # Cash Flow
        cash_flow = self._extract_statement(us_gaap, "cash_flow")
        if cash_flow:
            statements["cash_flow"] = cash_flow

        return statements

    def _extract_statement(self, us_gaap: Dict[str, Any], statement_type: str) -> Dict[str, Any]:
        """Extract specific financial statement"""
        statement_items = {
            "balance_sheet": ["Assets", "Liabilities", "Equity", "Inventory", "Debt"],
            "income_statement": ["Revenue", "NetIncomeLoss", "CostOfRevenue", "GrossProfit"],
            "cash_flow": ["NetCashProvidedByUsedInOperatingActivities"]
        }

        statement_data = {}

        for item in statement_items.get(statement_type, []):
            if item in us_gaap:
                latest = self._get_latest_period(us_gaap[item])
                if latest:
                    statement_data[item] = {
                        "value": latest["value"],
                        "unit": latest.get("unit", "USD"),
                        "period": latest["period"],
                        "decimals": latest.get("decimals", 0)
                    }

        return statement_data

    def _get_latest_period(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get latest period from item data"""
        if "units" not in item_data:
            return None

        for unit_type, periods in item_data["units"].items():
            if periods:
                # Get most recent period
                latest = sorted(periods, key=lambda x: x.get("end", ""), reverse=True)[0]
                return {
                    "value": float(latest.get("val", 0)),
                    "period": latest.get("end", ""),
                    "unit": unit_type,
                    "decimals": latest.get("decimals", 0)
                }

        return None

    def _parse_from_text(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse from unstructured text"""
        statements = {}
        text = filing_data.get("text", "")

        # Extract using regex patterns
        for metric, pattern in self.us_gaap_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the last match (most recent)
                    value_str = matches[-1].replace(",", "")
                    statements[metric] = {
                        "value": float(value_str),
                        "unit": "USD",
                        "source": "regex_extraction"
                    }
                except ValueError:
                    continue

        return statements

    def extract_risk_factors(self, filing_data: Dict[str, Any]) -> List[str]:
        """Extract risk factors from filing"""
        risk_factors = []

        # Try structured data first
        if "facts" in filing_data and "us-gaap" in filing_data["facts"]:
            risk_items = self._extract_structured_risks(filing_data["facts"])
            risk_factors.extend(risk_items)

        # Fallback to text extraction
        text = filing_data.get("text", "")
        risk_section = self._extract_section(text, "item_1a")

        if risk_section:
            # Split into individual risk factors
            risk_items = self._parse_risk_items(risk_section)
            risk_factors.extend(risk_items)

        # Deduplicate and return
        return list(set(risk_factors))[:50]  # Limit to 50

    def _extract_structured_risks(self, facts: Dict[str, Any]) -> List[str]:
        """Extract risks from structured data"""
        risks = []

        # Look for risk-related disclosures
        risk_keywords = ["RiskFactorsTextBlock", "RiskFactor", "Uncertainty"]

        for keyword in risk_keywords:
            if keyword in facts.get("us-gaap", {}):
                risk_data = facts["us-gaap"][keyword]
                if "units" in risk_data and "USD" in risk_data["units"]:
                    # This is actually a text block
                    text_blocks = risk_data["units"]["USD"]
                    for block in text_blocks:
                        if "val" in block:
                            risks.append(block["val"])

        return risks

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract specific section from text"""
        if section_name not in self.section_patterns:
            return None

        pattern = self.section_patterns[section_name]
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(2).strip()

        return None

    def _parse_risk_items(self, risk_section: str) -> List[str]:
        """Parse individual risk items from risk section"""
        # Split by common risk item patterns
        risk_items = []

        # Pattern for numbered risk items
        number_pattern = r"(\d+\.\s*)(.*?)(?=\d+\.\s*|\Z)"
        matches = re.findall(number_pattern, risk_section, re.DOTALL)

        for match in matches:
            risk_text = match[1].strip()
            # Clean up the text
            risk_text = re.sub(r'\s+', ' ', risk_text)  # Normalize whitespace
            risk_text = risk_text[:500]  # Limit length
            risk_items.append(risk_text)

        # If no numbered items, split by line breaks
        if not risk_items:
            lines = risk_section.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:  # Meaningful line
                    risk_items.append(line[:500])

        return risk_items

    def parse_management_discussion(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse management discussion and analysis"""
        text = filing_data.get("text", "")
        md_section = self._extract_section(text, "item_7")

        if not md_section:
            return {"content": "", "sections": []}

        # Parse MD&A sections
        sections = self._parse_mda_sections(md_section)

        # Extract key topics
        topics = self._extract_mda_topics(md_section)

        return {
            "content": md_section[:10000],  # Limit size
            "sections": sections,
            "topics": topics,
            "word_count": len(md_section.split())
        }

    def _parse_mda_sections(self, md_text: str) -> List[Dict[str, str]]:
        """Parse MD&A into sections"""
        sections = []

        # Look for common MD&A headings
        heading_patterns = [
            (r"(Results\s+of\s+Operations)(.*?)(?=(?:Liquidity|Capital|Outlook)|\Z)", "results"),
            (r"(Liquidity\s+and\s+Capital\s+Resources)(.*?)(?=(?:Results|Outlook)|\Z)", "liquidity"),
            (r"(Critical\s+Accounting\s+Policies)(.*?)(?=(?:Results|Liquidity)|\Z)", "accounting"),
            (r"(Outlook|Future\s+Operations)(.*?)(?=(?:Results|Liquidity)|\Z)", "outlook")
        ]

        for pattern, section_type in heading_patterns:
            match = re.search(pattern, md_text, re.IGNORECASE | re.DOTALL)
            if match:
                sections.append({
                    "type": section_type,
                    "heading": match.group(1).strip(),
                    "content": match.group(2).strip()[:2000],
                    "word_count": len(match.group(2).split())
                })

        return sections

    def _extract_mda_topics(self, md_text: str) -> List[str]:
        """Extract key topics from MD&A"""
        topics = []

        # Look for key topics
        topic_keywords = [
            "revenue growth", "margin", "expenses", "competition",
            "market share", "innovation", "acquisition", "restructuring",
            "regulation", "tax", "currency", "inflation", "supply chain"
        ]

        for keyword in topic_keywords:
            if re.search(keyword, md_text, re.IGNORECASE):
                topics.append(keyword)

        return topics

    def validate_financial_data(
            self,
            extracted_data: Dict[str, Any],
            source_cik: str
    ) -> Dict[str, Any]:
        """Validate extracted financial data"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": 0,
            "checks_total": 0
        }

        # Check 1: Basic accounting equation (Assets = Liabilities + Equity)
        if all(k in extracted_data for k in ["total_assets", "total_liabilities", "equity"]):
            validation_results["checks_total"] += 1
            assets = extracted_data["total_assets"]["value"]
            liabilities = extracted_data["total_liabilities"]["value"]
            equity = extracted_data["equity"]["value"]

            tolerance = abs(assets) * 0.01  # 1% tolerance
            if abs(assets - (liabilities + equity)) > tolerance:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Accounting equation imbalance: Assets ({assets}) != Liabilities ({liabilities}) + Equity ({equity})"
                )
            else:
                validation_results["checks_passed"] += 1

        # Check 2: Positive revenue
        if "revenue" in extracted_data:
            validation_results["checks_total"] += 1
            revenue = extracted_data["revenue"]["value"]
            if revenue <= 0:
                validation_results["warnings"].append(
                    f"Non-positive revenue: {revenue}"
                )
            else:
                validation_results["checks_passed"] += 1

        # Check 3: Valid dates
        for metric, data in extracted_data.items():
            if "period" in data:
                validation_results["checks_total"] += 1
                try:
                    datetime.fromisoformat(data["period"].replace("Z", ""))
                    validation_results["checks_passed"] += 1
                except ValueError:
                    validation_results["warnings"].append(
                        f"Invalid date format for {metric}: {data['period']}"
                    )

        return validation_results