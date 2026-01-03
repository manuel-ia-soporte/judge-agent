# agents/finance_agent/finance_agent.py
from typing import Dict, List, Any, Optional
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from domain.models.agent import Agent, AgentStatus, AgentCapabilities, AgentCapability
from domain.models.finance import FinancialAnalysis, SECDocument
from infrastructure.sec_edgar.sec_client import SECClient
from infrastructure.sec_edgar.edgar_parser import EdgarParser
from contracts.finance_contracts import CompanyFinancials


@dataclass
class FinanceAgent:
    """Finance Agent for SEC analysis"""

    agent_id: str
    agent_name: str = "FinanceAnalysisAgent"
    status: AgentStatus = AgentStatus.REGISTERED
    capabilities: AgentCapabilities = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = AgentCapabilities(
                capabilities=[
                    AgentCapability.FINANCIAL_ANALYSIS,
                    AgentCapability.SEC_FILING_ANALYSIS,
                    AgentCapability.RISK_ASSESSMENT,
                    AgentCapability.DATA_EXTRACTION
                ],
                max_concurrent_tasks=3,
                processing_timeout=120,
                supports_batch=True,
                requires_grounding=True
            )

        self.sec_client = SECClient()
        self.edgar_parser = EdgarParser()
        self.logger = logging.getLogger(__name__)
        self.is_active = False

    async def analyze_company(
            self,
            company_cik: str,
            analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze a company using SEC filings"""
        self.logger.info(f"Starting analysis for CIK: {company_cik}")

        try:
            # 1. Fetch recent filings
            filings = await self.sec_client.search_filings(
                company=company_cik,
                filing_types=["10-K", "10-Q"],
                start_date="2022-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d")
            )

            if not filings:
                return {"error": f"No filings found for CIK {company_cik}"}

            # 2. Parse filings
            sec_documents = []
            for filing in filings[:3]:  # Analyze 3 most recent
                doc = await self._parse_filing(filing)
                if doc:
                    sec_documents.append(doc)

            # 3. Perform analysis based on type
            if analysis_type == "comprehensive":
                analysis = await self._comprehensive_analysis(sec_documents)
            elif analysis_type == "risk_focused":
                analysis = await self._risk_analysis(sec_documents)
            elif analysis_type == "financial":
                analysis = await self._financial_analysis(sec_documents)
            else:
                analysis = await self._quick_analysis(sec_documents)

            # 4. Generate conclusions
            conclusions = self._generate_conclusions(analysis, sec_documents)

            return {
                "company_cik": company_cik,
                "analysis_type": analysis_type,
                "analysis_date": datetime.utcnow().isoformat(),
                "documents_analyzed": len(sec_documents),
                "analysis": analysis,
                "conclusions": conclusions,
                "key_metrics": self._extract_key_metrics(sec_documents),
                "risk_factors": self._extract_risk_factors(sec_documents)
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}

    async def compare_companies(
            self,
            company_ciks: List[str],
            comparison_metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple companies"""
        comparisons = {}

        for cik in company_ciks:
            try:
                analysis = await self.analyze_company(cik, "financial")
                comparisons[cik] = {
                    "analysis": analysis,
                    "metrics": self._extract_comparison_metrics(analysis, comparison_metrics)
                }
            except Exception as e:
                self.logger.warning(f"Failed to analyze {cik}: {e}")
                comparisons[cik] = {"error": str(e)}

        # Generate comparison insights
        insights = self._generate_comparison_insights(comparisons)

        return {
            "companies": company_ciks,
            "comparisons": comparisons,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _parse_filing(self, filing_data: Dict[str, Any]) -> Optional[SECDocument]:
        """Parse SEC filing into domain document"""
        try:
            # Extract full filing text
            filing_text = await self.sec_client.download_filing(
                filing_data.get("cik"),
                filing_data.get("accessionNumber")
            )

            if not filing_text:
                return None

            # Create SEC document
            return SECDocument(
                document_id=filing_data.get("accessionNumber", ""),
                company_cik=filing_data.get("cik", ""),
                company_name=filing_data.get("companyName", ""),
                filing_type=filing_data.get("form", ""),
                filing_date=datetime.fromisoformat(filing_data.get("filingDate", datetime.utcnow().isoformat())),
                period_end=datetime.fromisoformat(
                    filing_data.get("period", {}).get("end", datetime.utcnow().isoformat())),
                document_url=filing_data.get("filingUrl", ""),
                content=filing_data,
                raw_text=filing_text,
                items=self._extract_filing_items(filing_text)
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse filing: {e}")
            return None

    def _extract_filing_items(self, filing_text: str) -> Dict[str, str]:
        """Extract items from filing text"""
        items = {}

        # Extract standard items
        item_patterns = {
            "Item 1A": r"(ITEM\s+1A\.?\s*RISK\s+FACTORS)(.*?)(?=ITEM\s+1B|\Z)",
            "Item 7": r"(ITEM\s+7\.?\s*MANAGEMENT'S\s+DISCUSSION)(.*?)(?=ITEM\s+7A|\Z)",
            "Item 8": r"(ITEM\s+8\.?\s*FINANCIAL\s+STATEMENTS)(.*?)(?=ITEM\s+9|\Z)"
        }

        for item_name, pattern in item_patterns.items():
            match = re.search(pattern, filing_text, re.IGNORECASE | re.DOTALL)
            if match:
                items[item_name] = match.group(2).strip()

        return items

    async def _comprehensive_analysis(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        analysis = {}

        # Financial analysis
        analysis["financial"] = self._analyze_financials(documents)

        # Risk analysis
        analysis["risk"] = self._analyze_risks(documents)

        # Operational analysis
        analysis["operational"] = self._analyze_operations(documents)

        # Strategic analysis
        analysis["strategic"] = self._analyze_strategy(documents)

        return analysis

    async def _risk_analysis(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Focus on risk analysis"""
        return {
            "risk_factors": self._extract_risk_factors(documents),
            "risk_assessment": self._assess_risk_level(documents),
            "mitigations": self._identify_mitigations(documents),
            "risk_trends": self._analyze_risk_trends(documents)
        }

    async def _financial_analysis(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Focus on financial analysis"""
        return {
            "financial_statements": self._extract_financial_statements(documents),
            "ratios": self._calculate_ratios(documents),
            "trends": self._analyze_financial_trends(documents),
            "liquidity": self._assess_liquidity(documents),
            "profitability": self._assess_profitability(documents)
        }

    async def _quick_analysis(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Quick overview analysis"""
        if not documents:
            return {}

        latest_doc = max(documents, key=lambda d: d.filing_date)

        return {
            "latest_filing": {
                "type": latest_doc.filing_type,
                "date": latest_doc.filing_date.isoformat(),
                "period": latest_doc.period_end.isoformat()
            },
            "key_numbers": self._extract_key_numbers([latest_doc]),
            "top_risks": self._extract_risk_factors([latest_doc])[:5],
            "management_highlights": self._extract_management_highlights([latest_doc])
        }

    def _extract_key_metrics(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Extract key financial metrics"""
        metrics = {}

        for doc in documents:
            statements = self.edgar_parser.parse_financial_statements(doc.content)
            for statement_type, statement_data in statements.items():
                for metric, value in statement_data.items():
                    if isinstance(value, dict) and "value" in value:
                        metrics[metric] = {
                            "value": value["value"],
                            "period": doc.period_end.isoformat(),
                            "source": doc.document_id
                        }

        return metrics

    def _extract_risk_factors(self, documents: List[SECDocument]) -> List[str]:
        """Extract risk factors from documents"""
        risks = []

        for doc in documents:
            if "Item 1A" in doc.items:
                doc_risks = self.edgar_parser._parse_risk_items(doc.items["Item 1A"])
                risks.extend(doc_risks)

        return list(set(risks))[:10]  # Return top 10 unique risks

    def _generate_conclusions(self, analysis: Dict[str, Any], documents: List[SECDocument]) -> List[str]:
        """Generate conclusions from analysis"""
        conclusions = []

        # Add financial conclusions
        if "financial" in analysis and analysis["financial"].get("ratios"):
            ratios = analysis["financial"]["ratios"]

            if ratios.get("current_ratio", 0) < 1:
                conclusions.append("Potential liquidity concern based on current ratio")
            elif ratios.get("current_ratio", 0) > 3:
                conclusions.append("Strong liquidity position")

            if ratios.get("debt_to_equity", 0) > 2:
                conclusions.append("High leverage ratio indicates significant debt burden")

        # Add risk conclusions
        if "risk" in analysis:
            risk_count = len(analysis["risk"].get("risk_factors", []))
            if risk_count > 10:
                conclusions.append(f"Company discloses {risk_count} risk factors, indicating complex risk profile")

        # Add operational conclusions
        if len(documents) > 0:
            latest_date = max(doc.filing_date for doc in documents)
            conclusions.append(f"Analysis based on filings through {latest_date.strftime('%B %Y')}")

        return conclusions

    def _analyze_financials(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze financial statements"""
        # Implementation would parse and analyze financial data
        return {}

    def _analyze_risks(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze risks"""
        # Implementation would analyze risk factors
        return {}

    def _analyze_operations(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze operations"""
        # Implementation would analyze operational data
        return {}

    def _analyze_strategy(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze strategy"""
        # Implementation would analyze strategic information
        return {}

    def _extract_key_numbers(self, documents: List[SECDocument]) -> Dict[str, float]:
        """Extract key numbers"""
        # Implementation would extract key financial numbers
        return {}

    def _extract_management_highlights(self, documents: List[SECDocument]) -> List[str]:
        """Extract management highlights"""
        # Implementation would extract highlights from MD&A
        return []

    def _extract_financial_statements(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Extract financial statements"""
        # Implementation would extract and parse financial statements
        return {}

    def _calculate_ratios(self, documents: List[SECDocument]) -> Dict[str, float]:
        """Calculate financial ratios"""
        # Implementation would calculate ratios
        return {}

    def _analyze_financial_trends(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze financial trends"""
        # Implementation would analyze trends
        return {}

    def _assess_liquidity(self, documents: List[SECDocument]) -> str:
        """Assess liquidity"""
        # Implementation would assess liquidity
        return "adequate"

    def _assess_profitability(self, documents: List[SECDocument]) -> str:
        """Assess profitability"""
        # Implementation would assess profitability
        return "profitable"

    def _assess_risk_level(self, documents: List[SECDocument]) -> str:
        """Assess risk level"""
        # Implementation would assess risk level
        return "medium"

    def _identify_mitigations(self, documents: List[SECDocument]) -> List[str]:
        """Identify risk mitigations"""
        # Implementation would identify mitigations
        return []

    def _analyze_risk_trends(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze risk trends"""
        # Implementation would analyze risk trends
        return {}

    def _extract_comparison_metrics(self, analysis: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
        """Extract metrics for comparison"""
        # Implementation would extract comparison metrics
        return {metric: 0.0 for metric in metrics}

    def _generate_comparison_insights(self, comparisons: Dict[str, Any]) -> List[str]:
        """Generate comparison insights"""
        # Implementation would generate insights
        return []

    async def start(self):
        """Start the finance agent"""
        self.is_active = True
        self.status = AgentStatus.ACTIVE
        self.logger.info(f"Finance agent {self.agent_id} started")

    async def stop(self):
        """Stop the finance agent"""
        self.is_active = False
        self.status = AgentStatus.OFFLINE
        self.logger.info(f"Finance agent {self.agent_id} stopped")