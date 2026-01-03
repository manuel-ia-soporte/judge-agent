# infrastructure/adapters/finance_adapter.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from domain.models.finance import SECDocument, FinancialAnalysis, FinancialMetric
from contracts.finance_contracts import CompanyFinancials, FinancialMetricData, SECFilingRequest
from infrastructure.sec_edgar.edgar_parser import EdgarParser


class FinanceAdapter:
    """Adapter for financial data transformations"""

    def __init__(self):
        self.parser = EdgarParser()

    def sec_response_to_document(self, sec_data: Dict[str, Any]) -> SECDocument:
        """Convert SEC API response to domain document"""
        filing_date = datetime.fromisoformat(
            sec_data.get("filingDate", datetime.utcnow().isoformat()).replace("Z", "")
        )

        period_end_str = sec_data.get("period", {}).get("end", filing_date.isoformat())
        period_end = datetime.fromisoformat(period_end_str.replace("Z", ""))

        # Parse items from text
        items = {}
        text = sec_data.get("text", "")

        for item_num in ["1A", "7", "8"]:
            section = self.parser._extract_section(text, f"item_{item_num.lower()}")
            if section:
                items[f"Item {item_num}"] = section

        return SECDocument(
            document_id=sec_data.get("accessionNumber", ""),
            company_cik=sec_data.get("cik", ""),
            company_name=sec_data.get("companyName", ""),
            filing_type=sec_data.get("form", ""),
            filing_date=filing_date,
            period_end=period_end,
            document_url=sec_data.get("filingUrl", ""),
            content=sec_data,
            raw_text=text,
            items=items
        )

    def documents_to_company_financials(
            self,
            documents: List[SECDocument],
            ticker: Optional[str] = None
    ) -> CompanyFinancials:
        """Convert SEC documents to company financials"""
        if not documents:
            raise ValueError("No documents provided")

        # Use first document for company info
        primary_doc = documents[0]

        # Extract metrics from all documents
        metrics = self._extract_metrics_from_documents(documents)

        # Extract risk factors
        risk_factors = []
        for doc in documents:
            if "Item 1A" in doc.items:
                risks = self.parser._parse_risk_items(doc.items["Item 1A"])
                risk_factors.extend(risks)

        # Extract management discussion
        management_discussion = ""
        for doc in documents:
            if "Item 7" in doc.items:
                management_discussion = doc.items["Item 7"][:5000]  # Limit size
                break

        # Prepare filings list
        filings = []
        for doc in documents:
            filings.append({
                "document_id": doc.document_id,
                "filing_type": doc.filing_type,
                "filing_date": doc.filing_date.isoformat(),
                "period_end": doc.period_end.isoformat()
            })

        return CompanyFinancials(
            company_cik=primary_doc.company_cik,
            company_name=primary_doc.company_name,
            ticker=ticker,
            fiscal_year_end=primary_doc.period_end.strftime("%m-%d"),
            filings=filings,
            metrics=metrics,
            risk_factors=list(set(risk_factors))[:20],  # Limit to 20 unique
            management_discussion=management_discussion,
            recent_events=self._extract_recent_events(documents)
        )

    def _extract_metrics_from_documents(
            self,
            documents: List[SECDocument]
    ) -> Dict[str, List[FinancialMetricData]]:
        """Extract metrics from SEC documents"""
        metrics = {}

        for doc in documents:
            # Parse financial statements
            statements = self.parser.parse_financial_statements(doc.content)

            # Extract metrics from statements
            for statement_type, statement_data in statements.items():
                for metric_name, metric_data in statement_data.items():
                    if isinstance(metric_data, dict) and "value" in metric_data:
                        metric_key = metric_name.lower()

                        if metric_key not in metrics:
                            metrics[metric_key] = []

                        metrics[metric_key].append(
                            FinancialMetricData(
                                metric_name=metric_name,
                                value=metric_data["value"],
                                unit=metric_data.get("unit", "USD"),
                                period=datetime.fromisoformat(
                                    metric_data.get("period", doc.period_end.isoformat()).replace("Z", "")
                                ),
                                source_document=doc.document_id,
                                footnote=metric_data.get("footnote"),
                                is_estimated=False,
                                confidence=1.0
                            )
                        )

        return metrics

    def _extract_recent_events(self, documents: List[SECDocument]) -> List[Dict[str, Any]]:
        """Extract recent events from documents"""
        events = []

        for doc in documents:
            # Look for 8-K filings (current reports)
            if doc.filing_type == "8-K":
                event = {
                    "date": doc.filing_date.isoformat(),
                    "type": "current_report",
                    "document_id": doc.document_id,
                    "description": f"8-K filing on {doc.filing_date.strftime('%Y-%m-%d')}"
                }
                events.append(event)

        # Sort by date, most recent first
        events.sort(key=lambda x: x["date"], reverse=True)
        return events[:10]  # Return only 10 most recent

    def analysis_to_dict(self, analysis: FinancialAnalysis) -> Dict[str, Any]:
        """Convert financial analysis to dictionary"""
        return {
            "analysis_id": analysis.analysis_id,
            "agent_id": analysis.agent_id,
            "company_ticker": analysis.company_ticker,
            "analysis_date": analysis.analysis_date.isoformat(),
            "content": analysis.content,
            "metrics_used": [m.value for m in analysis.metrics_used],
            "source_count": len(analysis.source_documents),
            "conclusions": analysis.conclusions,
            "risks_identified": analysis.risks_identified,
            "assumptions": analysis.assumptions,
            "confidence_score": analysis.confidence_score,
            "source_citations": analysis.get_cited_sources()
        }

    def create_comparison_matrix(
            self,
            companies: List[CompanyFinancials],
            metrics: List[str]
    ) -> Dict[str, Any]:
        """Create comparison matrix for multiple companies"""
        matrix = {}

        for company in companies:
            company_data = {}

            for metric in metrics:
                latest = company.get_latest_metric(metric)
                if latest:
                    company_data[metric] = {
                        "value": latest.value,
                        "unit": latest.unit,
                        "period": latest.period.isoformat(),
                        "confidence": latest.confidence
                    }
                else:
                    company_data[metric] = None

            matrix[company.company_name] = company_data

        # Calculate relative performance
        relative_performance = self._calculate_relative_performance(matrix)

        return {
            "companies": list(matrix.keys()),
            "metrics": metrics,
            "matrix": matrix,
            "relative_performance": relative_performance,
            "generated_at": datetime.utcnow().isoformat()
        }

    def _calculate_relative_performance(
            self,
            matrix: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate relative performance between companies"""
        if not matrix:
            return {}

        # For each metric, rank companies
        rankings = {}

        for metric in list(next(iter(matrix.values())).keys()):
            valid_companies = [
                (company, data[metric]["value"])
                for company, data in matrix.items()
                if data[metric] is not None
            ]

            if valid_companies:
                # Sort by value (higher is better for most metrics)
                sorted_companies = sorted(valid_companies, key=lambda x: x[1], reverse=True)

                rankings[metric] = {
                    "rankings": [
                        {"company": company, "value": value, "rank": i + 1}
                        for i, (company, value) in enumerate(sorted_companies)
                    ],
                    "best": sorted_companies[0][0],
                    "worst": sorted_companies[-1][0],
                    "range": sorted_companies[0][1] - sorted_companies[-1][1]
                }

        return rankings