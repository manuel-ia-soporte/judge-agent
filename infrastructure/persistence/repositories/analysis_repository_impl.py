# infrastructure/persistence/repositories/analysis_repository_impl.py
"""Analysis Repository Implementation (Adapter)"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, UTC
import asyncio
import logging

from domain.repositories.analysis_repository import AnalysisRepository
from domain.models.entities import FinancialAnalysis
from domain.models.value_objects import FinancialMetric, RiskFactor


class InMemoryAnalysisRepository(AnalysisRepository):
    """In-memory implementation of AnalysisRepository"""

    def __init__(self):
        self.analyses: Dict[str, FinancialAnalysis] = {}
        self.company_index: Dict[str, List[str]] = {}
        self.agent_index: Dict[str, List[str]] = {}
        self.date_index: List[tuple[datetime, str]] = []
        self.logger = logging.getLogger(__name__)
        self.lock = asyncio.Lock()

    async def save(self, analysis: FinancialAnalysis) -> bool:
        """Save analysis to repository"""
        async with self.lock:
            try:
                # Store analysis
                self.analyses[analysis.analysis_id] = analysis

                # Update company index
                if analysis.company_cik not in self.company_index:
                    self.company_index[analysis.company_cik] = []
                self.company_index[analysis.company_cik].append(analysis.analysis_id)

                # Update agent index (if agent_id is tracked in the analysis)
                # Note: FinancialAnalysis in domain/models/entities.py doesn't have agent_id.
                # If we need to track agent, we might need to add it or use metadata.
                # For now, we skip agent index.

                # Update date index
                self.date_index.append((analysis.analysis_date, analysis.analysis_id))
                self.date_index.sort(key=lambda x: x[0], reverse=True)

                # Limit date index size
                if len(self.date_index) > 10000:
                    self.date_index = self.date_index[:5000]

                self.logger.debug(f"Analysis saved: {analysis.analysis_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to save analysis {analysis.analysis_id}: {e}")
                return False

    async def find_by_id(self, analysis_id: str) -> Optional[FinancialAnalysis]:
        """Find analysis by ID"""
        async with self.lock:
            return self.analyses.get(analysis_id)

    async def find_by_company(
            self,
            company_cik: str,
            analysis_type: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 100
    ) -> List[FinancialAnalysis]:
        """Find analyses by company CIK"""
        async with self.lock:
            if company_cik not in self.company_index:
                return []

            analysis_ids = self.company_index[company_cik]
            analyses = []

            for analysis_id in analysis_ids:
                if analysis_id in self.analyses:
                    analysis = self.analyses[analysis_id]

                    # Apply filters
                    if analysis_type and analysis.analysis_type.value != analysis_type:
                        continue

                    if start_date and analysis.analysis_date < start_date:
                        continue

                    if end_date and analysis.analysis_date > end_date:
                        continue

                    analyses.append(analysis)

            # Sort by date, most recent first
            analyses.sort(key=lambda a: a.analysis_date, reverse=True)

            return analyses[:limit]

    async def find_by_agent(
            self,
            agent_id: str,
            limit: int = 100
    ) -> List[FinancialAnalysis]:
        """Find analyses by agent ID"""
        # Note: This implementation requires agent_id to be stored in FinancialAnalysis.
        # Since our current FinancialAnalysis doesn't have agent_id, we return empty.
        # Alternatively, we could store agent_id in metadata.
        return []

    async def find_recent(
            self,
            days: int = 30,
            limit: int = 100
    ) -> List[FinancialAnalysis]:
        """Find recent analyses"""
        async with self.lock:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            recent_analyses = []

            for analysis_date, analysis_id in self.date_index:
                if analysis_date < cutoff:
                    break

                if analysis_id in self.analyses:
                    recent_analyses.append(self.analyses[analysis_id])

            return recent_analyses[:limit]

    async def update_metrics(
            self,
            analysis_id: str,
            metrics: List[Dict[str, Any]]
    ) -> bool:
        """Update analysis metrics"""
        async with self.lock:
            if analysis_id not in self.analyses:
                return False

            analysis = self.analyses[analysis_id]

            # Convert metrics to FinancialMetric objects
            financial_metrics = []
            for metric_data in metrics:
                try:
                    metric = FinancialMetric(
                        name=metric_data['name'],
                        value=metric_data['value'],
                        unit=metric_data.get('unit', 'USD'),
                        period=metric_data['period'],
                        source_document_id=metric_data['source_document_id'],
                        footnote=metric_data.get('footnote'),
                        is_estimated=metric_data.get('is_estimated', False),
                        confidence=metric_data.get('confidence', 1.0)
                    )
                    financial_metrics.append(metric)
                except Exception as e:
                    self.logger.warning(f"Failed to create metric: {e}")
                    continue

            # Update analysis metrics
            analysis.metrics = financial_metrics
            return True

    async def add_conclusion(
            self,
            analysis_id: str,
            conclusion: str
    ) -> bool:
        """Add conclusion to analysis"""
        async with self.lock:
            if analysis_id not in self.analyses:
                return False

            analysis = self.analyses[analysis_id]
            analysis.conclusions.append(conclusion)
            return True

    async def add_risk_factor(
            self,
            analysis_id: str,
            risk_factor: Dict[str, Any]
    ) -> bool:
        """Add the risk factor to analysis"""
        async with self.lock:
            if analysis_id not in self.analyses:
                return False

            analysis = self.analyses[analysis_id]

            try:
                risk = RiskFactor(
                    description=risk_factor['description'],
                    category=risk_factor['category'],
                    severity=risk_factor['severity'],
                    probability=risk_factor.get('probability', 0.5),
                    impact=risk_factor.get('impact', 'unknown'),
                    mitigation=risk_factor.get('mitigation')
                )
                analysis.risk_factors.append(risk)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to create risk factor: {e}")
                return False

    async def delete(self, analysis_id: str) -> bool:
        """Delete analysis from repository"""
        async with self.lock:
            if analysis_id not in self.analyses:
                return False

            analysis = self.analyses[analysis_id]

            # Remove from company index
            if analysis.company_cik in self.company_index:
                self.company_index[analysis.company_cik] = [
                    aid for aid in self.company_index[analysis.company_cik]
                    if aid != analysis_id
                ]

            # Remove from date index
            self.date_index = [
                (date, aid) for date, aid in self.date_index
                if aid != analysis_id
            ]

            # Remove from storage
            del self.analyses[analysis_id]

            self.logger.debug(f"Analysis deleted: {analysis_id}")
            return True

    async def count(self) -> int:
        """Count analyses in repository"""
        async with self.lock:
            return len(self.analyses)

    async def search(
            self,
            query: str,
            field: str = "content",
            limit: int = 50
    ) -> List[FinancialAnalysis]:
        """Search analyses by query"""
        # Note: This is a simple in-memory search. For production, use a search engine.
        async with self.lock:
            results = []
            query_lower = query.lower()

            for analysis in self.analyses.values():
                # Search in conclusions and insights
                searchable_text = ' '.join(analysis.conclusions + analysis.insights).lower()
                if query_lower in searchable_text:
                    results.append(analysis)

            return results[:limit]

    async def get_statistics(
            self,
            company_cik: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analysis statistics"""
        async with self.lock:
            stats = {
                "total_analyses": 0,
                "by_type": {},
                "by_company": {},
                "date_range": {
                    "earliest": None,
                    "latest": None
                }
            }

            for analysis in self.analyses.values():
                # Apply filters
                if company_cik and analysis.company_cik != company_cik:
                    continue

                if start_date and analysis.analysis_date < start_date:
                    continue

                if end_date and analysis.analysis_date > end_date:
                    continue

                # Count total
                stats["total_analyses"] += 1

                # Count by type
                analysis_type = analysis.analysis_type.value
                stats["by_type"][analysis_type] = stats["by_type"].get(analysis_type, 0) + 1

                # Count by company
                stats["by_company"][analysis.company_cik] = stats["by_company"].get(analysis.company_cik, 0) + 1

                # Update date range
                if not stats["date_range"]["earliest"] or analysis.analysis_date < stats["date_range"]["earliest"]:
                    stats["date_range"]["earliest"] = analysis.analysis_date

                if not stats["date_range"]["latest"] or analysis.analysis_date > stats["date_range"]["latest"]:
                    stats["date_range"]["latest"] = analysis.analysis_date

            # Convert dates to string for JSON serialization
            if stats["date_range"]["earliest"]:
                stats["date_range"]["earliest"] = stats["date_range"]["earliest"].isoformat()

            if stats["date_range"]["latest"]:
                stats["date_range"]["latest"] = stats["date_range"]["latest"].isoformat()

            return stats