# domain/repositories/analysis_repository.py
"""Analysis repository interface (Port)"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..models.entities import FinancialAnalysis


class AnalysisRepository(ABC):
    """Repository interface for FinancialAnalysis aggregate"""

    @abstractmethod
    async def save(self, analysis: FinancialAnalysis) -> bool:
        """Save analysis to repository"""
        pass

    @abstractmethod
    async def find_by_id(self, analysis_id: str) -> Optional[FinancialAnalysis]:
        """Find analysis by ID"""
        pass

    @abstractmethod
    async def find_by_company(
        self,
        company_cik: str,
        analysis_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[FinancialAnalysis]:
        """Find analyses by company CIK"""
        pass

    @abstractmethod
    async def find_by_agent(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[FinancialAnalysis]:
        """Find analyses by agent ID"""
        pass

    @abstractmethod
    async def find_recent(
        self,
        days: int = 30,
        limit: int = 100
    ) -> List[FinancialAnalysis]:
        """Find recent analyses"""
        pass

    @abstractmethod
    async def update_metrics(
        self,
        analysis_id: str,
        metrics: List[Dict[str, Any]]
    ) -> bool:
        """Update analysis metrics"""
        pass

    @abstractmethod
    async def add_conclusion(
        self,
        analysis_id: str,
        conclusion: str
    ) -> bool:
        """Add conclusion to analysis"""
        pass

    @abstractmethod
    async def add_risk_factor(
        self,
        analysis_id: str,
        risk_factor: Dict[str, Any]
    ) -> bool:
        """Add the risk factor to analysis"""
        pass

    @abstractmethod
    async def delete(self, analysis_id: str) -> bool:
        """Delete analysis from repository"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Count analyses in repository"""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        field: str = "content",
        limit: int = 50
    ) -> List[FinancialAnalysis]:
        """Search analyses by query"""
        pass

    @abstractmethod
    async def get_statistics(
        self,
        company_cik: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analysis statistics"""
        pass