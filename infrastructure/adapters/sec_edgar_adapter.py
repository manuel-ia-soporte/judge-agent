# infrastructure/adapters/sec_edgar_adapter.py
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import date, datetime
from domain.entities.sec_filing import SECFiling
from domain.models.entities import SECDocument

logger = logging.getLogger(__name__)


# Mock data for testing - Real financial data from public filings
MOCK_COMPANY_DATA: Dict[str, Dict[str, Any]] = {
    "0000320193": {  # Apple Inc.
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "revenue": 394328000000,
        "net_income": 96995000000,
        "total_assets": 352583000000,
        "total_liabilities": 290437000000,
        "stockholders_equity": 62146000000,
        "cash": 29965000000,
        "long_term_debt": 111088000000,
        "gross_margin": 0.433,
        "operating_margin": 0.298,
        "current_ratio": 0.94,
        "debt_to_equity": 1.79,
    },
    "0000789019": {  # Microsoft
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "revenue": 211915000000,
        "net_income": 72361000000,
        "total_assets": 411976000000,
        "total_liabilities": 205753000000,
        "stockholders_equity": 206223000000,
        "cash": 34704000000,
        "long_term_debt": 41990000000,
        "gross_margin": 0.689,
        "operating_margin": 0.417,
        "current_ratio": 1.77,
        "debt_to_equity": 0.20,
    },
    "0001318605": {  # Tesla
        "name": "Tesla, Inc.",
        "ticker": "TSLA",
        "revenue": 96773000000,
        "net_income": 14997000000,
        "total_assets": 106618000000,
        "total_liabilities": 43009000000,
        "stockholders_equity": 63609000000,
        "cash": 16398000000,
        "long_term_debt": 2857000000,
        "gross_margin": 0.184,
        "operating_margin": 0.094,
        "current_ratio": 1.73,
        "debt_to_equity": 0.04,
    },
}


class SECEdgarAdapter:
    """
    Infrastructure adapter: File system / Mock data → Domain filing
    """

    def load_filing(self, filing_path: Path) -> SECFiling:
        if not filing_path.exists():
            raise FileNotFoundError(f"SEC filing not found: {filing_path}")

        text = filing_path.read_text(encoding="utf-8")
        form = filing_path.stem.split("_")[0]
        filing_date = date.fromtimestamp(filing_path.stat().st_mtime)

        return SECFiling(
            form=form,
            filing_date=filing_date,
            text=text,
            source_path=str(filing_path),
        )

    def find_by_cik(self, cik: str) -> List[SECDocument]:
        """
        Find SEC documents by CIK. Uses mock data for testing.
        """
        normalized_cik = cik.zfill(10)

        if normalized_cik in MOCK_COMPANY_DATA:
            company = MOCK_COMPANY_DATA[normalized_cik]
            logger.info(f"Loading mock SEC data for {company['name']} (CIK: {normalized_cik})")
            return self._create_mock_documents(normalized_cik, company)

        logger.warning(f"No SEC data found for CIK: {cik}")
        return []

    def _create_mock_documents(self, cik: str, company: Dict[str, Any]) -> List[SECDocument]:
        """Create mock SEC documents with realistic financial data."""
        now = datetime.now()

        return [
            SECDocument(
                cik=cik,
                form_type="10-K",
                filing_date=now,
                content=self._generate_10k_content(cik, company),
                accession_number=f"{cik}-24-000001",
            ),
            SECDocument(
                cik=cik,
                form_type="10-Q",
                filing_date=now,
                content=self._generate_10q_content(cik, company),
                accession_number=f"{cik}-24-000002",
            ),
        ]

    def _generate_10k_content(self, cik: str, company: Dict[str, Any]) -> str:
        """Generate realistic 10-K content."""
        return f"""
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K

ANNUAL REPORT - {company['name']} ({company['ticker']})
CIK: {cik}

ITEM 1. BUSINESS
{company['name']} is a leading company in its industry, focused on innovation and growth.

ITEM 1A. RISK FACTORS
- Market competition risk: The company faces significant competition.
- Supply chain risk: Global supply chain disruptions may affect operations.
- Regulatory risk: Changes in regulations could impact business.
- Currency risk: Foreign exchange fluctuations affect international revenue.
- Technology risk: Rapid technological changes require continuous innovation.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Financial Highlights (Annual):
- Total Revenue: ${company['revenue']:,.0f}
- Net Income: ${company['net_income']:,.0f}
- Total Assets: ${company['total_assets']:,.0f}
- Total Liabilities: ${company['total_liabilities']:,.0f}
- Stockholders' Equity: ${company['stockholders_equity']:,.0f}

Key Financial Ratios:
- Gross Margin: {company['gross_margin']*100:.1f}%
- Operating Margin: {company['operating_margin']*100:.1f}%
- Current Ratio: {company['current_ratio']:.2f}
- Debt to Equity: {company['debt_to_equity']:.2f}

ITEM 8. FINANCIAL STATEMENTS

Consolidated Balance Sheet:
- Cash and Cash Equivalents: ${company['cash']:,.0f}
- Long-term Debt: ${company['long_term_debt']:,.0f}

The company maintains a strong financial position with adequate liquidity.
"""

    def _generate_10q_content(self, cik: str, company: Dict[str, Any]) -> str:
        """Generate realistic 10-Q content."""
        quarterly_revenue = company['revenue'] / 4
        quarterly_income = company['net_income'] / 4

        return f"""
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-Q

QUARTERLY REPORT - {company['name']} ({company['ticker']})
CIK: {cik}

PART I - FINANCIAL INFORMATION

ITEM 1. FINANCIAL STATEMENTS
- Quarterly Revenue: ${quarterly_revenue:,.0f}
- Quarterly Net Income: ${quarterly_income:,.0f}
- Current Ratio: {company['current_ratio']:.2f}

ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS
The company continued strong performance across all business segments.
Operating cash flow remained robust during the quarter.

PART II - OTHER INFORMATION

ITEM 1A. RISK FACTORS
No material changes from the most recent 10-K filing.
"""
