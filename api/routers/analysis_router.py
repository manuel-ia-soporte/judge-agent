# api/routers/analysis_router.py
import uuid
from fastapi import APIRouter, HTTPException, Request
from typing import List

from contracts.api.requests.analysis_requests import (
    AnalyzeCompanyRequest,
    CompareCompaniesRequest,
)
from contracts.api.responses.analysis_responses import AnalysisResultResponse
from application.commands.analyze_company_command import (
    AnalyzeCompanyCommand,
    AnalysisType,
)

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResultResponse)
async def analyze_company(request: AnalyzeCompanyRequest, req: Request):
    """
    Analyze a company based on SEC filings.
    """
    try:
        container = req.app.state.container
        finance_agent = container.get_finance_agent()

        # Map request to command
        analysis_type = AnalysisType(request.analysis_type) if request.analysis_type else AnalysisType.COMPREHENSIVE

        command = AnalyzeCompanyCommand(
            company_cik=request.company_cik,
            analysis_type=analysis_type,
            request_id=str(uuid.uuid4()),
        )

        result = await finance_agent.analyze(command)

        # Convert metrics to response format
        metrics_response = [
            {"name": m.get("name", ""), "value": float(m.get("value", 0))}
            for m in (result.metrics or [])
        ]

        # Convert ratios to response format
        ratios_response = [
            {"name": r.get("name", ""), "value": float(r.get("value", 0))}
            for r in (result.ratios or [])
        ]

        # Convert risk factors to response format
        risk_factors_response = [
            {
                "description": rf.get("description", ""),
                "category": rf.get("category"),
                "severity": rf.get("severity"),
                "probability": float(rf.get("probability", 0.5)),
                "impact": rf.get("impact"),
                "risk_score": float(rf.get("probability", 0.5)),
            }
            for rf in (result.risk_factors or [])
        ]

        return AnalysisResultResponse(
            analysis_id=result.analysis_id,
            company_cik=result.company_cik,
            analysis_type=result.analysis_type,
            analysis_date=result.analysis_date,
            metrics=metrics_response,
            ratios=ratios_response,
            risk_factors=risk_factors_response,
            risk_level=result.risk_assessment.get("risk_level", "unknown") if result.risk_assessment else "unknown",
            risk_score=result.risk_assessment.get("risk_score", 0.0) if result.risk_assessment else 0.0,
            conclusions=result.conclusions or [],
            recommendations=result.recommendations or [],
            documents_analyzed=result.documents_analyzed,
            confidence_score=result.confidence_score,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/compare")
async def compare_companies(request: CompareCompaniesRequest, req: Request):
    """
    Compare multiple companies.
    """
    try:
        container = req.app.state.container
        finance_agent = container.get_finance_agent()

        # Analyze each company
        analyses = []
        all_ciks = [request.primary_cik] + request.peer_ciks

        for cik in all_ciks:
            command = AnalyzeCompanyCommand(
                company_cik=cik,
                analysis_type=AnalysisType.COMPARATIVE,
                request_id=str(uuid.uuid4()),
            )
            result = await finance_agent.analyze(command)
            analyses.append(result.to_dict() if hasattr(result, 'to_dict') else result.__dict__)

        comparison = finance_agent.compare(analyses)

        return {
            "primary_company": request.primary_cik,
            "peer_companies": request.peer_ciks,
            "comparison": comparison,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/capabilities")
async def list_capabilities(req: Request):
    """
    List available analysis capabilities.
    """
    container = req.app.state.container
    finance_agent = container.get_finance_agent()

    capabilities = finance_agent.list_capabilities()

    return {
        "capabilities": [
            {
                "name": cap.name,
                "description": cap.schema.description,
                "required_permission": cap.required_permission,
            }
            for cap in capabilities.values()
        ]
    }
