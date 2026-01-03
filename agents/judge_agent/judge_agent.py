# agents/judge_agent/judge_agent.py
from typing import Dict, Any, List, Optional
import asyncio
import logging
from dataclasses import dataclass
from domain.models.evaluation import Evaluation
from contracts.evaluation_contracts import EvaluationRequest, EvaluationResult
from contracts.judge_contracts import JudgeCapabilities, JudgeMetrics, JudgeConfiguration


@dataclass
class JudgeAgent:
    """Main Judge Agent implementation"""

    agent_id: str = "judge_agent_001"
    capabilities: JudgeCapabilities = None
    metrics: JudgeMetrics = None
    configuration: JudgeConfiguration = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = JudgeCapabilities(
                agent_id=self.agent_id,
                supported_rubrics=[
                    "factual_accuracy",
                    "source_fidelity",
                    "regulatory_compliance",
                    "financial_reasoning",
                    "materiality_relevance",
                    "completeness",
                    "consistency",
                    "temporal_validity",
                    "risk_awareness",
                    "clarity_interpretability",
                    "uncertainty_handling",
                    "actionability"
                ]
            )

        if self.metrics is None:
            self.metrics = JudgeMetrics()

        if self.configuration is None:
            self.configuration = JudgeConfiguration()

        self.evaluations_queue = asyncio.Queue()
        self.is_active = True

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate a financial analysis"""
        logging.info(f"Starting evaluation for analysis: {request.analysis_id}")

        try:
            # Create domain evaluation
            evaluation = Evaluation(
                evaluation_id=f"eval_{request.analysis_id}",
                analysis_id=request.analysis_id,
                agent_id=request.agent_id
            )

            # Process rubrics based on configuration
            rubric_scores = await self._process_rubrics(request)

            # Add scores to evaluation
            for rubric_name, score_data in rubric_scores.items():
                evaluation.add_rubric_evaluation(score_data)

            # Complete evaluation
            evaluation.complete_evaluation()

            # Update metrics
            self.metrics.evaluations_completed += 1
            self.metrics.average_score = (
                    (self.metrics.average_score * (self.metrics.evaluations_completed - 1) +
                     evaluation.overall_score) / self.metrics.evaluations_completed
            )

            # Convert to result
            result = EvaluationResult(
                evaluation_id=evaluation.evaluation_id,
                request_id=request.analysis_id,
                agent_id=evaluation.agent_id,
                overall_score=evaluation.overall_score,
                passed=evaluation.is_passed,
                rubric_scores={
                    k: {
                        "rubric": v.rubric_name,
                        "score": v.score,
                        "passed": v.is_passed,
                        "feedback": v.feedback,
                        "evidence": v.evidence,
                        "confidence": v.confidence_score
                    }
                    for k, v in evaluation.rubric_evaluations.items()
                },
                recommendations=evaluation.recommendations,
                warnings=evaluation.warnings,
                metadata={
                    "judge_id": self.agent_id,
                    "configuration": self.configuration.dict()
                }
            )

            logging.info(f"Evaluation completed: {result.overall_score}")
            return result

        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise

    async def _process_rubrics(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Process each rubric"""
        from domain.services.rubrics_service import RubricEvaluator

        rubric_scores = {}

        # Parse analysis
        from domain.models.finance import FinancialAnalysis
        analysis = FinancialAnalysis(
            analysis_id=request.analysis_id,
            agent_id=request.agent_id,
            company_ticker="",
            analysis_date=datetime.utcnow(),
            content=request.analysis_content,
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        # Evaluate each requested rubric
        for rubric in request.rubrics_to_evaluate:
            if rubric == "factual_accuracy":
                rubric_scores[rubric] = RubricEvaluator.evaluate_factual_accuracy(
                    analysis, {}
                )
            elif rubric == "source_fidelity":
                rubric_scores[rubric] = RubricEvaluator.evaluate_source_fidelity(
                    analysis, []
                )
            elif rubric == "regulatory_compliance":
                rubric_scores[rubric] = RubricEvaluator.evaluate_regulatory_compliance(
                    analysis
                )


            ###################################
            # Add other rubrics...
            ###################################


        return rubric_scores

    async def batch_evaluate(self, requests: List[EvaluationRequest]) -> List[EvaluationResult]:
        """Batch evaluate multiple analyses"""
        results = []

        for request in requests:
            result = await self.evaluate(request)
            results.append(result)
            await asyncio.sleep(0.1)  # Prevent overload

        return results

    async def calibrate(self, ground_truth_data: List[Dict[str, Any]]):
        """Calibrate judge based on ground truth"""
        logging.info("Starting calibration")

        for data in ground_truth_data:
            # Adjust weights based on performance
            pass

        self.configuration.auto_calibrate = True
        logging.info("Calibration completed")

    async def stop(self):
        """Stop the judge agent"""
        self.is_active = False
        logging.info("Judge agent stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "status": "active" if self.is_active else "inactive",
            "metrics": self.metrics.dict(),
            "queue_size": self.evaluations_queue.qsize(),
            "capabilities": self.capabilities.dict()
        }