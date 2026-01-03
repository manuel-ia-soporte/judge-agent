# tests/test_performance.py
"""
Performance tests for Finance Judge System
"""
import pytest
import asyncio
import time
import statistics
import psutil
import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.judge_agent.judge_agent import JudgeAgent
from contracts.evaluation_contracts import EvaluationRequest, RubricCategory
from domain.models.evaluation import Evaluation, RubricEvaluation
from infrastructure.adapters.evaluation_adapter import EvaluationAdapter


@pytest.mark.performance
class TestEvaluationPerformance:
    """Performance tests for evaluation system"""

    @pytest.fixture
    def judge_agent(self):
        """Create judge agent for performance tests"""
        return JudgeAgent(agent_id="performance_test")

    @pytest.fixture
    def sample_evaluation_request(self):
        """Create sample evaluation request"""
        return EvaluationRequest(
            analysis_id="perf_analysis",
            agent_id="perf_agent",
            analysis_content="Apple Inc. reported revenue of $81.8 billion in Q3 2023, representing a 2% year-over-year increase. The company's net income was $19.9 billion, with earnings per share of $1.26. Current ratio improved to 1.5, indicating strong liquidity. Debt-to-equity ratio remains stable at 1.8. The company faces risks including market competition, supply chain disruptions, and regulatory changes. Management expects continued growth in the next quarter, assuming stable market conditions.",
            source_documents=[
                {
                    "cik": "0000320193",
                    "filing_type": "10-Q",
                    "date": "2023-06-30",
                    "content": "SEC filing content placeholder"
                }
            ],
            rubrics_to_evaluate=[
                RubricCategory.FACTUAL_ACCURACY,
                RubricCategory.SOURCE_FIDELITY,
                RubricCategory.REGULATORY_COMPLIANCE,
                RubricCategory.FINANCIAL_REASONING,
                RubricCategory.MATERIALITY_RELEVANCE,
                RubricCategory.COMPLETENESS,
                RubricCategory.CONSISTENCY,
                RubricCategory.RISK_AWARENESS,
                RubricCategory.CLARITY_INTERPRETABILITY,
                RubricCategory.UNCERTAINTY_HANDLING,
                RubricCategory.ACTIONABILITY
            ]
        )

    @pytest.mark.asyncio
    async def test_single_evaluation_latency(self, judge_agent, sample_evaluation_request):
        """Test latency of single evaluation"""
        # Mock rubric processing to isolate performance
        with patch.object(judge_agent, '_process_rubrics') as mock_process:
            mock_process.return_value = {
                rubric: RubricEvaluation(
                    rubric_name=rubric,
                    score=1.5,
                    is_passed=True,
                    feedback="Test",
                    evidence=[],
                    confidence_score=1.0
                )
                for rubric in sample_evaluation_request.rubrics_to_evaluate
            }

            # Warm-up
            await judge_agent.evaluate(sample_evaluation_request)

            # Measure performance
            latencies = []
            for _ in range(10):
                start_time = time.perf_counter()
                result = await judge_agent.evaluate(sample_evaluation_request)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                assert result is not None

            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

            print(f"\nSingle Evaluation Latency (10 runs):")
            print(f"  Average: {avg_latency:.2f} ms")
            print(f"  Minimum: {min_latency:.2f} ms")
            print(f"  Maximum: {max_latency:.2f} ms")
            print(f"  95th Percentile: {p95_latency:.2f} ms")

            # Assert performance requirements
            assert avg_latency < 1000  # Should be under 1 second
            assert p95_latency < 1500  # 95% under 1.5 seconds

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, judge_agent):
        """Test performance under concurrent load"""
        num_concurrent = 20
        num_iterations = 5

        async def evaluate_one(request):
            with patch.object(judge_agent, '_process_rubrics'):
                return await judge_agent.evaluate(request)

        # Create requests
        requests = [
            EvaluationRequest(
                analysis_id=f"concurrent_{i}",
                agent_id=f"agent_{i}",
                analysis_content=f"Concurrent test {i}",
                source_documents=[],
                rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
            )
            for i in range(num_concurrent * num_iterations)
        ]

        # Measure concurrent execution
        start_time = time.perf_counter()

        # Process in batches
        results = []
        for batch_num in range(num_iterations):
            batch_start = batch_num * num_concurrent
            batch_end = batch_start + num_concurrent
            batch = requests[batch_start:batch_end]

            batch_results = await asyncio.gather(*[
                evaluate_one(req) for req in batch
            ])
            results.extend(batch_results)

        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_evaluations = len(results)
        evaluations_per_second = total_evaluations / total_time

        print(f"\nConcurrent Evaluations Performance:")
        print(f"  Total evaluations: {total_evaluations}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Evaluations per second: {evaluations_per_second:.2f}")
        print(f"  Time per evaluation: {(total_time / total_evaluations) * 1000:.2f} ms")

        # Assert performance requirements
        assert evaluations_per_second > 5  # Should handle at least 5 eval/sec
        assert total_time < 30  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_memory_usage_growth(self, judge_agent):
        """Test memory usage growth during sustained load"""
        import gc

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run evaluations
        num_evaluations = 100

        with patch.object(judge_agent, '_process_rubrics'):
            for i in range(num_evaluations):
                request = EvaluationRequest(
                    analysis_id=f"memory_test_{i}",
                    agent_id="memory_agent",
                    analysis_content=f"Memory test content {i}",
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                await judge_agent.evaluate(request)

                # Force garbage collection every 10 evaluations
                if i % 10 == 0:
                    gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        print(f"\nMemory Usage Test:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory growth: {memory_growth:.2f} MB")
        print(f"  Growth per evaluation: {memory_growth / num_evaluations:.4f} MB")

        # Assert memory requirements
        assert memory_growth < 50  # Should not grow more than 50MB for 100 evaluations
        assert memory_growth / num_evaluations < 0.5  # Less than 0.5MB per evaluation

    @pytest.mark.asyncio
    async def test_cpu_utilization(self, judge_agent):
        """Test CPU utilization during evaluation"""
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()

        # Measure CPU during batch processing
        num_evaluations = 50

        # Get initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=0.1)

        start_time = time.perf_counter()

        with patch.object(judge_agent, '_process_rubrics'):
            # Process evaluations
            tasks = []
            for i in range(num_evaluations):
                request = EvaluationRequest(
                    analysis_id=f"cpu_test_{i}",
                    agent_id="cpu_agent",
                    analysis_content=f"CPU test content {i}",
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                tasks.append(judge_agent.evaluate(request))

            await asyncio.gather(*tasks)

        end_time = time.perf_counter()

        # Get CPU during processing
        processing_cpu = psutil.cpu_percent(interval=1)

        total_time = end_time - start_time
        evaluations_per_second = num_evaluations / total_time

        print(f"\nCPU Utilization Test:")
        print(f"  CPU cores: {cpu_count}")
        print(f"  Initial CPU: {initial_cpu:.1f}%")
        print(f"  Processing CPU: {processing_cpu:.1f}%")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Evaluations per second: {evaluations_per_second:.2f}")

        # Assert CPU requirements
        assert processing_cpu < 80  # Should not exceed 80% CPU utilization
        assert evaluations_per_second > 2  # Should process at least 2 eval/sec

    @pytest.mark.asyncio
    async def test_scalability(self):
        """Test system scalability with multiple agents"""
        num_agents = 5
        evaluations_per_agent = 10

        agents = [JudgeAgent(agent_id=f"scale_agent_{i}") for i in range(num_agents)]

        async def agent_workload(agent, agent_id):
            results = []
            with patch.object(agent, '_process_rubrics'):
                for i in range(evaluations_per_agent):
                    request = EvaluationRequest(
                        analysis_id=f"scale_{agent_id}_{i}",
                        agent_id=f"client_{i}",
                        analysis_content=f"Scalability test {i}",
                        source_documents=[],
                        rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                    )

                    result = await agent.evaluate(request)
                    results.append(result)

            return results

        # Run all agents concurrently
        start_time = time.perf_counter()

        tasks = [agent_workload(agent, i) for i, agent in enumerate(agents)]
        all_results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()

        total_evaluations = num_agents * evaluations_per_agent
        total_time = end_time - start_time
        throughput = total_evaluations / total_time

        print(f"\nScalability Test ({num_agents} agents, {evaluations_per_agent} each):")
        print(f"  Total evaluations: {total_evaluations}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} evaluations/second")
        print(f"  Time per evaluation: {(total_time / total_evaluations) * 1000:.2f} ms")

        # Assert scalability requirements
        assert throughput > num_agents * 2  # Should scale with number of agents
        assert total_time < 30  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_response_time_consistency(self, judge_agent):
        """Test response time consistency over time"""
        num_evaluations = 100
        warmup_evaluations = 10

        response_times = []

        with patch.object(judge_agent, '_process_rubrics'):
            # Warm-up phase
            for i in range(warmup_evaluations):
                request = EvaluationRequest(
                    analysis_id=f"warmup_{i}",
                    agent_id="warmup_agent",
                    analysis_content="Warm-up content",
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                await judge_agent.evaluate(request)

            # Measurement phase
            for i in range(num_evaluations):
                request = EvaluationRequest(
                    analysis_id=f"consistency_{i}",
                    agent_id="consistency_agent",
                    analysis_content=f"Consistency test {i}",
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                start_time = time.perf_counter()
                result = await judge_agent.evaluate(request)
                end_time = time.perf_counter()

                response_time = (end_time - start_time) * 1000
                response_times.append(response_time)

                assert result is not None

        # Calculate consistency metrics
        avg_response = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        cv = (std_dev / avg_response) * 100 if avg_response > 0 else 0  # Coefficient of variation

        # Calculate percentiles
        percentiles = {
            '50th': statistics.quantiles(response_times, n=2)[0],
            '90th': statistics.quantiles(response_times, n=10)[8],
            '95th': statistics.quantiles(response_times, n=20)[18],
            '99th': statistics.quantiles(response_times, n=100)[98]
        }

        print(f"\nResponse Time Consistency ({num_evaluations} evaluations):")
        print(f"  Average: {avg_response:.2f} ms")
        print(f"  Standard deviation: {std_dev:.2f} ms")
        print(f"  Coefficient of variation: {cv:.2f}%")
        print(f"  50th percentile: {percentiles['50th']:.2f} ms")
        print(f"  90th percentile: {percentiles['90th']:.2f} ms")
        print(f"  95th percentile: {percentiles['95th']:.2f} ms")
        print(f"  99th percentile: {percentiles['99th']:.2f} ms")

        # Assert consistency requirements
        assert cv < 30  # Coefficient of variation should be less than 30%
        assert percentiles['95th'] < avg_response * 2  # 95th percentile less than 2x average

    @pytest.mark.asyncio
    async def test_large_document_processing(self, judge_agent):
        """Test performance with large documents"""
        # Create large analysis content (simulating long reports)
        large_content = "Large financial analysis " * 1000  # ~25KB

        request = EvaluationRequest(
            analysis_id="large_doc_test",
            agent_id="large_doc_agent",
            analysis_content=large_content,
            source_documents=[
                {
                    "cik": "0000320193",
                    "filing_type": "10-K",
                    "date": "2023-12-31",
                    "content": "Large SEC filing " * 5000  # ~125KB
                }
            ],
            rubrics_to_evaluate=[
                RubricCategory.FACTUAL_ACCURACY,
                RubricCategory.SOURCE_FIDELITY,
                RubricCategory.COMPLETENESS,
                RubricCategory.CLARITY_INTERPRETABILITY
            ]
        )

        # Measure performance
        with patch.object(judge_agent, '_process_rubrics'):
            start_time = time.perf_counter()
            result = await judge_agent.evaluate(request)
            end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000

        print(f"\nLarge Document Processing:")
        print(f"  Document size: ~{len(large_content) / 1024:.1f} KB")
        print(f"  Processing time: {processing_time:.2f} ms")

        assert result is not None
        assert processing_time < 5000  # Should process within 5 seconds

    @pytest.mark.asyncio
    async def test_evaluation_adapter_performance(self):
        """Test evaluation adapter performance"""
        # Create evaluation with many rubric scores
        evaluation = Evaluation(
            evaluation_id="adapter_perf_test",
            analysis_id="adapter_analysis",
            agent_id="adapter_agent"
        )

        # Add many rubric evaluations
        for i in range(20):  # 20 rubrics
            evaluation.add_rubric_evaluation(RubricEvaluation(
                rubric_name=f"rubric_{i}",
                score=1.5 + (i * 0.02),
                is_passed=True,
                feedback=f"Feedback for rubric {i}",
                evidence=[f"Evidence {j}" for j in range(3)],
                confidence_score=0.9
            ))

        # Measure conversion performance
        iterations = 1000
        start_time = time.perf_counter()

        for _ in range(iterations):
            result = EvaluationAdapter.domain_to_result(evaluation)
            assert result is not None

        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_conversion = total_time / iterations

        print(f"\nEvaluation Adapter Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Average per conversion: {avg_time_per_conversion:.3f} ms")

        assert avg_time_per_conversion < 5  # Should be under 5ms per conversion

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, judge_agent):
        """Test performance under sustained load over time"""
        duration_seconds = 30
        evaluations_completed = 0
        start_time = time.perf_counter()

        with patch.object(judge_agent, '_process_rubrics'):
            while (time.perf_counter() - start_time) < duration_seconds:
                request = EvaluationRequest(
                    analysis_id=f"sustained_{evaluations_completed}",
                    agent_id="sustained_agent",
                    analysis_content=f"Sustained load test {evaluations_completed}",
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                await judge_agent.evaluate(request)
                evaluations_completed += 1

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        evaluations_per_second = evaluations_completed / actual_duration

        print(f"\nSustained Load Performance ({duration_seconds}s):")
        print(f"  Evaluations completed: {evaluations_completed}")
        print(f"  Actual duration: {actual_duration:.2f} seconds")
        print(f"  Evaluations per second: {evaluations_per_second:.2f}")

        # Check for performance degradation
        expected_min_rate = 5  # Minimum expected evaluations per second
        assert evaluations_per_second >= expected_min_rate


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_benchmark_evaluation_throughput(self):
        """Benchmark evaluation throughput"""
        judge_agent = JudgeAgent(agent_id="benchmark_test")

        # Test with different payload sizes
        payload_sizes = [100, 1000, 10000]  # Characters
        results = {}

        with patch.object(judge_agent, '_process_rubrics'):
            for size in payload_sizes:
                content = "X" * size

                request = EvaluationRequest(
                    analysis_id=f"benchmark_{size}",
                    agent_id="benchmark_agent",
                    analysis_content=content,
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                # Measure 100 evaluations
                start_time = time.perf_counter()
                for i in range(100):
                    await judge_agent.evaluate(request)
                end_time = time.perf_counter()

                total_time = end_time - start_time
                throughput = 100 / total_time

                results[size] = {
                    "throughput": throughput,
                    "time_per_eval": (total_time / 100) * 1000
                }

        print("\nThroughput Benchmark:")
        for size, metrics in results.items():
            print(f"  Payload {size} chars:")
            print(f"    Throughput: {metrics['throughput']:.2f} eval/sec")
            print(f"    Time per eval: {metrics['time_per_eval']:.2f} ms")

    @pytest.mark.benchmark
    def test_memory_footprint_benchmark(self):
        """Benchmark memory footprint"""
        import tracemalloc

        # Start tracking memory
        tracemalloc.start()

        # Create multiple evaluations
        evaluations = []
        for i in range(1000):
            eval = Evaluation(
                evaluation_id=f"mem_bench_{i}",
                analysis_id=f"analysis_{i}",
                agent_id=f"agent_{i}"
            )
            evaluations.append(eval)

        # Take snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("\nMemory Footprint Benchmark:")
        print(f"  Evaluations created: {len(evaluations)}")
        print(f"  Total allocated memory: {sum(stat.size for stat in top_stats) / 1024:.1f} KB")

        # Stop tracking
        tracemalloc.stop()