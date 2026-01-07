# infrastructure/metrics/capability_metrics.py
from collections import defaultdict


class CapabilityMetrics:
    """
    Simple in-memory metrics collector.
    """

    def __init__(self):
        self.calls = defaultdict(int)
        self.failures = defaultdict(int)
        self.latency = defaultdict(float)

    def record(self, name: str, duration: float, success: bool):
        self.calls[name] += 1
        self.latency[name] += duration
        if not success:
            self.failures[name] += 1
