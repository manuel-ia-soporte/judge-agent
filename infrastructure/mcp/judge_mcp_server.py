# infrastructure/mcp/judge_mcp_server.py
from agents.judge_agent.judge_agent import JudgeAgent


class JudgeMCPServer:
    def __init__(self, judge_agent: JudgeAgent) -> None:
        self._judge = judge_agent

    def evaluate(self, signals: dict):
        return self._judge.judge(signals)
