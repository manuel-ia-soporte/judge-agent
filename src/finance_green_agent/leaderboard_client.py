"""HTTP client for sending evaluation results to the leaderboard webhook."""

from __future__ import annotations

import os
from typing import Any

import aiohttp
import backoff


LEADERBOARD_WEBHOOK_URL = os.environ.get(
    "LEADERBOARD_WEBHOOK_URL",
    "https://api.github.com/repos/manuel-ia-soporte/leaderboard/dispatches",
)
LEADERBOARD_WEBHOOK_TOKEN = os.environ.get("LEADERBOARD_WEBHOOK_TOKEN", "")


class LeaderboardClient:
    def __init__(
        self,
        webhook_url: str | None = None,
        webhook_token: str | None = None,
        timeout: float = 30.0,
    ):
        self.webhook_url = webhook_url or LEADERBOARD_WEBHOOK_URL
        self.webhook_token = webhook_token or LEADERBOARD_WEBHOOK_TOKEN
        self.timeout = timeout

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
        }
        if self.webhook_token:
            headers["Authorization"] = f"Bearer {self.webhook_token}"
        return headers

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, aiohttp.ServerTimeoutError),
        max_tries=3,
        max_time=60,
    )
    async def send_results(
        self,
        evaluation_result: dict[str, Any],
        event_type: str = "evaluation-complete",
    ) -> dict[str, Any]:
        """
        Send evaluation results to the leaderboard webhook.
        
        Uses GitHub repository dispatch to trigger the leaderboard workflow.
        """
        payload = {
            "event_type": event_type,
            "client_payload": {
                "results": evaluation_result,
                "winner": evaluation_result.get("winner"),
                "participants": list(evaluation_result.get("participants", {}).keys()),
                "max_questions": evaluation_result.get("max_questions"),
                "seed": evaluation_result.get("seed"),
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout,
            ) as response:
                if response.status == 204:
                    return {"success": True, "status": 204, "message": "Dispatch accepted"}
                
                try:
                    body = await response.json()
                except Exception:
                    body = await response.text()

                if response.status >= 400:
                    return {
                        "success": False,
                        "status": response.status,
                        "error": body,
                    }

                return {
                    "success": True,
                    "status": response.status,
                    "response": body,
                }

    async def notify_evaluation_complete(
        self,
        task_id: str,
        evaluation_result: dict[str, Any],
        callback_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Notify via push notification / webhook when evaluation completes.
        
        If callback_url is provided, sends directly to that URL.
        Otherwise, uses the default leaderboard webhook.
        """
        target_url = callback_url or self.webhook_url
        
        payload = {
            "task_id": task_id,
            "event": "evaluation_complete",
            "result": evaluation_result,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                target_url,
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout,
            ) as response:
                try:
                    body = await response.json()
                except Exception:
                    body = await response.text()

                return {
                    "success": response.status < 400,
                    "status": response.status,
                    "response": body if response.status < 400 else None,
                    "error": body if response.status >= 400 else None,
                }


_default_client: LeaderboardClient | None = None


def get_leaderboard_client() -> LeaderboardClient:
    global _default_client
    if _default_client is None:
        _default_client = LeaderboardClient()
    return _default_client
