"""
HTTP adapter for the Student State tracking service (port 8885).

The orchestrator uses this adapter in two ways:
  1. get_context()  — called before each LLM invocation; returns a compact
                      student-state string that is prepended to the system prompt.
  2. record(...)    — called after each LLM response; fires-and-forgets the
                      interaction data returned by the LLM's record_interaction tool.

Both operations fail silently (logged at DEBUG) so a missing or restarting
student-state service never blocks the tutor.
"""

import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class StudentStateAdapter:
    """Thin HTTP client for the YLIP student-history service."""

    async def get_context(self) -> str:
        """Return formatted student-state text for system-prompt injection.

        Returns an empty string when the service is unavailable or has no data.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.student_state_base_url}/context")
                resp.raise_for_status()
                return resp.json().get("context", "")
        except Exception as exc:
            logger.debug("student-state /context unavailable: %s", exc)
            return ""

    async def record(
        self,
        session_id: str,
        topic: str,
        mastery_signal: int,
        approach: str,
        notes: str = "",
    ) -> None:
        """Post one interaction record. Non-blocking — errors are silently logged."""
        payload = {
            "session_id":     session_id,
            "topic":          topic,
            "mastery_signal": mastery_signal,
            "approach":       approach,
            "notes":          notes[:200],
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{settings.student_state_base_url}/record", json=payload
                )
        except Exception as exc:
            logger.debug("student-state /record failed: %s", exc)
