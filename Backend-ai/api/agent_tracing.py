"""In-memory, failure-isolated tracing for deterministic CaseLens agent runs."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from api.agent import NextAction, PhaseName


AgentEventType = Literal[
    "agent_started",
    "routing_decision",
    "question_requested",
    "user_answer_applied",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "rerun_started",
    "rerun_completed",
    "human_handoff",
    "agent_completed",
]


class AgentTraceEvent(BaseModel):
    trace_id: uuid.UUID
    timestamp: datetime
    event_type: AgentEventType
    current_phase: PhaseName | None = None
    next_action: NextAction | None = None
    tool_name: str | None = None
    success: bool | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    trace_id: uuid.UUID
    started_at: datetime
    events: list[AgentTraceEvent] = Field(default_factory=list)


class TraceCollector:
    def __init__(self):
        self.trace = self._new_trace()

    @staticmethod
    def _new_trace() -> AgentTrace:
        return AgentTrace(trace_id=uuid.uuid4(), started_at=datetime.now(timezone.utc))

    def start_trace(self) -> AgentTrace:
        self.trace = self._new_trace()
        return self.trace

    def emit(
        self,
        event_type: AgentEventType,
        *,
        current_phase: PhaseName | None = None,
        next_action: NextAction | None = None,
        tool_name: str | None = None,
        success: bool | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.trace.events.append(AgentTraceEvent(
                trace_id=self.trace.trace_id,
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                current_phase=current_phase,
                next_action=next_action,
                tool_name=tool_name,
                success=success,
                duration_ms=duration_ms,
                error=self.sanitize_error(error),
                metadata=metadata or {},
            ))
        except Exception:
            return

    def snapshot(self) -> AgentTrace:
        return self.trace.model_copy(deep=True)

    @staticmethod
    def sanitize_error(error: str | None) -> str | None:
        if error is None:
            return None
        sanitized = re.sub(
            r"(?i)(authorization|api[_-]?key)\s*[:=]\s*[^\s,;]+",
            r"\1=[REDACTED]",
            str(error),
        )
        sanitized = re.sub(r"(?i)bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", sanitized)
        return sanitized[:500]
