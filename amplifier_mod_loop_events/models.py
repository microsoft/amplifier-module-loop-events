"""
Decision event models for the loop-events orchestration module.
These models handle scheduler decisions for tools, agents, and context management.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionRequest:
    """Base class for decision request events."""

    event_id: str
    timestamp: float
    session_id: str
    metadata: dict[str, Any]


@dataclass
class ToolResolutionRequest(DecisionRequest):
    """Request for tool selection decision."""

    available_tools: list[str]
    context: dict[str, Any]


@dataclass
class ToolResolutionResponse:
    """Response from scheduler with tool selection."""

    selected_tool: str
    score: float
    rationale: str
    metadata: dict[str, Any]


@dataclass
class AgentResolutionRequest(DecisionRequest):
    """Request for agent selection decision."""

    available_agents: list[str]
    task: str
    context: dict[str, Any]


@dataclass
class AgentResolutionResponse:
    """Response from scheduler with agent selection."""

    selected_agent: str
    score: float
    rationale: str
    metadata: dict[str, Any]


@dataclass
class ContextResolutionRequest(DecisionRequest):
    """Request for context management decision."""

    context_size: int
    max_size: int
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class ContextResolutionResponse:
    """Response from scheduler with context management decision."""

    should_compact: bool
    compaction_strategy: str | None
    rationale: str
    metadata: dict[str, Any]


@dataclass
class ErrorEvent:
    """Structured error event for telemetry."""

    error_type: str  # "tool_failure", "timeout", "validation", etc.
    error_code: str | None
    error_message: str
    stack_trace: str | None
    recovery_attempted: bool
    recovery_successful: bool
    fallback_used: str | None
    severity: str  # "low", "medium", "high", "critical"
    metadata: dict[str, Any]
