"""Phase 2 tests for loop-events orchestrator.

Tests:
- reasoning_effort propagated to ChatRequest
- PROVIDER_ERROR constant used (not "error:provider")
- LLMError events include retryable and status_code
- Generic Exception events don't include retryable
- extended_thinking passed to provider.complete()
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.events import PROVIDER_ERROR
from amplifier_core.llm_errors import LLMError

from amplifier_module_loop_events import EventDrivenOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hooks() -> AsyncMock:
    """Create a mock HookRegistry that records emit calls."""
    hooks = AsyncMock()
    # emit returns a HookResult-like object with action="continue"
    result = MagicMock()
    result.action = "continue"
    result.ephemeral = False
    result.context_injection = None
    hooks.emit.return_value = result
    return hooks


def _make_context(messages: list[dict] | None = None) -> AsyncMock:
    """Create a mock context manager."""
    ctx = AsyncMock()
    ctx.get_messages_for_request = AsyncMock(
        return_value=messages or [{"role": "user", "content": "hello"}]
    )
    ctx.get_messages = AsyncMock(
        return_value=messages or [{"role": "user", "content": "hello"}]
    )
    ctx.add_message = AsyncMock()
    return ctx


def _make_provider(content: str = "response text", tool_calls: Any = None) -> MagicMock:
    """Create a mock provider that returns a simple text response."""
    provider = MagicMock()
    response = MagicMock()
    response.content = content
    response.metadata = None
    provider.complete = AsyncMock(return_value=response)
    provider.parse_tool_calls = MagicMock(return_value=tool_calls or [])
    return provider


# ---------------------------------------------------------------------------
# 1. reasoning_effort on ChatRequest
# ---------------------------------------------------------------------------


class TestReasoningEffort:
    """ChatRequest must receive reasoning_effort from config."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_propagated_to_main_loop(self):
        """Main-loop ChatRequest gets reasoning_effort from config."""
        config = {"reasoning_effort": "low"}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()
        provider = _make_provider()

        await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        # provider.complete was called; inspect the ChatRequest arg
        call_args = provider.complete.call_args
        chat_request = call_args[0][0]
        assert chat_request.reasoning_effort == "low"

    @pytest.mark.asyncio
    async def test_reasoning_effort_none_when_absent(self):
        """reasoning_effort is None when not in config."""
        config = {}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()
        provider = _make_provider()

        await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        chat_request = provider.complete.call_args[0][0]
        assert chat_request.reasoning_effort is None

    @pytest.mark.asyncio
    async def test_reasoning_effort_on_max_iterations_fallback(self):
        """Max-iterations fallback ChatRequest also gets reasoning_effort."""
        config = {"max_iterations": 1, "reasoning_effort": "high"}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()

        # First call returns tool_calls to consume the single iteration,
        # second call (fallback) returns text.
        tool_call = MagicMock()
        tool_call.name = "some_tool"
        tool_call.arguments = {}
        tool_call.id = "tc1"

        tool = MagicMock()
        tool.name = "some_tool"
        tool.description = "desc"
        tool.input_schema = {}
        result = MagicMock()
        result.success = True
        result.get_serialized_output = MagicMock(return_value="ok")
        result.model_dump = MagicMock(return_value={"output": "ok"})
        tool.execute = AsyncMock(return_value=result)

        response_with_tools = MagicMock()
        response_with_tools.content = ""
        response_with_tools.metadata = None

        response_text = MagicMock()
        response_text.content = "final answer"
        response_text.metadata = None

        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=[response_with_tools, response_text])
        provider.parse_tool_calls = MagicMock(side_effect=[[tool_call], []])

        await orch.execute("hi", ctx, {"default": provider}, {"some_tool": tool}, hooks)

        # The second complete() call is the fallback
        assert provider.complete.call_count == 2
        fallback_request = provider.complete.call_args_list[1][0][0]
        assert fallback_request.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# 2. PROVIDER_ERROR event name (bug fix)
# ---------------------------------------------------------------------------


class TestProviderErrorEventName:
    """Provider errors must emit PROVIDER_ERROR ("provider:error"), not "error:provider"."""

    @pytest.mark.asyncio
    async def test_generic_exception_emits_provider_error_constant(self):
        """A generic Exception from provider.complete emits PROVIDER_ERROR."""
        config = {}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()

        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("boom"))

        result = await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        # Should have emitted PROVIDER_ERROR (= "provider:error")
        emit_calls = hooks.emit.call_args_list
        error_events = [c for c in emit_calls if c[0][0] == PROVIDER_ERROR]
        assert len(error_events) == 1, (
            f"Expected exactly 1 PROVIDER_ERROR emit, got {len(error_events)}. "
            f"All emitted events: {[c[0][0] for c in emit_calls]}"
        )

        # Verify the old wrong name was NOT used
        wrong_events = [c for c in emit_calls if c[0][0] == "error:provider"]
        assert len(wrong_events) == 0, (
            "Bug: 'error:provider' was emitted instead of PROVIDER_ERROR"
        )

        assert "Error getting response" in result


# ---------------------------------------------------------------------------
# 3. LLMError events include retryable and status_code
# ---------------------------------------------------------------------------


class TestLLMErrorEnrichment:
    """LLMError must produce enriched event data with retryable and status_code."""

    @pytest.mark.asyncio
    async def test_llm_error_event_includes_retryable_and_status_code(self):
        """LLMError from provider.complete emits enriched PROVIDER_ERROR."""
        config = {}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()

        provider = MagicMock()
        provider.complete = AsyncMock(
            side_effect=LLMError(
                "rate limited",
                provider="anthropic",
                status_code=429,
                retryable=True,
            )
        )

        result = await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        emit_calls = hooks.emit.call_args_list
        error_events = [c for c in emit_calls if c[0][0] == PROVIDER_ERROR]
        assert len(error_events) == 1

        event_data = error_events[0][0][1]
        assert event_data["retryable"] is True
        assert event_data["status_code"] == 429
        assert event_data["error_type"] == "completion_failed"
        assert "rate limited" in event_data["error_message"]

        # loop-events stores error as response text (doesn't re-raise)
        assert "Error getting response" in result

    @pytest.mark.asyncio
    async def test_llm_error_in_max_iterations_fallback(self):
        """LLMError in max-iterations fallback also emits enriched event."""
        config = {"max_iterations": 1}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()

        # First call: return tool calls to use up the iteration
        tool_call = MagicMock()
        tool_call.name = "some_tool"
        tool_call.arguments = {}
        tool_call.id = "tc1"

        tool = MagicMock()
        tool.name = "some_tool"
        tool.description = "desc"
        tool.input_schema = {}
        result_mock = MagicMock()
        result_mock.success = True
        result_mock.get_serialized_output = MagicMock(return_value="ok")
        result_mock.model_dump = MagicMock(return_value={"output": "ok"})
        tool.execute = AsyncMock(return_value=result_mock)

        response_with_tools = MagicMock()
        response_with_tools.content = ""
        response_with_tools.metadata = None

        provider = MagicMock()
        provider.complete = AsyncMock(
            side_effect=[
                response_with_tools,  # main loop
                LLMError(
                    "server error", provider="openai", status_code=500, retryable=True
                ),  # fallback
            ]
        )
        provider.parse_tool_calls = MagicMock(side_effect=[[tool_call], []])

        await orch.execute("hi", ctx, {"default": provider}, {"some_tool": tool}, hooks)

        emit_calls = hooks.emit.call_args_list
        error_events = [c for c in emit_calls if c[0][0] == PROVIDER_ERROR]
        assert len(error_events) == 1

        event_data = error_events[0][0][1]
        assert event_data["retryable"] is True
        assert event_data["status_code"] == 500


# ---------------------------------------------------------------------------
# 4. Generic Exception events don't include retryable
# ---------------------------------------------------------------------------


class TestGenericExceptionEvent:
    """Generic Exception events must NOT include retryable or status_code."""

    @pytest.mark.asyncio
    async def test_generic_exception_no_retryable_field(self):
        """A plain Exception should emit event without retryable/status_code."""
        config = {}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()

        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=ValueError("bad input"))

        await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        emit_calls = hooks.emit.call_args_list
        error_events = [c for c in emit_calls if c[0][0] == PROVIDER_ERROR]
        assert len(error_events) == 1

        event_data = error_events[0][0][1]
        assert "retryable" not in event_data
        assert "status_code" not in event_data
        assert event_data["severity"] == "high"
        assert "bad input" in event_data["error_message"]


# ---------------------------------------------------------------------------
# 5. extended_thinking support
# ---------------------------------------------------------------------------


class TestExtendedThinking:
    """extended_thinking config should be forwarded to provider.complete()."""

    @pytest.mark.asyncio
    async def test_extended_thinking_passed_to_provider(self):
        """When extended_thinking=True, provider.complete gets the kwarg."""
        config = {"extended_thinking": True}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()
        provider = _make_provider()

        await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        call_kwargs = provider.complete.call_args[1]
        assert call_kwargs.get("extended_thinking") is True

    @pytest.mark.asyncio
    async def test_no_extended_thinking_by_default(self):
        """When extended_thinking not in config, kwarg is absent."""
        config = {}
        orch = EventDrivenOrchestrator(config)
        hooks = _make_hooks()
        ctx = _make_context()
        provider = _make_provider()

        await orch.execute("hi", ctx, {"default": provider}, {}, hooks)

        call_kwargs = provider.complete.call_args[1]
        assert "extended_thinking" not in call_kwargs
