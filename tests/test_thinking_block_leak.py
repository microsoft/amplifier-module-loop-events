"""Regression tests for thinking block content leaking into text extraction.

Root cause:
    There are TWO content block model systems:
      - content_models.ThinkingContent  → has .text  (was the bug hazard)
      - message_models.ThinkingBlock    → has .thinking (already safe)

    The old execute() inline text extraction used ``hasattr(block, "text")``
    which allowed ThinkingContent objects through unchanged, leaking thinking
    text into final_response at TWO locations:
      - Location 1: main loop no-tool-calls response path (~line 268)
      - Location 2: max-iterations fallback path (~line 621)

    Because this module has two identical patterns, the fix extracts a shared
    ``_extract_text_from_content`` helper (module-level, matching the existing
    ``_normalize_tool_call`` convention) and calls it from both sites.

Fix:
    Use an explicit ``block.type == "text"`` guard so only text blocks are
    included, regardless of which model system the block comes from.

RED / GREEN verification:
    Run against the unfixed code to see test_thinking_content_does_not_leak
    and test_thinking_content_filtered_at_max_iterations FAIL (thinking text
    present in result).
    Run after the fix to see all tests PASS.

Cross-ecosystem:
    Same fix pattern as amplifier-module-loop-streaming PR #25 (df5c0e1)
    and amplifier-module-loop-basic PR #11 (fa67fe2).
"""

from types import SimpleNamespace

import pytest

from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_events import EventDrivenOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orch(config=None):
    return EventDrivenOrchestrator(config or {})


def _make_fake_response(blocks):
    """Create a fake provider response with the given content blocks."""

    class FakeResponse:
        content = blocks
        tool_calls = None
        usage = None
        content_blocks = None
        metadata = None

    return FakeResponse()


class FakeProvider:
    """Mock provider that returns a pre-configured response."""

    name = "mock-thinking"

    def __init__(self, response):
        self._response = response

    async def complete(self, request, **kwargs):
        return self._response

    def parse_tool_calls(self, response):
        return []


class FakeProviderWithToolsThenText:
    """Mock provider that returns tool calls once, then a text response.

    Used to drive the max-iterations fallback path: set max_iterations=1,
    return a tool call on the first request (so final_response stays empty
    after the first iteration), then return a content response on the second
    call (the max-iterations fallback provider.complete call).
    """

    def __init__(self, tool_call_name, final_response_blocks):
        self._call_count = 0
        self._tool_call_name = tool_call_name
        self._final_response_blocks = final_response_blocks

    async def complete(self, request, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            # First call: return a response that has tool calls
            return _make_fake_response([])
        else:
            # Second call (max-iterations fallback): return content blocks
            return _make_fake_response(self._final_response_blocks)

    def parse_tool_calls(self, response):
        if self._call_count == 1:
            # Return a tool call for the first response
            return [
                SimpleNamespace(
                    id="tc_001",
                    name=self._tool_call_name,
                    arguments={},
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Location 1 tests: ThinkingContent at main loop no-tool-calls path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_content_does_not_leak_into_final_response():
    """content_models.ThinkingContent has .text — must be filtered by type check.

    PRIMARY REGRESSION TEST for Location 1 (main loop path).

    Before the fix, the hasattr(block, "text") guard would include
    ThinkingContent blocks because they *do* have a .text attribute, just
    with type="thinking".

    RED (before fix):  result contains "internal reasoning" — thinking text leaked.
    GREEN (after fix): result is ONLY "real response".
    """
    from amplifier_core.content_models import TextContent, ThinkingContent

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response(
        [
            ThinkingContent(text="internal reasoning"),
            TextContent(text="real response"),
        ]
    )
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    # Thinking text must NOT appear in the final response
    assert "internal reasoning" not in result, (
        f"Thinking text leaked into final_response: {result!r}"
    )
    # Only the TextContent payload should be present
    assert "real response" in result, (
        f"Expected 'real response' in result but got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Location 2 tests: ThinkingContent at max-iterations fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_content_filtered_at_max_iterations_fallback():
    """ThinkingContent must be filtered at the max-iterations fallback path.

    PRIMARY REGRESSION TEST for Location 2.

    Drives through the max-iterations code path by:
      1. Setting max_iterations=1
      2. Provider returns a tool call on iteration 1 (loop exits without
         setting final_response)
      3. Provider returns ThinkingContent + TextContent on the fallback call

    RED (before fix):  result contains "thinking at max iter" — leaked.
    GREEN (after fix): result is ONLY "max iter real response".
    """
    from amplifier_core.content_models import TextContent, ThinkingContent

    orchestrator = _orch({"max_iterations": 1})
    context = MockContextManager()
    hooks = EventRecorder()

    provider = FakeProviderWithToolsThenText(
        tool_call_name="nonexistent_tool",
        final_response_blocks=[
            ThinkingContent(text="thinking at max iter"),
            TextContent(text="max iter real response"),
        ],
    )

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},  # no tools registered → tool "not found" → loop exits cleanly
        hooks=hooks,
    )

    assert "thinking at max iter" not in result, (
        f"Thinking text leaked into final_response at max-iterations: {result!r}"
    )
    assert "max iter real response" in result, (
        f"Expected 'max iter real response' in result but got: {result!r}"
    )


@pytest.mark.asyncio
async def test_helper_directly_filters_thinking_content():
    """Direct test of _extract_text_from_content helper (module-level function).

    Verifies the shared helper filters ThinkingContent and passes TextContent
    through, independent of which code path calls it.
    """
    from amplifier_core.content_models import TextContent, ThinkingContent

    from amplifier_module_loop_events import _extract_text_from_content

    content = [
        ThinkingContent(text="should be filtered"),
        TextContent(text="should pass through"),
    ]

    result = _extract_text_from_content(content)

    assert "should be filtered" not in result, (
        f"ThinkingContent leaked via helper: {result!r}"
    )
    assert "should pass through" in result, (
        f"Expected 'should pass through' from helper but got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Regression guards: TextBlock and TextContent pass through correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_content_passes_through():
    """content_models.TextContent is included in final_response as expected."""
    from amplifier_core.content_models import TextContent

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response([TextContent(text="hello from TextContent")])
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "hello from TextContent", f"Unexpected result: {result!r}"


@pytest.mark.asyncio
async def test_text_block_passes_through():
    """message_models.TextBlock is included in final_response as expected."""
    from amplifier_core.message_models import TextBlock

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response([TextBlock(text="hello from TextBlock")])
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "hello from TextBlock", f"Unexpected result: {result!r}"


# ---------------------------------------------------------------------------
# Complementary test: ThinkingBlock (message_models) was already safe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_block_does_not_leak_into_final_response():
    """message_models.ThinkingBlock has .thinking (not .text) — already safe.

    ThinkingBlock was not affected by the original bug (no .text attribute),
    but this test documents that it remains excluded after the type-check refactor.
    """
    from amplifier_core.message_models import TextBlock, ThinkingBlock

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response(
        [
            ThinkingBlock(thinking="internal reasoning", signature="sig"),
            TextBlock(text="real response"),
        ]
    )
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert "internal reasoning" not in result, (
        f"ThinkingBlock content leaked into final_response: {result!r}"
    )
    assert "real response" in result, (
        f"Expected 'real response' in result but got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Dict block tests: type-aware filtering applies to dict blocks too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dict_text_block_passes_through():
    """Dict blocks with type='text' pass through correctly."""
    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response([{"type": "text", "text": "dict text block"}])
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "dict text block", f"Unexpected result: {result!r}"


@pytest.mark.asyncio
async def test_dict_thinking_block_is_filtered():
    """Dict blocks with type='thinking' are filtered out (consistency fix)."""
    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response(
        [
            {"type": "thinking", "text": "dict thinking block"},
            {"type": "text", "text": "dict real response"},
        ]
    )
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert "dict thinking block" not in result, (
        f"Dict thinking block leaked into final_response: {result!r}"
    )
    assert "dict real response" in result, (
        f"Expected 'dict real response' in result but got: {result!r}"
    )
