"""Regression tests for dict-based tool_calls.

Providers may return tool_calls as plain dicts instead of ToolCall Pydantic objects.
All attribute accesses (.id, .name, .arguments) must work regardless of whether
tool_calls are objects or dicts.

Regression test for: 16 unsafe bare attribute accesses on tool_call objects.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from amplifier_module_loop_events import EventDrivenOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hooks():
    """Create a mock HookRegistry that records emit calls."""
    hooks = AsyncMock()
    result = MagicMock()
    result.action = "continue"
    result.ephemeral = False
    result.context_injection = None
    result.data = None
    hooks.emit.return_value = result
    return hooks


def _make_context():
    """Create a mock context that captures add_message calls."""
    context = AsyncMock()
    messages_added = []

    async def capture_add_message(msg):
        messages_added.append(msg)

    context.add_message = AsyncMock(side_effect=capture_add_message)
    context.get_messages_for_request = AsyncMock(
        return_value=[{"role": "user", "content": "test"}]
    )
    context.get_messages = AsyncMock(return_value=[{"role": "user", "content": "test"}])
    return context, messages_added


def _make_tool_result(output="ok", success=True):
    """Create a mock ToolResult."""
    result = MagicMock()
    result.success = success
    result.output = output
    result.error = None

    def get_serialized_output():
        if isinstance(output, (dict, list)):
            return json.dumps(output)
        return str(output)

    result.get_serialized_output = get_serialized_output

    def model_dump():
        return {"success": success, "output": output, "error": None}

    result.model_dump = model_dump
    return result


def _make_provider_with_dict_tool_calls(dict_tool_calls, text_response="Done"):
    """Create a mock provider that returns dict-based tool_calls, then text.

    This simulates a provider that returns plain dicts instead of ToolCall objects.
    """
    # First response: triggers tool calls
    tool_response = MagicMock()
    tool_response.content = "Using tool"
    tool_response.metadata = None

    # Second response: text (ends the loop)
    text_block = MagicMock()
    text_block.text = text_response
    text_block.type = "text"
    final_response = MagicMock()
    final_response.content = [text_block]
    final_response.metadata = None

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=[tool_response, final_response])
    provider.parse_tool_calls = MagicMock(side_effect=[dict_tool_calls, []])
    return provider


# ---------------------------------------------------------------------------
# Tests: dict-based tool_calls must not crash
# ---------------------------------------------------------------------------


class TestDictToolCalls:
    """Tool calls returned as plain dicts must work identically to objects."""

    @pytest.mark.asyncio
    async def test_dict_tool_call_with_name_key(self):
        """Dict tool_call using 'name' key for tool name works."""
        dict_tool_calls = [
            {"id": "tc_1", "name": "test_tool", "arguments": {"key": "value"}}
        ]

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(return_value=_make_tool_result("tool output"))
        tools = {"test_tool": mock_tool}

        hooks = _make_hooks()
        context, messages_added = _make_context()
        provider = _make_provider_with_dict_tool_calls(dict_tool_calls)

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        result = await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools=tools,
            hooks=hooks,
        )

        # Should complete without AttributeError
        assert result == "Done"

        # Tool should have been executed
        mock_tool.execute.assert_called_once_with({"key": "value"})

        # Verify tool result was added to context
        tool_msgs = [m for m in messages_added if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc_1"
        assert tool_msgs[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_dict_tool_call_with_tool_key(self):
        """Dict tool_call using 'tool' key (alternate provider format) works."""
        dict_tool_calls = [{"id": "tc_2", "tool": "test_tool", "arguments": {"x": 1}}]

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(return_value=_make_tool_result("result"))
        tools = {"test_tool": mock_tool}

        hooks = _make_hooks()
        context, messages_added = _make_context()
        provider = _make_provider_with_dict_tool_calls(dict_tool_calls)

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        result = await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools=tools,
            hooks=hooks,
        )

        assert result == "Done"
        mock_tool.execute.assert_called_once_with({"x": 1})

    @pytest.mark.asyncio
    async def test_dict_tool_call_assistant_message_construction(self):
        """Dict tool_calls are correctly serialized in assistant message."""
        dict_tool_calls = [
            {"id": "tc_1", "name": "tool_a", "arguments": {"a": 1}},
            {"id": "tc_2", "name": "tool_b", "arguments": {"b": 2}},
        ]

        # Create tools for both
        def make_tool(name):
            t = MagicMock()
            t.name = name
            t.description = f"Tool {name}"
            t.input_schema = {"type": "object", "properties": {}}
            t.execute = AsyncMock(return_value=_make_tool_result(f"{name} output"))
            return t

        tools = {"tool_a": make_tool("tool_a"), "tool_b": make_tool("tool_b")}

        hooks = _make_hooks()
        context, messages_added = _make_context()
        provider = _make_provider_with_dict_tool_calls(dict_tool_calls)

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools=tools,
            hooks=hooks,
        )

        # Find the assistant message with tool_calls
        assistant_msgs = [
            m
            for m in messages_added
            if m.get("role") == "assistant" and "tool_calls" in m
        ]
        assert len(assistant_msgs) == 1

        tc_data = assistant_msgs[0]["tool_calls"]
        assert len(tc_data) == 2
        assert tc_data[0]["id"] == "tc_1"
        assert tc_data[0]["tool"] == "tool_a"
        assert tc_data[1]["id"] == "tc_2"
        assert tc_data[1]["tool"] == "tool_b"

    @pytest.mark.asyncio
    async def test_dict_tool_call_missing_arguments_defaults_to_empty(self):
        """Dict tool_call with no 'arguments' key defaults to empty dict."""
        dict_tool_calls = [
            {"id": "tc_1", "name": "test_tool"}  # no "arguments" key
        ]

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(return_value=_make_tool_result("ok"))
        tools = {"test_tool": mock_tool}

        hooks = _make_hooks()
        context, messages_added = _make_context()
        provider = _make_provider_with_dict_tool_calls(dict_tool_calls)

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools=tools,
            hooks=hooks,
        )

        # Should have called execute with empty dict (the default)
        mock_tool.execute.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_dict_tool_call_in_error_handler(self):
        """Dict tool_calls don't crash in error handlers (highest risk site)."""
        dict_tool_calls = [{"id": "tc_err", "name": "failing_tool", "arguments": {}}]

        # Tool that doesn't exist - forces the "tool not found" path
        # which accesses tool_call.id in the error handler
        hooks = _make_hooks()
        context, messages_added = _make_context()
        provider = _make_provider_with_dict_tool_calls(dict_tool_calls)

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        result = await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools={},  # No tools registered - triggers "not found" path
            hooks=hooks,
        )

        # Should complete without crashing
        assert result == "Done"

        # Should have added error tool message
        tool_msgs = [m for m in messages_added if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "not found" in tool_msgs[0]["content"]
        assert tool_msgs[0]["tool_call_id"] == "tc_err"

    @pytest.mark.asyncio
    async def test_dict_tool_call_tool_execution_error(self):
        """Dict tool_calls work when tool.execute raises an exception."""
        dict_tool_calls = [
            {"id": "tc_crash", "name": "crashing_tool", "arguments": {"x": 1}}
        ]

        mock_tool = MagicMock()
        mock_tool.name = "crashing_tool"
        mock_tool.description = "A tool that crashes"
        mock_tool.input_schema = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("tool exploded"))
        tools = {"crashing_tool": mock_tool}

        hooks = _make_hooks()
        context, messages_added = _make_context()
        provider = _make_provider_with_dict_tool_calls(dict_tool_calls)

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        result = await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools=tools,
            hooks=hooks,
        )

        # Should complete without crashing (error is caught internally)
        assert result == "Done"

        # Tool result should contain the error message
        tool_msgs = [m for m in messages_added if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc_crash"

    @pytest.mark.asyncio
    async def test_object_tool_calls_still_work(self):
        """Object-based tool_calls (MagicMock with attrs) still work after fix."""
        # This is the existing behavior - ensure we don't break it
        tool_call = MagicMock()
        tool_call.id = "tc_obj"
        tool_call.name = "test_tool"
        tool_call.arguments = {"key": "value"}

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(return_value=_make_tool_result("ok"))
        tools = {"test_tool": mock_tool}

        hooks = _make_hooks()
        context, messages_added = _make_context()

        # Use the same provider pattern but with object tool_calls
        tool_response = MagicMock()
        tool_response.content = "Using tool"
        tool_response.metadata = None

        text_block = MagicMock()
        text_block.text = "Done"
        text_block.type = "text"
        final_response = MagicMock()
        final_response.content = [text_block]
        final_response.metadata = None

        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=[tool_response, final_response])
        provider.parse_tool_calls = MagicMock(side_effect=[[tool_call], []])

        orch = EventDrivenOrchestrator({"max_iterations": 5})
        result = await orch.execute(
            prompt="test",
            context=context,
            providers={"default": provider},
            tools=tools,
            hooks=hooks,
        )

        assert result == "Done"
        mock_tool.execute.assert_called_once_with({"key": "value"})
