"""Tests for hook modify action on tool:post events in event-driven orchestrator.

Verifies that when a hook returns HookResult(action="modify", data={"result": ...})
on a tool:post event, the orchestrator uses the modified data in the context message
instead of the original result.get_serialized_output().
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_core.hooks import HookRegistry
from amplifier_core.models import HookResult


def _make_tool_result(output, success=True):
    """Create a mock tool result with get_serialized_output() and model_dump()."""
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


def _make_provider_responses(tool_calls_list, text_response_str="Done"):
    """Create mock provider that returns tool calls then text."""
    # First response: tool calls
    tool_response = MagicMock()
    tool_response.content = [MagicMock(type="text", text="Using tool")]
    tool_response.metadata = None

    # Second response: text (ends the loop)
    text_block = MagicMock()
    text_block.text = text_response_str
    text_block.type = "text"
    text_response = MagicMock()
    text_response.content = [text_block]
    text_response.metadata = None

    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=[tool_response, text_response])
    mock_provider.parse_tool_calls = MagicMock(side_effect=[tool_calls_list, []])
    mock_provider.priority = 1

    return mock_provider


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
    return context, messages_added


def _get_tool_result_messages(messages_added):
    """Extract tool result messages from captured messages."""
    return [msg for msg in messages_added if msg.get("role") == "tool"]


@pytest.mark.asyncio
async def test_tool_post_modify_replaces_context_message():
    """When hook returns modify on tool:post, context message uses modified data."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_events import EventDrivenOrchestrator

    orchestrator = EventDrivenOrchestrator({"max_iterations": 5})

    # Tool with original output
    original_output = {"original": True, "big_data": "x" * 1000}
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.input_schema = {"type": "object", "properties": {}}
    mock_tool.execute = AsyncMock(return_value=_make_tool_result(original_output))
    tools = {"test_tool": mock_tool}

    # Hook that modifies tool output
    modified_content = {"modified": True, "truncated": True}
    hooks = HookRegistry()

    async def modify_hook(event: str, data: dict) -> HookResult:
        if event == "tool:post":
            return HookResult(action="modify", data={"result": modified_content})
        return HookResult()

    hooks.register("tool:post", modify_hook, priority=50, name="test_modify")

    # Mock tool call
    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {"key": "value"}

    mock_provider = _make_provider_responses([tool_call])
    providers = {"test_provider": mock_provider}
    context, messages_added = _make_context()

    await orchestrator.execute(
        prompt="test prompt",
        context=context,
        providers=providers,
        tools=tools,
        hooks=hooks,
    )

    tool_msgs = _get_tool_result_messages(messages_added)
    assert len(tool_msgs) == 1, f"Expected 1 tool message, got {len(tool_msgs)}"

    tool_content = tool_msgs[0]["content"]

    # Content should be the MODIFIED data, not the original
    assert tool_content == json.dumps(modified_content), (
        f"Expected modified content {json.dumps(modified_content)}, got {tool_content}"
    )
    assert "big_data" not in tool_content


@pytest.mark.asyncio
async def test_tool_post_no_modify_uses_original():
    """When no hook returns modify, original get_serialized_output() is used."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_events import EventDrivenOrchestrator

    orchestrator = EventDrivenOrchestrator({"max_iterations": 5})

    original_output = {"original": True}
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.input_schema = {"type": "object", "properties": {}}
    mock_tool.execute = AsyncMock(return_value=_make_tool_result(original_output))
    tools = {"test_tool": mock_tool}

    # No modify hooks
    hooks = HookRegistry()

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {}

    mock_provider = _make_provider_responses([tool_call])
    providers = {"test_provider": mock_provider}
    context, messages_added = _make_context()

    await orchestrator.execute(
        prompt="test prompt",
        context=context,
        providers=providers,
        tools=tools,
        hooks=hooks,
    )

    tool_msgs = _get_tool_result_messages(messages_added)
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == json.dumps(original_output)


@pytest.mark.asyncio
async def test_tool_post_modify_with_string_result():
    """When hook returns modify with a string, it should be used as-is."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_events import EventDrivenOrchestrator

    orchestrator = EventDrivenOrchestrator({"max_iterations": 5})

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.input_schema = {"type": "object", "properties": {}}
    mock_tool.execute = AsyncMock(return_value=_make_tool_result("original text"))
    tools = {"test_tool": mock_tool}

    hooks = HookRegistry()

    async def modify_hook(event: str, data: dict) -> HookResult:
        if event == "tool:post":
            return HookResult(
                action="modify",
                data={"result": "truncated string result"},
            )
        return HookResult()

    hooks.register("tool:post", modify_hook, priority=50, name="test_modify")

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {}

    mock_provider = _make_provider_responses([tool_call])
    providers = {"test_provider": mock_provider}
    context, messages_added = _make_context()

    await orchestrator.execute(
        prompt="test prompt",
        context=context,
        providers=providers,
        tools=tools,
        hooks=hooks,
    )

    tool_msgs = _get_tool_result_messages(messages_added)
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "truncated string result"
