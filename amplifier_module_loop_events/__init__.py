"""
Event-driven orchestrator module for Amplifier.
Trusts LLM decisions with optional scheduler veto/modification.
"""

import logging
from typing import Any

from amplifier_core import HookRegistry
from amplifier_core import ModuleCoordinator
from amplifier_core import ToolResult

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the event-driven orchestrator module.

    Args:
        coordinator: Module coordinator
        config: Optional configuration

    Returns:
        Optional cleanup function
    """
    config = config or {}
    orchestrator = EventDrivenOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    logger.info("Mounted EventDrivenOrchestrator")
    return


class EventDrivenOrchestrator:
    """
    Event-driven orchestrator that trusts LLM decisions.
    Schedulers can observe and optionally veto/modify tool selections.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.max_iterations = config.get("max_iterations", 50)
        self.default_provider = config.get("default_provider")

    async def execute(
        self, prompt: str, context, providers: dict[str, Any], tools: dict[str, Any], hooks: HookRegistry
    ) -> str:
        """
        Execute the agent loop trusting LLM decisions.

        Args:
            prompt: User input prompt
            context: Context manager
            providers: Available providers
            tools: Available tools
            hooks: Hook registry

        Returns:
            Final response string
        """
        # Emit session start
        await hooks.emit("session:start", {"prompt": prompt})

        # Add user message to context
        await context.add_message({"role": "user", "content": prompt})

        # Select provider
        provider = self._select_provider(providers)
        if not provider:
            return "Error: No providers available"

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            # Get messages from context
            messages = await context.get_messages()

            # Get completion from provider
            try:
                # Pass tools to provider so LLM can use them
                response = await provider.complete(messages, tools=list(tools.values()))
            except Exception as e:
                logger.error(f"Provider error: {e}")
                # Emit error event
                await hooks.emit(
                    "error:provider",
                    {
                        "error_type": "completion_failed",
                        "error_message": str(e),
                        "severity": "high",
                    },
                )
                final_response = f"Error getting response: {e}"
                break

            # Check for tool calls
            tool_calls = provider.parse_tool_calls(response)

            if not tool_calls:
                # No tool calls - we're done
                final_response = response.content
                await context.add_message({"role": "assistant", "content": final_response})
                break

            # Add assistant message with tool calls to context
            await context.add_message(
                {
                    "role": "assistant",
                    "content": response.content if response.content else "",
                    "tool_calls": [{"tool": tc.tool, "arguments": tc.arguments, "id": tc.id} for tc in tool_calls],
                }
            )

            # Execute tool calls
            for tool_call in tool_calls:
                # Trust LLM's selection
                tool_name = tool_call.tool

                # Optional: Allow schedulers to veto or modify
                hook_result = await hooks.emit(
                    "tool:selecting",
                    {"tool": tool_name, "arguments": tool_call.arguments, "available_tools": list(tools.keys())},
                )

                if hook_result.action == "deny":
                    logger.info(f"Tool {tool_name} vetoed: {hook_result.reason}")
                    reason = hook_result.reason or "Tool execution denied by scheduler"
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {reason}",
                        }
                    )
                    continue
                if hook_result.action == "modify":
                    original_tool = tool_name
                    tool_name = hook_result.data.get("tool", tool_name) if hook_result.data else tool_name
                    logger.info(f"Tool changed by scheduler: {original_tool} → {tool_name}")

                # Emit selection for logging (AFTER decision is made)
                await hooks.emit(
                    "tool:selected",
                    {
                        "tool": tool_name,
                        "source": "scheduler" if hook_result.action == "modify" else "llm",
                        "original_tool": tool_call.tool if hook_result.action == "modify" else None,
                    },
                )

                # Get tool object first to pass to hook
                tool = tools.get(tool_name)

                # Pre-tool hook (backward compatibility, now includes tool object for metadata checking)
                hook_data = {"tool": tool_name, "arguments": tool_call.arguments}
                if tool:
                    hook_data["tool_obj"] = tool
                pre_hook_result = await hooks.emit("tool:pre", hook_data)

                if pre_hook_result.action == "deny":
                    # Tool denied by hook - MUST add tool_result for API compliance
                    reason = pre_hook_result.reason or "Tool execution denied"
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {reason}",
                        }
                    )
                    continue

                # Check if tool exists (we already got it earlier for the hook)
                if not tool:
                    # Tool not found - MUST add tool_result for API compliance
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call.id,
                            "content": f"Error: Tool {tool_name} not found",
                        }
                    )
                    # Emit error event
                    await hooks.emit(
                        "error:tool",
                        {
                            "error_type": "tool_not_found",
                            "error_message": f"Tool {tool_name} not found",
                            "severity": "medium",
                        },
                    )
                    continue

                # Execute tool
                try:
                    result = await tool.execute(tool_call.arguments)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    result = ToolResult(success=False, error={"message": str(e)})
                    # Emit error event
                    await hooks.emit(
                        "error:tool",
                        {
                            "error_type": "execution_failed",
                            "error_message": str(e),
                            "tool": tool_name,
                            "severity": "high",
                        },
                    )

                # Post-tool hook
                await hooks.emit(
                    "tool:post",
                    {
                        "tool": tool_name,
                        "result": result.model_dump() if hasattr(result, "model_dump") else str(result),
                    },
                )

                # Add tool result to context
                await context.add_message(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call.id,
                        "content": str(result.output) if result.success else f"Error: {result.error}",
                    }
                )

            # Check if we should compact context
            if await context.should_compact():
                await hooks.emit("context:pre-compact", {})
                await context.compact()

        # Emit session end
        await hooks.emit("session:end", {"response": final_response})

        return final_response

    def _select_provider(self, providers: dict[str, Any]) -> Any:
        """
        Select a provider to use.

        Args:
            providers: Available providers

        Returns:
            Selected provider or None
        """
        if not providers:
            return None

        # Use configured default if available
        if self.default_provider and self.default_provider in providers:
            return providers[self.default_provider]

        # Otherwise use first available
        return next(iter(providers.values()))
