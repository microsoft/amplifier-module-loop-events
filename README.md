# Amplifier Event-Driven Orchestrator Module

Event-driven agent loop orchestrator for Amplifier with scheduler integration.

## Purpose

Provides event-driven orchestration that:
- Queries schedulers for tool/agent/context decisions
- Reduces multiple scheduler responses to single decision
- Falls back gracefully if no schedulers respond
- Maintains all standard orchestrator functionality

## Contract

**Module Type:** Orchestrator
**Mount Point:** `orchestrator`
**Entry Point:** `amplifier_mod_loop_events:mount`

## Configuration

```toml
[session]
orchestrator = "loop-events"
context = "context-simple"

[[providers]]
module = "provider-anthropic"
name = "claude"

[[hooks]]
module = "hooks-scheduler-heuristic"

[[hooks]]
module = "hooks-scheduler-cost-aware"
config = { cost_weight = 0.6, latency_weight = 0.4 }
```

## Behavior

Standard agent loop with event-driven decision-making:
1. Get user prompt
2. Loop while tool calls needed:
   - Query schedulers for tool selection via `decision:tool_resolution` event
   - Reduce responses (highest score wins)
   - Fall back to first available if no responses
   - Execute selected tool
   - Feed results back to LLM
3. Return final response

## Dependencies

- `amplifier-core>=1.0.0`

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.