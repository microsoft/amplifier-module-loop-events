"""Actual public participating graph conformance, opt-in exact source only.

No private bridge seam, live inference, host activation or synthetic tool:post
readiness is used. The fixture graph is explicitly cooperative, not a claim of
universal native or arbitrary Python callback coverage.
"""

import asyncio
import base64
import hashlib
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
from amplifier_core import AmplifierSession
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.models import HookResult

from amplifier_module_loop_events import EventDrivenOrchestrator
from amplifier_module_loop_events._media import MEDIA
from tests.test_media_foundation_pairing import rich_images

pytest_plugins = ["tests.test_media_foundation_pairing"]

pytestmark = pytest.mark.skipif(
    os.environ.get("AMPLIFIER_MEDIA_BRIDGE_PROFILE") != "qualified",
    reason="public participating positive graph candidate not selected",
)


class GraphLoader:
    """Finite Python module mounts passed through the public bridge loader."""

    def __init__(self, modules, policy):
        self.modules = modules
        self.policy = policy
        self.queue = []

    async def load(self, module_id, config, *, source_hint=None, coordinator=None):
        async def mount(current):
            if module_id == "fixture-hooks":
                current.hooks.register("tool:post", self.policy, priority=10, name="policy")
            else:
                slot, value, name = self.modules[module_id]
                await current.mount(slot, value, name=name)
        return mount

    def enqueue_on_session_ready(self, name, callback):
        self.queue.append((name, callback))

    def get_on_session_ready_queue(self):
        return self.queue

    def clear_on_session_ready_queue(self):
        self.queue.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("limit", [1, 3], ids=["final", "main"])
@pytest.mark.parametrize("case", [
    "keep", "partial", "null", "missing", "inplace", "string", "deny",
    "error", "outer-redaction", "unknown-observed", "checkpoint-failure",
    "unsupported-none-budget", "visual-none-budget", "continue-null-echo",
    "special-null-echo", "count-mutation",
])
async def test_public_graph_actual_core_store_checkpoint_count_send(
    foundation, bridge_installer, tmp_path, fmt, limit, case
):
    from amplifier_foundation.session.media_provider import MediaPreparedProvider

    media, checkpoint = foundation
    session = AmplifierSession({
        "session": {"orchestrator": "loop-events", "context": "fixture-context"},
        "providers": [{"module": "fixture-provider"}],
        "tools": [{"module": "fixture-image"}],
        "hooks": [{"module": "fixture-hooks"}],
    }, session_id="events-positive")
    core = session.coordinator
    order, posts, counted, sent, canonical, checkpoints = [], [], [], [], [], []
    rich, pixels = rich_images(fmt)
    digest = [hashlib.sha256(base64.b64decode(value)).hexdigest() for value in pixels]

    class Context:
        def __init__(self):
            self.messages = []

        async def add_message(self, message):
            self.messages.append(deepcopy(message))
            if message["role"] == "tool":
                order.append("append")

        async def get_messages(self):
            return self.messages

        async def get_messages_for_request(self, provider=None):
            assert provider is adapter
            if case not in {"unsupported-none-budget", "visual-none-budget"} and any(
                MEDIA in (row.get("metadata") or {}) for row in self.messages
            ):
                preliminary = ChatRequest(messages=[Message(**row) for row in self.messages])
                await provider.request_budget(preliminary, request_options={})
            return self.messages

    context = Context()
    store = media.SessionMediaStore(
        tmp_path / "media", core.session_id,
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    cap = media.bind_session_media(core, store=store, enabled=True)
    publications = []
    publish = cap.publish

    async def observed_publish(refs, *, origin, disposition):
        publications.append((deepcopy(refs), deepcopy(origin), disposition))
        return await publish(refs, origin=origin, disposition=disposition)

    cap.publish = observed_publish

    class Wire:
        default_model = "fixture"

        def get_media_request_support(self, **kwargs):
            return {
                "v": 1,
                "positions": [] if case == "unsupported-none-budget" else ["tool"],
            }

        async def request_budget(self, request, *, request_options, context_estimate=0):
            cap.validate(request, purpose="provider-count", request_options=request_options)
            counted.append((request, request.model_dump(), deepcopy(request_options)))
            order.append("count")
            if case == "count-mutation" and request.tools:
                request.messages[-1].content = "changed after measured candidate"
            if case in {"unsupported-none-budget", "visual-none-budget"}:
                return None
            return {
                "estimated_input_tokens": 10, "input_limit_tokens": 100,
                "context_token_budget": 100,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 10},
            }

        async def complete(self, request, **kwargs):
            cap.validate(request, purpose="provider-send", request_options=kwargs)
            sent.append((request, request.model_dump(), deepcopy(kwargs)))
            order.append("send")
            return SimpleNamespace(
                content="" if len(sent) == 1 else "done", metadata=None,
                calls=[{"id": "actual-call", "name": "image", "arguments": {}}]
                if len(sent) == 1 else [],
            )

        def parse_tool_calls(self, response):
            return response.calls

    adapter = MediaPreparedProvider(
        Wire(), cap, provider_identity={"provider": "fixture", "account": "fixture", "generation": 1},
    )
    executions = []

    class ImageTool:
        name = "image"
        description = "Synthetic image"

        def __init__(self):
            self.input_schema = {"type": "object", "properties": {"detail": {"type": "string"}}}

        async def execute(self, arguments):
            executions.append(deepcopy(arguments))
            assert core._tool_dispatch_contexts[asyncio.current_task()][
                "tool_call_id"
            ] == "actual-call"
            return rich

    def policy(event, data):
        posts.append(deepcopy(data))
        order.append("post")
        if case == "partial":
            value = deepcopy(data["result"])
            value["output"][MEDIA]["refs"] = value["output"][MEDIA]["refs"][:1]
            return HookResult(action="modify", data={"result": value})
        if case == "null":
            return HookResult(action="modify", data={"result": None})
        if case == "continue-null-echo":
            return HookResult(data={"result": None})
        if case == "special-null-echo":
            return HookResult(
                action="inject_context", data={"result": None},
                context_injection="special reminder",
            )
        if case == "missing":
            return HookResult(action="modify", data={})
        if case == "inplace":
            data["result"]["output"].pop(MEDIA)
        if case == "string":
            return HookResult(action="modify", data={"result": "redacted text"})
        if case == "deny":
            return HookResult(action="deny", reason="redacted")
        if case == "error":
            raise RuntimeError("fixture policy failure")
        return HookResult()

    orchestrator = EventDrivenOrchestrator({
        "max_iterations": limit, "reasoning_effort": "low", "extended_thinking": True,
    })
    # Record canonical calls without replacing the actual provider adapter or
    # bridge emitter/processor identities.
    original_complete = adapter.complete
    preparation_failures = []
    original_fallback = adapter.media_text_fallback_witness

    async def observe_fallback(request, **kwargs):
        try:
            return await original_fallback(request, **kwargs)
        except ValueError as error:
            preparation_failures.append(str(error))
            raise

    adapter.media_text_fallback_witness = observe_fallback

    async def observe_canonical(request, **kwargs):
        canonical.append(request.model_dump())
        return await original_complete(request, **kwargs)

    adapter.complete = observe_canonical
    modules = {
        "loop-events": ("orchestrator", orchestrator, None),
        "fixture-context": ("context", context, None),
        "fixture-provider": ("providers", adapter, "fixture"),
        "fixture-image": ("tools", ImageTool(), "image"),
    }
    core.loader = GraphLoader(modules, policy)
    bridge = bridge_installer(core)
    graph = bridge.begin_participating_graph(
        session, owner="events.controlled.graph",
        participants=(*modules, "fixture-hooks", "host.policy"),
    )
    await session.initialize()

    def outer(event, data):
        if case in {"continue-null-echo", "special-null-echo"}:
            return HookResult(action="modify", data=data)
        if case == "outer-redaction":
            return HookResult(action="modify", data={"result": None})
        if case == "null":
            return HookResult(
                action="inject_context", context_injection="qualified reminder",
            )
        return HookResult()

    graph.participant("host.policy").hooks.register(
        "tool:post", outer, priority=20, name="outer-policy"
    )
    receipt = graph.seal()
    assert bridge.validate_receipt(receipt)
    if case == "unknown-observed":
        core.hooks.register("tool:post", lambda e, d: HookResult(), name="unqualified")

    def save():
        if case == "checkpoint-failure":
            raise OSError("fixture save refused")
        messages = deepcopy(context.messages)
        (tmp_path / "transcript.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in messages)
        )
        ack = checkpoint.acknowledge_legacy_checkpoint(
            tmp_path, session_id=core.session_id, expected_messages=messages,
        )
        store.acknowledge_checkpoint(ack)
        checkpoints.append(messages)
        order.append("checkpoint")
        return True

    core.register_capability("session.durable_checkpoint", save)
    try:
        if case == "checkpoint-failure":
            with pytest.raises(Exception, match="checkpoint"):
                await session.execute("Inspect the synthetic observation")
            assert len(sent) == 1
            assert len(counted) == 0
            return
        answer = await session.execute("Inspect the synthetic observation")
        if case in {"visual-none-budget", "unsupported-none-budget", "count-mutation"}:
            assert len(sent) == 1
            assert len(counted) == (2 if case == "count-mutation" else 1)
            assert answer != "done"
            if case == "unsupported-none-budget":
                # The tool-call history contains opaque Core message extras.
                # F1 owns this refusal; never drop them to manufacture fallback.
                assert preparation_failures == ["Text fallback cannot omit opaque message state"]
            return
        assert answer == "done", preparation_failures
        assert len(executions) == len(posts) == len(publications) == 1
        assert len(sent) == 2
        tool_row = next(row for row in context.messages if row["role"] == "tool")
        expected = 2 if case in {"keep", "unsupported-none-budget"} else 1 if case == "partial" else 0
        assert publications[0][2] == ("retain" if expected else "drop")
        assert publications[0][1]["id"] == "actual-call"
        if expected:
            assert "useful text" in tool_row["content"]
            assert "observation" in tool_row["content"]
            refs = tool_row["metadata"][MEDIA]["refs"]
            assert [item["sha256"] for item in refs] == digest[:expected]
            assert counted[-1][0] is sent[-1][0]
            assert counted[-1][1] == sent[-1][1]
            assert counted[-1][2] == sent[-1][2] == {"extended_thinking": True}
            assert counted[-1][1]["tools"] and counted[-1][1]["reasoning_effort"] == "low"
            assert order.index("append") < order.index("checkpoint") < order.index("count")
            assert order[-2:] == ["count", "send"]
            outgoing_tool = next(row for row in sent[-1][1]["messages"] if row["role"] == "tool")
            image_blocks = [block for block in outgoing_tool["content"] if block["type"] == "image"]
            if case == "unsupported-none-budget":
                assert not image_blocks
                assert "Image unavailable" in json.dumps(outgoing_tool["content"])
            else:
                assert [block["source"]["data"] for block in image_blocks] == pixels[:expected]
        else:
            assert "metadata" not in tool_row
            assert not any(value in json.dumps(sent[-1][1]) for value in pixels)
        if case in {"null", "missing", "outer-redaction", "continue-null-echo"}:
            assert tool_row["content"] == "null"
            assert "useful text" not in tool_row["content"]
        if case == "string":
            assert tool_row["content"] == "redacted text"
        if case == "deny":
            assert tool_row["content"] == "Error: redacted"
        if case == "special-null-echo":
            assert "metadata" not in tool_row
            assert tool_row["content"].startswith("null")
        if case == "null":
            assert any(row["content"] == "qualified reminder" for row in context.messages)
        persisted = json.dumps([context.messages, canonical, posts, checkpoints, list(store.journal())])
        for value in pixels:
            assert value not in persisted
        for path in tmp_path.rglob("*.json*"):
            assert not any(value in path.read_text() for value in pixels)
    finally:
        await adapter.aclose()
        store.close()
        await session.cleanup()