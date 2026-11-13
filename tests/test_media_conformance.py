"""Synthetic structural media-v1 contract tests: no inference, captures or stores."""

import asyncio
import base64
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from amplifier_core import ToolResult
from amplifier_core.hooks import HookRegistry
from amplifier_core.models import HookResult

from amplifier_module_loop_events import EventDrivenOrchestrator, mount
from amplifier_module_loop_events._media import MEDIA, UNAVAILABLE, MediaCheckpointError

PIXELS = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+jC1sAAAAASUVORK5CYII="
)


def ref(call_id, label="a"):
    return {
        "v": 1,
        "session_id": "session-test",
        "sha256": hashlib.sha256(base64.b64decode(PIXELS)).hexdigest(),
        "mime": "image/png",
        "bytes": len(base64.b64decode(PIXELS)),
        "width": 1,
        "height": 1,
        "provenance": {
            "kind": "tool",
            "session_id": "session-test",
            "id": call_id + label,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": None,
        },
    }


def image_result():
    return ToolResult(
        output={"note": "ordinary useful text"},
        content=[
            {"type": "text", "text": "image label"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": PIXELS},
            },
        ],
    )


class Media:
    """Contract double, not evidence of Foundation storage or actual durable ACK."""

    version = 1
    session_id = "session-test"

    def __init__(self):
        self.ingested = []
        self.published = []
        self.staged = {}

    async def ingest(self, value, *, source_format, origin, policy):
        assert isinstance(value, ToolResult)
        assert source_format == "core-tool-result"
        assert policy is None
        self.ingested.append(deepcopy(origin))
        if not any(block.type == "image" for block in (value.content or [])):
            return {"value": value, "staged_refs": []}
        refs = [ref(origin["id"], "a"), ref(origin["id"], "b")]
        self.staged[origin["id"]] = refs
        output = dict(value.output)
        output[MEDIA] = {"refs": refs}
        return {
            "value": ToolResult(success=value.success, output=output, error=value.error),
            "staged_refs": refs,
        }

    async def publish(self, refs, *, origin, disposition):
        assert disposition == "retain"
        assert all(r in self.staged[origin["id"]] for r in refs)
        self.published.append((deepcopy(refs), deepcopy(origin)))
        return {
            "refs": deepcopy(refs),
            "publication": {"state": "published", "durability": "assets-only"},
        }

    def validate(self, *args, **kwargs):
        raise AssertionError("Only host/provider validates private projection")

    def resolve(self, *args, **kwargs):
        raise AssertionError("Loop must not resolve pixels")

    async def prepare_request(self, *args, **kwargs):
        raise AssertionError("Loop must not hydrate its canonical request")

    def report(self, *args, **kwargs):
        raise AssertionError("Loop must not manufacture a committed report")


class Context:
    def __init__(self):
        self.messages = []
        self.providers = []
        self.on_tool = None

    async def add_message(self, message):
        self.messages.append(deepcopy(message))
        if message["role"] == "tool" and self.on_tool:
            self.on_tool()

    async def get_messages_for_request(self, provider=None):
        self.providers.append(provider)
        if hasattr(provider, "request_budget"):
            await provider.request_budget(self.messages)
        return self.messages


class Provider:
    def __init__(self, calls=None):
        self.calls = calls or [
            SimpleNamespace(id="call-1", name="image", arguments={"path": "synthetic"})
        ]
        self.requests = []
        self.counts = []

    async def request_budget(self, messages):
        # Deliberately no token or exact-image-cost claim.
        self.counts.append(deepcopy(messages))

    async def complete(self, request, **kwargs):
        self.requests.append(request.model_dump())
        assert PIXELS not in json.dumps(self.requests)
        return SimpleNamespace(
            content="" if len(self.requests) == 1 else "done",
            metadata=None,
            calls=self.calls if len(self.requests) == 1 else [],
        )

    def parse_tool_calls(self, response):
        return response.calls


class Tool:
    name = "image"
    description = "synthetic image"

    def __init__(self, result=None):
        self.input_schema = {}
        self.executions = []
        self.result = result if result is not None else image_result()

    async def execute(self, arguments):
        self.executions.append(deepcopy(arguments))
        return self.result


class Hooks:
    def __init__(self, post=None, selecting=None):
        self.post = post
        self.selecting = selecting
        self.events = []

    async def emit(self, event, data):
        self.events.append((event, deepcopy(data)))
        assert PIXELS not in json.dumps(data)
        if event == "tool:post" and self.post:
            return self.post(data)
        if event == "tool:selecting" and self.selecting:
            return self.selecting(data)
        return HookResult()


def setup(media=True, checkpoint=True, calls=None):
    ctx = Context()
    cap = Media() if media else None
    provider = Provider(calls)
    checkpoints = []

    def save():
        checkpoints.append(deepcopy(ctx.messages))
        return checkpoint

    capabilities = {MEDIA: cap, "session.durable_checkpoint": save}
    coord = SimpleNamespace(
        session_id="session-test",
        get_capability=capabilities.get,
        cancellation=SimpleNamespace(is_cancelled=False, is_immediate=False),
    )

    async def process(result, event, name):
        return result

    coord.process_hook_result = process
    return ctx, cap, provider, coord, capabilities, checkpoints


async def run(ctx, provider, coord, hooks=None, tools=None, limit=3):
    return await EventDrivenOrchestrator({"max_iterations": limit}).execute(
        "inspect synthetic", ctx, {"selected": provider},
        tools if tools is not None else {"image": Tool()}, hooks or Hooks(), coord,
    )


def tool_messages(ctx):
    return [m for m in ctx.messages if m["role"] == "tool"]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3], ids=["final-completion", "main-completion"])
async def test_surviving_ordered_refs_checkpoint_and_canonical_requests(limit):
    ctx, cap, provider, coord, _, checkpoints = setup()
    assert await run(ctx, provider, coord, limit=limit) == "done"
    message = tool_messages(ctx)[0]
    assert message["metadata"][MEDIA]["refs"] == cap.staged["call-1"]
    assert message["tool_call_id"] == "call-1"
    assert message["name"] == "image"
    assert "ordinary useful text" in message["content"]
    assert checkpoints[0][-1] == message
    assert all(p is provider for p in ctx.providers)
    assert len(provider.requests) == 2
    request_tool = next(m for m in provider.requests[1]["messages"] if m["role"] == "tool")
    assert request_tool["metadata"] == message["metadata"]
    assert PIXELS not in json.dumps([ctx.messages, checkpoints, provider.counts])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["null", "missing", "remove-result", "remove-output", "remove-namespace", "in-place", "string",
     "json-string", "list", "deny", "exception"],
)
async def test_posthook_authority_never_resurrects_refs(case):
    ctx, cap, provider, coord, _, _ = setup()

    def post(data):
        if case == "exception":
            raise RuntimeError("synthetic hook failure")
        if case == "deny":
            return HookResult(action="deny", reason="redacted")
        if case == "null":
            return HookResult(action="modify", data={"result": None})
        if case == "missing":
            return HookResult(action="modify", data={})
        if case == "remove-result":
            data.pop("result")
            return HookResult(data=data)
        if case == "in-place":
            data["result"]["output"].pop(MEDIA)
            return HookResult()
        replacements = {
            "remove-output": {"success": True},
            "remove-namespace": {"success": True, "output": {"text": "redacted"}},
            "string": "redacted",
            "json-string": json.dumps(data["result"]),
            "list": [data["result"]],
        }
        return HookResult(action="modify", data={"result": replacements[case]})

    await run(ctx, provider, coord, Hooks(post=post))
    assert not cap.published
    message = tool_messages(ctx)[0]
    assert "metadata" not in message
    if case in ("null", "missing", "remove-result"):
        assert message["content"] == "null"
    if case == "json-string":
        assert message["content"].startswith("{")
    assert message["tool_call_id"] == "call-1"


@pytest.mark.asyncio
async def test_copied_reordered_effective_mapping_is_only_publication_source():
    ctx, cap, provider, coord, _, _ = setup()

    def post(data):
        value = deepcopy(data["result"])
        value["output"][MEDIA]["refs"].reverse()
        return HookResult(action="modify", data={"result": value})

    await run(ctx, provider, coord, Hooks(post=post))
    expected = list(reversed(cap.staged["call-1"]))
    assert cap.published[0][0] == expected
    assert tool_messages(ctx)[0]["metadata"][MEDIA]["refs"] == expected


@pytest.mark.asyncio
async def test_processed_hook_result_overrides_emit_result():
    ctx, cap, provider, coord, _, _ = setup()

    async def process(result, event, name):
        if event == "tool:post":
            return HookResult(action="modify", data={"result": None})
        return result

    coord.process_hook_result = process
    await run(ctx, provider, coord)
    assert not cap.published
    assert tool_messages(ctx)[0]["content"] == "null"


@pytest.mark.asyncio
@pytest.mark.parametrize("veto", [False, True])
async def test_scheduler_binds_actual_dispatch_and_veto_never_stages(veto):
    ctx, cap, provider, coord, _, _ = setup()
    original = Tool()
    result = image_result()
    result.output["note"] = "replacement"
    replacement = Tool(result)
    replacement.name = "replacement"
    hooks = Hooks(selecting=lambda data: (
        HookResult(action="deny", reason="scheduler veto") if veto
        else HookResult(action="modify", data={"tool": "replacement"})
    ))
    await run(ctx, provider, coord, hooks, {"image": original, "replacement": replacement})
    assert not original.executions
    if veto:
        assert not replacement.executions
        assert not cap.ingested and not cap.published
        assert "metadata" not in tool_messages(ctx)[0]
    else:
        assert replacement.executions == [{"path": "synthetic"}]
        assert tool_messages(ctx)[0]["name"] == "replacement"
        assert cap.ingested[0]["id"] == "call-1"
        assert "replacement" in tool_messages(ctx)[0]["content"]
        post = next(data for event, data in hooks.events if event == "tool:post")
        assert post["tool_name"] == "replacement"
        assert post["tool_call_id"] == "call-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "exception", "cancel"])
@pytest.mark.parametrize("nested", [False, True])
async def test_dispatch_context_actual_scheduler_call_restored(outcome, nested):
    ctx, _, provider, coord, _, _ = setup()
    task = asyncio.current_task()
    previous = {"tool_call_id": "outer-call"}
    coord._tool_dispatch_contexts = {task: previous} if nested else {}
    seen = []

    class Producer(Tool):
        name = "producer"

        async def execute(self, arguments):
            seen.append(deepcopy(coord._tool_dispatch_contexts[asyncio.current_task()]))
            if outcome == "exception":
                raise RuntimeError("synthetic tool failure")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            return image_result()

    await run(
        ctx, provider, coord,
        Hooks(selecting=lambda data: HookResult(action="modify", data={"tool": "producer"})),
        {"image": Tool(), "producer": Producer()},
    )
    assert seen == [{"tool_call_id": "call-1"}]
    assert coord._tool_dispatch_contexts == ({task: previous} if nested else {})
    assert tool_messages(ctx)[0]["name"] == "producer"


@pytest.mark.asyncio
async def test_capability_session_id_fallback_when_coordinator_has_no_id():
    ctx, cap, provider, coord, _, _ = setup()
    del coord.session_id
    await run(ctx, provider, coord)
    assert cap.ingested[0]["session_id"] == cap.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", [False, None, "raises"])
@pytest.mark.parametrize("limit", [1, 3])
async def test_checkpoint_failure_blocks_every_later_count_and_send(ack, limit):
    ctx, _, provider, coord, capabilities, _ = setup(checkpoint=ack)
    if ack == "raises":
        async def fail():
            raise OSError("synthetic checkpoint failure")
        capabilities["session.durable_checkpoint"] = fail
    with pytest.raises(MediaCheckpointError):
        await run(ctx, provider, coord, limit=limit)
    assert len(provider.requests) == 1
    assert len(provider.counts) == 1
    assert "metadata" in tool_messages(ctx)[0]  # Settled result was not erased/replayed.


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
async def test_async_checkpoint_and_cancellation_preserve_settled_evidence(limit):
    ctx, cap, provider, coord, capabilities, _ = setup()
    saved = []

    async def save():
        saved.append(deepcopy(ctx.messages))
        return True

    capabilities["session.durable_checkpoint"] = save
    ctx.on_tool = lambda: setattr(coord.cancellation, "is_cancelled", True)
    await run(ctx, provider, coord, limit=limit)
    assert saved[0][-1]["metadata"][MEDIA]["refs"] == cap.published[0][0]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["capability", "incompatible", "checkpoint"])
async def test_unavailable_keeps_text_without_base64_or_unsupported_refs(missing):
    ctx, cap, provider, coord, capabilities, _ = setup()
    if missing == "checkpoint":
        capabilities.pop("session.durable_checkpoint")
    elif missing == "incompatible":
        cap.version = 2
    else:
        capabilities.pop(MEDIA)
    hooks = Hooks()
    await run(ctx, provider, coord, hooks)
    message = tool_messages(ctx)[0]
    assert "ordinary useful text" in message["content"]
    assert "Image unavailable" in message["content"]
    assert "metadata" not in message
    assert not cap.published
    assert PIXELS not in json.dumps([ctx.messages, hooks.events, provider.requests])


@pytest.mark.asyncio
async def test_inherited_refs_without_capability_unavailable_before_dispatch_not_mutated():
    ctx, _, provider, coord, _, _ = setup(media=False)
    inherited = {
        "role": "user", "content": "prior image",
        "metadata": {MEDIA: {"refs": [ref("prior")]}, "owner": {"opaque": "preserve"}},
    }
    ctx.messages.append(deepcopy(inherited))
    provider.calls = []
    await run(ctx, provider, coord, tools={})
    assert ctx.messages[0] == inherited
    # Context may measure only via the actual host adapter. The loop passes that
    # adapter, and its own outbound canonical request is an explicit text fallback.
    assert "Image unavailable" in provider.requests[0]["messages"][0]["content"]
    assert provider.requests[0]["messages"][0]["metadata"]["owner"] == {"opaque": "preserve"}
    assert MEDIA not in provider.requests[0]["messages"][0]["metadata"]


@pytest.mark.asyncio
async def test_real_hook_registry_noop_and_modify_use_same_effective_contract():
    for modify in (False, True):
        ctx, cap, provider, coord, _, _ = setup()
        hooks = HookRegistry()

        async def post(event, data, modify=modify):
            assert PIXELS not in json.dumps(data)
            if modify:
                return HookResult(action="modify", data={"result": None})
            return HookResult()

        hooks.register("tool:post", post)
        await run(ctx, provider, coord, hooks)
        assert bool(cap.published) is not modify


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("vision", [False, True])
async def test_host_adapter_owns_private_count_dispatch_projection(limit, vision):
    """Provider seam conformance only; not an actual Foundation integration pass."""
    ctx, _, _, coord, _, checkpoints = setup(media=vision)

    class Adapter(Provider):
        def __init__(self):
            super().__init__()
            self.count_projections = []
            self.send_projections = []

        def project(self, messages):
            private = deepcopy(messages)
            for message in private:
                metadata = message.get("metadata") or {}
                if MEDIA not in metadata:
                    continue
                refs = metadata.pop(MEDIA)["refs"]
                if vision:
                    message["content"] = [
                        {"type": "text", "text": message["content"]},
                        *[
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64", "media_type": "image/png",
                                    "data": PIXELS,
                                },
                            }
                            for _ in refs
                        ],
                    ]
                else:
                    message["content"] += "\n\n" + UNAVAILABLE
            return private

        async def request_budget(self, messages):
            # The actual installed host must prepare before any real count.
            self.count_projections.append(self.project(messages))

        async def complete(self, request, **kwargs):
            self.send_projections.append(self.project(request.model_dump()["messages"]))
            return await super().complete(request, **kwargs)

    provider = Adapter()
    if not vision:
        ctx.messages.append({
            "role": "user", "content": "inherited image",
            "metadata": {MEDIA: {"refs": [ref("prior")]}},
        })
    await run(ctx, provider, coord, limit=limit)
    for counted, sent in zip(provider.count_projections, provider.send_projections):
        # Finalization adds only its noncanonical loop-limit reminder after context.
        if limit == 1 and len(sent) == len(counted) + 1:
            sent = sent[:-1]
        # Core fills absent optional fields with None; compare semantic content.
        assert [(m["role"], m["content"]) for m in counted] == [
            (m["role"], m["content"]) for m in sent
        ]
    assert all(p is provider for p in ctx.providers)
    assert PIXELS not in json.dumps([ctx.messages, checkpoints, provider.requests])
    if vision:
        assert PIXELS in json.dumps(provider.send_projections[1])
        projected_tool = next(
            m for m in provider.send_projections[1] if m["role"] == "tool"
        )
        assert projected_tool["tool_call_id"] == "call-1"
    else:
        assert PIXELS not in json.dumps(provider.send_projections)
        assert "Image unavailable" in provider.count_projections[0][0]["content"]


@pytest.mark.asyncio
async def test_sequential_tools_keep_pairing_order_status_and_no_refs_on_failure():
    calls = [
        SimpleNamespace(id="call-1", name="image", arguments={}),
        SimpleNamespace(id="call-2", name="failure", arguments={}),
    ]
    ctx, cap, provider, coord, _, _ = setup(calls=calls)
    await run(ctx, provider, coord, tools={
        "image": Tool(),
        "failure": Tool(ToolResult(
            success=False, output={"stderr": "failure detail", "returncode": 1},
            error={"message": "failed"},
        )),
    })
    messages = tool_messages(ctx)
    assert [m["tool_call_id"] for m in messages] == ["call-1", "call-2"]
    assert MEDIA in messages[0]["metadata"]
    assert "metadata" not in messages[1]
    assert json.loads(messages[1]["content"]) == {"stderr": "failure detail", "returncode": 1}
    assert len(cap.published) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["ingest", "publish"])
async def test_capability_error_never_leaks_pixels_or_reexecutes(seam, caplog):
    ctx, cap, provider, coord, _, _ = setup()

    async def fail(*args, **kwargs):
        raise ValueError("untrusted decoder payload: " + PIXELS)

    setattr(cap, seam, fail)
    tool = Tool()
    await run(ctx, provider, coord, tools={"image": tool})
    assert len(tool.executions) == 1
    assert "metadata" not in tool_messages(ctx)[0]
    assert PIXELS not in caplog.text
    assert PIXELS not in json.dumps([ctx.messages, provider.requests])


@pytest.mark.asyncio
async def test_text_only_rich_content_survives_missing_capability():
    ctx, _, provider, coord, _, _ = setup(media=False)
    hooks = Hooks()
    result = ToolResult(content=[{"type": "text", "text": "useful rich text"}])
    await run(ctx, provider, coord, hooks, {"image": Tool(result)})
    post = next(data for event, data in hooks.events if event == "tool:post")
    assert post["result"]["content"] == [{"type": "text", "text": "useful rich text"}]
    assert tool_messages(ctx)[0]["content"] == "useful rich text"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_report", [False, True])
async def test_mount_reports_consumer_readiness_observation_only(async_report):
    _, cap, _, coord, _, _ = setup()
    observations = []
    mounted = []

    def report(origin, **kwargs):
        observations.append((origin, kwargs))

    async def areport(origin, **kwargs):
        report(origin, **kwargs)

    async def save_mount(slot, value):
        mounted.append(slot)

    cap.report = areport if async_report else report
    coord.mount = save_mount
    await mount(coord)
    assert mounted == ["orchestrator"]
    assert observations == [(None, {
        "event": "consumer-ready", "outcome": "ready",
        "details": {"contract": 1, "role": "loop"},
    })]
    assert not cap.ingested and not cap.published


@pytest.mark.asyncio
async def test_legacy_context_signature_is_not_called_twice_on_internal_typeerror():
    ctx, _, provider, coord, _, _ = setup()
    calls = []

    async def get_messages():
        calls.append(True)
        return ctx.messages

    ctx.get_messages_for_request = get_messages
    await run(ctx, provider, coord)
    assert len(calls) == 2

    async def failing_messages(provider=None):
        calls.append(True)
        raise TypeError("synthetic compaction internal error")

    ctx.get_messages_for_request = failing_messages
    calls.clear()
    with pytest.raises(TypeError, match="synthetic compaction"):
        await run(ctx, provider, coord)
    assert calls == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [None, "redacted", {"output": {"note": "redacted"}}])
async def test_result_field_presence_is_authoritative_even_without_modify(replacement):
    ctx, cap, provider, coord, _, _ = setup()
    await run(ctx, provider, coord, Hooks(post=lambda data: HookResult(
        data={"result": replacement},
    )))
    assert not cap.published
    assert "metadata" not in tool_messages(ctx)[0]
    assert "ordinary useful text" not in tool_messages(ctx)[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", [
    {"refs": "wrong-type"}, {"refs": [], "unexpected": True},
    {"refs": [ref("other-origin")]},
])
async def test_invalid_or_unstaged_effective_refs_never_become_metadata(carrier):
    ctx, cap, provider, coord, _, _ = setup()
    await run(ctx, provider, coord, Hooks(post=lambda data: HookResult(
        action="modify",
        data={"result": {"success": True, "output": {MEDIA: carrier}}},
    )))
    assert not cap.published
    message = tool_messages(ctx)[0]
    assert "metadata" not in message
    assert message["tool_call_id"] == "call-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [None, "redacted", {"output": {"note": "redacted"}}])
async def test_in_place_event_result_replacement_cannot_restore_original(replacement):
    ctx, cap, provider, coord, _, _ = setup()

    def post(data):
        data["result"] = replacement
        return HookResult()

    await run(ctx, provider, coord, Hooks(post=post))
    assert not cap.published
    assert "metadata" not in tool_messages(ctx)[0]
    assert "ordinary useful text" not in tool_messages(ctx)[0]["content"]


@pytest.mark.asyncio
async def test_typed_posthook_result_normalized_before_rich_image_fallback():
    ctx, cap, provider, coord, _, checkpoints = setup()
    await run(ctx, provider, coord, Hooks(post=lambda data: HookResult(
        action="modify", data={"result": image_result()},
    )))
    assert not cap.published
    message = tool_messages(ctx)[0]
    assert "metadata" not in message
    assert "Image unavailable" in message["content"]
    assert PIXELS not in json.dumps([ctx.messages, checkpoints, provider.requests])