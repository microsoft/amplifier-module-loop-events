"""Opt-in immutable Foundation pairing; runtime module still depends only on Core.

Set AMPLIFIER_MEDIA_F1_TEST_ROOT to a git-archived candidate, placed first on
PYTHONPATH. This suite does not install or enable anything in the running host.
"""

import base64
import importlib.util
import io
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from amplifier_core import AmplifierSession, ToolResult
from amplifier_core.hooks import HookRegistry
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.models import HookResult

from amplifier_module_loop_events import EventDrivenOrchestrator
from amplifier_module_loop_events._media import MEDIA
from tests.test_media_conformance import (
    Context,
    Hooks,
    Provider,
    Tool,
    ref,
    run,
    setup,
    tool_messages,
)

ROOT = os.environ.get("AMPLIFIER_MEDIA_F1_TEST_ROOT")
BRIDGE = os.environ.get("AMPLIFIER_MEDIA_BRIDGE_TEST_FILE")
pytestmark = pytest.mark.skipif(not ROOT, reason="explicit immutable F1 pairing not selected")


@pytest.fixture
def foundation():
    from amplifier_foundation.session import checkpoint, media

    assert Path(media.__file__).is_relative_to(Path(ROOT).resolve())
    assert Path(checkpoint.__file__).is_relative_to(Path(ROOT).resolve())
    return media, checkpoint


def rich_images(fmt="PNG"):
    from PIL import Image

    encoded = []
    for color in ("red", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (3, 2), color).save(output, format=fmt)
        encoded.append(base64.b64encode(output.getvalue()).decode())
    return ToolResult(output={"note": "useful text"}, content=[
        {"type": "text", "text": "observation"},
        *[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg" if fmt == "JPEG" else "image/png",
                    "data": value,
                },
            }
            for value in encoded
        ],
    ]), encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("case", ["keep", "partial", "null", "missing", "inplace", "exception"])
async def test_actual_f1_store_single_origin_disposition_and_checkpoint(
    foundation, tmp_path, case, limit, fmt
):
    media, checkpoint = foundation
    ctx, _, provider, coord, capabilities, _ = setup()
    session = tmp_path / "session"
    session.mkdir()
    store = media.SessionMediaStore(
        session / "media", coord.session_id,
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    capabilities.pop(MEDIA)
    coord.register_capability = capabilities.__setitem__
    cap = media.bind_session_media(coord, store=store, enabled=True)
    publications = []
    original_publish = cap.publish

    async def publish(refs, *, origin, disposition):
        publications.append((deepcopy(refs), disposition))
        return await original_publish(refs, origin=origin, disposition=disposition)

    cap.publish = publish
    saved = []

    def save():
        messages = deepcopy(ctx.messages)
        (session / "transcript.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in messages)
        )
        ack = checkpoint.acknowledge_legacy_checkpoint(
            session, session_id=coord.session_id, expected_messages=messages
        )
        store.acknowledge_checkpoint(ack)
        saved.append(messages)
        return True

    capabilities["session.durable_checkpoint"] = save

    def post(event):
        if case == "exception":
            raise RuntimeError("synthetic callback error")
        if case == "partial":
            value = deepcopy(event["result"])
            value["output"][MEDIA]["refs"] = value["output"][MEDIA]["refs"][:1]
            return HookResult(action="modify", data={"result": value})
        if case == "null":
            return HookResult(action="modify", data={"result": None})
        if case == "missing":
            return HookResult(action="modify", data={})
        if case == "inplace":
            event["result"]["output"].pop(MEDIA)
        return HookResult()

    result, pixels = rich_images(fmt)
    tool = Tool(result)
    try:
        await run(ctx, provider, coord, Hooks(post=post), {"image": tool}, limit=limit)
        assert len(tool.executions) == 1
        assert len(publications) == 1
        expected = 2 if case == "keep" else 1 if case == "partial" else 0
        rows = tool_messages(ctx)
        if expected:
            refs = rows[0]["metadata"][MEDIA]["refs"]
            assert "observation" in rows[0]["content"]
            assert len(refs) == expected
            assert publications[0] == (refs, "retain")
            for item in refs:
                assert cap.validate(item, purpose="model") == {"valid": True}
        else:
            assert "metadata" not in rows[0]
            assert publications[0][1] == "drop"
        assert saved and len(provider.requests) == 2
        journal = list(store.journal())
        assert any(row.get("state") == "committed" for row in journal) is bool(expected)
        persisted = json.dumps([ctx.messages, saved, journal, provider.requests])
        for value in pixels:
            assert value not in persisted
        # Report is observational, but its component argument must be real F1-valid.
        readiness = cap.report("pairing-test", event="readiness", outcome="read")
        if case != "exception":
            assert "loop-events" in readiness["consumers"]
    finally:
        store.close()


@pytest.fixture
def bridge_installer():
    if not BRIDGE:
        pytest.skip("explicit immutable Foundation bridge candidate not selected")
    # The bridge is a new, independently owned module. Load its exact candidate
    # while F1 remains the separately selected immutable library revision.
    spec = importlib.util.spec_from_file_location("pairing_media_hook_bridge", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.install_media_hook_bridge


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["qualified", "safe-negative"])
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("case", [
    "keep", "partial", "missing", "null-injection", "inplace-remove",
    "inplace-null", "exception", "process-exception", "late-uncovered", "ordinary-text",
    "ordinary-error", "text-hook-adds-refs",
])
async def test_unsealed_bridge_core_registry_store_and_events_fail_closed(
    foundation, bridge_installer, tmp_path, case, limit, profile
):
    if os.environ.get("AMPLIFIER_MEDIA_BRIDGE_PROFILE") != profile:
        pytest.skip("this candidate does not qualify the other bridge profile")
    media, checkpoint = foundation
    text_only = case in {"ordinary-text", "ordinary-error", "text-hook-adds-refs"}
    session = AmplifierSession(
        {"session": {"orchestrator": "fixture", "context": "fixture"}},
        session_id="session-test",
    )
    core = session.coordinator
    ctx, provider = Context(), Provider()
    await core.mount("context", ctx)
    bridge = bridge_installer(core)
    store = media.SessionMediaStore(
        tmp_path / "media", core.session_id,
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    cap = media.bind_session_media(core, store=store, enabled=True)
    publications, posts, processed, emitted, saved = [], [], [], [], []
    original_publish = cap.publish

    async def publish(refs, *, origin, disposition):
        publications.append((deepcopy(refs), disposition))
        return await original_publish(refs, origin=origin, disposition=disposition)

    cap.publish = publish

    def save():
        messages = deepcopy(ctx.messages)
        (tmp_path / "transcript.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in messages)
        )
        receipt = checkpoint.acknowledge_legacy_checkpoint(
            tmp_path, session_id=core.session_id, expected_messages=messages
        )
        store.acknowledge_checkpoint(receipt)
        saved.append(messages)
        return True

    core.register_capability("session.durable_checkpoint", save)

    async def handler(event, data):
        posts.append(deepcopy(data))
        if case == "text-hook-adds-refs":
            return HookResult(action="modify", data={"result": {
                "success": True, "output": {
                    "note": "hook text", MEDIA: {"refs": [ref("unowned")]},
                },
            }})
        if case == "partial":
            value = deepcopy(data["result"])
            value["output"][MEDIA]["refs"] = value["output"][MEDIA]["refs"][:1]
            return HookResult(action="modify", data={"result": value})
        if case == "missing":
            return HookResult(action="modify", data={})
        if case == "null-injection":
            return HookResult(action="modify", data={"result": None})
        if case == "inplace-remove":
            data["result"]["output"].pop(MEDIA)
        if case == "inplace-null":
            data["result"] = None
        if case == "exception":
            raise RuntimeError("synthetic callback failure")
        return HookResult()

    bridge.hooks.register("tool:post", handler, priority=10, name="observed")
    if case == "null-injection":
        bridge.hooks.register(
            "tool:post",
            lambda event, data: HookResult(
                action="inject_context", context_injection="fixture reminder"
            ),
            priority=20,
            name="later-injection",
        )
    if case == "late-uncovered":
        core.hooks.register("tool:post", lambda e, d: HookResult(), name="uncovered")

    class CoordinatorObserver:
        def __getattr__(self, name):
            return getattr(core, name)

        async def process_hook_result(self, result, event, name):
            if event == "tool:post":
                processed.append(name)
                if case == "process-exception":
                    raise RuntimeError("synthetic processed failure")
            return await core.process_hook_result(result, event, name)

    class HooksObserver:
        async def emit(self, event, data):
            if event == "tool:post":
                emitted.append(deepcopy(data))
            return await core.hooks.emit(event, data)

    coord = CoordinatorObserver()
    result, pixels = rich_images()
    if text_only:
        result, pixels = ToolResult(output="ordinary text"), []
    if case == "ordinary-error":
        result = ToolResult(
            success=False, output={"stderr": "useful failure", "returncode": 1},
            error={"message": "failed"},
        )
    tool = Tool(result)
    try:
        await run(ctx, provider, coord, HooksObserver(), {"image": tool}, limit=limit)
        assert len(tool.executions) == 1
        # New graph bridge rejects these unenrolled arbitrary observer wrappers
        # before emission. It must not replay via legacy hooks after refusal.
        expected_emissions = 0 if profile == "qualified" and not text_only else 1
        assert len(posts) == len(emitted) == len(processed) == expected_emissions
        assert len(publications) == (0 if text_only else 1)
        # Even a positive-capable implementation cannot qualify an unsealed
        # module/host graph. Positive full initialization has a separate fixture.
        expected = 0
        message = tool_messages(ctx)[0]
        if expected:
            refs = message["metadata"][MEDIA]["refs"]
            assert len(refs) == expected
            assert publications[0] == (refs, "retain")
            for item in refs:
                assert cap.validate(item, purpose="model") == {"valid": True}
        else:
            assert "metadata" not in message
            if not text_only:
                assert publications[0][1] == "drop"
        assert len(provider.requests) == 2 and saved
        readiness = cap.report("pairing-test", event="readiness", outcome="read")
        assert not readiness["consumers"]
        if profile == "safe-negative" and case != "process-exception" and not text_only:
            assert "Image unavailable" in message["content"]
        if case == "null-injection":
            if profile == "safe-negative":
                assert message["content"].startswith("null")
                assert any(m["content"] == "fixture reminder" for m in ctx.messages)
            else:
                assert not any(m["content"] == "fixture reminder" for m in ctx.messages)
        if case == "ordinary-text":
            assert message["content"] == "ordinary text"
        if case == "ordinary-error":
            assert posts[0]["result"]["success"] is False
            assert posts[0]["result"]["error"] == {"message": "failed"}
            assert json.loads(message["content"]) == {
                "stderr": "useful failure", "returncode": 1,
            }
        if case == "text-hook-adds-refs":
            assert "hook text" in message["content"]
            assert "Image unavailable" in message["content"]
            assert MEDIA not in message["content"]
        if case == "keep" and profile == "safe-negative":
            # f943 deliberately supplies no authoritative effective ordinary
            # result. Preserve explicit null, never reconstruct old useful text.
            assert message["content"].startswith("null")
            assert "useful text" not in message["content"]
            assert "Image unavailable" in message["content"]
        persisted = json.dumps([
            ctx.messages, saved, list(store.journal()), emitted, provider.requests,
        ])
        for value in pixels:
            assert value not in persisted
    finally:
        store.close()
        await session.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("vision", [False, True])
async def test_actual_f1_private_provider_complete_preserves_canonical_state(
    foundation, tmp_path, limit, vision, fmt
):
    from amplifier_foundation.session.media_provider import MediaPreparedProvider

    media, checkpoint = foundation
    ctx, _, raw, coord, capabilities, _ = setup()
    capabilities.pop(MEDIA)
    coord.register_capability = capabilities.__setitem__
    store = media.SessionMediaStore(
        tmp_path / "media", coord.session_id,
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    cap = media.bind_session_media(coord, store=store, enabled=True)
    projected = []

    class WireFixture:
        default_model = "fixture"

        def get_media_request_support(self, **kwargs):
            return {"v": 1, "positions": ["tool"] if vision else []}

        async def request_budget(self, request, *, request_options, context_estimate):
            cap.validate(request, purpose="provider-count", request_options=request_options)
            return {
                "estimated_input_tokens": 1, "input_limit_tokens": 100,
                "context_token_budget": context_estimate,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 1},
            }

        async def complete(self, request, **kwargs):
            cap.validate(request, purpose="provider-send", request_options=kwargs)
            projected.append(request.model_dump())
            return SimpleNamespace(
                content="" if len(projected) == 1 else "done",
                metadata=None, calls=raw.calls if len(projected) == 1 else [],
            )

        def parse_tool_calls(self, response):
            return response.calls

    adapter = MediaPreparedProvider(
        WireFixture(), cap,
        provider_identity={"provider": "fixture", "account": "test", "generation": 1},
    )
    canonical = []

    async def messages(provider=None):
        assert provider is adapter
        canonical.append(deepcopy(ctx.messages))
        return ctx.messages

    ctx.get_messages_for_request = messages

    def save():
        (tmp_path / "transcript.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ctx.messages)
        )
        ack = checkpoint.acknowledge_legacy_checkpoint(
            tmp_path, session_id=coord.session_id, expected_messages=ctx.messages
        )
        store.acknowledge_checkpoint(ack)
        return True

    capabilities["session.durable_checkpoint"] = save
    result, pixels = rich_images(fmt)
    try:
        await run(ctx, adapter, coord, tools={"image": Tool(result)}, limit=limit)
        assert len(projected) == 2
        tool_message = next(m for m in projected[1]["messages"] if m["role"] == "tool")
        if vision:
            images = [b for b in tool_message["content"] if b["type"] == "image"]
            assert [b["source"]["data"] for b in images] == pixels, tool_message
        else:
            assert "Image unavailable" in json.dumps(tool_message["content"])
            assert not any(value in json.dumps(projected) for value in pixels)
        assert tool_message["tool_call_id"] == "call-1"
        for value in pixels:
            assert value not in json.dumps([ctx.messages, canonical, list(store.journal())])
            assert value not in (tmp_path / "transcript.jsonl").read_text()
    finally:
        await adapter.aclose()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("kind", ["rich", "refs", "text", "failure"])
async def test_actual_disabled_f1_preserves_optional_text_without_media_calls(
    foundation, kind, limit
):
    media, _ = foundation
    ctx, _, provider, coord, capabilities, _ = setup()
    touched = []

    class NoIOStore:
        session_id = coord.session_id

        def __getattr__(self, name):
            touched.append(name)
            raise AssertionError("Disabled media must not touch its backing store")

    cap = media.SessionMediaCapability(NoIOStore(), enabled=False)
    calls = []
    for name in ("ingest", "publish", "report"):
        original = getattr(cap, name)

        if name == "report":
            def report(*args, _original=original, **kwargs):
                calls.append("report")
                return _original(*args, **kwargs)
            cap.report = report
        else:
            async def operation(*args, _name=name, _original=original, **kwargs):
                calls.append(_name)
                return await _original(*args, **kwargs)
            setattr(cap, name, operation)
    capabilities[MEDIA] = cap
    bridge_calls = []

    async def forbidden_bridge(*args, **kwargs):
        bridge_calls.append(True)
        raise AssertionError("Disabled media must use the ordinary sanitized path")

    capabilities["session.media.tool_post"] = forbidden_bridge
    result, pixels = rich_images()
    if kind == "refs":
        result = ToolResult(output={"note": "useful text", MEDIA: {"refs": [ref("old")]}})
    elif kind == "text":
        result = ToolResult(output="useful text")
    elif kind == "failure":
        result = ToolResult(
            success=False, output={"note": "useful text", "returncode": 1},
            error={"message": "actual failure"},
        )
    original_result = result.model_dump()
    posts = []
    hooks = HookRegistry()

    async def post(event, data):
        posts.append(deepcopy(data))
        return HookResult()

    hooks.register("tool:post", post)
    tool = Tool(result)
    await run(ctx, provider, coord, hooks, {"image": tool}, limit=limit)
    assert len(tool.executions) == len(posts) == 1
    assert not calls and not touched and not bridge_calls
    assert not cap._consumers
    assert cap.enabled is False
    assert result.model_dump() == original_result
    message = tool_messages(ctx)[0]
    assert "useful text" in message["content"]
    assert "Internal error" not in message["content"]
    assert "metadata" not in message
    assert posts[0]["result"]["success"] == result.success
    assert posts[0]["result"]["error"] == result.error
    if kind in {"rich", "refs"}:
        assert "Image unavailable" in message["content"]
        assert MEDIA not in json.dumps(posts)
        assert "content" not in posts[0]["result"]
    else:
        assert "Image unavailable" not in message["content"]
    if kind == "rich":
        assert "observation" in message["content"]
    for value in pixels:
        assert value not in json.dumps([posts, ctx.messages, provider.requests])
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_enabled_f1_ingest_refusal_is_not_optional_disabled_fallback(foundation):
    from amplifier_module_loop_events._media import capability, stage_result

    media, _ = foundation
    _, _, _, coord, capabilities, _ = setup()
    calls = []

    class RefusingStore:
        session_id = coord.session_id

        async def ingest(self, *args, **kwargs):
            calls.append(kwargs)
            raise media.MediaRefusal("required policy refuses this media")

    cap = media.SessionMediaCapability(RefusingStore(), enabled=True)
    capabilities[MEDIA] = cap
    assert capability(coord) is cap
    result, _ = rich_images()
    with pytest.raises(ValueError, match="Media staging failed"):
        await stage_result(result, cap, {
            "kind": "tool", "session_id": coord.session_id, "id": "actual-call",
        })
    assert len(calls) == 1
    assert calls[0]["policy"] is None
    assert not cap._consumers


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("preliminary_error", [False, True])
async def test_actual_f1_final_budget_and_complete_share_full_canonical_candidate(
    foundation, tmp_path, limit, preliminary_error, caplog
):
    """A context-selected count is not a final request/tools/options count."""
    from amplifier_foundation.session.media_provider import MediaPreparedProvider

    media, checkpoint = foundation
    ctx, _, raw, coord, capabilities, _ = setup()
    capabilities.pop(MEDIA)
    coord.register_capability = capabilities.__setitem__
    store = media.SessionMediaStore(
        tmp_path / "media", coord.session_id,
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    cap = media.bind_session_media(coord, store=store, enabled=True)
    counts, sends, order = [], [], []

    class Wire:
        default_model = "fixture"

        def get_media_request_support(self, **kwargs):
            return {"v": 1, "positions": ["tool"]}

        async def request_budget(self, request, *, request_options, context_estimate):
            assert type(context_estimate) is int and context_estimate > 0
            cap.validate(request, purpose="provider-count", request_options=request_options)
            if preliminary_error and not request.tools:
                raise ValueError("private preliminary payload " + rich_images()[1][0])
            counts.append((request, request.model_dump(), deepcopy(request_options)))
            order.append("count")
            return {
                "estimated_input_tokens": 10, "input_limit_tokens": 100,
                "context_token_budget": 100,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 10},
            }

        async def complete(self, request, **kwargs):
            cap.validate(request, purpose="provider-send", request_options=kwargs)
            sends.append((request, request.model_dump(), deepcopy(kwargs)))
            order.append("send")
            return SimpleNamespace(
                content="" if len(sends) == 1 else "done", metadata=None,
                calls=raw.calls if len(sends) == 1 else [],
            )

        def parse_tool_calls(self, response):
            return response.calls

    adapter = MediaPreparedProvider(
        Wire(), cap,
        provider_identity={"provider": "fixture", "account": "test", "generation": 1},
    )

    async def messages(provider=None):
        assert provider is adapter
        if any(MEDIA in (row.get("metadata") or {}) for row in ctx.messages):
            # Actual context APIs can count a selection before the loop adds
            # tools, reasoning and finalization reminder. It is not dispatch fit.
            preliminary = ChatRequest(messages=[Message(**row) for row in ctx.messages])
            await provider.request_budget(preliminary, request_options={}, context_estimate=7)
        return ctx.messages

    ctx.get_messages_for_request = messages

    def save():
        (tmp_path / "transcript.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ctx.messages)
        )
        receipt = checkpoint.acknowledge_legacy_checkpoint(
            tmp_path, session_id=coord.session_id, expected_messages=ctx.messages
        )
        store.acknowledge_checkpoint(receipt)
        order.append("checkpoint")
        return True

    capabilities["session.durable_checkpoint"] = save
    result, pixels = rich_images()
    try:
        if preliminary_error:
            with pytest.raises(ValueError, match="private diagnostics were not retained") as caught:
                await EventDrivenOrchestrator({"max_iterations": limit}).execute(
                    "inspect", ctx, {"selected": adapter}, {"image": Tool(result)}, Hooks(), coord
                )
            assert len(sends) == 1
            for value in pixels:
                assert value not in str(caught.value) + caplog.text
            from amplifier_module_loop_events._media import request_messages
            # The historical committed ref can still reach context counting when
            # outgoing media is ineligible. Privacy must not follow eligibility.
            with pytest.raises(ValueError, match="private diagnostics were not retained") as blocked:
                await request_messages(ctx, adapter, media=None, private_media=cap)
            for value in pixels:
                assert value not in str(blocked.value) + caplog.text
            assert len(sends) == 1
            return
        answer = await EventDrivenOrchestrator({
            "max_iterations": limit, "reasoning_effort": "low", "extended_thinking": True,
        }).execute(
            "inspect", ctx, {"selected": adapter}, {"image": Tool(result)},
            Hooks(post=lambda event: HookResult(
                action="inject_context", context_injection="ephemeral observation",
                ephemeral=True, append_to_last_tool_result=True,
            )),
            coord,
        )
        assert answer == "done"
        assert len(sends) == 2
        # The final count must use the exact private projection sent, including
        # tool schema, reasoning, reminder and effective kwargs.
        assert counts[-1][0] is sends[-1][0]
        assert counts[-1][1] == sends[-1][1]
        assert counts[-1][2] == sends[-1][2] == {"extended_thinking": True}
        assert counts[-1][1]["tools"]
        assert counts[-1][1]["reasoning_effort"] == "low"
        assert "ephemeral observation" in json.dumps(counts[-1][1]["messages"])
        if limit == 1:
            assert "orchestrator-loop-limit" in json.dumps(counts[-1][1]["messages"])
        assert "ephemeral observation" not in json.dumps(ctx.messages)
        assert order.index("checkpoint") < order.index("count", order.index("checkpoint"))
        assert order[-2:] == ["count", "send"]
        for value in pixels:
            assert value not in json.dumps(ctx.messages)
    finally:
        await adapter.aclose()
        store.close()


@pytest.mark.asyncio
async def test_actual_f1_owned_warning_only_without_count(foundation, tmp_path):
    from amplifier_foundation.session.media_provider import MediaPreparedProvider

    from amplifier_module_loop_events._media import check_final_budget

    media, checkpoint = foundation
    sent = []
    store = media.SessionMediaStore(
        tmp_path / "media", "session-test",
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    cap = media.SessionMediaCapability(store, enabled=True)
    origin = {"kind": "tool", "session_id": "session-test", "id": "fixture"}
    rich, _ = rich_images()
    staged = await cap.ingest(
        rich, source_format="core-tool-result", origin=origin, policy=None
    )
    refs = staged["staged_refs"]
    await cap.publish(refs, origin=origin, disposition="retain")
    row = {"role": "user", "content": "image observation", "metadata": {MEDIA: {"refs": refs}}}
    (tmp_path / "transcript.jsonl").write_text(json.dumps(row) + "\n")
    receipt = checkpoint.acknowledge_legacy_checkpoint(
        tmp_path, session_id="session-test", expected_messages=[row]
    )
    store.acknowledge_checkpoint(receipt)

    class TextOnly:
        default_model = "text-fixture"

        async def complete(self, request, **kwargs):
            cap.validate(request, purpose="provider-send", request_options=kwargs)
            sent.append(request.model_dump())
            return "done"

    adapter = MediaPreparedProvider(
        TextOnly(), cap,
        provider_identity={"provider": "fixture", "account": "test", "generation": 1},
    )
    request = ChatRequest(messages=[Message(**row)])
    before = request.model_dump()
    try:
        await check_final_budget(adapter, request, {})
        assert await adapter.complete(request) == "done"
        assert "Image unavailable" in json.dumps(sent)
        assert request.model_dump() == before
        assert not hasattr(adapter, "request_budget")
    finally:
        await adapter.aclose()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["over-limit", "malformed", "cancel-before", "cancel-after", "witness-refusal"])
async def test_actual_f1_local_abandon_releases_exact_candidate_not_shared_provider(
    foundation, tmp_path, decision
):
    from amplifier_foundation.session.media_provider import MediaPreparedProvider

    from amplifier_module_loop_events._media import complete_media_request

    media, checkpoint = foundation
    if not callable(getattr(MediaPreparedProvider, "release_media_request", None)):
        pytest.skip("exact public per-candidate abandon API not in selected candidate")
    store = media.SessionMediaStore(
        tmp_path / "media", "session-test",
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=checkpoint.verified_checkpoint_refs,
    )
    cap = media.SessionMediaCapability(store, enabled=True)
    rich, _ = rich_images()
    origin = {"kind": "tool", "session_id": "session-test", "id": "actual"}
    staged = await cap.ingest(rich, source_format="core-tool-result", origin=origin, policy=None)
    refs = staged["staged_refs"]
    await cap.publish(refs, origin=origin, disposition="retain")
    row = {"role": "user", "content": "synthetic image", "metadata": {MEDIA: {"refs": refs}}}
    (tmp_path / "transcript.jsonl").write_text(json.dumps(row) + "\n")
    store.acknowledge_checkpoint(checkpoint.acknowledge_legacy_checkpoint(
        tmp_path, session_id="session-test", expected_messages=[row]
    ))
    cancelled, budget_calls, sends, releases = [], [], [], []
    options = {"temperature": 0.2}
    request = ChatRequest(messages=[Message(**row)])

    class Wire:
        default_model = "fixture"

        def get_media_request_support(self, **kwargs):
            return {"v": 1, "positions": ["user"]}

        async def request_budget(self, private, *, context_estimate, request_options):
            cap.validate(private, purpose="provider-count", request_options=request_options)
            budget_calls.append(private)
            if decision == "cancel-after":
                cancelled.append(True)
            if decision == "malformed":
                return {"estimated_input_tokens": True, "input_limit_tokens": 100}
            if decision == "witness-refusal":
                return None
            return {
                "estimated_input_tokens": 101 if decision == "over-limit" else 1,
                "input_limit_tokens": 100,
            }

        async def complete(self, private, **kwargs):
            sends.append(private)
            return "done"

    adapter = MediaPreparedProvider(
        Wire(), cap,
        provider_identity={"provider": "fixture", "account": "test", "generation": 1},
    )
    release = adapter.release_media_request

    def observed_release(candidate, *, request_options=None):
        result = release(candidate, request_options=request_options)
        releases.append((candidate, deepcopy(request_options), result))
        return result

    adapter.release_media_request = observed_release
    try:
        if decision == "cancel-before":
            await adapter.request_budget(request, request_options=options, context_estimate=7)
            cancelled.append(True)
        expected = __import__("asyncio").CancelledError if "cancel" in decision else (ValueError, TypeError)
        with pytest.raises(expected):
            await complete_media_request(
                adapter, request, options, is_cancelled=lambda: bool(cancelled),
            )
        assert not sends
        assert len(releases) == 1
        assert releases[0][0] is request and releases[0][1] == options
        # The adapter already releases its candidate on its own witness refusal;
        # a subsequent loop-owned abandon is honestly idempotent False.
        assert releases[0][2] is (decision != "witness-refusal")
        assert not adapter._active and adapter._candidate is None
        assert adapter._closed is False
        assert release(request, request_options=options) is False
        # Shared wrapper remains usable for a distinct later request.
        plain = ChatRequest(messages=[Message(role="user", content="later text")])
        assert await adapter.complete(plain) == "done"
    finally:
        await adapter.aclose()
        store.close()