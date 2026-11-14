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
from amplifier_core.models import HookResult

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


def rich_images():
    from PIL import Image

    encoded = []
    for color in ("red", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (3, 2), color).save(output, format="PNG")
        encoded.append(base64.b64encode(output.getvalue()).decode())
    return ToolResult(output={"note": "useful text"}, content=[
        {"type": "text", "text": "observation"},
        *[
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": value},
            }
            for value in encoded
        ],
    ]), encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("case", ["keep", "partial", "null", "missing", "inplace", "exception"])
async def test_actual_f1_store_single_origin_disposition_and_checkpoint(
    foundation, tmp_path, case, limit
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

    result, pixels = rich_images()
    tool = Tool(result)
    try:
        await run(ctx, provider, coord, Hooks(post=post), {"image": tool}, limit=limit)
        assert len(tool.executions) == 1
        assert len(publications) == 1
        expected = 2 if case == "keep" else 1 if case == "partial" else 0
        rows = tool_messages(ctx)
        if expected:
            refs = rows[0]["metadata"][MEDIA]["refs"]
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
async def test_actual_bridge_core_registry_store_and_events_end_to_end(
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
        assert len(tool.executions) == len(posts) == len(emitted) == len(processed) == 1
        assert len(publications) == (0 if text_only else 1)
        expected = (
            (2 if case == "keep" else 1 if case == "partial" else 0)
            if profile == "qualified" else 0
        )
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
        if profile == "safe-negative" or case in {
            "exception", "process-exception", "late-uncovered",
            "ordinary-text", "ordinary-error", "text-hook-adds-refs",
        }:
            readiness = cap.report("pairing-test", event="readiness", outcome="read")
            assert not readiness["consumers"]
            if case != "process-exception" and not text_only:
                assert "Image unavailable" in message["content"]
        else:
            assert "loop-events" in cap.report(
                "pairing-test", event="readiness", outcome="read"
            )["consumers"]
        if case == "null-injection":
            assert message["content"].startswith("null")
            assert any(m["content"] == "fixture reminder" for m in ctx.messages)
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
        if profile == "safe-negative" and case == "keep":
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
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("vision", [False, True])
async def test_actual_f1_private_provider_complete_preserves_canonical_state(
    foundation, tmp_path, limit, vision
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
    result, pixels = rich_images()
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