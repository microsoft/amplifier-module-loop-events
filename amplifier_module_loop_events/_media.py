"""Structural media-v1 integration. Storage and private pixels stay with the host."""

import asyncio
import inspect
import json
import logging
import math
from collections.abc import Mapping
from copy import deepcopy

from amplifier_core import ToolResult
from amplifier_core.models import HookResult

logger = logging.getLogger(__name__)

MEDIA = "experimental.media.v1"
TOOL_POST_BOUNDARY = "session.media.tool_post"
BOUNDARY_UNAVAILABLE = (
    "[Image unavailable: a complete media tool-post boundary is not installed; "
    "no pixels were sent.]"
)
UNAVAILABLE = (
    "[Image unavailable to this session: compatible media handling is not installed; "
    "no pixels were sent.]"
)


class MediaCheckpointError(RuntimeError):
    """Settled evidence must not be counted/sent before its durable checkpoint."""


def capability(coordinator):
    getter = getattr(coordinator, "get_capability", None)
    media = getter(MEDIA) if callable(getter) else None
    if (
        type(getattr(media, "version", None)) is int
        and media.version == 1
        # An installed neutral/disabled capability is not an ingress grant.
        # Treat only explicit False as unavailable; older structural consumers
        # need not expose this optional host-owned activation flag.
        and getattr(media, "enabled", None) is not False
        and all(
            callable(getattr(media, op, None))
            for op in (
                "ingest", "publish", "validate", "resolve", "prepare_request", "report"
            )
        )
    ):
        return media
    return None


def checkpoint_callback(coordinator):
    getter = getattr(coordinator, "get_capability", None)
    checkpoint = getter("session.durable_checkpoint") if callable(getter) else None
    return checkpoint if callable(checkpoint) else None


def post_boundary(coordinator):
    getter = getattr(coordinator, "get_capability", None)
    boundary = getter(TOOL_POST_BOUNDARY) if callable(getter) else None
    return boundary if callable(boundary) else None


def refs_from_result(value):
    """Only the actual effective ToolResult.output carrier is eligible."""
    if isinstance(value, ToolResult):
        value = value.model_dump()
    if not isinstance(value, Mapping) or not isinstance(value.get("output"), Mapping):
        return []
    output = value["output"]
    if MEDIA not in output:
        return []
    envelope = output[MEDIA]
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"refs"}
        or not isinstance(envelope["refs"], list)
    ):
        raise ValueError("Invalid media-v1 output carrier")
    return deepcopy(envelope["refs"])


def without_media(value, reason=UNAVAILABLE):
    """Remove only known canonical image fields, never scan arbitrary JSON text."""
    if isinstance(value, ToolResult):
        data = value.model_dump()
    else:
        data = deepcopy(value)
    if not isinstance(data, Mapping):
        return data
    data = dict(data)
    output = data.get("output")
    had_media = isinstance(output, Mapping) and MEDIA in output
    if had_media:
        output = {k: v for k, v in output.items() if k != MEDIA}
    content = data.get("content")
    texts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "image":
                had_media = True
            elif isinstance(block, Mapping) and block.get("type") == "text":
                texts.append(block["text"])
    if had_media:
        # Core rich content has only text/image blocks. Do not keep pixel transport.
        data.pop("content", None)
        if isinstance(output, (dict, list)):
            output = json.dumps(output)
        elif output is None:
            output = ""
        else:
            output = str(output)
        data["output"] = "\n\n".join(p for p in [output, *texts, reason] if p)
    return data


async def stage_result(result, media, origin):
    has_media = isinstance(result, ToolResult) and (
        any(block.type == "image" for block in (result.content or []))
        or isinstance(result.output, Mapping) and MEDIA in result.output
    )
    if media is not None and has_media:
        try:
            staged = await media.ingest(
                result, source_format="core-tool-result", origin=origin, policy=None
            )
            value = staged["value"]
            if not isinstance(value, ToolResult):
                raise TypeError("Media ingest must return a Core ToolResult")
            data = value.model_dump()
            if any(b.get("type") == "image" for b in data.get("content", [])):
                raise ValueError("Media ingest returned pixel transport")
            return data
        except Exception:  # noqa: BLE001 - sanitize arbitrary decoder/store diagnostics
            # A decoder error may include input bytes. Never expose it to logs or
            # the canonical safety-net response. Tool effects are not retried.
            raise ValueError(
                "Media staging failed; tool effect was not rolled back"
            ) from None
    if isinstance(result, ToolResult):
        return without_media(result)
    return result.model_dump() if hasattr(result, "model_dump") else str(result)


async def report_ready(media, *, complete):
    # Mount-time structural availability is not evidence about the actual chain.
    if media is not None and complete:
        receipt = media.report(
            "loop-events",
            event="consumer-ready",
            outcome="ready",
            details={"contract": 1, "role": "loop"},
        )
        if inspect.isawaitable(receipt):
            await receipt


async def tool_post(event, *, hooks, coordinator):
    """One host-owned emission/processing, or one unqualified legacy emission."""
    # Text-only results do not need a media witness. A safe-negative bridge must
    # not replace their ordinary output with its intentionally unavailable result.
    boundary = post_boundary(coordinator) if refs_from_result(event.get("result")) else None
    if boundary is not None:
        try:
            outcome = await boundary(event, hooks=hooks, coordinator=coordinator)
        except Exception:  # noqa: BLE001 - never replay a possibly emitted callback
            raise ValueError("Media tool-post boundary failed; no replay attempted") from None
        if not isinstance(outcome, Mapping):
            raise ValueError("Media tool-post boundary returned an invalid outcome")
        result = outcome.get("hook_result")
        if not isinstance(result, HookResult):
            raise ValueError("Media tool-post boundary did not return a processed HookResult")
        complete = outcome.get("complete") is True and "result" in outcome
        # Even explicit null is authoritative. Never recover the original event
        # after a boundary call: it may have emitted effects before refusing.
        value = outcome.get("result")
        if result.action == "deny":
            value = f"Error: {result.reason or 'Tool result denied'}"
        return result, value, complete
    result = await hooks.emit("tool:post", event)
    if coordinator is not None:
        result = await coordinator.process_hook_result(
            result, "tool:post", event["tool_name"]
        )
    return result, effective_result(result, event), False


async def drop_staged(media, staged_value, surviving, *, origin):
    if media is None:
        return
    staged = refs_from_result(staged_value)
    dropped = [ref for ref in staged if ref not in surviving]
    if dropped:
        try:
            await media.publish(dropped, origin=origin, disposition="drop")
        except Exception:  # noqa: BLE001 - do not serialize private store diagnostics
            raise ValueError("Media staging drop failed; no tool replay attempted") from None


async def drop_after_failure(media, staged_value, *, origin):
    """Cleanup must not replace a pending cancellation with a recoverable error."""
    try:
        await drop_staged(media, staged_value, [], origin=origin)
    except Exception:  # noqa: BLE001 - original failure/cancellation remains authoritative
        # Unsettled staging stays owner-reconcilable; never claim it was deleted.
        return False
    return True


def effective_result(post_result, event_data):
    """Presence/action, not identity or truthiness, determines acceptance."""
    if post_result.action == "deny":
        return f"Error: {post_result.reason or 'Tool result denied'}"
    data = post_result.data
    if isinstance(data, Mapping) and "result" in data:
        return data["result"]
    if post_result.action == "modify" or "result" not in event_data:
        return None
    return event_data["result"]


def serialize_result(value, *, modified):
    # Preserve ordinary output semantics even when HookRegistry copies a no-op event.
    if not modified and isinstance(value, Mapping) and "output" in value:
        fields = {"success", "output", "error", "content"}
        if set(value) <= fields:
            result = ToolResult(**value)
            if result.output is None and result.content:
                return "\n\n".join(block.text for block in result.content)
            output = result.get_serialized_output()
            if result.content:
                output += "\n\n" + "\n\n".join(block.text for block in result.content)
            return output
    if isinstance(value, (dict, list)) or value is None:
        return json.dumps(value)
    return str(value)


def normalize_effective(value):
    if isinstance(value, ToolResult):
        value = value.model_dump()
    if isinstance(value, Mapping) and any(
        isinstance(block, Mapping) and block.get("type") == "image"
        for block in (value.get("content") or [])
    ):
        # A post-hook cannot introduce an un-staged raw rich image into history.
        value = without_media(value)
    return value


async def settle_result(value, *, media, origin, checkpoint, modified, complete):
    value = normalize_effective(value)
    refs = refs_from_result(value)
    metadata = {}
    if refs:
        if media is None or checkpoint is None or not complete:
            value = without_media(
                value,
                BOUNDARY_UNAVAILABLE
                if not complete
                else UNAVAILABLE
                if media is None
                else (
                    "[Image unavailable: durable media checkpoint is not installed; "
                    "no pixels were sent.]"
                ),
            )
        else:
            try:
                published = await media.publish(
                    refs, origin=origin, disposition="retain"
                )
            except Exception:  # noqa: BLE001 - never persist raw storage diagnostics
                raise ValueError(
                    "Media publication refused the effective refs"
                ) from None
            accepted = published["refs"]
            # Publication validates scope and origin; it cannot replace/reorder output.
            if accepted != refs:
                raise ValueError("Media publication changed the effective refs")
            metadata[MEDIA] = {"refs": deepcopy(accepted)}
    return serialize_result(value, modified=modified), metadata


async def checkpoint_settled(checkpoint, *, has_media):
    if checkpoint is None:
        if has_media:
            raise MediaCheckpointError("Media requires a durable checkpoint")
        return
    try:
        acknowledgment = checkpoint()
        if inspect.isawaitable(acknowledgment):
            acknowledgment = await acknowledgment
    except Exception as exc:
        raise MediaCheckpointError("Settled tool checkpoint failed") from exc
    if acknowledgment is False or (has_media and acknowledgment is None):
        raise MediaCheckpointError("Settled tool checkpoint was not acknowledged")


async def request_messages(context, provider, *, media, private_media=None):
    getter = getattr(context, "get_messages_for_request", None)
    if getter is not None:
        # Inspect before calling: catching TypeError could repeat a compaction effect.
        parameters = inspect.signature(getter).parameters.values()
        supports_provider = any(
            (p.name == "provider" and p.kind != inspect.Parameter.POSITIONAL_ONLY)
            or p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters
        )
        try:
            messages = (
                await getter(provider=provider) if supports_provider else await getter()
            )
        except Exception:
            if media is not None or private_media is not None:
                raise ValueError(
                    "Media context preparation failed; private diagnostics were not retained"
                ) from None
            raise
    else:
        messages = await context.get_messages()
    messages = deepcopy(list(messages))
    if media is None:
        for message in messages:
            metadata = message.get("metadata")
            if isinstance(metadata, dict) and MEDIA in metadata:
                metadata.pop(MEDIA)
                content = message.get("content", "")
                if isinstance(content, list):
                    content.append({"type": "text", "text": UNAVAILABLE})
                else:
                    message["content"] = f"{content}\n\n{UNAVAILABLE}"
    return messages


async def check_final_budget(provider, request, request_options):
    """Count the exact assembled canonical media candidate, never hydrate here.

    Context selection/compaction can count earlier candidates. Its fit is not a
    witness for the later tools/reminder/options assembled by this loop.
    """
    if not any(MEDIA in (message.metadata or {}) for message in request.messages):
        return
    budget = getattr(provider, "request_budget", None)
    if not callable(budget):
        if not await check_warning_only_fallback(provider, request, request_options):
            raise ValueError("Media request requires a budget or owned warning-only fallback")
        return
    parameters = inspect.signature(budget).parameters
    options_parameter = parameters.get("request_options")
    if options_parameter is None or options_parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        raise ValueError("Media budget requires named request_options support")
    kwargs = {"request_options": deepcopy(request_options)}
    # Providers use this only as a local text-coordinate hint, not an image or
    # provider count. Core has no public context-estimate getter; use the same
    # deliberately partial chars/4 convention as the standalone streaming loop.
    if "context_estimate" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["context_estimate"] = estimate_message_text(request)
    result = budget(request, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if result is None:
        if await check_warning_only_fallback(provider, request, request_options):
            return
        raise ValueError("Media budget is unavailable for this request")
    if not isinstance(result, Mapping):
        raise TypeError("Media budget is unavailable for this request")
    if result.get("fits") is False:
        raise ValueError("Media request exceeds the provider budget")
    # Accept the owner's numeric estimate without calling it an exact count.
    # A malformed or missing cost/limit cannot certify a media fit.
    cost = result.get("estimated_input_tokens")
    limit = result.get("input_limit_tokens")
    if (
        type(cost) not in (int, float) or type(limit) not in (int, float)
        or not math.isfinite(cost) or not math.isfinite(limit)
        or cost < 0 or limit <= 0
    ):
        raise ValueError("Media budget lacks a valid numeric cost and limit")
    if cost > limit:
        raise ValueError("Media request exceeds the provider budget")
    target = result.get("context_token_budget")
    if target is not None and (type(target) is not int or target < 0):
        raise ValueError("Media budget has an invalid context rebuild target")


def estimate_message_text(request):
    """Partial local message-coordinate hint: never pixels, tools or a fit."""
    tokens = 0
    for message in request.messages:
        value = message.model_dump(mode="json", exclude_none=True)
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            value["metadata"] = {key: item for key, item in metadata.items() if key != MEDIA}
        content = value.get("content")
        if isinstance(content, list):
            value["content"] = [
                {key: item for key, item in block.items()
                 if key not in {"source", "image_url", "url", "data", "file_id"}}
                if isinstance(block, dict) and block.get("type") in {
                    "image", "image_url", "input_image"
                } else block
                for block in content
            ]
        tokens += len(str(value)) // 4
    return tokens


async def complete_media_request(provider, request, request_options, *, is_cancelled):
    """Own the exact final candidate until transport, releasing local refusals."""
    options = deepcopy(request_options)
    dispatched = False
    try:
        if is_cancelled():
            raise asyncio.CancelledError()
        await check_final_budget(provider, request, options)
        if is_cancelled():
            raise asyncio.CancelledError()
        dispatched = True
        return await provider.complete(request, **options)
    finally:
        if not dispatched:
            release = getattr(provider, "release_media_request", None)
            if callable(release):
                try:
                    # Public synchronous per-candidate abandon, not session close.
                    # Mismatch/absence is not a claim that another fit was released.
                    release(request, request_options=options)
                except Exception:  # noqa: BLE001 - cleanup must not replace the original failure/cancellation
                    logger.warning("Media candidate cleanup failed; owner reconciliation required")


async def check_warning_only_fallback(provider, request, request_options):
    """Use only the owner's exact warning-only witness, never a fake count.

    Missing optional adapter support makes no fit claim and returns False for
    local refusal. A present witness must refuse visual/mandatory/opaque
    candidates itself. Its errors never trigger fallback or another count call.
    """
    witness = getattr(provider, "media_text_fallback_witness", None)
    if not callable(witness):
        return False
    value = witness(request, request_options=deepcopy(request_options))
    if inspect.isawaitable(value):
        value = await value
    if (
        not isinstance(value, Mapping)
        or type(value.get("v")) is not int or value["v"] != 1
        or value.get("validated") is not True
        or not isinstance(value.get("projection_digest"), str)
        or len(value["projection_digest"]) != 64
        or any(char not in "0123456789abcdef" for char in value["projection_digest"])
        or not isinstance(value.get("messages"), list)
    ):
        raise ValueError("Media text fallback witness is invalid")
    return True


def provider_error_message(error, request):
    # Hydrated provider copies stay private even when a decoder/SDK includes its
    # request in an exception. Preserve legacy text diagnostics outside media.
    if request is None or any(MEDIA in (message.metadata or {}) for message in request.messages):
        return "Media provider request failed; private transport diagnostics were not retained"
    return str(error)