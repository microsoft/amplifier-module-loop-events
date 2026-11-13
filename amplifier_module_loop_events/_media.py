"""Structural media-v1 integration. Storage and private pixels stay with the host."""

import inspect
import json
from collections.abc import Mapping
from copy import deepcopy

from amplifier_core import ToolResult

MEDIA = "experimental.media.v1"
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


def refs_from_result(value):
    """Only the actual effective ToolResult.output carrier is eligible."""
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
    if media is not None and isinstance(result, ToolResult):
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


async def report_ready(coordinator):
    media = capability(coordinator)
    if media is not None:
        receipt = media.report(
            None,
            event="consumer-ready",
            outcome="ready",
            details={"contract": 1, "role": "loop"},
        )
        if inspect.isawaitable(receipt):
            await receipt


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
            return result.get_serialized_output()
    if isinstance(value, (dict, list)) or value is None:
        return json.dumps(value)
    return str(value)


async def settle_result(value, *, media, origin, checkpoint, modified):
    if isinstance(value, ToolResult):
        value = value.model_dump()
    if isinstance(value, Mapping) and any(
        isinstance(block, Mapping) and block.get("type") == "image"
        for block in (value.get("content") or [])
    ):
        # A post-hook cannot introduce an un-staged raw rich image into history.
        value = without_media(value)
    refs = refs_from_result(value)
    metadata = {}
    if refs:
        if media is None or checkpoint is None:
            value = without_media(
                value,
                UNAVAILABLE
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


async def request_messages(context, provider, *, media):
    getter = getattr(context, "get_messages_for_request", None)
    if getter is not None:
        # Inspect before calling: catching TypeError could repeat a compaction effect.
        parameters = inspect.signature(getter).parameters.values()
        supports_provider = any(
            p.name == "provider" or p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters
        )
        messages = (
            await getter(provider=provider) if supports_provider else await getter()
        )
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