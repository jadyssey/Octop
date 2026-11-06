"""Streaming fallback for providers that reject non-streaming chat requests.

Some OpenAI-compatible endpoints (CodeBuddy / copilot.tencent.com among them)
only implement SSE and answer a plain stream=false call with HTTP 400:

    {"code": 11101, "msg": "Non-stream chat request is currently not supported"}

Octop's one-shot helpers (infra/text_assist.py, harness memory extraction) call
invoke / ainvoke, so they fail against such providers even though the agent
conversation (streaming) works fine.

wrap_chat_model() returns a transparent proxy that retries a rejected
non-streaming call through stream / astream and concatenates the chunks.
install_chat_model_stream_fallback() applies it to the isolated model
instances the harness builds for memory extraction - the main conversation
model is never wrapped.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# "Non-stream chat request is currently not supported" / "non_streaming ..."
_NON_STREAM_RE = re.compile(r"non[-_ ]?stream", re.IGNORECASE)
_UNSUPPORTED_RE = re.compile(r"not\s+support|unsupported|not\s+allowed|不支持", re.IGNORECASE)


def is_non_stream_unsupported(exc: BaseException) -> bool:
    """True when *exc* reports that non-streaming requests are rejected."""
    text = str(exc)
    if not text:
        return False
    return bool(_NON_STREAM_RE.search(text) and _UNSUPPORTED_RE.search(text))


def _chunk_text(chunk: Any) -> str:
    """Best-effort text of one streamed chunk."""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


class StreamFallbackModel:
    """Chat-model proxy that falls back to streaming on non-stream rejection.

    Everything except bind / invoke / ainvoke is delegated to the wrapped model,
    so provider-specific attributes (_harness_*, profile, bind_tools) keep
    working. Callers that branch on type(model) (e.g. harness
    _is_langchain_openai_model) see the wrapper, so OpenAI-only bind options are
    skipped for wrapped models - they only apply to the isolated memory calls
    that used to fail outright against a stream-only provider.
    """

    def __init__(self, model: Any) -> None:
        self._inner = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"StreamFallbackModel({self._inner!r})"

    def bind(self, **kwargs: Any) -> Any:
        bind = getattr(self._inner, "bind", None)
        if bind is None:
            return self
        return StreamFallbackModel(bind(**kwargs))

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return self._inner.invoke(input, config=config, **kwargs)
        except Exception as exc:
            if not is_non_stream_unsupported(exc):
                raise
            logger.info("provider rejected non-stream request; retrying with streaming")
        return _collect(self._inner.stream(input, config=config, **kwargs))

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return await self._inner.ainvoke(input, config=config, **kwargs)
        except Exception as exc:
            if not is_non_stream_unsupported(exc):
                raise
            logger.info("provider rejected non-stream request; retrying with streaming")
        return await _collect_async(self._inner.astream(input, config=config, **kwargs))


def wrap_chat_model(model: Any) -> Any:
    """Return *model* wrapped with streaming fallback (idempotent)."""
    if isinstance(model, StreamFallbackModel):
        return model
    return StreamFallbackModel(model)


def _collect(chunks: Iterator[Any]) -> AIMessage:
    return AIMessage(content="".join(_chunk_text(chunk) for chunk in chunks))


async def _collect_async(chunks: AsyncIterator[Any]) -> AIMessage:
    return AIMessage(content="".join([_chunk_text(chunk) async for chunk in chunks]))


def install_chat_model_stream_fallback() -> None:
    """Wrap the isolated models the harness builds for memory extraction.

    get_chat_model(ref, max_retries=0) produces short-lived instances owned by
    the memory LLM client (octop_harness.memory.llm_client); the conversation
    model is built without max_retries and stays untouched. Idempotent.
    """
    from octop_harness.llm.factory import ChatModelFactory

    if getattr(ChatModelFactory, "_octop_stream_fallback", False):
        return
    original = ChatModelFactory.get_chat_model

    def get_chat_model(self: Any, model_ref: str, *, max_retries: int | None = None) -> Any:
        model = original(self, model_ref, max_retries=max_retries)
        if max_retries is None:
            return model
        return wrap_chat_model(model)

    ChatModelFactory.get_chat_model = get_chat_model  # type: ignore[method-assign]
    ChatModelFactory._octop_stream_fallback = True  # type: ignore[attr-defined]
    logger.debug("installed chat-model streaming fallback for isolated models")


__all__ = [
    "StreamFallbackModel",
    "install_chat_model_stream_fallback",
    "is_non_stream_unsupported",
    "wrap_chat_model",
]
