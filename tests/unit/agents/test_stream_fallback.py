"""Streaming fallback for providers that reject non-streaming calls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from octop_harness.config import ModelConfig, ProviderConfig
from octop_harness.llm.factory import ChatModelFactory

from octop.infra.agents.providers.stream_fallback import (
    StreamFallbackModel,
    install_chat_model_stream_fallback,
    is_non_stream_unsupported,
    wrap_chat_model,
)

NON_STREAM_ERROR = (
    "Error code: 400 - {'code': 11101, 'msg': 'Non-stream chat request is currently not supported'}"
)


class _StreamOnlyModel:
    """Fake chat model that only answers streamed requests."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.stream_calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError(NON_STREAM_ERROR)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError(NON_STREAM_ERROR)

    def stream(self, input: Any, config: Any = None, **kwargs: Any) -> Iterator[AIMessageChunk]:
        self.stream_calls += 1
        for text in self._chunks:
            yield AIMessageChunk(content=text)

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[AIMessageChunk]:
        self.stream_calls += 1
        for text in self._chunks:
            yield AIMessageChunk(content=text)

    def bind(self, **kwargs: Any) -> Any:
        return self


class _BrokenModel(_StreamOnlyModel):
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("insufficient_quota")

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("insufficient_quota")


def test_detects_stream_only_rejection() -> None:
    assert is_non_stream_unsupported(RuntimeError(NON_STREAM_ERROR))
    assert not is_non_stream_unsupported(RuntimeError("insufficient_quota"))
    assert not is_non_stream_unsupported(RuntimeError("stream timeout"))


async def test_ainvoke_falls_back_to_streaming() -> None:
    inner = _StreamOnlyModel(["he", "llo"])
    model = wrap_chat_model(inner)

    result = await model.ainvoke([HumanMessage("hi")])

    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert inner.stream_calls == 1


def test_invoke_falls_back_to_streaming() -> None:
    inner = _StreamOnlyModel(["wo", "rld"])
    model = wrap_chat_model(inner)

    result = model.invoke([HumanMessage("hi")])

    assert result.content == "world"
    assert inner.stream_calls == 1


def test_unrelated_errors_propagate() -> None:
    model = wrap_chat_model(_BrokenModel([]))
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        model.invoke([HumanMessage("hi")])


async def test_bound_model_keeps_fallback() -> None:
    inner = _StreamOnlyModel(["bo", "und"])
    model = wrap_chat_model(inner).bind(max_tokens=8)

    result = await model.ainvoke([HumanMessage("hi")])

    assert result.content == "bound"


def test_wrap_is_idempotent() -> None:
    inner = _StreamOnlyModel([])
    once = wrap_chat_model(inner)
    assert wrap_chat_model(once) is once


def test_install_wraps_only_isolated_models(monkeypatch: pytest.MonkeyPatch) -> None:
    install_chat_model_stream_fallback()
    provider = ProviderConfig(
        id="codebuddy",
        base_url="https://example.invalid/v2",
        api_key="sk-test",
        name="codebuddy",
        models=[ModelConfig(id="demo")],
    )
    monkeypatch.setattr(
        ChatModelFactory,
        "_build",
        lambda self, provider, model, *, max_retries=None: _StreamOnlyModel(["x"]),
    )
    factory = ChatModelFactory([provider])

    plain = factory.get_chat_model("codebuddy/demo")
    isolated = factory.get_chat_model("codebuddy/demo", max_retries=0)

    assert type(plain) is _StreamOnlyModel
    assert type(isolated) is StreamFallbackModel
