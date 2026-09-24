"""Fake Anthropic client for tests: replays scripted streaming turns."""
from types import SimpleNamespace

import anthropic
import httpx2


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool(name, id_="t1", input_=None):
    return SimpleNamespace(type="tool_use", name=name, id=id_, input={} if input_ is None else input_)


class FakeStream:
    """Mimics client.beta.messages.stream(...): a context manager that yields
    `text` events chunk by chunk, then get_final_message()."""

    def __init__(self, stop_reason, content, fail_after_text=None):
        self.final = SimpleNamespace(stop_reason=stop_reason, content=content)
        self.fail_after_text = fail_after_text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for block in self.final.content:
            if block.type == "text":
                half = len(block.text) // 2
                for chunk in (block.text[:half], block.text[half:]):
                    if chunk:
                        yield SimpleNamespace(type="text", text=chunk)
        if self.fail_after_text:
            raise self.fail_after_text

    def get_final_message(self):
        return self.final


class FakeClient:
    """Replays scripted turns and records every request's kwargs."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        turn = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        if isinstance(turn, Exception):
            raise turn
        return FakeStream(*turn)


def connection_error():
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))


def claude(turns):
    """An LLM wired to a scripted fake Anthropic client."""
    from aegis_core.coordinator import LLM, MODEL

    return LLM("claude", MODEL, FakeClient(turns))


# ----- Groq (OpenAI-compatible chat completions) -------------------------------------------

def _chunk(content=None, tool_calls=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def _tc(index, id_=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id_, function=fn)


def groq_text(text, finish="stop"):
    """A streamed text-only turn, split into two chunks."""
    half = len(text) // 2
    return [_chunk(content=text[:half]), _chunk(content=text[half:]), _chunk(finish=finish)]


def groq_tool_calls(*calls, text=None, finish="tool_calls"):
    """A streamed tool-call turn. Each call is (id, name, arguments_json); the
    id/name and the arguments arrive in separate fragments, like the real API."""
    chunks = [_chunk(content=text)] if text else []
    for i, (id_, name, args) in enumerate(calls):
        chunks.append(_chunk(tool_calls=[_tc(i, id_=id_, name=name, arguments="")]))
        half = len(args) // 2
        chunks.append(_chunk(tool_calls=[_tc(i, arguments=args[:half])]))
        chunks.append(_chunk(tool_calls=[_tc(i, arguments=args[half:])]))
    chunks.append(_chunk(finish=finish))
    return chunks


class FakeGroqClient:
    """Mimics groq.Groq().chat.completions.create(stream=True)."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        turn = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        if isinstance(turn, Exception):
            raise turn
        return iter(turn)


def groq(turns):
    from aegis_core.coordinator import GROQ_MODEL, LLM

    return LLM("groq", GROQ_MODEL, FakeGroqClient(turns))
