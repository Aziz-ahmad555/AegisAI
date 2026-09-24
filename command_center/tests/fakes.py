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
