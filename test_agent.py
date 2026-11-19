"""Tests for the VaaniAI agent loop using a fake streaming OpenAI client."""

import asyncio
import unittest
from types import SimpleNamespace

from agent import VoiceAgent
from voice_tools import TOOLS


def chunk(text=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text, tool_calls=tool_calls))])


def tool_frag(index, name=None, arguments=None, tid=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tid, function=fn)


class FakeAsyncIter:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class FakeOpenAI:
    """Scripted streaming client. Each script item is a list of chunks."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        return FakeAsyncIter(self.script.pop(0))


def make_agent(script):
    openai = FakeOpenAI(script)
    agent = VoiceAgent(openai_client=openai, model="test-model")
    return agent, openai


def collect(agent, message, persona="english"):
    async def run():
        return [evt async for evt
                in agent.reply_stream("test", message, TOOLS, persona)]
    return asyncio.run(run())


class AgentTest(unittest.TestCase):
    def test_streamed_text_answer(self):
        script = [[chunk("Hello"), chunk(" there"), chunk("!")]]
        agent, openai = make_agent(script)
        events = collect(agent, "hi")
        types = [e["type"] for e in events]
        self.assertEqual(types, ["delta", "delta", "delta", "text", "done"])
        deltas = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertEqual(deltas, "Hello there!")
        self.assertEqual(events[-2]["text"], "Hello there!")
        # system prompt + user message were sent
        sent = openai.requests[0]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[-1]["content"], "hi")

    def test_tool_call_assembled_from_fragments(self):
        script = [
            [
                chunk(tool_calls=[tool_frag(0, name="calc")]),
                chunk(tool_calls=[tool_frag(0, name="ulate")]),
                chunk(tool_calls=[tool_frag(0, arguments='{"expr')]),
                chunk(tool_calls=[tool_frag(0, arguments='ession": "2+2"}',
                                  tid="c1")]),
            ],
            [chunk("It is 4.")],
        ]
        agent, openai = make_agent(script)
        events = collect(agent, "what is 2+2")
        types = [e["type"] for e in events]
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        call = [e for e in events if e["type"] == "tool_call"][0]
        self.assertEqual(call["name"], "calculate")
        self.assertEqual(call["arguments"], {"expression": "2+2"})
        result = [e for e in events if e["type"] == "tool_result"][0]
        self.assertTrue(result["ok"])
        self.assertIn("2+2 = 4", result["result"])
        self.assertEqual(events[-2]["text"], "It is 4.")

        # the tool result was fed back before the final answer
        second = openai.requests[1]["messages"]
        tool_msgs = [m for m in second if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("2+2 = 4", tool_msgs[0]["content"])

    def test_hinglish_persona_extends_system(self):
        script = [[chunk("Namaste!")]]
        agent, openai = make_agent(script)
        events = collect(agent, "namaste", persona="hinglish")
        self.assertEqual(events[-1]["type"], "done")
        sent = openai.requests[0]["messages"]
        self.assertIn("Hinglish", sent[0]["content"])

    def test_kannada_and_auto_personas(self):
        agent, openai = make_agent([[chunk("ನಮಸ್ಕಾರ")]])
        collect(agent, "namaskara", persona="kannada")
        self.assertIn("ಕನ್ನಡ", openai.requests[0]["messages"][0]["content"])

        agent2, openai2 = make_agent([[chunk("hello")]])
        collect(agent2, "hi", persona="auto")
        self.assertIn("language the user just used",
                      openai2.requests[0]["messages"][0]["content"])

    def test_history_memory_and_clear(self):
        script = [[chunk("One")], [chunk("Two")]]
        agent, openai = make_agent(script)
        collect(agent, "first")
        self.assertEqual(len(agent.history("test")), 2)  # user + assistant
        collect(agent, "second")
        sent = openai.requests[-1]["messages"]
        roles = [m["role"] for m in sent]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        agent.clear("test")
        self.assertEqual(agent.history("test"), [])

    def test_error_on_exception(self):
        class Broken(FakeOpenAI):
            async def _create(self, **kwargs):
                raise RuntimeError("boom")

        agent = VoiceAgent(openai_client=Broken([]), model="m")
        events = collect(agent, "hi")
        types = [e["type"] for e in events]
        self.assertIn("error", types)
        self.assertEqual(types[-1], "done")


if __name__ == "__main__":
    unittest.main()
