"""DhvaniAI agent: streaming tool-calling loop tuned for voice latency."""

import json
from threading import Lock
from typing import Any, AsyncIterator

MAX_ROUNDS = 6
MAX_HISTORY = 16

SYSTEM_PROMPT = """You are DhvaniAI, a warm, quick-witted voice assistant.

Your answers are SPOKEN ALOUD, so:
- Keep them short: 1-3 spoken sentences unless the user asks for detail.
- Plain conversational text only: no markdown, no lists, no emojis, no
  asterisks — nothing that sounds odd when read out.
- You speak English, Hindi, Hinglish and Kannada (ಕನ್ನಡ).

You have tools: get_current_time, calculate, add_note, list_notes,
delete_note. Use them whenever they help — for any math, for the current
date/time, and whenever the user asks you to remember something.
Never claim you did something without calling the tool."""

PERSONA_INSTRUCTIONS = {
    "english": "ANSWER LANGUAGE: English only. Even if the user speaks Hindi, "
               "Hinglish or Kannada, you must answer in English — do not "
               "switch languages.",
    "hinglish": "ANSWER LANGUAGE: Hinglish only, written in Latin script "
                "(e.g. 'Haan bilkul, main aapki help kar sakta hoon'). Never "
                "answer in Devanagari, English-only or Kannada.",
    "kannada": "ANSWER LANGUAGE: Kannada only, written in ಕನ್ನಡ script. Never "
               "answer in English or Hindi — even if the user's sentence "
               "contains English words, reply fully in Kannada and keep only "
               "proper nouns or technical terms in English.",
    "auto": "ANSWER LANGUAGE: reply in the language the user just used. If "
            "their message is Kannada, answer in Kannada script; if Hindi, "
            "answer in Hinglish Latin script; if English, answer in English.",
}


class VoiceAgent:
    """Per-session conversation loop with streaming text + visible tools."""

    def __init__(self, openai_client, model, max_rounds=MAX_ROUNDS):
        self.openai = openai_client
        self.model = model
        self.max_rounds = max_rounds
        self.sessions: dict[str, list[dict]] = {}
        self.lock = Lock()

    # ------------------------------------------------------------------ #
    # Session memory
    # ------------------------------------------------------------------ #

    def history(self, sid: str) -> list[dict]:
        with self.lock:
            return list(self.sessions.get(sid, []))

    def clear(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)

    def _commit(self, sid: str, message: str, reply: str) -> None:
        with self.lock:
            history = self.sessions.setdefault(sid, [])
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            if len(history) > MAX_HISTORY:
                del history[:-MAX_HISTORY]

    # ------------------------------------------------------------------ #
    # Full loop
    # ------------------------------------------------------------------ #

    async def reply_stream(self, sid: str, message: str, tools: dict,
                           persona: str = "auto") -> AsyncIterator[dict]:
        """Yield live events: status / delta / tool_call / tool_result /
        text / done / error."""
        try:
            async for event in self._loop(sid, message, tools, persona):
                yield event
        except Exception as e:                       # noqa: BLE001
            yield {"type": "error", "text": f"DhvaniAI failure: {e}"}
            yield {"type": "done"}

    async def _loop(self, sid: str, message: str, tools: dict,
                    persona: str = "auto"):
        import time
        started = time.perf_counter()

        system = SYSTEM_PROMPT
        instruction = PERSONA_INSTRUCTIONS.get(persona)
        if instruction:
            system += "\n\n" + instruction

        messages = [
            {"role": "system", "content": system},
            *self.history(sid),
            {"role": "user", "content": message},
        ]
        specs = [t["schema"] for t in tools.values()]

        # Inline streaming loop: text deltas go out as they arrive; tool
        # calls are assembled from streamed fragments, executed, and fed
        # back until the model produces a final answer.
        for round_no in range(1, self.max_rounds + 1):
            stream = await self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=specs or None,
                stream=True,
                temperature=0.6,
            )
            content_parts: list[str] = []
            tool_slots: dict[int, dict] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if text:
                    content_parts.append(text)
                    yield {"type": "delta", "text": text}
                for tc in (getattr(delta, "tool_calls", None) or []):
                    slot = tool_slots.setdefault(tc.index, {
                        "id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments

            content = "".join(content_parts)
            calls = [tool_slots[i] for i in sorted(tool_slots)]

            if not calls:
                if not content:
                    yield {"type": "error",
                           "text": "The model returned an empty answer."}
                    yield {"type": "done"}
                    return
                self._commit(sid, message, content)
                yield {"type": "text", "text": content,
                       "llm_ms": int((time.perf_counter() - started) * 1000)}
                yield {"type": "done"}
                return

            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": c["arguments"]}}
                    for c in calls
                ],
            })
            for call in calls:
                try:
                    args = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_call",
                       "name": call["name"], "arguments": args}
                ok, result = self._run_tool(tools, call["name"], args)
                yield {"type": "tool_result",
                       "name": call["name"], "result": result, "ok": ok}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })

        yield {"type": "error",
               "text": "The model used too many tool-calling rounds."}
        yield {"type": "done"}

    @staticmethod
    def _run_tool(tools: dict, name: str, arguments: dict):
        tool = tools.get(name)
        if tool is None:
            return False, f"Error: unknown tool {name}"
        try:
            return True, str(tool["impl"](**arguments))
        except TypeError as e:
            return False, f"Error: bad arguments for {name}: {e}"
