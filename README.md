# DhvaniAI

**A multilingual voice agent you can interrupt.**

ಧ್ವನಿ (dhvani) means *voice* in Kannada and Hindi — and that's the whole
idea: talk to it in **English, Hinglish or Kannada**, and it answers out loud.
Speech is transcribed by Groq Whisper, answered by a streaming LLM with real
tools, and spoken back through your browser — or a server voice when your
device has no Kannada speech installed.

## The voice loop

```
Mic → Groq Whisper (STT, en/hi/kn) → streaming LLM + tools (Groq) → TTS
```

- 🗣 **Three languages** — Auto, English, Hinglish and **ಕನ್ನಡ (Kannada)**.
  Kannada speech is transcribed with Whisper's `kn` hint and answered in
  Kannada script; if the browser has no Kannada voice, the server speaks it
  with gTTS automatically.
- ▶️⏹ **Start / stop speaking** — every answer has a play button, a floating
  **Stop speaking** pill appears while it talks, `Esc` stops instantly, and
  starting a new listen stops playback.
- ✂️ **Barge-in** — just talk over it: playback stops and the mic takes over.
- 🖐 **Hands-free** — client-side voice detection starts listening on its own,
  stops after your silence, and re-arms after each answer.
- 🔧 **Voice tool-calling** — time, safe calculator, and notes
  (add/list/delete, persisted); every call appears as a card as it happens.
- ⚡ **Latency panel** — STT / first-token / speech-start / total, colour-coded.
- ⌨️ **Type too** — every feature works from the text box, so you can test
  without a mic.

## Setup

```bash
cd DhvaniAI
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then fill in GROQ_API_KEY
uvicorn app:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080, allow microphone access, tap the orb and talk.

> “What's 17% of 2490?” · “Remember I have a demo on Friday” ·
> “Kya time hua hai?” · “ಈಗ ಸಮಯ ಎಷ್ಟು?” · “ನನ್ನ ನೋಟ್ಸ್ ಏನಿದೆ?”

## API

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/transcribe?language=` | Multipart audio → `{text, stt_ms, language}` via Whisper (en/hi/kn hints) |
| `POST` | `/api/chat/stream?message=&persona=` | SSE agent events: `delta` / `tool_call` / `tool_result` / `text` / `done` |
| `POST` | `/api/tts?text=&language=` | Server TTS (gTTS) — Kannada fallback when the browser has no voice |
| `POST` | `/api/clear` | Clear this session's memory |

## Tools

| Tool | What it does |
|---|---|
| `get_current_time` | Current local date & time |
| `calculate` | Safe arithmetic (AST-walked — no `eval` of arbitrary code) |
| `add_note` / `list_notes` / `delete_note` | Notes persisted to `notes.json` |

## Notes

- Conversation memory is per browser session (cookie), capped in-memory.
- Browser voices depend on your OS/browser; Chrome and Edge have the best
  sets. Kannada usually falls back to the server voice, which always works.
- No LiveKit, no paid TTS keys, no vector DB — one Groq key, `uvicorn`, done.
