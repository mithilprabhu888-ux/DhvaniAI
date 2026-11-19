import json
import os
import secrets
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from agent import VoiceAgent
from voice_tools import TOOLS

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Set GROQ_API_KEY in the .env file")

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
)
STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

agent = VoiceAgent(
    openai_client=client,
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
)

SESSION_COOKIE = "dhvani_sid"

# persona -> Whisper language hint (None = auto-detect)
STT_LANG = {
    "english": "en",
    "hinglish": "hi",      # lock Hindi so mixed speech stays consistent
    "kannada": "kn",
    "auto": None,
}

# persona -> gTTS language code (server TTS fallback)
TTS_LANG = {
    "english": "en",
    "hinglish": "en",
    "kannada": "kn",
    "auto": None,
}

try:
    from gtts import gTTS
except ImportError:                              # optional fallback
    gTTS = None

# A short language hint helps Whisper spell Kannada more accurately.
STT_PROMPT = {
    "kannada": "ಇದು ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ಸಮಯ, ದಿನಾಂಕ, ಪ್ರಶ್ನೆ.",
}


def clean_transcript(text: str) -> str:
    """Drop punctuation-only output and obvious Whisper repeat loops."""
    text = " ".join((text or "").split())
    if not any(ch.isalnum() for ch in text):
        return ""
    words = text.split()
    if len(words) >= 3:
        for size in (1, 2, 3):
            if len(words) >= size * 3:
                chunk = " ".join(words[:size]).lower()
                i, reps = size, 1
                while i + size <= len(words) and \
                        " ".join(words[i:i + size]).lower() == chunk:
                    reps += 1
                    i += size
                if reps >= 3 and i >= len(words):
                    return ""
    return text


def sid_of(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE) or secrets.token_urlsafe(24)


def with_sid(response: Response, sid: str) -> Response:
    response.set_cookie(SESSION_COOKIE, sid, max_age=7 * 24 * 60 * 60,
                        httponly=True, samesite="Lax")
    return response


app = FastAPI(title="DhvaniAI")


@app.post("/api/transcribe")
async def transcribe(request: Request, audio: UploadFile,
                     language: str = "auto"):
    """Speech -> text via Groq Whisper. Returns {text, stt_ms, language}."""
    sid = sid_of(request)
    data = await audio.read()
    if not data:
        return JSONResponse({"error": "empty audio"}, status_code=400)

    hint = STT_LANG.get(language)
    started = time.perf_counter()
    try:
        kwargs = {"model": STT_MODEL,
                  "file": (audio.filename or "speech.webm", data),
                  "temperature": 0}
        if hint:
            kwargs["language"] = hint
        if language in STT_PROMPT:
            kwargs["prompt"] = STT_PROMPT[language]
        result = await client.audio.transcriptions.create(**kwargs)
    except Exception as e:                       # noqa: BLE001
        return JSONResponse({"error": f"transcription failed: {e}"},
                            status_code=502)

    stt_ms = int((time.perf_counter() - started) * 1000)
    response = JSONResponse({"text": clean_transcript(result.text),
                             "stt_ms": stt_ms,
                             "language": hint or "auto"})
    return with_sid(response, sid)


@app.post("/api/chat/stream")
async def chat_stream(request: Request, message: str, persona: str = "auto"):
    """SSE stream of agent events for one user utterance."""
    sid = sid_of(request)

    async def generate():
        async for event in agent.reply_stream(sid, message, TOOLS, persona):
            yield {"data": json.dumps(event)}

    return with_sid(EventSourceResponse(generate()), sid)


@app.post("/api/tts")
async def tts(text: str, language: str = "auto"):
    """Server-side TTS (gTTS) used when the browser lacks a voice for the
    language — notably Kannada. Returns audio/mpeg."""
    text = (text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    if len(text) > 1200:
        text = text[:1200]
    if gTTS is None:
        return JSONResponse({"error": "server TTS not installed"},
                            status_code=501)

    lang = TTS_LANG.get(language)
    if not lang:
        # auto: pick by script
        if any("\u0c80" <= ch <= "\u0cff" for ch in text):
            lang = "kn"
        elif any("\u0900" <= ch <= "\u097f" for ch in text):
            lang = "hi"
        else:
            lang = "en"

    try:
        from io import BytesIO
        buffer = BytesIO()
        await run_in_threadpool(
            lambda: gTTS(text=text, lang=lang).write_to_fp(buffer))
        audio = buffer.getvalue()
    except Exception as e:                       # noqa: BLE001
        return JSONResponse({"error": f"tts failed: {e}"}, status_code=502)

    return Response(content=audio, media_type="audio/mpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/clear")
async def clear(request: Request):
    agent.clear(sid_of(request))
    return {"ok": True}


@app.get("/")
def index():
    response = FileResponse(HERE / "static" / "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
