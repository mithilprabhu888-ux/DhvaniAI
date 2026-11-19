"""Manual live test for DhvaniAI: Whisper STT (en + kn), streaming agent,
Kannada TTS fallback."""

import json
import sys

import requests
from gtts import gTTS

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8080"


def stream_turn(message, persona="auto"):
    print(f"\n--- turn: {message!r} (persona={persona}) ---")
    r = requests.post(f"{BASE}/api/chat/stream",
                      params={"message": message, "persona": persona},
                      timeout=120, stream=True)
    answer = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        evt = json.loads(line[5:].strip())
        t = evt["type"]
        if t == "delta":
            answer += evt["text"]
        elif t == "tool_call":
            print(f"  TOOL_CALL   {evt['name']} {json.dumps(evt['arguments'])}")
        elif t == "tool_result":
            print(f"  TOOL_RESULT ({'ok' if evt['ok'] else 'FAIL'}) "
                  f"{evt['name']}: {evt['result'][:70]}")
        elif t == "text":
            answer = evt["text"]
        elif t == "error":
            print(f"  ERROR       {evt['text']}")
    print(f"  ANSWER      {answer}")
    return answer


def transcribe(path, language):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/transcribe",
                          params={"language": language},
                          files={"audio": ("speech.mp3", f, "audio/mpeg")},
                          timeout=90)
    data = r.json()
    print(f"TRANSCRIBE [{language}]: {r.status_code} "
          f"stt_ms={data.get('stt_ms')} text={data.get('text')!r}")
    return data.get("text", "")


# 1) English speech -> Whisper
en = transcribe(r"D:\Temp\opencode\vaani_test.wav", "english")
stream_turn(en, persona="english")

# 2) Kannada speech -> Whisper (generate speech with gTTS)
gTTS("ನಮಸ್ಕಾರ, ಈಗ ಸಮಯ ಎಷ್ಟು? ದಯವಿಟ್ಟು ಹೇಳಿ.", lang="kn").save(
    r"D:\Temp\opencode\kn_question.mp3")
kn = transcribe(r"D:\Temp\opencode\kn_question.mp3", "kannada")
stream_turn(kn, persona="kannada")

# 3) Kannada tool use: notes in Kannada
stream_turn("ನನ್ನ ಡೆಮೊ ಶುಕ್ರವಾರ ಇದೆ ಎಂದು ನೆನಪಿಡಿ", persona="kannada")
stream_turn("ನನ್ನ ನೋಟ್ಸ್ ಏನಿದೆ?", persona="kannada")

# 4) Hinglish + auto mode
stream_turn("Yaar, 234 times 12 kitna hota hai?", persona="hinglish")
stream_turn("ಧನ್ಯವಾದ, ಸಧ್ಯಕ್ಕೆ ಸಾಕು", persona="auto")

# 5) server TTS fallback (Kannada)
r = requests.post(f"{BASE}/api/tts",
                  params={"text": "ನಮಸ್ಕಾರ, ನಾನು ಧ್ವನಿ ಐ.",
                          "language": "kannada"}, timeout=60)
print(f"\nTTS fallback: {r.status_code} {r.headers.get('content-type')} "
      f"{len(r.content)} bytes")

print("\nclear:", requests.post(f"{BASE}/api/clear", timeout=30).json())
