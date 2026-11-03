"""Synthesize the English demo narration with ElevenLabs, one WAV per shot.

Reads out/demo/voice/narration.json (written by web/tools/demo_video.py
--voice-text) and writes out/demo/voice_el/<shot>.wav. A shot that already has a
WAV is skipped, so credits are spent once per shot. Without --send it only prints
the shots and the credits they would use.

The key is ELEVENLABS_API_KEY in ~/graspgrade/.env. Credits are characters times
the model rate (1.0 for eleven_multilingual_v2). The free plan allows 10,000 a month.

Run in WSL from ~/graspgrade:

    .venv/bin/python web/tools/demo_voice_el.py --shots 00          # cost only
    .venv/bin/python web/tools/demo_voice_el.py --shots 00 --send   # call the API
    .venv/bin/python web/tools/demo_voice_el.py --send              # every shot
    .venv/bin/python web/tools/demo_voice_el.py --rewav             # WAVs again from the saved MP3s
"""
import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "out" / "demo" / "voice" / "narration.json"
OUT = ROOT / "out" / "demo" / "voice_el"
VOICE_ID = "cjVigY5qzO86Huf0OWal"  # Eric - Smooth, Trustworthy (premade)
MODEL = "eleven_multilingual_v2"
SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}
TEMPO = 1.1  # Eric reads slower than the Windows voice; 1.1 keeps the English video under 3 minutes


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ELEVENLABS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ELEVENLABS_API_KEY is not in .env")


def speak(text, key):
    body = json.dumps({"text": text, "model_id": MODEL, "voice_settings": SETTINGS}).encode()
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128",
                                 data=body, headers={"xi-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", help="comma separated shot keys, default every shot")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--rewav", action="store_true", help="remake every WAV from the saved MP3s at TEMPO, no API call")
    a = ap.parse_args()
    text = json.loads(SRC.read_text(encoding="utf-8"))
    shots = a.shots.split(",") if a.shots else list(text)
    if a.rewav:
        for k in shots:
            wav(k)
        print("remade", len(shots), "WAVs at tempo", TEMPO)
        return
    todo = [k for k in shots if not (OUT / f"{k}.wav").exists()]
    print("shots", ",".join(todo) or "none", "credits", sum(len(text[k]) for k in todo))
    if not a.send:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    for k in todo:
        (OUT / f"{k}.mp3").write_bytes(speak(text[k], key))
        wav(k)
        print("wrote", k)


def wav(k):
    """The WAV the video uses: the saved MP3, sped up to TEMPO without changing the pitch."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / f"{k}.mp3"), "-af", f"atempo={TEMPO}",
                    "-ar", "24000", "-ac", "1", str(OUT / f"{k}.wav")], check=True)


if __name__ == "__main__":
    main()
