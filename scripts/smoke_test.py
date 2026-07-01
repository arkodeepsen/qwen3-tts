"""End-to-end smoke test against a deployed RunPod endpoint.

Usage:
  ENDPOINT_ID=... RUNPOD_API_KEY=... python scripts/smoke_test.py
Registers a voice from the public sample clip, then generates speech + SRT.
"""
import base64, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))
import requests
from cli import poll_result

# Load ENDPOINT_ID / RUNPOD_API_KEY from the repo-root .env if python-dotenv is
# installed; otherwise fall back to real environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

URL = f"https://api.runpod.ai/v2/{os.environ['ENDPOINT_ID']}/runsync"
KEY = os.environ["RUNPOD_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
REF = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you blew it!"

reg = poll_result(URL, KEY, requests.post(URL, headers=H, json={"input": {
    "action": "register_voice", "name": "smoke", "ref_audio": REF,
    "ref_text": REF_TEXT, "language": "English"}}).json())
print("register ->", reg)
vid = reg["voice_id"]

gen = poll_result(URL, KEY, requests.post(URL, headers=H, json={"input": {
    "action": "generate", "voice_id": vid,
    "text": "This is a cloned voice speaking a long-form sentence. And a second one, for SRT.",
    "language": "English", "response_format": "mp3", "return_srt": True}}).json())
assert gen["success"] and gen["audio_base64"], gen
open("smoke.mp3", "wb").write(base64.b64decode(gen["audio_base64"]))
open("smoke.srt", "w", encoding="utf-8").write(gen["srt"])
print(f"OK: {gen['duration_sec']}s, {gen['chunks']} chunks, sr={gen['sample_rate']} -> smoke.mp3 + smoke.srt")
