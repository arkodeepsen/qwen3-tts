"""Streamlit GUI for the Qwen3-TTS RunPod endpoint: register a voice, then generate."""
import base64, os
import requests
import streamlit as st
from cli import build_generate_payload, build_register_payload, poll_result  # noqa: E402

st.set_page_config(page_title="Qwen3-TTS Voice Clone", layout="wide")

default_id = os.getenv("ENDPOINT_ID", "")
url = st.sidebar.text_input("RunPod runsync URL",
        value=(f"https://api.runpod.ai/v2/{default_id}/runsync" if default_id else ""))
api_key = st.sidebar.text_input("RunPod API Key", value=os.getenv("RUNPOD_API_KEY", ""), type="password")


def _post(payload):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, json=payload, headers=headers, timeout=300).json()


st.title("Qwen3-TTS Voice Clone")
tab_reg, tab_gen = st.tabs(["1 · Register voice", "2 · Generate"])

with tab_reg:
    name = st.text_input("Voice name", "client_narrator")
    ref_text = st.text_area("Reference transcript (exact words in the clip)")
    ref_file = st.file_uploader("Reference audio (wav/mp3)", type=["wav", "mp3", "flac"])
    language = st.selectbox("Language", ["English", "Chinese", "Japanese", "Korean", "German",
                                          "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"])
    if st.button("Register", type="primary") and ref_file and ref_text:
        payload = {"input": {"action": "register_voice", "name": name,
                             "ref_audio": base64.b64encode(ref_file.getvalue()).decode(),
                             "ref_text": ref_text, "language": language}}
        out = poll_result(url, api_key, _post(payload))
        if out.get("success"):
            st.success(f"Registered! voice_id = {out['voice_id']}")
            st.code(out["voice_id"])
        else:
            st.error(out)

with tab_gen:
    voice_id = st.text_input("voice_id (from step 1)")
    text = st.text_area("Text to synthesize", height=180)
    g_lang = st.selectbox("Language ", ["English", "Chinese", "Japanese", "Korean", "German",
                                         "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"], key="glang")
    fmt = st.selectbox("Format", ["wav", "mp3", "flac", "opus"])
    want_srt = st.checkbox("Return SRT subtitles")
    seed = st.number_input("Seed", value=42, step=1)
    if st.button("Generate", type="primary") and voice_id and text:
        payload = build_generate_payload(voice_id, text, g_lang, fmt, want_srt, int(seed))
        out = poll_result(url, api_key, _post(payload))
        if out.get("success") and out.get("audio_base64"):
            audio = base64.b64decode(out["audio_base64"])
            st.audio(audio, format=f"audio/{fmt}")
            st.download_button("Download audio", audio, f"output.{fmt}", mime=f"audio/{fmt}")
            if out.get("srt"):
                st.text_area("SRT", out["srt"], height=200)
                st.download_button("Download SRT", out["srt"], "output.srt", "text/plain")
        else:
            st.error(out)
