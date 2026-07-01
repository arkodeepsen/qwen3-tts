import base64
from client.cli import build_register_payload, build_generate_payload

def test_register_payload_encodes_audio(tmp_path):
    p = tmp_path / "ref.wav"; p.write_bytes(b"RIFFfake")
    payload = build_register_payload("Narr", str(p), "hello", "English")
    assert payload["input"]["action"] == "register_voice"
    assert base64.b64decode(payload["input"]["ref_audio"]) == b"RIFFfake"
    assert payload["input"]["name"] == "Narr"

def test_generate_payload_fields():
    payload = build_generate_payload("v-1", "Hi there.", "English", "mp3", True, 7)
    inp = payload["input"]
    assert inp["action"] == "generate" and inp["voice_id"] == "v-1"
    assert inp["response_format"] == "mp3" and inp["return_srt"] is True and inp["seed"] == 7
