import json
import tomllib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_runpod_toml_scale_to_zero():
    cfg = tomllib.loads((ROOT / "runpod.toml").read_text())["runpod"]
    assert cfg["min_workers"] == 0
    assert cfg["idle_timeout"] == 5
    assert cfg["flashboot"] is True
    assert cfg["volume_gb"] >= 20
    assert cfg["execution_timeout"] == 300


def test_hub_json_serverless_audio():
    hub = json.loads((ROOT / ".runpod" / "hub.json").read_text())
    assert hub["type"] == "serverless"
    assert hub["config"]["env"] and any(e["key"] == "VOICE_DIR" for e in hub["config"]["env"])


def test_tests_json_has_register_and_generate():
    t = json.loads((ROOT / ".runpod" / "tests.json").read_text())
    actions = [c["input"]["action"] for c in t["tests"]]
    assert "register_voice" in actions and "generate" in actions
