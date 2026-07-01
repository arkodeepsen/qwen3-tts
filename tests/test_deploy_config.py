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


def test_tests_json_hub_ready():
    t = json.loads((ROOT / ".runpod" / "tests.json").read_text())
    actions = [c["input"]["action"] for c in t["tests"]]
    # Hub tests must be self-contained (no cross-test output chaining), so we
    # exercise the heavy register path + a no-arg list, not a voice_id-dependent generate.
    assert "register_voice" in actions and "list_voices" in actions
    assert all("timeout" in c for c in t["tests"])  # Hub requires per-test timeout
    env_keys = {e["key"] for e in t["config"]["env"]}
    assert {"HF_HOME", "VOICE_DIR"} <= env_keys      # model + voices persist on the volume
    assert t["config"]["allowedCudaVersions"]        # non-empty CUDA compatibility list
