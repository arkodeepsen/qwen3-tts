import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_workflow_parses_and_targets_ghcr():
    try:
        import yaml
    except ImportError:
        import pytest; pytest.skip("pyyaml not installed")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml").read_text())
    assert "jobs" in wf
    assert "ghcr.io" in (ROOT / ".github" / "workflows" / "build.yml").read_text()


def test_readme_has_api_and_billing_sections():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ["register_voice", "generate", "voice_id", "scale to zero", "/status"]:
        assert needle in txt
