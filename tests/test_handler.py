import handler as h


def test_handler_delegates(monkeypatch):
    seen = {}
    def mock_handle(job_input):
        seen["in"] = job_input
        return {"success": True, "ok": 1}
    monkeypatch.setattr(h, "handle", mock_handle)
    out = h.handler({"input": {"action": "list_voices"}})
    assert out == {"success": True, "ok": 1}
    assert seen["in"] == {"action": "list_voices"}


def test_handler_missing_input():
    out = h.handler({})
    assert out["success"] is False and "input" in out["error"]


def test_handler_wraps_exceptions(monkeypatch):
    def boom(_):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(h, "handle", boom)
    out = h.handler({"input": {"action": "generate"}})
    assert out["success"] is False and "kaboom" in out["error"]
