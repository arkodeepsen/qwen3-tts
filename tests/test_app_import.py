import importlib.util, pathlib

def test_app_module_parses():
    path = pathlib.Path(__file__).resolve().parents[1] / "client" / "app.py"
    spec = importlib.util.spec_from_file_location("app_check", path)
    assert spec is not None
    compile(path.read_text(encoding="utf-8"), str(path), "exec")  # parses without syntax error
