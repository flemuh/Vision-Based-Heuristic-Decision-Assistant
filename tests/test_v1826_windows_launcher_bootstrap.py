from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def _labels(text: str) -> list[str]:
    return re.findall(r"(?m)^:([A-Za-z0-9_]+)\s*$", text)


def test_run_windows_uses_host_pip_and_short_external_venv():
    text = _read("run_windows.bat")
    assert ":VALIDATE_HOST_PIP" in text
    assert "SpeedloraJewelBingo\\v18026-py" in text
    assert '-m venv --without-pip "%VENV_DIR%"' in text
    assert '-m pip --python "%VENV_PY%" install' in text
    assert ':VALIDATE_VENV_PYTHON' in text
    assert '-m ensurepip' not in text.lower()
    assert '-m venv --without-pip .venv' not in text
    assert '".venv\\Scripts\\python.exe" app.py' not in text
    assert '"%VENV_PY%" app.py' in text


def test_run_windows_does_not_require_private_venv_pip():
    text = _read("run_windows.bat")
    assert '"%VENV_PY%" -m pip' not in text
    assert "pip inside venv is not required" in text
    assert "pip._vendor.urllib3.packages" not in text


