from pathlib import Path

import pytest

from vision_engine.input.geometry import AutoPlayPoint
from vision_engine.live.input_executor import LiveInputExecutor


class _Coordinator:
    remote_solver = True


class _Lease:
    coordinator = _Coordinator()
    released = False
    monitor_quiesced = True
    solver_token = "token"


def test_gui_routes_physical_input_through_live_executor():
    src = (Path(__file__).parents[1] / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "self.input_executor.move_and_left_click" in src
    assert "windows_move_and_left_click" not in src


def test_live_executor_requires_valid_exclusive_lease(monkeypatch):
    called = []
    monkeypatch.setattr(
        "vision_engine.live.input_executor.windows_move_and_left_click",
        lambda point, duration_ms: called.append((point, duration_ms)) or "mock",
    )
    executor = LiveInputExecutor()
    lease = _Lease()
    assert executor.move_and_left_click(lease, AutoPlayPoint(10, 20), duration_ms=250) == "mock"
    assert len(called) == 1

    lease.released = True
    with pytest.raises(RuntimeError, match="already released"):
        executor.move_and_left_click(lease, AutoPlayPoint(10, 20), duration_ms=250)


def test_launchers_are_v18_0_26_and_no_v7_marker():
    root = Path(__file__).parents[1]
    run = (root / "run_windows.bat").read_text(encoding="utf-8")
    setup = (root / "setup_windows.bat").read_text(encoding="utf-8")
    wsl = (root / "setup_wsl_solver.bat").read_text(encoding="utf-8")
    assert "V18.0.26" in run and "requirements_ok_v18" in run
    assert "V18.0.26" in setup and "requirements_ok_v18" in setup
    assert "V18.0.26" in wsl
    assert "V18.0.11" not in run
    assert "requirements_ok_v7" not in run + setup


def test_clean_start_uses_packaged_defaults_without_runtime_logs(tmp_path, monkeypatch):
    from vision_engine.config import AppConfig
    from vision_engine.data_home import prepare_user_data
    from vision_engine.vision import VisionConfig

    project = tmp_path / "speedlora-jewel-bingo-assistant-v18.0.26-clean"
    packaged_data = project / "data"
    packaged_data.mkdir(parents=True)
    source_data = Path(__file__).parents[1] / "data"
    for name in ["app_config.json", "vision_config.json", "boards.json", "history.json", "episodes.json"]:
        (packaged_data / name).write_bytes((source_data / name).read_bytes())
    for name in ["board_templates", "icon_templates", "templates"]:
        (packaged_data / name).mkdir()

    user = tmp_path / "user-data"
    monkeypatch.setenv("SPEEDLORA_JEWEL_DATA_DIR", str(user))
    resolved, _ = prepare_user_data(project)
    assert resolved == user
    assert AppConfig.load(user / "app_config.json").solver_backend == "wsl"
    assert VisionConfig.load(user / "vision_config.json").board_roi is None
    assert not (project / ".venv").exists()
    assert not (project / "data" / "pip-install.log").exists()
