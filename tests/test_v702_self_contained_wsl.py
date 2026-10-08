from pathlib import Path

from vision_engine import __version__
from vision_engine.solver import wsl

ROOT = Path(__file__).parents[1]


def test_release_version_matches_current_release():
    assert __version__.startswith("18.")


def test_wsl_project_command_uses_explicit_distro_and_own_directory_without_wslpath(tmp_path):
    cmd = wsl.wsl_project_command(tmp_path, "Ubuntu", "python3", "-m", "vision_engine.solver.service")
    assert cmd[:4] == ["wsl.exe", "-d", "Ubuntu", "--cd"]
    assert cmd[4] == str(tmp_path.resolve())
    assert cmd[5:] == ["--", "python3", "-m", "vision_engine.solver.service"]


def test_release_sources_do_not_depend_on_wslpath_or_default_distro():
    wsl_src = (ROOT / "vision_engine" / "solver" / "wsl.py").read_text(encoding="utf-8")
    run_src = (ROOT / "run_windows.bat").read_text(encoding="utf-8")
    setup_src = (ROOT / "setup_wsl_solver.bat").read_text(encoding="utf-8")
    assert '"wslpath"' not in wsl_src
    assert "-- wslpath" not in run_src
    assert "-- wslpath" not in setup_src
    assert '--cd "%CD%" -- python3' in run_src
    assert '--cd "%CD%" -- python3' in setup_src
    assert 'wsl.exe -d "%WSL_DISTRO%"' in run_src
    assert 'wsl.exe -d "%WSL_DISTRO%"' in setup_src


def test_running_solver_must_match_current_release():
    assert wsl._running_solver_matches({"ok": True, "app_version": __version__})
    assert not wsl._running_solver_matches({"ok": True, "app_version": "0.7.0.1"})
    assert not wsl._running_solver_matches({"ok": True})
