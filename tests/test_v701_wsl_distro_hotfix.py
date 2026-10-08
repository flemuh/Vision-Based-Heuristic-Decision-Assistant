from pathlib import Path

from vision_engine.config import AppConfig
from vision_engine.solver import wsl

ROOT = Path(__file__).parents[1]


def test_packaged_config_pins_ubuntu_instead_of_default_wsl():
    cfg = AppConfig.load(ROOT / "data" / "app_config.json")
    assert cfg.solver_backend == "wsl"
    assert cfg.solver_wsl_distro == "Ubuntu"


def test_wsl_prefix_always_contains_explicit_distro():
    assert wsl._wsl_prefix("Ubuntu") == ["wsl.exe", "-d", "Ubuntu", "--"]
    assert wsl._wsl_prefix("") == ["wsl.exe", "-d", "Ubuntu", "--"]


def test_windows_launch_sources_do_not_use_implicit_default_distro():
    wsl_src = (ROOT / "vision_engine" / "solver" / "wsl.py").read_text(encoding="utf-8")
    run_src = (ROOT / "run_windows.bat").read_text(encoding="utf-8")
    setup_src = (ROOT / "setup_wsl_solver.bat").read_text(encoding="utf-8")
    assert '["wsl.exe", "--"' not in wsl_src
    assert "wsl.exe -- bash" not in run_src
    assert "wsl.exe -- bash" not in setup_src
    assert 'wsl.exe -d "%WSL_DISTRO%" -- python3 --version' in run_src
    assert 'wsl.exe -d "%WSL_DISTRO%" -- python3 --version' in setup_src
    assert 'wsl.exe -d "%WSL_DISTRO%" --cd "%CD%" -- python3' in setup_src
    assert '-- wslpath' not in setup_src
