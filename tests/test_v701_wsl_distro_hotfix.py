from pathlib import Path

from vision_engine.solver import wsl

ROOT = Path(__file__).parents[1]


def test_wsl_prefix_always_contains_explicit_distro():
    assert wsl._wsl_prefix("Ubuntu") == ["wsl.exe", "-d", "Ubuntu", "--"]
    assert wsl._wsl_prefix("") == ["wsl.exe", "-d", "Ubuntu", "--"]


