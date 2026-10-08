from types import SimpleNamespace

from vision_engine.autoplay import cell_center


def test_cell_center_uses_absolute_region_and_board_roi():
    region = SimpleNamespace(left=100, top=200)
    # 500x500 board begins at +50,+60 inside selected panel.
    point = cell_center(region, (50, 60, 500, 500), 0)
    assert (point.x, point.y) == (200, 310)

    point = cell_center(region, (50, 60, 500, 500), 24)
    assert (point.x, point.y) == (600, 710)


def test_cell_center_middle_cell_geometry():
    region = SimpleNamespace(left=0, top=0)
    point = cell_center(region, (10, 20, 250, 250), 12)
    assert (point.x, point.y) == (135, 145)
