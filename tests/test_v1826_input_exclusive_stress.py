from vision_engine.diagnostics.input_exclusive_stress import run_input_exclusive_stress


def test_v1826_input_exclusive_stress_1000_cycles_has_no_leaks():
    report = run_input_exclusive_stress(1000, rotate_every=100, seed=1826)
    assert report.failures == 0, report.to_dict()
    assert report.acquisitions == 1000
    assert report.releases == 1000
    assert report.session_rotations == 9
    assert report.pending6_cycles == 20
    assert report.stale_bridge_rejections == 1000
    assert report.forbidden_input_admissions == 0
    assert report.final_bridge_active == 0
    assert report.final_bridge_queued == 0
    assert report.final_foreground == 0
    assert report.final_background == 0
    assert report.final_frozen is False
    assert report.final_freeze_pending is False
    assert report.final_monitor_requested is False
    assert report.final_monitor_quiesced is False
