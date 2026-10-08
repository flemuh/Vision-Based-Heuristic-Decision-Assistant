from vision_engine.diagnostics.fence_simulation import run_fence_simulation


def test_v1826_global_fence_stress_has_no_orphaned_work():
    report = run_fence_simulation(300)
    assert report.failures == 0
    assert report.freezes == 300
    assert report.releases == 300
    assert report.stale_bridge_rejections == 300
    assert report.frozen_solver_rejections == 300
    assert report.final_bridge_active == 0
    assert report.final_foreground == 0
    assert report.final_background == 0
