from pixel_diff.timing import StageTimer


def test_stage_timer_records_named_intervals_and_total() -> None:
    values = iter([10.0, 10.2, 10.5, 11.0])
    timer = StageTimer(clock=lambda: next(values))

    timer.checkpoint("render")
    timer.checkpoint("alignment")
    metrics = timer.finish()

    assert metrics == {
        "timing_render_ms": 200,
        "timing_alignment_ms": 300,
        "elapsed_ms": 1000,
    }

