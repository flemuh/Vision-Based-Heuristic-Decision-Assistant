from vision_engine.round_lifecycle import NewRoundDetector


def test_new_round_requires_two_matching_low_count_frames():
    d = NewRoundDetector(required=2, max_selected=1)
    assert not d.observe("board-a", 0, True)
    assert d.observe("board-a", 0, True)


def test_new_round_rejects_more_than_one_selected_and_resets_consensus():
    d = NewRoundDetector(required=2, max_selected=1)
    assert not d.observe("board-a", 0, True)
    assert not d.observe("board-a", 0b11, True)
    assert not d.observe("board-a", 0, True)
    assert d.observe("board-a", 0, True)


def test_invalid_board_resets_consensus():
    d = NewRoundDetector(required=2, max_selected=1)
    assert not d.observe("board-a", 1, True)
    assert not d.observe("board-a", 1, False)
    assert not d.observe("board-a", 1, True)
