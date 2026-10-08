from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NewRoundDetector:
    """Small deterministic consensus gate for post-game round detection."""

    required: int = 2
    max_selected: int = 1
    candidate: tuple[str, int] | None = None
    seen: int = 0

    def reset(self) -> None:
        self.candidate = None
        self.seen = 0

    def observe(self, board_key: str | None, mask: int, valid_board: bool) -> bool:
        if not valid_board or board_key is None or int(mask).bit_count() > self.max_selected:
            self.reset()
            return False
        current = (str(board_key), int(mask))
        if current == self.candidate:
            self.seen += 1
        else:
            self.candidate = current
            self.seen = 1
        return self.seen >= max(1, int(self.required))


@dataclass(frozen=True)
class CompletionEvidence:
    confirmed: bool
    reason: str
    final_mask: int | None = None


@dataclass
class CompletionDetector:
    """Detect a completed round even if the normal board disappears immediately.

    Real MU transitions can briefly expose the true 14/14 mask and then replace
    the board with the reward/result screen, making the ordinary selector see
    0/14. The old monotonic filter interpreted that as a regression and could
    remain stuck at 13/14 forever.

    This detector remembers a recent raw 14/14 snapshot and treats a subsequent
    result-screen-like disappearance as completion. The optional external fields
    remain for compatibility/tests, but the P5 live loop does not perform OCR.
    """

    min_accepted: int = 12
    disappear_required: int = 2
    recent_14_window_frames: int = 10

    recent_full_mask: int | None = None
    recent_full_age: int = 0
    disappearance_streak: int = 0

    def reset(self) -> None:
        self.recent_full_mask = None
        self.recent_full_age = 0
        self.disappearance_streak = 0

    def observe(
        self,
        *,
        accepted_mask: int,
        raw_mask: int,
        jewel_visible: bool,
        remaining_ui: int | None = None,
        result_plausible: bool = False,
    ) -> CompletionEvidence:
        accepted_mask = int(accepted_mask)
        raw_mask = int(raw_mask)
        accepted_n = accepted_mask.bit_count()
        raw_n = raw_mask.bit_count()

        if self.recent_full_mask is not None:
            self.recent_full_age += 1
            if self.recent_full_age > max(1, int(self.recent_14_window_frames)):
                self.recent_full_mask = None
                self.recent_full_age = 0

        if raw_n == 14:
            self.recent_full_mask = raw_mask
            self.recent_full_age = 0
            self.disappearance_streak = 0
            # One raw full-board frame is remembered but not committed by itself.
            # A second independent signal below will confirm completion.

        strong_external = (remaining_ui == 0) or bool(result_plausible)
        if strong_external and accepted_n >= self.min_accepted:
            final_mask = self.recent_full_mask
            if final_mask is None and raw_n == 14:
                final_mask = raw_mask
            return CompletionEvidence(True, "external-result-evidence", final_mask)

        # Reward/result screens commonly make the normal selected-cell detector
        # see zero while the current-jewel icon disappears. Only consider this
        # near the end of a round and only after a recent real 14/14 raw mask.
        result_like_disappearance = raw_n <= 1 and not jewel_visible and accepted_n >= self.min_accepted
        if result_like_disappearance and self.recent_full_mask is not None:
            self.disappearance_streak += 1
        else:
            self.disappearance_streak = 0

        if self.disappearance_streak >= max(1, int(self.disappear_required)):
            return CompletionEvidence(True, "recent-14-then-result-screen", self.recent_full_mask)

        return CompletionEvidence(False, "waiting-completion-evidence", None)
