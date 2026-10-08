from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

MAX_PLACEMENTS = 14


@dataclass(frozen=True)
class StableState:
    mask: int
    jewel: str | None
    stable: bool
    reason: str
    jewel_vote_confidence: float = 0.0


class BoardConsensusFilter:
    """Require the same decoded board across multiple frames before accepting it."""

    def __init__(self, frames: int = 3, required: int = 2):
        self.frames = max(1, int(frames))
        self.required = max(1, min(int(required), self.frames))
        self._hashes: deque[str] = deque(maxlen=self.frames)

    def reset(self) -> None:
        self._hashes.clear()

    def observe(self, board_hash: str) -> tuple[bool, int]:
        self._hashes.append(str(board_hash))
        n = Counter(self._hashes)[str(board_hash)]
        return n >= self.required, n


class TemporalStateFilter:
    """Multi-frame filter for screen-derived state.

    V7.0.3 keeps two concepts separate:

    * ``accepted_mask`` is the durable checkpoint for the current round.
    * a +1 move is first *provisional* and becomes durable only after several
      consecutive raw observations.

    A recently committed +1 is also reversible for a short observation window.
    This is intentionally narrower than ordinary regression handling: it only
    rolls back to the immediately previous confirmed checkpoint when the exact
    previous mask reappears repeatedly. It fixes the real 12/14 -> false 13/14
    glow case without allowing arbitrary state regression.

    Multi-cell catch-up remains quarantined separately and requires extended
    raw stability before it may become persistent accepted state.
    """

    def __init__(
        self,
        frames: int = 5,
        required: int = 3,
        max_mask_jump: int = 1,
        max_catchup_jump: int = 14,
        catchup_required_frames: int = 5,
        single_step_confirm_frames: int = 2,
        single_step_rollback_frames: int = 3,
        single_step_rollback_window_frames: int = 10,
    ):
        self.frames = max(1, int(frames))
        self.required = max(1, min(int(required), self.frames))
        self.max_mask_jump = max(1, int(max_mask_jump))
        self.max_catchup_jump = min(MAX_PLACEMENTS, max(self.max_mask_jump, int(max_catchup_jump)))
        self.catchup_required_frames = max(self.required + 1, int(catchup_required_frames))
        self.single_step_confirm_frames = max(self.required, int(single_step_confirm_frames))
        self.single_step_rollback_frames = max(2, int(single_step_rollback_frames))
        self.single_step_rollback_window_frames = max(
            self.single_step_rollback_frames + 1, int(single_step_rollback_window_frames)
        )

        self._masks: deque[int] = deque(maxlen=self.frames)
        self._jewels: deque[tuple[str | None, float]] = deque(maxlen=self.frames)
        self.accepted_mask = 0
        self.accepted_jewel: str | None = None

        self._catchup_candidate: int | None = None
        self._catchup_streak = 0

        self._plus_one_candidate: int | None = None
        self._plus_one_streak = 0

        # A just-committed +1 can be rolled back only to this exact checkpoint
        # for a short window. Once the window expires (or another move commits),
        # ordinary monotonicity becomes absolute again.
        self._recent_previous_mask: int | None = None
        self._recent_committed_mask: int | None = None
        self._recent_commit_age = 0
        self._recent_previous_streak = 0

    @property
    def provisional_mask(self) -> int | None:
        return self._plus_one_candidate

    @staticmethod
    def _valid_mask(mask: int) -> bool:
        return int(mask).bit_count() <= MAX_PLACEMENTS

    def reset(self, mask: int = 0) -> None:
        self._masks.clear()
        self._jewels.clear()
        # Fail closed on physically impossible vision state. A bad first frame
        # must never become the durable checkpoint for the whole round.
        self.accepted_mask = int(mask) if self._valid_mask(mask) else 0
        self.accepted_jewel = None
        self._reset_catchup()
        self._reset_plus_one()
        self._clear_recent_commit()

    def _reset_catchup(self) -> None:
        self._catchup_candidate = None
        self._catchup_streak = 0

    def _reset_plus_one(self) -> None:
        self._plus_one_candidate = None
        self._plus_one_streak = 0

    def _clear_recent_commit(self) -> None:
        self._recent_previous_mask = None
        self._recent_committed_mask = None
        self._recent_commit_age = 0
        self._recent_previous_streak = 0

    def _observe_raw_catchup(self, raw_mask: int) -> None:
        raw = int(raw_mask)
        if self.accepted_mask & ~raw:
            self._reset_catchup()
            return
        jump = (raw & ~self.accepted_mask).bit_count()
        if jump <= self.max_mask_jump or jump > self.max_catchup_jump:
            self._reset_catchup()
            return
        if self._catchup_candidate == raw:
            self._catchup_streak += 1
        else:
            self._catchup_candidate = raw
            self._catchup_streak = 1

    def _observe_plus_one(self, raw_mask: int) -> None:
        raw = int(raw_mask)
        if self.accepted_mask & ~raw:
            self._reset_plus_one()
            return
        jump = (raw & ~self.accepted_mask).bit_count()
        if jump != 1:
            self._reset_plus_one()
            return
        if self._plus_one_candidate == raw:
            self._plus_one_streak += 1
        else:
            self._plus_one_candidate = raw
            self._plus_one_streak = 1

    def _advance_recent_commit_window(self, raw_mask: int) -> StableState | None:
        """Return a controlled rollback state when a fresh +1 was false.

        Only the immediately previous mask is eligible and only inside a small
        observation window. This is deliberately not a generic regression path.
        """
        if self._recent_committed_mask is None or self._recent_previous_mask is None:
            return None

        self._recent_commit_age += 1
        raw = int(raw_mask)
        if raw == self._recent_previous_mask:
            self._recent_previous_streak += 1
        else:
            self._recent_previous_streak = 0

        if self._recent_previous_streak >= self.single_step_rollback_frames:
            previous = self._recent_previous_mask
            current = self.accepted_mask
            self.accepted_mask = int(previous)
            self._masks.clear()
            self._masks.append(self.accepted_mask)
            self._reset_plus_one()
            self._reset_catchup()
            self._clear_recent_commit()
            # Preserve jewel votes. The next jewel is often already visible while
            # the board glow is stabilising, so throwing these votes away causes
            # unnecessary waiting after rollback.
            jewel_mode, jewel_n, jewel_vote_conf = self._weighted_jewel_vote()
            if jewel_mode is not None and jewel_n >= self.required:
                self.accepted_jewel = jewel_mode
            return StableState(
                self.accepted_mask,
                self.accepted_jewel,
                True,
                f"rollback-false-plus-one:{current.bit_count()}->{self.accepted_mask.bit_count()}",
                jewel_vote_conf,
            )

        if self._recent_commit_age >= self.single_step_rollback_window_frames:
            self._clear_recent_commit()
        return None

    @staticmethod
    def _mode(values):
        if not values:
            return None, 0
        value, count = Counter(values).most_common(1)[0]
        return value, count

    def _weighted_jewel_vote(self) -> tuple[str | None, int, float]:
        votes: dict[str | None, float] = defaultdict(float)
        counts: Counter = Counter()
        for label, conf in self._jewels:
            if label is None:
                continue
            votes[label] += max(0.05, float(conf))
            counts[label] += 1
        if not votes:
            return None, 0, 0.0
        label = max(votes, key=votes.get)
        total = sum(votes.values())
        return label, int(counts[label]), float(votes[label] / total if total else 0.0)

    def observe(self, raw_mask: int, raw_jewel: str | None, jewel_confidence: float = 1.0) -> StableState:
        raw_mask = int(raw_mask)
        raw_count = raw_mask.bit_count()
        if raw_count > MAX_PLACEMENTS:
            # Vision can transiently classify hover/glow cells as selected. Never
            # let impossible 15/14, 16/14, ... states enter consensus history.
            self._reset_catchup()
            self._reset_plus_one()
            return StableState(
                self.accepted_mask, self.accepted_jewel, False,
                f"invalid-mask-count:{raw_count}/{MAX_PLACEMENTS}", 0.0,
            )
        self._observe_raw_catchup(raw_mask)
        self._observe_plus_one(raw_mask)
        self._masks.append(raw_mask)
        self._jewels.append((raw_jewel, float(jewel_confidence)))

        rollback = self._advance_recent_commit_window(raw_mask)
        if rollback is not None:
            return rollback

        mask_mode, mask_n = self._mode(self._masks)
        jewel_mode, jewel_n, jewel_vote_conf = self._weighted_jewel_vote()

        buffered_jewel = jewel_mode if (jewel_mode is not None and jewel_n >= self.required) else None

        if mask_n < self.required:
            return StableState(self.accepted_mask, buffered_jewel or self.accepted_jewel, False, "waiting-mask-consensus", jewel_vote_conf)

        candidate = int(mask_mode)
        if not self._valid_mask(candidate):
            self._reset_catchup()
            self._reset_plus_one()
            return StableState(
                self.accepted_mask, buffered_jewel or self.accepted_jewel, False,
                f"invalid-mask-count:{candidate.bit_count()}/{MAX_PLACEMENTS}", jewel_vote_conf,
            )
        if candidate == self.accepted_mask and raw_mask != self.accepted_mask:
            return StableState(self.accepted_mask, buffered_jewel or self.accepted_jewel, False, "waiting-transition-consensus", jewel_vote_conf)

        if self.accepted_mask & ~candidate:
            return StableState(self.accepted_mask, buffered_jewel or self.accepted_jewel, False, "ignored-mask-regression", jewel_vote_conf)

        delta = candidate & ~self.accepted_mask
        jump = delta.bit_count()
        if jump > self.max_catchup_jump:
            self._reset_catchup()
            self._reset_plus_one()
            return StableState(self.accepted_mask, self.accepted_jewel, False, "multi-cell-jump/desync", jewel_vote_conf)

        if jump > self.max_mask_jump:
            if self._catchup_candidate != candidate or self._catchup_streak < self.catchup_required_frames:
                streak = self._catchup_streak if self._catchup_candidate == candidate else 0
                return StableState(
                    self.accepted_mask,
                    buffered_jewel or self.accepted_jewel,
                    False,
                    f"catch-up-quarantine:{jump}:{streak}/{self.catchup_required_frames}",
                    jewel_vote_conf,
                )
            reason = f"catch-up:{jump}"
        elif jump == 1:
            # +1 used to be committed after the ordinary 2-of-3 vote. A click
            # glow can satisfy that briefly, so keep it provisional until the raw
            # mask itself persists long enough.
            if self._plus_one_candidate != candidate or self._plus_one_streak < self.single_step_confirm_frames:
                streak = self._plus_one_streak if self._plus_one_candidate == candidate else 0
                return StableState(
                    self.accepted_mask,
                    buffered_jewel or self.accepted_jewel,
                    False,
                    f"single-step-provisional:{streak}/{self.single_step_confirm_frames}",
                    jewel_vote_conf,
                )
            reason = "stable"
        else:
            reason = "stable"

        transitioned = candidate != self.accepted_mask
        previous_mask = self.accepted_mask
        self.accepted_mask = candidate
        self._reset_catchup()
        self._reset_plus_one()

        if transitioned:
            if jump == 1:
                self._recent_previous_mask = previous_mask
                self._recent_committed_mask = candidate
                self._recent_commit_age = 0
                self._recent_previous_streak = 0
            else:
                self._clear_recent_commit()

            # Keep the durable mask isolated from old frames and clear top-jewel
            # votes after a forward placement. Those votes describe the jewel that
            # was just placed. Rollback is handled separately and preserves votes.
            self._masks.clear()
            self._masks.append(candidate)
            self._jewels.clear()
            self.accepted_jewel = None
            return StableState(self.accepted_mask, None, True, reason, 0.0)

        if buffered_jewel is not None:
            self.accepted_jewel = buffered_jewel
        return StableState(self.accepted_mask, self.accepted_jewel, True, reason, jewel_vote_conf)
