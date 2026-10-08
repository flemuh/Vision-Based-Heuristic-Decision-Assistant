from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading


class InputCommitPhase(str, Enum):
    IDLE = "idle"
    SCHEDULED = "scheduled"
    PREPARING = "preparing"
    QUIESCING = "quiescing"
    COMMITTING = "committing"
    AWAIT_CONFIRM = "await-confirm"


@dataclass(frozen=True)
class InputActionToken:
    episode_id: str
    generation: int
    mask: int
    jewel: str
    position: int

    @property
    def legacy_key(self) -> tuple[str, int, str, int]:
        return (self.episode_id, self.mask, self.jewel, self.position)


class AutoPlayCommitCoordinator:
    """Single-owner state machine for one Auto Play physical-input transaction.

    Generation is bumped whenever live state is invalidated/confirmed. Any late
    callback carrying an older token is therefore stale and cannot acquire the
    commit path. Timing is intentionally absent: this class is synchronization
    and idempotency only.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._phase = InputCommitPhase.IDLE
        self._token: InputActionToken | None = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def phase(self) -> InputCommitPhase:
        with self._lock:
            return self._phase

    @property
    def token(self) -> InputActionToken | None:
        with self._lock:
            return self._token

    @property
    def active(self) -> bool:
        with self._lock:
            return self._phase is not InputCommitPhase.IDLE

    def reserve(self, episode_id: str, mask: int, jewel: str, position: int) -> InputActionToken | None:
        with self._lock:
            if self._phase is not InputCommitPhase.IDLE:
                return None
            token = InputActionToken(
                episode_id=str(episode_id),
                generation=self._generation,
                mask=int(mask),
                jewel=str(jewel),
                position=int(position),
            )
            self._token = token
            self._phase = InputCommitPhase.SCHEDULED
            return token

    def _advance(self, token: InputActionToken, expected: InputCommitPhase, target: InputCommitPhase) -> bool:
        with self._lock:
            if self._token != token or token.generation != self._generation or self._phase is not expected:
                return False
            self._phase = target
            return True

    def begin_prepare(self, token: InputActionToken) -> bool:
        return self._advance(token, InputCommitPhase.SCHEDULED, InputCommitPhase.PREPARING)

    def begin_quiesce(self, token: InputActionToken) -> bool:
        return self._advance(token, InputCommitPhase.PREPARING, InputCommitPhase.QUIESCING)

    def begin_commit(self, token: InputActionToken) -> bool:
        return self._advance(token, InputCommitPhase.QUIESCING, InputCommitPhase.COMMITTING)

    def mark_clicked(self, token: InputActionToken) -> bool:
        return self._advance(token, InputCommitPhase.COMMITTING, InputCommitPhase.AWAIT_CONFIRM)

    def is_current(self, token: InputActionToken) -> bool:
        with self._lock:
            return self._token == token and token.generation == self._generation

    def confirm(self, token: InputActionToken) -> bool:
        with self._lock:
            if self._token != token or token.generation != self._generation:
                return False
            if self._phase is not InputCommitPhase.AWAIT_CONFIRM:
                return False
            self._generation += 1
            self._phase = InputCommitPhase.IDLE
            self._token = None
            return True

    def invalidate(self) -> int:
        """Invalidate every outstanding callback and return the new generation."""
        with self._lock:
            self._generation += 1
            self._phase = InputCommitPhase.IDLE
            self._token = None
            return self._generation

    def abort(self, token: InputActionToken | None = None) -> bool:
        """Abort the current transaction without reusing its generation."""
        with self._lock:
            if token is not None and self._token != token:
                return False
            self._generation += 1
            self._phase = InputCommitPhase.IDLE
            self._token = None
            return True
