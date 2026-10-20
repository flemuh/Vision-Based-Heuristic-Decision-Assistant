"""Conservative, exact-history match from internally consistent >=999 games.

No future jewel is consulted in live decisions. The DB remains the source of
truth and does not need to be migrated or modified to build this index.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .board import Board
from .constants import JEWELS
from .persistence import BingoDB
from .scoring import score_mask


@dataclass(frozen=True)
class WinningExample:
    game_id: str
    board: tuple[str | None, ...]
    sequence: tuple[str, ...]
    actions: tuple[int, ...]
    total: int


def validated_episodes(db: BingoDB) -> tuple[WinningExample, ...]:
    """Validate ALL 14-step recorded episodes before promoting any history to ML.

    Never trust a displayed score alone: validate jewel order, complete action
    prefix/masks, final mask and the computed/independently verified totals.
    """
    examples = []
    with db.connect() as con:
        games = con.execute(
            """SELECT id,board_json,sequence_json,counts_json,selected_mask,
                      computed_score_json,verified_score_json
               FROM games WHERE finished_at IS NOT NULL"""
        ).fetchall()
        for game in games:
            try:
                board = Board(tuple(json.loads(game["board_json"])))
                sequence = tuple(json.loads(game["sequence_json"]))
                if len(sequence) != 14 or any(j not in JEWELS for j in sequence):
                    continue
                rows = con.execute(
                    "SELECT step,mask_before,jewel,action_pos FROM actions WHERE game_id=? ORDER BY step",
                    (game["id"],),
                ).fetchall()
                if len(rows) != 14:
                    continue
                positions = []
                mask = 0
                valid = True
                for step, row in enumerate(rows):
                    pos = int(row["action_pos"])
                    if (int(row["step"]) != step or int(row["mask_before"]) != mask
                            or not 0 <= pos < 25 or pos == 12 or (mask >> pos) & 1
                            or row["jewel"] != sequence[step] or board.cells[pos] != sequence[step]):
                        valid = False
                        break
                    positions.append(pos)
                    mask |= 1 << pos
                if not valid or mask != int(game["selected_mask"]):
                    continue
                if game["counts_json"]:
                    stored_counts = tuple(int(x) for x in json.loads(game["counts_json"]))
                    if stored_counts != board.jewel_counts_in_mask(mask):
                        continue
                computed = json.loads(game["computed_score_json"] or "null")
                verified = json.loads(game["verified_score_json"] or "null")
                actual = score_mask(mask)
                if (not isinstance(computed, dict) or int(computed.get("total", -1)) != actual.total):
                    continue
                if verified is not None and (not isinstance(verified, dict)
                        or int(verified.get("total", -1)) != actual.total):
                    continue
                examples.append(WinningExample(
                    str(game["id"]), board.cells, sequence, tuple(positions), actual.total,
                ))
            except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
                continue
    return tuple(examples)


class WinningMemory:
    """The only live-learning signal is a genuinely matching winning path.

    Key requires exact 25-cell board, accepted positions, previous jewel order,
    previous position order, and current jewel. No approximate layouts or
    predictions from a model trained on unrelated boards are eligible.
    """

    def __init__(self, examples: Iterable[WinningExample] = ()):
        self.examples = tuple(examples)
        self._by_state: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for example in self.examples:
            mask = 0
            for step, (jewel, pos) in enumerate(zip(example.sequence, example.actions)):
                key = (example.board, mask, example.sequence[:step], example.actions[:step], jewel)
                self._by_state[key][pos].append(example.total)
                mask |= 1 << pos

    @classmethod
    def from_db(cls, db: BingoDB) -> 'WinningMemory':
        # Threshold is inclusive: 999 itself is a valid factual outcome.
        return cls(x for x in validated_episodes(db) if x.total >= 999)

    @property
    def winning_game_ids(self) -> frozenset[str]:
        return frozenset(x.game_id for x in self.examples)

    def matching_scores(
        self,
        board: Board,
        mask: int,
        current_jewel: str,
        legal_positions: Iterable[int],
        observed_jewels: Iterable[str] | None,
        observed_positions: Iterable[int] | None,
    ) -> dict[int, float]:
        if observed_jewels is None or observed_positions is None:
            return {}
        jewels = tuple(observed_jewels)
        positions = tuple(int(p) for p in observed_positions)
        if len(jewels) != mask.bit_count() or len(positions) != len(jewels):
            return {}
        # Reject stale/desynced tracking; partial matching must never suffice.
        observed_mask = 0
        for jewel, pos in zip(jewels, positions):
            if not 0 <= pos < 25 or (observed_mask >> pos) & 1 or board.cells[pos] != jewel:
                return {}
            observed_mask |= 1 << pos
        if observed_mask != mask:
            return {}
        state = (board.cells, mask, jewels, positions, current_jewel)
        historical = self._by_state.get(state)
        if not historical:
            return {}
        allowed = set(legal_positions)
        # Only observed winning decisions receive a positive score. Absence of
        # evidence is not evidence that other actions lose. Advisor uses scores
        # only to break mathematical ties.
        return {
            pos: 1.0 + max(0.0, (sum(totals) / len(totals) - 999.0) / 1000.0)
            for pos, totals in historical.items() if pos in allowed
        }
