"""R3.7 experiments: integer-mask radix tree and immutable shared binary table.

Not wired into live solver: only promote an alternative after deterministic
correctness + actual end-to-end performance improve. Shared memory holds only
board-independent score totals, NOT Monte-Carlo samples or mutable ML state.
"""
from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory


class RadixTreeCache:
    """5-level radix-32 trie for 25-bit occupancy masks (experimental)."""

    def __init__(self) -> None:
        self.root: dict = {}
        self.size = 0

    def put(self, key: int, value: int) -> None:
        if not 0 <= key < (1 << 25):
            raise ValueError('invalid board mask')
        node = self.root
        for shift in (20, 15, 10, 5):
            node = node.setdefault((key >> shift) & 31, {})
        leaf = key & 31
        if leaf not in node:
            self.size += 1
        node[leaf] = int(value)

    def get(self, key: int) -> int | None:
        node = self.root
        for shift in (20, 15, 10, 5):
            node = node.get((key >> shift) & 31)
            if node is None:
                return None
        return node.get(key & 31)


def _slot(mask: int, size: int) -> int:
    # Knuth's multiplicative hash; 25-bit keys have no hash collisions, but
    # table slots will collide and are resolved with bounded linear probing.
    k = mask ^ (mask >> 13)
    return (k * 2654435761) & (size - 1)


class SharedScoreTable:
    """Read-only, open-addressed shared score table (mask,total) int32 pairs.

    Parent populates once before worker spawn. Readers do NOT lock or mutate.
    The owner must call unlink only after readers exit; readers only close.
    Score total is a scalar; this table is NOT a drop-in replacement for
    score_mask(), which also returns line names and cleared-cell counts.
    """

    def __init__(self, shm: SharedMemory, capacity: int, owner: bool = False):
        self.shm = shm
        self.capacity = capacity
        self.owner = owner
        self.data = shm.buf.cast('i')

    @classmethod
    def build(cls, entries: dict[int, int]) -> 'SharedScoreTable':
        capacity = 2
        while capacity < max(2, len(entries) * 2):
            capacity <<= 1
        shm = SharedMemory(create=True, size=capacity * 8)
        table = cls(shm, capacity, owner=True)
        try:
            for slot in range(capacity):
                table.data[slot * 2] = -1
            for key, val in entries.items():
                if not 0 <= key < (1 << 25):
                    raise ValueError('invalid 25-bit mask')
                i = _slot(key, capacity)
                for _ in range(capacity):
                    if table.data[2*i] == -1:
                        table.data[2*i], table.data[2*i+1] = int(key), int(val)
                        break
                    i = (i + 1) & (capacity - 1)
                else:
                    raise RuntimeError('shared hash table unexpectedly full')
        except Exception:
            table.close()
            table.unlink()
            raise
        return table

    @classmethod
    def attach(cls, name: str, capacity: int) -> 'SharedScoreTable':
        return cls(SharedMemory(name=name), capacity, owner=False)

    def get(self, key: int) -> int | None:
        idx = _slot(key, self.capacity)
        for _ in range(self.capacity):
            existing = self.data[idx * 2]
            if existing == -1:
                return None
            if existing == key:
                return int(self.data[idx * 2 + 1])
            idx = (idx + 1) & (self.capacity - 1)
        return None

    def close(self) -> None:
        self.data.release()
        self.shm.close()

    def unlink(self) -> None:
        if self.owner:
            self.shm.unlink()
