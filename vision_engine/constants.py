from __future__ import annotations

JEWELS = ("BL", "SO", "LI", "CR", "HA", "CH")
JEWEL_NAMES = {
    "BL": "Bless",
    "SO": "Soul",
    "LI": "Life",
    "CR": "Creation",
    "HA": "Harmony",
    "CH": "Chaos",
}
JEWEL_INDEX = {j: i for i, j in enumerate(JEWELS)}
CENTER = 12

# Four Lucky lines crossing MU. The center itself is not a jewel cell.
LUCKY_LINES = {
    "H": (10, 11, 13, 14),
    "V": (2, 7, 17, 22),
    "\\": (0, 6, 18, 24),
    "/": (4, 8, 16, 20),
}

# Normal lines are rows/columns that do not cross MU.
NORMAL_LINES = {
    "R1": (0, 1, 2, 3, 4),
    "R2": (5, 6, 7, 8, 9),
    "R4": (15, 16, 17, 18, 19),
    "R5": (20, 21, 22, 23, 24),
    "C1": (0, 5, 10, 15, 20),
    "C2": (1, 6, 11, 16, 21),
    "C4": (3, 8, 13, 18, 23),
    "C5": (4, 9, 14, 19, 24),
}

LUCKY_SCORE = 312
NORMAL_SCORE = 240
JEWEL_SCORE = 45
MAX_DRAWS = 14
COPIES_PER_JEWEL = 4
