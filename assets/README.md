# Vision bootstrap assets

`seed_board_features.npz` contains **numeric feature descriptors only** for 24 board-cell examples (four examples for each of the six jewel labels).

They bootstrap board-cell recognition before enough user-specific examples have been collected. The repository intentionally does **not** ship raw MU Online screenshots or sprite/image assets. Runtime visual examples learned from the user's own screen are written under `data/board_templates/` and `data/icon_templates/` and are ignored by Git.

The two visual domains are deliberately separate:

- **board cells**: framed cells that may contain blue selection glow;
- **current/x4 icons**: small standalone colored icons used for dynamic current-jewel matching.

Keeping these domains separate fixes a failure mode in V5.1 where one classifier attempted to model both appearances.
