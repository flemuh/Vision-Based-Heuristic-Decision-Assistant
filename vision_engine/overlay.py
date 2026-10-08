from __future__ import annotations

import sys
import tkinter as tk

from .capture import ScreenRegion


class MoveOverlay:
    """Small transparent, click-through highlight over the recommended cell.

    Works best on Windows. On other platforms it degrades to a normal topmost
    borderless label and can be disabled from app_config.json.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: tk.Toplevel | None = None

    def hide(self) -> None:
        if self.win is not None:
            try:
                self.win.destroy()
            except Exception:
                pass
            self.win = None

    def show_cell(
        self,
        region: ScreenRegion,
        board_roi: tuple[int, int, int, int],
        row: int,
        col: int,
        text: str,
    ) -> None:
        self.hide()
        bx, by, bw, bh = board_roi
        cw, ch = bw / 5.0, bh / 5.0
        x = int(region.left + bx + (col - 1) * cw)
        y = int(region.top + by + (row - 1) * ch)
        w = max(30, int(cw))
        h = max(30, int(ch))

        win = tk.Toplevel(self.root)
        self.win = win
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        transparent = "#ff00ff"
        win.configure(bg=transparent)
        try:
            win.wm_attributes("-transparentcolor", transparent)
        except tk.TclError:
            pass
        win.geometry(f"{w}x{h}+{x}+{y}")
        canvas = tk.Canvas(win, width=w, height=h, bg=transparent, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_rectangle(3, 3, w - 4, h - 4, outline="#00ff66", width=4)
        canvas.create_text(w / 2, max(10, h * 0.18), text=text, fill="white", font=("Segoe UI", 9, "bold"))

        if sys.platform.startswith("win"):
            try:
                import ctypes
                hwnd = win.winfo_id()
                GWL_EXSTYLE = -20
                WS_EX_TRANSPARENT = 0x00000020
                WS_EX_LAYERED = 0x00080000
                style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED)
            except Exception:
                pass
