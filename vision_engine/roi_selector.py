from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .calibration import CalibrationCheck, component_band, quality_band


def _bgr_to_photo(img: np.ndarray, max_w: int, max_h: int):
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    dw = max(1, int(round(w * scale)))
    dh = max(1, int(round(h * scale)))
    if scale < 0.999:
        disp = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        disp = img
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), scale, dw, dh


def select_roi_dialog(
    parent: tk.Misc,
    img: np.ndarray,
    title: str,
    instruction: str,
    confirm_text: str = "Confirm selection",
) -> Optional[tuple[int, int, int, int]]:
    """Tk-native ROI selector.

    The complete source image is scaled to fit the user's screen. The user drags
    a rectangle and clicks an explicit Confirm button; Enter/Space is never
    required. Returned coordinates are mapped back to the original image.
    """
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.grab_set()
    win.attributes("-topmost", True)

    sw = max(800, int(win.winfo_screenwidth() * 0.86))
    sh = max(600, int(win.winfo_screenheight() * 0.72))
    photo, scale, dw, dh = _bgr_to_photo(img, sw, sh)

    outer = ttk.Frame(win, padding=10)
    outer.pack(fill="both", expand=True)
    ttk.Label(
        outer,
        text=instruction,
        font=("Segoe UI", 11, "bold"),
        wraplength=max(500, dw),
        justify="left",
    ).pack(anchor="w", pady=(0, 8))
    ttk.Label(
        outer,
        text="Drag with the left mouse button. Adjust by drawing again. Then click Confirm selection.",
        wraplength=max(500, dw),
    ).pack(anchor="w", pady=(0, 8))

    canvas = tk.Canvas(outer, width=dw, height=dh, highlightthickness=1, highlightbackground="#777")
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo

    state = {"start": None, "rect": None, "coords": None, "result": None}
    selected_var = tk.StringVar(value="No area selected yet")

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def on_down(event):
        x = clamp(canvas.canvasx(event.x), 0, dw - 1)
        y = clamp(canvas.canvasy(event.y), 0, dh - 1)
        state["start"] = (x, y)
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(x, y, x, y, outline="#00d7ff", width=3)
        state["coords"] = None
        confirm_btn.config(state="disabled")

    def on_move(event):
        if state["start"] is None or state["rect"] is None:
            return
        x0, y0 = state["start"]
        x1 = clamp(canvas.canvasx(event.x), 0, dw - 1)
        y1 = clamp(canvas.canvasy(event.y), 0, dh - 1)
        canvas.coords(state["rect"], x0, y0, x1, y1)

    def on_up(event):
        if state["start"] is None:
            return
        x0, y0 = state["start"]
        x1 = clamp(canvas.canvasx(event.x), 0, dw - 1)
        y1 = clamp(canvas.canvasy(event.y), 0, dh - 1)
        xa, xb = sorted((x0, x1)); ya, yb = sorted((y0, y1))
        if xb - xa < 5 or yb - ya < 5:
            selected_var.set("Selection too small. Draw the area again.")
            confirm_btn.config(state="disabled")
            return
        state["coords"] = (xa, ya, xb, yb)
        ow = int(round((xb - xa) / scale)); oh = int(round((yb - ya) / scale))
        selected_var.set(f"Selected area: {ow} × {oh} px. Click Confirm selection if it looks correct.")
        confirm_btn.config(state="normal")

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)

    bottom = ttk.Frame(outer)
    bottom.pack(fill="x", pady=(8, 0))
    ttk.Label(bottom, textvariable=selected_var).pack(side="left", fill="x", expand=True)

    def confirm():
        coords = state["coords"]
        if coords is None:
            return
        xa, ya, xb, yb = coords
        x = int(round(xa / scale)); y = int(round(ya / scale))
        w = int(round((xb - xa) / scale)); h = int(round((yb - ya) / scale))
        ih, iw = img.shape[:2]
        x = max(0, min(iw - 1, x)); y = max(0, min(ih - 1, y))
        w = max(1, min(iw - x, w)); h = max(1, min(ih - y, h))
        state["result"] = (x, y, w, h)
        win.destroy()

    def cancel():
        state["result"] = None
        win.destroy()

    ttk.Button(bottom, text="Cancel", command=cancel).pack(side="right", padx=(6, 0))
    confirm_btn = ttk.Button(bottom, text=confirm_text, command=confirm, state="disabled")
    confirm_btn.pack(side="right")
    win.protocol("WM_DELETE_WINDOW", cancel)

    win.update_idletasks()
    req_w = min(win.winfo_reqwidth(), win.winfo_screenwidth() - 40)
    req_h = min(win.winfo_reqheight(), win.winfo_screenheight() - 80)
    x = max(0, (win.winfo_screenwidth() - req_w) // 2)
    y = max(0, (win.winfo_screenheight() - req_h) // 2)
    win.geometry(f"{req_w}x{req_h}+{x}+{y}")
    parent.wait_window(win)
    return state["result"]


def select_board_cell_dialog(
    parent: tk.Misc,
    board_img: np.ndarray,
    title: str,
    instruction: str,
    center_index: int = 12,
) -> Optional[int]:
    """Select one visible 5x5 board cell with a click + explicit confirmation."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.grab_set()
    win.attributes("-topmost", True)

    sw = max(500, int(win.winfo_screenwidth() * 0.55))
    sh = max(500, int(win.winfo_screenheight() * 0.62))
    photo, scale, dw, dh = _bgr_to_photo(board_img, sw, sh)

    outer = ttk.Frame(win, padding=10)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text=instruction, font=("Segoe UI", 11, "bold"), wraplength=max(450, dw)).pack(anchor="w", pady=(0, 8))
    ttk.Label(outer, text="Click the matching cell once. The selected cell will be highlighted; then click Confirm cell.").pack(anchor="w", pady=(0, 8))

    canvas = tk.Canvas(outer, width=dw, height=dh, highlightthickness=1, highlightbackground="#777")
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo
    for i in range(1, 5):
        canvas.create_line(i * dw / 5, 0, i * dw / 5, dh, fill="#ffffff", width=1)
        canvas.create_line(0, i * dh / 5, dw, i * dh / 5, fill="#ffffff", width=1)

    state = {"idx": None, "highlight": None, "result": None}
    chosen_var = tk.StringVar(value="No cell selected")

    def choose(event):
        c = min(4, max(0, int(event.x * 5 / max(1, dw))))
        r = min(4, max(0, int(event.y * 5 / max(1, dh))))
        idx = r * 5 + c
        if idx == center_index:
            chosen_var.set("The MU center cannot be used. Choose a jewel cell.")
            confirm_btn.config(state="disabled")
            return
        state["idx"] = idx
        if state["highlight"] is not None:
            canvas.delete(state["highlight"])
        x0, y0 = c * dw / 5, r * dh / 5
        x1, y1 = (c + 1) * dw / 5, (r + 1) * dh / 5
        state["highlight"] = canvas.create_rectangle(x0, y0, x1, y1, outline="#00d7ff", width=4)
        chosen_var.set(f"Selected L{r+1}C{c+1}")
        confirm_btn.config(state="normal")

    canvas.bind("<Button-1>", choose)

    bottom = ttk.Frame(outer)
    bottom.pack(fill="x", pady=(8, 0))
    ttk.Label(bottom, textvariable=chosen_var).pack(side="left", fill="x", expand=True)

    def confirm():
        state["result"] = state["idx"]
        win.destroy()

    def cancel():
        state["result"] = None
        win.destroy()

    ttk.Button(bottom, text="Cancel", command=cancel).pack(side="right", padx=(6, 0))
    confirm_btn = ttk.Button(bottom, text="Confirm cell", command=confirm, state="disabled")
    confirm_btn.pack(side="right")
    win.protocol("WM_DELETE_WINDOW", cancel)

    win.update_idletasks()
    req_w = min(win.winfo_reqwidth(), win.winfo_screenwidth() - 40)
    req_h = min(win.winfo_reqheight(), win.winfo_screenheight() - 80)
    x = max(0, (win.winfo_screenwidth() - req_w) // 2)
    y = max(0, (win.winfo_screenheight() - req_h) // 2)
    win.geometry(f"{req_w}x{req_h}+{x}+{y}")
    parent.wait_window(win)
    return state["result"]



def _band_color(band: str) -> str:
    return {
        "HIGH": "#167a2d",
        "MEDIUM": "#8a6d00",
        "LOW": "#b05a00",
        "REJECT": "#b00020",
        "GOOD": "#167a2d",
        "OK": "#8a6d00",
        "WEAK": "#b05a00",
        "BAD": "#b00020",
    }.get(str(band).upper(), "#444444")


def _metric_rows(kind: str, check: CalibrationCheck):
    """Human-readable sub-scores for every first-run geometry check."""
    m = check.metrics
    if kind == "game":
        return [
            ("Panel shape", m.get("score_shape", 0.0), f"aspect {m.get('aspect', 0):.2f}", "The Jewel Bingo panel should be close to square and must stop before Event Inventory."),
            ("Dark UI coverage", m.get("score_dark_ui", 0.0), f"dark pixels {m.get('dark_fraction', 0)*100:.0f}%", "A correct crop is mostly the dark Jewel Bingo interface, not the game world/desktop."),
            ("Capture size", m.get("score_size", 0.0), f"{m.get('width_px', 0):.0f}×{m.get('height_px', 0):.0f}px", "Enough pixels are needed for the board and jewel icons to be recognized reliably."),
        ]
    if kind == "board":
        return [
            ("Square shape", m.get("score_shape", 0.0), f"aspect {m.get('aspect', 0):.2f}", "The 5×5 board crop should be nearly square."),
            ("Board size", m.get("score_size", 0.0), f"panel fraction {m.get('width_fraction',0):.2f}×{m.get('height_fraction',0):.2f}", "The crop should contain only the 25 cells, not title/counters/instructions."),
            ("Board position", m.get("score_location", 0.0), f"center ({m.get('center_x_fraction',0):.2f}, {m.get('center_y_fraction',0):.2f})", "The board normally sits in the left/center region of the Jewel Bingo panel."),
            ("Cell resolution", m.get("score_cell_size", 0.0), f"~{m.get('cell_px',0):.1f}px/cell", "Each cell needs enough pixels for stable jewel classification."),
            ("5×5 periodicity", m.get("score_grid", 0.0), f"period {m.get('period_regularity',0):.2f}, projection {m.get('projection_grid_score',0):.2f}", "This measures repeated cell spacing/borders. Blue selection glow can reduce this score, so visual overlay confirmation remains important."),
        ]
    if kind == "current":
        return [
            ("Icon shape", m.get("score_shape", 0.0), f"aspect {m.get('aspect',0):.2f}", "The ROI should tightly surround one jewel icon and be roughly square."),
            ("Icon size", m.get("score_size", 0.0), f"panel fraction {m.get('width_fraction',0):.3f}×{m.get('height_fraction',0):.3f}", "Do not include the box icon or the remaining-count number."),
            ("Top position", m.get("score_location", 0.0), f"vertical center {m.get('center_y_fraction',0):.2f}", "The current jewel normally appears near the top of the panel."),
            ("Visual contrast", m.get("score_contrast", 0.0), f"contrast {m.get('contrast',0):.1f}", "A visible jewel should have enough light/dark structure to recognize."),
            ("Color signal", m.get("score_color", 0.0), f"colorful {m.get('colorful_fraction',0)*100:.0f}%", "A blank/grey area is likely not the current jewel."),
        ]
    if kind == "number":
        return [
            ("ROI size", m.get("score_size", 0.0), f"panel fraction {m.get('width_fraction',0):.3f}×{m.get('height_fraction',0):.3f}", "Select only the digits, not the surrounding label/panel."),
            ("Digit contrast", m.get("score_contrast", 0.0), f"contrast {m.get('contrast',0):.1f}", "Visible digits need enough contrast for OCR."),
            ("Digit edges", m.get("score_digit_edges", 0.0), f"edge pixels {m.get('edge_fraction',0)*100:.0f}%", "Text normally creates a moderate amount of sharp edge structure."),
        ]
    return []


def confirm_geometry_dialog(
    parent: tk.Misc,
    preview_img: np.ndarray,
    title: str,
    check: CalibrationCheck,
    kind: str,
    visual_instruction: str,
    accept_text: str = "Use this selection",
) -> bool:
    """Explain geometry score + sub-scores before anything is persisted.

    HIGH confidence may be accepted directly. MEDIUM/LOW require an explicit
    checkbox acknowledging the visual preview. REJECT/hard-rule failures are
    blocked so setup cannot silently continue with bad geometry.
    """
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.grab_set()
    win.attributes("-topmost", True)

    sw = max(520, int(win.winfo_screenwidth() * 0.58))
    sh = max(380, int(win.winfo_screenheight() * 0.38))
    photo, _scale, dw, dh = _bgr_to_photo(preview_img, sw, sh)
    outer = ttk.Frame(win, padding=12)
    outer.pack(fill="both", expand=True)

    band = quality_band(check.score)
    headline = (
        f"{band} confidence — score {check.score:.2f}"
        if check.ok else f"REJECTED — score {check.score:.2f}"
    )
    ttk.Label(outer, text=headline, font=("Segoe UI", 13, "bold"), foreground=_band_color(band if check.ok else "REJECT")).pack(anchor="w")
    ttk.Label(
        outer,
        text="Bands: HIGH ≥ 0.80  |  MEDIUM 0.65–0.79  |  LOW 0.50–0.64  |  REJECT < 0.50",
        foreground="#555555",
    ).pack(anchor="w", pady=(2, 6))
    ttk.Label(outer, text=visual_instruction, wraplength=760, justify="left").pack(anchor="w", pady=(0, 8))

    body = ttk.Frame(outer)
    body.pack(fill="both", expand=True)
    img_frame = ttk.Frame(body)
    img_frame.pack(side="left", anchor="n", padx=(0, 12))
    canvas = tk.Canvas(img_frame, width=dw, height=dh, highlightthickness=1, highlightbackground="#777")
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo

    metrics = ttk.Frame(body)
    metrics.pack(side="left", fill="both", expand=True, anchor="n")
    rows = _metric_rows(kind, check)
    if rows:
        ttk.Label(metrics, text="Why this score", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
    for label, val, raw, explanation in rows:
        cb = component_band(val)
        line = ttk.Frame(metrics)
        line.pack(fill="x", anchor="w", pady=2)
        ttk.Label(line, text=f"{label}: {val:.2f}  {cb}", foreground=_band_color(cb), width=28).pack(side="left", anchor="n")
        ttk.Label(line, text=f"{raw} — {explanation}", wraplength=500, justify="left").pack(side="left", fill="x", expand=True)

    if check.messages:
        ttk.Label(metrics, text="Hard-rule notes", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Label(metrics, text="\n".join(f"• {m}" for m in check.messages), foreground="#b00020", wraplength=600, justify="left").pack(anchor="w")

    result = {"value": False}
    review_var = tk.BooleanVar(value=False)
    bottom = ttk.Frame(outer)
    bottom.pack(fill="x", pady=(10, 0))

    def accept():
        result["value"] = True
        win.destroy()

    def reject():
        result["value"] = False
        win.destroy()

    ttk.Button(bottom, text="Select again", command=reject).pack(side="right", padx=(6, 0))
    accept_btn = ttk.Button(bottom, text=accept_text, command=accept)
    accept_btn.pack(side="right")

    can_review = check.ok and check.score >= 0.50
    if not check.ok or check.score < 0.50:
        accept_btn.config(state="disabled")
        ttk.Label(bottom, text="Automatic rules rejected this selection. Adjust it.", foreground="#b00020").pack(side="left")
    elif check.score < 0.80:
        accept_btn.config(state="disabled")
        text = "I visually checked the preview and it matches the requested area."
        cb = ttk.Checkbutton(bottom, text=text, variable=review_var)
        cb.pack(side="left")
        def toggle(*_):
            accept_btn.config(state="normal" if review_var.get() and can_review else "disabled")
        review_var.trace_add("write", toggle)
    else:
        ttk.Label(bottom, text="High-confidence automatic geometry check.", foreground="#167a2d").pack(side="left")

    win.protocol("WM_DELETE_WINDOW", reject)
    win.update_idletasks()
    req_w = min(win.winfo_reqwidth(), win.winfo_screenwidth() - 40)
    req_h = min(win.winfo_reqheight(), win.winfo_screenheight() - 80)
    x = max(0, (win.winfo_screenwidth() - req_w) // 2)
    y = max(0, (win.winfo_screenheight() - req_h) // 2)
    win.geometry(f"{req_w}x{req_h}+{x}+{y}")
    parent.wait_window(win)
    return bool(result["value"])

def confirm_board_alignment_dialog(
    parent: tk.Misc,
    board_img: np.ndarray,
    check: CalibrationCheck,
) -> bool:
    """Explain board geometry and require visual review when confidence < HIGH."""
    preview = board_img.copy()
    h, w = preview.shape[:2]
    # Draw exactly the geometry the vision code will use later.
    for i in range(1, 5):
        x = int(round(i * w / 5))
        y = int(round(i * h / 5))
        cv2.line(preview, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(preview, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(preview, (1, 1), (max(1, w - 2), max(1, h - 2)), (255, 220, 0), 2)
    cv2.rectangle(
        preview,
        (int(round(2*w/5)), int(round(2*h/5))),
        (int(round(3*w/5))-1, int(round(3*h/5))-1),
        (0, 255, 120), 2,
    )
    return confirm_geometry_dialog(
        parent,
        preview,
        "Validate 5x5 board geometry",
        check,
        "board",
        "Every white rectangle must cover exactly ONE board cell, and the green center rectangle must contain MU. "
        "The overall score is NOT a binary truth: MEDIUM/LOW requires your visual confirmation; REJECT cannot be saved.",
        accept_text="Use this 5x5 board",
    )


def confirm_reference_templates_dialog(
    parent: tk.Misc,
    templates: dict[str, np.ndarray],
    confidence: float,
    names: dict[str, str],
    metrics: dict[str, float] | None = None,
) -> bool:
    """Show six automatically detected right-side x4 jewel references."""
    win = tk.Toplevel(parent)
    win.title("Validate automatically detected jewels")
    win.transient(parent)
    win.grab_set()
    win.attributes("-topmost", True)

    outer = ttk.Frame(win, padding=12)
    outer.pack(fill="both", expand=True)
    band = quality_band(confidence)
    ttk.Label(outer, text="Six jewel references detected", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(
        outer,
        text=f"Reference geometry: {band} confidence — {confidence:.2f}",
        foreground=_band_color(band),
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w", pady=(4, 2))
    ttk.Label(
        outer,
        text=(
            "Check the six icons below. They must read Bless, Soul, Life, Creation, Harmony, Chaos in this exact order. "
            "The score measures vertical spacing, horizontal alignment and color signal; your icon review is still the final guard."
        ),
        wraplength=760,
        justify="left",
    ).pack(anchor="w", pady=(0, 6))
    if metrics:
        score_rows = [
            ("Vertical spacing", metrics.get("score_spacing", 0.0), f"variation {metrics.get('spacing_cv',0):.3f}"),
            ("Horizontal alignment", metrics.get("score_alignment", 0.0), f"x spread {metrics.get('x_std_px',0):.1f}px"),
            ("Color signal", metrics.get("score_color_signal", 0.0), f"peak {metrics.get('mean_peak_strength',0):.1f}"),
            ("List span", metrics.get("score_span", 0.0), f"span ratio {metrics.get('span_ratio',0):.2f}"),
        ]
        metrics_box = ttk.Frame(outer)
        metrics_box.pack(fill="x", pady=(0, 8))
        for label, val, raw in score_rows:
            cb = component_band(val)
            ttk.Label(metrics_box, text=f"{label}: {val:.2f} {cb}  ({raw})", foreground=_band_color(cb)).pack(anchor="w")

    row = ttk.Frame(outer)
    row.pack(fill="x")
    photos = []
    for c, label in enumerate(("BL", "SO", "LI", "CR", "HA", "CH")):
        card = ttk.Frame(row, padding=5)
        card.grid(row=0, column=c, sticky="n")
        patch = templates[label]
        # Enlarge tiny game sprites without smoothing so the user can verify them.
        h, w = patch.shape[:2]
        scale = max(1, min(5, int(72 / max(1, max(h, w)))))
        disp = cv2.resize(patch, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        photos.append(photo)
        ttk.Label(card, image=photo).pack()
        ttk.Label(card, text=f"{label}\n{names.get(label, label)}", justify="center").pack(pady=(4,0))
    win._template_photos = photos

    result = {"value": False}
    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(12,0))

    def accept():
        result["value"] = True
        win.destroy()

    def reject():
        result["value"] = False
        win.destroy()

    ttk.Label(buttons, text="If any icon/name is wrong, choose Manual instead.").pack(side="left")
    ttk.Button(buttons, text="Manual instead", command=reject).pack(side="right", padx=(6,0))
    use_btn = ttk.Button(buttons, text="Use these 6", command=accept)
    use_btn.pack(side="right")
    if confidence < 0.50:
        use_btn.config(state="disabled")
        ttk.Label(buttons, text="Reference geometry is too weak; use Manual instead.", foreground="#b00020").pack(side="left", padx=(8,0))
    win.protocol("WM_DELETE_WINDOW", reject)
    win.update_idletasks()
    req_w = min(win.winfo_reqwidth(), win.winfo_screenwidth() - 40)
    req_h = min(win.winfo_reqheight(), win.winfo_screenheight() - 80)
    x = max(0, (win.winfo_screenwidth() - req_w) // 2)
    y = max(0, (win.winfo_screenheight() - req_h) // 2)
    win.geometry(f"{req_w}x{req_h}+{x}+{y}")
    parent.wait_window(win)
    return bool(result["value"])


def show_vision_validation_dialog(
    parent: tk.Misc,
    passed: bool,
    rows: list[tuple[str, float, str, bool, str]],
    geometry_rows: list[tuple[str, float, str]],
) -> None:
    """Final setup report with both geometry and classification diagnostics."""
    win = tk.Toplevel(parent)
    win.title("Step 5/5 – Vision validation report")
    win.transient(parent)
    win.grab_set()
    win.attributes("-topmost", True)
    outer = ttk.Frame(win, padding=12)
    outer.pack(fill="both", expand=True)

    ttk.Label(
        outer,
        text="VISION TEST PASSED" if passed else "VISION TEST FAILED",
        foreground="#167a2d" if passed else "#b00020",
        font=("Segoe UI", 13, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        outer,
        text="This final test combines the saved geometry with live jewel-classification confidence. START stays locked if any required live check fails.",
        wraplength=780,
        justify="left",
    ).pack(anchor="w", pady=(4, 10))

    ttk.Label(outer, text="Saved geometry", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    for name, score, explanation in geometry_rows:
        band = quality_band(score) if score > 0 else "REVIEWED"
        color = _band_color(band) if band != "REVIEWED" else "#24527a"
        ttk.Label(
            outer,
            text=f"• {name}: {score:.2f} / {band} — {explanation}" if score > 0 else f"• {name}: reviewed — {explanation}",
            foreground=color,
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=1)

    ttk.Label(outer, text="Live recognition", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
    for name, value, requirement, ok, explanation in rows:
        ttk.Label(
            outer,
            text=f"{'✅' if ok else '❌'} {name}: {value:.3f}  | required {requirement} — {explanation}",
            foreground="#167a2d" if ok else "#b00020",
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=1)

    ttk.Label(
        outer,
        text=(
            "If a geometry score is MEDIUM/LOW but was visually confirmed, it may still be used. "
            "A failed live-recognition threshold is different: recalibrate/relearn the indicated item before START."
        ),
        foreground="#555555",
        wraplength=780,
        justify="left",
    ).pack(anchor="w", pady=(10, 8))
    ttk.Button(outer, text="Close", command=win.destroy).pack(anchor="e")
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.update_idletasks()
    req_w = min(win.winfo_reqwidth(), win.winfo_screenwidth() - 40)
    req_h = min(win.winfo_reqheight(), win.winfo_screenheight() - 80)
    x = max(0, (win.winfo_screenwidth() - req_w) // 2)
    y = max(0, (win.winfo_screenheight() - req_h) // 2)
    win.geometry(f"{req_w}x{req_h}+{x}+{y}")
    parent.wait_window(win)
