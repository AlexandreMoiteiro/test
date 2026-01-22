from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pymupdf  # PyMuPDF


# =========================
# Assets / Modes
# =========================
ASSETS = {
    "landing": {
        "title": "Landing Ground Roll",
        "bg_default": "ldg_ground_roll.pdf",
        "json_default": "ldg_ground_roll.json",
        "bg_kind": "pdf",
        "page_default": 0,
        "round_to": 5,    # ft
        "label_font": 24, # bigger for landing
    },
    "takeoff": {
        "title": "Takeoff Ground Roll",
        "bg_default": "to_ground_roll.jpg",
        "json_default": "to_ground_roll.json",
        "bg_kind": "image",
        "page_default": 0,
        "round_to": 5,     # ft
        "label_font": 18,
    },
    "climb": {
        "title": "Climb Performance",
        "bg_default": "climb_perf.jpg",
        "json_default": "climb_perf.json",
        "bg_kind": "image",
        "page_default": 0,
        "round_to": 10,    # fpm
        "label_font": 18,
    },
}


# =========================
# File helpers
# =========================
def _here(name: str) -> Optional[Path]:
    p = Path(name)
    if p.exists():
        return p
    if "__file__" in globals():
        p2 = Path(__file__).resolve().parent / name
        if p2.exists():
            return p2
    return None


@st.cache_data(show_spinner=False)
def render_pdf_to_image(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def load_background(mode: str, upload_bg, page_index: int, zoom: float) -> Image.Image:
    info = ASSETS[mode]
    if info["bg_kind"] == "pdf":
        if upload_bg is not None:
            pdf_bytes = upload_bg.read()
        else:
            p = _here(info["bg_default"])
            if not p:
                raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
            pdf_bytes = p.read_bytes()
        return render_pdf_to_image(pdf_bytes, page_index=page_index, zoom=zoom)

    if upload_bg is not None:
        return Image.open(upload_bg).convert("RGB")

    p = _here(info["bg_default"])
    if not p:
        raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
    return Image.open(p).convert("RGB")


def load_json(mode: str, upload_json) -> Dict[str, Any]:
    info = ASSETS[mode]
    if upload_json is not None:
        return json.loads(upload_json.read().decode("utf-8"))
    p = _here(info["json_default"])
    if not p:
        raise FileNotFoundError(f"Não encontrei {info['json_default']}")
    return json.loads(p.read_text(encoding="utf-8"))


# =========================
# Normalization helpers
# =========================
def pt_xy(p: Any) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    if isinstance(p, (list, tuple)) and len(p) == 2:
        return float(p[0]), float(p[1])
    raise ValueError(f"Invalid point: {p}")


def normalize_panel(panel_pts: Any) -> List[Dict[str, float]]:
    if not isinstance(panel_pts, list) or len(panel_pts) != 4:
        return []
    out = []
    for p in panel_pts:
        x, y = pt_xy(p)
        out.append({"x": x, "y": y})
    return out


def normalize_panels(cap: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    out = {}
    pc = cap.get("panel_corners", {})
    if not isinstance(pc, dict):
        return out
    for k, pts in pc.items():
        out[k] = normalize_panel(pts)
    return out


# =========================
# Math helpers
# =========================
def fit_axis_value_from_ticks(ticks: List[Dict[str, float]], coord: str) -> Tuple[float, float]:
    xs = np.array([float(t[coord]) for t in ticks], dtype=float)
    vs = np.array([float(t["value"]) for t in ticks], dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, vs, rcond=None)[0]
    return float(a), float(b)


def axis_value(a: float, b: float, coord_val: float) -> float:
    return a * coord_val + b


def axis_coord_from_value(a: float, b: float, value: float) -> float:
    if abs(a) < 1e-12:
        raise ValueError("Axis fit degenerate (a ~ 0).")
    return (value - b) / a


def line_y_at_x(seg: Dict[str, float], x: float) -> float:
    x1, y1, x2, y2 = map(float, (seg["x1"], seg["y1"], seg["x2"], seg["y2"]))
    if abs(x2 - x1) < 1e-12:
        return y1
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def parse_pa_levels_ft(lines: Dict[str, List[Dict[str, float]]]) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for k, segs in lines.items():
        if not k.startswith("pa_"):
            continue
        if not segs:
            continue
        if k == "pa_sea_level":
            out.append((0.0, k))
            continue
        try:
            out.append((float(k.replace("pa_", "")), k))
        except Exception:
            pass
    out.sort(key=lambda t: t[0])
    return out


def interp_between_levels(v: float, levels: List[Tuple[float, str]]) -> Tuple[Tuple[float, str], Tuple[float, str], float]:
    if not levels:
        raise ValueError("No PA levels available (all pa_* lines empty?).")
    if v <= levels[0][0]:
        return levels[0], levels[0], 0.0
    if v >= levels[-1][0]:
        return levels[-1], levels[-1], 0.0
    for i in range(len(levels) - 1):
        a, ka = levels[i]
        b, kb = levels[i + 1]
        if a <= v <= b:
            alpha = (v - a) / (b - a) if b != a else 0.0
            return (a, ka), (b, kb), float(alpha)
    return levels[-1], levels[-1], 0.0


def round_to_step(x: float, step: float) -> float:
    return step * round(x / step)


def x_of_vertical_ref(seg: Dict[str, float]) -> float:
    return 0.5 * (float(seg["x1"]) + float(seg["x2"]))


# =========================
# Guide interpolation (robust)
# =========================
def interp_guides_y(
    guides: List[Dict[str, float]],
    x_ref: float,
    y_ref: float,
    x_target: float
) -> Tuple[float, Dict[str, Any]]:
    if not guides:
        return y_ref, {"used": "none"}

    rows = []
    for g in guides:
        yr = line_y_at_x(g, x_ref)
        yt = line_y_at_x(g, x_target)
        rows.append((yr, yt))

    rows.sort(key=lambda t: t[0])

    if y_ref <= rows[0][0]:
        return float(rows[0][1]), {"used": "clamp_low"}
    if y_ref >= rows[-1][0]:
        return float(rows[-1][1]), {"used": "clamp_high"}

    for i in range(len(rows) - 1):
        y0_ref, y0_tgt = rows[i]
        y1_ref, y1_tgt = rows[i + 1]
        if y0_ref <= y_ref <= y1_ref:
            denom = (y1_ref - y0_ref)
            a = 0.0 if abs(denom) < 1e-12 else (y_ref - y0_ref) / denom
            y_tgt = (1 - a) * y0_tgt + a * y1_tgt
            return float(y_tgt), {"used": "interp", "i0": i, "i1": i + 1, "alpha": float(a)}

    return y_ref, {"used": "fallback"}


def pick_guides(cap: Dict[str, Any], mode: str) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    g = cap.get("guides", {}) or {}

    if mode == "takeoff":
        return g.get("guides_weight", []) or [], g.get("guides_wind", []) or []

    # landing: prefer middle/right if present, else guides_weight/guides_wind
    mid = g.get("middle", []) or []
    rgt = g.get("right", []) or []
    if len(mid) == 0 and len(rgt) == 0:
        return g.get("guides_weight", []) or [], g.get("guides_wind", []) or []
    return mid, rgt


# =========================
# Drawing (arrows + smart label)
# =========================
def draw_arrow(draw: ImageDraw.ImageDraw, p1: Tuple[float, float], p2: Tuple[float, float], color: Tuple[int, int, int], w: int = 4):
    draw.line([p1, p2], fill=color, width=w)

    x1, y1 = p1
    x2, y2 = p2
    ang = math.atan2(y2 - y1, x2 - x1)
    L = 14
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    h1 = (x2 + L * math.cos(a1), y2 + L * math.sin(a1))
    h2 = (x2 + L * math.cos(a2), y2 + L * math.sin(a2))
    draw.polygon([p2, h1, h2], fill=color)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # DejaVuSans costuma existir em Linux/Streamlit Cloud
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    # returns (w,h)
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return int(x1 - x0), int(y1 - y0)
    except Exception:
        # fallback rough
        return (8 * len(text), 14)


def place_label_smart(
    draw: ImageDraw.ImageDraw,
    img_w: int,
    img_h: int,
    tip: Tuple[float, float],
    text: str,
    font,
    pad: int = 4,
    safe_margin: int = 8,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Choose a label position around tip that:
      - stays inside bounds
      - avoids being glued to the right edge (where axis labels live)
    Returns: (text_xy), (rect_xyxy)
    """
    tx, ty = int(tip[0]), int(tip[1])
    tw, th = text_bbox(draw, text, font)

    # candidate offsets (dx,dy) relative to tip
    candidates = [
        (10, -th - 10),          # right-up
        (-tw - 10, -th - 10),    # left-up
        (10, 10),                # right-down
        (-tw - 10, 10),          # left-down
        (-tw - 10, -th // 2),    # left-mid
        (10, -th // 2),          # right-mid
        (-tw // 2, -th - 12),    # above
        (-tw // 2, 12),          # below
    ]

    def ok(x: int, y: int) -> bool:
        # label rect with padding
        rx0 = x - pad
        ry0 = y - pad
        rx1 = x + tw + pad
        ry1 = y + th + pad
        if rx0 < safe_margin or ry0 < safe_margin:
            return False
        if rx1 > img_w - safe_margin or ry1 > img_h - safe_margin:
            return False
        # extra: keep away from extreme right edge (axis labels)
        if rx1 > img_w - 30:
            return False
        return True

    for dx, dy in candidates:
        x = tx + dx
        y = ty + dy
        if ok(x, y):
            rect = (x - pad, y - pad, x + tw + pad, y + th + pad)
            return (x, y), rect

    # fallback: clamp inside
    x = min(max(tx - tw - 10, safe_margin), img_w - tw - safe_margin - 30)
    y = min(max(ty - th - 10, safe_margin), img_h - th - safe_margin)
    rect = (x - pad, y - pad, x + tw + pad, y + th + pad)
    return (x, y), rect


def draw_label_smart(
    draw: ImageDraw.ImageDraw,
    img_w: int,
    img_h: int,
    tip: Tuple[float, float],
    text: str,
    font,
    color_text: Tuple[int, int, int] = (255, 140, 0),
):
    (x, y), rect = place_label_smart(draw, img_w, img_h, tip, text, font)
    # white background box
    draw.rectangle(rect, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((x, y), text, fill=color_text, font=font)


# =========================
# SOLVERS
# =========================
def solve_ground_roll(
    cap: Dict[str, Any],
    mode: str,
    oat_c: float,
    pa_ft: float,
    weight_lb: float,
    wind_kt: float
) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]], Dict[str, Any]]:
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    panels = normalize_panels(cap)

    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_wt_a, ax_wt_b = fit_axis_value_from_ticks(ticks["weight_x100_lb"], "x")
    ax_wind_a, ax_wind_b = fit_axis_value_from_ticks(ticks["wind_kt"], "x")

    out_axis_key = "ground_roll_ft" if mode == "landing" else "takeoff_gr_ft"
    ax_out_a, ax_out_b = fit_axis_value_from_ticks(ticks[out_axis_key], "y")

    if not lines.get("weight_ref_line"):
        raise ValueError("Missing lines['weight_ref_line']")
    if not lines.get("wind_ref_zero"):
        raise ValueError("Missing lines['wind_ref_zero']")

    x_ref_mid = x_of_vertical_ref(lines["weight_ref_line"][0])
    x_ref_right = x_of_vertical_ref(lines["wind_ref_zero"][0])

    x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)

    pa_levels = parse_pa_levels_ft(lines)
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels)
    seg_lo = lines[k_lo][0]
    seg_hi = lines[k_hi][0]
    y_entry = (1 - alpha) * line_y_at_x(seg_lo, x_oat) + alpha * line_y_at_x(seg_hi, x_oat)

    x_wt = axis_coord_from_value(ax_wt_a, ax_wt_b, weight_lb / 100.0)

    g_mid, g_right = pick_guides(cap, mode=mode)
    y_mid, dbg_mid = interp_guides_y(g_mid, x_ref=x_ref_mid, y_ref=y_entry, x_target=x_wt)

    x_wind = axis_coord_from_value(ax_wind_a, ax_wind_b, wind_kt)
    y_out, dbg_right = interp_guides_y(g_right, x_ref=x_ref_right, y_ref=y_mid, x_target=x_wind)

    out_val = axis_value(ax_out_a, ax_out_b, y_out)

    segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    left_panel = panels.get("left") or []
    if not left_panel:
        raise ValueError("Missing panel_corners['left']")
    y_bottom_left = float(left_panel[2]["y"])

    segs.append(((x_oat, y_bottom_left), (x_oat, y_entry)))
    segs.append(((x_oat, y_entry), (x_ref_mid, y_entry)))
    segs.append(((x_ref_mid, y_entry), (x_wt, y_mid)))
    segs.append(((x_wt, y_mid), (x_ref_right, y_mid)))
    segs.append(((x_ref_right, y_mid), (x_wind, y_out)))

    right_panel = panels.get("right") or []
    if not right_panel:
        raise ValueError("Missing panel_corners['right']")
    x_right_edge = float(right_panel[1]["x"])
    segs.append(((x_wind, y_out), (x_right_edge, y_out)))

    debug = {
        "x_oat": x_oat,
        "y_entry": y_entry,
        "x_ref_mid": x_ref_mid,
        "x_wt": x_wt,
        "y_mid": y_mid,
        "x_ref_right": x_ref_right,
        "x_wind": x_wind,
        "y_out": y_out,
        "pa_interp": {"lo": (lo_ft, k_lo), "hi": (hi_ft, k_hi), "alpha": alpha},
        "guide_mid": dbg_mid,
        "guide_right": dbg_right,
    }
    return out_val, segs, debug


def solve_climb(
    cap: Dict[str, Any],
    oat_c: float,
    pa_ft: float
) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]], Dict[str, Any]]:
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    panels = normalize_panels(cap)

    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_roc_a, ax_roc_b = fit_axis_value_from_ticks(ticks["roc_fpm"], "y")

    x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)

    pa_levels = parse_pa_levels_ft(lines)
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels)
    seg_lo = lines[k_lo][0]
    seg_hi = lines[k_hi][0]
    y = (1 - alpha) * line_y_at_x(seg_lo, x_oat) + alpha * line_y_at_x(seg_hi, x_oat)

    roc = axis_value(ax_roc_a, ax_roc_b, y)

    main = panels.get("main") or []
    if not main:
        raise ValueError("Missing panel_corners['main']")
    y_bottom = float(main[2]["y"])
    x_right_edge = float(main[1]["x"])

    segs = [
        ((x_oat, y_bottom), (x_oat, y)),
        ((x_oat, y), (x_right_edge, y)),
    ]

    debug = {
        "x_oat": x_oat,
        "y": y,
        "x_right_edge": x_right_edge,
        "pa_interp": {"lo": (lo_ft, k_lo), "hi": (hi_ft, k_hi), "alpha": alpha},
    }
    return roc, segs, debug


# =========================
# Overlay drawing
# =========================
def draw_overlay(
    base: Image.Image,
    cap: Dict[str, Any],
    show_overlay: bool,
    show_path: bool,
    path_segs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    label_text: Optional[str],
    label_tip: Optional[Tuple[float, float]],
    label_font_size: int,
) -> Image.Image:
    img = base.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size

    panels = normalize_panels(cap)

    if show_overlay:
        for _, seglist in cap.get("lines", {}).items():
            for s in seglist:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

        for _, seglist in cap.get("guides", {}).items():
            for s in seglist:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

        for _, tlist in cap.get("axis_ticks", {}).items():
            for t in tlist:
                x, y = float(t["x"]), float(t["y"])
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 0), width=3)

        for panel, pts in panels.items():
            if len(pts) == 4:
                poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
                d.line(poly + [poly[0]], fill=(0, 140, 255), width=3)
                d.text((poly[0][0] + 6, poly[0][1] + 6), str(panel), fill=(0, 140, 255))

    if show_path:
        for p1, p2 in path_segs:
            draw_arrow(d, p1, p2, color=(255, 140, 0), w=5)

    if label_text and label_tip:
        font = load_font(label_font_size)
        draw_label_smart(d, W, H, label_tip, label_text, font)

    return img


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PA28 Solvers", layout="wide")
st.title("PA28 — Solvers (Landing / Takeoff / Climb)")

mode = st.sidebar.selectbox(
    "Escolhe o gráfico",
    options=["landing", "takeoff", "climb"],
    format_func=lambda k: ASSETS[k]["title"],
)
info = ASSETS[mode]

st.sidebar.markdown("---")
st.sidebar.subheader("Background")
upload_bg = st.sidebar.file_uploader("Upload background (se não estiver na pasta)", type=["pdf", "png", "jpg", "jpeg"])
zoom = st.sidebar.number_input("Zoom PDF", value=2.3, step=0.1)
page_index = st.sidebar.number_input("Página (0-index) [PDF]", value=int(info["page_default"]), step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("JSON")
upload_json = st.sidebar.file_uploader("Upload JSON (se não estiver na pasta)", type=["json"])

cap = load_json(mode, upload_json)
bg = load_background(mode, upload_bg, page_index=int(page_index), zoom=float(zoom))

left, right = st.columns([1.7, 1])

with right:
    st.subheader("Inputs")

    segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    dbg: Dict[str, Any] = {}
    raw = 0.0

    label_text: Optional[str] = None
    label_tip: Optional[Tuple[float, float]] = None

    if mode in ("landing", "takeoff"):
        oat = st.number_input("OAT (°C)", value=21.0 if mode == "landing" else 23.0, step=1.0)
        pa = st.number_input("Pressure Altitude (ft)", value=2500.0 if mode == "landing" else 2000.0, step=500.0)
        wt = st.number_input("Weight (lb)", value=2240.0 if mode == "landing" else 2400.0, step=50.0)
        wind = st.number_input("Headwind (kt)", value=5.0 if mode == "landing" else 8.0, step=1.0)

        raw, segs, dbg = solve_ground_roll(
            cap, mode=mode, oat_c=float(oat), pa_ft=float(pa), weight_lb=float(wt), wind_kt=float(wind)
        )
        out = round_to_step(raw, info["round_to"])
        name = "Ground roll (ft)" if mode == "landing" else "Takeoff GR (ft)"
        st.metric(name, f"{out:.0f}", help=f"Raw={raw:.1f} — arredondado de {info['round_to']} em {info['round_to']}")

        if segs:
            _, tip = segs[-1]
            label_text = f"{out:.0f} ft"
            label_tip = tip

    else:
        oat = st.number_input("OAT (°C)", value=19.0, step=1.0)
        pa = st.number_input("Pressure Altitude (ft)", value=4000.0, step=500.0)

        raw, segs, dbg = solve_climb(cap, oat_c=float(oat), pa_ft=float(pa))
        out = round_to_step(raw, info["round_to"])
        st.metric("Rate of climb (FPM)", f"{out:.0f}", help=f"Raw={raw:.1f} — arredondado de {info['round_to']} em {info['round_to']}")

        if segs:
            _, tip = segs[-1]
            label_text = f"{out:.0f} fpm"
            label_tip = tip

    st.markdown("---")
    show_overlay = st.checkbox("Mostrar overlay das linhas capturadas", value=True)
    show_path = st.checkbox("Mostrar caminho do solver (setas)", value=True)

    with st.expander("Debug"):
        st.json(dbg)

with left:
    img = draw_overlay(
        bg,
        cap,
        show_overlay=show_overlay,
        show_path=show_path,
        path_segs=segs,
        label_text=label_text,
        label_tip=label_tip,
        label_font_size=int(info["label_font"]),
    )
    st.image(img, use_container_width=True)
    st.caption("Vermelho: linhas. Azul: guides. Verde: ticks. Ciano: painéis. Laranja: caminho do solver + valor.")



