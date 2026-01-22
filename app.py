from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
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
        "round_to": 5,  # ft
    },
    "takeoff": {
        "title": "Takeoff Ground Roll",
        "bg_default": "to_ground_roll.jpg",
        "json_default": "to_ground_roll.json",
        "bg_kind": "image",
        "page_default": 0,
        "round_to": 5,  # ft
    },
    "climb": {
        "title": "Climb Performance",
        "bg_default": "climb_perf.jpg",
        "json_default": "climb_perf.json",
        "bg_kind": "image",
        "page_default": 0,
        "round_to": 10,  # fpm
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

    # image
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
    """Accepts {"x":..,"y":..} or [x,y]."""
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    if isinstance(p, (list, tuple)) and len(p) == 2:
        return float(p[0]), float(p[1])
    raise ValueError(f"Invalid point: {p}")


def normalize_panel(panel_pts: Any) -> List[Dict[str, float]]:
    """Return list of 4 dict points [{'x':..,'y':..},...]"""
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
    """Fit: value ≈ a*coord + b  (coord is 'x' or 'y')"""
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
    for k in lines.keys():
        if not k.startswith("pa_"):
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
    """Returns (low, high, alpha) clamped."""
    if not levels:
        raise ValueError("No levels provided.")
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
    """For a vertical ref line segment, return mean x."""
    return 0.5 * (float(seg["x1"]) + float(seg["x2"]))


# =========================
# Guide interpolation (ROBUST)
# =========================
def y_on_guide(seg: Dict[str, float], x: float) -> float:
    return line_y_at_x(seg, x)


def interp_guides_y(
    guides: List[Dict[str, float]],
    x_ref: float,
    y_ref: float,
    x_target: float
) -> Tuple[float, Dict[str, Any]]:
    """
    Interpolate between the two guides that bracket y_ref (evaluated at x_ref),
    then return y_target on the interpolated guide (evaluated at x_target).

    Robust approach:
      - For each guide g, compute y_g_ref = y_g(x_ref) and y_g_tgt = y_g(x_target)
      - Find two guides with y_g_ref surrounding y_ref
      - Interpolate y_tgt between those two y_g_tgt using alpha based on y_ref position
    """
    if not guides:
        return y_ref, {"used": "none"}

    # compute pairs
    rows = []
    for g in guides:
        yr = y_on_guide(g, x_ref)
        yt = y_on_guide(g, x_target)
        rows.append((yr, yt, g))

    # sort by y at ref (in image coords: smaller y = higher)
    rows.sort(key=lambda t: t[0])

    # clamp outside range
    if y_ref <= rows[0][0]:
        return rows[0][1], {"used": "clamp_low", "i": 0}
    if y_ref >= rows[-1][0]:
        return rows[-1][1], {"used": "clamp_high", "i": len(rows) - 1}

    # find bracket
    for i in range(len(rows) - 1):
        y0_ref, y0_tgt, _ = rows[i]
        y1_ref, y1_tgt, _ = rows[i + 1]
        if y0_ref <= y_ref <= y1_ref:
            denom = (y1_ref - y0_ref)
            a = 0.0 if abs(denom) < 1e-12 else (y_ref - y0_ref) / denom
            y_tgt = (1 - a) * y0_tgt + a * y1_tgt
            return float(y_tgt), {"used": "interp", "i0": i, "i1": i + 1, "alpha": float(a)}

    # fallback (shouldn't happen)
    return y_ref, {"used": "fallback"}


# =========================
# Drawing arrows
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


def draw_point(draw: ImageDraw.ImageDraw, p: Tuple[float, float], color: Tuple[int, int, int], r: int = 6, w: int = 3):
    x, y = p
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=w)


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
    """
    Corrected approach (Landing & Takeoff):
      - Use REF LINES (weight_ref_line and wind_ref_zero)
      - Use guide interpolation between two neighbor guides (not just nearest slope)
      - Steps:
        1) x_oat from axis ticks
        2) y_entry from PA line interpolation at x_oat (in left panel)
        3) Transfer horizontally to x_ref_mid (weight_ref_line)
        4) From x_ref_mid to x_weight, move along interpolated WEIGHT guide family -> y_mid
        5) Transfer horizontally to x_ref_right (wind_ref_zero)
        6) From x_ref_right to x_wind, move along interpolated WIND guide family -> y_out
        7) Read output on right axis
    """
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    guides = cap.get("guides", {})
    panels = normalize_panels(cap)

    # axis fits
    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_wt_a, ax_wt_b = fit_axis_value_from_ticks(ticks["weight_x100_lb"], "x")
    ax_wind_a, ax_wind_b = fit_axis_value_from_ticks(ticks["wind_kt"], "x")

    out_axis_key = "ground_roll_ft" if mode == "landing" else "takeoff_gr_ft"
    ax_out_a, ax_out_b = fit_axis_value_from_ticks(ticks[out_axis_key], "y")

    # REF lines (CRITICAL FIX)
    if "weight_ref_line" not in lines or not lines["weight_ref_line"]:
        raise ValueError("Missing lines['weight_ref_line'] in JSON.")
    if "wind_ref_zero" not in lines or not lines["wind_ref_zero"]:
        raise ValueError("Missing lines['wind_ref_zero'] in JSON.")

    x_ref_mid = x_of_vertical_ref(lines["weight_ref_line"][0])
    x_ref_right = x_of_vertical_ref(lines["wind_ref_zero"][0])

    # 1) OAT -> x
    x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)

    # 2) PA line interpolation at x_oat
    pa_levels = parse_pa_levels_ft(lines)
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels)

    seg_lo = lines[k_lo][0]
    seg_hi = lines[k_hi][0]
    y_lo = line_y_at_x(seg_lo, x_oat)
    y_hi = line_y_at_x(seg_hi, x_oat)
    y_entry = (1 - alpha) * y_lo + alpha * y_hi

    # 3) Weight target x
    weight_x100 = weight_lb / 100.0
    x_wt = axis_coord_from_value(ax_wt_a, ax_wt_b, weight_x100)

    # 4) Use guides (interpolate between TWO guides)
    if mode == "landing":
        g_mid = guides.get("middle", [])
        g_right = guides.get("right", [])
    else:
        g_mid = guides.get("guides_weight", [])
        g_right = guides.get("guides_wind", [])

    y_mid, dbg_mid = interp_guides_y(
        guides=g_mid,
        x_ref=x_ref_mid,
        y_ref=y_entry,
        x_target=x_wt
    )

    # 5) Wind target x
    x_wind = axis_coord_from_value(ax_wind_a, ax_wind_b, wind_kt)

    y_out, dbg_right = interp_guides_y(
        guides=g_right,
        x_ref=x_ref_right,
        y_ref=y_mid,
        x_target=x_wind
    )

    out_val = axis_value(ax_out_a, ax_out_b, y_out)

    # Build draw segments (arrows)
    segs = []
    # Start vertical in LEFT panel from bottom to entry
    if "left" not in panels or not panels["left"]:
        raise ValueError("Missing panel_corners['left']")
    left_panel = panels["left"]
    y_bottom_left = float(left_panel[2]["y"])
    segs.append(((x_oat, y_bottom_left), (x_oat, y_entry)))
    # horizontal to weight ref line (not border)
    segs.append(((x_oat, y_entry), (x_ref_mid, y_entry)))
    # weight guide to x_wt
    segs.append(((x_ref_mid, y_entry), (x_wt, y_mid)))
    # horizontal to wind ref line
    segs.append(((x_wt, y_mid), (x_ref_right, y_mid)))
    # wind guide to x_wind
    segs.append(((x_ref_right, y_mid), (x_wind, y_out)))

    # horizontal to right edge (for reading)
    if "right" not in panels or not panels["right"]:
        raise ValueError("Missing panel_corners['right']")
    x_right_edge = float(panels["right"][1]["x"])
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
    """
    Corrected CLIMB:
      - x from OAT
      - y from PA line (interpolated between pa_XXXX families) evaluated at x
      - ROC from y-axis ticks
    """
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
    y_lo = line_y_at_x(seg_lo, x_oat)
    y_hi = line_y_at_x(seg_hi, x_oat)
    y = (1 - alpha) * y_lo + alpha * y_hi

    roc = axis_value(ax_roc_a, ax_roc_b, y)

    # draw segment: vertical marker at x_oat from bottom to y
    if "main" not in panels or not panels["main"]:
        raise ValueError("Missing panel_corners['main']")
    y_bottom = float(panels["main"][2]["y"])
    segs = [((x_oat, y_bottom), (x_oat, y))]

    debug = {
        "x_oat": x_oat,
        "y": y,
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
    key_points: List[Tuple[Tuple[float, float], Tuple[int, int, int]]],
) -> Image.Image:
    img = base.copy()
    d = ImageDraw.Draw(img)

    panels = normalize_panels(cap)

    if show_overlay:
        # lines (red)
        for _, seglist in cap.get("lines", {}).items():
            for s in seglist:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

        # guides (blue)
        for _, seglist in cap.get("guides", {}).items():
            for s in seglist:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

        # ticks (green)
        for _, tlist in cap.get("axis_ticks", {}).items():
            for t in tlist:
                x, y = float(t["x"]), float(t["y"])
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 0), width=3)

        # panels (cyan)
        for panel, pts in panels.items():
            if len(pts) == 4:
                poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
                d.line(poly + [poly[0]], fill=(0, 140, 255), width=3)
                d.text((poly[0][0] + 6, poly[0][1] + 6), str(panel), fill=(0, 140, 255))

    # path arrows (orange)
    if show_path:
        for p1, p2 in path_segs:
            draw_arrow(d, p1, p2, color=(255, 140, 0), w=5)

    # key points (colored circles)
    for p, c in key_points:
        draw_point(d, p, color=c, r=7, w=4)

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
    key_points: List[Tuple[Tuple[float, float], Tuple[int, int, int]]] = []

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

        # key points for visualization
        key_points = [
            ((dbg["x_oat"], dbg["y_entry"]), (255, 0, 255)),          # entry (magenta)
            ((dbg["x_wt"], dbg["y_mid"]), (0, 200, 0)),               # weight point (green)
            ((dbg["x_wind"], dbg["y_out"]), (0, 0, 0)),               # wind point (black)
            ((dbg["x_ref_mid"], dbg["y_entry"]), (120, 0, 255)),      # ref mid (purple)
            ((dbg["x_ref_right"], dbg["y_mid"]), (120, 0, 255)),      # ref right (purple)
        ]

    else:
        oat = st.number_input("OAT (°C)", value=19.0, step=1.0)
        pa = st.number_input("Pressure Altitude (ft)", value=4000.0, step=500.0)

        raw, segs, dbg = solve_climb(cap, oat_c=float(oat), pa_ft=float(pa))
        out = round_to_step(raw, info["round_to"])
        st.metric("Rate of climb (FPM)", f"{out:.0f}", help=f"Raw={raw:.1f} — arredondado de {info['round_to']} em {info['round_to']}")

        key_points = [((dbg["x_oat"], dbg["y"]), (255, 0, 255))]  # magenta at PA curve

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
        key_points=key_points,
    )
    st.image(img, use_container_width=True)
    st.caption("Vermelho: linhas (ISA/PA). Azul: guides. Verde: ticks. Ciano: painéis. Laranja: caminho. Magenta/verde/preto: pontos chave.")
