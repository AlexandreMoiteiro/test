from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import pymupdf  # PyMuPDF


# ----------------------------
# Config
# ----------------------------
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
        "round_to": 10,  # fpm (podes mudar)
    },
}


# ----------------------------
# IO helpers
# ----------------------------
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


# ----------------------------
# Math helpers (axes, lines, intersection)
# ----------------------------
def fit_axis_value_from_ticks(ticks: List[Dict[str, float]], coord: str) -> Tuple[float, float]:
    """
    Fit: value ≈ a*coord + b
    coord is 'x' or 'y'
    """
    xs = np.array([float(t[coord]) for t in ticks], dtype=float)
    vs = np.array([float(t["value"]) for t in ticks], dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, vs, rcond=None)[0]
    return float(a), float(b)


def axis_value(a: float, b: float, coord_val: float) -> float:
    return a * coord_val + b


def axis_coord_from_value(a: float, b: float, value: float) -> float:
    # value = a*coord + b => coord = (value-b)/a
    if abs(a) < 1e-9:
        raise ValueError("Axis fit degenerate (a ~ 0).")
    return (value - b) / a


def line_y_at_x(seg: Dict[str, float], x: float) -> float:
    x1, y1, x2, y2 = map(float, (seg["x1"], seg["y1"], seg["x2"], seg["y2"]))
    if abs(x2 - x1) < 1e-9:
        # vertical line -> return y1 (fallback)
        return y1
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def segment_slope(seg: Dict[str, float]) -> float:
    x1, y1, x2, y2 = map(float, (seg["x1"], seg["y1"], seg["x2"], seg["y2"]))
    if abs(x2 - x1) < 1e-9:
        return 0.0
    return (y2 - y1) / (x2 - x1)


def closest_guide_slope(guides: List[Dict[str, float]], y_ref: float) -> float:
    # escolhe a guide cujo "meio" está mais perto em y
    best = None
    for g in guides:
        ym = (float(g["y1"]) + float(g["y2"])) / 2.0
        d = abs(ym - y_ref)
        if best is None or d < best[0]:
            best = (d, g)
    return segment_slope(best[1]) if best else 0.0


def parse_pa_levels(lines: Dict[str, List[Dict[str, float]]]) -> List[Tuple[float, str]]:
    out = []
    for k in lines.keys():
        if k.startswith("pa_"):
            if k == "pa_sea_level":
                out.append((0.0, k))
            else:
                try:
                    lvl = float(k.replace("pa_", ""))
                    out.append((lvl, k))
                except Exception:
                    pass
    out.sort(key=lambda t: t[0])
    return out


def interp_between_levels(v: float, levels: List[Tuple[float, str]]) -> Tuple[Tuple[float, str], Tuple[float, str], float]:
    """
    Returns (low, high, alpha) where alpha in [0,1]
    If v outside range -> clamps.
    """
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


def line_intersection(seg1: Dict[str, float], seg2: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """
    Infinite-line intersection of two segments (treated as lines).
    """
    x1, y1, x2, y2 = map(float, (seg1["x1"], seg1["y1"], seg1["x2"], seg1["y2"]))
    x3, y3, x4, y4 = map(float, (seg2["x1"], seg2["y1"], seg2["x2"], seg2["y2"]))

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


# ----------------------------
# Drawing arrows
# ----------------------------
def draw_arrow(draw: ImageDraw.ImageDraw, p1: Tuple[float, float], p2: Tuple[float, float], color: Tuple[int, int, int], w: int = 4):
    x1, y1 = p1
    x2, y2 = p2
    draw.line([p1, p2], fill=color, width=w)

    # arrowhead
    ang = math.atan2(y2 - y1, x2 - x1)
    L = 14
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    h1 = (x2 + L * math.cos(a1), y2 + L * math.sin(a1))
    h2 = (x2 + L * math.cos(a2), y2 + L * math.sin(a2))
    draw.polygon([p2, h1, h2], fill=color)


# ----------------------------
# Solver implementations
# ----------------------------
def solve_ground_roll(cap: Dict[str, Any], mode: str, oat_c: float, pa_ft: float, weight_lb: float, wind_kt: float) -> Tuple[float, List[Tuple[Tuple[float,float], Tuple[float,float]]], Dict[str, Any]]:
    """
    Modelo prático (seguindo as 'guides'):
    - LEFT: escolhe PA line (interp por PA) e avalia y na vertical do OAT (x_oat).
    - MIDDLE: aplica slope da guide mais próxima (por y) para ir do x_left_mid até x_weight.
    - RIGHT: aplica slope da guide mais próxima (por y) para ir do x_left_right até x_wind.
    - Converte y_final em ground_roll_ft.
    Retorna: (ground_roll, segments_to_draw, debug)
    """
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    panels = cap["panel_corners"]
    guides = cap.get("guides", {})

    # axis fits
    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_wt_a, ax_wt_b = fit_axis_value_from_ticks(ticks["weight_x100_lb"], "x")
    ax_wind_a, ax_wind_b = fit_axis_value_from_ticks(ticks["wind_kt"], "x")

    # y-axis for result differs by mode
    out_axis_key = "ground_roll_ft" if mode == "landing" else "takeoff_gr_ft"
    ax_out_a, ax_out_b = fit_axis_value_from_ticks(ticks[out_axis_key], "y")

    # 1) OAT -> x
    x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)

    # 2) PA line interpolation at x_oat
    pa_levels = parse_pa_levels(lines)
    (pa_lo, k_lo), (pa_hi, k_hi), alpha = interp_between_levels(pa_ft / 1000.0 if "pa_" in k_lo else pa_ft, pa_levels)
    # (nota: os teus nomes são pa_2000 etc (ft). Para landing json, é pa_2000, pa_4000, etc e sea_level.)
    # Vamos interpretar assim:
    # - se key == pa_sea_level => 0
    # - se key == pa_2000 => 2000
    # Por isso usamos pa_ft diretamente (em ft).
    # Vamos refazer o interp em ft:
    pa_levels_ft = []
    for lvl, k in pa_levels:
        if k == "pa_sea_level":
            pa_levels_ft.append((0.0, k))
        else:
            pa_levels_ft.append((lvl, k))
    pa_levels_ft.sort(key=lambda t: t[0])
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels_ft)

    seg_lo = lines[k_lo][0]
    seg_hi = lines[k_hi][0]
    y_lo = line_y_at_x(seg_lo, x_oat)
    y_hi = line_y_at_x(seg_hi, x_oat)
    y_entry = (1 - alpha) * y_lo + alpha * y_hi

    # 3) MIDDLE: weight -> x
    weight_x100 = weight_lb / 100.0
    x_wt = axis_coord_from_value(ax_wt_a, ax_wt_b, weight_x100)

    # middle panel left edge x
    mid = panels["middle"]
    x_left_mid = float(mid[0]["x"])
    # choose correct guide set name (landing uses guides["middle"], takeoff uses guides["guides_weight"])
    if mode == "landing":
        g_mid = guides.get("middle", [])
    else:
        g_mid = guides.get("guides_weight", [])
    slope_mid = closest_guide_slope(g_mid, y_entry) if g_mid else 0.0
    y_mid = y_entry + slope_mid * (x_wt - x_left_mid)

    # 4) RIGHT: wind -> x
    x_wind = axis_coord_from_value(ax_wind_a, ax_wind_b, wind_kt)
    right = panels["right"]
    x_left_right = float(right[0]["x"])
    if mode == "landing":
        g_right = guides.get("right", [])
    else:
        g_right = guides.get("guides_wind", [])
    slope_right = closest_guide_slope(g_right, y_mid) if g_right else 0.0
    y_out = y_mid + slope_right * (x_wind - x_left_right)

    # 5) y_out -> ground roll
    out_val = axis_value(ax_out_a, ax_out_b, y_out)

    debug = {
        "x_oat": x_oat,
        "y_entry": y_entry,
        "x_wt": x_wt,
        "y_mid": y_mid,
        "x_wind": x_wind,
        "y_out": y_out,
        "pa_interp": {"lo": (lo_ft, k_lo), "hi": (hi_ft, k_hi), "alpha": alpha},
        "slopes": {"middle": slope_mid, "right": slope_right},
        "axis_fit": {
            "oat": (ax_oat_a, ax_oat_b),
            "weight": (ax_wt_a, ax_wt_b),
            "wind": (ax_wind_a, ax_wind_b),
            "out": (ax_out_a, ax_out_b),
        },
    }

    # segments to draw (with arrows)
    segs = []
    # LEFT: vertical from bottom-ish to entry y
    left_panel = panels["left"]
    y_bottom_left = float(left_panel[2]["y"])
    segs.append(((x_oat, y_bottom_left), (x_oat, y_entry)))
    # LEFT -> MIDDLE: horizontal
    segs.append(((x_oat, y_entry), (x_left_mid, y_entry)))
    # MIDDLE: to x_wt
    segs.append(((x_left_mid, y_entry), (x_wt, y_mid)))
    # MIDDLE -> RIGHT: horizontal at y_mid
    segs.append(((x_wt, y_mid), (x_left_right, y_mid)))
    # RIGHT: to x_wind
    segs.append(((x_left_right, y_mid), (x_wind, y_out)))
    # RIGHT: horizontal to axis (right edge) at y_out
    x_right_edge = float(right[1]["x"])
    segs.append(((x_wind, y_out), (x_right_edge, y_out)))

    return out_val, segs, debug


def solve_climb(cap: Dict[str, Any], oat_c: float, pa_ft: float, isa_dev: float) -> Tuple[float, List[Tuple[Tuple[float,float], Tuple[float,float]]], Dict[str, Any]]:
    """
    Para o climb, usamos uma abordagem geométrica:
    - Interpola uma linha ISA (entre -15, 0, +35) como y(x) pela tua captura.
    - Interpola uma linha PA (entre pa_XXXX).
    - Interseção (ISA vs PA) dá um ponto cujo y -> ROC via ticks roc_fpm.
    """
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    panels = cap["panel_corners"]

    # axis fits
    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_roc_a, ax_roc_b = fit_axis_value_from_ticks(ticks["roc_fpm"], "y")

    # build ISA segments
    isa_m15 = lines["isa_m15"][0]
    isa_0 = lines["isa"][0]
    isa_p35 = lines["isa_p35"][0]

    # interpolate ISA in "deviation space"
    # map dev -15 -> 0, dev 0 -> 1, dev +35 -> 2 (piecewise)
    def lerp_seg(a: Dict[str,float], b: Dict[str,float], t: float) -> Dict[str,float]:
        return {
            "x1": (1-t)*a["x1"] + t*b["x1"],
            "y1": (1-t)*a["y1"] + t*b["y1"],
            "x2": (1-t)*a["x2"] + t*b["x2"],
            "y2": (1-t)*a["y2"] + t*b["y2"],
        }

    if isa_dev <= 0:
        t = (isa_dev - (-15.0)) / (0.0 - (-15.0))  # [-15..0]
        t = float(np.clip(t, 0, 1))
        isa_seg = lerp_seg(isa_m15, isa_0, t)
    else:
        t = (isa_dev - 0.0) / (35.0 - 0.0)  # [0..35]
        t = float(np.clip(t, 0, 1))
        isa_seg = lerp_seg(isa_0, isa_p35, t)

    # PA interpolate
    pa_levels = []
    for k in lines.keys():
        if k.startswith("pa_"):
            if k == "pa_sea_level":
                pa_levels.append((0.0, k))
            else:
                try:
                    pa_levels.append((float(k.replace("pa_", "")), k))
                except Exception:
                    pass
    pa_levels.sort(key=lambda t: t[0])
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels)

    pa_lo = lines[k_lo][0]
    pa_hi = lines[k_hi][0]
    pa_seg = {
        "x1": (1-alpha)*pa_lo["x1"] + alpha*pa_hi["x1"],
        "y1": (1-alpha)*pa_lo["y1"] + alpha*pa_hi["y1"],
        "x2": (1-alpha)*pa_lo["x2"] + alpha*pa_hi["x2"],
        "y2": (1-alpha)*pa_lo["y2"] + alpha*pa_hi["y2"],
    }

    ip = line_intersection(isa_seg, pa_seg)
    if ip is None:
        # fallback: lê ROC no x do OAT na PA line
        x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)
        y = line_y_at_x(pa_seg, x_oat)
        roc = axis_value(ax_roc_a, ax_roc_b, y)
        segs = [((x_oat, float(panels["main"][2]["y"])), (x_oat, y))]
        debug = {"fallback": True, "x_oat": x_oat, "y": y}
        return roc, segs, debug

    x_int, y_int = ip
    roc = axis_value(ax_roc_a, ax_roc_b, y_int)

    # segments to draw
    y_bottom = float(panels["main"][2]["y"])
    segs = [
        ((x_int, y_bottom), (x_int, y_int)),  # vertical marker
    ]
    debug = {
        "intersection": {"x": x_int, "y": y_int},
        "pa_interp": {"lo": (lo_ft, k_lo), "hi": (hi_ft, k_hi), "alpha": alpha},
    }
    return roc, segs, debug


# ----------------------------
# UI
# ----------------------------
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

    if mode in ["landing", "takeoff"]:
        oat = st.number_input("OAT (°C)", value=20.0, step=1.0)
        pa = st.number_input("Pressure Altitude (ft)", value=2000.0, step=500.0)
        wt = st.number_input("Weight (lb)", value=2400.0, step=50.0)
        wind = st.number_input("Headwind (kt)", value=0.0, step=1.0)

        raw, segs, dbg = solve_ground_roll(cap, mode=mode, oat_c=float(oat), pa_ft=float(pa), weight_lb=float(wt), wind_kt=float(wind))
        out = round_to_step(raw, info["round_to"])

        out_name = "Ground roll (ft)" if mode == "landing" else "Takeoff GR (ft)"
        st.metric(out_name, f"{out:.0f}", help=f"Raw={raw:.1f} (arredondado de {info['round_to']} em {info['round_to']})")

        with st.expander("Debug"):
            st.json(dbg)

    else:
        oat = st.number_input("OAT (°C)", value=20.0, step=1.0)
        pa = st.number_input("Pressure Altitude (ft)", value=5000.0, step=500.0)
        isa_dev = st.number_input("ISA deviation (°C)", value=0.0, step=1.0)

        raw, segs, dbg = solve_climb(cap, oat_c=float(oat), pa_ft=float(pa), isa_dev=float(isa_dev))
        out = round_to_step(raw, info["round_to"])

        st.metric("Rate of climb (FPM)", f"{out:.0f}", help=f"Raw={raw:.1f} (arredondado de {info['round_to']} em {info['round_to']})")
        with st.expander("Debug"):
            st.json(dbg)

    st.markdown("---")
    show_overlay = st.checkbox("Mostrar overlay das linhas capturadas", value=True)
    show_path = st.checkbox("Mostrar caminho do solver (setas)", value=True)


with left:
    base = bg.copy()
    d = ImageDraw.Draw(base)

    # overlay capturado (linhas + guides + ticks) simples
    if show_overlay:
        # lines (vermelho)
        for _, seglist in cap.get("lines", {}).items():
            for s in seglist:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

        # guides (azul)
        for _, seglist in cap.get("guides", {}).items():
            for s in seglist:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

        # ticks (verde)
        for _, tlist in cap.get("axis_ticks", {}).items():
            for t in tlist:
                x, y = float(t["x"]), float(t["y"])
                d.ellipse((x-4, y-4, x+4, y+4), outline=(0, 160, 0), width=3)

        # panels (ciano)
        for panel, pts in cap.get("panel_corners", {}).items():
            if isinstance(pts, list) and len(pts) == 4 and isinstance(pts[0], dict):
                poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
                d.line(poly + [poly[0]], fill=(0, 140, 255), width=3)
                d.text((poly[0][0] + 6, poly[0][1] + 6), str(panel), fill=(0, 140, 255))

    # solver path (amarelo/laranja)
    if show_path and "segs" in locals():
        for p1, p2 in segs:
            draw_arrow(d, p1, p2, color=(255, 140, 0), w=5)

    st.image(base, use_container_width=True)
    st.caption("Vermelho: linhas (ISA/PA). Azul: guides. Verde: ticks. Ciano: painéis. Laranja: caminho do solver.")

