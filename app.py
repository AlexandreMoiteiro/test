from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import pymupdf  # PyMuPDF


# ============================================================
# Config (fixo para estes 3 modos)
# ============================================================
ASSETS = {
    "landing": {
        "title": "Landing Ground Roll (PA-28-181 Archer III)",
        "bg_default": "ldg_ground_roll.pdf",
        "json_default": "ldg_ground_roll.json",
        "bg_kind": "pdf",
        "page_default": 0,
        "round_to": 5,   # ft
    },
    "takeoff": {
        "title": "Takeoff Ground Roll (PA-28-181 Archer III)",
        "bg_default": "to_ground_roll.jpg",
        "json_default": "to_ground_roll.json",
        "bg_kind": "image",
        "page_default": 0,
        "round_to": 5,   # ft
    },
    "climb": {
        "title": "Climb Performance (PA-28-181 Archer III)",
        "bg_default": "climb_perf.jpg",
        "json_default": "climb_perf.json",
        "bg_kind": "image",
        "page_default": 0,
        "round_to": 10,  # fpm (podes ajustar no UI)
    },
}

Point = Tuple[float, float]
Seg = Tuple[Point, Point]


# ============================================================
# IO helpers
# ============================================================
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
def load_bytes(path: Path) -> bytes:
    return path.read_bytes()


@st.cache_data(show_spinner=False)
def render_pdf_to_image(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def _as_xy(pt: Union[Dict[str, Any], List[Any], Tuple[Any, Any]]) -> Dict[str, float]:
    """Aceita {'x','y'} ou [x,y]/(x,y) e devolve dict com floats."""
    if isinstance(pt, dict):
        return {"x": float(pt["x"]), "y": float(pt["y"])}
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        return {"x": float(pt[0]), "y": float(pt[1])}
    raise TypeError(f"Ponto inválido: {pt!r}")


def normalize_capture(cap: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza estruturas do JSON para o solver ser robusto."""
    cap = dict(cap)  # shallow copy

    # panel_corners: {name: [pt, pt, pt, pt]} com pt dict(x,y)
    panels = {}
    for k, v in cap.get("panel_corners", {}).items():
        if isinstance(v, list) and len(v) == 4:
            panels[k] = [_as_xy(p) for p in v]
        else:
            # fallback: tenta converter listas/tuplos nested
            if isinstance(v, (list, tuple)):
                panels[k] = [_as_xy(p) for p in list(v)]
            else:
                raise TypeError(f"panel_corners[{k}] inválido: {type(v)}")
    cap["panel_corners"] = panels

    # axis_ticks
    ticks2 = {}
    for ax, tlist in cap.get("axis_ticks", {}).items():
        out = []
        for t in tlist:
            out.append({"value": float(t["value"]), "x": float(t["x"]), "y": float(t["y"])})
        ticks2[ax] = out
    cap["axis_ticks"] = ticks2

    # lines (segmentos)
    lines2 = {}
    for lk, segs in cap.get("lines", {}).items():
        out = []
        for s in segs:
            out.append({"x1": float(s["x1"]), "y1": float(s["y1"]), "x2": float(s["x2"]), "y2": float(s["y2"])})
        lines2[lk] = out
    cap["lines"] = lines2

    # guides (dict de listas)
    guides2 = {}
    g = cap.get("guides", {})
    if isinstance(g, dict):
        for gk, segs in g.items():
            out = []
            if isinstance(segs, list):
                for s in segs:
                    out.append({"x1": float(s["x1"]), "y1": float(s["y1"]), "x2": float(s["x2"]), "y2": float(s["y2"])})
            guides2[gk] = out
    cap["guides"] = guides2

    # zoom/page_index
    if "zoom" in cap:
        cap["zoom"] = float(cap["zoom"])
    if "page_index" in cap:
        cap["page_index"] = int(cap["page_index"])
    return cap


def load_json(mode: str, upload_json) -> Dict[str, Any]:
    info = ASSETS[mode]
    if upload_json is not None:
        cap = json.loads(upload_json.read().decode("utf-8"))
        return normalize_capture(cap)
    p = _here(info["json_default"])
    if not p:
        raise FileNotFoundError(f"Não encontrei {info['json_default']}. Coloca o ficheiro na mesma pasta do app.py.")
    cap = json.loads(p.read_text(encoding="utf-8"))
    return normalize_capture(cap)


# ============================================================
# Math helpers (axes, lines)
# ============================================================
def fit_axis_value_from_ticks(ticks: List[Dict[str, float]], coord: str) -> Tuple[float, float]:
    """Fit: value ≈ a*coord + b  (coord is 'x' or 'y')."""
    xs = np.array([float(t[coord]) for t in ticks], dtype=float)
    ys = np.array([float(t["value"]) for t in ticks], dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(a), float(b)


def axis_value(a: float, b: float, coord: float) -> float:
    return a * float(coord) + b


def axis_coord_from_value(a: float, b: float, value: float) -> float:
    if abs(a) < 1e-12:
        raise ValueError("Axis fit com 'a'≈0; não dá para inverter.")
    return (float(value) - b) / a


def segment_slope(seg: Dict[str, float]) -> float:
    dx = float(seg["x2"]) - float(seg["x1"])
    if abs(dx) < 1e-9:
        return 0.0
    return (float(seg["y2"]) - float(seg["y1"])) / dx


def line_y_at_x(seg: Dict[str, float], x: float) -> float:
    x1, y1, x2, y2 = map(float, (seg["x1"], seg["y1"], seg["x2"], seg["y2"]))
    if abs(x2 - x1) < 1e-9:
        return (y1 + y2) / 2.0
    t = (float(x) - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def parse_pa_levels_ft(lines: Dict[str, List[Dict[str, float]]]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for k in lines.keys():
        if k.startswith("pa_"):
            tail = k.replace("pa_", "")
            if tail == "sea_level":
                out.append((0, k))
            else:
                try:
                    out.append((int(tail), k))
                except ValueError:
                    pass
    out.sort(key=lambda x: x[0])
    return out


def interp_between_levels(pa_ft: float, pa_levels_ft: List[Tuple[int, str]]):
    pa_ft = float(pa_ft)
    if not pa_levels_ft:
        raise ValueError("Não encontrei linhas PA no JSON.")
    if pa_ft <= pa_levels_ft[0][0]:
        (lo_ft, k_lo) = pa_levels_ft[0]
        return (lo_ft, k_lo), (lo_ft, k_lo), 0.0
    if pa_ft >= pa_levels_ft[-1][0]:
        (hi_ft, k_hi) = pa_levels_ft[-1]
        return (hi_ft, k_hi), (hi_ft, k_hi), 0.0
    for i in range(len(pa_levels_ft) - 1):
        lo_ft, k_lo = pa_levels_ft[i]
        hi_ft, k_hi = pa_levels_ft[i + 1]
        if lo_ft <= pa_ft <= hi_ft:
            alpha = 0.0 if hi_ft == lo_ft else (pa_ft - lo_ft) / (hi_ft - lo_ft)
            return (lo_ft, k_lo), (hi_ft, k_hi), float(alpha)
    (lo_ft, k_lo) = pa_levels_ft[0]
    return (lo_ft, k_lo), (lo_ft, k_lo), 0.0


def round_to_step(x: float, step: float) -> float:
    step = float(step)
    if step <= 0:
        return float(x)
    return round(float(x) / step) * step


# ============================================================
# Guides: interpolação "entre duas réguas"
# ============================================================
def guide_interpolated_y(
    guides: List[Dict[str, float]],
    x_ref: float,
    y_ref: float,
    x_target: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Em vez de escolher a guide mais próxima, interpola ENTRE as duas guides que,
    no x_ref (borda esquerda do painel), enquadram o y_ref.
    Retorna y_target e debug (guides usadas, t, slope equivalente).
    """
    if not guides or abs(x_target - x_ref) < 1e-9:
        return float(y_ref), {"method": "none_or_dx0"}

    rows = []
    for idx, g in enumerate(guides):
        y0 = line_y_at_x(g, x_ref)
        y1 = line_y_at_x(g, x_target)
        rows.append((y0, y1, idx, g))
    rows.sort(key=lambda r: r[0])  # ordenar por y em x_ref

    y0s = [r[0] for r in rows]

    if y_ref <= y0s[0]:
        y_target = rows[0][1]
        chosen = [rows[0]]
        t = 0.0
    elif y_ref >= y0s[-1]:
        y_target = rows[-1][1]
        chosen = [rows[-1]]
        t = 0.0
    else:
        i = max(0, int(np.searchsorted(y0s, y_ref) - 1))
        lo = rows[i]
        hi = rows[i + 1]
        denom = (hi[0] - lo[0])
        t = 0.0 if abs(denom) < 1e-9 else (y_ref - lo[0]) / denom
        t = float(np.clip(t, 0.0, 1.0))
        y_target = (1 - t) * lo[1] + t * hi[1]
        chosen = [lo, hi]

    slope = (float(y_target) - float(y_ref)) / (float(x_target) - float(x_ref))
    dbg = {
        "method": "interp",
        "x_ref": float(x_ref),
        "x_target": float(x_target),
        "y_ref": float(y_ref),
        "y_target": float(y_target),
        "t": float(t),
        "slope_equiv": float(slope),
        "guides_used": [
            {
                "idx": int(r[2]),
                "y_at_ref": float(r[0]),
                "y_at_target": float(r[1]),
                "slope": float(segment_slope(r[3])),
            }
            for r in chosen
        ],
    }
    return float(y_target), dbg


# ============================================================
# Solvers
# ============================================================
def solve_ground_roll(
    cap: Dict[str, Any],
    mode: str,
    oat_c: float,
    pa_ft: float,
    weight_lb: float,
    wind_kt: float,
) -> Tuple[float, List[Seg], Dict[str, Any], Dict[str, Point]]:
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    panels = cap["panel_corners"]
    guides = cap.get("guides", {})

    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_wt_a, ax_wt_b = fit_axis_value_from_ticks(ticks["weight_x100_lb"], "x")
    ax_wind_a, ax_wind_b = fit_axis_value_from_ticks(ticks["wind_kt"], "x")

    out_axis_key = "ground_roll_ft" if mode == "landing" else "takeoff_gr_ft"
    ax_out_a, ax_out_b = fit_axis_value_from_ticks(ticks[out_axis_key], "y")

    # 1) OAT -> x
    x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)

    # 2) PA interp em x_oat (pixel y)
    pa_levels_ft = parse_pa_levels_ft(lines)
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels_ft)

    seg_lo = lines[k_lo][0]
    seg_hi = lines[k_hi][0]
    y_lo = line_y_at_x(seg_lo, x_oat)
    y_hi = line_y_at_x(seg_hi, x_oat)
    y_entry = (1 - alpha) * y_lo + alpha * y_hi

    # 3) Borda do painel middle
    mid = panels["middle"]
    x_left_mid = float(mid[0]["x"])

    # 4) Painel middle: guides (interp) até x_wt
    weight_x100 = weight_lb / 100.0
    x_wt = axis_coord_from_value(ax_wt_a, ax_wt_b, weight_x100)

    g_mid = guides.get("middle", []) if mode == "landing" else guides.get("guides_weight", [])
    y_mid, dbg_mid = guide_interpolated_y(g_mid, x_left_mid, y_entry, x_wt)

    # 5) Borda do painel right
    right = panels["right"]
    x_left_right = float(right[0]["x"])

    # 6) Painel right: guides até x_wind
    x_wind = axis_coord_from_value(ax_wind_a, ax_wind_b, wind_kt)
    g_right = guides.get("right", []) if mode == "landing" else guides.get("guides_wind", [])
    y_out, dbg_right = guide_interpolated_y(g_right, x_left_right, y_mid, x_wind)

    # 7) y_out -> ground roll
    out_val = axis_value(ax_out_a, ax_out_b, y_out)

    # pontos chave
    pts: Dict[str, Point] = {}
    left_panel = panels["left"]
    y_bottom_left = float(left_panel[2]["y"])
    x_right_edge = float(right[1]["x"])
    pts["oat_base"] = (x_oat, y_bottom_left)
    pts["entry"] = (x_oat, y_entry)
    pts["mid_in"] = (x_left_mid, y_entry)
    pts["weight"] = (x_wt, y_mid)
    pts["right_in"] = (x_left_right, y_mid)
    pts["wind"] = (x_wind, y_out)
    pts["readout"] = (x_right_edge, y_out)

    segs: List[Seg] = [
        (pts["oat_base"], pts["entry"]),
        (pts["entry"], pts["mid_in"]),
        (pts["mid_in"], pts["weight"]),
        (pts["weight"], pts["right_in"]),
        (pts["right_in"], pts["wind"]),
        (pts["wind"], pts["readout"]),
    ]

    debug = {
        "pa_interp": {"lo": (lo_ft, k_lo), "hi": (hi_ft, k_hi), "alpha": alpha},
        "middle": dbg_mid,
        "right": dbg_right,
        "axes": {
            "oat": {"a": ax_oat_a, "b": ax_oat_b},
            "weight_x100": {"a": ax_wt_a, "b": ax_wt_b},
            "wind": {"a": ax_wind_a, "b": ax_wind_b},
            "out_y": {"a": ax_out_a, "b": ax_out_b},
        },
        "pixels": {
            "x_oat": x_oat,
            "y_entry": y_entry,
            "x_wt": x_wt,
            "y_mid": y_mid,
            "x_wind": x_wind,
            "y_out": y_out,
        },
    }

    return out_val, segs, debug, pts


def solve_climb(
    cap: Dict[str, Any],
    oat_c: float,
    pa_ft: float,
    isa_dev: float,
) -> Tuple[float, List[Seg], Dict[str, Any], Dict[str, Point]]:
    ticks = cap["axis_ticks"]
    lines = cap["lines"]
    panels = cap["panel_corners"]

    ax_oat_a, ax_oat_b = fit_axis_value_from_ticks(ticks["oat_c"], "x")
    ax_roc_a, ax_roc_b = fit_axis_value_from_ticks(ticks["roc_fpm"], "y")

    isa_m15 = lines["isa_m15"][0]
    isa_0 = lines["isa"][0]
    isa_p35 = lines["isa_p35"][0]

    def lerp_seg(a: Dict[str, float], b: Dict[str, float], t: float) -> Dict[str, float]:
        return {
            "x1": (1 - t) * a["x1"] + t * b["x1"],
            "y1": (1 - t) * a["y1"] + t * b["y1"],
            "x2": (1 - t) * a["x2"] + t * b["x2"],
            "y2": (1 - t) * a["y2"] + t * b["y2"],
        }

    if isa_dev <= 0:
        t = (isa_dev - (-15.0)) / 15.0
        t = float(np.clip(t, 0.0, 1.0))
        isa_seg = lerp_seg(isa_m15, isa_0, t)
        isa_bracket = {"lo": -15.0, "hi": 0.0, "t": t}
    else:
        t = isa_dev / 35.0
        t = float(np.clip(t, 0.0, 1.0))
        isa_seg = lerp_seg(isa_0, isa_p35, t)
        isa_bracket = {"lo": 0.0, "hi": 35.0, "t": t}

    pa_levels_ft = parse_pa_levels_ft(lines)
    (lo_ft, k_lo), (hi_ft, k_hi), alpha = interp_between_levels(pa_ft, pa_levels_ft)

    # Resolver interseção ISA x PA por bisseção no x dentro do painel
    main = panels["main"]
    x_min = float(min(p["x"] for p in main))
    x_max = float(max(p["x"] for p in main))

    def ydiff(x: float) -> float:
        y_i = line_y_at_x(isa_seg, x)
        y_l = line_y_at_x(lines[k_lo][0], x)
        y_h = line_y_at_x(lines[k_hi][0], x)
        y_p = (1 - alpha) * y_l + alpha * y_h
        return y_i - y_p

    # bom chute inicial: x correspondente ao OAT
    x_oat = axis_coord_from_value(ax_oat_a, ax_oat_b, oat_c)

    a, b = x_min, x_max
    fa, fb = ydiff(a), ydiff(b)
    x_star = x_oat
    if fa == 0:
        x_star = a
    elif fb == 0:
        x_star = b
    elif fa * fb < 0:
        for _ in range(40):
            m = 0.5 * (a + b)
            fm = ydiff(m)
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        x_star = 0.5 * (a + b)

    y_star = line_y_at_x(isa_seg, x_star)
    roc_fpm = axis_value(ax_roc_a, ax_roc_b, y_star)

    pts = {
        "oat": (x_oat, line_y_at_x(isa_seg, x_oat)),
        "intersection": (x_star, y_star),
    }

    # caminho para desenhar (simples e legível)
    segs = [
        ((x_oat, float(main[2]["y"])), (x_oat, pts["oat"][1])),
        (pts["oat"], pts["intersection"]),
        (pts["intersection"], (float(main[1]["x"]), y_star)),
    ]

    debug = {
        "isa": {"isa_dev": isa_dev, "bracket": isa_bracket},
        "pa_interp": {"lo": (lo_ft, k_lo), "hi": (hi_ft, k_hi), "alpha": alpha},
        "x_oat": x_oat,
        "x_star": x_star,
        "roc_fit": {"a": ax_roc_a, "b": ax_roc_b},
    }

    return roc_fpm, segs, debug, pts


# ============================================================
# Drawing helpers
# ============================================================
def draw_arrow(draw: ImageDraw.ImageDraw, p1: Point, p2: Point, color=(255, 140, 0), w: int = 5):
    x1, y1 = p1
    x2, y2 = p2
    draw.line([p1, p2], fill=color, width=w)

    ang = math.atan2(y2 - y1, x2 - x1)
    L = max(10, int(2.6 * w))
    a1 = ang + math.radians(155)
    a2 = ang - math.radians(155)
    h1 = (x2 + L * math.cos(a1), y2 + L * math.sin(a1))
    h2 = (x2 + L * math.cos(a2), y2 + L * math.sin(a2))
    draw.polygon([p2, h1, h2], fill=color)


def draw_point(draw: ImageDraw.ImageDraw, p: Point, r: int = 6, fill=(255, 200, 0), outline=(0, 0, 0)):
    x, y = p
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2)


def draw_label(draw: ImageDraw.ImageDraw, p: Point, text: str, fill=(0, 0, 0)):
    x, y = p
    draw.text((x + 8, y - 14), text, fill=fill)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="PA-28-181 Performance Solver", layout="wide")
st.title("PA-28-181 Archer III — Performance Solver (JSON overlays + guides)")

mode = st.sidebar.selectbox("Modo", ["landing", "takeoff", "climb"], format_func=lambda m: m.capitalize())

st.sidebar.markdown("### Ficheiros")
up_bg = st.sidebar.file_uploader("Background (opcional)", type=["pdf", "jpg", "jpeg", "png"])
up_json = st.sidebar.file_uploader("JSON captura (opcional)", type=["json"])

cap = load_json(mode, up_json)
info = ASSETS[mode]

# background: se for PDF, precisa usar zoom/page do JSON (pixel-perfect)
if info["bg_kind"] == "pdf":
    if up_bg is not None:
        pdf_bytes = up_bg.read()
    else:
        p = _here(info["bg_default"])
        if not p:
            st.error(f"Não encontrei {info['bg_default']}")
            st.stop()
        pdf_bytes = load_bytes(p)
    bg = render_pdf_to_image(pdf_bytes, cap.get("page_index", 0), cap.get("zoom", 2.0))
else:
    if up_bg is not None:
        bg = Image.open(up_bg).convert("RGB")
    else:
        p = _here(info["bg_default"])
        if not p:
            st.error(f"Não encontrei {info['bg_default']}")
            st.stop()
        bg = Image.open(p).convert("RGB")

# Inputs
st.sidebar.markdown("### Inputs")
oat_c = st.sidebar.number_input("OAT (°C)", value=20.0, step=1.0)
pa_ft = st.sidebar.number_input("Pressure Altitude (ft)", value=0.0, step=500.0)

if mode in ("landing", "takeoff"):
    weight_lb = st.sidebar.number_input("Weight (lb)", value=2550.0, step=50.0)
    wind_kt = st.sidebar.number_input("Headwind (kt)", value=0.0, step=1.0)
else:
    weight_lb = 0.0
    wind_kt = 0.0

if mode == "climb":
    isa_dev = st.sidebar.number_input("ISA deviation (°C)", value=0.0, step=1.0)
    round_step = st.sidebar.number_input(
        "Arredondar ROC (fpm)",
        value=float(info["round_to"]),
        step=10.0,
        min_value=1.0,
    )
else:
    isa_dev = 0.0
    round_step = float(info["round_to"])

st.sidebar.markdown("### Visual")
show_overlay = st.sidebar.checkbox("Mostrar overlay (linhas/ticks/painéis/guides)", value=True)
show_path = st.sidebar.checkbox("Mostrar caminho do solver (setas)", value=True)
show_points = st.sidebar.checkbox("Marcar pontos chave", value=True)
show_debug = st.sidebar.checkbox("Mostrar debug", value=False)

# Solve
if mode in ("landing", "takeoff"):
    raw, segs, debug, pts = solve_ground_roll(cap, mode, oat_c, pa_ft, weight_lb, wind_kt)
    rounded = round_to_step(raw, info["round_to"])
    units = "ft"
    label = "Ground roll"
else:
    raw, segs, debug, pts = solve_climb(cap, oat_c, pa_ft, isa_dev)
    rounded = round_to_step(raw, round_step)
    units = "fpm"
    label = "Rate of climb"

left, right = st.columns([1.35, 1])

with right:
    st.subheader(info["title"])
    st.metric(f"{label} (arredondado)", f"{rounded:.0f} {units}")
    st.caption(f"Raw: {raw:.2f} {units}")
    if show_debug:
        st.markdown("#### Debug")
        st.json(debug)

with left:
    base = bg.copy()
    d = ImageDraw.Draw(base)

    if show_overlay:
        # linhas (vermelho)
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
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 0), width=3)

        # painéis (ciano)
        for panel, pts_panel in cap.get("panel_corners", {}).items():
            poly = [(p["x"], p["y"]) for p in pts_panel]
            d.line(poly + [poly[0]], fill=(0, 140, 255), width=3)
            d.text((poly[0][0] + 6, poly[0][1] + 6), str(panel), fill=(0, 140, 255))

    if show_path and segs:
        for p1, p2 in segs:
            draw_arrow(d, p1, p2, color=(255, 140, 0), w=5)

    if show_points and pts:
        for name, p in pts.items():
            draw_point(d, p, r=6, fill=(255, 200, 0))
            draw_label(d, p, name, fill=(0, 0, 0))

    st.image(base, use_container_width=True)
    st.caption("Vermelho: linhas (ISA/PA). Azul: guides. Verde: ticks. Ciano: painéis. Laranja: caminho do solver.")

