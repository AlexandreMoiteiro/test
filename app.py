from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import pymupdf
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# CONFIG
# =========================
PDF_NAME_DEFAULT = "PA28POH-ground-roll.pdf"
CAPTURE_NAME_DEFAULT = "capture.json"


# =========================
# GEOMETRY
# =========================
@dataclass(frozen=True)
class LineABC:
    """Reta a*x + b*y + c = 0 (a,b) normalizado."""
    a: float
    b: float
    c: float

    @staticmethod
    def from_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> "LineABC":
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = -(a * x1 + b * y1)
        n = float(np.hypot(a, b))
        if n == 0:
            return LineABC(0.0, 0.0, 0.0)
        return LineABC(a / n, b / n, c / n)

    def intersect(self, other: "LineABC") -> Optional[Tuple[float, float]]:
        det = self.a * other.b - other.a * self.b
        if abs(det) < 1e-10:
            return None
        x = (self.b * other.c - other.b * self.c) / det
        y = (other.a * self.c - self.a * other.c) / det
        return (float(x), float(y))


def horiz_line(y: float) -> LineABC:
    return LineABC(0.0, 1.0, -float(y))


def vert_line(x: float) -> LineABC:
    return LineABC(1.0, 0.0, -float(x))


def parallel_line_through(line: LineABC, p: Tuple[float, float]) -> LineABC:
    """Reta paralela (mesma normal a,b) que passa pelo ponto p."""
    x, y = p
    return LineABC(line.a, line.b, -(line.a * x + line.b * y))


# =========================
# IO
# =========================
def locate_file(name: str) -> Optional[str]:
    candidates = [Path.cwd() / name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def render_pdf_page_to_pil(pdf_path: str, page_index: int = 0, zoom: float = 2.3) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


# =========================
# ROBUST CALIBRATION (Theil–Sen)
# =========================
def robust_fit_x_from_value(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = np.array([t["value"] for t in ticks], dtype=float)
    xs = np.array([t["x"] for t in ticks], dtype=float)
    n = len(vals)
    if n < 2:
        raise ValueError("Poucos ticks para calibrar (precisas >=2).")
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dv = vals[j] - vals[i]
            if abs(dv) < 1e-9:
                continue
            slopes.append((xs[j] - xs[i]) / dv)
    m = float(np.median(slopes))
    b = float(np.median(xs - m * vals))
    return m, b


def robust_fit_value_from_y(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    ys = np.array([t["y"] for t in ticks], dtype=float)
    vals = np.array([t["value"] for t in ticks], dtype=float)
    n = len(ys)
    if n < 2:
        raise ValueError("Poucos ticks para calibrar (precisas >=2).")
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dy = ys[j] - ys[i]
            if abs(dy) < 1e-9:
                continue
            slopes.append((vals[j] - vals[i]) / dy)
    m = float(np.median(slopes))
    b = float(np.median(vals - m * ys))
    return m, b


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# NOMOGRAM HELPERS
# =========================
def std_atm_isa_temp_c(pressure_alt_ft: float) -> float:
    return 15.0 - 1.98 * (pressure_alt_ft / 1000.0)


def interpolate_line(pairs: List[Tuple[float, LineABC]], value: float) -> LineABC:
    pairs = sorted(pairs, key=lambda t: t[0])
    values = [v for v, _ in pairs]
    v = clamp(value, values[0], values[-1])

    for i in range(len(pairs) - 1):
        v1, l1 = pairs[i]
        v2, l2 = pairs[i + 1]
        if v1 <= v <= v2:
            t = 0.0 if v2 == v1 else (v - v1) / (v2 - v1)
            a = (1 - t) * l1.a + t * l2.a
            b = (1 - t) * l1.b + t * l2.b
            c = (1 - t) * l1.c + t * l2.c
            n = float(np.hypot(a, b))
            if n == 0:
                return LineABC(0.0, 0.0, 0.0)
            return LineABC(a / n, b / n, c / n)

    return pairs[-1][1]


def seg_to_line_from_capture(capture: Dict[str, Any], key: str) -> LineABC:
    seg = capture["lines"][key][0]
    return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


# =========================
# SOLVER (usa guias capturadas se existirem)
# =========================
def solve_ground_roll(
    capture: Dict[str, Any],
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    headwind_kt: float,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:

    ticks_oat = capture["axis_ticks"]["oat_c"]
    ticks_wt = capture["axis_ticks"]["weight_x100_lb"]
    ticks_wnd = capture["axis_ticks"]["wind_kt"]
    ticks_gr = capture["axis_ticks"]["ground_roll_ft"]

    mo, bo = robust_fit_x_from_value(ticks_oat)
    mw, bw = robust_fit_x_from_value(ticks_wt)
    mwd, bwd = robust_fit_x_from_value(ticks_wnd)
    mgr, bgr = robust_fit_value_from_y(ticks_gr)

    oat_axis_y = float(np.median([t["y"] for t in ticks_oat]))
    gr_axis_x = float(np.median([t["x"] for t in ticks_gr]))

    def x_from_oat(v): return mo * v + bo
    def x_from_weight_lb(v): return mw * (v / 100.0) + bw
    def x_from_wind(v): return mwd * v + bwd
    def gr_from_y(y): return mgr * y + bgr

    # left family lines
    isa_m15 = seg_to_line_from_capture(capture, "isa_m15")
    isa_0   = seg_to_line_from_capture(capture, "isa")
    isa_p35 = seg_to_line_from_capture(capture, "isa_p35")

    pa0 = seg_to_line_from_capture(capture, "pa_sea_level")
    pa2 = seg_to_line_from_capture(capture, "pa_2000")
    pa4 = seg_to_line_from_capture(capture, "pa_4000")
    pa6 = seg_to_line_from_capture(capture, "pa_6000")
    pa7 = seg_to_line_from_capture(capture, "pa_7000")

    w_ref  = seg_to_line_from_capture(capture, "weight_ref_line")
    z_wind = seg_to_line_from_capture(capture, "wind_ref_zero")

    # ISA deviation
    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15.0, isa_m15), (0.0, isa_0), (35.0, isa_p35)], dev)
    pa_line  = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA (ver linhas capturadas).")

    # extra arrow up from OAT axis
    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    # to weight ref (horizontal)
    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou interseção com weight_ref_line.")

    # MIDDLE: usar guia capturada se existir, senão fallback geométrico simples
    x_weight = float(x_from_weight_lb(weight_lb))
    if "weight_guide" in capture["lines"]:
        g = seg_to_line_from_capture(capture, "weight_guide")
        g_through = parallel_line_through(g, p_wref)
        p_w = g_through.intersect(vert_line(x_weight))
        if p_w is None:
            p_w = (x_weight, p_wref[1])
    else:
        # fallback: diagonal "genérica" (só para não crashar)
        p_w = (x_weight, p_wref[1])

    # to wind zero (horizontal)
    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou interseção com wind_ref_zero.")

    # RIGHT: usar guia capturada se existir
    x_wind = float(x_from_wind(headwind_kt))
    if "wind_guide" in capture["lines"]:
        g = seg_to_line_from_capture(capture, "wind_guide")
        g_through = parallel_line_through(g, p_z)
        p_wind = g_through.intersect(vert_line(x_wind))
        if p_wind is None:
            p_wind = (x_wind, p_z[1])
    else:
        p_wind = (x_wind, p_z[1])

    # ground roll
    gr_raw = float(gr_from_y(p_wind[1]))
    gr_round = float(5.0 * round(gr_raw / 5.0))
    p_gr = (gr_axis_x, p_wind[1])

    points = {
        "p_oat": p_oat,
        "p_left": p_left,
        "p_wref": p_wref,
        "p_w": p_w,
        "p_z": p_z,
        "p_wind": p_wind,
        "p_gr": p_gr,
    }
    return gr_raw, gr_round, points


# =========================
# DRAWING (preview only; PDF export opcional depois)
# =========================
def draw_arrow(draw: ImageDraw.ImageDraw, p1, p2, width=5):
    draw.line([p1, p2], fill=(255, 0, 0), width=width)
    x1, y1 = p1
    x2, y2 = p2
    vx, vy = x2 - x1, y2 - y1
    L = float(np.hypot(vx, vy))
    if L < 2:
        return
    ux, uy = vx / L, vy / L
    ah, aw = 18.0, 10.0
    bx, by = x2 - ah * ux, y2 - ah * uy
    px, py = -uy, ux
    tri = [(x2, y2), (bx + aw * px, by + aw * py), (bx - aw * px, by - aw * py)]
    draw.polygon(tri, fill=(255, 0, 0))


def draw_debug(img: Image.Image, points: Dict[str, Tuple[float, float]], gr_round: float) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    draw_arrow(d, points["p_oat"], points["p_left"], width=5)
    draw_arrow(d, points["p_left"], points["p_wref"], width=5)
    draw_arrow(d, points["p_wref"], points["p_w"], width=5)
    draw_arrow(d, points["p_w"], points["p_z"], width=5)
    draw_arrow(d, points["p_z"], points["p_wind"], width=5)
    draw_arrow(d, points["p_wind"], points["p_gr"], width=5)

    # label
    xg, yg = points["p_gr"]
    d.text((xg + 8, yg - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))

    return out


# =========================
# CAPTURE UI HELPERS
# =========================
def ensure_capture_schema(cap: Dict[str, Any]) -> Dict[str, Any]:
    cap.setdefault("lines", {})
    cap.setdefault("axis_ticks", {})
    cap.setdefault("panel_corners", {})
    cap.setdefault("zoom", 2.3)
    return cap


def add_line_segment(cap: Dict[str, Any], key: str, p1: Dict[str, float], p2: Dict[str, float]) -> None:
    cap = ensure_capture_schema(cap)
    cap["lines"][key] = [{
        "x1": float(p1["x"]), "y1": float(p1["y"]),
        "x2": float(p2["x"]), "y2": float(p2["y"]),
    }]


def draw_existing_lines(img: Image.Image, cap: Dict[str, Any]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    if "lines" not in cap:
        return out
    for k, segs in cap["lines"].items():
        if not segs:
            continue
        s = segs[0]
        p1 = (s["x1"], s["y1"])
        p2 = (s["x2"], s["y2"])
        d.line([p1, p2], fill=(0, 180, 0), width=3)
        d.text((p1[0] + 6, p1[1] - 10), k, fill=(0, 0, 0))
    return out


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Ground Roll Nomogram", layout="wide")
st.title("PA-28 Landing Ground Roll — Solver + Editor de linhas (JSON)")

# Load files from repo OR upload
colA, colB = st.columns(2)
with colA:
    pdf_up = st.file_uploader("PDF (opcional)", type=["pdf"])
with colB:
    json_up = st.file_uploader("capture.json (opcional)", type=["json"])

pdf_path = locate_file(PDF_NAME_DEFAULT)
json_path = locate_file(CAPTURE_NAME_DEFAULT)

if pdf_up is not None:
    Path("uploaded.pdf").write_bytes(pdf_up.read())
    pdf_path = "uploaded.pdf"

if json_up is not None:
    Path("capture.json").write_bytes(json_up.read())
    json_path = "capture.json"

if not pdf_path:
    st.error(f"Não encontrei '{PDF_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()
if not json_path:
    st.error(f"Não encontrei '{CAPTURE_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

cap = ensure_capture_schema(json.loads(Path(json_path).read_text(encoding="utf-8")))
zoom = float(cap.get("zoom", 2.3))
base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)

tab1, tab2 = st.tabs(["✅ Solver", "🛠️ Editor / Captura de linhas (JSON)"])

# -------------------------
# Solver tab
# -------------------------
with tab1:
    st.subheader("Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pressure_alt_ft = st.number_input("Pressure Altitude (ft)", value=2500.0, step=100.0)
    with c2:
        oat_c = st.number_input("OAT (°C)", value=21.0, step=1.0)
    with c3:
        weight_lb = st.number_input("Weight (lb)", value=2240.0, step=10.0)
    with c4:
        headwind_kt = st.number_input("Headwind (kt)", value=5.0, step=1.0)

    run = st.button("Calcular", key="run_solver")
    show_raw = st.checkbox("Mostrar valor bruto", value=False)

    if run:
        try:
            gr_raw, gr_round, pts = solve_ground_roll(
                capture=cap,
                pressure_alt_ft=float(pressure_alt_ft),
                oat_c=float(oat_c),
                weight_lb=float(weight_lb),
                headwind_kt=float(headwind_kt),
            )
        except Exception as e:
            st.exception(e)
            st.stop()

        st.success(f"Landing Ground Roll ≈ **{gr_round:.0f} ft** (arredondado de 5 em 5)")
        if show_raw:
            st.caption(f"Bruto: {gr_raw:.1f} ft")

        dbg = draw_debug(base_img, pts, gr_round)
        st.image(dbg, use_container_width=True)

        if "weight_guide" not in cap["lines"] or "wind_guide" not in cap["lines"]:
            st.warning("Dica: adiciona 'weight_guide' e 'wind_guide' na aba Editor para ficar 100% pilot-like.")


# -------------------------
# Editor tab
# -------------------------
with tab2:
    st.subheader("Adicionar/atualizar linhas no capture.json")
    st.write(
        "Escolhe uma linha (ex: `weight_guide` ou `wind_guide`) e clica **2 pontos** na imagem.\n\n"
        "**Recomendação de ordem dos cliques**:\n"
        "- verticais: **baixo → cima**\n"
        "- horizontais: **esquerda → direita**\n"
        "- diagonais-guia: **esquerda → direita** (bem afastados)\n"
    )

    # default keys + permitir escrever uma key nova
    default_keys = [
        "weight_guide",
        "wind_guide",
        "custom_guide_1",
        "custom_guide_2",
    ]
    existing_keys = sorted(set(list(cap.get("lines", {}).keys()) + default_keys))

    key_choice = st.selectbox("Linha a capturar (key no JSON)", existing_keys, index=0)
    new_key = st.text_input("Ou escreve uma key nova (opcional)", value="")
    line_key = new_key.strip() if new_key.strip() else key_choice

    # session state for clicks
    st.session_state.setdefault("click_points", [])
    st.session_state.setdefault("last_click", None)

    # show image with existing lines (para orientar)
    preview = draw_existing_lines(base_img, cap)
    preview_np = np.array(preview)  # IMPORTANT: streamlit_image_coordinates aceita numpy array

    st.info("Clica na imagem para capturar pontos. Vais ver o contador de pontos em baixo.")
    click = streamlit_image_coordinates(preview_np, key="img_clicks")

    if click is not None:
        st.session_state.last_click = click
        # click tem 'x','y'
        st.session_state.click_points.append({"x": float(click["x"]), "y": float(click["y"])})

    colu1, colu2, colu3 = st.columns(3)
    with colu1:
        st.write(f"**Pontos capturados nesta linha**: {len(st.session_state.click_points)}/2")
        if st.session_state.last_click:
            st.caption(f"Último clique: x={st.session_state.last_click['x']}, y={st.session_state.last_click['y']}")
    with colu2:
        if st.button("↩️ Undo último ponto"):
            if st.session_state.click_points:
                st.session_state.click_points.pop()
    with colu3:
        if st.button("🧹 Limpar pontos desta linha"):
            st.session_state.click_points = []

    if len(st.session_state.click_points) >= 2:
        p1, p2 = st.session_state.click_points[0], st.session_state.click_points[1]

        st.success(f"Pronto para gravar: **{line_key}**")
        st.json({"key": line_key, "p1": p1, "p2": p2})

        if st.button("💾 Gravar no JSON"):
            add_line_segment(cap, line_key, p1, p2)
            # limpar para próxima linha
            st.session_state.click_points = []
            # guardar em disco local do app
            Path("capture.json").write_text(json.dumps(cap, indent=2), encoding="utf-8")
            st.success(f"Guardado: {line_key} (atualizado em capture.json)")

    st.divider()
    st.subheader("Download do capture.json atualizado")
    cap_bytes = json.dumps(cap, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download capture.json",
        data=cap_bytes,
        file_name="capture.json",
        mime="application/json",
    )



