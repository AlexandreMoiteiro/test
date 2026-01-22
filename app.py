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


PDF_NAME_DEFAULT = "PA28POH-ground-roll.pdf"
CAPTURE_NAME_DEFAULT = "capture.json"


# =========================
# Geometry
# =========================
@dataclass(frozen=True)
class LineABC:
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
# Robust calibration (Theil–Sen)
# =========================
def robust_fit_x_from_value(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = np.array([t["value"] for t in ticks], dtype=float)
    xs = np.array([t["x"] for t in ticks], dtype=float)
    n = len(vals)
    if n < 2:
        raise ValueError("Poucos ticks para calibrar eixo (>=2).")
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
        raise ValueError("Poucos ticks para calibrar eixo (>=2).")
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
# Nomogram model helpers
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


def get_line_from_capture(capture: Dict[str, Any], key: str) -> Optional[LineABC]:
    if "lines" in capture and key in capture["lines"] and capture["lines"][key]:
        seg = capture["lines"][key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))
    if "guides" in capture and key in capture["guides"]:
        seg = capture["guides"][key]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))
    return None


def solve_ground_roll(
    capture: Dict[str, Any],
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    headwind_kt: float,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    ticks_oat = capture["axis_ticks"]["oat_c"]
    ticks_wt  = capture["axis_ticks"]["weight_x100_lb"]
    ticks_wnd = capture["axis_ticks"]["wind_kt"]
    ticks_gr  = capture["axis_ticks"]["ground_roll_ft"]

    mo, bo   = robust_fit_x_from_value(ticks_oat)
    mw, bw   = robust_fit_x_from_value(ticks_wt)
    mwd, bwd = robust_fit_x_from_value(ticks_wnd)
    mgr, bgr = robust_fit_value_from_y(ticks_gr)

    oat_axis_y = float(np.median([t["y"] for t in ticks_oat]))
    gr_axis_x  = float(np.median([t["x"] for t in ticks_gr]))

    def x_from_oat(v): return mo * v + bo
    def x_from_weight_lb(v): return mw * (v / 100.0) + bw
    def x_from_wind(v): return mwd * v + bwd
    def gr_from_y(y): return mgr * y + bgr

    # Required lines
    isa_m15 = get_line_from_capture(capture, "isa_m15")
    isa_0   = get_line_from_capture(capture, "isa")
    isa_p35 = get_line_from_capture(capture, "isa_p35")

    pa0 = get_line_from_capture(capture, "pa_sea_level")
    pa2 = get_line_from_capture(capture, "pa_2000")
    pa4 = get_line_from_capture(capture, "pa_4000")
    pa6 = get_line_from_capture(capture, "pa_6000")
    pa7 = get_line_from_capture(capture, "pa_7000")

    w_ref  = get_line_from_capture(capture, "weight_ref_line")
    z_wind = get_line_from_capture(capture, "wind_ref_zero")

    if not all([isa_m15, isa_0, isa_p35, pa0, pa2, pa4, pa6, pa7, w_ref, z_wind]):
        raise ValueError("Faltam linhas base no capture.json (ISA/PA/ref lines).")

    # Optional guides you will capture:
    weight_guide = get_line_from_capture(capture, "weight_guide")
    wind_guide   = get_line_from_capture(capture, "wind_guide")

    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15.0, isa_m15), (0.0, isa_0), (35.0, isa_p35)], dev)
    pa_line  = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA.")

    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou interseção com weight_ref_line.")

    # middle
    xw = float(x_from_weight_lb(weight_lb))
    if weight_guide is not None:
        ln = parallel_line_through(weight_guide, p_wref)
        p_w = ln.intersect(vert_line(xw))
    else:
        theta = np.deg2rad(-55.0)
        fallback = LineABC.from_points(p_wref, (p_wref[0] + 100*np.cos(theta), p_wref[1] + 100*np.sin(theta)))
        p_w = fallback.intersect(vert_line(xw))

    if p_w is None:
        raise ValueError("Falhou projeção no painel do weight (captura weight_guide).")

    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou interseção com wind_ref_zero.")

    # right
    xv = float(x_from_wind(headwind_kt))
    if wind_guide is not None:
        ln = parallel_line_through(wind_guide, p_z)
        p_wind = ln.intersect(vert_line(xv))
    else:
        theta = np.deg2rad(-55.0)
        fallback = LineABC.from_points(p_z, (p_z[0] + 100*np.cos(theta), p_z[1] + 100*np.sin(theta)))
        p_wind = fallback.intersect(vert_line(xv))

    if p_wind is None:
        raise ValueError("Falhou projeção no painel do vento (captura wind_guide).")

    gr_raw = float(gr_from_y(p_wind[1]))
    gr_round = float(5.0 * round(gr_raw / 5.0))
    p_gr = (gr_axis_x, p_wind[1])

    pts = {
        "p_oat": p_oat,
        "p_left": p_left,
        "p_wref": p_wref,
        "p_w": p_w,
        "p_z": p_z,
        "p_wind": p_wind,
        "p_gr": p_gr,
    }
    return gr_raw, gr_round, pts


# =========================
# Drawing
# =========================
def draw_arrow_pil(draw: ImageDraw.ImageDraw, p1, p2, width=5):
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


def overlay_segments(img: Image.Image, capture: Dict[str, Any], pending: Optional[Tuple[float, float]] = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    if "lines" in capture:
        for _, segs in capture["lines"].items():
            for seg in segs:
                d.line([(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])], fill=(0, 140, 0), width=2)

    if "guides" in capture:
        for _, seg in capture["guides"].items():
            d.line([(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])], fill=(0, 0, 200), width=3)

    if pending is not None:
        x, y = pending
        r = 7
        d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 120, 0), width=4)

    return out


def draw_solution(img: Image.Image, pts: Dict[str, Tuple[float, float]], gr_round: float) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    draw_arrow_pil(d, pts["p_oat"], pts["p_left"], width=5)
    draw_arrow_pil(d, pts["p_left"], pts["p_wref"], width=5)
    draw_arrow_pil(d, pts["p_wref"], pts["p_w"], width=5)
    draw_arrow_pil(d, pts["p_w"], pts["p_z"], width=5)
    draw_arrow_pil(d, pts["p_z"], pts["p_wind"], width=5)
    draw_arrow_pil(d, pts["p_wind"], pts["p_gr"], width=5)

    def mark(p, color, r=6):
        x, y = p
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)

    for k in ["p_oat", "p_left", "p_wref", "p_w", "p_z", "p_wind", "p_gr"]:
        mark(pts[k], (0, 120, 0) if k != "p_wind" else (0, 0, 200))

    xg, yg = pts["p_gr"]
    d.text((xg + 8, yg - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))
    return out


# =========================
# Streamlit
# =========================
st.set_page_config(page_title="PA28 Ground Roll", layout="wide")
st.title("PA-28 Landing Ground Roll — Solver + Editor (2 cliques = guarda)")

colA, colB = st.columns(2)
with colA:
    pdf_up = st.file_uploader("PDF (opcional, senão usa o da pasta)", type=["pdf"])
with colB:
    cap_up = st.file_uploader("capture.json (opcional, senão usa o da pasta)", type=["json"])

pdf_path = locate_file(PDF_NAME_DEFAULT)
cap_path = locate_file(CAPTURE_NAME_DEFAULT)

if pdf_up is not None:
    Path("uploaded.pdf").write_bytes(pdf_up.read())
    pdf_path = "uploaded.pdf"
if cap_up is not None:
    Path("capture.json").write_bytes(cap_up.read())
    cap_path = "capture.json"

if not pdf_path:
    st.error(f"Não encontrei '{PDF_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()
if not cap_path:
    st.error(f"Não encontrei '{CAPTURE_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

capture_disk = json.loads(Path(cap_path).read_text(encoding="utf-8"))
if "capture" not in st.session_state:
    st.session_state.capture = capture_disk

capture = st.session_state.capture
capture.setdefault("lines", {})
capture.setdefault("guides", {})
zoom = float(capture.get("zoom", 2.3))

# editor state
if "pending_point" not in st.session_state:
    st.session_state.pending_point = None
if "last_click" not in st.session_state:
    st.session_state.last_click = None  # to dedupe repeated clicks
if "selected_kind" not in st.session_state:
    st.session_state.selected_kind = "guides"
if "selected_key" not in st.session_state:
    st.session_state.selected_key = "weight_guide"

tabs = st.tabs(["✅ Solver", "🛠️ Editor do JSON (2 cliques = guarda e avança)"])


with tabs[0]:
    st.subheader("Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pressure_alt_ft = st.number_input("Pressure Altitude (ft)", value=2500.0, step=100.0)
    with c2:
        oat_c = st.number_input("OAT (°C)", value=21.0, step=1.0)
    with c3:
        weight_lb = st.number_input("Weight (lb)", value=2240.0, step=10.0)
    with c4:
        headwind_kt = st.number_input("Headwind component (kt)", value=5.0, step=1.0)

    run = st.button("Calcular e desenhar", key="run_solver")

    base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)
    base_overlay = overlay_segments(base_img, capture)

    if run:
        try:
            gr_raw, gr_round, pts = solve_ground_roll(
                capture=capture,
                pressure_alt_ft=float(pressure_alt_ft),
                oat_c=float(oat_c),
                weight_lb=float(weight_lb),
                headwind_kt=float(headwind_kt),
            )
            st.success(f"Landing Ground Roll ≈ **{gr_round:.0f} ft** (arredondado de 5 em 5)")
            st.caption(f"Bruto: {gr_raw:.1f} ft")
            out_img = draw_solution(base_overlay, pts, gr_round)
            st.image(out_img, use_container_width=True)
        except Exception as e:
            st.exception(e)
    else:
        st.image(base_overlay, use_container_width=True)


with tabs[1]:
    st.subheader("Editor — dois cliques guardam um segmento e ficam prontos para o próximo")

    st.markdown(
        """
**Fluxo:**
- 1º clique: fica “armado” (ponto laranja).
- 2º clique: grava o segmento e **reseta automaticamente** para começares outro.

**Dica:** para `weight_guide` e `wind_guide`, clica em 2 pontos bem afastados na diagonal grossa.
"""
    )

    # selection UI
    kind = st.radio("Tipo", ["guides (1 segmento)", "lines (lista de segmentos)"], horizontal=True)
    kind_norm = "guides" if kind.startswith("guides") else "lines"
    st.session_state.selected_kind = kind_norm

    if kind_norm == "guides":
        key = st.selectbox("Guide key", ["weight_guide", "wind_guide"] + sorted([k for k in capture["guides"].keys() if k not in ("weight_guide", "wind_guide")]))
    else:
        default_line_keys = [
            "isa", "isa_m15", "isa_p35",
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_7000",
            "weight_ref_line", "wind_ref_zero",
        ]
        existing = sorted(list(capture["lines"].keys()))
        key = st.selectbox("Line key", sorted(list(set(default_line_keys + existing))))

    # whenever key changes, cancel pending point to avoid carry-over
    if st.session_state.selected_key != key:
        st.session_state.selected_key = key
        st.session_state.pending_point = None
        st.session_state.last_click = None

    # buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Cancelar ponto pendente"):
            st.session_state.pending_point = None
            st.session_state.last_click = None
    with b2:
        if st.button("Undo último (desta key)"):
            if kind_norm == "guides":
                capture["guides"].pop(key, None)
            else:
                if key in capture["lines"] and capture["lines"][key]:
                    capture["lines"][key].pop()
                    if not capture["lines"][key]:
                        capture["lines"].pop(key, None)
            st.session_state.pending_point = None
            st.session_state.last_click = None
    with b3:
        st.download_button(
            "⬇️ Download capture.json atualizado",
            data=json.dumps(capture, indent=2).encode("utf-8"),
            file_name="capture.json",
            mime="application/json",
        )

    # image with overlay + pending point marker
    base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)
    img_show = overlay_segments(base_img, capture, pending=st.session_state.pending_point)
    img_arr = np.array(img_show)

    click = streamlit_image_coordinates(img_arr, key="img_clicker_fixed")

    # process click with de-dupe
    if click is not None:
        x = float(click["x"])
        y = float(click["y"])
        current = (round(x, 1), round(y, 1))

        # ignore repeated last click (component sometimes repeats on rerun)
        if st.session_state.last_click != current:
            st.session_state.last_click = current

            if st.session_state.pending_point is None:
                st.session_state.pending_point = (x, y)
                st.toast(f"1º ponto: ({x:.1f}, {y:.1f}) — clica no 2º ponto")
            else:
                x1, y1 = st.session_state.pending_point
                x2, y2 = x, y
                seg = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}

                if kind_norm == "guides":
                    capture["guides"][key] = seg
                else:
                    capture["lines"].setdefault(key, [])
                    capture["lines"][key].append(seg)

                st.session_state.pending_point = None
                st.toast(f"Guardado: {kind_norm}.{key}  ({x1:.1f},{y1:.1f}) → ({x2:.1f},{y2:.1f})")

    st.image(img_show, use_container_width=True)


