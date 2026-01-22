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

    def distance_to_point(self, p: Tuple[float, float]) -> float:
        # line normalized => |ax+by+c|
        return abs(self.a * p[0] + self.b * p[1] + self.c)


def horiz_line(y: float) -> LineABC:
    return LineABC(0.0, 1.0, -float(y))


def vert_line(x: float) -> LineABC:
    return LineABC(1.0, 0.0, -float(x))


def parallel_line_through(line: LineABC, p: Tuple[float, float]) -> LineABC:
    x, y = p
    return LineABC(line.a, line.b, -(line.a * x + line.b * y))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# IO + RENDER
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
# ROBUST CALIBRATION
# =========================
def robust_fit_x_from_value(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = np.array([t["value"] for t in ticks], dtype=float)
    xs = np.array([t["x"] for t in ticks], dtype=float)
    n = len(vals)
    if n < 2:
        raise ValueError("Poucos ticks para calibrar eixo (precisas >=2).")
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
        raise ValueError("Poucos ticks para calibrar eixo (precisas >=2).")
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
            b = (1 - t) * l2.b + t * l2.b
            # BUG FIX: correct interpolation b
            b = (1 - t) * l1.b + t * l2.b
            c = (1 - t) * l1.c + t * l2.c
            n = float(np.hypot(a, b))
            if n == 0:
                return LineABC(0.0, 0.0, 0.0)
            return LineABC(a / n, b / n, c / n)
    return pairs[-1][1]


def seg_to_line(seg: Dict[str, float]) -> LineABC:
    return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


def choose_nearest_guide(guide_segs: List[Dict[str, float]], p_start: Tuple[float, float]) -> Optional[LineABC]:
    if not guide_segs:
        return None
    lines = [seg_to_line(g) for g in guide_segs]
    return min(lines, key=lambda ln: ln.distance_to_point(p_start))


# =========================
# DRAWING (ARROWS)
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


def add_arrow_path_preview(img: Image.Image, pts: List[Tuple[float, float]], label: Optional[str] = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for a, b in zip(pts[:-1], pts[1:]):
        draw_arrow_pil(d, a, b, width=5)
    for p in pts:
        x, y = p
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 120, 0), width=4)
    if label and len(pts) >= 1:
        x, y = pts[-1]
        d.text((x + 8, y - 14), label, fill=(0, 0, 0))
    return out


# =========================
# SOLVER (uses guide lines if present)
# =========================
def solve_with_guides(
    capture: Dict[str, Any],
    base_img: Image.Image,
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    wind_kt_signed: float,  # negative = tailwind
) -> Tuple[float, float, List[Tuple[float, float]]]:

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

    lines = capture["lines"]

    def get_line(key: str) -> LineABC:
        seg = lines[key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))

    # base lines
    isa_m15 = get_line("isa_m15")
    isa_0 = get_line("isa")
    isa_p35 = get_line("isa_p35")
    pa0 = get_line("pa_sea_level")
    pa2 = get_line("pa_2000")
    pa4 = get_line("pa_4000")
    pa6 = get_line("pa_6000")
    pa7 = get_line("pa_7000")
    w_ref = get_line("weight_ref_line")
    z_wind = get_line("wind_ref_zero")

    # left: ISA deviation + PA
    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15.0, isa_m15), (0.0, isa_0), (35.0, isa_p35)], dev)
    pa_line = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)
    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA")

    # arrow up from OAT axis
    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    # to weight ref horizontal
    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou interseção com weight ref line")

    # weight guide: choose nearest captured guide, make parallel through p_wref
    weight_guides = capture.get("guide_lines", {}).get("weight_guides", [])
    guide_w = choose_nearest_guide(weight_guides, p_wref)

    xw = float(x_from_weight_lb(weight_lb))
    if guide_w is None:
        # fallback: simple diagonal-ish, but better to capture guides
        p_w = (xw, p_wref[1])
    else:
        ln = parallel_line_through(guide_w, p_wref)
        p_w = ln.intersect(vert_line(xw))
        if p_w is None:
            p_w = (xw, p_wref[1])

    # to wind ref (horizontal)
    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou interseção com wind zero ref line")

    # wind guide set based on sign
    wind_mag = abs(float(wind_kt_signed))
    if wind_kt_signed >= 0:
        wind_guides = capture.get("guide_lines", {}).get("wind_headwind_guides", [])
    else:
        wind_guides = capture.get("guide_lines", {}).get("wind_tailwind_guides", [])

    guide_wind = choose_nearest_guide(wind_guides, p_z)
    xwind = float(x_from_wind(wind_mag))

    if guide_wind is None:
        p_wind = (xwind, p_z[1])
    else:
        ln = parallel_line_through(guide_wind, p_z)
        p_wind = ln.intersect(vert_line(xwind))
        if p_wind is None:
            p_wind = (xwind, p_z[1])

    # ground roll from y
    gr_raw = float(gr_from_y(p_wind[1]))
    gr_round = float(5.0 * round(gr_raw / 5.0))
    p_gr = (gr_axis_x, p_wind[1])

    path = [p_oat, p_left, p_wref, p_w, p_z, p_wind, p_gr]
    return gr_raw, gr_round, path


# =========================
# JSON EDITOR (capture guide lines)
# =========================
def ensure_guide_struct(capture: Dict[str, Any]) -> None:
    if "guide_lines" not in capture or not isinstance(capture["guide_lines"], dict):
        capture["guide_lines"] = {}
    gl = capture["guide_lines"]
    gl.setdefault("weight_guides", [])
    gl.setdefault("wind_headwind_guides", [])
    gl.setdefault("wind_tailwind_guides", [])


def add_segment(capture: Dict[str, Any], key: str, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
    ensure_guide_struct(capture)
    seg = {"x1": float(p1[0]), "y1": float(p1[1]), "x2": float(p2[0]), "y2": float(p2[1])}
    capture["guide_lines"][key].append(seg)


def overlay_existing_guides(img: Image.Image, capture: Dict[str, Any], key: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    ensure_guide_struct(capture)
    for seg in capture["guide_lines"][key]:
        d.line([(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])], fill=(0, 160, 0), width=4)
    return out


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Nomogram Editor + Solver", layout="wide")
st.title("Nomograma — Editor (guias) + Solver (headwind/tailwind)")

# Load files
colA, colB = st.columns(2)
with colA:
    pdf_up = st.file_uploader("PDF (opcional, senão usa o da pasta)", type=["pdf"])
with colB:
    cap_up = st.file_uploader("capture.json (opcional, senão usa o da pasta)", type=["json"])

pdf_path = locate_file(PDF_NAME_DEFAULT)
cap_path = locate_file(CAPTURE_NAME_DEFAULT)

if pdf_up is not None:
    Path("uploaded.pdf").write_bytes(pdf_up.read())
    pdf_path = str(Path("uploaded.pdf"))

if cap_up is not None:
    Path("capture.json").write_bytes(cap_up.read())
    cap_path = str(Path("capture.json"))

if not pdf_path:
    st.error(f"Não encontrei '{PDF_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

# load capture
if cap_path and Path(cap_path).exists():
    capture = json.loads(Path(cap_path).read_text(encoding="utf-8"))
else:
    capture = {"zoom": 2.3, "lines": {}, "axis_ticks": {}, "panel_corners": {}}

ensure_guide_struct(capture)
zoom = float(capture.get("zoom", 2.3))
base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)

mode = st.radio("Modo", ["Editor de guias (JSON)", "Solver"], horizontal=True)

# session state for clicks
if "pending_p1" not in st.session_state:
    st.session_state.pending_p1 = None

if mode == "Editor de guias (JSON)":
    st.subheader("Editor: adiciona linhas-guia (weight / wind headwind / wind tailwind)")
    st.markdown(
        "- Clica 2 pontos por linha (ponto 1 e ponto 2).\n"
        "- Recomendo: **de baixo para cima** para ficar organizado.\n"
        "- Captura ~5–7 linhas por conjunto.\n"
    )

    key = st.selectbox(
        "Que conjunto queres editar?",
        ["weight_guides", "wind_headwind_guides", "wind_tailwind_guides"],
        help="weight = painel do meio; wind_headwind e wind_tailwind = painel da direita"
    )

    img_overlay = overlay_existing_guides(base_img, capture, key)
    click = streamlit_image_coordinates(img_overlay, key=f"click_{key}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Ponto pendente:", st.session_state.pending_p1)
    with c2:
        if st.button("Limpar ponto pendente"):
            st.session_state.pending_p1 = None
    with c3:
        if st.button("Apagar última linha desse conjunto"):
            if capture["guide_lines"][key]:
                capture["guide_lines"][key].pop()

    if click is not None:
        p = (float(click["x"]), float(click["y"]))
        if st.session_state.pending_p1 is None:
            st.session_state.pending_p1 = p
        else:
            p1 = st.session_state.pending_p1
            p2 = p
            add_segment(capture, key, p1, p2)
            st.session_state.pending_p1 = None

    st.write("Total de linhas capturadas:", {k: len(capture["guide_lines"][k]) for k in capture["guide_lines"]})

    json_bytes = json.dumps(capture, indent=2).encode("utf-8")
    st.download_button("⬇️ Download capture.json atualizado", data=json_bytes, file_name="capture.json", mime="application/json")

else:
    st.subheader("Solver")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pressure_alt_ft = st.number_input("Pressure Altitude (ft)", value=2500.0, step=100.0)
    with c2:
        oat_c = st.number_input("OAT (°C)", value=21.0, step=1.0)
    with c3:
        weight_lb = st.number_input("Weight (lb)", value=2240.0, step=10.0)
    with c4:
        wind_signed = st.number_input("Wind component (kt) (tailwind negativo)", value=5.0, step=1.0)

    show_raw = st.checkbox("Mostrar bruto", value=False)

    if st.button("Calcular"):
        try:
            gr_raw, gr_round, path = solve_with_guides(
                capture=capture,
                base_img=base_img,
                pressure_alt_ft=float(pressure_alt_ft),
                oat_c=float(oat_c),
                weight_lb=float(weight_lb),
                wind_kt_signed=float(wind_signed),
            )
        except Exception as e:
            st.exception(e)
            st.stop()

        st.success(f"Landing Ground Roll ≈ **{gr_round:.0f} ft** (arredondado de 5 em 5)")
        if show_raw:
            st.caption(f"Bruto: {gr_raw:.1f} ft")

        # draw path with arrows
        img = base_img.copy()
        d = ImageDraw.Draw(img)
        for a, b in zip(path[:-1], path[1:]):
            draw_arrow_pil(d, a, b, width=5)
        # label at end (ground roll axis)
        x, y = path[-1]
        d.text((x + 8, y - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))

        st.image(img, use_container_width=True)

