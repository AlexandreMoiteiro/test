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


def parallel_line_through(base: LineABC, p: Tuple[float, float]) -> LineABC:
    x, y = p
    return LineABC(base.a, base.b, -(base.a * x + base.b * y))


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
        raise ValueError("Poucos ticks para calibrar.")
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
        raise ValueError("Poucos ticks para calibrar.")
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
# Nomogram model
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
            return LineABC(a / n, b / n, c / n)
    return pairs[-1][1]


def choose_guide_and_intersect(
    guide_low: Optional[LineABC],
    guide_high: Optional[LineABC],
    p_start: Tuple[float, float],
    x_target: float,
) -> Optional[Tuple[float, float]]:
    """
    Usa 1 ou 2 guias capturadas.
    - Se só existir uma: usa essa direção (reta paralela por p_start).
    - Se existirem duas: escolhe a mais próxima em y (ou interpola direção).
    """
    if guide_low is None and guide_high is None:
        return None

    if guide_low is not None and guide_high is None:
        ln = parallel_line_through(guide_low, p_start)
        return ln.intersect(vert_line(x_target))
    if guide_high is not None and guide_low is None:
        ln = parallel_line_through(guide_high, p_start)
        return ln.intersect(vert_line(x_target))

    # ambas existem: escolher a mais próxima em y ao ponto inicial
    assert guide_low is not None and guide_high is not None

    # distância perpendicular ao ponto
    d_low = abs(guide_low.a * p_start[0] + guide_low.b * p_start[1] + guide_low.c)
    d_high = abs(guide_high.a * p_start[0] + guide_high.b * p_start[1] + guide_high.c)

    # opção simples: escolher a mais próxima
    base = guide_low if d_low <= d_high else guide_high

    ln = parallel_line_through(base, p_start)
    return ln.intersect(vert_line(x_target))


def solve_ground_roll(capture: Dict[str, Any], base_img: Image.Image,
                      pressure_alt_ft: float, oat_c: float, weight_lb: float, headwind_kt: float):
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

    lines = capture["lines"]

    def seg_to_line(key: str) -> LineABC:
        seg = lines[key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))

    # fixed lines
    isa_m15 = seg_to_line("isa_m15")
    isa_0   = seg_to_line("isa")
    isa_p35 = seg_to_line("isa_p35")

    pa0 = seg_to_line("pa_sea_level")
    pa2 = seg_to_line("pa_2000")
    pa4 = seg_to_line("pa_4000")
    pa6 = seg_to_line("pa_6000")
    pa7 = seg_to_line("pa_7000")

    w_ref  = seg_to_line("weight_ref_line")
    z_wind = seg_to_line("wind_ref_zero")

    # optional guides (NEW)
    guide_lines = capture.get("guide_lines", {})
    def read_guide(name: str) -> Optional[LineABC]:
        segs = guide_lines.get(name)
        if not segs:
            return None
        s = segs[0]
        return LineABC.from_points((s["x1"], s["y1"]), (s["x2"], s["y2"]))

    w_low = read_guide("weight_guide_low")
    w_high = read_guide("weight_guide_high")
    v_low = read_guide("wind_guide_low")
    v_high = read_guide("wind_guide_high")

    # left intersection
    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15.0, isa_m15), (0.0, isa_0), (35.0, isa_p35)], dev)
    pa_line  = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA.")

    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    # to weight ref
    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou weight_ref_line.")

    # to weight using guides
    x_weight = x_from_weight_lb(weight_lb)
    p_w = choose_guide_and_intersect(w_low, w_high, p_wref, x_weight)
    if p_w is None:
        # fallback simples: diagonal aproximada
        p_w = (x_weight, p_wref[1])

    # to wind ref
    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou wind_ref_zero.")

    # to wind using guides
    x_wind = x_from_wind(headwind_kt)
    p_wind = choose_guide_and_intersect(v_low, v_high, p_z, x_wind)
    if p_wind is None:
        p_wind = (x_wind, p_z[1])

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
# Drawing
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


def draw_preview(img: Image.Image, points: Dict[str, Tuple[float, float]], gr_round: float) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    draw_arrow(d, points["p_oat"], points["p_left"], width=5)
    draw_arrow(d, points["p_left"], points["p_wref"], width=5)
    draw_arrow(d, points["p_wref"], points["p_w"], width=5)
    draw_arrow(d, points["p_w"], points["p_z"], width=5)
    draw_arrow(d, points["p_z"], points["p_wind"], width=5)
    draw_arrow(d, points["p_wind"], points["p_gr"], width=5)

    def mark(p, r=6, col=(0, 140, 0)):
        x, y = p
        d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=4)

    for k in ["p_oat", "p_left", "p_wref", "p_w", "p_z", "p_wind", "p_gr"]:
        mark(points[k], col=(0, 0, 200) if k == "p_wind" else (0, 140, 0))

    xg, yg = points["p_gr"]
    d.text((xg + 8, yg - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))
    return out


# =========================
# CAPTURE / EDIT JSON helpers
# =========================
def ensure_capture_structure(cap: Dict[str, Any]) -> Dict[str, Any]:
    cap.setdefault("guide_lines", {})
    return cap


def add_segment(cap: Dict[str, Any], key: str, p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
    cap = ensure_capture_structure(cap)
    cap["guide_lines"][key] = [{
        "x1": int(p1[0]), "y1": int(p1[1]),
        "x2": int(p2[0]), "y2": int(p2[1]),
    }]


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Nomogram Tool", layout="wide")
st.title("PA-28 Landing Ground Roll — Captura + Solver")

tab1, tab2 = st.tabs(["1) Capturar/Editar JSON (guias)", "2) Solver (calcular + desenhar)"])

# Load PDF and capture.json
with tab1:
    st.subheader("Ficheiros")
    c1, c2 = st.columns(2)
    with c1:
        pdf_up = st.file_uploader("PDF (opcional; senão usa o da pasta)", type=["pdf"], key="pdf_up_1")
    with c2:
        cap_up = st.file_uploader("capture.json (opcional; senão usa o da pasta)", type=["json"], key="cap_up_1")

    pdf_path = locate_file(PDF_NAME_DEFAULT)
    cap_path = locate_file(CAPTURE_NAME_DEFAULT)

    if pdf_up is not None:
        Path("uploaded.pdf").write_bytes(pdf_up.read())
        pdf_path = "uploaded.pdf"
    if cap_up is not None:
        Path("capture.json").write_bytes(cap_up.read())
        cap_path = "capture.json"

    if not pdf_path or not cap_path:
        st.warning("Faz upload do PDF e do capture.json ou mete-os na mesma pasta do app.py")
        st.stop()

    capture = ensure_capture_structure(json.loads(Path(cap_path).read_text(encoding="utf-8")))
    zoom = float(capture.get("zoom", 2.3))
    base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)

    st.markdown("### Adicionar linhas-guia (2 cliques por linha)")
    st.markdown(
        "- Recomendo criares 2 por painel (weight e wind):\n"
        "  - `weight_guide_low` e `weight_guide_high`\n"
        "  - `wind_guide_low` e `wind_guide_high`\n"
        "- Ordem: **de baixo para cima** (low primeiro, high depois)."
    )

    key = st.selectbox(
        "Qual linha queres capturar?",
        ["weight_guide_low", "weight_guide_high", "wind_guide_low", "wind_guide_high"],
    )

    if "click_points" not in st.session_state:
        st.session_state.click_points = []

    st.info("Clica 2 pontos em cima da linha grossa escolhida. Quando tiver 2 pontos, a linha é guardada.")

    click = streamlit_image_coordinates(base_img, key="img_coords")
    if click is not None:
        st.session_state.click_points.append((int(click["x"]), int(click["y"])))

    # show pending points
    st.write("Pontos selecionados (pendentes):", st.session_state.click_points)

    if len(st.session_state.click_points) >= 2:
        p1 = st.session_state.click_points[-2]
        p2 = st.session_state.click_points[-1]
        add_segment(capture, key, p1, p2)
        st.success(f"Guardado: {key} = {p1} -> {p2}")
        st.session_state.click_points = []

    st.markdown("### Preview das guias no JSON")
    st.json(capture.get("guide_lines", {}))

    out_json = json.dumps(capture, indent=2).encode("utf-8")
    st.download_button("⬇️ Download capture.json atualizado", data=out_json, file_name="capture.json", mime="application/json")


with tab2:
    st.subheader("Solver")
    c1, c2 = st.columns(2)
    with c1:
        pdf_up2 = st.file_uploader("PDF (opcional; senão usa o da pasta)", type=["pdf"], key="pdf_up_2")
    with c2:
        cap_up2 = st.file_uploader("capture.json (opcional; senão usa o da pasta)", type=["json"], key="cap_up_2")

    pdf_path2 = locate_file(PDF_NAME_DEFAULT)
    cap_path2 = locate_file(CAPTURE_NAME_DEFAULT)

    if pdf_up2 is not None:
        Path("uploaded2.pdf").write_bytes(pdf_up2.read())
        pdf_path2 = "uploaded2.pdf"
    if cap_up2 is not None:
        Path("capture2.json").write_bytes(cap_up2.read())
        cap_path2 = "capture2.json"

    if not pdf_path2 or not cap_path2:
        st.warning("Faz upload do PDF e do capture.json ou mete-os na mesma pasta do app.py")
        st.stop()

    capture2 = ensure_capture_structure(json.loads(Path(cap_path2).read_text(encoding="utf-8")))
    zoom2 = float(capture2.get("zoom", 2.3))
    base_img2 = render_pdf_page_to_pil(pdf_path2, zoom=zoom2)

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        pressure_alt_ft = st.number_input("Pressure Altitude (ft)", value=2500.0, step=100.0)
    with i2:
        oat_c = st.number_input("OAT (°C)", value=21.0, step=1.0)
    with i3:
        weight_lb = st.number_input("Weight (lb)", value=2240.0, step=10.0)
    with i4:
        headwind_kt = st.number_input("Headwind (kt)", value=5.0, step=1.0)

    run = st.button("Calcular", key="run_solver")
    if run:
        try:
            gr_raw, gr_round, pts = solve_ground_roll(
                capture2, base_img2,
                float(pressure_alt_ft), float(oat_c), float(weight_lb), float(headwind_kt)
            )
        except Exception as e:
            st.exception(e)
            st.stop()

        st.success(f"Landing Ground Roll ≈ {gr_round:.0f} ft (arredondado 5 em 5)")
        st.caption(f"Bruto: {gr_raw:.1f} ft")

        preview = draw_preview(base_img2, pts, gr_round)
        st.image(preview, use_container_width=True)



