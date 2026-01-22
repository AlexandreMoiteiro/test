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
# CAPTURE STRUCT
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
    capture["guide_lines"][key].append(
        {"x1": float(p1[0]), "y1": float(p1[1]), "x2": float(p2[0]), "y2": float(p2[1])}
    )


def seg_to_line(seg: Dict[str, float]) -> LineABC:
    return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


def choose_nearest_guide(guide_segs: List[Dict[str, float]], p_start: Tuple[float, float]) -> Optional[LineABC]:
    if not guide_segs:
        return None
    lines = [seg_to_line(g) for g in guide_segs]
    return min(lines, key=lambda ln: ln.distance_to_point(p_start))


# =========================
# PANEL BBOX + CROPS
# =========================
def panel_bbox(panel_quad: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in panel_quad]
    ys = [p[1] for p in panel_quad]
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return x0, y0, x1, y1


def get_panel_for_key(capture: Dict[str, Any], key: str) -> str:
    if key == "weight_guides":
        return "middle"
    if key in ("wind_headwind_guides", "wind_tailwind_guides"):
        return "right"
    return "middle"


# =========================
# EDITOR: FIT LINE FROM MANY POINTS
# =========================
def fit_line_pca(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    PCA line fit:
      returns (point_on_line, direction_unit_vector)
    """
    P = np.array(points, dtype=float)
    mu = P.mean(axis=0)
    C = np.cov(P.T)
    w, v = np.linalg.eigh(C)          # eigenvectors
    d = v[:, np.argmax(w)]            # principal direction
    d = d / (np.linalg.norm(d) + 1e-12)
    return (float(mu[0]), float(mu[1])), (float(d[0]), float(d[1]))


def line_rect_intersections(p0: Tuple[float, float], d: Tuple[float, float],
                            rect: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    """
    Interseções da reta p(t)=p0+t*d com o retângulo [x0,x1]x[y0,y1].
    Retorna até 2 pontos (idealmente 2).
    """
    x0, y0, x1, y1 = rect
    px, py = p0
    dx, dy = d
    pts = []

    def add_if_inside(x, y):
        if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1:
            pts.append((float(x), float(y)))

    # Intersect with x = x0, x1
    if abs(dx) > 1e-9:
        t = (x0 - px) / dx
        add_if_inside(x0, py + t * dy)
        t = (x1 - px) / dx
        add_if_inside(x1, py + t * dy)

    # Intersect with y = y0, y1
    if abs(dy) > 1e-9:
        t = (y0 - py) / dy
        add_if_inside(px + t * dx, y0)
        t = (y1 - py) / dy
        add_if_inside(px + t * dx, y1)

    # remover duplicados aproximados
    uniq = []
    for p in pts:
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > 4 for q in uniq):
            uniq.append(p)

    # se houver mais de 2, escolhe os 2 mais afastados
    if len(uniq) <= 2:
        return uniq
    best = None
    best_d = -1
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            dd = (uniq[i][0]-uniq[j][0])**2 + (uniq[i][1]-uniq[j][1])**2
            if dd > best_d:
                best_d = dd
                best = (uniq[i], uniq[j])
    return list(best) if best else uniq[:2]


def overlay_editor_crop(
    crop: Image.Image,
    capture: Dict[str, Any],
    key: str,
    offset: Tuple[int, int],
    sample_pts_crop: List[Tuple[float, float]],
) -> Image.Image:
    out = crop.copy()
    d = ImageDraw.Draw(out)
    ensure_guide_struct(capture)

    ox, oy = offset

    # existing segments for this key (draw in crop coords)
    for seg in capture["guide_lines"][key]:
        x1, y1 = seg["x1"] - ox, seg["y1"] - oy
        x2, y2 = seg["x2"] - ox, seg["y2"] - oy
        d.line([(x1, y1), (x2, y2)], fill=(0, 170, 0), width=4)

    # sample points being clicked (blue dots)
    for i, (x, y) in enumerate(sample_pts_crop):
        r = 5
        d.ellipse((x-r, y-r, x+r, y+r), outline=(0, 0, 255), width=3)
        if i == 0:
            d.text((x+8, y-10), "P", fill=(0, 0, 255))

    return out


# =========================
# SOLVER (igual, usa guias)
# =========================
def robust_fit_x_from_value(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = np.array([t["value"] for t in ticks], dtype=float)
    xs = np.array([t["x"] for t in ticks], dtype=float)
    slopes = []
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            dv = vals[j] - vals[i]
            if abs(dv) < 1e-9:
                continue
            slopes.append((xs[j]-xs[i])/dv)
    m = float(np.median(slopes))
    b = float(np.median(xs - m*vals))
    return m, b


def robust_fit_value_from_y(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    ys = np.array([t["y"] for t in ticks], dtype=float)
    vals = np.array([t["value"] for t in ticks], dtype=float)
    slopes = []
    for i in range(len(ys)):
        for j in range(i+1, len(ys)):
            dy = ys[j] - ys[i]
            if abs(dy) < 1e-9:
                continue
            slopes.append((vals[j]-vals[i])/dy)
    m = float(np.median(slopes))
    b = float(np.median(vals - m*ys))
    return m, b


def std_atm_isa_temp_c(pressure_alt_ft: float) -> float:
    return 15.0 - 1.98 * (pressure_alt_ft / 1000.0)


def interpolate_line(pairs: List[Tuple[float, LineABC]], value: float) -> LineABC:
    pairs = sorted(pairs, key=lambda t: t[0])
    values = [v for v, _ in pairs]
    v = clamp(value, values[0], values[-1])
    for i in range(len(pairs)-1):
        v1, l1 = pairs[i]
        v2, l2 = pairs[i+1]
        if v1 <= v <= v2:
            t = 0.0 if v2 == v1 else (v - v1) / (v2 - v1)
            a = (1-t)*l1.a + t*l2.a
            b = (1-t)*l1.b + t*l2.b
            c = (1-t)*l1.c + t*l2.c
            n = float(np.hypot(a,b)) + 1e-12
            return LineABC(a/n, b/n, c/n)
    return pairs[-1][1]


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


def solve_with_guides(
    capture: Dict[str, Any],
    base_img: Image.Image,
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    wind_kt_signed: float,
) -> Tuple[float, float, List[Tuple[float, float]]]:

    ensure_guide_struct(capture)

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

    def x_from_oat(v): return mo*v + bo
    def x_from_weight_lb(v): return mw*(v/100.0) + bw
    def x_from_wind(v): return mwd*v + bwd
    def gr_from_y(y): return mgr*y + bgr

    def get_line(key: str) -> LineABC:
        seg = capture["lines"][key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))

    isa_m15 = get_line("isa_m15")
    isa_0   = get_line("isa")
    isa_p35 = get_line("isa_p35")

    pa0 = get_line("pa_sea_level")
    pa2 = get_line("pa_2000")
    pa4 = get_line("pa_4000")
    pa6 = get_line("pa_6000")
    pa7 = get_line("pa_7000")

    w_ref  = get_line("weight_ref_line")
    z_wind = get_line("wind_ref_zero")

    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15.0, isa_m15), (0.0, isa_0), (35.0, isa_p35)], dev)
    pa_line  = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou ISA/PA")

    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou weight_ref_line")

    guide_w = choose_nearest_guide(capture["guide_lines"]["weight_guides"], p_wref)
    xw = float(x_from_weight_lb(weight_lb))
    if guide_w is None:
        p_w = (xw, p_wref[1])
    else:
        ln = parallel_line_through(guide_w, p_wref)
        p_w = ln.intersect(vert_line(xw)) or (xw, p_wref[1])

    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou wind_ref_zero")

    wind_mag = abs(float(wind_kt_signed))
    wind_key = "wind_headwind_guides" if wind_kt_signed >= 0 else "wind_tailwind_guides"
    guide_wind = choose_nearest_guide(capture["guide_lines"][wind_key], p_z)
    xwind = float(x_from_wind(wind_mag))

    if guide_wind is None:
        p_wind = (xwind, p_z[1])
    else:
        ln = parallel_line_through(guide_wind, p_z)
        p_wind = ln.intersect(vert_line(xwind)) or (xwind, p_z[1])

    gr_raw = float(gr_from_y(p_wind[1]))
    gr_round = float(5.0 * round(gr_raw / 5.0))
    p_gr = (gr_axis_x, p_wind[1])

    path = [p_oat, p_left, p_wref, p_w, p_z, p_wind, p_gr]
    return gr_raw, gr_round, path


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Nomogram Editor + Solver", layout="wide")
st.title("Nomograma — Editor (fit por pontos) + Solver (Headwind/Tailwind)")

colA, colB = st.columns(2)
with colA:
    pdf_up = st.file_uploader("PDF (opcional)", type=["pdf"])
with colB:
    cap_up = st.file_uploader("capture.json (opcional)", type=["json"])

pdf_path = locate_file(PDF_NAME_DEFAULT)
cap_path = locate_file(CAPTURE_NAME_DEFAULT)

if pdf_up is not None:
    Path("uploaded.pdf").write_bytes(pdf_up.read())
    pdf_path = str(Path("uploaded.pdf"))

if not pdf_path:
    st.error(f"Não encontrei '{PDF_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

# ---- capture in session state (persist) ----
if "capture" not in st.session_state:
    if cap_up is not None:
        st.session_state.capture = json.loads(cap_up.read().decode("utf-8"))
    elif cap_path and Path(cap_path).exists():
        st.session_state.capture = json.loads(Path(cap_path).read_text(encoding="utf-8"))
    else:
        st.session_state.capture = {"zoom": 2.3, "lines": {}, "axis_ticks": {}, "panel_corners": {}}
    ensure_guide_struct(st.session_state.capture)
else:
    if cap_up is not None:
        st.session_state.capture = json.loads(cap_up.read().decode("utf-8"))
        ensure_guide_struct(st.session_state.capture)

capture = st.session_state.capture
ensure_guide_struct(capture)

zoom = float(capture.get("zoom", 2.3))
base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)

mode = st.radio("Modo", ["Editor de guias (JSON)", "Solver"], horizontal=True)

# editor state: points for current line (in crop coords)
if "sample_pts" not in st.session_state:
    st.session_state.sample_pts = []

if mode == "Editor de guias (JSON)":
    st.subheader("Editor (melhor): clica vários pontos na guia e depois 'Criar linha (fit)'")

    key = st.selectbox("Conjunto",
                       ["weight_guides", "wind_headwind_guides", "wind_tailwind_guides"],
                       help="WEIGHT= painel do meio; WIND= painel direito (headwind e tailwind separados)")

    # choose panel bbox for this key
    panel_name = get_panel_for_key(capture, key)
    if "panel_corners" not in capture or panel_name not in capture["panel_corners"]:
        st.error("Faltam panel_corners no capture.json (preciso de left/middle/right).")
        st.stop()

    x0, y0, x1, y1 = panel_bbox(capture["panel_corners"][panel_name])
    crop = base_img.crop((x0, y0, x1, y1))

    zoom_editor = st.slider("Zoom do editor", 1.0, 4.0, 2.0, 0.1)

    # overlay + resize for easier clicking
    img_overlay = overlay_editor_crop(
        crop=crop,
        capture=capture,
        key=key,
        offset=(x0, y0),
        sample_pts_crop=st.session_state.sample_pts,
    )

    if zoom_editor != 1.0:
        w, h = img_overlay.size
        img_overlay = img_overlay.resize((int(w * zoom_editor), int(h * zoom_editor)), Image.NEAREST)

    click = streamlit_image_coordinates(np.array(img_overlay), key=f"coords_{key}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("Pontos na linha:", len(st.session_state.sample_pts))
    with c2:
        if st.button("Undo ponto"):
            if st.session_state.sample_pts:
                st.session_state.sample_pts.pop()
                st.rerun()
    with c3:
        if st.button("Limpar pontos"):
            st.session_state.sample_pts = []
            st.rerun()
    with c4:
        if st.button("Apagar última linha guardada"):
            if capture["guide_lines"][key]:
                capture["guide_lines"][key].pop()
            st.rerun()

    if click is not None:
        # map click back from zoomed overlay into crop coords
        cx = float(click["x"]) / float(zoom_editor)
        cy = float(click["y"]) / float(zoom_editor)
        st.session_state.sample_pts.append((cx, cy))
        st.rerun()

    # Create fitted line
    st.markdown("### Criar linha (fit)")
    st.caption("Clica 4–10 pontos ao longo da guia grossa. Depois carrega em 'Criar linha (fit)'.")
    if st.button("✅ Criar linha (fit) a partir dos pontos"):
        if len(st.session_state.sample_pts) < 2:
            st.warning("Precisas de pelo menos 2 pontos (recomendo 4+).")
        else:
            # fit on CROP coords
            p0, d = fit_line_pca(st.session_state.sample_pts)

            # find intersections with crop rect
            rect = (0.0, 0.0, float(crop.size[0]-1), float(crop.size[1]-1))
            inter = line_rect_intersections(p0, d, rect)

            if len(inter) < 2:
                st.warning("Não consegui esticar a linha até às bordas (tenta mais pontos).")
            else:
                pA, pB = inter[0], inter[1]

                # convert to GLOBAL coords
                gA = (pA[0] + x0, pA[1] + y0)
                gB = (pB[0] + x0, pB[1] + y0)

                add_segment(capture, key, gA, gB)
                st.session_state.sample_pts = []
                st.rerun()

    st.write("Contagens:", {k: len(capture["guide_lines"][k]) for k in capture["guide_lines"]})
    st.download_button(
        "⬇️ Download capture.json atualizado",
        data=json.dumps(capture, indent=2).encode("utf-8"),
        file_name="capture.json",
        mime="application/json",
    )

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

        img = base_img.copy()
        d = ImageDraw.Draw(img)
        for a, b in zip(path[:-1], path[1:]):
            draw_arrow(d, a, b, width=5)
        x, y = path[-1]
        d.text((x + 8, y - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))

        st.image(img, use_container_width=True)

