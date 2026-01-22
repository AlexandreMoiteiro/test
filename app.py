from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import pymupdf
from PIL import Image, ImageDraw

import cv2  # opencv-python-headless


# =========================
# CONFIG
# =========================
PDF_NAME_DEFAULT = "PA28POH-ground-roll.pdf"
CAPTURE_NAME_DEFAULT = "capture.json"


# =========================
# GEOMETRY PRIMITIVES
# =========================
@dataclass(frozen=True)
class LineABC:
    """Reta na forma a*x + b*y + c = 0, com (a,b) normalizado."""
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


def line_through_point_with_angle(p: Tuple[float, float], theta_rad: float) -> LineABC:
    x0, y0 = p
    dx = float(np.cos(theta_rad))
    dy = float(np.sin(theta_rad))
    p2 = (x0 + dx * 100.0, y0 + dy * 100.0)
    return LineABC.from_points((x0, y0), p2)


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
# CALIBRATION
# =========================
def fit_linear_x_from_value(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    """Ajusta x ≈ m*value + b."""
    vals = np.array([t["value"] for t in ticks], dtype=float)
    xs = np.array([t["x"] for t in ticks], dtype=float)
    A = np.vstack([vals, np.ones_like(vals)]).T
    m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
    return float(m), float(b)


def fit_linear_value_from_y(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    """Ajusta value ≈ m*y + b (para ground roll vs y)."""
    ys = np.array([t["y"] for t in ticks], dtype=float)
    vals = np.array([t["value"] for t in ticks], dtype=float)
    A = np.vstack([ys, np.ones_like(ys)]).T
    m, b = np.linalg.lstsq(A, vals, rcond=None)[0]
    return float(m), float(b)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# GUIDES (Hough) — “linhas do piloto”
# =========================
def _crop_panel(img: Image.Image, panel_quad: List[List[float]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    xs = [p[0] for p in panel_quad]
    ys = [p[1] for p in panel_quad]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
    arr = np.asarray(crop)
    return arr, (x0, y0, x1, y1)


def _line_distance_to_point(line: LineABC, p: Tuple[float, float]) -> float:
    # line normalizado => distância = |ax+by+c|
    return abs(line.a * p[0] + line.b * p[1] + line.c)


def detect_guide_lines(img: Image.Image, panel_quad: List[List[float]]) -> List[LineABC]:
    """
    Deteta as diagonais grossas (linhas-guia) num painel via Hough.
    Retorna linhas em coords globais da imagem (a*x+b*y+c=0).
    """
    arr, (x0, y0, x1, y1) = _crop_panel(img, panel_quad)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Realçar as linhas grossas:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180, apertureSize=3)

    # Hough standard
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return []

    guides: List[LineABC] = []
    # limitar quantidade
    for rho_theta in lines[:250]:
        rho, theta = float(rho_theta[0][0]), float(rho_theta[0][1])

        # Equação Hough: x*cosθ + y*sinθ = rho  => a*x + b*y + c=0 com a=cosθ, b=sinθ, c=-rho
        a = float(np.cos(theta))
        b = float(np.sin(theta))
        c = -rho

        # rejeitar quase horizontais/verticais
        ang = theta % np.pi
        if abs(ang - 0) < np.deg2rad(10) or abs(ang - np.pi / 2) < np.deg2rad(10):
            continue

        # converter para coords globais
        # a*(x-x0)+b*(y-y0)+c=0 => a*x + b*y + (c - a*x0 - b*y0)=0
        c_global = c - a * x0 - b * y0

        n = float(np.hypot(a, b))
        if n == 0:
            continue
        guides.append(LineABC(a / n, b / n, c_global / n))

    # Dedup: há muitas repetidas
    dedup: List[LineABC] = []
    for ln in guides:
        dup = False
        for ex in dedup:
            # praticamente paralelas e offset parecido
            if abs(ln.a - ex.a) < 0.02 and abs(ln.b - ex.b) < 0.02 and abs(ln.c - ex.c) < 12:
                dup = True
                break
        if not dup:
            dedup.append(ln)

    return dedup


def snap_to_nearest_guide(guides: List[LineABC], p: Tuple[float, float]) -> Optional[LineABC]:
    if not guides:
        return None
    return min(guides, key=lambda ln: _line_distance_to_point(ln, p))


def estimate_dominant_diagonal_angle_fallback(img: Image.Image, panel_quad: List[List[float]]) -> float:
    """
    Fallback caso o Hough falhe. Estima ângulo dominante por gradientes.
    """
    arr, (x0, y0, x1, y1) = _crop_panel(img, panel_quad)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0

    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang_g = np.arctan2(gy, gx)
    ang_line = ang_g + np.pi / 2.0
    ang_line = (ang_line + np.pi / 2) % np.pi - np.pi / 2

    m = mag > np.percentile(mag, 85)
    ang = ang_line[m]
    w = mag[m]

    if ang.size < 200:
        return float(np.deg2rad(45.0))

    mask_diag = (np.abs(ang) > np.deg2rad(15)) & (np.abs(ang) < np.deg2rad(75))
    ang = ang[mask_diag]
    w = w[mask_diag]
    if ang.size < 200:
        return float(np.deg2rad(45.0))

    hist, edges = np.histogram(ang, bins=180, range=(-np.pi / 2, np.pi / 2), weights=w)
    idx = int(np.argmax(hist))
    theta = float((edges[idx] + edges[idx + 1]) / 2)

    # orientar para dx>0
    dx = np.cos(theta)
    if dx < 0:
        theta += np.pi
    return float(theta)


# =========================
# NOMOGRAM CORE
# =========================
def interpolate_line(pairs: List[Tuple[float, LineABC]], value: float) -> LineABC:
    """
    Interpola entre linhas por value (interpolação em (a,b,c) com renormalização).
    """
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


def std_atm_isa_temp_c(pressure_alt_ft: float) -> float:
    """ISA temp aproximada: 15 - 1.98°C por 1000 ft."""
    return 15.0 - 1.98 * (pressure_alt_ft / 1000.0)


def solve_landing_ground_roll(
    capture: Dict[str, Any],
    base_img: Image.Image,
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    headwind_kt: float,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """
    Retorna:
      - ground_roll_raw
      - ground_roll_rounded_to_5
      - pontos do percurso (para desenhar)
    """
    ticks_oat = capture["axis_ticks"]["oat_c"]
    ticks_wt = capture["axis_ticks"]["weight_x100_lb"]
    ticks_wnd = capture["axis_ticks"]["wind_kt"]
    ticks_gr = capture["axis_ticks"]["ground_roll_ft"]

    mo, bo = fit_linear_x_from_value(ticks_oat)       # x = mo*oat + bo
    mw, bw = fit_linear_x_from_value(ticks_wt)        # x = mw*(w/100) + bw
    mwd, bwd = fit_linear_x_from_value(ticks_wnd)     # x = mwd*wind + bwd
    mgr, bgr = fit_linear_value_from_y(ticks_gr)      # gr = mgr*y + bgr

    def x_from_weight_lb(w_lb: float) -> float:
        return mw * (w_lb / 100.0) + bw

    def x_from_wind(w: float) -> float:
        return mwd * w + bwd

    def gr_from_y(y: float) -> float:
        return mgr * y + bgr

    lines = capture["lines"]

    def seg_to_line(key: str) -> LineABC:
        seg = lines[key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))

    # ISA family (-15, 0, +35)
    line_isa_m15 = seg_to_line("isa_m15")
    line_isa_0 = seg_to_line("isa")
    line_isa_p35 = seg_to_line("isa_p35")

    # PA family
    pa0 = seg_to_line("pa_sea_level")
    pa2 = seg_to_line("pa_2000")
    pa4 = seg_to_line("pa_4000")
    pa6 = seg_to_line("pa_6000")
    pa7 = seg_to_line("pa_7000")

    # references
    w_ref = seg_to_line("weight_ref_line")
    z_wind = seg_to_line("wind_ref_zero")

    # --- left panel (ISA deviation line + PA line) ---
    isa_temp = std_atm_isa_temp_c(pressure_alt_ft)
    dev = oat_c - isa_temp  # OAT - ISA

    isa_line = interpolate_line(
        [(-15.0, line_isa_m15), (0.0, line_isa_0), (35.0, line_isa_p35)],
        dev,
    )
    pa_line = interpolate_line(
        [(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)],
        pressure_alt_ft,
    )

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA (linhas quase paralelas).")

    # horizontal from left point to weight ref line
    y_left = p_left[1]
    horiz = LineABC(0.0, 1.0, -y_left)
    p_wref = w_ref.intersect(horiz)
    if p_wref is None:
        raise ValueError("Falhou interseção com weight_ref_line.")

    # --- middle panel: snap to nearest guide line ---
    guides_mid = detect_guide_lines(base_img, capture["panel_corners"]["middle"])
    snap_mid = snap_to_nearest_guide(guides_mid, p_wref)
    if snap_mid is None:
        theta_mid = estimate_dominant_diagonal_angle_fallback(base_img, capture["panel_corners"]["middle"])
        snap_mid = line_through_point_with_angle(p_wref, theta_mid)

    x_weight = x_from_weight_lb(weight_lb)
    vert_w = LineABC(1.0, 0.0, -x_weight)
    p_w = snap_mid.intersect(vert_w)
    if p_w is None:
        raise ValueError("Falhou projeção para o peso (interseção no painel middle).")

    # horizontal to wind zero line
    y_w = p_w[1]
    horiz2 = LineABC(0.0, 1.0, -y_w)
    p_z = z_wind.intersect(horiz2)
    if p_z is None:
        raise ValueError("Falhou interseção com wind_ref_zero.")

    # --- right panel: snap to nearest guide line ---
    guides_r = detect_guide_lines(base_img, capture["panel_corners"]["right"])
    snap_r = snap_to_nearest_guide(guides_r, p_z)
    if snap_r is None:
        theta_r = estimate_dominant_diagonal_angle_fallback(base_img, capture["panel_corners"]["right"])
        snap_r = line_through_point_with_angle(p_z, theta_r)

    x_wind = x_from_wind(headwind_kt)
    vert_wind = LineABC(1.0, 0.0, -x_wind)
    p_wind = snap_r.intersect(vert_wind)
    if p_wind is None:
        raise ValueError("Falhou projeção para o vento (interseção no painel right).")

    ground_roll_raw = float(gr_from_y(p_wind[1]))
    ground_roll_rounded = float(5.0 * round(ground_roll_raw / 5.0))

    points = {
        "p_left": p_left,
        "p_wref": p_wref,
        "p_w": p_w,
        "p_z": p_z,
        "p_wind": p_wind,
    }
    return ground_roll_raw, ground_roll_rounded, points


# =========================
# DRAWING (SETAS)
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

    ah = 18.0
    aw = 10.0
    bx, by = x2 - ah * ux, y2 - ah * uy
    px, py = -uy, ux

    pA = (x2, y2)
    pB = (bx + aw * px, by + aw * py)
    pC = (bx - aw * px, by - aw * py)
    draw.polygon([pA, pB, pC], fill=(255, 0, 0))


def draw_path_preview(img: Image.Image, points: Dict[str, Tuple[float, float]]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    order = ["p_left", "p_wref", "p_w", "p_z", "p_wind"]
    for a, b in zip(order[:-1], order[1:]):
        draw_arrow_pil(d, points[a], points[b], width=5)

    # marcar pontos
    def mark(p, color, r=6):
        x, y = p
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)

    for name in order:
        p = points[name]
        mark(p, (0, 120, 0) if name != "p_wind" else (0, 0, 200))
        d.text((p[0] + 8, p[1] - 12), name, fill=(0, 0, 0))

    return out


def draw_arrow_pdf(page, p1: pymupdf.Point, p2: pymupdf.Point, width=2.5):
    page.draw_line(p1, p2, color=(1, 0, 0), width=width)

    vx = p2.x - p1.x
    vy = p2.y - p1.y
    L = float(np.hypot(vx, vy))
    if L < 1e-6:
        return
    ux, uy = vx / L, vy / L

    ah = 10.0
    aw = 6.0
    bx, by = p2.x - ah * ux, p2.y - ah * uy
    px, py = -uy, ux

    tri = [
        p2,
        pymupdf.Point(bx + aw * px, by + aw * py),
        pymupdf.Point(bx - aw * px, by - aw * py),
    ]
    page.draw_polygon(tri, color=(1, 0, 0), fill=(1, 0, 0))


def add_path_to_pdf(pdf_path: str, zoom: float, points: Dict[str, Tuple[float, float]]) -> bytes:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect

    # render size at zoom
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    w_img, h_img = pix.width, pix.height

    sx = rect.width / w_img
    sy = rect.height / h_img

    def map_pt(p):
        return pymupdf.Point(p[0] * sx, p[1] * sy)

    order = ["p_left", "p_wref", "p_w", "p_z", "p_wind"]
    for a, b in zip(order[:-1], order[1:]):
        draw_arrow_pdf(page, map_pt(points[a]), map_pt(points[b]), width=2.5)

    # pontos
    for k in order:
        page.draw_circle(map_pt(points[k]), radius=3.0, color=(0, 0.6, 0), width=2.0)

    out = doc.tobytes()
    doc.close()
    return out


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Landing Ground Roll Solver", layout="wide")
st.title("Landing Ground Roll — Solver do nomograma (snap às guias + setas)")

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
if not cap_path:
    st.error(f"Não encontrei '{CAPTURE_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

capture = json.loads(Path(cap_path).read_text(encoding="utf-8"))
zoom = float(capture.get("zoom", 2.3))

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

show_raw = st.checkbox("Mostrar valor bruto (antes de arredondar)", value=False)

run = st.button("Calcular")

if run:
    base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)

    try:
        gr_raw, gr_rounded, points = solve_landing_ground_roll(
            capture=capture,
            base_img=base_img,
            pressure_alt_ft=float(pressure_alt_ft),
            oat_c=float(oat_c),
            weight_lb=float(weight_lb),
            headwind_kt=float(headwind_kt),
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success(f"Landing Ground Roll ≈ **{gr_rounded:.0f} ft**  (arredondado de 5 em 5)")
    if show_raw:
        st.caption(f"Bruto: {gr_raw:.1f} ft")

    debug_img = draw_path_preview(base_img, points)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.subheader("Debug visual (trajetória com setas)")
        st.image(debug_img, use_container_width=True)
    with col2:
        st.subheader("Pontos (pixels)")
        st.json({k: {"x": v[0], "y": v[1]} for k, v in points.items()})

        pdf_out = add_path_to_pdf(pdf_path, zoom=zoom, points=points)
        st.download_button(
            "⬇️ Download PDF com trajetória (setas)",
            data=pdf_out,
            file_name="landing_ground_roll_debug_arrows.pdf",
            mime="application/pdf",
        )

st.caption("Se a trajetória ainda não estiver 100% colada às guias, ajustamos só 2 parâmetros do Hough (threshold e Canny).")


