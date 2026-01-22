from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import pymupdf
from PIL import Image, ImageDraw


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
    """Reta na forma a*x + b*y + c = 0"""
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
        # normalizar para estabilidade
        n = np.hypot(a, b)
        if n == 0:
            return LineABC(0.0, 0.0, 0.0)
        return LineABC(a / n, b / n, c / n)

    def intersect(self, other: "LineABC") -> Optional[Tuple[float, float]]:
        # resolve:
        # a1 x + b1 y + c1 = 0
        # a2 x + b2 y + c2 = 0
        det = self.a * other.b - other.a * self.b
        if abs(det) < 1e-9:
            return None
        x = (self.b * other.c - other.b * self.c) / det
        y = (other.a * self.c - self.a * other.c) / det
        return (float(x), float(y))


def line_through_point_with_angle(p: Tuple[float, float], theta_rad: float) -> LineABC:
    """Reta que passa em p com direção theta (rad)."""
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
# IMAGE-BASED DIAGONAL ANGLE ESTIMATION
# =========================
def estimate_dominant_diagonal_angle(img: Image.Image, panel_quad: List[List[float]]) -> float:
    """
    Estima o ângulo dominante das diagonais grossas dentro do painel
    usando histograma de orientação de gradientes.

    Retorna theta em rad, escolhido perto de uma diagonal (não horizontal/vertical),
    e orientado para "descer para a direita" (slope positivo em coord. imagem? atenção: y cresce para baixo).
    """
    # recorta bounding box do painel
    xs = [p[0] for p in panel_quad]
    ys = [p[1] for p in panel_quad]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    crop = img.crop((x0, y0, x1, y1)).convert("L")
    arr = np.asarray(crop, dtype=np.float32) / 255.0

    # suavização leve (box blur simples)
    k = 3
    arr_blur = np.copy(arr)
    arr_blur[1:-1, 1:-1] = (
        arr[:-2, :-2] + arr[:-2, 1:-1] + arr[:-2, 2:] +
        arr[1:-1, :-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
        arr[2:, :-2] + arr[2:, 1:-1] + arr[2:, 2:]
    ) / 9.0

    # gradientes
    gy, gx = np.gradient(arr_blur)
    mag = np.hypot(gx, gy)

    # ângulo do gradiente (perp à linha); queremos orientação da LINHA,
    # então somamos 90° (pi/2)
    ang_g = np.arctan2(gy, gx)
    ang_line = ang_g + np.pi / 2.0

    # normalizar para [-pi/2, pi/2] (linhas sem direção)
    ang_line = (ang_line + np.pi/2) % np.pi - np.pi/2

    # rejeitar magnitudes pequenas
    m = mag > np.percentile(mag, 85)

    ang = ang_line[m]
    w = mag[m]

    if ang.size < 100:
        # fallback: uma diagonal típica (-35° a -55° em coords de imagem),
        # mas como y cresce para baixo, "descer para a direita" é theta ~ +35°..+55°
        return float(np.deg2rad(45.0))

    # histograma focado em diagonais (exclui quase horizontais/verticais)
    # queremos |ang| entre 15° e 75°
    mask_diag = (np.abs(ang) > np.deg2rad(15)) & (np.abs(ang) < np.deg2rad(75))
    ang = ang[mask_diag]
    w = w[mask_diag]
    if ang.size < 100:
        return float(np.deg2rad(45.0))

    bins = 180
    hist, edges = np.histogram(ang, bins=bins, range=(-np.pi/2, np.pi/2), weights=w)
    idx = int(np.argmax(hist))
    theta = (edges[idx] + edges[idx+1]) / 2.0

    # queremos direção que "desce para a direita": dx>0 e dy>0 em coords de imagem
    # se theta estiver a apontar para "subir", inverte (adiciona pi)
    dx = np.cos(theta)
    dy = np.sin(theta)
    if dx < 0:
        theta += np.pi
        dx = np.cos(theta)
        dy = np.sin(theta)
    if dy < 0:
        # virar 180°
        theta += np.pi
    # reduzir para [-pi, pi] não é necessário
    return float(theta)


# =========================
# NOMOGRAM SOLVER
# =========================
def interpolate_line(pairs: List[Tuple[float, LineABC]], value: float) -> LineABC:
    """
    Interpola entre linhas por 'value'. Cada par é (value_i, line_i).
    Retorna uma reta interpolada em (a,b,c) (funciona bem porque todas são quase paralelas).
    """
    pairs = sorted(pairs, key=lambda t: t[0])
    values = [v for v, _ in pairs]
    vmin, vmax = values[0], values[-1]
    v = clamp(value, vmin, vmax)

    # encontrar intervalo
    for i in range(len(pairs) - 1):
        v1, l1 = pairs[i]
        v2, l2 = pairs[i + 1]
        if v1 <= v <= v2:
            t = 0.0 if v2 == v1 else (v - v1) / (v2 - v1)
            a = (1 - t) * l1.a + t * l2.a
            b = (1 - t) * l1.b + t * l2.b
            c = (1 - t) * l1.c + t * l2.c
            # renormalizar
            n = np.hypot(a, b)
            if n == 0:
                return LineABC(0.0, 0.0, 0.0)
            return LineABC(float(a / n), float(b / n), float(c / n))
    # fallback
    return pairs[-1][1]


def std_atm_isa_temp_c(pressure_alt_ft: float) -> float:
    """
    ISA temperatura aproximada (troposfera): 15 - 1.98°C por 1000 ft.
    """
    return 15.0 - 1.98 * (pressure_alt_ft / 1000.0)


def solve_landing_ground_roll(
    capture: Dict[str, Any],
    base_img: Image.Image,
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    headwind_kt: float,
) -> Tuple[float, Dict[str, Tuple[float, float]], float, float]:
    """
    Retorna:
      - ground_roll_ft
      - pontos do percurso (para desenhar)
      - theta_middle (rad)
      - theta_right  (rad)
    """
    zoom = float(capture["zoom"])

    # --- calibrations from ticks ---
    ticks_oat = capture["axis_ticks"]["oat_c"]
    ticks_wt  = capture["axis_ticks"]["weight_x100_lb"]
    ticks_wnd = capture["axis_ticks"]["wind_kt"]
    ticks_gr  = capture["axis_ticks"]["ground_roll_ft"]

    mo, bo = fit_linear_x_from_value(ticks_oat)      # x = mo*oat + bo
    mw, bw = fit_linear_x_from_value(ticks_wt)       # x = mw*w + bw (w em "x100 lb")
    mwd, bwd = fit_linear_x_from_value(ticks_wnd)    # x = mwd*wind + bwd
    mgr, bgr = fit_linear_value_from_y(ticks_gr)     # gr = mgr*y + bgr

    def x_from_oat(oat: float) -> float:
        return mo * oat + bo

    def x_from_weight_lb(w_lb: float) -> float:
        return mw * (w_lb / 100.0) + bw

    def x_from_wind(w: float) -> float:
        return mwd * w + bwd

    def gr_from_y(y: float) -> float:
        return mgr * y + bgr

    # --- get fixed lines from capture ---
    lines = capture["lines"]

    def seg_to_line(key: str) -> LineABC:
        seg = lines[key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))

    # ISA family is captured: -15, 0, +35
    line_isa_m15 = seg_to_line("isa_m15")
    line_isa_0   = seg_to_line("isa")
    line_isa_p35 = seg_to_line("isa_p35")

    # PA family
    pa0 = seg_to_line("pa_sea_level")
    pa2 = seg_to_line("pa_2000")
    pa4 = seg_to_line("pa_4000")
    pa6 = seg_to_line("pa_6000")
    pa7 = seg_to_line("pa_7000")

    # refs
    w_ref = seg_to_line("weight_ref_line")
    z_wind = seg_to_line("wind_ref_zero")

    # --- choose ISA deviation line based on ISA temp ---
    isa_temp = std_atm_isa_temp_c(pressure_alt_ft)
    dev = oat_c - isa_temp  # OAT - ISA

    isa_line = interpolate_line(
        [
            (-15.0, line_isa_m15),
            (0.0, line_isa_0),
            (35.0, line_isa_p35),
        ],
        dev
    )

    pa_line = interpolate_line(
        [
            (0.0, pa0),
            (2000.0, pa2),
            (4000.0, pa4),
            (6000.0, pa6),
            (7000.0, pa7),
        ],
        pressure_alt_ft
    )

    # --- left-panel result point: intersection(ISA_dev_line, PA_line) ---
    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Não consegui intersectar ISA e PA (linhas paralelas?).")

    # --- horizontal transfer to weight ref line ---
    y_left = p_left[1]
    # horizontal line: y = y_left => 0*x + 1*y - y_left = 0
    horiz = LineABC(0.0, 1.0, -y_left)
    p_wref = w_ref.intersect(horiz)
    if p_wref is None:
        raise ValueError("Não consegui intersectar com a weight_ref_line.")

    # --- middle panel: use dominant diagonal direction to project to weight x ---
    theta_middle = estimate_dominant_diagonal_angle(base_img, capture["panel_corners"]["middle"])
    diag_mid = line_through_point_with_angle(p_wref, theta_middle)

    x_weight = x_from_weight_lb(weight_lb)
    # vertical line at x_weight
    vert_w = LineABC(1.0, 0.0, -x_weight)
    p_w = diag_mid.intersect(vert_w)
    if p_w is None:
        raise ValueError("Não consegui projetar para o peso (interseção falhou).")

    # --- horizontal transfer to wind zero ref line ---
    y_w = p_w[1]
    horiz2 = LineABC(0.0, 1.0, -y_w)
    p_z = z_wind.intersect(horiz2)
    if p_z is None:
        raise ValueError("Não consegui intersectar com a wind_ref_zero.")

    # --- right panel: same idea with dominant diagonal direction ---
    theta_right = estimate_dominant_diagonal_angle(base_img, capture["panel_corners"]["right"])
    diag_r = line_through_point_with_angle(p_z, theta_right)

    x_wind = x_from_wind(headwind_kt)
    vert_wind = LineABC(1.0, 0.0, -x_wind)
    p_wind = diag_r.intersect(vert_wind)
    if p_wind is None:
        raise ValueError("Não consegui projetar para o vento (interseção falhou).")

    # --- read ground roll from y ---
    ground_roll = gr_from_y(p_wind[1])

    points = {
        "p_left": p_left,
        "p_wref": p_wref,
        "p_w": p_w,
        "p_z": p_z,
        "p_wind": p_wind,
    }

    return float(ground_roll), points, float(theta_middle), float(theta_right)


# =========================
# DRAWING
# =========================
def draw_path(img: Image.Image, points: Dict[str, Tuple[float, float]]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    def mark(p, color, r=6):
        x, y = p
        d.ellipse((x-r, y-r, x+r, y+r), outline=color, width=4)

    # segments: left -> wref (horizontal), wref -> w (diagonal), w -> z (horizontal), z -> wind (diagonal)
    p_left = points["p_left"]
    p_wref = points["p_wref"]
    p_w = points["p_w"]
    p_z = points["p_z"]
    p_wind = points["p_wind"]

    d.line([p_left, p_wref], fill=(255, 0, 0), width=4)
    d.line([p_wref, p_w], fill=(255, 0, 0), width=4)
    d.line([p_w, p_z], fill=(255, 0, 0), width=4)
    d.line([p_z, p_wind], fill=(255, 0, 0), width=4)

    for name, p in points.items():
        mark(p, (0, 0, 255) if name == "p_wind" else (0, 200, 0))
        d.text((p[0] + 8, p[1] - 10), name, fill=(0, 0, 0))

    return out


def add_path_to_pdf(pdf_path: str, zoom: float, points: Dict[str, Tuple[float, float]]) -> bytes:
    """
    Desenha a trajetória como vetores diretamente no PDF.
    Converte coords da imagem renderizada (zoom) para coords da página.
    """
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect

    # tamanho real da render no zoom usado
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    w_img, h_img = pix.width, pix.height

    sx = rect.width / w_img
    sy = rect.height / h_img

    def map_pt(p):
        return pymupdf.Point(p[0] * sx, p[1] * sy)

    # linhas
    order = ["p_left", "p_wref", "p_w", "p_z", "p_wind"]
    for a, b in zip(order[:-1], order[1:]):
        page.draw_line(map_pt(points[a]), map_pt(points[b]), color=(1, 0, 0), width=2.5)

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
st.title("Landing Ground Roll — Solver do nomograma (com debug visual)")

# Load PDF and capture.json either from disk or upload
colA, colB = st.columns(2)
with colA:
    pdf_up = st.file_uploader("PDF (opcional, senão usa o da pasta)", type=["pdf"])
with colB:
    cap_up = st.file_uploader("capture.json (opcional, senão usa o da pasta)", type=["json"])

pdf_path = locate_file(PDF_NAME_DEFAULT)
cap_path = locate_file(CAPTURE_NAME_DEFAULT)

if pdf_up is not None:
    pdf_bytes = pdf_up.read()
    pdf_path = str(Path("uploaded.pdf"))
    Path(pdf_path).write_bytes(pdf_bytes)

if cap_up is not None:
    cap_bytes = cap_up.read()
    cap_path = str(Path("capture.json"))
    Path(cap_path).write_bytes(cap_bytes)

if not pdf_path:
    st.error(f"Não encontrei '{PDF_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

if not cap_path:
    st.error(f"Não encontrei '{CAPTURE_NAME_DEFAULT}' e não fizeste upload.")
    st.stop()

capture = json.loads(Path(cap_path).read_text(encoding="utf-8"))

# Inputs
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

# Render base image at the same zoom as capture.json (important!)
zoom = float(capture.get("zoom", 2.3))
base_img = render_pdf_page_to_pil(pdf_path, zoom=zoom)

run = st.button("Calcular")

if run:
    try:
        gr, points, th_mid, th_r = solve_landing_ground_roll(
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

    st.success(f"Landing Ground Roll ≈ **{gr:.0f} ft**")

    st.caption(f"Ângulo dominante (middle) ≈ {np.rad2deg(th_mid):.1f}° | (right) ≈ {np.rad2deg(th_r):.1f}°")

    # Preview with path
    debug_img = draw_path(base_img, points)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.subheader("Debug visual (trajetória)")
        st.image(debug_img, use_container_width=True)
    with col2:
        st.subheader("Pontos (pixels)")
        st.json({k: {"x": v[0], "y": v[1]} for k, v in points.items()})

        pdf_out = add_path_to_pdf(pdf_path, zoom=zoom, points=points)
        st.download_button(
            "⬇️ Download PDF com trajetória",
            data=pdf_out,
            file_name="landing_ground_roll_debug.pdf",
            mime="application/pdf",
        )

st.caption("Se o resultado não bater com o exemplo do POH, diz-me quais inputs usaste e eu ajusto o passo do painel esquerdo (ISA/PA) ou a leitura do GR.")

