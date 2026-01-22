from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import pymupdf
from PIL import Image, ImageDraw
import cv2


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


def line_direction_angle(line: LineABC) -> float:
    """
    Ângulo da direção da reta (em rad) normalizado para [-pi/2, pi/2).
    Direção de (dx,dy) pode ser (b, -a) (perpendicular ao normal).
    """
    dx = line.b
    dy = -line.a
    ang = float(np.arctan2(dy, dx))
    # normalizar a "sem direção" para banda de pi
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    return ang


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
    """
    Ajusta x ≈ m*value + b de forma robusta (Theil–Sen):
      m = mediana das slopes entre pares
      b = mediana de (x - m*value)
    Muito resistente a outliers (os teus ticks tinham alguns).
    """
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

    if not slopes:
        raise ValueError("Ticks degenerados (valores repetidos).")

    m = float(np.median(slopes))
    b = float(np.median(xs - m * vals))
    return m, b


def robust_fit_value_from_y(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    """Ajusta value ≈ m*y + b robusto (Theil–Sen)."""
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

    if not slopes:
        raise ValueError("Ticks degenerados (Y repetidos).")

    m = float(np.median(slopes))
    b = float(np.median(vals - m * ys))
    return m, b


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# PANELS + GUIDES (pilot lines)
# =========================
def panel_bbox(panel_quad: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in panel_quad]
    ys = [p[1] for p in panel_quad]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def detect_guide_lines(img_rgb: Image.Image, panel_quad: List[List[float]]) -> List[LineABC]:
    """
    Deteta diagonais grossas via Hough, em coords globais.
    """
    x0, y0, x1, y1 = panel_bbox(panel_quad)
    crop = img_rgb.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
    arr = np.asarray(crop)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180, apertureSize=3)

    raw = cv2.HoughLines(edges, 1, np.pi / 180, 155)  # ligeiramente mais seletivo
    if raw is None:
        return []

    guides: List[LineABC] = []
    for rho_theta in raw[:300]:
        rho, theta = float(rho_theta[0][0]), float(rho_theta[0][1])

        a = float(np.cos(theta))
        b = float(np.sin(theta))
        c = -rho

        # rejeitar quase horizontais/verticais (queremos diagonais)
        ang = theta % np.pi
        if abs(ang - 0) < np.deg2rad(12) or abs(ang - np.pi / 2) < np.deg2rad(12):
            continue

        # converter para global:
        c_global = c - a * x0 - b * y0
        n = float(np.hypot(a, b))
        if n == 0:
            continue
        guides.append(LineABC(a / n, b / n, c_global / n))

    # dedup
    dedup: List[LineABC] = []
    for ln in guides:
        dup = False
        for ex in dedup:
            if abs(ln.a - ex.a) < 0.02 and abs(ln.b - ex.b) < 0.02 and abs(ln.c - ex.c) < 12:
                dup = True
                break
        if not dup:
            dedup.append(ln)

    # filtrar para o cluster angular dominante (mata grelha/ruído)
    if len(dedup) < 3:
        return dedup

    angs = np.array([line_direction_angle(l) for l in dedup], dtype=float)
    hist, edges = np.histogram(angs, bins=60, range=(-np.pi/2, np.pi/2))
    idx = int(np.argmax(hist))
    ang0 = float((edges[idx] + edges[idx + 1]) / 2)
    keep = []
    for ln in dedup:
        a = line_direction_angle(ln)
        if abs(a - ang0) <= np.deg2rad(6.0):
            keep.append(ln)
    return keep


def line_distance(line: LineABC, p: Tuple[float, float]) -> float:
    return abs(line.a * p[0] + line.b * p[1] + line.c)


def segment_darkness_score(gray_full: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], n=80) -> float:
    """
    Média da intensidade (0 preto, 255 branco) ao longo do segmento.
    Quanto MENOR, mais "guia grossa" parece.
    """
    h, w = gray_full.shape[:2]
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    vals = []
    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            vals.append(gray_full[yi, xi])
        else:
            vals.append(255)
    return float(np.mean(vals))


def pick_best_pilot_guide(
    guides: List[LineABC],
    p_start: Tuple[float, float],
    x_target: float,
    panel_bb: Tuple[float, float, float, float],
    gray_full: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], LineABC]]:
    """
    Escolhe a guia “pilot-like” correta:
      - a linha tem de passar perto do ponto inicial (dist <= 22 px)
      - a interseção com x_target tem de cair dentro do painel
      - escolhe a que dá o segmento mais escuro (guia grossa)
    """
    if not guides:
        return None

    x_min, y_min, x_max, y_max = panel_bb
    vline = vert_line(x_target)

    best_end = None
    best_ln = None
    best_score = 1e9

    for ln in guides:
        # tem de passar perto do ponto de partida (piloto encosta régua)
        if line_distance(ln, p_start) > 22.0:
            continue

        p_end = ln.intersect(vline)
        if p_end is None:
            continue

        x, y = p_end
        if not (x_min - 2 <= x <= x_max + 2 and y_min - 2 <= y <= y_max + 2):
            continue

        score = segment_darkness_score(gray_full, p_start, p_end, n=90)
        # leve preferência por segmentos mais longos (evita “agarra” curto na grelha)
        seg_len = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
        score2 = score - 0.02 * seg_len

        if score2 < best_score:
            best_score = score2
            best_end = p_end
            best_ln = ln

    if best_end is None or best_ln is None:
        return None
    return best_end, best_ln


# =========================
# NOMOGRAM MODEL
# =========================
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


def std_atm_isa_temp_c(pressure_alt_ft: float) -> float:
    return 15.0 - 1.98 * (pressure_alt_ft / 1000.0)


def solve_landing_ground_roll(
    capture: Dict[str, Any],
    base_img: Image.Image,
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    headwind_kt: float,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:

    gray_full = np.asarray(base_img.convert("L"))

    # ticks
    ticks_oat = capture["axis_ticks"]["oat_c"]
    ticks_wt = capture["axis_ticks"]["weight_x100_lb"]
    ticks_wnd = capture["axis_ticks"]["wind_kt"]
    ticks_gr = capture["axis_ticks"]["ground_roll_ft"]

    # robust fits (evita ticks maus)
    mo, bo = robust_fit_x_from_value(ticks_oat)
    mw, bw = robust_fit_x_from_value(ticks_wt)
    mwd, bwd = robust_fit_x_from_value(ticks_wnd)
    mgr, bgr = robust_fit_value_from_y(ticks_gr)

    def x_from_oat(oat: float) -> float:
        return mo * oat + bo

    def x_from_weight_lb(w_lb: float) -> float:
        return mw * (w_lb / 100.0) + bw

    def x_from_wind(w: float) -> float:
        return mwd * w + bwd

    def gr_from_y(y: float) -> float:
        return mgr * y + bgr

    # eixo OAT baseline y e eixo GR x (robustos)
    oat_axis_y = float(np.median([t["y"] for t in ticks_oat]))
    gr_axis_x = float(np.median([t["x"] for t in ticks_gr]))

    lines = capture["lines"]

    def seg_to_line(key: str) -> LineABC:
        seg = lines[key][0]
        return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))

    # ISA family
    line_isa_m15 = seg_to_line("isa_m15")
    line_isa_0 = seg_to_line("isa")
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

    # panel bboxes
    left_bb = panel_bbox(capture["panel_corners"]["left"])
    mid_bb = panel_bbox(capture["panel_corners"]["middle"])
    right_bb = panel_bbox(capture["panel_corners"]["right"])

    # -------- LEFT (ISA dev + PA) --------
    isa_temp = std_atm_isa_temp_c(pressure_alt_ft)
    dev = oat_c - isa_temp

    isa_line = interpolate_line([(-15.0, line_isa_m15), (0.0, line_isa_0), (35.0, line_isa_p35)], dev)
    pa_line = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA.")

    # seta "para cima" a partir do OAT no eixo (não no fundo do painel)
    x_oat = x_from_oat(oat_c)
    x_min_l, y_min_l, x_max_l, y_max_l = left_bb
    p_oat = (float(clamp(x_oat, x_min_l, x_max_l)), oat_axis_y)

    # -------- to WEIGHT REF (horizontal) --------
    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou interseção com weight_ref_line.")

    # -------- MIDDLE (pilot guide) --------
    guides_mid = detect_guide_lines(base_img, capture["panel_corners"]["middle"])
    x_weight = x_from_weight_lb(weight_lb)

    pick_mid = pick_best_pilot_guide(
        guides=guides_mid,
        p_start=p_wref,
        x_target=x_weight,
        panel_bb=mid_bb,
        gray_full=gray_full,
    )
    if pick_mid is None:
        # fallback: reta qualquer por interseção simples (evita crash)
        p_w = (x_weight, p_wref[1])
    else:
        p_w, _ = pick_mid

    # -------- to WIND REF ZERO (horizontal) --------
    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou interseção com wind_ref_zero.")

    # -------- RIGHT (pilot guide) --------
    guides_r = detect_guide_lines(base_img, capture["panel_corners"]["right"])
    x_wind = x_from_wind(headwind_kt)

    pick_r = pick_best_pilot_guide(
        guides=guides_r,
        p_start=p_z,
        x_target=x_wind,
        panel_bb=right_bb,
        gray_full=gray_full,
    )
    if pick_r is None:
        p_wind = (x_wind, p_z[1])
    else:
        p_wind, _ = pick_r

    # -------- GR + rounding + point on GR axis --------
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
# DRAWING (ARROWS + LABEL)
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

    tri = [(x2, y2), (bx + aw * px, by + aw * py), (bx - aw * px, by - aw * py)]
    draw.polygon(tri, fill=(255, 0, 0))


def draw_path_preview(img: Image.Image, points: Dict[str, Tuple[float, float]], gr_round: float) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # setas
    draw_arrow_pil(d, points["p_oat"], points["p_left"], width=5)      # 1º bloco: subir
    draw_arrow_pil(d, points["p_left"], points["p_wref"], width=5)
    draw_arrow_pil(d, points["p_wref"], points["p_w"], width=5)
    draw_arrow_pil(d, points["p_w"], points["p_z"], width=5)
    draw_arrow_pil(d, points["p_z"], points["p_wind"], width=5)
    draw_arrow_pil(d, points["p_wind"], points["p_gr"], width=5)       # até ao eixo GR

    # pontos
    def mark(p, color, r=6):
        x, y = p
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)

    for k in ["p_oat", "p_left", "p_wref", "p_w", "p_z", "p_wind", "p_gr"]:
        mark(points[k], (0, 120, 0) if k != "p_wind" else (0, 0, 200))

    # label do GR junto ao eixo
    xg, yg = points["p_gr"]
    d.text((xg + 8, yg - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))

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

    tri = [p2, pymupdf.Point(bx + aw * px, by + aw * py), pymupdf.Point(bx - aw * px, by - aw * py)]
    page.draw_polygon(tri, color=(1, 0, 0), fill=(1, 0, 0))


def add_path_to_pdf(pdf_path: str, zoom: float, points: Dict[str, Tuple[float, float]], gr_round: float) -> bytes:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect

    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    w_img, h_img = pix.width, pix.height

    sx = rect.width / w_img
    sy = rect.height / h_img

    def map_pt(p):
        return pymupdf.Point(p[0] * sx, p[1] * sy)

    order = ["p_oat", "p_left", "p_wref", "p_w", "p_z", "p_wind", "p_gr"]
    for a, b in zip(order[:-1], order[1:]):
        draw_arrow_pdf(page, map_pt(points[a]), map_pt(points[b]), width=2.5)

    for k in order:
        page.draw_circle(map_pt(points[k]), radius=3.0, color=(0, 0.6, 0), width=2.0)

    pgr = map_pt(points["p_gr"])
    page.insert_text(pymupdf.Point(pgr.x + 6, pgr.y - 6), f"{int(gr_round)} ft", fontsize=10, color=(0, 0, 0))

    out = doc.tobytes()
    doc.close()
    return out


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Landing Ground Roll Solver", layout="wide")
st.title("Landing Ground Roll — piloto-like (robusto + setas + GR no eixo)")

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
        gr_raw, gr_round, points = solve_landing_ground_roll(
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

    st.success(f"Landing Ground Roll ≈ **{gr_round:.0f} ft** (arredondado de 5 em 5)")
    if show_raw:
        st.caption(f"Bruto: {gr_raw:.1f} ft")

    debug_img = draw_path_preview(base_img, points, gr_round)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.subheader("Debug visual")
        st.image(debug_img, use_container_width=True)
    with col2:
        st.subheader("Pontos (pixels)")
        st.json({k: {"x": v[0], "y": v[1]} for k, v in points.items()})

        pdf_out = add_path_to_pdf(pdf_path, zoom=zoom, points=points, gr_round=gr_round)
        st.download_button(
            "⬇️ Download PDF com setas + GR no eixo",
            data=pdf_out,
            file_name="landing_ground_roll_debug_arrows_gr.pdf",
            mime="application/pdf",
        )

st.caption("Esta versão aguenta ticks maus (robusta) e escolhe a guia do piloto por ângulo + proximidade + escuridão.")
