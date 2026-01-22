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
# FILES
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

    def distance_to_point(self, p: Tuple[float, float]) -> float:
        return abs(self.a * p[0] + self.b * p[1] + self.c)  # normalizado => distância


def horiz_line(y: float) -> LineABC:
    return LineABC(0.0, 1.0, -float(y))


def vert_line(x: float) -> LineABC:
    return LineABC(1.0, 0.0, -float(x))


def parallel_through(line: LineABC, p: Tuple[float, float]) -> LineABC:
    # mesma normal (a,b), ajustar c para passar em p: a*x+b*y+c=0 => c=-(a*x+b*y)
    x, y = p
    return LineABC(line.a, line.b, -(line.a * x + line.b * y))


# =========================
# IO / RENDER
# =========================
def locate_file(name: str) -> Optional[str]:
    candidates = [Path.cwd() / name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return None


@st.cache_data(show_spinner=False)
def render_pdf_page_to_pil_cached(pdf_bytes: bytes, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def load_pdf_bytes(pdf_up) -> bytes:
    if pdf_up is not None:
        return pdf_up.read()
    pdf_path = locate_file(PDF_NAME_DEFAULT)
    if not pdf_path:
        raise FileNotFoundError(f"Não encontrei {PDF_NAME_DEFAULT}")
    return Path(pdf_path).read_bytes()


# =========================
# ROBUST CALIBRATION (Theil–Sen)
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# NOMOGRAM MODEL
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


def get_seg_line(capture: Dict[str, Any], key: str) -> LineABC:
    seg = capture["lines"][key][0]
    return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


def get_guides(capture: Dict[str, Any], panel: str) -> List[LineABC]:
    # guardamos em capture["guides"]["middle"] / ["right"]
    guides = capture.get("guides", {}).get(panel, [])
    out = []
    for g in guides:
        out.append(LineABC.from_points((g["x1"], g["y1"]), (g["x2"], g["y2"])))
    return out


def nearest_guide_line(guides: List[LineABC], p: Tuple[float, float]) -> Optional[LineABC]:
    if not guides:
        return None
    return min(guides, key=lambda ln: ln.distance_to_point(p))


def solve_landing_ground_roll(
    capture: Dict[str, Any],
    pressure_alt_ft: float,
    oat_c: float,
    weight_lb: float,
    headwind_kt: float,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    # Ticks
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

    # Base lines
    line_isa_m15 = get_seg_line(capture, "isa_m15")
    line_isa_0   = get_seg_line(capture, "isa")
    line_isa_p35 = get_seg_line(capture, "isa_p35")

    pa0 = get_seg_line(capture, "pa_sea_level")
    pa2 = get_seg_line(capture, "pa_2000")
    pa4 = get_seg_line(capture, "pa_4000")
    pa6 = get_seg_line(capture, "pa_6000")
    pa7 = get_seg_line(capture, "pa_7000")

    w_ref  = get_seg_line(capture, "weight_ref_line")
    z_wind = get_seg_line(capture, "wind_ref_zero")

    # Left panel: ISA deviation + PA
    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15.0, line_isa_m15), (0.0, line_isa_0), (35.0, line_isa_p35)], dev)
    pa_line  = interpolate_line([(0.0, pa0), (2000.0, pa2), (4000.0, pa4), (6000.0, pa6), (7000.0, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Falhou interseção ISA/PA.")

    # seta "para cima" no primeiro bloco (do eixo OAT)
    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    # horizontal -> weight ref
    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Falhou interseção com weight_ref_line.")

    # Weight panel: usar guia capturada (mais próxima do ponto) => paralela por p_wref
    x_weight = float(x_from_weight_lb(weight_lb))
    guides_mid = get_guides(capture, "middle")
    guide_mid = nearest_guide_line(guides_mid, p_wref)

    if guide_mid is not None:
        pilot_mid = parallel_through(guide_mid, p_wref)
        p_w = pilot_mid.intersect(vert_line(x_weight))
        if p_w is None:
            p_w = (x_weight, p_wref[1])
    else:
        # fallback simples (se não houver guias ainda)
        p_w = (x_weight, p_wref[1])

    # horizontal -> wind ref zero
    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Falhou interseção com wind_ref_zero.")

    # Wind panel: guia capturada (mais próxima) => paralela por p_z
    x_wind = float(x_from_wind(headwind_kt))
    guides_right = get_guides(capture, "right")
    guide_right = nearest_guide_line(guides_right, p_z)

    if guide_right is not None:
        pilot_right = parallel_through(guide_right, p_z)
        p_wind = pilot_right.intersect(vert_line(x_wind))
        if p_wind is None:
            p_wind = (x_wind, p_z[1])
    else:
        p_wind = (x_wind, p_z[1])

    gr_raw = float(gr_from_y(p_wind[1]))
    gr_round = float(5.0 * round(gr_raw / 5.0))

    # ponto no eixo de ground roll (para desenhar seta até lá)
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
# DRAWING
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


def draw_debug_overlay(img: Image.Image, points: Dict[str, Tuple[float, float]], gr_round: float) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # Setas principais
    draw_arrow_pil(d, points["p_oat"], points["p_left"], width=5)
    draw_arrow_pil(d, points["p_left"], points["p_wref"], width=5)
    draw_arrow_pil(d, points["p_wref"], points["p_w"], width=5)
    draw_arrow_pil(d, points["p_w"], points["p_z"], width=5)
    draw_arrow_pil(d, points["p_z"], points["p_wind"], width=5)
    draw_arrow_pil(d, points["p_wind"], points["p_gr"], width=5)

    # Pontos
    def mark(p, color, r=6):
        x, y = p
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)

    for k in ["p_oat", "p_left", "p_wref", "p_w", "p_z", "p_wind", "p_gr"]:
        mark(points[k], (0, 120, 0) if k != "p_wind" else (0, 0, 200))

    xg, yg = points["p_gr"]
    d.text((xg + 8, yg - 14), f"{int(gr_round)} ft", fill=(0, 0, 0))
    return out


# =========================
# CAPTURE EDITOR HELPERS
# =========================
def ensure_capture_defaults(cap: Dict[str, Any]) -> Dict[str, Any]:
    cap.setdefault("zoom", 2.3)
    cap.setdefault("axis_ticks", {})
    cap["axis_ticks"].setdefault("oat_c", [])
    cap["axis_ticks"].setdefault("weight_x100_lb", [])
    cap["axis_ticks"].setdefault("wind_kt", [])
    cap["axis_ticks"].setdefault("ground_roll_ft", [])
    cap.setdefault("lines", {})
    cap.setdefault("guides", {})
    cap["guides"].setdefault("middle", [])  # weight guides
    cap["guides"].setdefault("right", [])   # wind guides
    # panel_corners optional (não mexemos aqui)
    cap.setdefault("panel_corners", {})
    return cap


def overlay_draw_capture(img: Image.Image, cap: Dict[str, Any]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # ticks
    def draw_tick_list(lst, color):
        for t in lst:
            x, y = t["x"], t["y"]
            d.ellipse((x-4, y-4, x+4, y+4), outline=color, width=3)

    draw_tick_list(cap["axis_ticks"]["oat_c"], (0, 120, 0))
    draw_tick_list(cap["axis_ticks"]["weight_x100_lb"], (0, 120, 0))
    draw_tick_list(cap["axis_ticks"]["wind_kt"], (0, 120, 0))
    draw_tick_list(cap["axis_ticks"]["ground_roll_ft"], (0, 120, 0))

    # lines (segments)
    for key, segs in cap["lines"].items():
        for s in segs:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

    # guides
    for s in cap["guides"].get("middle", []):
        d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)
    for s in cap["guides"].get("right", []):
        d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

    return out


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PA28 Ground Roll – Editor + Solver", layout="wide")
st.title("PA28 Landing Ground Roll — Editor do capture.json + Solver")

# Uploads
colA, colB = st.columns(2)
with colA:
    pdf_up = st.file_uploader("PDF (opcional; senão usa o da pasta)", type=["pdf"])
with colB:
    cap_up = st.file_uploader("capture.json (opcional; senão usa o da pasta)", type=["json"])

pdf_bytes = load_pdf_bytes(pdf_up)

# Load capture
if "capture" not in st.session_state:
    cap_path = locate_file(CAPTURE_NAME_DEFAULT)
    if cap_up is not None:
        capture = json.loads(cap_up.read().decode("utf-8"))
    elif cap_path:
        capture = json.loads(Path(cap_path).read_text(encoding="utf-8"))
    else:
        capture = {"zoom": 2.3}
    st.session_state.capture = ensure_capture_defaults(capture)

cap = st.session_state.capture
cap = ensure_capture_defaults(cap)

# Zoom control
with st.sidebar:
    st.header("Render")
    cap["zoom"] = float(st.number_input("Zoom (render)", value=float(cap.get("zoom", 2.3)), step=0.1))
    show_overlay = st.checkbox("Mostrar overlay (ticks/linhas/guias)", value=True)

img = render_pdf_page_to_pil_cached(pdf_bytes, zoom=float(cap["zoom"]))
img_for_click = img.copy()
if show_overlay:
    img_for_click = overlay_draw_capture(img_for_click, cap)

tabs = st.tabs(["1) Editor capture.json", "2) Solver"])

# -------------------------
# TAB 1: Editor
# -------------------------
with tabs[0]:
    st.subheader("Editor do capture.json (clicar para adicionar / apagar outliers)")

    left, right = st.columns([1.4, 1])

    with right:
        st.markdown("### Modo de captura")
        mode = st.selectbox(
            "O que queres adicionar?",
            [
                "Tick: OAT (°C)",
                "Tick: Weight (x100 lb)",
                "Tick: Wind (kt)",
                "Tick: Ground Roll (ft)",
                "Linha: segmento (2 cliques) — escolher key",
                "Guia: Weight (2 cliques) (middle)",
                "Guia: Wind (2 cliques) (right)",
            ],
        )

        line_key = None
        if mode.startswith("Linha:"):
            # Mostra keys existentes + campo
            existing = sorted(list(cap["lines"].keys()))
            line_key = st.text_input("Key da linha (ex: pa_2000, isa, weight_ref_line...)", value=existing[0] if existing else "pa_2000")

        tick_value = None
        if mode.startswith("Tick:"):
            tick_value = st.number_input("Valor do tick (ex: -10, 21, 2240/100=22.4, 5, 820)", value=0.0, step=1.0)

        st.markdown("---")
        st.markdown("### Apagar (outliers)")
        del_kind = st.selectbox("O que queres apagar?", ["—", "Ticks", "Linhas (segmentos)", "Guias"])

        if del_kind == "Ticks":
            axis = st.selectbox("Qual eixo?", ["oat_c", "weight_x100_lb", "wind_kt", "ground_roll_ft"])
            ticks = cap["axis_ticks"][axis]
            if ticks:
                options = [f"[{i}] value={t['value']}  (x={int(t['x'])}, y={int(t['y'])})" for i, t in enumerate(ticks)]
                to_del = st.multiselect("Seleciona índices para apagar", options)
                if st.button("Apagar ticks selecionados"):
                    idxs = sorted([int(s.split("]")[0][1:]) for s in to_del], reverse=True)
                    for i in idxs:
                        cap["axis_ticks"][axis].pop(i)
                    st.success("Ticks apagados.")
            else:
                st.info("Sem ticks neste eixo.")

        elif del_kind == "Linhas (segmentos)":
            keys = sorted(list(cap["lines"].keys()))
            if keys:
                k = st.selectbox("Key", keys)
                segs = cap["lines"][k]
                options = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(segs)]
                to_del = st.multiselect("Seleciona segmentos para apagar", options)
                if st.button("Apagar segmentos selecionados"):
                    idxs = sorted([int(s.split("]")[0][1:]) for s in to_del], reverse=True)
                    for i in idxs:
                        cap["lines"][k].pop(i)
                    st.success("Segmentos apagados.")
            else:
                st.info("Sem linhas ainda.")

        elif del_kind == "Guias":
            panel = st.selectbox("Painel", ["middle (weight)", "right (wind)"])
            pkey = "middle" if panel.startswith("middle") else "right"
            segs = cap["guides"][pkey]
            if segs:
                options = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(segs)]
                to_del = st.multiselect("Seleciona guias para apagar", options)
                if st.button("Apagar guias selecionadas"):
                    idxs = sorted([int(s.split(']')[0][1:]) for s in to_del], reverse=True)
                    for i in idxs:
                        cap["guides"][pkey].pop(i)
                    st.success("Guias apagadas.")
            else:
                st.info("Sem guias neste painel.")

        st.markdown("---")
        st.markdown("### Guardar / Exportar JSON")
        json_text = json.dumps(cap, indent=2)
        st.download_button("⬇️ Download capture.json", data=json_text, file_name="capture.json", mime="application/json")
        if st.button("Guardar como capture.json na app (Streamlit Cloud)"):
            Path("capture.json").write_text(json_text, encoding="utf-8")
            st.success("Gravei capture.json no diretório da app.")

        st.markdown("---")
        st.caption("Dica: para as guias, captura 2–4 linhas grossas por painel (de baixo para cima se quiseres).")

    with left:
        st.markdown("### Clicar na imagem para capturar")
        click = streamlit_image_coordinates(np.array(img_for_click), key="img_click")

        # state para segmentos (2 cliques)
        if "pending_point" not in st.session_state:
            st.session_state.pending_point = None

        if click is not None:
            x = float(click["x"])
            y = float(click["y"])

            st.write(f"Click: x={int(x)}, y={int(y)}")

            if mode.startswith("Tick:"):
                axis_map = {
                    "Tick: OAT (°C)": "oat_c",
                    "Tick: Weight (x100 lb)": "weight_x100_lb",
                    "Tick: Wind (kt)": "wind_kt",
                    "Tick: Ground Roll (ft)": "ground_roll_ft",
                }
                axis = axis_map[mode]
                cap["axis_ticks"][axis].append({"x": x, "y": y, "value": float(tick_value)})
                st.success(f"Tick adicionado em {axis}.")

            elif mode.startswith("Linha: segmento"):
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto guardado. Clica no segundo ponto.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["lines"].setdefault(line_key, [])
                    cap["lines"][line_key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success(f"Segmento adicionado em lines['{line_key}'].")

            elif mode.startswith("Guia: Weight"):
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto da guia (weight) guardado. Clica no segundo.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["guides"]["middle"].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success("Guia (weight/middle) adicionada.")

            elif mode.startswith("Guia: Wind"):
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto da guia (wind) guardado. Clica no segundo.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["guides"]["right"].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success("Guia (wind/right) adicionada.")

        # status rápido
        st.markdown("#### Estado atual")
        st.write({
            "ticks_oat": len(cap["axis_ticks"]["oat_c"]),
            "ticks_weight": len(cap["axis_ticks"]["weight_x100_lb"]),
            "ticks_wind": len(cap["axis_ticks"]["wind_kt"]),
            "ticks_ground_roll": len(cap["axis_ticks"]["ground_roll_ft"]),
            "lines_keys": list(cap["lines"].keys()),
            "guides_middle": len(cap["guides"]["middle"]),
            "guides_right": len(cap["guides"]["right"]),
        })


# -------------------------
# TAB 2: Solver
# -------------------------
with tabs[1]:
    st.subheader("Solver (usa capture.json atual)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pressure_alt_ft = st.number_input("Pressure Altitude (ft)", value=2500.0, step=100.0)
    with c2:
        oat_c = st.number_input("OAT (°C)", value=21.0, step=1.0)
    with c3:
        weight_lb = st.number_input("Weight (lb)", value=2240.0, step=10.0)
    with c4:
        headwind_kt = st.number_input("Headwind component (kt)", value=5.0, step=1.0)

    show_raw = st.checkbox("Mostrar bruto (antes de arredondar)", value=False)
    run = st.button("Calcular", key="solve_btn")

    if run:
        try:
            gr_raw, gr_round, pts = solve_landing_ground_roll(
                cap,
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

        base_img = render_pdf_page_to_pil_cached(pdf_bytes, zoom=float(cap["zoom"]))
        debug = draw_debug_overlay(base_img, pts, gr_round)

        st.image(debug, use_container_width=True)

        with st.expander("Pontos (pixels)"):
            st.json({k: {"x": v[0], "y": v[1]} for k, v in pts.items()})

        st.info(
            "Para ficar 100% 'piloto-like', adiciona 2–4 guias no painel middle (weight) "
            "e 2–4 guias no painel right (wind). O solver escolhe sempre a mais próxima."
        )


