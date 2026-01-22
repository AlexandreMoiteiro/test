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
# FILE DEFAULTS (ajustados ao teu repo)
# =========================
DEFAULT_LANDING_PDF = "ldg_ground_roll.pdf"         # <- tens este no repo
DEFAULT_LANDING_CAPTURE = "capture_landing_gr.json" # <- tens este no repo

DEFAULT_TAKEOFF_IMAGE = "to_ground_roll.jpg"        # tens este
DEFAULT_CLIMB_IMAGE   = "climb_perf.jpg"            # tens este


# =========================
# MODE CONFIG
# =========================
MODE_CONFIG = {
    "landing_gr": {
        "title": "Landing Ground Roll",
        "capture_file": DEFAULT_LANDING_CAPTURE,
        "pdf_default": DEFAULT_LANDING_PDF,
        "is_pdf": True,
        "panels": ["left", "middle", "right"],
        "axis_ticks": ["oat_c", "weight_x100_lb", "wind_kt", "ground_roll_ft"],
        "lines": [
            "isa_m15", "isa", "isa_p35",
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_7000",
            "weight_ref_line", "wind_ref_zero",
        ],
        "guides": ["middle", "right"],
        "solver": "landing",
    },
    "takeoff_gr": {
        "title": "Flaps Up Takeoff Ground Roll",
        "capture_file": "capture_takeoff_gr.json",
        "img_default": DEFAULT_TAKEOFF_IMAGE,
        "is_pdf": False,
        "panels": ["left", "right"],
        "axis_ticks": ["oat_c", "weight_x100_lb", "wind_kt", "takeoff_gr_ft"],
        "lines": [
            "isa_m15", "isa", "isa_p35",
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_8000",
            "weight_ref_line", "wind_ref_zero",
        ],
        "guides": ["right"],
        "solver": "todo",
    },
    "climb_perf": {
        "title": "Climb Performance",
        "capture_file": "capture_climb_perf.json",
        "img_default": DEFAULT_CLIMB_IMAGE,
        "is_pdf": False,
        "panels": ["main"],
        "axis_ticks": ["oat_c", "roc_fpm"],
        "lines": [
            "isa_m15", "isa", "isa_p35",
            "pa_sea_level", "pa_1000", "pa_2000", "pa_3000", "pa_4000", "pa_5000",
            "pa_6000", "pa_7000", "pa_8000", "pa_9000", "pa_10000", "pa_11000",
            "pa_12000", "pa_13000",
        ],
        "guides": ["main"],
        "solver": "todo",
    },
}


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


def parallel_through(line: LineABC, p: Tuple[float, float]) -> LineABC:
    x, y = p
    return LineABC(line.a, line.b, -(line.a * x + line.b * y))


# =========================
# IO
# =========================
def locate_file(name: str) -> Optional[str]:
    p = Path(name)
    if p.exists():
        return str(p)
    if "__file__" in globals():
        p2 = Path(__file__).resolve().parent / name
        if p2.exists():
            return str(p2)
    return None


@st.cache_data(show_spinner=False)
def render_pdf_page_to_pil_cached(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


@st.cache_data(show_spinner=False)
def load_image_cached(img_bytes: bytes) -> Image.Image:
    return Image.open(Path("/dev/null")).convert("RGB")  # dummy to satisfy cache signature


def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_pdf_bytes(upload, default_path: str) -> bytes:
    if upload is not None:
        return upload.read()
    p = locate_file(default_path)
    if not p:
        raise FileNotFoundError(f"Não encontrei {default_path}")
    return Path(p).read_bytes()


# =========================
# CAPTURE STRUCTURE
# =========================
def new_capture(mode_key: str, zoom: float = 2.3) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode_key]
    cap = {
        "mode": mode_key,
        "zoom": float(zoom),
        "page_index": 0,
        "panel_corners": {p: [] for p in cfg["panels"]},
        "axis_ticks": {k: [] for k in cfg["axis_ticks"]},
        "lines": {k: [] for k in cfg["lines"]},
        "guides": {p: [] for p in cfg.get("guides", [])},
    }
    return cap


def ensure_capture(cap: Dict[str, Any], mode_key: str) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode_key]
    cap.setdefault("mode", mode_key)
    cap.setdefault("zoom", 2.3)
    cap.setdefault("page_index", 0)
    cap.setdefault("panel_corners", {})
    for p in cfg["panels"]:
        cap["panel_corners"].setdefault(p, [])
    cap.setdefault("axis_ticks", {})
    for k in cfg["axis_ticks"]:
        cap["axis_ticks"].setdefault(k, [])
    cap.setdefault("lines", {})
    for k in cfg["lines"]:
        cap["lines"].setdefault(k, [])
    cap.setdefault("guides", {})
    for p in cfg.get("guides", []):
        cap["guides"].setdefault(p, [])
    return cap


def overlay_draw(img: Image.Image, cap: Dict[str, Any]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # panel corners
    for panel, pts in cap.get("panel_corners", {}).items():
        if len(pts) == 4:
            poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
            d.line(poly + [poly[0]], fill=(0, 100, 255), width=3)
            d.text((poly[0][0] + 5, poly[0][1] + 5), panel, fill=(0, 100, 255))

    # ticks
    for axis, ticks in cap.get("axis_ticks", {}).items():
        for t in ticks:
            x, y = t["x"], t["y"]
            d.ellipse((x-4, y-4, x+4, y+4), outline=(0, 150, 0), width=3)

    # lines
    for key, segs in cap.get("lines", {}).items():
        for s in segs:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

    # guides
    for panel, segs in cap.get("guides", {}).items():
        for s in segs:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

    return out


# =========================
# Robust axis calibration (Theil–Sen)
# =========================
def robust_fit_x_from_value(ticks: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = np.array([t["value"] for t in ticks], dtype=float)
    xs = np.array([t["x"] for t in ticks], dtype=float)
    n = len(vals)
    if n < 2:
        raise ValueError("Precisas de >= 2 ticks.")
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
        raise ValueError("Precisas de >= 2 ticks.")
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


def interpolate_line(pairs: List[Tuple[float, LineABC]], value: float) -> LineABC:
    pairs = sorted(pairs, key=lambda t: t[0])
    values = [v for v, _ in pairs]
    v = max(values[0], min(values[-1], value))

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


def std_atm_isa_temp_c(pressure_alt_ft: float) -> float:
    return 15.0 - 1.98 * (pressure_alt_ft / 1000.0)


def seg_to_line(cap: Dict[str, Any], key: str) -> LineABC:
    seg = cap["lines"][key][0]
    return LineABC.from_points((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


def guides_to_lines(cap: Dict[str, Any], panel: str) -> List[LineABC]:
    out = []
    for s in cap.get("guides", {}).get(panel, []):
        out.append(LineABC.from_points((s["x1"], s["y1"]), (s["x2"], s["y2"])))
    return out


def nearest_guide(guides: List[LineABC], p: Tuple[float, float]) -> Optional[LineABC]:
    if not guides:
        return None
    return min(guides, key=lambda ln: ln.distance_to_point(p))


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


def landing_solver(cap: Dict[str, Any], pressure_alt_ft: float, oat_c: float, weight_lb: float, headwind_kt: float):
    # axis fits
    mo, bo = robust_fit_x_from_value(cap["axis_ticks"]["oat_c"])
    mw, bw = robust_fit_x_from_value(cap["axis_ticks"]["weight_x100_lb"])
    mwd, bwd = robust_fit_x_from_value(cap["axis_ticks"]["wind_kt"])
    mgr, bgr = robust_fit_value_from_y(cap["axis_ticks"]["ground_roll_ft"])

    def x_from_oat(v): return mo * v + bo
    def x_from_weight(v): return mw * (v / 100.0) + bw
    def x_from_wind(v): return mwd * v + bwd
    def gr_from_y(y): return mgr * y + bgr

    oat_axis_y = float(np.median([t["y"] for t in cap["axis_ticks"]["oat_c"]]))
    gr_axis_x = float(np.median([t["x"] for t in cap["axis_ticks"]["ground_roll_ft"]]))

    # lines
    isa_m15 = seg_to_line(cap, "isa_m15")
    isa_0 = seg_to_line(cap, "isa")
    isa_p35 = seg_to_line(cap, "isa_p35")

    pa0 = seg_to_line(cap, "pa_sea_level")
    pa2 = seg_to_line(cap, "pa_2000")
    pa4 = seg_to_line(cap, "pa_4000")
    pa6 = seg_to_line(cap, "pa_6000")
    pa7 = seg_to_line(cap, "pa_7000")

    w_ref = seg_to_line(cap, "weight_ref_line")
    z_wind = seg_to_line(cap, "wind_ref_zero")

    dev = oat_c - std_atm_isa_temp_c(pressure_alt_ft)
    isa_line = interpolate_line([(-15, isa_m15), (0, isa_0), (35, isa_p35)], dev)
    pa_line = interpolate_line([(0, pa0), (2000, pa2), (4000, pa4), (6000, pa6), (7000, pa7)], pressure_alt_ft)

    p_left = isa_line.intersect(pa_line)
    if p_left is None:
        raise ValueError("Interseção ISA/PA falhou.")

    p_oat = (float(x_from_oat(oat_c)), oat_axis_y)

    p_wref = w_ref.intersect(horiz_line(p_left[1]))
    if p_wref is None:
        raise ValueError("Interseção com weight_ref_line falhou.")

    x_w = float(x_from_weight(weight_lb))
    gmid = nearest_guide(guides_to_lines(cap, "middle"), p_wref)
    if gmid is not None:
        pilot = parallel_through(gmid, p_wref)
        p_w = pilot.intersect(vert_line(x_w)) or (x_w, p_wref[1])
    else:
        p_w = (x_w, p_wref[1])

    p_z = z_wind.intersect(horiz_line(p_w[1]))
    if p_z is None:
        raise ValueError("Interseção com wind_ref_zero falhou.")

    x_h = float(x_from_wind(headwind_kt))
    gright = nearest_guide(guides_to_lines(cap, "right"), p_z)
    if gright is not None:
        pilot = parallel_through(gright, p_z)
        p_wind = pilot.intersect(vert_line(x_h)) or (x_h, p_z[1])
    else:
        p_wind = (x_h, p_z[1])

    gr_raw = float(gr_from_y(p_wind[1]))
    gr_round = float(5 * round(gr_raw / 5))
    p_gr = (gr_axis_x, p_wind[1])

    pts = {"p_oat": p_oat, "p_left": p_left, "p_wref": p_wref, "p_w": p_w, "p_z": p_z, "p_wind": p_wind, "p_gr": p_gr}
    return gr_raw, gr_round, pts


def draw_solution_overlay(img: Image.Image, pts: Dict[str, Tuple[float, float]], value_text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    order = ["p_oat", "p_left", "p_wref", "p_w", "p_z", "p_wind", "p_gr"]
    for a, b in zip(order[:-1], order[1:]):
        draw_arrow(d, pts[a], pts[b], width=5)

    def mark(p, color, r=6):
        x, y = p
        d.ellipse((x-r, y-r, x+r, y+r), outline=color, width=4)

    for k in order:
        mark(pts[k], (0, 120, 0) if k != "p_wind" else (0, 0, 200))

    xg, yg = pts["p_gr"]
    d.text((xg + 8, yg - 14), value_text, fill=(0, 0, 0))
    return out


# =========================
# APP
# =========================
st.set_page_config(page_title="PA28 — Editor + Solver", layout="wide")
st.title("PA28 — Editor + Solver (3 gráficos)")

mode_key = st.sidebar.selectbox(
    "Modo",
    options=list(MODE_CONFIG.keys()),
    format_func=lambda k: MODE_CONFIG[k]["title"],
)
cfg = MODE_CONFIG[mode_key]

st.sidebar.markdown("---")
st.sidebar.header("Ficheiros")

# Capture load/upload
cap_up = st.sidebar.file_uploader("Upload capture.json (opcional)", type=["json"])
if cap_up is not None:
    cap = json.loads(cap_up.read().decode("utf-8"))
else:
    cap_path = locate_file(cfg["capture_file"])
    if cap_path:
        cap = json.loads(Path(cap_path).read_text(encoding="utf-8"))
    else:
        cap = new_capture(mode_key)

cap = ensure_capture(cap, mode_key)

# PDF ou imagem de fundo
bg_pdf_up = None
bg_img_up = None

if cfg["is_pdf"]:
    bg_pdf_up = st.sidebar.file_uploader("Upload PDF (opcional)", type=["pdf"])
    pdf_bytes = load_pdf_bytes(bg_pdf_up, cfg["pdf_default"])
    page_index = st.sidebar.number_input("Página (0-index)", value=int(cap.get("page_index", 0)), step=1)
    zoom = st.sidebar.number_input("Zoom", value=float(cap.get("zoom", 2.3)), step=0.1)
    cap["page_index"] = int(page_index)
    cap["zoom"] = float(zoom)
    img = render_pdf_page_to_pil_cached(pdf_bytes, int(page_index), float(zoom))
else:
    bg_img_up = st.sidebar.file_uploader("Upload imagem (opcional)", type=["png", "jpg", "jpeg"])
    if bg_img_up is not None:
        Path("_bg.jpg").write_bytes(bg_img_up.read())
        img = Image.open("_bg.jpg").convert("RGB")
    else:
        p = locate_file(cfg["img_default"])
        if not p:
            st.error(f"Não encontrei imagem default: {cfg['img_default']}")
            st.stop()
        img = Image.open(p).convert("RGB")

show_overlay = st.sidebar.checkbox("Mostrar overlay no editor", value=True)

# Tabs
tab_editor, tab_solver, tab_export = st.tabs(["Editor", "Solver", "Export/Apagar"])

# -------------------------
# Editor
# -------------------------
with tab_editor:
    st.subheader(f"Editor — {cfg['title']}")

    img_show = overlay_draw(img, cap) if show_overlay else img

    colL, colR = st.columns([1.4, 1])

    with colR:
        st.markdown("### Capturar")
        modes = []
        modes.append("Panel corners (4 cliques)")
        modes += [f"Tick: {k}" for k in cfg["axis_ticks"]]
        modes += [f"Line segment: {k}" for k in cfg["lines"]]
        modes += [f"Guide segment: {p}" for p in cfg.get("guides", [])]
        task = st.selectbox("Modo de captura", modes)

        tick_val = 0.0
        if task.startswith("Tick:"):
            tick_val = st.number_input("Valor do tick", value=0.0, step=1.0)

        st.markdown("---")
        st.write("Estado:")
        st.write({
            "ticks": {k: len(cap["axis_ticks"][k]) for k in cfg["axis_ticks"]},
            "lines": {k: len(cap["lines"][k]) for k in cfg["lines"]},
            "guides": {k: len(cap.get("guides", {}).get(k, [])) for k in cfg.get("guides", [])},
            "panel_corners": {k: len(cap["panel_corners"][k]) for k in cfg["panels"]},
        })

    with colL:
        click = streamlit_image_coordinates(np.array(img_show), key=f"click_{mode_key}_main")
        if "pending_point" not in st.session_state:
            st.session_state.pending_point = None
        if "pending_corners" not in st.session_state:
            st.session_state.pending_corners = []

        if click is not None:
            x = float(click["x"])
            y = float(click["y"])
            st.write(f"Click: x={int(x)}, y={int(y)}")

            if task == "Panel corners (4 cliques)":
                st.session_state.pending_corners.append({"x": x, "y": y})
                if len(st.session_state.pending_corners) == 4:
                    panel = st.selectbox("Qual painel?", cfg["panels"], key=f"panel_pick_{mode_key}")
                    cap["panel_corners"][panel] = st.session_state.pending_corners
                    st.session_state.pending_corners = []
                    st.success(f"Corners guardados em '{panel}'.")

            elif task.startswith("Tick:"):
                axis = task.split("Tick: ")[1]
                cap["axis_ticks"][axis].append({"x": x, "y": y, "value": float(tick_val)})
                st.success(f"Tick adicionado em {axis}")

            elif task.startswith("Line segment:"):
                key = task.split("Line segment: ")[1]
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto guardado. Clica no segundo.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["lines"][key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success(f"Segmento adicionado em lines['{key}'].")

            elif task.startswith("Guide segment:"):
                panel = task.split("Guide segment: ")[1]
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto guardado. Clica no segundo.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["guides"][panel].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success(f"Guia adicionada em guides['{panel}'].")

        st.image(img_show, use_container_width=True)

# -------------------------
# Solver
# -------------------------
with tab_solver:
    st.subheader(f"Solver — {cfg['title']}")
    if cfg["solver"] == "landing":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pressure_alt_ft = st.number_input("Pressure Altitude (ft)", value=2500.0, step=100.0)
        with c2:
            oat_c = st.number_input("OAT (°C)", value=21.0, step=1.0)
        with c3:
            weight_lb = st.number_input("Weight (lb)", value=2240.0, step=10.0)
        with c4:
            headwind_kt = st.number_input("Headwind component (kt)", value=5.0, step=1.0)

        run = st.button("Calcular")
        if run:
            try:
                gr_raw, gr_round, pts = landing_solver(
                    cap,
                    float(pressure_alt_ft),
                    float(oat_c),
                    float(weight_lb),
                    float(headwind_kt),
                )
            except Exception as e:
                st.exception(e)
                st.stop()

            st.success(f"Landing Ground Roll ≈ {gr_round:.0f} ft (arredondado de 5 em 5)")
            st.caption(f"Bruto: {gr_raw:.1f} ft")

            out_img = draw_solution_overlay(img, pts, f"{int(gr_round)} ft")
            st.image(out_img, use_container_width=True)

            with st.expander("Pontos (pixels)"):
                st.json({k: {"x": v[0], "y": v[1]} for k, v in pts.items()})

    else:
        st.info("Solver ainda não implementado para este gráfico — por agora usa o Editor para criar o JSON.")

# -------------------------
# Export / Apagar
# -------------------------
with tab_export:
    st.subheader("Export / Apagar outliers")

    col1, col2 = st.columns(2)
    with col1:
        txt = json.dumps(cap, indent=2)
        st.download_button("⬇️ Download capture JSON", data=txt, file_name=cfg["capture_file"], mime="application/json")
        if st.button("Guardar no repo (na app)"):
            Path(cfg["capture_file"]).write_text(txt, encoding="utf-8")
            st.success(f"Guardado: {cfg['capture_file']}")

    with col2:
        kind = st.selectbox("O que queres apagar?", ["Ticks", "Lines", "Guides", "Panel corners"])
        if kind == "Ticks":
            axis = st.selectbox("Axis", cfg["axis_ticks"])
            items = cap["axis_ticks"][axis]
            if not items:
                st.info("Sem ticks.")
            else:
                opts = [f"[{i}] value={t['value']} (x={int(t['x'])}, y={int(t['y'])})" for i, t in enumerate(items)]
                pick = st.multiselect("Seleciona", opts)
                if st.button("Apagar selecionados"):
                    idxs = sorted([int(s.split(']')[0][1:]) for s in pick], reverse=True)
                    for i in idxs:
                        cap["axis_ticks"][axis].pop(i)
                    st.success("Apagado.")

        elif kind == "Lines":
            key = st.selectbox("Line key", cfg["lines"])
            items = cap["lines"][key]
            if not items:
                st.info("Sem segmentos.")
            else:
                opts = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(items)]
                pick = st.multiselect("Seleciona", opts)
                if st.button("Apagar selecionados"):
                    idxs = sorted([int(s.split(']')[0][1:]) for s in pick], reverse=True)
                    for i in idxs:
                        cap["lines"][key].pop(i)
                    st.success("Apagado.")

        elif kind == "Guides":
            if not cfg.get("guides", []):
                st.info("Este modo não tem guides.")
            else:
                panel = st.selectbox("Guide panel", cfg["guides"])
                items = cap["guides"][panel]
                if not items:
                    st.info("Sem guias.")
                else:
                    opts = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(items)]
                    pick = st.multiselect("Seleciona", opts)
                    if st.button("Apagar selecionados"):
                        idxs = sorted([int(s.split(']')[0][1:]) for s in pick], reverse=True)
                        for i in idxs:
                            cap["guides"][panel].pop(i)
                        st.success("Apagado.")

        elif kind == "Panel corners":
            panel = st.selectbox("Panel", cfg["panels"])
            if st.button("Limpar corners"):
                cap["panel_corners"][panel] = []
                st.success("Limpo.")


