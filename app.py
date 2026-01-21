from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import streamlit as st
import pymupdf
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# =========================
# CONFIG
# =========================
PDF_NAME = "PA28POH-ground-roll.pdf"

PANELS = ["left", "middle", "right"]

# Linhas relevantes (ajusta se necessário)
LINE_KEYS = [
    # Painel esquerdo (atmosfera/altitude)
    "isa",
    "isa_m15",
    "isa_p35",
    "pa_sea_level",
    "pa_2000",
    "pa_4000",
    "pa_6000",
    "pa_7000",
    # Referências essenciais (peso/vento)
    "weight_ref_line",   # "REF. LINE MAX. WEIGHT 2,550 LBS."
    "wind_ref_zero",     # "REF. LINE ZERO WIND"
]

# Eixos a calibrar (ticks)
AXES = [
    ("oat_c", "Outside Air Temperature (°C)"),
    ("weight_x100_lb", "Weight x100 (lb)"),
    ("wind_kt", "Wind components (kt)"),
    ("ground_roll_ft", "Landing Ground Roll (ft)"),
]


# =========================
# HELPERS
# =========================
def locate_pdf(name: str) -> str:
    candidates = [Path.cwd() / name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Não encontrei '{name}'. Procurei em: " + ", ".join(str(x) for x in candidates)
    )


def render_pdf(pdf_path: str, zoom: float) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def init_state():
    """Inicializa e reconcilia o estado para evitar KeyError quando listas mudam."""
    if "zoom" not in st.session_state:
        st.session_state.zoom = 2.0

    # panel corners
    if "panel_corners" not in st.session_state:
        st.session_state.panel_corners = {p: [] for p in PANELS}
    else:
        for p in PANELS:
            st.session_state.panel_corners.setdefault(p, [])

    # axis ticks
    if "axis_ticks" not in st.session_state:
        st.session_state.axis_ticks = {a[0]: [] for a in AXES}
    else:
        for a, _ in AXES:
            st.session_state.axis_ticks.setdefault(a, [])

    # lines
    if "lines" not in st.session_state:
        st.session_state.lines = {k: [] for k in LINE_KEYS}
    else:
        for k in LINE_KEYS:
            st.session_state.lines.setdefault(k, [])
        # remove keys antigas (opcional)
        for k in list(st.session_state.lines.keys()):
            if k not in LINE_KEYS:
                del st.session_state.lines[k]

    # flow
    if "mode" not in st.session_state:
        st.session_state.mode = "panel"
    if "selected_panel" not in st.session_state:
        st.session_state.selected_panel = PANELS[0]
    if "selected_axis" not in st.session_state:
        st.session_state.selected_axis = AXES[0][0]
    if "axis_value" not in st.session_state:
        st.session_state.axis_value = 0.0
    if "selected_line" not in st.session_state:
        st.session_state.selected_line = LINE_KEYS[0]
    if "pending_point" not in st.session_state:
        st.session_state.pending_point = None  # type: Optional[Tuple[float, float]]
    if "last_click" not in st.session_state:
        st.session_state.last_click = None  # type: Optional[Tuple[float, float]]


def draw_overlay(img: Image.Image) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # Panels (azul)
    for p, pts in st.session_state.panel_corners.items():
        if len(pts) == 4:
            d.polygon(pts, outline=(0, 200, 255), width=4)
            d.text((pts[0][0] + 6, pts[0][1] + 6), p, fill=(0, 200, 255))

    # Axis ticks (laranja)
    for axis_name, ticks in st.session_state.axis_ticks.items():
        for t in ticks:
            x, y = t["x"], t["y"]
            d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 165, 0), width=3)
            d.text((x + 8, y - 10), f'{axis_name}:{t["value"]}', fill=(255, 165, 0))

    # Lines (vermelho)
    for k, segs in st.session_state.lines.items():
        for seg in segs:
            d.line([(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])], fill=(255, 0, 0), width=3)

    # Pending point (verde)
    if st.session_state.pending_point is not None:
        x, y = st.session_state.pending_point
        d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(0, 255, 0), width=4)

    return out


def register_click(x: float, y: float):
    # debounce simples
    if st.session_state.last_click == (x, y):
        return
    st.session_state.last_click = (x, y)

    mode = st.session_state.mode

    if mode == "panel":
        p = st.session_state.selected_panel
        pts = st.session_state.panel_corners.setdefault(p, [])
        if len(pts) < 4:
            pts.append((x, y))

    elif mode == "axis":
        axis = st.session_state.selected_axis
        st.session_state.axis_ticks.setdefault(axis, []).append({"value": float(st.session_state.axis_value), "x": x, "y": y})

    elif mode == "line":
        key = st.session_state.selected_line
        st.session_state.lines.setdefault(key, [])
        if st.session_state.pending_point is None:
            st.session_state.pending_point = (x, y)
        else:
            x1, y1 = st.session_state.pending_point
            st.session_state.lines[key].append({"x1": float(x1), "y1": float(y1), "x2": float(x), "y2": float(y)})
            st.session_state.pending_point = None


# =========================
# APP
# =========================
st.set_page_config(layout="wide", page_title="Capture Nomogram Geometry")
init_state()

st.title("Captura guiada — painéis, eixos e retas (por cliques)")

try:
    pdf_path = locate_pdf(PDF_NAME)
except Exception as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    st.session_state.zoom = st.slider("Zoom", 1.0, 4.0, float(st.session_state.zoom), 0.1)

    st.header("Modo de captura")
    st.session_state.mode = st.radio(
        "Escolhe",
        ["panel", "axis", "line"],
        index=["panel", "axis", "line"].index(st.session_state.mode),
    )

    if st.session_state.mode == "panel":
        st.subheader("Painel (4 cantos)")
        st.session_state.selected_panel = st.selectbox("Qual painel?", PANELS, index=PANELS.index(st.session_state.selected_panel))
        st.caption("Clica 4 cantos da grelha: TL → TR → BR → BL")
        if st.button("Limpar cantos deste painel"):
            st.session_state.panel_corners[st.session_state.selected_panel] = []
            st.rerun()

    elif st.session_state.mode == "axis":
        st.subheader("Eixo / Tick")
        axis_names = [a[0] for a in AXES]
        st.session_state.selected_axis = st.selectbox("Qual eixo?", axis_names, index=axis_names.index(st.session_state.selected_axis))
        st.session_state.axis_value = st.number_input("Valor do tick", value=float(st.session_state.axis_value))
        st.caption("Clica exatamente em cima do tick/número correspondente.")
        if st.button("Limpar ticks deste eixo"):
            st.session_state.axis_ticks[st.session_state.selected_axis] = []
            st.rerun()

    elif st.session_state.mode == "line":
        st.subheader("Reta (2 pontos)")
        st.session_state.selected_line = st.selectbox("Qual reta?", LINE_KEYS, index=LINE_KEYS.index(st.session_state.selected_line))
        st.caption("Clica 2 pontos (extremos) para guardar 1 segmento.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Cancelar ponto 1"):
                st.session_state.pending_point = None
                st.rerun()
        with c2:
            if st.button("Limpar esta reta"):
                st.session_state.lines[st.session_state.selected_line] = []
                st.session_state.pending_point = None
                st.rerun()

    st.divider()
    export = {
        "pdf_name": PDF_NAME,
        "zoom": float(st.session_state.zoom),
        "panel_corners": st.session_state.panel_corners,
        "axis_ticks": st.session_state.axis_ticks,
        "lines": st.session_state.lines,
        "notes": "Tudo em pixels da imagem renderizada com o zoom acima.",
    }
    st.download_button(
        "⬇️ Download capture.json",
        data=json.dumps(export, indent=2),
        file_name="capture.json",
        mime="application/json",
    )

    if st.button("RESET TOTAL"):
        st.session_state.panel_corners = {p: [] for p in PANELS}
        st.session_state.axis_ticks = {a[0]: [] for a in AXES}
        st.session_state.lines = {k: [] for k in LINE_KEYS}
        st.session_state.pending_point = None
        st.rerun()

# Render
try:
    img = render_pdf(pdf_path, float(st.session_state.zoom))
except Exception as e:
    st.exception(e)
    st.stop()

preview = draw_overlay(img)

col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("Imagem (clica para capturar)")
    click = streamlit_image_coordinates(preview, key="click_img")
    if click and "x" in click and "y" in click:
        register_click(float(click["x"]), float(click["y"]))
        st.rerun()

with col2:
    st.subheader("Estado")

    st.write("Painéis completos:", {p: (len(st.session_state.panel_corners.get(p, [])) == 4) for p in PANELS})
    st.write("Ticks por eixo:", {a[0]: len(st.session_state.axis_ticks.get(a[0], [])) for a in AXES})
    st.write("Segmentos por reta:", {k: len(st.session_state.lines.get(k, [])) for k in LINE_KEYS})

    if st.session_state.pending_point is not None:
        st.warning(f"Ponto 1 pendente: {st.session_state.pending_point}")

    st.markdown("### Progresso detalhado")
    st.write("Panel corners:", {p: len(st.session_state.panel_corners.get(p, [])) for p in PANELS})
    st.write("Axis ticks:", {a[0]: len(st.session_state.axis_ticks.get(a[0], [])) for a in AXES})
    st.write("Lines:", {k: len(st.session_state.lines.get(k, [])) for k in LINE_KEYS})


