from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
import pymupdf
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

PDF_NAME = "PA28POH-ground-roll.pdf"

# O que queremos capturar
PANELS = ["left", "middle", "right"]

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

    # Painel peso / vento (referências essenciais)
    "weight_ref_line",     # "REF. LINE MAX. WEIGHT 2,550 LBS."
    "wind_ref_zero",       # "REF. LINE ZERO WIND"
]

AXES = [
    # name, description
    ("oat_c", "Outside Air Temperature (°C)"),
    ("weight_x100_lb", "Weight x100 (lb)"),
    ("wind_kt", "Wind components (kt)"),
    ("ground_roll_ft", "Landing Ground Roll (ft)"),
]

def locate_pdf(name: str) -> str:
    candidates = [Path.cwd() / name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Não encontrei '{name}'")

def render_pdf(pdf_path: str, zoom: float) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def init_state():
    if "zoom" not in st.session_state:
        st.session_state.zoom = 2.0

    # panel corners: {panel: [ (x,y), ... 4 ]}
    if "panel_corners" not in st.session_state:
        st.session_state.panel_corners = {p: [] for p in PANELS}

    # axis ticks: {axis_name: [ {"value":..., "x":..., "y":...}, ... ]}
    if "axis_ticks" not in st.session_state:
        st.session_state.axis_ticks = {a[0]: [] for a in AXES}

    # lines: {line_key: [ {"x1":..,"y1":..,"x2":..,"y2":..}, ... ]}
    if "lines" not in st.session_state:
        st.session_state.lines = {k: [] for k in LINE_KEYS}

    # capture flow
    if "mode" not in st.session_state:
        st.session_state.mode = "panel"
    if "selected_panel" not in st.session_state:
        st.session_state.selected_panel = "left"
    if "selected_axis" not in st.session_state:
        st.session_state.selected_axis = AXES[0][0]
    if "axis_value" not in st.session_state:
        st.session_state.axis_value = 0.0
    if "selected_line" not in st.session_state:
        st.session_state.selected_line = LINE_KEYS[0]
    if "pending_point" not in st.session_state:
        st.session_state.pending_point = None  # for line capture
    if "last_click" not in st.session_state:
        st.session_state.last_click = None

def draw_overlay(img: Image.Image) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # Panels
    for p, pts in st.session_state.panel_corners.items():
        if len(pts) == 4:
            d.polygon(pts, outline=(0, 200, 255), width=4)
            d.text((pts[0][0]+5, pts[0][1]+5), f"{p}", fill=(0, 200, 255))

    # Axis ticks
    for axis_name, ticks in st.session_state.axis_ticks.items():
        for t in ticks:
            x, y = t["x"], t["y"]
            d.ellipse((x-6, y-6, x+6, y+6), outline=(255, 165, 0), width=3)
            d.text((x+8, y-8), f'{axis_name}:{t["value"]}', fill=(255, 165, 0))

    # Lines
    for k, segs in st.session_state.lines.items():
        for seg in segs:
            d.line([(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])], fill=(255, 0, 0), width=3)

    # Pending point (for line capture)
    if st.session_state.pending_point is not None:
        x, y = st.session_state.pending_point
        d.ellipse((x-8, y-8, x+8, y+8), outline=(0, 255, 0), width=4)

    return out

def register_click(x: float, y: float):
    # basic debounce
    if st.session_state.last_click == (x, y):
        return
    st.session_state.last_click = (x, y)

    mode = st.session_state.mode

    if mode == "panel":
        p = st.session_state.selected_panel
        pts = st.session_state.panel_corners[p]
        if len(pts) < 4:
            pts.append((x, y))

    elif mode == "axis":
        axis = st.session_state.selected_axis
        st.session_state.axis_ticks[axis].append({"value": float(st.session_state.axis_value), "x": x, "y": y})

    elif mode == "line":
        key = st.session_state.selected_line
        if st.session_state.pending_point is None:
            st.session_state.pending_point = (x, y)
        else:
            x1, y1 = st.session_state.pending_point
            st.session_state.lines[key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
            st.session_state.pending_point = None

st.set_page_config(layout="wide", page_title="Capture Nomogram Geometry")
init_state()

st.title("Captura guiada — painéis, eixos e retas (por cliques)")

pdf_path = locate_pdf(PDF_NAME)

with st.sidebar:
    st.session_state.zoom = st.slider("Zoom", 1.0, 4.0, st.session_state.zoom, 0.1)

    st.header("Modo de captura")
    st.session_state.mode = st.radio("Escolhe", ["panel", "axis", "line"], index=["panel","axis","line"].index(st.session_state.mode))

    if st.session_state.mode == "panel":
        st.subheader("Painel")
        st.session_state.selected_panel = st.selectbox("Qual painel?", PANELS, index=PANELS.index(st.session_state.selected_panel))
        st.write("Clica 4 cantos da grelha: TL → TR → BR → BL")
        if st.button("Limpar cantos deste painel"):
            st.session_state.panel_corners[st.session_state.selected_panel] = []
            st.rerun()

    elif st.session_state.mode == "axis":
        st.subheader("Eixo / Tick")
        st.session_state.selected_axis = st.selectbox("Qual eixo?", [a[0] for a in AXES], index=[a[0] for a in AXES].index(st.session_state.selected_axis))
        st.session_state.axis_value = st.number_input("Valor do tick", value=float(st.session_state.axis_value))
        st.caption("Clica exatamente em cima do tick/numero correspondente.")
        if st.button("Limpar ticks deste eixo"):
            st.session_state.axis_ticks[st.session_state.selected_axis] = []
            st.rerun()

    elif st.session_state.mode == "line":
        st.subheader("Reta")
        st.session_state.selected_line = st.selectbox("Qual reta?", LINE_KEYS, index=LINE_KEYS.index(st.session_state.selected_line))
        st.write("Clica 2 pontos (extremos) para guardar 1 segmento.")
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
    st.download_button("⬇️ Download capture.json", data=json.dumps(export, indent=2), file_name="capture.json", mime="application/json")

    if st.button("RESET TOTAL"):
        for p in PANELS:
            st.session_state.panel_corners[p] = []
        for a, _ in AXES:
            st.session_state.axis_ticks[a] = []
        for k in LINE_KEYS:
            st.session_state.lines[k] = []
        st.session_state.pending_point = None
        st.rerun()

img = render_pdf(pdf_path, st.session_state.zoom)
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
    st.write("Painéis completos:", {p: (len(pts)==4) for p, pts in st.session_state.panel_corners.items()})
    st.write("Ticks por eixo:", {a[0]: len(st.session_state.axis_ticks[a[0]]) for a in AXES})
    st.write("Segmentos por reta:", {k: len(st.session_state.lines[k]) for k in LINE_KEYS})
    if st.session_state.pending_point is not None:
        st.warning(f"Ponto 1 pendente: {st.session_state.pending_point}")



