from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from streamlit_image_coordinates import streamlit_image_coordinates

PDF_NAME = "PA28POH-ground-roll.pdf"

# ---------- O que vamos capturar ----------
PANELS = ["panel_oat", "panel_weight", "panel_wind"]

AXIS_POINTS = [
    # y axis (ground roll)
    ("y_gr_600",  "Clica no tick **600 ft** no eixo de Ground Roll (direita)"),
    ("y_gr_1300", "Clica no tick **1300 ft** no eixo de Ground Roll (direita)"),

    # x axis oat
    ("x_oat_-10", "Clica no tick **-10°C** no eixo OAT (painel esquerdo)"),
    ("x_oat_50",  "Clica no tick **50°C** no eixo OAT (painel esquerdo)"),

    # x axis weight
    ("x_wt_25",   "Clica no tick **25** no eixo WEIGHT x100 (painel do meio)"),
    ("x_wt_20",   "Clica no tick **20** no eixo WEIGHT x100 (painel do meio)"),

    # x axis wind
    ("x_wind_0",  "Clica no tick **0 kt** no eixo WIND (painel direito)"),
    ("x_wind_15", "Clica no tick **15 kt** no eixo WIND (painel direito)"),
]

LINES = [
    # Painel OAT (altitude / ISA)
    "sea_level",
    "ft_2000",
    "ft_4000",
    "ft_6000",
    "ft_7000",
    "isa",
    "isa_m15",
    "isa_p35",

    # Painel WEIGHT
    "ref_line_max_weight_2550",

    # Painel WIND
    "ref_line_zero_wind",
    "headwind_main_1",
    "headwind_main_2",
    "headwind_main_3",
]

def locate_pdf(pdf_name: str) -> str:
    for c in [Path.cwd()/pdf_name, Path(__file__).resolve().parent/pdf_name]:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Não encontrei '{pdf_name}' ao lado do app.py")

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
    if "step" not in st.session_state:
        st.session_state.step = "panels"  # panels -> axes -> lines -> export
    if "panel_rects" not in st.session_state:
        st.session_state.panel_rects = {}  # name -> {"x1","y1","x2","y2"}
    if "axis_pts" not in st.session_state:
        st.session_state.axis_pts = {}     # key -> {"x","y"}
    if "lines" not in st.session_state:
        st.session_state.lines = {}        # name -> [{"x1","y1","x2","y2"}, ...] (permite 2 capturas)
    if "pending" not in st.session_state:
        st.session_state.pending = None    # guarda 1º clique (x,y)
    if "current_key" not in st.session_state:
        st.session_state.current_key = PANELS[0]
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "last_click" not in st.session_state:
        st.session_state.last_click = None

def overlay(img: Image.Image) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # rects dos painéis (verde)
    for name, r in st.session_state.panel_rects.items():
        d.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=(0, 255, 0), width=4)
        d.text((r["x1"]+5, r["y1"]+5), name, fill=(0, 255, 0))

    # pontos de eixo (azul)
    for k, p in st.session_state.axis_pts.items():
        x, y = p["x"], p["y"]
        d.ellipse([x-6, y-6, x+6, y+6], outline=(0, 128, 255), width=3)
        d.text((x+8, y-8), k, fill=(0, 128, 255))

    # linhas (vermelho)
    for name, arr in st.session_state.lines.items():
        for ln in arr:
            d.line([(ln["x1"], ln["y1"]), (ln["x2"], ln["y2"])], fill=(255, 0, 0), width=4)
        # label perto do primeiro segmento
        if arr:
            ln0 = arr[0]
            d.text((ln0["x1"]+5, ln0["y1"]+5), name, fill=(255, 0, 0))

    # ponto pendente (amarelo)
    if st.session_state.pending is not None:
        x, y = st.session_state.pending
        d.ellipse([x-8, y-8, x+8, y+8], outline=(255, 255, 0), width=4)

    return out

def next_item():
    st.session_state.pending = None
    st.session_state.current_index += 1

def set_current_from_step():
    if st.session_state.step == "panels":
        st.session_state.current_key = PANELS[min(st.session_state.current_index, len(PANELS)-1)]
    elif st.session_state.step == "axes":
        st.session_state.current_key = AXIS_POINTS[min(st.session_state.current_index, len(AXIS_POINTS)-1)][0]
    elif st.session_state.step == "lines":
        st.session_state.current_key = LINES[min(st.session_state.current_index, len(LINES)-1)]
    else:
        st.session_state.current_key = ""

def step_done() -> bool:
    if st.session_state.step == "panels":
        return st.session_state.current_index >= len(PANELS)
    if st.session_state.step == "axes":
        return st.session_state.current_index >= len(AXIS_POINTS)
    if st.session_state.step == "lines":
        return st.session_state.current_index >= len(LINES)
    return True

def handle_click(x: float, y: float):
    # evita processar o mesmo clique em reruns
    if st.session_state.last_click == (x, y):
        return
    st.session_state.last_click = (x, y)

    if st.session_state.step == "panels":
        key = st.session_state.current_key
        if st.session_state.pending is None:
            st.session_state.pending = (x, y)
        else:
            x1, y1 = st.session_state.pending
            st.session_state.panel_rects[key] = {
                "x1": float(min(x1, x)), "y1": float(min(y1, y)),
                "x2": float(max(x1, x)), "y2": float(max(y1, y)),
            }
            next_item()

    elif st.session_state.step == "axes":
        key = st.session_state.current_key
        st.session_state.axis_pts[key] = {"x": float(x), "y": float(y)}
        next_item()

    elif st.session_state.step == "lines":
        key = st.session_state.current_key
        if st.session_state.pending is None:
            st.session_state.pending = (x, y)
        else:
            x1, y1 = st.session_state.pending
            st.session_state.lines.setdefault(key, []).append(
                {"x1": float(x1), "y1": float(y1), "x2": float(x), "y2": float(y)}
            )
            next_item()

def export_payload() -> Dict[str, Any]:
    return {
        "pdf_name": PDF_NAME,
        "zoom_used_for_capture": float(st.session_state.zoom),
        "panel_rects": st.session_state.panel_rects,
        "axis_points": st.session_state.axis_pts,
        "lines": st.session_state.lines,
        "notes": "Tudo em pixels da imagem renderizada (zoom acima).",
    }

# ---------- UI ----------
st.set_page_config(layout="wide", page_title="Nomograma Capture Wizard")
init_state()

pdf_path = locate_pdf(PDF_NAME)

with st.sidebar:
    st.header("Config")
    st.session_state.zoom = st.slider("Zoom", 1.0, 4.0, st.session_state.zoom, 0.1)

    st.divider()
    st.header("Modo")
    mode = st.radio("Etapa", ["panels", "axes", "lines", "export"], index=["panels","axes","lines","export"].index(st.session_state.step))
    st.session_state.step = mode
    st.session_state.current_index = 0
    set_current_from_step()
    st.session_state.pending = None

    st.divider()
    st.header("Controlo")
    if st.button("↩️ Cancelar ponto pendente"):
        st.session_state.pending = None
        st.rerun()
    if st.button("🗑️ Reset tudo"):
        for k in ["panel_rects","axis_pts","lines"]:
            st.session_state[k] = {}
        st.session_state.pending = None
        st.rerun()

    st.divider()
    payload = export_payload()
    st.download_button("⬇️ Download schema.json", data=json.dumps(payload, indent=2), file_name="schema.json", mime="application/json")

# instruções do passo
set_current_from_step()

instr = ""
if st.session_state.step == "panels" and not step_done():
    instr = f"**PAINEL:** `{st.session_state.current_key}` → clica canto sup-esq e depois inf-dir."
elif st.session_state.step == "axes" and not step_done():
    key, msg = AXIS_POINTS[st.session_state.current_index]
    instr = f"**EIXO:** `{key}` → {msg}"
elif st.session_state.step == "lines" and not step_done():
    instr = f"**RETA:** `{st.session_state.current_key}` → clica 2 pontos (extremos) da reta."
else:
    instr = "Etapa concluída. Faz download do `schema.json` e volta aqui."

st.title("Wizard de captura do nomograma")
st.info(instr)

# imagem + overlay
base = render_pdf(pdf_path, st.session_state.zoom)
img = overlay(base)

col1, col2 = st.columns([1.3, 1])

with col1:
    click = streamlit_image_coordinates(img, key="img")
    if click and "x" in click and "y" in click:
        handle_click(float(click["x"]), float(click["y"]))
        st.rerun()

with col2:
    st.subheader("Estado")
    st.write("Step:", st.session_state.step)
    st.write("Index:", st.session_state.current_index)

    st.markdown("### Painéis")
    st.code(json.dumps(st.session_state.panel_rects, indent=2), language="json")

    st.markdown("### Eixos")
    st.code(json.dumps(st.session_state.axis_pts, indent=2), language="json")

    st.markdown("### Linhas")
    st.code(json.dumps(st.session_state.lines, indent=2)[:2000] + ("\n... (truncado)" if len(json.dumps(st.session_state.lines)) > 2000 else ""), language="json")


