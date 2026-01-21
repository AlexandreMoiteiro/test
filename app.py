from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import pymupdf
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# CONFIG
# =========================
PDF_NAME = "PA28POH-ground-roll.pdf"

LINE_KEYS = [
    "sea_level",
    "ft_2000",
    "ft_4000",
    "ft_6000",
    "ft_7000",
    "isa",
    "isa_m15",
    "isa_p15",
    "headwind_0",
    "headwind_5",
    "headwind_10",
    "headwind_15",
]

CAPTURES_PER_LINE_DEFAULT = 1


# =========================
# HELPERS
# =========================
def locate_pdf(pdf_name: str) -> str:
    candidates = [Path.cwd() / pdf_name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / pdf_name)
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Não encontrei '{pdf_name}'. Procurei em: " + ", ".join(str(x) for x in candidates)
    )


def render_pdf_page_to_pil(pdf_path: str, page_index: int = 0, zoom: float = 2.0) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def init_state():
    if "captures" not in st.session_state:
        st.session_state.captures = {}  # Dict[str, List[Dict[str,float]]]
    if "selected_key" not in st.session_state:
        st.session_state.selected_key = LINE_KEYS[0]
    if "target_per_line" not in st.session_state:
        st.session_state.target_per_line = CAPTURES_PER_LINE_DEFAULT
    if "zoom" not in st.session_state:
        st.session_state.zoom = 2.0
    if "pending_point" not in st.session_state:
        st.session_state.pending_point = None  # Optional[Tuple[float,float]]
    if "last_click" not in st.session_state:
        st.session_state.last_click = None


def count_done(key: str) -> int:
    return len(st.session_state.captures.get(key, []))


def total_done() -> int:
    return sum(len(v) for v in st.session_state.captures.values())


def draw_overlay_preview(img: Image.Image, captures: Dict[str, List[Dict[str, float]]], highlight_key: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # desenhar todas as retas guardadas (vermelho)
    for k, lines in captures.items():
        for ln in lines:
            x1, y1, x2, y2 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
            width = 5 if k == highlight_key else 3
            d.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=width)

    # se houver 1 ponto pendente, desenhar um circulo
    if st.session_state.pending_point is not None:
        x, y = st.session_state.pending_point
        r = 8
        d.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0), width=3)

    return out


def save_line(key: str, p1: Tuple[float, float], p2: Tuple[float, float]):
    st.session_state.captures.setdefault(key, []).append(
        {"x1": float(p1[0]), "y1": float(p1[1]), "x2": float(p2[0]), "y2": float(p2[1])}
    )


# =========================
# APP
# =========================
st.set_page_config(page_title="Captura de Retas (por cliques)", layout="wide")
init_state()

st.title("Captura guiada de retas — (clicar 2 pontos por reta)")

try:
    pdf_path = locate_pdf(PDF_NAME)
except Exception as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Config")
    st.session_state.zoom = st.slider("Zoom do preview", 1.0, 4.0, st.session_state.zoom, 0.1)
    st.session_state.target_per_line = st.number_input(
        "Capturas por reta (1 recomendado)",
        min_value=1, max_value=5,
        value=int(st.session_state.target_per_line),
        step=1,
    )

    st.divider()
    st.header("Reta atual")
    st.session_state.selected_key = st.selectbox(
        "Seleciona a reta",
        LINE_KEYS,
        index=LINE_KEYS.index(st.session_state.selected_key),
    )
    st.write("Progresso:", f"{count_done(st.session_state.selected_key)}/{st.session_state.target_per_line}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Limpar esta reta"):
            st.session_state.captures[st.session_state.selected_key] = []
            st.session_state.pending_point = None
            st.rerun()
    with c2:
        if st.button("🗑️ Reset total"):
            st.session_state.captures = {}
            st.session_state.pending_point = None
            st.rerun()

    st.divider()
    export = {
        "pdf_name": PDF_NAME,
        "zoom_used_for_capture": float(st.session_state.zoom),
        "coords_space": "image_pixels",
        "captures": st.session_state.captures,
    }
    st.download_button(
        "⬇️ Download lines.json",
        data=json.dumps(export, indent=2),
        file_name="lines.json",
        mime="application/json",
    )
    st.caption("Quando terminares, faz download do JSON e traz-me aqui.")

# Render PDF -> imagem
try:
    base_img = render_pdf_page_to_pil(pdf_path, page_index=0, zoom=st.session_state.zoom)
except Exception as e:
    st.exception(e)
    st.stop()

# Overlay para preview
preview = draw_overlay_preview(base_img, st.session_state.captures, st.session_state.selected_key)

colA, colB = st.columns([1.35, 1])

with colA:
    st.subheader("Clica 2 pontos para definir a reta")
    st.write(
        f"**Reta atual:** `{st.session_state.selected_key}`  \n"
        "- Clica o **ponto 1** (fica marcado a verde)  \n"
        "- Clica o **ponto 2** para guardar a reta"
    )

    # componente devolve {"x":..., "y":...} quando clicas
    click = streamlit_image_coordinates(preview, key="img_click")

    if click is not None and ("x" in click and "y" in click):
        x, y = float(click["x"]), float(click["y"])
        # evitar reprocessar o mesmo clique em reruns (heurística simples)
        if st.session_state.last_click != (x, y):
            st.session_state.last_click = (x, y)

            if st.session_state.pending_point is None:
                st.session_state.pending_point = (x, y)
                st.rerun()
            else:
                p1 = st.session_state.pending_point
                p2 = (x, y)
                save_line(st.session_state.selected_key, p1, p2)
                st.session_state.pending_point = None
                st.success(f"Guardado `{st.session_state.selected_key}`: {p1} -> {p2}")
                st.rerun()

    b1, b2 = st.columns(2)
    with b1:
        if st.button("↩️ Cancelar ponto 1"):
            st.session_state.pending_point = None
            st.rerun()
    with b2:
        if st.button("➡️ Próxima reta"):
            idx = LINE_KEYS.index(st.session_state.selected_key)
            st.session_state.selected_key = LINE_KEYS[(idx + 1) % len(LINE_KEYS)]
            st.session_state.pending_point = None
            st.rerun()

with colB:
    st.subheader("Resumo")
    st.write("Total de retas capturadas:", total_done())

    st.markdown("### Capturas desta reta")
    st.code(json.dumps(st.session_state.captures.get(st.session_state.selected_key, []), indent=2), language="json")

    st.markdown("### Checklist")
    for k in LINE_KEYS:
        st.write(f"- `{k}`: {count_done(k)}/{st.session_state.target_per_line}")

