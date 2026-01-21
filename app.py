import os
import io
import streamlit as st

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

# =========================
# CONFIG
# =========================

PDF_NAME = "PA28POH-ground-roll-20.pdf"  # <-- está na mesma pasta do app.py

# Tamanho do "sistema de coordenadas" das tuas coordenadas (o print que mostraste)
# (se as tuas coords vierem desse print, deixa assim)
REF_W = 1822
REF_H = 1178

# =========================
# COORDENADAS (EDITA AQUI)
# =========================
# Coloca aqui as tuas polilinhas.
# Cada item em LINES é uma lista de pontos [(x1,y1), (x2,y2), ...] que serão ligados em sequência.
#
# Exemplo mínimo (SUBSTITUI pelas tuas coords reais):
LINES = [
    # Exemplo: uma linha/rota
    [(122, 84), (170, 52), (217, 52), (265, 52)],
    # Exemplo: outra linha
    [(396, 137), (409, 86), (397, 67)],
]

# Se quiseres pontos soltos (opcional):
POINTS = [
    # (135, 202),
]


# =========================
# FUNÇÕES
# =========================

def render_pdf_page_to_image(pdf_path: str, page_index: int = 0, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def draw_overlay_on_image(
    img: Image.Image,
    lines,
    points,
    ref_w: int,
    ref_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    line_width: int = 4,
):
    """
    Desenha as polilinhas em cima da imagem, convertendo coords do espaço REF_(W,H)
    para o tamanho real da imagem (img.size), com ajustes opcionais.
    """
    w, h = img.size
    sx = (w / ref_w) * scale_x
    sy = (h / ref_h) * scale_y

    out = img.copy()
    draw = ImageDraw.Draw(out)

    # Polilinhas
    for poly in lines:
        if len(poly) < 2:
            continue
        mapped = [((x * sx) + offset_x, (y * sy) + offset_y) for (x, y) in poly]
        draw.line(mapped, width=line_width, fill=(255, 0, 0))  # vermelho

    # Pontos (opcional)
    r = max(2, line_width)
    for (x, y) in points:
        px = (x * sx) + offset_x
        py = (y * sy) + offset_y
        draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 0, 0), width=2)

    return out

def pdf_with_vector_overlay(
    pdf_path: str,
    lines,
    points,
    ref_w: int,
    ref_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    stroke_width: float = 2.0,
):
    """
    Cria um NOVO PDF baseado no original, desenhando as linhas como VETORES no PDF.
    Mapeia coords do espaço REF_(W,H) para o espaço da página (page.rect).
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect

    # Map ref coords -> page coords
    # Primeiro escala proporcional ao tamanho da página; depois aplica ajustes (offset em "ref pixels")
    # convertidos também para page coords.
    sx = (rect.width / ref_w) * scale_x
    sy = (rect.height / ref_h) * scale_y

    ox = (offset_x / ref_w) * rect.width
    oy = (offset_y / ref_h) * rect.height

    # Desenhar polilinhas
    for poly in lines:
        if len(poly) < 2:
            continue
        for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
            p1 = fitz.Point((x1 * sx) + ox, (y1 * sy) + oy)
            p2 = fitz.Point((x2 * sx) + ox, (y2 * sy) + oy)
            page.draw_line(p1, p2, color=(1, 0, 0), width=stroke_width)  # vermelho em RGB (0..1)

    # Pontos (opcional)
    for (x, y) in points:
        p = fitz.Point((x * sx) + ox, (y * sy) + oy)
        page.draw_circle(p, radius=max(1.0, stroke_width * 1.2), color=(1, 0, 0), width=stroke_width)

    # Exportar bytes
    out_bytes = doc.tobytes()
    doc.close()
    return out_bytes


# =========================
# UI
# =========================

st.set_page_config(page_title="PDF Overlay (Performance Chart)", layout="wide")
st.title("PDF Overlay – Retas por cima do gráfico")

pdf_path = os.path.join(os.path.dirname(__file__), PDF_NAME)
if not os.path.exists(pdf_path):
    st.error(f"Não encontrei o ficheiro '{PDF_NAME}' na mesma pasta do app.py.")
    st.stop()

with st.sidebar:
    st.header("Ajustes")
    zoom = st.slider("Zoom do preview (imagem)", 1.0, 4.0, 2.0, 0.1)

    st.subheader("Transformação das coordenadas")
    scale_x = st.slider("Scale X", 0.8, 1.2, 1.0, 0.001)
    scale_y = st.slider("Scale Y", 0.8, 1.2, 1.0, 0.001)
    offset_x = st.slider("Offset X (px ref)", -200.0, 200.0, 0.0, 1.0)
    offset_y = st.slider("Offset Y (px ref)", -200.0, 200.0, 0.0, 1.0)

    st.subheader("Estilo")
    line_width_img = st.slider("Espessura (preview)", 1, 12, 4, 1)
    stroke_width_pdf = st.slider("Espessura (PDF)", 0.5, 8.0, 2.0, 0.5)

col1, col2 = st.columns([1, 1])

# Preview
with col1:
    st.subheader("Preview (imagem + overlay)")
    base_img = render_pdf_page_to_image(pdf_path, page_index=0, zoom=zoom)
    over_img = draw_overlay_on_image(
        base_img,
        lines=LINES,
        points=POINTS,
        ref_w=REF_W,
        ref_h=REF_H,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
        line_width=line_width_img,
    )
    st.image(over_img, use_container_width=True)

# Gerar PDF com overlay vetorial
with col2:
    st.subheader("Download do PDF com overlay (vetorial)")
    pdf_bytes = pdf_with_vector_overlay(
        pdf_path,
        lines=LINES,
        points=POINTS,
        ref_w=REF_W,
        ref_h=REF_H,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
        stroke_width=stroke_width_pdf,
    )

    st.download_button(
        label="⬇️ Download PDF com overlay",
        data=pdf_bytes,
        file_name="landing_ground_roll_overlay.pdf",
        mime="application/pdf",
    )

    st.caption("Dica: se as linhas estiverem ligeiramente fora, ajusta Scale/Offset até bater certo e volta a fazer download.")
