import io
from pathlib import Path

import streamlit as st
import pymupdf  # ✅ PyMuPDF (import robusto)
from PIL import Image, ImageDraw

# =========================
# CONFIG
# =========================

PDF_NAME = "PA28POH-ground-roll-20.pdf"  # mesmo diretório do app.py

# Tamanho do sistema de coordenadas das tuas coords (o print)
# Se as tuas coords vierem desse print, deixa assim.
REF_W = 1822
REF_H = 1178

# =========================
# COORDENADAS (EDITA AQUI)
# =========================
# Cada item é uma polilinha: lista de pontos [(x1,y1), (x2,y2), ...]
LINES = [
    # EXEMPLOS (substitui pelas tuas):
    [(122, 84), (170, 52), (217, 52), (265, 52)],
    [(396, 137), (409, 86), (397, 67)],
]

# Opcional: pontos soltos
POINTS = [
    # (135, 202),
]


# =========================
# FUNÇÕES
# =========================

def locate_pdf(pdf_name: str) -> str:
    """Procura o PDF no CWD e na pasta do ficheiro do app (robusto para Streamlit Cloud)."""
    candidates = [Path.cwd() / pdf_name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / pdf_name)

    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        f"Não encontrei '{pdf_name}'. Procurei em: " + ", ".join(str(x) for x in candidates)
    )


def render_pdf_page_to_image(pdf_path: str, page_index: int = 0, zoom: float = 2.0) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    mat = pymupdf.Matrix(zoom, zoom)
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
) -> Image.Image:
    """
    Desenha polilinhas por cima da imagem, convertendo coords (ref_w,ref_h) -> tamanho real da imagem.
    offset_x/offset_y estão em "pixels do sistema REF".
    """
    w, h = img.size
    sx = (w / ref_w) * scale_x
    sy = (h / ref_h) * scale_y

    # offset em px do espaço da imagem (aplicando o mesmo sx/sy)
    ox = offset_x * sx
    oy = offset_y * sy

    out = img.copy()
    draw = ImageDraw.Draw(out)

    for poly in lines:
        if len(poly) < 2:
            continue
        mapped = [((x * sx) + ox, (y * sy) + oy) for (x, y) in poly]
        draw.line(mapped, width=line_width, fill=(255, 0, 0))  # vermelho

    r = max(2, line_width)
    for (x, y) in points:
        px = (x * sx) + ox
        py = (y * sy) + oy
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
) -> bytes:
    """
    Cria um novo PDF (bytes) com o overlay desenhado como VETORES na página 0.
    offset_x/offset_y estão em "pixels do sistema REF".
    """
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect

    sx = (rect.width / ref_w) * scale_x
    sy = (rect.height / ref_h) * scale_y

    # offsets convertidos do espaço REF para espaço da página
    ox = (offset_x / ref_w) * rect.width
    oy = (offset_y / ref_h) * rect.height

    # linhas
    for poly in lines:
        if len(poly) < 2:
            continue
        for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
            p1 = pymupdf.Point((x1 * sx) + ox, (y1 * sy) + oy)
            p2 = pymupdf.Point((x2 * sx) + ox, (y2 * sy) + oy)
            page.draw_line(p1, p2, color=(1, 0, 0), width=stroke_width)

    # pontos (opcional)
    for (x, y) in points:
        p = pymupdf.Point((x * sx) + ox, (y * sy) + oy)
        page.draw_circle(p, radius=max(1.0, stroke_width * 1.2), color=(1, 0, 0), width=stroke_width)

    out_bytes = doc.tobytes()
    doc.close()
    return out_bytes


# =========================
# APP
# =========================

st.set_page_config(page_title="PDF Overlay (Performance Chart)", layout="wide")
st.title("PDF Overlay – Retas por cima do gráfico")

# Mostrar erros no próprio browser (em vez do “Oh no.” genérico)
try:
    pdf_path = locate_pdf(PDF_NAME)
except Exception as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Debug (paths)")
    st.write("CWD:", str(Path.cwd()))
    if "__file__" in globals():
        st.write("APP DIR:", str(Path(__file__).resolve().parent))
    st.write("PDF:", pdf_path)

    st.header("Ajustes")
    zoom = st.slider("Zoom do preview (imagem)", 1.0, 4.0, 2.0, 0.1)

    st.subheader("Transformação das coordenadas")
    scale_x = st.slider("Scale X", 0.8, 1.2, 1.0, 0.001)
    scale_y = st.slider("Scale Y", 0.8, 1.2, 1.0, 0.001)
    offset_x = st.slider("Offset X (px ref)", -300.0, 300.0, 0.0, 1.0)
    offset_y = st.slider("Offset Y (px ref)", -300.0, 300.0, 0.0, 1.0)

    st.subheader("Estilo")
    line_width_img = st.slider("Espessura (preview)", 1, 12, 4, 1)
    stroke_width_pdf = st.slider("Espessura (PDF)", 0.5, 8.0, 2.0, 0.5)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Preview (imagem + overlay)")
    try:
        base_img = render_pdf_page_to_image(pdf_path, page_index=0, zoom=zoom)
    except Exception as e:
        st.exception(e)
        st.stop()

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

with col2:
    st.subheader("Download do PDF com overlay (vetorial)")
    try:
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
    except Exception as e:
        st.exception(e)
        st.stop()

    st.download_button(
        label="⬇️ Download PDF com overlay",
        data=pdf_bytes,
        file_name="landing_ground_roll_overlay.pdf",
        mime="application/pdf",
    )

    st.caption("Se as linhas não baterem certo, ajusta Scale/Offset até alinhar e volta a fazer download.")

