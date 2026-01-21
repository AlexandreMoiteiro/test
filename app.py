
from pathlib import Path
import streamlit as st
import pymupdf  # PyMuPDF
from PIL import Image, ImageDraw

PDF_NAME = "PA28POH-ground-roll-20.pdf"

# Sistema de coordenadas do teu print (imagem 1822x1178)
REF_W = 1822
REF_H = 1178

# =========================================================
# RETAS / POLILINHAS (decididas por mim a partir dos pontos)
# =========================================================
# Convenção:
# - Cada item é uma polilinha: [(x1,y1), (x2,y2), ...]
# - Eu agrupei por alinhamento (colunas verticais do gráfico, diagonais, base, etc.)
LINES = [
    # --- Base inferior (linha de referência em baixo) ---
    [(121, 52), (178, 52), (265, 52), (409, 53)],

    # --- Coluna vertical ~ x=397 (meio) ---
    [(397, 97), (397, 118), (396, 137), (396, 175)],

    # --- Coluna vertical ~ x=409 (meio-direita) ---
    [(409, 53), (409, 150), (408, 183), (408, 214), (408, 247), (408, 271)],

    # --- Coluna vertical ~ x=436/438 (direita) ---
    [(438, 64), (438, 95), (437, 150), (435, 183), (434, 215), (436, 259)],

    # --- Coluna vertical ~ x=497 (extrema direita) ---
    [(497, 78), (497, 105), (496, 193), (496, 337)],

    # --- Diagonal (grupo 1) - zona esquerda/meio ---
    # (linha “a subir” para a esquerda no print)
    [(254, 270), (259, 256), (269, 228), (287, 193), (308, 136)],

    # --- Diagonal (grupo 2) - zona esquerda ---
    [(176, 210), (186, 187), (194, 171), (205, 156), (265, 52)],

    # --- Vertical curta ~ x=299/301 ---
    [(299, 276), (299, 249), (301, 204)],

    # --- Diagonal muito inclinada (3 pontos) ---
    [(158, 170), (159, 153), (178, 52)],

    # --- Segmento no painel de vento (2 pontos visíveis) ---
    [(468, 73), (497, 78)],

    # --- Segmento “passando” pelo ponto (368,180) (alinhado com a zona central) ---
    [(368, 180), (396, 175), (408, 183), (435, 183)],

    # --- Pequeno segmento na esquerda (pontos soltos visíveis) ---
    [(142, 189), (171, 222)],
]

# =========================================================
# FUNÇÕES (só render + overlay)
# =========================================================

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
    ref_w: int,
    ref_h: int,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    line_width: int,
) -> Image.Image:
    w, h = img.size
    sx = (w / ref_w) * scale_x
    sy = (h / ref_h) * scale_y

    ox = offset_x * sx
    oy = offset_y * sy

    out = img.copy()
    draw = ImageDraw.Draw(out)

    for poly in lines:
        if len(poly) < 2:
            continue
        mapped = [((x * sx) + ox, (y * sy) + oy) for (x, y) in poly]
        draw.line(mapped, width=line_width, fill=(255, 0, 0))

    return out

def pdf_with_vector_overlay(
    pdf_path: str,
    lines,
    ref_w: int,
    ref_h: int,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    stroke_width: float,
) -> bytes:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect

    sx = (rect.width / ref_w) * scale_x
    sy = (rect.height / ref_h) * scale_y

    ox = (offset_x / ref_w) * rect.width
    oy = (offset_y / ref_h) * rect.height

    for poly in lines:
        if len(poly) < 2:
            continue
        for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
            p1 = pymupdf.Point((x1 * sx) + ox, (y1 * sy) + oy)
            p2 = pymupdf.Point((x2 * sx) + ox, (y2 * sy) + oy)
            page.draw_line(p1, p2, color=(1, 0, 0), width=stroke_width)

    out_bytes = doc.tobytes()
    doc.close()
    return out_bytes

# =========================================================
# UI
# =========================================================

st.set_page_config(page_title="PDF Overlay (Landing Ground Roll)", layout="wide")
st.title("Landing Ground Roll — Overlay de Retas (pré-definidas)")

try:
    pdf_path = locate_pdf(PDF_NAME)
except Exception as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Ajustes (só para alinhar)")
    zoom = st.slider("Zoom do preview", 1.0, 4.0, 2.0, 0.1)

    scale_x = st.slider("Scale X", 0.8, 1.2, 1.0, 0.001)
    scale_y = st.slider("Scale Y", 0.8, 1.2, 1.0, 0.001)
    offset_x = st.slider("Offset X (px ref)", -300.0, 300.0, 0.0, 1.0)
    offset_y = st.slider("Offset Y (px ref)", -300.0, 300.0, 0.0, 1.0)

    line_width_img = st.slider("Espessura (preview)", 1, 12, 4, 1)
    stroke_width_pdf = st.slider("Espessura (PDF)", 0.5, 8.0, 2.0, 0.5)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Preview")
    try:
        base_img = render_pdf_page_to_image(pdf_path, page_index=0, zoom=zoom)
    except Exception as e:
        st.exception(e)
        st.stop()

    over_img = draw_overlay_on_image(
        base_img,
        lines=LINES,
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
    st.subheader("Download PDF com overlay (vetorial)")
    try:
        pdf_bytes = pdf_with_vector_overlay(
            pdf_path,
            lines=LINES,
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
        "⬇️ Download landing_ground_roll_overlay.pdf",
        data=pdf_bytes,
        file_name="landing_ground_roll_overlay.pdf",
        mime="application/pdf",
    )

st.caption("Se alguma reta estiver ligeiramente fora, ajusta Scale/Offset até bater certo.")

