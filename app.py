from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pymupdf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =========================
# CONFIG
# =========================
PDF_NAME = "PA28POH-ground-roll.pdf"  # ✅ usar este pdf

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
    """Renderiza PDF -> PIL.Image (RGB)."""
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def ensure_png_pil(img: Image.Image) -> Image.Image:
    """
    FIX: força encode PNG e reabre para garantir compatibilidade com st_canvas,
    evitando crashes por falta de metadata/format.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # força carregar antes de buf sair de scope
    return out


def extract_last_line_object(canvas_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not canvas_json:
        return None
    objs = canvas_json.get("objects") or []
    for obj in reversed(objs):
        if obj.get("type") == "line":
            return obj
    return None


def normalize_line_obj(obj: Dict[str, Any]) -> Dict[str, float]:
    """
    Converte 'line' do fabric.js para coords absolutas na imagem:
    x = left + x1, y = top + y1, etc.
    """
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    x1 = float(obj["x1"]) + left
    y1 = float(obj["y1"]) + top
    x2 = float(obj["x2"]) + left
    y2 = float(obj["y2"]) + top
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def init_state():
    if "captures" not in st.session_state:
        st.session_state.captures = {}  # type: Dict[str, List[Dict[str, float]]]
    if "selected_key" not in st.session_state:
        st.session_state.selected_key = LINE_KEYS[0]
    if "target_per_line" not in st.session_state:
        st.session_state.target_per_line = CAPTURES_PER_LINE_DEFAULT
    if "zoom" not in st.session_state:
        st.session_state.zoom = 2.0


def count_done(key: str) -> int:
    return len(st.session_state.captures.get(key, []))


def total_done() -> int:
    return sum(len(v) for v in st.session_state.captures.values())


# =========================
# APP
# =========================
st.set_page_config(page_title="Captura guiada de retas", layout="wide")
init_state()

st.title("Captura guiada de retas — Landing Ground Roll")

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
            st.rerun()
    with c2:
        if st.button("🗑️ Reset total"):
            st.session_state.captures = {}
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

# Render PDF -> PIL -> PNG PIL (FIX)
try:
    base = render_pdf_page_to_pil(pdf_path, page_index=0, zoom=st.session_state.zoom)
    img = ensure_png_pil(base)
except Exception as e:
    st.exception(e)
    st.stop()

img_w, img_h = img.size

colA, colB = st.columns([1.35, 1])

with colA:
    st.subheader("Desenha uma linha por cima da reta selecionada")
    canvas = st_canvas(
        background_image=img,   # ✅ agora vem sempre como PNG "de verdade"
        drawing_mode="line",
        stroke_width=3,
        stroke_color="#ff0000",
        fill_color="rgba(255,0,0,0)",
        update_streamlit=True,
        height=img_h,
        width=img_w,
        key=f"canvas_{st.session_state.selected_key}",
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("💾 Guardar captura (última linha)"):
            if canvas.json_data is None:
                st.error("Ainda não desenhaste nada.")
            else:
                obj = extract_last_line_object(canvas.json_data)
                if not obj:
                    st.error("Não encontrei nenhuma 'line'. Desenha uma linha.")
                else:
                    coords = normalize_line_obj(obj)
                    st.session_state.captures.setdefault(st.session_state.selected_key, []).append(coords)
                    st.success(f"Guardado em {st.session_state.selected_key}: {coords}")
    with b2:
        if st.button("➡️ Próxima reta"):
            idx = LINE_KEYS.index(st.session_state.selected_key)
            st.session_state.selected_key = LINE_KEYS[(idx + 1) % len(LINE_KEYS)]
            st.rerun()

with colB:
    st.subheader("Resumo")
    st.write("Total de capturas:", total_done())

    st.markdown("### Capturas desta reta")
    st.code(json.dumps(st.session_state.captures.get(st.session_state.selected_key, []), indent=2), language="json")

    st.markdown("### Checklist")
    for k in LINE_KEYS:
        st.write(f"- `{k}`: {count_done(k)}/{st.session_state.target_per_line}")
