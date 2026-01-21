from __future__ import annotations

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
PDF_NAME = "PA28POH-ground-roll-20.pdf"

# Lista de retas “relevantes” (edita/expande à vontade)
LINE_KEYS = [
    "sea_level",
    "ft_2000",
    "ft_4000",
    "ft_6000",
    "ft_7000",
    "isa",
    "isa_m15",
    "isa_p15",
    # vento (exemplos)
    "headwind_line_0",
    "headwind_line_5",
    "headwind_line_10",
    "headwind_line_15",
]

# Quantas capturas por reta queres?
# (Normalmente 1 chega: uma linha com 2 pontos. Se quiseres refinar, podes capturar 2-3 e depois eu faço média.)
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


def render_pdf_page_to_image(pdf_path: str, page_index: int = 0, zoom: float = 2.0) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def extract_last_line_object(canvas_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Apanha o último objeto do tipo 'line' desenhado no canvas."""
    if not canvas_json:
        return None
    objs = canvas_json.get("objects") or []
    for obj in reversed(objs):
        if obj.get("type") == "line":
            return obj
    return None


def normalize_line_obj(obj: Dict[str, Any]) -> Dict[str, float]:
    """
    Converte o objeto 'line' do fabric.js para coordenadas absolutas na imagem:
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
    if "stroke_pdf" not in st.session_state:
        st.session_state.stroke_pdf = 2.0


def count_done(key: str) -> int:
    return len(st.session_state.captures.get(key, []))


def total_done() -> int:
    return sum(len(v) for v in st.session_state.captures.values())


def all_keys_status() -> List[Dict[str, Any]]:
    rows = []
    for k in LINE_KEYS:
        rows.append({"reta": k, "capturas": count_done(k), "alvo": st.session_state.target_per_line})
    return rows


# =========================
# APP
# =========================
st.set_page_config(page_title="Captura de Retas (Nomograma)", layout="wide")
init_state()

st.title("Captura guiada de retas — Nomograma (POH)")

try:
    pdf_path = locate_pdf(PDF_NAME)
except Exception as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("1) Config")
    st.session_state.zoom = st.slider("Zoom da imagem", 1.0, 4.0, st.session_state.zoom, 0.1)
    st.session_state.target_per_line = st.number_input(
        "Capturas por reta (recomendado: 1 ou 2)",
        min_value=1,
        max_value=5,
        value=int(st.session_state.target_per_line),
        step=1,
    )

    st.session_state.stroke_pdf = st.slider("Espessura (apenas preview)", 1.0, 8.0, st.session_state.stroke_pdf, 0.5)

    st.header("2) Seleciona a reta")
    st.session_state.selected_key = st.selectbox("Reta atual", LINE_KEYS, index=LINE_KEYS.index(st.session_state.selected_key))

    done = count_done(st.session_state.selected_key)
    target = st.session_state.target_per_line

    st.markdown("### Instruções")
    st.write(f"**Reta atual:** `{st.session_state.selected_key}`")
    st.write(f"**Progresso:** {done}/{target} capturas")

    st.info(
        "Desenha **uma única linha** por cima da reta selecionada.\n"
        "Depois carrega **Guardar captura**.\n\n"
        "Se desenhares várias linhas, a app guarda **a última**."
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Limpar capturas desta reta"):
            st.session_state.captures[st.session_state.selected_key] = []
            st.rerun()
    with c2:
        if st.button("🗑️ Reset TOTAL"):
            st.session_state.captures = {}
            st.rerun()

    st.divider()
    st.header("3) Export")
    export = {
        "pdf_name": PDF_NAME,
        "zoom_used_for_capture": float(st.session_state.zoom),
        "image_space": "pixel",
        "notes": "Coordenadas em pixels no espaço da imagem renderizada (com o zoom selecionado).",
        "captures": st.session_state.captures,
    }
    st.download_button(
        "⬇️ Download lines.json",
        data=json.dumps(export, indent=2),
        file_name="lines.json",
        mime="application/json",
    )
    st.caption("Quando acabares, faz download do JSON e traz-me aqui.")

# Render da imagem
img = render_pdf_page_to_image(pdf_path, page_index=0, zoom=st.session_state.zoom)
img_w, img_h = img.size

colA, colB = st.columns([1.35, 1])

with colA:
    st.subheader("Canvas (desenha a reta)")
    canvas = st_canvas(
        background_image=img,
        drawing_mode="line",
        stroke_width=3,
        stroke_color="#ff0000",
        fill_color="rgba(255, 0, 0, 0.0)",
        update_streamlit=True,
        height=img_h,
        width=img_w,
        key=f"canvas_{st.session_state.selected_key}",
    )

    save_col1, save_col2 = st.columns([1, 1])
    with save_col1:
        if st.button("💾 Guardar captura (última linha desenhada)"):
            if canvas.json_data is None:
                st.error("Ainda não desenhaste nada.")
            else:
                obj = extract_last_line_object(canvas.json_data)
                if not obj:
                    st.error("Não encontrei nenhum objeto do tipo 'line'. Desenha uma linha.")
                else:
                    coords = normalize_line_obj(obj)
                    st.session_state.captures.setdefault(st.session_state.selected_key, []).append(coords)
                    st.success(f"Guardado: {st.session_state.selected_key} → {coords}")
    with save_col2:
        if st.button("➡️ Próxima reta"):
            # passa para a próxima na lista
            idx = LINE_KEYS.index(st.session_state.selected_key)
            st.session_state.selected_key = LINE_KEYS[(idx + 1) % len(LINE_KEYS)]
            st.rerun()

with colB:
    st.subheader("Estado / Dados guardados")

    # Tabela simples de progresso
    status_rows = all_keys_status()
    st.dataframe(status_rows, use_container_width=True, hide_index=True)

    st.markdown("### Capturas desta reta")
    current = st.session_state.captures.get(st.session_state.selected_key, [])
    st.code(json.dumps(current, indent=2), language="json")

    st.markdown("### Total de capturas")
    st.write(total_done())

    st.markdown("### Dicas rápidas")
    st.write(
        "- Faz zoom suficiente para acertar nos endpoints.\n"
        "- Para uma reta longa, desenha de extremo a extremo.\n"
        "- Se uma reta for curvada (não devia), faz 2 capturas em segmentos."
    )
