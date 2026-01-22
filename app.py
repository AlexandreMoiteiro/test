from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import pymupdf
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# CONFIG (defaults)
# =========================
DEFAULT_PDF = "PA28POH-ground-roll.pdf"  # podes trocar
MODE_CONFIG = {
    # 1) Landing Ground Roll (3 painéis)
    "landing_gr": {
        "title": "Landing Ground Roll",
        "default_page": 0,
        "capture_file": "capture_landing_gr.json",
        "panels": ["left", "middle", "right"],
        "axis_ticks": {
            "oat_c": {"label": "OAT (°C)", "panel": "left"},
            "weight_x100_lb": {"label": "Weight (lb/100)", "panel": "middle"},
            "wind_kt": {"label": "Wind (kt)", "panel": "right"},
            "ground_roll_ft": {"label": "Ground Roll (ft)", "panel": "right"},  # eixo vertical à direita
        },
        "lines": [
            # ISA family
            "isa_m15", "isa", "isa_p35",
            # PA family
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_7000",
            # refs
            "weight_ref_line", "wind_ref_zero",
        ],
        "guides": {
            "middle": {"label": "Guia diagonais (WEIGHT)", "count_hint": "2–4 linhas grossas"},
            "right": {"label": "Guia diagonais (WIND)", "count_hint": "2–4 linhas grossas"},
        },
    },

    # 2) Flaps Up Takeoff Ground Roll (2 painéis: left (ISA/PA) + right (Weight+Wind+Result))
    "takeoff_gr": {
        "title": "Flaps Up Takeoff Ground Roll",
        "default_page": 0,
        "capture_file": "capture_takeoff_gr.json",
        "panels": ["left", "right"],
        "axis_ticks": {
            "oat_c": {"label": "OAT (°C)", "panel": "left"},
            "weight_x100_lb": {"label": "Weight (lb/100)", "panel": "right"},
            "wind_kt": {"label": "Wind (kt)", "panel": "right"},
            "takeoff_gr_ft": {"label": "Takeoff Ground Roll (ft)", "panel": "right"},
        },
        "lines": [
            # ISA family
            "isa_m15", "isa", "isa_p35",
            # PA family (no takeoff vai até 8000 no teu gráfico)
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_8000",
            # refs
            "weight_ref_line", "wind_ref_zero",
        ],
        "guides": {
            "right": {"label": "Guia diagonais (WEIGHT/WIND/RESULT)", "count_hint": "4–6 linhas grossas"},
        },
    },

    # 3) Climb Performance (1 painel)
    "climb_perf": {
        "title": "Climb Performance",
        "default_page": 0,
        "capture_file": "capture_climb_perf.json",
        "panels": ["main"],
        "axis_ticks": {
            "oat_c": {"label": "OAT (°C)", "panel": "main"},
            "roc_fpm": {"label": "Rate of Climb (FPM)", "panel": "main"},
        },
        "lines": [
            # ISA family +35 / ISA / -15 (no gráfico tens esses)
            "isa_m15", "isa", "isa_p35",
            # pressure altitude family (linhas grossas inclinadas dentro do “paralelogramo”)
            "pa_sea_level", "pa_1000", "pa_2000", "pa_3000", "pa_4000", "pa_5000",
            "pa_6000", "pa_7000", "pa_8000", "pa_9000", "pa_10000", "pa_11000",
            "pa_12000", "pa_13000",
        ],
        "guides": {
            "main": {"label": "Guia (opcional) para paralelas internas", "count_hint": "0–4 (se precisares)"},
        },
    },
}


# =========================
# IO / Render
# =========================
def locate_file(name: str) -> Optional[str]:
    candidates = [Path.cwd() / name]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def load_pdf_bytes(upload) -> bytes:
    if upload is not None:
        return upload.read()
    pdf_path = locate_file(DEFAULT_PDF)
    if not pdf_path:
        raise FileNotFoundError(f"Não encontrei {DEFAULT_PDF}")
    return Path(pdf_path).read_bytes()


@st.cache_data(show_spinner=False)
def render_pdf_page_to_pil_cached(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


# =========================
# Capture structure
# =========================
def new_capture(mode_key: str, zoom: float = 2.3) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode_key]
    cap = {
        "mode": mode_key,
        "zoom": float(zoom),
        "page_index": int(cfg["default_page"]),
        "panel_corners": {p: [] for p in cfg["panels"]},   # each: 4 points (x,y)
        "axis_ticks": {k: [] for k in cfg["axis_ticks"].keys()},
        "lines": {k: [] for k in cfg["lines"]},            # each: list of segments
        "guides": {p: [] for p in cfg.get("guides", {}).keys()},  # each: list of segments
    }
    return cap


def ensure_capture_mode(cap: Dict[str, Any], mode_key: str) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode_key]
    cap.setdefault("mode", mode_key)
    cap.setdefault("zoom", 2.3)
    cap.setdefault("page_index", int(cfg["default_page"]))
    cap.setdefault("panel_corners", {})
    for p in cfg["panels"]:
        cap["panel_corners"].setdefault(p, [])
    cap.setdefault("axis_ticks", {})
    for k in cfg["axis_ticks"].keys():
        cap["axis_ticks"].setdefault(k, [])
    cap.setdefault("lines", {})
    for k in cfg["lines"]:
        cap["lines"].setdefault(k, [])
    cap.setdefault("guides", {})
    for p in cfg.get("guides", {}).keys():
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

    # lines (red)
    for key, segs in cap.get("lines", {}).items():
        for s in segs:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

    # guides (blue)
    for panel, segs in cap.get("guides", {}).items():
        for s in segs:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PA28 Capture Studio (3 modos)", layout="wide")
st.title("Capture Studio — Landing GR / Takeoff GR / Climb Performance")

with st.sidebar:
    mode_key = st.selectbox(
        "Escolhe o gráfico",
        options=list(MODE_CONFIG.keys()),
        format_func=lambda k: MODE_CONFIG[k]["title"],
    )

    pdf_up = st.file_uploader("PDF (opcional)", type=["pdf"])
    pdf_bytes = load_pdf_bytes(pdf_up)

    st.markdown("---")
    st.header("Render")
    zoom = st.number_input("Zoom", value=2.3, step=0.1)

    page_index = st.number_input("Página (0-index)", value=MODE_CONFIG[mode_key]["default_page"], step=1)

    st.markdown("---")
    if st.button("Novo capture para este modo (reset)"):
        st.session_state.capture = new_capture(mode_key, zoom=float(zoom))
        st.session_state.capture["page_index"] = int(page_index)
        st.success("Capture reset.")

# load capture from session or disk
if "capture" not in st.session_state:
    cap_path = locate_file(MODE_CONFIG[mode_key]["capture_file"])
    if cap_path:
        cap = json.loads(Path(cap_path).read_text(encoding="utf-8"))
    else:
        cap = new_capture(mode_key, zoom=float(zoom))
    st.session_state.capture = cap

cap = st.session_state.capture
cap = ensure_capture_mode(cap, mode_key)
cap["zoom"] = float(zoom)
cap["page_index"] = int(page_index)

cfg = MODE_CONFIG[mode_key]

img = render_pdf_page_to_pil_cached(pdf_bytes, int(page_index), float(zoom))
show_overlay = st.checkbox("Mostrar overlay", value=True)
img_show = overlay_draw(img, cap) if show_overlay else img

tabs = st.tabs(["Editor", "Export/Apagar", "Preview"])

# =========================
# Editor tab
# =========================
with tabs[0]:
    st.subheader(cfg["title"])

    left, right = st.columns([1.4, 1])

    with right:
        st.markdown("### O que queres capturar agora?")

        options = []
        options.append("Panel corners (4 cliques)")
        options += [f"Tick: {k}" for k in cfg["axis_ticks"].keys()]
        options += [f"Line segment: {k}" for k in cfg["lines"]]
        for p in cfg.get("guides", {}).keys():
            options.append(f"Guide segment: {p}")

        task = st.selectbox("Modo", options)

        tick_val = None
        if task.startswith("Tick:"):
            axis_key = task.split("Tick: ")[1].strip()
            tick_val = st.number_input("Valor do tick", value=0.0, step=1.0)

        st.markdown("---")
        st.markdown("### Dicas rápidas")
        st.write("- **Panel corners**: clica 4 cantos do painel (ordem qualquer).")
        st.write("- **Ticks**: clica no traço do eixo e mete o valor correto.")
        st.write("- **Line segment / Guide segment**: 2 cliques (início e fim).")

        if task.startswith("Guide segment:"):
            p = task.split("Guide segment: ")[1].strip()
            st.info(f"{cfg['guides'][p]['label']} — {cfg['guides'][p]['count_hint']}")

    with left:
        st.markdown("### Clica na imagem")
        click = streamlit_image_coordinates(np.array(img_show), key=f"click_{mode_key}_{page_index}")

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
                    # escolher qual panel
                    panel = st.selectbox("Em que painel são estes corners?", cfg["panels"], key=f"panel_pick_{len(cap['panel_corners'])}")
                    cap["panel_corners"][panel] = st.session_state.pending_corners
                    st.session_state.pending_corners = []
                    st.success(f"Panel corners guardados para '{panel}'.")

            elif task.startswith("Tick:"):
                axis_key = task.split("Tick: ")[1].strip()
                cap["axis_ticks"][axis_key].append({"x": x, "y": y, "value": float(tick_val)})
                st.success(f"Tick adicionado em {axis_key}.")

            elif task.startswith("Line segment:"):
                key = task.split("Line segment: ")[1].strip()
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto guardado. Clica no segundo.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["lines"][key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success(f"Segmento adicionado em lines['{key}'].")

            elif task.startswith("Guide segment:"):
                panel = task.split("Guide segment: ")[1].strip()
                if st.session_state.pending_point is None:
                    st.session_state.pending_point = (x, y)
                    st.info("Primeiro ponto da guia guardado. Clica no segundo.")
                else:
                    x1, y1 = st.session_state.pending_point
                    st.session_state.pending_point = None
                    cap["guides"][panel].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                    st.success(f"Guia adicionada em guides['{panel}'].")

        st.markdown("#### Estado rápido")
        st.write({
            "panel_corners": {p: len(cap["panel_corners"][p]) for p in cfg["panels"]},
            "ticks": {k: len(cap["axis_ticks"][k]) for k in cfg["axis_ticks"].keys()},
            "lines": {k: len(cap["lines"][k]) for k in cfg["lines"]},
            "guides": {k: len(cap["guides"].get(k, [])) for k in cap.get("guides", {}).keys()},
        })


# =========================
# Export / delete tab
# =========================
with tabs[1]:
    st.subheader("Exportar e apagar outliers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Export JSON")
        txt = json.dumps(cap, indent=2)
        st.download_button("⬇️ Download capture JSON", data=txt, file_name=cfg["capture_file"], mime="application/json")

        if st.button("Guardar na pasta da app"):
            Path(cfg["capture_file"]).write_text(txt, encoding="utf-8")
            st.success(f"Gravado: {cfg['capture_file']}")

        st.markdown("---")
        st.markdown("### Ver JSON")
        st.code(txt, language="json")

    with col2:
        st.markdown("### Apagar coisas")

        what = st.selectbox("O que apagar?", ["Ticks", "Lines", "Guides", "Panel corners"])
        if what == "Ticks":
            axis = st.selectbox("Axis", list(cfg["axis_ticks"].keys()))
            items = cap["axis_ticks"][axis]
            if not items:
                st.info("Sem ticks.")
            else:
                opts = [f"[{i}] value={t['value']} (x={int(t['x'])}, y={int(t['y'])})" for i, t in enumerate(items)]
                pick = st.multiselect("Seleciona", opts)
                if st.button("Apagar selecionados"):
                    idxs = sorted([int(s.split("]")[0][1:]) for s in pick], reverse=True)
                    for i in idxs:
                        cap["axis_ticks"][axis].pop(i)
                    st.success("Apagado.")

        elif what == "Lines":
            key = st.selectbox("Line key", cfg["lines"])
            items = cap["lines"][key]
            if not items:
                st.info("Sem segmentos.")
            else:
                opts = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(items)]
                pick = st.multiselect("Seleciona", opts)
                if st.button("Apagar selecionados"):
                    idxs = sorted([int(s.split("]")[0][1:]) for s in pick], reverse=True)
                    for i in idxs:
                        cap["lines"][key].pop(i)
                    st.success("Apagado.")

        elif what == "Guides":
            gkeys = list(cfg.get("guides", {}).keys())
            if not gkeys:
                st.info("Este modo não tem guides.")
            else:
                panel = st.selectbox("Guide panel", gkeys)
                items = cap["guides"][panel]
                if not items:
                    st.info("Sem guias.")
                else:
                    opts = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(items)]
                    pick = st.multiselect("Seleciona", opts)
                    if st.button("Apagar selecionados"):
                        idxs = sorted([int(s.split("]")[0][1:]) for s in pick], reverse=True)
                        for i in idxs:
                            cap["guides"][panel].pop(i)
                        st.success("Apagado.")

        elif what == "Panel corners":
            panel = st.selectbox("Panel", cfg["panels"])
            if st.button(f"Limpar corners de {panel}"):
                cap["panel_corners"][panel] = []
                st.success("Limpo.")


# =========================
# Preview tab
# =========================
with tabs[2]:
    st.subheader("Preview overlay")
    st.image(img_show, use_container_width=True)
    st.caption("Verifica se ticks/corners/linhas/guias estão onde queres antes de exportar.")

