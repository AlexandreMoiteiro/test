from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import streamlit as st
import pymupdf
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# Repo file names (os teus)
# =========================
FILES = {
    "landing": {
        "bg_default": "ldg_ground_roll.pdf",
        "json_default": "ldg_ground_roll.json",
        "kind": "pdf",
        "page_default": 0,
    },
    "takeoff": {
        # se um dia tiveres o PDF, troca para .pdf e kind="pdf"
        "bg_default": "to_ground_roll.jpg",
        "json_default": "to_ground_roll.json",
        "kind": "image",
        "page_default": 0,
    },
    "climb": {
        "bg_default": "climb_perf.jpg",
        "json_default": "climb_perf.json",
        "kind": "image",
        "page_default": 0,
    },
}

# =========================
# MODE CONFIG (sem invenções)
# =========================
MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "landing": {
        "title": "Landing Ground Roll",
        "panels": ["left", "middle", "right"],
        "axis_ticks": {
            "oat_c": "OAT (°C) [painel left]",
            "weight_x100_lb": "Weight (lb/100) [painel middle]",
            "wind_kt": "Wind (kt) [painel right]",
            "ground_roll_ft": "Ground Roll (ft) [eixo vertical right]",
        },
        "lines": [
            "isa_m15", "isa", "isa_p35",
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_7000",
            "weight_ref_line", "wind_ref_zero",
        ],
        "guides": {
            "guides_weight": "Guias WEIGHT (painel middle)",
            "guides_wind": "Guias WIND (painel right)",
        },
    },
    "takeoff": {
        "title": "Flaps Up Takeoff Ground Roll",
        "panels": ["left", "middle", "right"],  # <- como pediste
        "axis_ticks": {
            "oat_c": "OAT (°C) [painel left]",
            "weight_x100_lb": "Weight (lb/100) [painel middle]",
            "wind_kt": "Wind (kt) [painel right]",
            "takeoff_gr_ft": "Takeoff GR (ft) [eixo vertical right]",
        },
        "lines": [
            "isa_m15", "isa", "isa_p35",
            "pa_sea_level", "pa_2000", "pa_4000", "pa_6000", "pa_8000",
            "weight_ref_line", "wind_ref_zero",
        ],
        "guides": {
            "guides_weight": "Guias WEIGHT (painel middle)",
            "guides_wind": "Guias WIND (painel right)",
        },
    },
    "climb": {
        "title": "Climb Performance",
        "panels": ["main"],  # <- 1 painel
        "axis_ticks": {
            "oat_c": "OAT (°C) [eixo X]",
            "roc_fpm": "Rate of Climb (FPM) [eixo Y]",
        },
        "lines": [
            "isa_m15", "isa", "isa_p35",
            "pa_sea_level", "pa_1000", "pa_2000", "pa_3000", "pa_4000", "pa_5000",
            "pa_6000", "pa_7000", "pa_8000", "pa_9000", "pa_10000", "pa_11000",
            "pa_12000", "pa_13000",
        ],
        "guides": {},  # <- como pediste: sem guide main nenhum
    },
}


# =========================
# Helpers IO/render
# =========================
def locate_file(name: str) -> Optional[str]:
    p = Path(name)
    if p.exists():
        return str(p)
    if "__file__" in globals():
        p2 = Path(__file__).resolve().parent / name
        if p2.exists():
            return str(p2)
    return None


@st.cache_data(show_spinner=False)
def render_pdf_page_to_pil(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def load_background(mode: str, upload_bg, page_index: int, zoom: float) -> Image.Image:
    info = FILES[mode]
    if info["kind"] == "pdf":
        if upload_bg is not None:
            pdf_bytes = upload_bg.read()
        else:
            p = locate_file(info["bg_default"])
            if not p:
                raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
            pdf_bytes = Path(p).read_bytes()
        return render_pdf_page_to_pil(pdf_bytes, page_index=page_index, zoom=zoom)

    # image
    if upload_bg is not None:
        img = Image.open(upload_bg).convert("RGB")
    else:
        p = locate_file(info["bg_default"])
        if not p:
            raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
        img = Image.open(p).convert("RGB")
    return img


def json_default_name(mode: str) -> str:
    return FILES[mode]["json_default"]


# =========================
# Capture schema
# =========================
def new_capture(mode: str, zoom: float, page_index: int) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode]
    cap = {
        "mode": mode,
        "zoom": float(zoom),
        "page_index": int(page_index),
        "panel_corners": {p: [] for p in cfg["panels"]},  # 4 pontos dicts {"x","y"}
        "axis_ticks": {k: [] for k in cfg["axis_ticks"].keys()},  # dict {"x","y","value"}
        "lines": {k: [] for k in cfg["lines"]},  # dict {"x1","y1","x2","y2"}
        "guides": {k: [] for k in cfg.get("guides", {}).keys()},  # idem
        "notes": "Tudo em pixels do render atual (zoom e page_index).",
    }
    return cap


def ensure_capture(cap: Dict[str, Any], mode: str, zoom: float, page_index: int) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode]
    cap.setdefault("mode", mode)
    cap["zoom"] = float(zoom)
    cap["page_index"] = int(page_index)

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
    for gk in cfg.get("guides", {}).keys():
        cap["guides"].setdefault(gk, [])

    cap.setdefault("notes", "")
    return cap


# =========================
# Drawing overlay (com ponto do clique!)
# =========================
def draw_overlay(
    base: Image.Image,
    cap: Dict[str, Any],
    show_panels: bool = True,
    show_ticks: bool = True,
    show_lines: bool = True,
    show_guides: bool = True,
    last_click: Optional[Tuple[float, float]] = None,
    pending_point: Optional[Tuple[float, float]] = None,
    pending_corners: Optional[List[Dict[str, float]]] = None,
) -> Image.Image:
    out = base.copy()
    d = ImageDraw.Draw(out)

    # Panels (cyan)
    if show_panels:
        for panel, pts in cap.get("panel_corners", {}).items():
            if len(pts) == 4:
                poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
                d.line(poly + [poly[0]], fill=(0, 140, 255), width=3)
                d.text((poly[0][0] + 6, poly[0][1] + 6), panel, fill=(0, 140, 255))

    # Ticks (green)
    if show_ticks:
        for axis, ticks in cap.get("axis_ticks", {}).items():
            for t in ticks:
                x, y = float(t["x"]), float(t["y"])
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 0), width=3)

    # Lines (red)
    if show_lines:
        for _, segs in cap.get("lines", {}).items():
            for s in segs:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

    # Guides (blue)
    if show_guides:
        for _, segs in cap.get("guides", {}).items():
            for s in segs:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

    # Pending corners (magenta)
    if pending_corners:
        for p in pending_corners:
            x, y = p["x"], p["y"]
            d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 0, 255), width=3)

    # Pending first point for a segment (orange)
    if pending_point is not None:
        x, y = pending_point
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 140, 0), width=4)

    # Last click point (black)
    if last_click is not None:
        x, y = last_click
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 0, 0), width=4)

    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PA28 JSON Builder (3 gráficos)", layout="wide")
st.title("PA28 — Builder de JSON (Landing / Takeoff / Climb)")

mode = st.sidebar.selectbox(
    "Escolhe o gráfico",
    options=["landing", "takeoff", "climb"],
    format_func=lambda k: MODE_CONFIG[k]["title"],
)

cfg = MODE_CONFIG[mode]
files = FILES[mode]

st.sidebar.markdown("---")
st.sidebar.header("Background")

upload_bg = st.sidebar.file_uploader(
    "Upload background (opcional)",
    type=["pdf", "png", "jpg", "jpeg"],
)

zoom = st.sidebar.number_input("Zoom (PDF)", value=2.3, step=0.1)
page_index = st.sidebar.number_input("Página (0-index) [PDF]", value=int(files["page_default"]), step=1)

# Load background
try:
    bg = load_background(mode, upload_bg, page_index=int(page_index), zoom=float(zoom))
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("JSON")
upload_json = st.sidebar.file_uploader("Upload JSON (opcional)", type=["json"])

default_json_name = json_default_name(mode)

def load_json_or_new() -> Dict[str, Any]:
    if upload_json is not None:
        return json.loads(upload_json.read().decode("utf-8"))
    p = locate_file(default_json_name)
    if p:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    return new_capture(mode, zoom=float(zoom), page_index=int(page_index))

if "cap" not in st.session_state or st.session_state.get("cap_mode") != mode:
    st.session_state.cap = load_json_or_new()
    st.session_state.cap_mode = mode

cap = st.session_state.cap
cap = ensure_capture(cap, mode=mode, zoom=float(zoom), page_index=int(page_index))

# State for capture workflow
st.session_state.setdefault("pending_point", None)     # (x,y) for segment first click
st.session_state.setdefault("pending_corners", [])     # list of {"x","y"} up to 4
st.session_state.setdefault("last_click", None)        # (x,y)
st.session_state.setdefault("status_msg", "")

# Controls: what to show
with st.sidebar:
    st.markdown("---")
    st.header("Overlay")
    show_panels = st.checkbox("Mostrar painéis", value=True)
    show_ticks = st.checkbox("Mostrar ticks", value=True)
    show_lines = st.checkbox("Mostrar linhas", value=True)
    show_guides = st.checkbox("Mostrar guides", value=True)

    st.markdown("---")
    if st.button("Reset pending (limpar clique pendente)"):
        st.session_state.pending_point = None
        st.session_state.pending_corners = []
        st.session_state.status_msg = "Pendentes limpos."

    if st.button("Novo JSON (reset tudo)"):
        st.session_state.cap = new_capture(mode, zoom=float(zoom), page_index=int(page_index))
        st.session_state.pending_point = None
        st.session_state.pending_corners = []
        st.session_state.last_click = None
        st.session_state.status_msg = "JSON reset."
        st.rerun()

# Layout
left, right = st.columns([1.6, 1])

with right:
    st.subheader("Modo de captura (como antes)")
    tasks = []
    tasks.append("PANEL: 4 cliques (corners)")
    tasks += [f"TICK: {k}" for k in cfg["axis_ticks"].keys()]
    tasks += [f"LINE: {k} (2 cliques)" for k in cfg["lines"]]
    for gk in cfg.get("guides", {}).keys():
        tasks.append(f"GUIDE: {gk} (2 cliques)")

    task = st.selectbox("Escolhe o que estás a fazer agora", tasks)

    # panel choice for corners
    panel_pick = None
    if task.startswith("PANEL"):
        panel_pick = st.selectbox("Qual painel?", cfg["panels"])

    # tick axis + value
    tick_axis = None
    tick_value = None
    if task.startswith("TICK:"):
        tick_axis = task.split("TICK: ")[1].strip()
        tick_value = st.number_input("Valor do tick", value=0.0, step=1.0)

    # line key
    line_key = None
    if task.startswith("LINE:"):
        line_key = task.split("LINE: ")[1].split(" (")[0].strip()

    # guide key
    guide_key = None
    if task.startswith("GUIDE:"):
        guide_key = task.split("GUIDE: ")[1].split(" (")[0].strip()
        st.info(cfg["guides"][guide_key])

    st.markdown("---")
    st.subheader("Último clique")
    if st.session_state.last_click is None:
        st.write("—")
    else:
        x, y = st.session_state.last_click
        st.write(f"x={int(x)}, y={int(y)}")

    st.subheader("Pendentes")
    st.write({
        "pending_point": st.session_state.pending_point,
        "pending_corners_count": len(st.session_state.pending_corners),
    })

    if st.session_state.status_msg:
        st.success(st.session_state.status_msg)

    st.markdown("---")
    st.subheader("Resumo do JSON")
    st.write({
        "panels": {p: len(cap["panel_corners"].get(p, [])) for p in cfg["panels"]},
        "ticks": {k: len(cap["axis_ticks"].get(k, [])) for k in cfg["axis_ticks"].keys()},
        "lines": {k: len(cap["lines"].get(k, [])) for k in cfg["lines"]},
        "guides": {k: len(cap["guides"].get(k, [])) for k in cfg.get("guides", {}).keys()},
    })

    st.markdown("---")
    st.subheader("Export / Guardar")
    txt = json.dumps(cap, indent=2)
    st.download_button("⬇️ Download JSON", data=txt, file_name=default_json_name, mime="application/json")
    if st.button(f"Guardar no repo como {default_json_name}"):
        Path(default_json_name).write_text(txt, encoding="utf-8")
        st.success(f"Guardado: {default_json_name}")

    st.markdown("---")
    st.subheader("Apagar outliers (como antes)")
    del_kind = st.selectbox("Apagar:", ["—", "Ticks", "Lines", "Guides", "Panel corners"])
    if del_kind == "Ticks":
        ax = st.selectbox("Axis", list(cfg["axis_ticks"].keys()))
        items = cap["axis_ticks"][ax]
        if items:
            opts = [f"[{i}] v={t['value']} (x={int(t['x'])}, y={int(t['y'])})" for i, t in enumerate(items)]
            pick = st.multiselect("Seleciona índices", opts)
            if st.button("Apagar ticks selecionados"):
                idxs = sorted([int(s.split("]")[0][1:]) for s in pick], reverse=True)
                for i in idxs:
                    cap["axis_ticks"][ax].pop(i)
                st.session_state.status_msg = "Ticks apagados."
                st.rerun()
        else:
            st.info("Sem ticks.")

    elif del_kind == "Lines":
        lk = st.selectbox("Line key", cfg["lines"])
        items = cap["lines"][lk]
        if items:
            opts = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(items)]
            pick = st.multiselect("Seleciona índices", opts)
            if st.button("Apagar segmentos selecionados"):
                idxs = sorted([int(s.split("]")[0][1:]) for s in pick], reverse=True)
                for i in idxs:
                    cap["lines"][lk].pop(i)
                st.session_state.status_msg = "Segmentos apagados."
                st.rerun()
        else:
            st.info("Sem segmentos.")

    elif del_kind == "Guides":
        if not cfg.get("guides", {}):
            st.info("Este gráfico não tem guides.")
        else:
            gk = st.selectbox("Guide key", list(cfg["guides"].keys()))
            items = cap["guides"][gk]
            if items:
                opts = [f"[{i}] ({int(s['x1'])},{int(s['y1'])})→({int(s['x2'])},{int(s['y2'])})" for i, s in enumerate(items)]
                pick = st.multiselect("Seleciona índices", opts)
                if st.button("Apagar guides selecionadas"):
                    idxs = sorted([int(s.split("]")[0][1:]) for s in pick], reverse=True)
                    for i in idxs:
                        cap["guides"][gk].pop(i)
                    st.session_state.status_msg = "Guides apagadas."
                    st.rerun()
            else:
                st.info("Sem guides.")

    elif del_kind == "Panel corners":
        p = st.selectbox("Painel", cfg["panels"])
        if st.button(f"Limpar corners do painel {p}"):
            cap["panel_corners"][p] = []
            st.session_state.status_msg = "Corners apagados."
            st.rerun()


with left:
    st.subheader("Imagem (mostra clique, painéis, etc.)")

    # Build overlay with last click & pending markers
    overlay = draw_overlay(
        bg,
        cap,
        show_panels=show_panels,
        show_ticks=show_ticks,
        show_lines=show_lines,
        show_guides=show_guides,
        last_click=st.session_state.last_click,
        pending_point=st.session_state.pending_point,
        pending_corners=st.session_state.pending_corners,
    )

    click = streamlit_image_coordinates(np.array(overlay), key=f"img_{mode}_{cap.get('page_index',0)}")

    if click is not None:
        x = float(click["x"])
        y = float(click["y"])
        st.session_state.last_click = (x, y)
        st.session_state.status_msg = ""  # limpar msg antiga

        # APPLY the click according to task
        if task.startswith("PANEL"):
            st.session_state.pending_corners.append({"x": x, "y": y})
            if len(st.session_state.pending_corners) == 4:
                cap["panel_corners"][panel_pick] = st.session_state.pending_corners
                st.session_state.pending_corners = []
                st.session_state.status_msg = f"Corners guardados em '{panel_pick}'."

        elif task.startswith("TICK:"):
            cap["axis_ticks"][tick_axis].append({"x": x, "y": y, "value": float(tick_value)})
            st.session_state.status_msg = f"Tick adicionado em {tick_axis} (v={tick_value})."

        elif task.startswith("LINE:"):
            if st.session_state.pending_point is None:
                st.session_state.pending_point = (x, y)
                st.session_state.status_msg = "Primeiro ponto da linha guardado. Agora clica no segundo."
            else:
                x1, y1 = st.session_state.pending_point
                st.session_state.pending_point = None
                cap["lines"][line_key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                st.session_state.status_msg = f"Segmento adicionado em lines['{line_key}']."

        elif task.startswith("GUIDE:"):
            if st.session_state.pending_point is None:
                st.session_state.pending_point = (x, y)
                st.session_state.status_msg = "Primeiro ponto da guide guardado. Agora clica no segundo."
            else:
                x1, y1 = st.session_state.pending_point
                st.session_state.pending_point = None
                cap["guides"][guide_key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                st.session_state.status_msg = f"Guide adicionada em guides['{guide_key}']."

        # re-render to show immediate feedback
        st.rerun()

    st.image(overlay, use_container_width=True)
    st.caption("O ponto preto é o último clique. Laranja = 1º ponto pendente (segmentos). Magenta = corners pendentes.")

