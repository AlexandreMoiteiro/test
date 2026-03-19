from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw
import pymupdf  # PyMuPDF


# =========================
# Assets
# =========================
ASSETS = {
    "landing_perf": {
        "title": "Landing Performance",
        "bg_default": "ldg_perf.pdf",
        "bg_kind": "pdf",
        "page_default": 0,
        "json_default_name": "ldg_perf.json",
        "default_axis_keys": [
            "oat_c",
            "weight_x100_lb",
            "wind_kt",
            "landing_50ft_ft",
        ],
        "default_line_keys": [
            "pa_sea_level",
            "pa_2000",
            "pa_4000",
            "pa_6000",
            "pa_7000",
            "weight_ref_line",
            "wind_ref_zero",
        ],
        "default_guide_keys": [
            "guides_weight",
            "guides_wind",
        ],
        "default_panels": ["left", "middle", "right"],
    },
    "takeoff_perf": {
        "title": "Flaps Up Takeoff Performance",
        "bg_default": "to_perf.pdf",
        "bg_kind": "pdf",
        "page_default": 0,
        "json_default_name": "to_perf.json",
        "default_axis_keys": [
            "oat_c",
            "weight_x100_lb",
            "wind_kt",
            "takeoff_50ft_ft",
        ],
        "default_line_keys": [
            "pa_sea_level",
            "pa_2000",
            "pa_4000",
            "pa_6000",
            "pa_8000",
            "weight_ref_line",
            "wind_ref_zero",
        ],
        "default_guide_keys": [
            "guides_weight",
            "guides_wind",
        ],
        "default_panels": ["left", "middle", "right"],
    },
}


# =========================
# Helpers
# =========================
def _here(name: str) -> Optional[Path]:
    p = Path(name)
    if p.exists():
        return p
    if "__file__" in globals():
        p2 = Path(__file__).resolve().parent / name
        if p2.exists():
            return p2
    return None


@st.cache_data(show_spinner=False)
def render_pdf_to_image(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def load_background(asset_key: str, upload_bg, page_index: int, zoom: float) -> Image.Image:
    info = ASSETS[asset_key]
    if info["bg_kind"] == "pdf":
        if upload_bg is not None:
            pdf_bytes = upload_bg.read()
        else:
            p = _here(info["bg_default"])
            if not p:
                raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
            pdf_bytes = p.read_bytes()
        return render_pdf_to_image(pdf_bytes, page_index=page_index, zoom=zoom)

    if upload_bg is not None:
        return Image.open(upload_bg).convert("RGB")

    p = _here(info["bg_default"])
    if not p:
        raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
    return Image.open(p).convert("RGB")


def empty_capture(asset_key: str) -> Dict[str, Any]:
    info = ASSETS[asset_key]
    return {
        "meta": {
            "asset": asset_key,
            "title": info["title"],
            "source_file": info["bg_default"],
            "notes": "",
        },
        "panel_corners": {k: [] for k in info["default_panels"]},
        "axis_ticks": {k: [] for k in info["default_axis_keys"]},
        "lines": {k: [] for k in info["default_line_keys"]},
        "guides": {k: [] for k in info["default_guide_keys"]},
    }


def ensure_structure(cap: Dict[str, Any], asset_key: str) -> Dict[str, Any]:
    info = ASSETS[asset_key]
    cap.setdefault("meta", {})
    cap.setdefault("panel_corners", {})
    cap.setdefault("axis_ticks", {})
    cap.setdefault("lines", {})
    cap.setdefault("guides", {})

    for k in info["default_panels"]:
        cap["panel_corners"].setdefault(k, [])
    for k in info["default_axis_keys"]:
        cap["axis_ticks"].setdefault(k, [])
    for k in info["default_line_keys"]:
        cap["lines"].setdefault(k, [])
    for k in info["default_guide_keys"]:
        cap["guides"].setdefault(k, [])

    return cap


def normalize_panel(panel_pts: Any) -> List[Dict[str, float]]:
    if not isinstance(panel_pts, list):
        return []
    out = []
    for p in panel_pts:
        if isinstance(p, dict) and "x" in p and "y" in p:
            out.append({"x": float(p["x"]), "y": float(p["y"])})
    return out


def draw_overlay(base: Image.Image, cap: Dict[str, Any]) -> Image.Image:
    img = base.copy()
    d = ImageDraw.Draw(img)

    # lines
    for _, seglist in cap.get("lines", {}).items():
        for s in seglist:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

    # guides
    for _, seglist in cap.get("guides", {}).items():
        for s in seglist:
            d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

    # ticks
    for key, tlist in cap.get("axis_ticks", {}).items():
        for i, t in enumerate(tlist):
            x, y = float(t["x"]), float(t["y"])
            d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 160, 0), width=3)
            d.text((x + 6, y - 14), f"{key}:{i}", fill=(0, 110, 0))

    # panels
    for panel, pts in cap.get("panel_corners", {}).items():
        pts = normalize_panel(pts)
        if len(pts) == 4:
            poly = [(p["x"], p["y"]) for p in pts]
            d.line(poly + [poly[0]], fill=(0, 200, 255), width=3)
            d.text((poly[0][0] + 6, poly[0][1] + 6), panel, fill=(0, 140, 255))

    return img


def add_tick(cap: Dict[str, Any], axis_key: str, x: float, y: float, value: float):
    cap["axis_ticks"].setdefault(axis_key, [])
    cap["axis_ticks"][axis_key].append({
        "x": float(x),
        "y": float(y),
        "value": float(value),
    })


def add_segment(cap: Dict[str, Any], section: str, key: str, x1: float, y1: float, x2: float, y2: float):
    cap[section].setdefault(key, [])
    cap[section][key].append({
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    })


def set_panel(cap: Dict[str, Any], panel_key: str, pts: List[Tuple[float, float]]):
    cap["panel_corners"][panel_key] = [{"x": float(x), "y": float(y)} for x, y in pts[:4]]


def json_download_name(asset_key: str) -> str:
    return ASSETS[asset_key]["json_default_name"]


# =========================
# Session init
# =========================
st.set_page_config(page_title="PA28 JSON Builder", layout="wide")
st.title("PA28 — JSON Builder para os gráficos")

asset_key = st.sidebar.selectbox(
    "Escolhe o gráfico",
    options=list(ASSETS.keys()),
    format_func=lambda k: ASSETS[k]["title"],
)

info = ASSETS[asset_key]

if "cap_asset" not in st.session_state or st.session_state.cap_asset != asset_key:
    st.session_state.cap_asset = asset_key
    st.session_state.cap = empty_capture(asset_key)

cap = ensure_structure(st.session_state.cap, asset_key)

st.sidebar.markdown("---")
st.sidebar.subheader("Background")
upload_bg = st.sidebar.file_uploader("Upload background", type=["pdf", "png", "jpg", "jpeg"])
zoom = st.sidebar.number_input("Zoom PDF", value=2.4, step=0.1)
page_index = st.sidebar.number_input("Página PDF (0-index)", value=int(info["page_default"]), step=1)

bg = load_background(asset_key, upload_bg, int(page_index), float(zoom))

st.sidebar.markdown("---")
st.sidebar.subheader("JSON")
upload_json = st.sidebar.file_uploader("Importar JSON existente", type=["json"])
if upload_json is not None:
    st.session_state.cap = ensure_structure(json.loads(upload_json.read().decode("utf-8")), asset_key)
    cap = st.session_state.cap

col_img, col_edit = st.columns([1.5, 1])

with col_edit:
    st.subheader("Editor")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Meta",
        "Panels",
        "Axis ticks",
        "Lines / Guides",
        "JSON",
    ])

    with tab1:
        cap["meta"]["asset"] = asset_key
        cap["meta"]["title"] = info["title"]
        cap["meta"]["source_file"] = st.text_input(
            "Source file",
            value=cap["meta"].get("source_file", info["bg_default"]),
        )
        cap["meta"]["notes"] = st.text_area(
            "Notas",
            value=cap["meta"].get("notes", ""),
            height=120,
        )

    with tab2:
        st.markdown("### Panel corners")
        panel_key = st.selectbox("Panel", info["default_panels"])

        pcols = st.columns(2)
        pts: List[Tuple[float, float]] = []
        for i in range(4):
            with pcols[i % 2]:
                x = st.number_input(f"{panel_key} P{i+1} x", value=0.0, key=f"{panel_key}_x_{i}")
                y = st.number_input(f"{panel_key} P{i+1} y", value=0.0, key=f"{panel_key}_y_{i}")
                pts.append((x, y))

        if st.button("Guardar panel", key=f"save_panel_{panel_key}"):
            set_panel(cap, panel_key, pts)
            st.success(f"Panel '{panel_key}' guardado.")

        if st.button("Limpar panel", key=f"clear_panel_{panel_key}"):
            cap["panel_corners"][panel_key] = []
            st.warning(f"Panel '{panel_key}' limpo.")

        st.json(cap["panel_corners"].get(panel_key, []))

    with tab3:
        st.markdown("### Axis ticks")
        axis_key = st.selectbox("Axis key", info["default_axis_keys"])
        x = st.number_input("Tick x", value=0.0, key=f"tick_x_{axis_key}")
        y = st.number_input("Tick y", value=0.0, key=f"tick_y_{axis_key}")
        value = st.number_input("Tick value", value=0.0, key=f"tick_value_{axis_key}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Adicionar tick", key=f"add_tick_{axis_key}"):
                add_tick(cap, axis_key, x, y, value)
                st.success("Tick adicionado.")
        with c2:
            if st.button("Limpar ticks deste eixo", key=f"clear_ticks_{axis_key}"):
                cap["axis_ticks"][axis_key] = []
                st.warning("Ticks limpos.")

        st.dataframe(cap["axis_ticks"].get(axis_key, []), use_container_width=True)

    with tab4:
        st.markdown("### Segments")
        section = st.radio("Secção", ["lines", "guides"], horizontal=True)

        if section == "lines":
            key = st.selectbox("Line key", info["default_line_keys"])
        else:
            key = st.selectbox("Guide key", info["default_guide_keys"])

        x1 = st.number_input("x1", value=0.0, key=f"{section}_{key}_x1")
        y1 = st.number_input("y1", value=0.0, key=f"{section}_{key}_y1")
        x2 = st.number_input("x2", value=0.0, key=f"{section}_{key}_x2")
        y2 = st.number_input("y2", value=0.0, key=f"{section}_{key}_y2")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Adicionar segmento", key=f"add_seg_{section}_{key}"):
                add_segment(cap, section, key, x1, y1, x2, y2)
                st.success("Segmento adicionado.")
        with c2:
            if st.button("Limpar segmentos desta chave", key=f"clear_seg_{section}_{key}"):
                cap[section][key] = []
                st.warning("Segmentos limpos.")

        st.dataframe(cap[section].get(key, []), use_container_width=True)

    with tab5:
        st.markdown("### JSON atual")
        json_text = json.dumps(cap, indent=2, ensure_ascii=False)
        st.code(json_text, language="json")

        st.download_button(
            "Descarregar JSON",
            data=json_text.encode("utf-8"),
            file_name=json_download_name(asset_key),
            mime="application/json",
        )

        if st.button("Reset JSON completo"):
            st.session_state.cap = empty_capture(asset_key)
            st.warning("JSON reiniciado.")
            st.rerun()

with col_img:
    st.subheader("Preview")
    img = draw_overlay(bg, cap)
    st.image(img, use_container_width=True)
    st.caption("Vermelho: lines • Azul: guides • Verde: axis ticks • Ciano: panel_corners")



