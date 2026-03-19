from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw
import pymupdf  # PyMuPDF


# ======================================
# Targets to build JSON for new charts
# ======================================
TARGETS = {
    "landing_50ft": {
        "title": "Landing Performance (50 ft)",
        "bg_default": "ldg_perf.pdf",
        "bg_kind": "pdf",
        "page_default": 0,
        "json_default_name": "ldg_perf.json",
        "suggested_axis_keys": [
            "oat_c",
            "weight_x100_lb",
            "headwind_kt",
            "landing_50ft_ft",
        ],
        "suggested_line_keys": [
            "pa_sea_level",
            "pa_2000",
            "pa_4000",
            "pa_6000",
            "pa_7000",
            "weight_ref_line",
            "wind_ref_zero",
            "headwind_line",
            "tailwind_line",
        ],
        "suggested_guide_groups": ["middle", "right"],
        "suggested_panels": ["left", "middle", "right"],
    },
    "takeoff_50ft": {
        "title": "Flaps Up Takeoff Performance (50 ft)",
        "bg_default": "to_perf.pdf",
        "bg_kind": "pdf",
        "page_default": 0,
        "json_default_name": "to_perf.json",
        "suggested_axis_keys": [
            "oat_c",
            "weight_x100_lb",
            "headwind_kt",
            "takeoff_50ft_ft",
        ],
        "suggested_line_keys": [
            "pa_sea_level",
            "pa_2000",
            "pa_4000",
            "pa_6000",
            "pa_8000",
            "weight_ref_line",
            "wind_ref_zero",
            "headwind_line",
            "tailwind_line",
        ],
        "suggested_guide_groups": ["middle", "right"],
        "suggested_panels": ["left", "middle", "right"],
    },
}


# ======================================
# Helpers
# ======================================
def _here(name: str) -> Optional[Path]:
    p = Path(name)
    if p.exists():
        return p
    if "__file__" in globals():
        p2 = Path(__file__).resolve().parent / name
        if p2.exists():
            return p2
    p3 = Path("/mnt/data") / name
    if p3.exists():
        return p3
    return None


@st.cache_data(show_spinner=False)
def render_pdf_to_image(pdf_bytes: bytes, page_index: int, zoom: float) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def load_background(target: str, upload_bg, page_index: int, zoom: float) -> Image.Image:
    info = TARGETS[target]
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


def ensure_capture_shape(cap: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cap or {})
    out.setdefault("panel_corners", {})
    out.setdefault("axis_ticks", {})
    out.setdefault("lines", {})
    out.setdefault("guides", {})
    return out


def parse_json_text(text: str) -> Dict[str, Any]:
    if not text.strip():
        return ensure_capture_shape({})
    return ensure_capture_shape(json.loads(text))


def pt_xy(p: Any) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    if isinstance(p, (list, tuple)) and len(p) == 2:
        return float(p[0]), float(p[1])
    raise ValueError(f"Invalid point: {p}")


def normalize_panel(panel_pts: Any) -> List[Dict[str, float]]:
    if not isinstance(panel_pts, list) or len(panel_pts) != 4:
        return []
    out = []
    for p in panel_pts:
        x, y = pt_xy(p)
        out.append({"x": x, "y": y})
    return out


def normalize_panels(cap: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    out = {}
    pc = cap.get("panel_corners", {})
    if not isinstance(pc, dict):
        return out
    for k, pts in pc.items():
        out[k] = normalize_panel(pts)
    return out


def as_pretty_json(cap: Dict[str, Any]) -> str:
    return json.dumps(cap, ensure_ascii=False, indent=2)


def blank_template(target: str) -> Dict[str, Any]:
    info = TARGETS[target]
    return {
        "meta": {
            "title": info["title"],
            "source": info["bg_default"],
            "notes": "Preencher coordenadas manualmente e validar pelo overlay.",
        },
        "panel_corners": {k: [] for k in info["suggested_panels"]},
        "axis_ticks": {k: [] for k in info["suggested_axis_keys"]},
        "lines": {k: [] for k in info["suggested_line_keys"]},
        "guides": {k: [] for k in info["suggested_guide_groups"]},
    }


def coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def draw_capture_overlay(base: Image.Image, cap: Dict[str, Any], show_labels: bool = True) -> Image.Image:
    img = base.copy()
    d = ImageDraw.Draw(img)
    panels = normalize_panels(cap)

    # Lines
    for name, seglist in (cap.get("lines", {}) or {}).items():
        for s in seglist:
            x1, y1 = coerce_float(s.get("x1")), coerce_float(s.get("y1"))
            x2, y2 = coerce_float(s.get("x2")), coerce_float(s.get("y2"))
            d.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=3)
            if show_labels:
                d.text((x1 + 4, y1 + 4), name, fill=(255, 0, 0))

    # Guides
    for group, seglist in (cap.get("guides", {}) or {}).items():
        for i, s in enumerate(seglist):
            x1, y1 = coerce_float(s.get("x1")), coerce_float(s.get("y1"))
            x2, y2 = coerce_float(s.get("x2")), coerce_float(s.get("y2"))
            d.line([(x1, y1), (x2, y2)], fill=(0, 0, 255), width=4)
            if show_labels:
                d.text((x1 + 4, y1 - 14), f"{group}[{i}]", fill=(0, 0, 255))

    # Axis ticks
    for axis_name, tlist in (cap.get("axis_ticks", {}) or {}).items():
        for t in tlist:
            x, y = coerce_float(t.get("x")), coerce_float(t.get("y"))
            d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 0), width=3)
            if show_labels:
                d.text((x + 6, y - 10), f"{axis_name}:{t.get('value')}", fill=(0, 160, 0))

    # Panels
    for panel, pts in panels.items():
        if len(pts) == 4:
            poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
            d.line(poly + [poly[0]], fill=(0, 180, 255), width=3)
            if show_labels:
                d.text((poly[0][0] + 6, poly[0][1] + 6), panel, fill=(0, 180, 255))

    return img


# ======================================
# Streamlit app
# ======================================
st.set_page_config(page_title="PA28 JSON Builder", layout="wide")
st.title("PA28 — JSON Builder para os novos gráficos")

st.markdown(
    "Usa esta app para montar e validar os JSONs dos gráficos de **takeoff 50 ft** e **landing 50 ft**. "
    "Os PDFs que enviaste são o *Landing Performance* (fig. 5-41) e o *Flaps Up Takeoff Performance* (fig. 5-7)."
)

target = st.sidebar.selectbox(
    "Gráfico",
    options=["landing_50ft", "takeoff_50ft"],
    format_func=lambda k: TARGETS[k]["title"],
)
info = TARGETS[target]

st.sidebar.markdown("---")
st.sidebar.subheader("Background")
upload_bg = st.sidebar.file_uploader("Upload background opcional", type=["pdf", "png", "jpg", "jpeg"])
zoom = st.sidebar.number_input("Zoom PDF", value=2.4, step=0.1)
page_index = st.sidebar.number_input("Página (0-index)", value=int(info["page_default"]), step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("JSON source")
use_template = st.sidebar.checkbox("Começar com template sugerido", value=True)
upload_json = st.sidebar.file_uploader("Upload de JSON já existente", type=["json"])

if upload_json is not None:
    initial_text = upload_json.read().decode("utf-8")
elif use_template:
    initial_text = as_pretty_json(blank_template(target))
else:
    initial_text = as_pretty_json(ensure_capture_shape({}))

bg = load_background(target, upload_bg, page_index=int(page_index), zoom=float(zoom))

left, right = st.columns([1.7, 1.2])

with right:
    st.subheader("Editor JSON")
    st.caption(
        "Edita as coordenadas diretamente. O overlay à esquerda mostra painéis, ticks, linhas e guides por cima do gráfico."
    )

    st.markdown("**Chaves sugeridas**")
    st.write(
        {
            "panel_corners": info["suggested_panels"],
            "axis_ticks": info["suggested_axis_keys"],
            "lines": info["suggested_line_keys"],
            "guides": info["suggested_guide_groups"],
        }
    )

    json_text = st.text_area(
        "Conteúdo JSON",
        value=initial_text,
        height=780,
    )

    try:
        cap = parse_json_text(json_text)
        valid = True
        st.success("JSON válido.")
    except Exception as e:
        cap = ensure_capture_shape({})
        valid = False
        st.error(f"JSON inválido: {e}")

    if valid:
        out_name = st.text_input("Nome do ficheiro", value=info["json_default_name"])
        st.download_button(
            "Descarregar JSON",
            data=as_pretty_json(cap).encode("utf-8"),
            file_name=out_name,
            mime="application/json",
        )

        st.markdown("---")
        st.subheader("Checklist")
        st.write(
            {
                "panel_corners": list((cap.get("panel_corners", {}) or {}).keys()),
                "axis_ticks": list((cap.get("axis_ticks", {}) or {}).keys()),
                "lines": list((cap.get("lines", {}) or {}).keys()),
                "guides": list((cap.get("guides", {}) or {}).keys()),
            }
        )

with left:
    st.subheader(info["title"])
    if valid:
        img = draw_capture_overlay(bg, cap, show_labels=True)
    else:
        img = bg.copy()
    st.image(img, use_container_width=True)
    st.caption("Vermelho: lines. Azul: guides. Verde: axis ticks. Ciano: panel_corners.")

st.markdown("---")
st.markdown(
    "### Estrutura recomendada\n"
    "Para estes dois gráficos, a estrutura do JSON pode continuar praticamente igual ao solver atual:"
)

st.code(
    json.dumps(
        {
            "panel_corners": {
                "left": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
                "middle": [],
                "right": [],
            },
            "axis_ticks": {
                "oat_c": [{"x": 0, "y": 0, "value": -20}],
                "weight_x100_lb": [{"x": 0, "y": 0, "value": 25}],
                "headwind_kt": [{"x": 0, "y": 0, "value": 0}],
                "landing_50ft_ft": [{"x": 0, "y": 0, "value": 1000}],
            },
            "lines": {
                "pa_sea_level": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}],
                "pa_2000": [],
                "pa_4000": [],
                "weight_ref_line": [],
                "wind_ref_zero": [],
            },
            "guides": {
                "middle": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}],
                "right": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}],
            },
        },
        ensure_ascii=False,
        indent=2,
    ),
    language="json",
)

st.info(
    "Depois de fechares os dois JSONs, o passo seguinte é trocar no solver as chaves do output para `landing_50ft_ft` e `takeoff_50ft_ft` e adaptar os labels do UI."
)




