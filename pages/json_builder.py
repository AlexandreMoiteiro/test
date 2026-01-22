from __future__ import annotations

import json
import copy
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
        "panels": ["left", "middle", "right"],
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
        "panels": ["main"],
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
        "guides": {},
    },
}


# =========================
# IO / Render
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
        return Image.open(upload_bg).convert("RGB")

    p = locate_file(info["bg_default"])
    if not p:
        raise FileNotFoundError(f"Não encontrei {info['bg_default']}")
    return Image.open(p).convert("RGB")


def json_default_name(mode: str) -> str:
    return FILES[mode]["json_default"]


# =========================
# Capture schema
# =========================
def new_capture(mode: str, zoom: float, page_index: int) -> Dict[str, Any]:
    cfg = MODE_CONFIG[mode]
    return {
        "mode": mode,
        "zoom": float(zoom),
        "page_index": int(page_index),
        "panel_corners": {p: [] for p in cfg["panels"]},  # 4 pts {"x","y"}
        "axis_ticks": {k: [] for k in cfg["axis_ticks"].keys()},  # {"x","y","value"}
        "lines": {k: [] for k in cfg["lines"]},  # {"x1","y1","x2","y2"}
        "guides": {k: [] for k in cfg.get("guides", {}).keys()},  # {"x1","y1","x2","y2"}
        "notes": "Tudo em pixels do render atual (zoom e page_index).",
    }


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
# Overlay drawing
# =========================
def draw_overlay(
    base: Image.Image,
    cap: Dict[str, Any],
    show_panels: bool,
    show_ticks: bool,
    show_lines: bool,
    show_guides: bool,
    last_click: Optional[Tuple[float, float]],
    pending_segment: Optional[Dict[str, Any]],
    pending_corners: List[Dict[str, float]],
) -> Image.Image:
    out = base.copy()
    d = ImageDraw.Draw(out)

    # panels
    if show_panels:
        for panel, pts in cap.get("panel_corners", {}).items():
            if len(pts) == 4:
                poly = [(pts[i]["x"], pts[i]["y"]) for i in range(4)]
                d.line(poly + [poly[0]], fill=(0, 140, 255), width=3)
                d.text((poly[0][0] + 6, poly[0][1] + 6), panel, fill=(0, 140, 255))

    # ticks
    if show_ticks:
        for _, ticks in cap.get("axis_ticks", {}).items():
            for t in ticks:
                x, y = float(t["x"]), float(t["y"])
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 160, 0), width=3)

    # lines
    if show_lines:
        for _, segs in cap.get("lines", {}).items():
            for s in segs:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(255, 0, 0), width=3)

    # guides
    if show_guides:
        for _, segs in cap.get("guides", {}).items():
            for s in segs:
                d.line([(s["x1"], s["y1"]), (s["x2"], s["y2"])], fill=(0, 0, 255), width=4)

    # pending corners
    for p in pending_corners:
        x, y = p["x"], p["y"]
        d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 0, 255), width=3)

    # pending segment first point
    if pending_segment is not None:
        x, y = pending_segment["p"]
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 140, 0), width=4)
        d.text((x + 8, y + 8), f"{pending_segment['kind']}:{pending_segment['key']}", fill=(255, 140, 0))

    # last click
    if last_click is not None:
        x, y = last_click
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 0, 0), width=4)

    return out


# =========================
# History (UNDO)
# =========================
def push_history():
    st.session_state.history.append({
        "cap": copy.deepcopy(st.session_state.cap),
        "pending_segment": copy.deepcopy(st.session_state.pending_segment),
        "pending_corners": copy.deepcopy(st.session_state.pending_corners),
        "last_click": st.session_state.last_click,
        "status_msg": st.session_state.status_msg,
        "prev_task_id": st.session_state.prev_task_id,
        "last_processed_click_id": st.session_state.last_processed_click_id,
    })
    if len(st.session_state.history) > 80:
        st.session_state.history = st.session_state.history[-80:]


def undo_history() -> bool:
    if not st.session_state.history:
        return False
    snap = st.session_state.history.pop()
    st.session_state.cap = snap["cap"]
    st.session_state.pending_segment = snap["pending_segment"]
    st.session_state.pending_corners = snap["pending_corners"]
    st.session_state.last_click = snap["last_click"]
    st.session_state.status_msg = "UNDO ✓"
    st.session_state.prev_task_id = snap["prev_task_id"]
    st.session_state.last_processed_click_id = snap["last_processed_click_id"]
    return True


# =========================
# Click de-dup (CRÍTICO)
# =========================
def click_id(click: Dict[str, Any], mode: str, page_index: int) -> str:
    # quantização para evitar diferenças minúsculas
    x = int(round(float(click["x"])))
    y = int(round(float(click["y"])))
    return f"{mode}|{page_index}|{x}|{y}"


# =========================
# App
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
upload_bg = st.sidebar.file_uploader("Upload background (opcional)", type=["pdf", "png", "jpg", "jpeg"])
zoom = st.sidebar.number_input("Zoom (PDF)", value=2.3, step=0.1)
page_index = st.sidebar.number_input("Página (0-index) [PDF]", value=int(files["page_default"]), step=1)

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


# init per-mode
if "cap" not in st.session_state or st.session_state.get("cap_mode") != mode:
    st.session_state.cap = load_json_or_new()
    st.session_state.cap_mode = mode
    # reset click de-dup when switching mode
    st.session_state.last_processed_click_id = None
    st.session_state.prev_task_id = None
    st.session_state.pending_segment = None
    st.session_state.pending_corners = []
    st.session_state.last_click = None
    st.session_state.status_msg = ""

st.session_state.cap = ensure_capture(st.session_state.cap, mode=mode, zoom=float(zoom), page_index=int(page_index))

# session state
st.session_state.setdefault("pending_segment", None)  # {"kind":"LINE"/"GUIDE","key":str,"p":(x,y)}
st.session_state.setdefault("pending_corners", [])
st.session_state.setdefault("last_click", None)
st.session_state.setdefault("status_msg", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("prev_task_id", None)
st.session_state.setdefault("last_processed_click_id", None)

# Sidebar overlay toggles + actions
with st.sidebar:
    st.markdown("---")
    st.header("Overlay")
    show_panels = st.checkbox("Mostrar painéis", value=True)
    show_ticks = st.checkbox("Mostrar ticks", value=True)
    show_lines = st.checkbox("Mostrar linhas", value=True)
    show_guides = st.checkbox("Mostrar guides", value=True)

    st.markdown("---")
    if st.button("↩️ Undo (voltar atrás 1 passo)"):
        if undo_history():
            st.rerun()
        else:
            st.info("Nada para desfazer.")

    if st.button("Novo JSON (reset tudo)"):
        push_history()
        st.session_state.cap = new_capture(mode, zoom=float(zoom), page_index=int(page_index))
        st.session_state.pending_segment = None
        st.session_state.pending_corners = []
        st.session_state.last_click = None
        st.session_state.last_processed_click_id = None
        st.session_state.status_msg = "JSON reset."
        st.rerun()

    if st.button("Limpar pendentes"):
        push_history()
        st.session_state.pending_segment = None
        st.session_state.pending_corners = []
        st.session_state.status_msg = "Pendentes limpos."
        st.rerun()


# Layout
left, right = st.columns([1.6, 1])

with right:
    st.subheader("Modo de captura")

    tasks = ["PANEL: 4 cliques (corners)"]
    tasks += [f"TICK: {k}" for k in cfg["axis_ticks"].keys()]
    tasks += [f"LINE: {k} (2 cliques)" for k in cfg["lines"]]
    for gk in cfg.get("guides", {}).keys():
        tasks.append(f"GUIDE: {gk} (2 cliques)")

    task = st.selectbox("O que estás a fazer agora", tasks)

    panel_pick = None
    tick_axis = None
    tick_value = None
    line_key = None
    guide_key = None

    if task.startswith("PANEL"):
        panel_pick = st.selectbox("Qual painel?", cfg["panels"])

    elif task.startswith("TICK:"):
        tick_axis = task.split("TICK: ")[1].strip()
        tick_value = st.number_input("Valor do tick", value=0.0, step=1.0)

    elif task.startswith("LINE:"):
        line_key = task.split("LINE: ")[1].split(" (")[0].strip()

    elif task.startswith("GUIDE:"):
        guide_key = task.split("GUIDE: ")[1].split(" (")[0].strip()
        st.info(cfg["guides"][guide_key])

    # task id + auto reset pending if changed
    if task.startswith("PANEL"):
        task_id = f"PANEL:{panel_pick}"
    elif task.startswith("TICK:"):
        task_id = f"TICK:{tick_axis}"
    elif task.startswith("LINE:"):
        task_id = f"LINE:{line_key}"
    elif task.startswith("GUIDE:"):
        task_id = f"GUIDE:{guide_key}"
    else:
        task_id = task

    if st.session_state.prev_task_id is not None and st.session_state.prev_task_id != task_id:
        st.session_state.pending_segment = None
        st.session_state.pending_corners = []
    st.session_state.prev_task_id = task_id

    # delete last for current task
    def delete_last_for_task() -> bool:
        cap_local = st.session_state.cap

        if st.session_state.pending_segment is not None:
            st.session_state.pending_segment = None
            st.session_state.status_msg = "Ponto pendente apagado."
            return True

        if st.session_state.pending_corners:
            st.session_state.pending_corners.pop()
            st.session_state.status_msg = "Último corner pendente apagado."
            return True

        if task.startswith("TICK:") and tick_axis:
            if cap_local["axis_ticks"][tick_axis]:
                cap_local["axis_ticks"][tick_axis].pop()
                st.session_state.status_msg = f"Último tick apagado ({tick_axis})."
                return True

        if task.startswith("LINE:") and line_key:
            if cap_local["lines"][line_key]:
                cap_local["lines"][line_key].pop()
                st.session_state.status_msg = f"Último segmento apagado (line {line_key})."
                return True

        if task.startswith("GUIDE:") and guide_key:
            if cap_local["guides"][guide_key]:
                cap_local["guides"][guide_key].pop()
                st.session_state.status_msg = f"Última guide apagada ({guide_key})."
                return True

        if task.startswith("PANEL") and panel_pick:
            if cap_local["panel_corners"][panel_pick]:
                cap_local["panel_corners"][panel_pick] = []
                st.session_state.status_msg = f"Corners do painel {panel_pick} apagados."
                return True

        return False

    if st.button("🗑️ Apagar último (pendente/último item)"):
        push_history()
        if delete_last_for_task():
            st.rerun()
        else:
            st.info("Nada para apagar neste modo.")

    st.markdown("---")
    st.subheader("Último clique")
    if st.session_state.last_click is None:
        st.write("—")
    else:
        x, y = st.session_state.last_click
        st.write(f"x={int(x)}, y={int(y)}")

    st.subheader("Pendentes")
    st.write({
        "pending_segment": st.session_state.pending_segment,
        "pending_corners_count": len(st.session_state.pending_corners),
        "last_processed_click_id": st.session_state.last_processed_click_id,
    })

    if st.session_state.status_msg:
        st.success(st.session_state.status_msg)

    st.markdown("---")
    st.subheader("Resumo")
    cap_local = st.session_state.cap
    st.write({
        "panels": {p: len(cap_local["panel_corners"].get(p, [])) for p in cfg["panels"]},
        "ticks": {k: len(cap_local["axis_ticks"].get(k, [])) for k in cfg["axis_ticks"].keys()},
        "lines": {k: len(cap_local["lines"].get(k, [])) for k in cfg["lines"]},
        "guides": {k: len(cap_local["guides"].get(k, [])) for k in cfg.get("guides", {}).keys()},
    })

    st.markdown("---")
    st.subheader("Export / Guardar")
    txt = json.dumps(cap_local, indent=2)
    st.download_button("⬇️ Download JSON", data=txt, file_name=default_json_name, mime="application/json")
    if st.button(f"Guardar no repo como {default_json_name}"):
        Path(default_json_name).write_text(txt, encoding="utf-8")
        st.success(f"Guardado: {default_json_name}")

    st.markdown("---")
    with st.expander("Ver JSON"):
        st.code(txt, language="json")


with left:
    st.subheader("Imagem (sem duplicar cliques)")

    # overlay BEFORE (serve de input ao componente)
    overlay_before = draw_overlay(
        bg,
        st.session_state.cap,
        show_panels=show_panels,
        show_ticks=show_ticks,
        show_lines=show_lines,
        show_guides=show_guides,
        last_click=st.session_state.last_click,
        pending_segment=st.session_state.pending_segment,
        pending_corners=st.session_state.pending_corners,
    )

    click = streamlit_image_coordinates(np.array(overlay_before), key=f"img_{mode}_{int(page_index)}")

    # Processar clique APENAS se for novo (CRÍTICO)
    if click is not None:
        cid = click_id(click, mode=mode, page_index=int(page_index))
        if cid != st.session_state.last_processed_click_id:
            push_history()
            st.session_state.last_processed_click_id = cid

            x = float(click["x"])
            y = float(click["y"])
            st.session_state.last_click = (x, y)
            st.session_state.status_msg = ""

            cap_local = st.session_state.cap

            if task.startswith("PANEL"):
                st.session_state.pending_corners.append({"x": x, "y": y})
                if len(st.session_state.pending_corners) == 4:
                    cap_local["panel_corners"][panel_pick] = st.session_state.pending_corners
                    st.session_state.pending_corners = []
                    st.session_state.status_msg = f"Corners guardados em '{panel_pick}'."

            elif task.startswith("TICK:"):
                cap_local["axis_ticks"][tick_axis].append({"x": x, "y": y, "value": float(tick_value)})
                st.session_state.status_msg = f"Tick adicionado em {tick_axis} (v={tick_value})."

            elif task.startswith("LINE:"):
                pend = st.session_state.pending_segment
                if pend is None:
                    st.session_state.pending_segment = {"kind": "LINE", "key": line_key, "p": (x, y)}
                    st.session_state.status_msg = "Primeiro ponto guardado. Agora clica no segundo."
                else:
                    if pend["kind"] != "LINE" or pend["key"] != line_key:
                        st.session_state.pending_segment = {"kind": "LINE", "key": line_key, "p": (x, y)}
                        st.session_state.status_msg = "Mudaste de linha — novo primeiro ponto guardado."
                    else:
                        x1, y1 = pend["p"]
                        st.session_state.pending_segment = None
                        cap_local["lines"][line_key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                        st.session_state.status_msg = f"Segmento adicionado em lines['{line_key}']."

            elif task.startswith("GUIDE:"):
                pend = st.session_state.pending_segment
                if pend is None:
                    st.session_state.pending_segment = {"kind": "GUIDE", "key": guide_key, "p": (x, y)}
                    st.session_state.status_msg = "Primeiro ponto da guide guardado. Agora clica no segundo."
                else:
                    if pend["kind"] != "GUIDE" or pend["key"] != guide_key:
                        st.session_state.pending_segment = {"kind": "GUIDE", "key": guide_key, "p": (x, y)}
                        st.session_state.status_msg = "Mudaste de guide — novo primeiro ponto guardado."
                    else:
                        x1, y1 = pend["p"]
                        st.session_state.pending_segment = None
                        cap_local["guides"][guide_key].append({"x1": x1, "y1": y1, "x2": x, "y2": y})
                        st.session_state.status_msg = f"Guide adicionada em guides['{guide_key}']."

            st.session_state.cap = cap_local

    # overlay AFTER (mostra resultado final sempre)
    overlay_after = draw_overlay(
        bg,
        st.session_state.cap,
        show_panels=show_panels,
        show_ticks=show_ticks,
        show_lines=show_lines,
        show_guides=show_guides,
        last_click=st.session_state.last_click,
        pending_segment=st.session_state.pending_segment,
        pending_corners=st.session_state.pending_corners,
    )

    st.image(overlay_after, use_container_width=True)
    st.caption("Preto=último clique | Laranja=1º ponto pendente | Magenta=corners pendentes")

