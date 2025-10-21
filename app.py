# NAVLOG — Folium VFR + PDF — rev6 (refactor + cleanup)
# - OpenTopoMap por defeito
# - Seleção de WPs com desambiguação (rádio + botões ➕) e mini-mapa de preview
# - Mantém “adicionar no mapa (clique)” e “colar lista”
# - Mapa principal com:
#     • pílulas limpas: MH 197° • 198T • 90kt • 05:50 • 23.7nm
#       (sem amarelo; pílula branca com borda preta)
#     • leader line curta, offset dinâmico e anti-sobreposição
#     • riscas rigorosas a cada 2 min (pela GS)
#     • nomes de WPs a preto com halo branco
#     • botões: Fullscreen + Export PDF/PNG
# - Toggles acima do mapa: mostrar pílulas, mostrar riscas, tamanho do texto
# - Limpeza de imports, funções utilitárias com type hints, cache nas leituras CSV

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Dict, List, Optional, Tuple

import difflib
import pandas as pd
import streamlit as st
from branca.element import MacroElement, Template
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

# ================== CONSTANTES ==================
CLIMB_TAS = 70.0   # kt
CRUISE_TAS = 90.0  # kt
DESCENT_TAS = 90.0 # kt
FUEL_FLOW = 20.0   # L/h
EARTH_NM  = 3440.065

LABEL_MIN_NM  = 4.0
BASE_SIDE_OFF = 0.85  # afastamento lateral base (NM)
LINE_OFF      = 0.38  # comprimento da leader line (NM)

AD_CSV  = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"

# ================== PÁGINA / ESTILO ==================
st.set_page_config(page_title="NAVLOG — Folium VFR + PDF", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    :root{--line:#e5e7eb;--chip:#f3f4f6}
    *{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
    .card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
    .kvrow{display:flex;gap:8px;flex-wrap:wrap}
    .kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
    .sep{height:1px;background:var(--line);margin:10px 0}
    .sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid var(--line);padding-bottom:8px}
    small.muted{color:#6b7280}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== HELPERS ==================

def _round_to_10s(seconds: float) -> int:
    """Arredonda para o múltiplo de 10s mais próximo (mínimo 0)."""
    if seconds <= 0:
        return 0
    return max(10, int(round(seconds / 10.0) * 10))


def _mmss(total_seconds: float) -> str:
    total = int(total_seconds)
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def _hhmmss(total_seconds: float) -> str:
    total = int(total_seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ri(x: float) -> int:
    return int(round(float(x)))


def _r10f(x: float) -> float:
    return round(float(x), 1)


def _wrap360(x: float) -> int:
    return int((x % 360 + 360) % 360)


def _angdiff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def wind_triangle(tc: float, tas: float, wdir: float, wkt: float) -> Tuple[float, float, float]:
    """Resolve triângulo do vento. Devolve (WCA, TH, GS).
    - tc: true course (°)
    - tas: true airspeed (kt)
    - wdir: direção de onde vem o vento (°T)
    - wkt: intensidade do vento (kt)
    """
    if tas <= 0:
        return 0.0, _wrap360(tc), 0.0
    d = math.radians(_angdiff(wdir, tc))
    cross = wkt * math.sin(d)
    s = max(-1.0, min(1.0, cross / max(tas, 1e-9)))
    wca = math.degrees(math.asin(s))
    th  = _wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs


def apply_variation(true_heading: float, variation: float, east_is_negative: bool = False) -> int:
    """Converte TH→MH aplicando variação magnética. EAST subtrai se east_is_negative=True."""
    return _wrap360(true_heading - variation if east_is_negative else true_heading + variation)


def gc_dist_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância ortodrómica em NM."""
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_NM * c


def gc_course_tc(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Rumo verdadeiro inicial (TC) da ortodrómica."""
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360) % 360


def dest_point(lat: float, lon: float, bearing_deg: float, dist_nm: float) -> Tuple[float, float]:
    """Ponto destino a partir de (lat,lon), rumo e distância (NM)."""
    θ = math.radians(bearing_deg)
    δ = dist_nm / EARTH_NM
    φ1, λ1 = math.radians(lat), math.radians(lon)
    sinφ2 = math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ)
    φ2 = math.asin(sinφ2)
    y = math.sin(θ) * math.sin(δ) * math.cos(φ1)
    x = math.cos(δ) - math.sin(φ1) * sinφ2
    λ2 = λ1 + math.atan2(y, x)
    return math.degrees(φ2), ((math.degrees(λ2) + 540) % 360) - 180


def point_along_gc(lat1: float, lon1: float, lat2: float, lon2: float, dist_from_start_nm: float) -> Tuple[float, float]:
    total = gc_dist_nm(lat1, lon1, lat2, lon2)
    if total <= 0:
        return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)


def _nm_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return gc_dist_nm(a[0], a[1], b[0], b[1])


# ============ LABELS / ANTI-OVERLAP ============

def add_text_marker(map_obj: folium.Map, lat: float, lon: float, text: str, *, size_px: int = 16, color: str = "#111111", halo: bool = True, weight: str = "900") -> None:
    shadow = "text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;" if halo else ""
    html = f"""
    <div style="font-size:{size_px}px;color:{color};font-weight:{weight};{shadow};white-space:nowrap;">{text}</div>
    """
    folium.Marker(location=(lat, lon), icon=folium.DivIcon(html=html, icon_size=(0, 0))).add_to(map_obj)


def add_pill(m: folium.Map, lat: float, lon: float, text: str, *, font_px: int = 13) -> None:
    html = f"""
    <div style="font-size:{font_px}px; background:#fff; border:1px solid #000; border-radius:14px; padding:4px 10px; box-shadow:0 1px 2px rgba(0,0,0,.25); white-space:nowrap;">{text}</div>
    """
    folium.Marker((lat, lon), icon=folium.DivIcon(html=html, icon_size=(0, 0))).add_to(m)


def best_label_anchor(L: Dict, used_points: List[Tuple[float, float]], other_mids: List[Tuple[float, float]], side_off_nm: float, *, min_clear_nm: float = 0.7):
    """Escolhe âncora para a pílula. Candidatos: 1/3 e 2/3 do segmento, ambos os lados.
    Pontuação = menor distância às entidades (quanto maior, melhor), com bónus por afastamento lateral.
    Retorna (anchor_latlon, base_latlon, side_sign, clear_ok).
    """
    candidates = []
    crowd = used_points + other_mids + [(L["A"]["lat"], L["A"]["lon"]), (L["B"]["lat"], L["B"]["lon"])]
    for frac in (0.33, 0.67):
        along = max(0.7, min(L["Dist"] - 0.7, L["Dist"] * frac))
        base_lat, base_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], along)
        for side in (-1, +1):
            lab_lat, lab_lon = dest_point(base_lat, base_lon, L["TC"] + 90 * side, side_off_nm)
            dists = [_nm_dist((lab_lat, lab_lon), (p[0], p[1])) for p in crowd]
            near_route = _nm_dist((lab_lat, lab_lon), (base_lat, base_lon))  # ≈ lateral offset
            score = min(dists + [999]) + max(0.0, near_route - 0.6)  # bónus por afastamento da rota
            candidates.append((score, (lab_lat, lab_lon), (base_lat, base_lon), side))
    best = max(candidates, key=lambda x: x[0])
    clear_ok = best[0] >= min_clear_nm
    return (best[1], best[2], best[3], clear_ok)


# ================== STATE DEFAULTS ==================

def ens(key: str, default):
    return st.session_state.setdefault(key, default)


ens("wind_from", 0)
ens("wind_kt", 0)
ens("mag_var", 1.0)
ens("mag_is_e", False)
ens("roc_fpm", 600)
ens("desc_angle", 3.0)
ens("start_clock", "")
ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", [])
ens("legs", [])
ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")
ens("maptiler_key", "")
ens("show_pills", True)
ens("show_ticks", True)
ens("text_size", "Normal")  # Pequeno | Normal | Grande
ens("db_points", None)
ens("qadd", "")
ens("alt_qadd", 3000.0)
ens("radio_pick", None)

# ================== HEADER ==================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
a, b, c, d = st.columns([3, 3, 2, 2])
with a:
    st.title("NAVLOG — Folium VFR + PDF")
with b:
    st.caption("TAS 70/90/90 · FF 20 L/h · offsets em NM · pronto a imprimir")
with c:
    if st.button("➕ WP", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with d:
    if st.button("🗑️ Limpar", use_container_width=True):
        for k in ["wps", "legs", "route_nodes"]:
            st.session_state[k] = []
st.markdown("</div>", unsafe_allow_html=True)

# ================== PARÂMETROS ==================
with st.form("globals"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from))
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt))
    with c2:
        st.session_state.mag_var   = st.number_input("Variação magnética (±°)", -30.0, 30.0, float(st.session_state.mag_var))
        st.session_state.mag_is_e  = st.toggle("Var. é EAST (subtrai)", value=st.session_state.mag_is_e)
    with c3:
        st.session_state.roc_fpm    = st.number_input("ROC global (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10)
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
    with c4:
        st.session_state.start_efob  = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default))

    b1, b2 = st.columns([2, 2])
    with b1:
        bases = [
            "OpenTopoMap (VFR-ish)",
            "EOX Sentinel-2 (satélite)",
            "Esri World Imagery (satélite + labels)",
            "Esri World TopoMap (topo)",
            "OSM Standard",
            "MapTiler Satellite Hybrid (requer key)",
        ]
        cur = st.session_state.get("map_base", bases[0])
        idx = bases.index(cur) if cur in bases else 0
        st.session_state.map_base = st.selectbox("Base do mapa", bases, index=idx, key="map_base_choice")
    with b2:
        if "MapTiler" in st.session_state.map_base:
            st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key)

    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================== CSV / PARSE ==================

@st.cache_data(show_spinner=False)
def dms_to_dd(token: str, *, is_lon: bool = False) -> Optional[float]:
    """Converte DMS compactas para decimal.
    Aceita: "390712N" / "0083155W" (ddmmssH / dddmmssH) e também "39.4667N" / "8.1234W".
    """
    if token is None:
        return None
    s = str(token).strip().upper()

    # Decimal + hemisfério (ex: 39.4667N)
    m_dec = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([NSEW])", s)
    if m_dec:
        val = float(m_dec.group(1))
        hemi = m_dec.group(2)
        if hemi in ("S", "W"):
            val = -val
        return val

    # Compacto DMS: ddmmssH (lat) ou dddmmssH (lon)
    m_dms = re.fullmatch(r"(\d{2,3})(\d{2})(\d{2}(?:\.[0-9]+)?)([NSEW])", s)
    if not m_dms:
        return None
    deg = int(m_dms.group(1))
    minutes = int(m_dms.group(2))
    seconds = float(m_dms.group(3))
    hemi = m_dms.group(4)

    # validação simples
    if (is_lon and not (0 <= deg <= 180)) or ((not is_lon) and not (0 <= deg <= 90)):
        return None
    if not (0 <= minutes < 60) or not (0 <= seconds < 60.0):
        return None

    dd = deg + minutes / 60.0 + seconds / 3600.0
    if hemi in ("S", "W"):
        dd = -dd
    return dd


@st.cache_data(show_spinner=False)
def parse_ad_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:, 0].dropna().tolist():
        s = str(line).strip()
        if not s or s.startswith(("Ident", "DEP/")):
            continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.fullmatch(r"\d+(?:\.[0-9]+)?[NSEW]", t, flags=re.I)]
        if len(coord_toks) < 2:
            continue
        lat_tok, lon_tok = coord_toks[-2], coord_toks[-1]
        lat = dms_to_dd(lat_tok, is_lon=False)
        lon = dms_to_dd(lon_tok, is_lon=True)
        if lat is None or lon is None:
            continue
        ident = tokens[0] if re.fullmatch(r"[A-Z0-9]{4,}", tokens[0]) else None
        try:
            name = " ".join(tokens[1 : tokens.index(coord_toks[0])]).strip()
        except Exception:
            name = " ".join(tokens[1:]).strip()
        try:
            lon_idx = tokens.index(lon_tok)
            city = " ".join(tokens[lon_idx + 1 :]) or None
        except Exception:
            city = None
        rows.append({"src": "AD", "code": ident or name, "name": name, "city": city, "lat": lat, "lon": lon, "alt": 0.0})
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"]) if rows else pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"]) 


@st.cache_data(show_spinner=False)
def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:, 0].dropna().tolist():
        s = str(line).strip()
        if not s or "Total de registos" in s:
            continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.fullmatch(r"\d{6,7}(?:\.[0-9]+)?[NSEW]", t)]
        if len(coord_toks) < 2:
            continue
        lat_tok, lon_tok = coord_toks[0], coord_toks[1]
        lat = dms_to_dd(lat_tok, is_lon=False)
        lon = dms_to_dd(lon_tok, is_lon=True)
        if lat is None or lon is None:
            continue
        try:
            lon_idx = tokens.index(lon_tok)
        except ValueError:
            continue
        code = tokens[lon_idx + 1] if lon_idx + 1 < len(tokens) else None
        sector = " ".join(tokens[lon_idx + 2 :]) if lon_idx + 2 < len(tokens) else None
        name = " ".join(tokens[: tokens.index(lat_tok)]).strip()
        rows.append({"src": "LOC", "code": code or name, "name": name, "sector": sector, "lat": lat, "lon": lon, "alt": 0.0})
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"]) if rows else pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"]) 


@st.cache_data(show_spinner=False)
def load_databases(ad_csv: str, loc_csv: str) -> pd.DataFrame:
    try:
        ad_raw = pd.read_csv(ad_csv)
        ad_df = parse_ad_df(ad_raw)
    except Exception:
        ad_df = pd.DataFrame(columns=["src", "code", "name", "city", "lat", "lon", "alt"])
    try:
        loc_raw = pd.read_csv(loc_csv)
        loc_df = parse_loc_df(loc_raw)
    except Exception:
        loc_df = pd.DataFrame(columns=["src", "code", "name", "sector", "lat", "lon", "alt"])

    if ad_df.empty and loc_df.empty:
        st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")
        return pd.DataFrame(columns=["src", "code", "name", "lat", "lon", "alt"])

    db = pd.concat([ad_df, loc_df]).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return db


# base única de pesquisa (memo na sessão)
if st.session_state.db_points is None:
    st.session_state.db_points = load_databases(AD_CSV, LOC_CSV)

db = st.session_state.db_points

# ================== ADICIONAR WPs — DESAMBIGUAÇÃO ==================
st.subheader("Adicionar waypoints")


def add_wp_unique(name: str, lat: float, lon: float, alt: float) -> bool:
    """Evita duplicados por nome+proximidade (<= 0.2 NM)."""
    for w in st.session_state.wps:
        if str(w["name"]).strip().lower() == str(name).strip().lower():
            if gc_dist_nm(w["lat"], w["lon"], lat, lon) <= 0.2:
                return False
    st.session_state.wps.append({"name": str(name), "lat": float(lat), "lon": float(lon), "alt": float(alt)})
    return True


def _score_row(row: pd.Series, tq: str, last_wp: Optional[Dict]) -> float:
    code = str(row.get("code") or "").lower()
    name = str(row.get("name") or "").lower()
    text = f"{code} {name}"
    sim = difflib.SequenceMatcher(None, tq, text).ratio()
    starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
    near = 0.0
    if last_wp:
        near = 1.0 / (1.0 + gc_dist_nm(last_wp["lat"], last_wp["lon"], row["lat"], row["lon"]))
    exact = 1.0 if code == tq or name == tq else 0.0
    return exact * 3 + starts * 2 + sim + near * 0.3


def _search_points(tq: str) -> pd.DataFrame:
    if not tq:
        return db.head(0)
    tq = tq.lower().strip()
    last = st.session_state.wps[-1] if st.session_state.wps else None
    sel = db[db.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)].copy()
    if sel.empty:
        return sel
    sel["__score"] = sel.apply(lambda r: _score_row(r, tq, last), axis=1)
    return sel.sort_values("__score", ascending=False)


srch1, srch2 = st.columns([3, 1])
with srch1:
    st.text_input("Pesquisar (escolhe 1 abaixo)", key="qadd", placeholder="Ex: LPPT, ALPAL, ÉVORA, NISA…")
with srch2:
    st.number_input("Alt (ft) para novos WPs", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0, key="alt_qadd")

res_df = _search_points(st.session_state.qadd.strip()) if st.session_state.qadd.strip() else pd.DataFrame()
if not res_df.empty:
    labels: List[str] = []
    idx_map: Dict[str, int] = {}
    for i, r in res_df.head(20).reset_index(drop=True).iterrows():
        lbl = f"[{r['src']}] {r.get('code','')} — {r.get('name','')}  ({r['lat']:.4f}, {r['lon']:.4f})"
        labels.append(lbl)
        idx_map[lbl] = i
    st.session_state.radio_pick = st.radio("Resultados (1 escolha):", labels, index=0, key="radio_pick_lbl")

    pick_idx = idx_map.get(st.session_state.radio_pick, 0)
    row_pick = res_df.iloc[pick_idx]

    cadd1, cadd2, cadd3 = st.columns([1, 1, 4])
    with cadd1:
        if st.button("Adicionar selecionado"):
            ok = add_wp_unique(row_pick.get("code") or row_pick.get("name"), float(row_pick["lat"]), float(row_pick["lon"]), float(st.session_state.alt_qadd))
            (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")
    with cadd2:
        if st.button("Adicionar 1.º"):
            r0 = res_df.iloc[0]
            ok = add_wp_unique(r0.get("code") or r0.get("name"), float(r0["lat"]), float(r0["lon"]), float(st.session_state.alt_qadd))
            (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")
    with cadd3:
        st.caption("Pré-visualização do selecionado")
        pm = folium.Map(
            location=[row_pick["lat"], row_pick["lon"]],
            zoom_start=10,
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="© OpenTopoMap",
            control_scale=True,
        )
        # WPs já existentes (azul)
        for w in st.session_state.wps:
            folium.CircleMarker((w["lat"], w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1).add_to(pm)
        # candidato (laranja)
        folium.CircleMarker((row_pick["lat"], row_pick["lon"]), radius=7, color="#ff7f00", fill=True, fill_opacity=1).add_to(pm)
        st_folium(pm, width=None, height=260, key="preview_map")

# Tabs extra
_tabs = st.tabs(["🗺️ Adicionar no mapa (clique)", "📋 Colar lista"])
with _tabs[0]:
    st.caption("Clica no mapa e depois em **Adicionar**.")
    m0 = folium.Map(
        location=[39.7, -8.1],
        zoom_start=7,
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap",
        control_scale=True,
    )
    for w in st.session_state.wps:
        folium.CircleMarker((w["lat"], w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1).add_to(m0)
    map_out = st_folium(m0, width=None, height=420, key="pickmap")
    with st.form("add_by_click"):
        cA, cB, _ = st.columns([2, 1, 1])
        with cA:
            nm = st.text_input("Nome", "WP novo")
        with cB:
            alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
        clicked = map_out.get("last_clicked")
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            ok = add_wp_unique(nm, clicked["lat"], clicked["lng"], alt)
            (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")

with _tabs[1]:
    st.caption("Formato por linha: `NOME; LAT; LON; ALT` — aceita DD ou DMS compactas (`390712N 0083155W`). ALT opcional.")
    txt = st.text_area(
        "Lista",
        height=120,
        placeholder="ABRANTES; 39.4667; -8.2; 3000\nPONTO X; 390712N; 0083155W; 2500",
    )
    alt_def = st.number_input("Alt (ft) se faltar", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0, key="alt_def_paste")

    if st.button("Adicionar da lista"):
        n = 0
        for line in txt.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 3:
                continue
            name = parts[0]
            lat_token = parts[1].replace(",", ".")
            lon_token = parts[2].replace(",", ".")
            lat = dms_to_dd(lat_token, is_lon=False) if re.search(r"[NnSs]$", lat_token) else float(lat_token)
            lon = dms_to_dd(lon_token, is_lon=True)  if re.search(r"[EeWw]$", lon_token) else float(lon_token)
            alt = float(parts[3]) if len(parts) >= 4 and parts[3] else alt_def
            if lat is None or lon is None:
                continue
            n += 1 if add_wp_unique(name, lat, lon, alt) else 0
        st.success(f"Adicionados {n} WPs.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================== EDITOR WPs ==================
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
            with c1:
                name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2:
                lat = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3:
                lon = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4:
                alt = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                up = st.button("↑", key=f"up{i}")
                dn = st.button("↓", key=f"dn{i}")
                if up and i > 0:
                    st.session_state.wps[i - 1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i - 1]
                if dn and i < len(st.session_state.wps) - 1:
                    st.session_state.wps[i + 1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i + 1]
            if (name, lat, lon, alt) != (w["name"], w["lat"], w["lon"], w["alt"]):
                st.session_state.wps[i] = {"name": name, "lat": float(lat), "lon": float(lon), "alt": float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================== TOC/TOD ==================

def build_route_nodes(user_wps: List[Dict], wind_from: float, wind_kt: float, roc_fpm: float, desc_angle_deg: float) -> List[Dict]:
    nodes: List[Dict] = []
    if len(user_wps) < 2:
        return nodes
    for i in range(len(user_wps) - 1):
        A, B = user_wps[i], user_wps[i + 1]
        nodes.append(A)
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"]) 
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"]) 
        _, _, gs_cl = wind_triangle(tc, CLIMB_TAS,   wind_from, wind_kt)
        _, _, gs_de = wind_triangle(tc, DESCENT_TAS, wind_from, wind_kt)

        if B["alt"] > A["alt"]:
            dh = B["alt"] - A["alt"]
            t_need = dh / max(roc_fpm, 1)
            d_need = gs_cl * (t_need / 60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:
            # ROD aproximado a partir do plano de 3°: ROD ≈ GS × 5 × (ângulo/3)
            rod_fpm = max(100.0, gs_de * 5.0 * (desc_angle_deg / 3.0))
            dh = A["alt"] - B["alt"]
            t_need = dh / max(rod_fpm, 1)
            d_need = gs_de * (t_need / 60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
    nodes.append(user_wps[-1])
    return nodes


# ================== LEGS ==================

def build_legs_from_nodes(
    nodes: List[Dict],
    wind_from: float,
    wind_kt: float,
    mag_var: float,
    mag_is_e: bool,
    ck_every_min: int,
) -> List[Dict]:
    legs: List[Dict] = []
    if len(nodes) < 2:
        return legs

    base_time: Optional[dt.datetime] = None
    if st.session_state.start_clock.strip():
        try:
            h, m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h, m))
        except Exception:
            base_time = None

    carry_efob = float(st.session_state.start_efob)
    t_cursor = 0

    for i in range(len(nodes) - 1):
        A, B = nodes[i], nodes[i + 1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        profile = "LEVEL" if abs(B["alt"] - A["alt"]) < 1e-6 else ("CLIMB" if B["alt"] > A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile == "CLIMB" else (DESCENT_TAS if profile == "DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
        mh = apply_variation(th, mag_var, mag_is_e)
        time_sec = _round_to_10s((dist / max(gs, 1e-9)) * 3600.0) if gs > 0 else 0
        burn = FUEL_FLOW * (time_sec / 3600.0)
        efob_start = carry_efob
        efob_end = max(0.0, _r10f(efob_start - burn))
        clk_start = (base_time + dt.timedelta(seconds=t_cursor)).strftime("%H:%M") if base_time else f"T+{_mmss(t_cursor)}"
        clk_end = (base_time + dt.timedelta(seconds=t_cursor + time_sec)).strftime("%H:%M") if base_time else f"T+{_mmss(t_cursor + time_sec)}"

        cps: List[Dict] = []
        if ck_every_min > 0 and gs > 0:
            k = 1
            while k * ck_every_min * 60 <= time_sec:
                t = k * ck_every_min * 60
                d = gs * (t / 3600.0)
                eto = (base_time + dt.timedelta(seconds=t_cursor + t)).strftime("%H:%M") if base_time else ""
                cps.append({"t": t, "min": int(t / 60), "nm": round(d, 1), "eto": eto})
                k += 1

        legs.append(
            {
                "i": i + 1,
                "A": A,
                "B": B,
                "profile": profile,
                "TC": tc,
                "TH": th,
                "MH": mh,
                "TAS": tas,
                "GS": gs,
                "Dist": dist,
                "time_sec": time_sec,
                "burn": _r10f(burn),
                "efob_start": efob_start,
                "efob_end": efob_end,
                "clock_start": clk_start,
                "clock_end": clk_end,
                "cps": cps,
            }
        )
        t_cursor += time_sec
        carry_efob = efob_end
    return legs


# ================== GERAR ROTA/LEGS ==================
cgen, _ = st.columns([2, 6])
with cgen:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(
            st.session_state.wps,
            st.session_state.wind_from,
            st.session_state.wind_kt,
            st.session_state.roc_fpm,
            st.session_state.desc_angle,
        )
        st.session_state.legs = build_legs_from_nodes(
            st.session_state.route_nodes,
            st.session_state.wind_from,
            st.session_state.wind_kt,
            st.session_state.mag_var,
            st.session_state.mag_is_e,
            st.session_state.ck_default,
        )

# ================== RESUMO ==================
if st.session_state.legs:
    total_sec = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = _r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final = st.session_state.legs[-1]["efob_end"]
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{_hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b> (20 L/h)</div>"
        + f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>"
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================== CONTROLOS DO MAPA (TOGGLES) ==================
t1, t2, t3 = st.columns([1.2, 1.2, 2])
with t1:
    st.session_state.show_pills = st.toggle("Mostrar pílulas", value=st.session_state.show_pills)
with t2:
    st.session_state.show_ticks = st.toggle("Mostrar riscas 2 min", value=st.session_state.show_ticks)
with t3:
    st.session_state.text_size = st.selectbox("Tamanho do texto", ["Pequeno", "Normal", "Grande"], index=["Pequeno", "Normal", "Grande"].index(st.session_state.text_size))


# ================== MAPA (FOLIUM) ==================

def _bounds_from_nodes(nodes: List[Dict]) -> List[Tuple[float, float]]:
    lats = [n["lat"] for n in nodes]
    lons = [n["lon"] for n in nodes]
    return [(min(lats), min(lons)), (max(lats), max(lons))]


def _route_latlngs(legs: List[Dict]) -> List[List[Tuple[float, float]]]:
    return [[(L["A"]["lat"], L["A"]["lon"]), (L["B"]["lat"], L["B"]["lon"])] for L in legs]


def add_print_and_fullscreen(m: folium.Map) -> None:
    # Export PDF/PNG
    m.get_root().header.add_child(
        folium.Element('<script src="https://unpkg.com/leaflet.browser.print/dist/leaflet.browser.print.min.js"></script>')
    )
    tpl = """
    {% macro script(this, kwargs) %}
      const mp = {{this._parent.get_name()}};
      L.control.browserPrint({position:'topleft', title:'Export PDF/PNG',
                              printModes:['Portrait','Landscape','Auto','Custom']}).addTo(mp);
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(tpl)
    m.get_root().script.add_child(macro)
    # Fullscreen
    Fullscreen(position='topleft', title='Fullscreen', force_separate_button=True).add_to(m)


def render_map(nodes: List[Dict], legs: List[Dict], *, base_choice: str, maptiler_key: str = "") -> None:
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return

    mean_lat = sum(n["lat"] for n in nodes) / len(nodes)
    mean_lon = sum(n["lon"] for n in nodes) / len(nodes)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8, tiles=None, control_scale=True, prefer_canvas=True)

    # Camadas base
    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer(
            tiles="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
            attr="© EOX Sentinel-2",
            name="Sentinel-2",
            overlay=False,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png",
            attr="© EOX Overlay",
            name="Labels",
            overlay=True,
        ).add_to(m)
    elif base_choice == "Esri World Imagery (satélite + labels)":
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri",
            name="Esri Imagery",
            overlay=False,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri",
            name="Labels",
            overlay=True,
        ).add_to(m)
    elif base_choice == "Esri World TopoMap (topo)":
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri",
            name="Esri Topo",
            overlay=False,
        ).add_to(m)
    elif base_choice == "OpenTopoMap (VFR-ish)":
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="© OpenTopoMap (CC-BY-SA)",
            name="OpenTopoMap",
            overlay=False,
        ).add_to(m)
    elif base_choice == "MapTiler Satellite Hybrid (requer key)" and maptiler_key:
        folium.TileLayer(
            tiles=f"https://api.maptiler.com/maps/hybrid/256/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_key}",
            attr="© MapTiler",
            name="MapTiler Hybrid",
            overlay=False,
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="© OpenStreetMap",
            name="OSM",
            overlay=False,
        ).add_to(m)

    # Rota com halo
    for latlngs in _route_latlngs(legs):
        folium.PolyLine(latlngs, color="#ffffff", weight=9, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color="#C000FF", weight=4, opacity=1.0).add_to(m)

    # Riscas de 2 minutos
    if st.session_state.show_ticks:
        for L in legs:
            if L["GS"] <= 0 or L["time_sec"] < 120:
                continue
            t = 120
            while t <= L["time_sec"]:
                d = min(L["Dist"], L["GS"] * (t / 3600.0))
                latm, lonm = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], d)
                llat, llon = dest_point(latm, lonm, L["TC"] - 90, 0.18)
                rlat, rlon = dest_point(latm, lonm, L["TC"] + 90, 0.18)
                folium.PolyLine([(llat, llon), (rlat, rlon)], color="#555555", weight=2, opacity=1).add_to(m)
                t += 120

    # Pontos médios para evitar choques
    other_mids: List[Tuple[float, float]] = []
    for L in legs:
        mid_lat, mid_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], L["Dist"] / 2.0)
        other_mids.append((mid_lat, mid_lon))

    # Pílulas de info (MH + TH • GS • ETE • NM)
    if st.session_state.show_pills:
        font_px = 12 if st.session_state.text_size == "Pequeno" else (15 if st.session_state.text_size == "Grande" else 13)
        used: List[Tuple[float, float]] = []
        for ix, L in enumerate(legs):
            if L["Dist"] < LABEL_MIN_NM or L["GS"] <= 0 or L["time_sec"] <= 0:
                continue
            mids = other_mids[:ix] + other_mids[ix + 1 :]

            side_off = BASE_SIDE_OFF
            for _ in range(5):  # tenta com offset crescente
                anchor, base, side, ok = best_label_anchor(L, used, mids, side_off)
                if ok:
                    break
                side_off += 0.15

            # Leader line
            mid_pt = dest_point(base[0], base[1], L["TC"] + 90 * side, min(LINE_OFF, side_off - 0.05))
            folium.PolyLine([(base[0], base[1]), (mid_pt[0], mid_pt[1])], color="#000000", weight=2, opacity=1).add_to(m)

            info_txt = f"<b>MH {_wrap360(L['MH'])}°</b> • {_wrap360(L['TH'])}T • {_ri(L['GS'])}kt • {_mmss(L['time_sec'])} • {L['Dist']:.1f}nm"
            add_pill(m, anchor[0], anchor[1], info_txt, font_px=font_px)
            used.append(anchor)

    # WPs (ponto + nome legível)
    font_wp = 14 if st.session_state.text_size != "Pequeno" else 13
    for idx, N in enumerate(nodes):
        is_toc_tod = str(N["name"]).startswith(("TOC", "TOD"))
        color = "#FF5050" if is_toc_tod else "#007AFF"
        folium.CircleMarker((N["lat"], N["lon"]), radius=6, color="#FFFFFF", weight=2, fill=True, fill_color=color, fill_opacity=1).add_to(m)
        add_text_marker(m, N["lat"], N["lon"], f"{idx+1}. {N['name']}", size_px=font_wp, color="#111111", halo=True)

    try:
        m.fit_bounds(_bounds_from_nodes(nodes), padding=(30, 30))
    except Exception:
        pass

    add_print_and_fullscreen(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=760)


# ---- render ----
if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(
        st.session_state.route_nodes,
        st.session_state.legs,
        base_choice=st.session_state.map_base,
        maptiler_key=st.session_state.maptiler_key,
    )
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")


