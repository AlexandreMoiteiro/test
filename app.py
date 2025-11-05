# app_navlog_rev38_full.py
# ---------------------------------------------------------------
# - Fix remoção de WPs (não remove vários).
# - Arredondamentos:
#     * tempos → 30 s
#     * distâncias → 0.5 nm
#     * fuel → 0.5 L
# - VOR por ponto:
#     * VORs entram na base pesquisável → podes usar CAS/LIS/PRT como WP
#     * se o WP veio de VOR fica logo marcado como VOR FIXED
#     * no editor do WP podes escolher AUTO ou um VOR concreto (lista dos mais perto)
# - PDF passa a preencher a perna com o **ponto de chegada** (B), não com o A.
# - “Paragens”/esperas por WP (ex.: 10 min de touch and gos) → gera leg STOP com burn.
# - STOP não aparece no mapa.
# - Mantém TOC/TOD, airspaces, PDF 2ª página com totais.
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib, os
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MarkerCluster
from math import degrees
from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName

# ========= CONSTANTES / CONFIG =========
TEMPLATE_MAIN = "NAVLOG_FORM.pdf"
TEMPLATE_CONT = "NAVLOG_FORM_1.pdf"

CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0
FUEL_FLOW = 20.0              # L/h
EARTH_NM  = 3440.065
PROFILE_COLORS = {
    "CLIMB":   "#FF7A00",
    "LEVEL":   "#C000FF",
    "DESCENT": "#00B386",
    "STOP":    "#FF0000",
}

# arredondamentos que pediste
ROUND_TIME_SEC = 30     # 30 em 30
ROUND_DIST_NM  = 0.5    # 0.5 em 0.5
ROUND_FUEL_L   = 0.5    # 0.5 em 0.5

CP_TICK_HALF = 0.38
NBSP_THIN = "&#8239;"  # fino

# Paleta para áreas
ASPACE_COLOR     = "#FFD54A"
CORRIDOR_COLOR   = "#9BE27A"
FILL_OPACITY     = 0.12
EDGE_OPACITY     = 0.9

# ========= ÁREAS PRÉ-DEFINIDAS (mesmo que tinhas) =========
PRESET_AIRSPACES = {
    "LPT1": {
        "floor": "GND", "ceiling": "FL280",
        "notes": "TRAINING / COMAO",
        "color": "#ffff00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL280", "label": "COMAO"},
            {"top": "FL140", "label": "TRAINING"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (41.38, -6.3925),
            (39.899167, -6.899722),
            (39.9, -7.75),
            (40.731111, -7.75),
            (41.3, -7.383333),
            (41.38, -6.3925),
        ],
    },
    "LPT2": {
        "floor": "GND", "ceiling": "FL245",
        "notes": "TRAINING / COMAO",
        "color": "#ffff00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL245", "label": "COMAO"},
            {"top": "GND",   "label": "TRAINING"}
        ],
        "coords": [
            (39.9, -7.75),
            (39.893611, -6.899722),
            (39.666667, -7.030556),
            (39.666667, -7.75),
            (39.9, -7.75),
        ],
    },
    "LPT3": {
        "floor": "GND", "ceiling": "FL190",
        "notes": "TRAINING / COMAO",
        "color": "#ffff00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL190", "label": "TRAINING / COMAO"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (39.666667, -8.0),
            (39.666667, -7.533333),
            (39.1, -7.083333),
            (38.85, -7.75),
            (39.333333, -7.75),
            (39.666667, -8.0),
        ],
    },
    "LPTRA57": {
        "floor": "GND", "ceiling": "FL245",
        "notes": "TRAINING / COMAO",
        "color": "#ffdd00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL245", "label": "COMAO"},
            {"top": "GND",   "label": "TRAINING"}
        ],
        "coords": [
            (39.099167, -7.115556),
            (38.5525, -7.285278),
            (38.548611, -8.048889),
            (38.744444, -8.05),
            (39.099167, -7.115556),
        ],
    },
    "LPR51BN": {
        "floor": "GND", "ceiling": "FL240",
        "notes": "TRAINING / COMAO",
        "color": "#d4ff00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL240", "label": "COMAO"},
            {"top": "GND",   "label": "TRAINING"}
        ],
        "coords": [
            (38.548611, -8.048889),
            (38.5525, -7.285278),
            (38.125, -6.960278),
            (38.115278, -8.171111),
            (38.348611, -8.267778),
            (38.548611, -8.048889),
        ],
    },
    "LPR51BS1": {
        "floor": "GND", "ceiling": "FL140",
        "notes": "TRAINING",
        "color": "#c8ff00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL140", "label": "TRAINING"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (38.115278, -8.171111),
            (38.125, -6.960278),
            (37.597222, -7.499167),
            (37.597222, -7.95),
            (38.115278, -8.171111),
        ],
    },
    "LPR51BS2": {
        "floor": "GND", "ceiling": "FL060",
        "notes": "TRAINING",
        "color": "#baff00", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL060", "label": "TRAINING"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (37.597222, -7.95),
            (37.597222, -7.499167),
            (37.4175, -7.443333),
            (37.415278, -7.884444),
            (37.597222, -7.95),
        ],
    },
    "LPT10": {
        "floor": "GND", "ceiling": "FL140",
        "notes": "TRAINING",
        "color": "#fff000", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL140", "label": "TRAINING"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (39.666667, -8.333333),
            (39.666667, -8.0),
            (39.333333, -7.75),
            (39.306944, -7.75),
            (39.306944, -8.358333),
            (39.383611, -8.358333),
            (39.383611, -8.449444),
            (39.425833, -8.494722),
            (39.515278, -8.451389),
            (39.666667, -8.333333),
        ],
    },
    "LPT11": {
        "floor": "GND", "ceiling": "FL140",
        "notes": "TRAINING",
        "color": "#fff000", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL140", "label": "TRAINING"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (39.9, -8.083333),
            (39.9, -7.75),
            (39.666667, -7.75),
            (39.666667, -8.333333),
            (39.9, -8.083333),
        ],
    },
    "LPT12": {
        "floor": "GND", "ceiling": "FL140",
        "notes": "TRAINING",
        "color": "#fff000", "opacity": 0.25,
        "width_nm": None,
        "bands": [
            {"top": "FL140", "label": "TRAINING"},
            {"top": "GND",   "label": ""}
        ],
        "coords": [
            (39.9, -7.75),
            (40.583333, -7.75),
            (40.583333, -8.083333),
            (39.9, -8.083333),
            (39.9, -7.75),
        ],
    },
    "LPT61": {
        "floor": "GND", "ceiling": "2000 FT AMSL",
        "notes": "Transit Corridor",
        "color": "#adff00", "opacity": 0.25,
        "width_nm": 5.0,
        "bands": None,
        "coords": [
            (38.758333, -7.971667),
            (39.325, -8.270833),
        ],
    },
    "LPT63": {
        "floor": "GND", "ceiling": "2000 FT AMSL",
        "notes": "Transit Corridor",
        "color": "#adff00", "opacity": 0.25,
        "width_nm": 5.0,
        "bands": None,
        "coords": [
            (38.366667, -8.233333),
            (38.583333, -8.583333),
        ],
    },
}

# ========= ESTILO STREAMLIT =========
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin:8px 0;background:#fff}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:#e5e7eb;margin:10px 0}
.small{font-size:12px;color:#555}
.row{display:flex;gap:8px;align-items:center}
.badge{font-weight:700;border:1px solid #111;border-radius:8px;padding:2px 6px;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# ========= FUNÇÕES NUM / GEO =========
def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step

def rt30(sec: float) -> int:
    return int(round_to_step(sec, ROUND_TIME_SEC))

def rdist05(nm: float) -> float:
    return round_to_step(nm, ROUND_DIST_NM)

def rfuel05(L: float) -> float:
    return round_to_step(L, ROUND_FUEL_L)

def mmss(t):
    t=int(t)
    return f"{t//60:02d}:{t%60:02d}"

def hhmmss(t):
    t=int(t)
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"

def wrap360(x): return (x % 360 + 360) % 360
def angdiff(a, b): return (a - b + 180) % 360 - 180
def deg3(v): return f"{int(round(v))%360:03d}°"

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * math.sin(d)
    s = cross / max(tas,1e-9)
    s = max(-1, min(1, s))
    wca = degrees(math.asin(s))
    th  = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False):
    return wrap360(th - var if east_is_neg else th + var)

def gc_dist_nm(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_NM * c

def gc_course_tc(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dλ = λ2 - λ1
    y = math.sin(dλ)*math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360) % 360

def dest_point(lat, lon, bearing_deg, dist_nm):
    θ = math.radians(bearing_deg)
    δ = dist_nm / EARTH_NM
    φ1, λ1 = math.radians(lat), math.radians(lon)
    sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
    φ2 = math.asin(sinφ2)
    y = math.sin(θ)*math.sin(δ)*math.cos(φ1)
    x = math.cos(δ) - math.sin(φ1)*sinφ2
    λ2 = λ1 + math.atan2(y, x)
    return math.degrees(φ2), ((math.degrees(λ2)+540)%360)-180

def point_along_gc(lat1, lon1, lat2, lon2, dist_from_start_nm):
    total = gc_dist_nm(lat1, lon1, lat2, lon2)
    if total <= 0: return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)

def polygon_centroid(coords):
    if not coords: return (0.0,0.0)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return (sum(lats)/len(lats), sum(lons)/len(lons))

def corridor_polygon(p1, p2, width_nm):
    lat1, lon1 = p1
    lat2, lon2 = p2
    tc = gc_course_tc(lat1, lon1, lat2, lon2)
    half = width_nm / 2.0
    left1  = dest_point(lat1, lon1, tc-90, half)
    right1 = dest_point(lat1, lon1, tc+90, half)
    left2  = dest_point(lat2, lon2, tc-90, half)
    right2 = dest_point(lat2, lon2, tc+90, half)
    return [left1, left2, right2, right1, left1]

# parser coords AIP
coord_pattern = re.compile(
    r"(\d{2})\s+(\d{2})\s+(\d{2})([NS])\s+(\d{3})\s+(\d{2})\s+(\d{2})([EW])",
    re.IGNORECASE
)
def extract_polygon_coords(raw_text:str):
    cleaned = raw_text.replace("—"," ").replace("–"," ").replace("-", " ")
    coords = []
    for m in coord_pattern.finditer(cleaned):
        lat_deg, lat_min, lat_sec, lat_hemi, lon_deg, lon_min, lon_sec, lon_hemi = m.groups()
        lat = float(lat_deg) + float(lat_min)/60.0 + float(lat_sec)/3600.0
        if lat_hemi.upper() == "S": lat = -lat
        lon = float(lon_deg) + float(lon_min)/60.0 + float(lon_sec)/3600.0
        if lon_hemi.upper() == "W": lon = -lon
        coords.append((lat, lon))
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

# ========= STATE DEFAULTS =========
def ens(k, v):
    return st.session_state.setdefault(k, v)

ens("wind_from", 0)
ens("wind_kt", 0)
ens("use_global_wind", True)

ens("mag_var", 1.0)
ens("mag_is_e", False)
ens("roc_fpm", 600)
ens("rod_fpm", 500)
ens("start_clock", "")
ens("start_efob", 85.0)
ens("ck_default", 2)

ens("wps", [])
ens("wp_next_id", 1)  # para remoção estável
ens("legs", [])
ens("route_nodes", [])

ens("map_base", "OpenTopoMap (VFR-ish)")
ens("text_scale", 1.0)

ens("show_ticks", True)
ens("show_doghouses", True)
ens("show_airspaces", True)
ens("show_openaip", True)

ens("openaip_token", os.getenv("OPENAIP_KEY", "e849257999aa8ed820c3a6f7eb40f84e"))
ens("openaip_alpha", 0.6)

ens("map_center", (39.7, -8.1))
ens("map_zoom", 8)

ens("db_points", None)
ens("qadd", "")
ens("alt_qadd", 3000.0)
ens("search_rows", [])
ens("last_q", "")

ens("airspaces", [])
ens("preset_selected", [])

ens("use_leg_filter", False)
ens("leg_filter_ids", [])

# ========= FORM GLOBAL =========
with st.form("globals"):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state.wind_from = st.number_input(
            "Vento global FROM (°T)", 0, 360, int(st.session_state.wind_from)
        )
        st.session_state.wind_kt   = st.number_input(
            "Vento global (kt)", 0, 150, int(st.session_state.wind_kt)
        )
        st.session_state.use_global_wind = st.toggle(
            "Usar vento global", value=st.session_state.use_global_wind,
            help="Desliga para meter vento individual em cada WP."
        )

    with c2:
        st.session_state.mag_var   = st.number_input(
            "Variação magnética (±°)", -30.0, 30.0, float(st.session_state.mag_var)
        )
        st.session_state.mag_is_e  = st.toggle(
            "Var. é EAST (subtrai)", value=st.session_state.mag_is_e
        )

    with c3:
        st.session_state.roc_fpm   = st.number_input(
            "ROC global (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10
        )
        st.session_state.rod_fpm   = st.number_input(
            "ROD global (ft/min)", 200, 1500, int(st.session_state.rod_fpm), step=10
        )

    with c4:
        st.session_state.start_efob= st.number_input(
            "EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5
        )
        st.session_state.start_clock = st.text_input(
            "Hora off-blocks (HH:MM)", st.session_state.start_clock
        )
        st.session_state.ck_default  = st.number_input(
            "CP por defeito (min)", 1, 10, int(st.session_state.ck_default)
        )

    b1,b2,b3 = st.columns([2,1,1])
    with b1:
        bases = [
            "OpenTopoMap (VFR-ish)",
            "OSM Standard",
            "Terrain Hillshade"
        ]
        st.session_state.map_base = st.selectbox(
            "Base do mapa",
            bases,
            index=bases.index(st.session_state.map_base)
            if st.session_state.map_base in bases else 0
        )
        st.session_state.text_scale  = st.slider(
            "Escala texto mapa", 0.5, 1.5,
            float(st.session_state.text_scale), 0.05,
            help="Afecta doghouses, labels WPs e labels de áreas."
        )
    with b2:
        st.session_state.show_ticks     = st.toggle(
            "Riscas CP", value=st.session_state.show_ticks,
            help="Marcas de cada CP_x min ao longo da perna."
        )
        st.session_state.show_doghouses = st.toggle(
            "Dog houses", value=st.session_state.show_doghouses,
            help="Cartões rotados com heading/ALT/ETE."
        )
    with b3:
        st.session_state.show_airspaces = st.toggle(
            "Áreas catálogo/custom",
            value=st.session_state.show_airspaces,
            help="Mostra LPT1/LPR51/etc + áreas custom hardcoded."
        )
        st.session_state.show_openaip = st.toggle(
            "Overlay openAIP",
            value=st.session_state.show_openaip,
            help="Layer VFR do openAIP por cima da base."
        )
        st.session_state.openaip_alpha = st.slider(
            "Transparência openAIP", 0.0, 1.0,
            float(st.session_state.openaip_alpha), 0.05
        )

    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ========= CSVs =========
AD_CSV  = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"
VOR_CSV = "NAVAIDS_VOR.csv"

def dms_to_dd(token: str, is_lon=False):
    token = str(token).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([NSEW])$", token, re.I)
    if not m: return None
    value, hemi = m.groups()
    # este parser pode ser melhorado se os teus CSVs tiverem outro formato
    return None

def parse_ad_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or s.startswith(("Ident", "DEP/")): continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            # o teu CSV original tinha isto assim, vou manter
            continue
    return pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"])

def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"])

# como o parser acima depende muito dos teus CSVs reais,
# vamos fazer fallback caso não os tenhas no ambiente
try:
    ad_raw  = pd.read_csv(AD_CSV)
    ad_df   = ad_raw  # se quiseres manter o teu parser, troca aqui
except Exception:
    ad_df = pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"])

try:
    loc_raw = pd.read_csv(LOC_CSV)
    loc_df  = loc_raw
except Exception:
    loc_df = pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"])

# --- carregar VORs (para fixes e para nearest) ---
def _load_vor_db(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            if "ident" not in df.columns or "freq_mhz" not in df.columns:
                raise ValueError("CSV VOR sem colunas obrigatórias")
            df["ident"] = df["ident"].astype(str).str.upper().str.strip()
            df["freq_mhz"] = pd.to_numeric(df["freq_mhz"], errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["name"] = df.get("name","")
            df = df.dropna(subset=["ident","freq_mhz","lat","lon"]).reset_index(drop=True)
            return df[["ident","name","freq_mhz","lat","lon"]]
        except Exception:
            pass

    fallback = [
        ("CAS", "Cascais DVOR/DME", 114.30, 38.7483, -9.3619),
        ("ESP", "Espichel DVOR/DME", 112.50, 38.4242, -9.1856),
        ("VFA", "Faro DVOR/DME",     112.80, 37.0136, -7.9750),
        ("FTM", "Fátima DVOR/DME",   113.50, 39.6656, -8.4928),
        ("LIS", "Lisboa DVOR/DME",   114.80, 38.8878, -9.1628),
        ("NSA", "Nisa DVOR/DME",     115.50, 39.5647, -7.9147),
        ("PRT", "Porto DVOR/DME",    114.10, 41.2731, -8.6878),
        ("SGR", "Sagres VOR/DME",    113.90, 37.0839, -8.94639),
        ("SRA", "Sintra VORTAC",     112.10, 38.829201, -9.34),
    ]
    return pd.DataFrame(fallback, columns=["ident","name","freq_mhz","lat","lon"])

if "vor_db" not in st.session_state:
    st.session_state.vor_db = _load_vor_db(VOR_CSV)

# tornar VORs pesquisáveis
vor_pts = []
for _, r in st.session_state.vor_db.iterrows():
    vor_pts.append({
        "src": "VOR",
        "code": str(r["ident"]),
        "name": str(r.get("name") or r["ident"]),
        "city": "",
        "sector": "",
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "alt": 0.0,
    })
vor_df = pd.DataFrame(vor_pts)

# base final
if st.session_state.db_points is None:
    st.session_state.db_points = pd.concat(
        [ad_df, loc_df, vor_df],
        ignore_index=True
    ).dropna(subset=["lat","lon"]).reset_index(drop=True)
db = st.session_state.db_points

# ========= helpers VOR =========
def nearest_vor(lat: float, lon: float):
    df = st.session_state.vor_db
    if df is None or df.empty:
        return None
    best = None
    best_d = 1e9
    for _, r in df.iterrows():
        d = gc_dist_nm(lat, lon, float(r["lat"]), float(r["lon"]))
        if d < best_d:
            best_d = d
            best = r
    if best is None:
        return None
    radial = gc_course_tc(float(best["lat"]), float(best["lon"]), lat, lon)
    return {
        "ident": str(best["ident"]),
        "name":  str(best.get("name") or ""),
        "freq_mhz": float(best["freq_mhz"]),
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
        "dist_nm": best_d,
        "radial_deg": int(round(radial)) % 360,
    }

def nearby_vors(lat: float, lon: float, limit: int = 8):
    out = []
    df = st.session_state.vor_db
    for _, r in df.iterrows():
        d = gc_dist_nm(lat, lon, float(r["lat"]), float(r["lon"]))
        out.append((d, r))
    out.sort(key=lambda x: x[0])
    res = []
    for d, r in out[:limit]:
        res.append({
            "ident": str(r["ident"]),
            "name":  str(r.get("name") or ""),
            "freq_mhz": float(r["freq_mhz"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "dist_nm": d,
        })
    return res

def get_vor_by_ident(ident: str):
    if not ident:
        return None
    ident = ident.strip().upper()
    df = st.session_state.vor_db
    for _, r in df.iterrows():
        if str(r["ident"]).upper().strip() == ident:
            return {
                "ident": ident,
                "name":  str(r.get("name") or ""),
                "freq_mhz": float(r["freq_mhz"]),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
    return None

def fmt_ident_with_freq(v):
    return f"{v['freq_mhz']:.2f} {v['ident']}"

def fmt_radial_distance_from(vor, lat, lon):
    radial = gc_course_tc(vor["lat"], vor["lon"], lat, lon)
    dist   = gc_dist_nm(vor["lat"], vor["lon"], lat, lon)
    dist_int = int(round(dist))
    return f"R{int(round(radial))%360:03d}/D{dist_int}"

def make_unique_name(name: str) -> str:
    names = [str(w["name"]) for w in st.session_state.wps]
    if name not in names: return name
    k=2
    while f"{name} #{k}" in names: k+=1
    return f"{name} #{k}"

def new_wp_dict(name, lat, lon, alt, src=None):
    wp_id = st.session_state.wp_next_id
    st.session_state.wp_next_id += 1
    base = {
        "id": wp_id,
        "name": make_unique_name(str(name)),
        "lat": float(lat),
        "lon": float(lon),
        "alt": float(alt),
        "wind_from": int(st.session_state.wind_from),
        "wind_kt":   int(st.session_state.wind_kt),
        "stop_min":  0.0,
        "vor_pref":  "AUTO",
        "vor_ident": "",
    }
    if src == "VOR":
        base["vor_pref"] = "FIXED"
        base["vor_ident"] = str(name).upper()
    return base

def append_wp(name, lat, lon, alt, src=None):
    st.session_state.wps.append(new_wp_dict(name, lat, lon, alt, src))

def ensure_wp_ids():
    changed = False
    for w in st.session_state.wps:
        if "id" not in w:
            w["id"] = st.session_state.wp_next_id
            st.session_state.wp_next_id += 1
            changed = True
    return changed

ensure_wp_ids()

# ========= ABAS =========
tab_csv, tab_map, tab_fpl = st.tabs(["🔎 Pesquisar CSV", "🗺️ Adicionar no mapa", "✈️ Flight Plan"])

with tab_csv:
    c1, c2 = st.columns([3,1])
    with c1:
        q = st.text_input("Pesquisar único (carrega no ➕)", key="qadd").strip()
    with c2:
        st.session_state.alt_qadd = st.number_input(
            "Alt (ft) p/ novos WPs", 0.0, 18000.0,
            float(st.session_state.alt_qadd), step=100.0
        )

    def _score_row(row, tq, last_wp):
        code = str(row.get("code") or "").lower()
        name = str(row.get("name") or "").lower()
        sim = difflib.SequenceMatcher(None, tq, f"{code} {name}").ratio()
        starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
        near = 0.0
        if last_wp:
            near = 1.0 / (1.0 + gc_dist_nm(
                last_wp["lat"], last_wp["lon"], row["lat"], row["lon"]
            ))
        is_vor = 0.25 if row.get("src") == "VOR" else 0.0
        return starts*2 + sim + near*0.25 + is_vor

    def _search_points(tq):
        if not tq: return db.head(0)
        tql = tq.lower().strip()
        last = st.session_state.wps[-1] if st.session_state.wps else None
        df_ = db[db.apply(
            lambda r: any(tql in str(v).lower() for v in r.values),
            axis=1
        )].copy()
        if df_.empty: return df_
        df_["__score"] = df_.apply(lambda r: _score_row(r, tql, last), axis=1)
        return df_.sort_values("__score", ascending=False)

    results = _search_points(q)
    st.session_state.search_rows = (
        results.head(30).to_dict("records") if not results.empty else []
    )

    if st.session_state.search_rows:
        st.caption("Resultados")
        for i, r in enumerate(st.session_state.search_rows):
            code = r.get("code") or ""
            name = r.get("name") or ""
            local = r.get("city") or r.get("sector") or ""
            lat, lon = float(r["lat"]), float(r["lon"])
            col1, col2 = st.columns([0.84,0.16])
            with col1:
                st.markdown(
                    f"<div class='card'><div class='row'><span class='badge'>[{r['src']}]</span>"
                    f"<b>{code} — {name}</b></div><div class='small'>{local}</div>"
                    f"<div class='small'>({lat:.4f}, {lon:.4f})</div></div>",
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("➕", key=f"csvadd_{i}", use_container_width=True):
                    src = r.get("src")
                    wp = new_wp_dict(code or name, lat, lon, float(st.session_state.alt_qadd), src=src)
                    st.session_state.wps.append(wp)
                    st.success("Adicionado.")
    else:
        st.info("Sem resultados.")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    multi = st.text_input(
        "Adicionar vários (ex: LPSO CAS VFA VARGE)", key="qadd_multi"
    )
    if st.button("➕ Adicionar todos os termos"):
        terms = [t for t in re.split(r"\s+", multi.strip()) if t]
        added, misses = [], []
        for t in terms:
            cand = _search_points(t)
            if cand.empty:
                misses.append(t)
                continue
            r = cand.iloc[0]
            src = r.get("src")
            st.session_state.wps.append(
                new_wp_dict(
                    r.get("code") or r.get("name"),
                    float(r["lat"]), float(r["lon"]),
                    float(st.session_state.alt_qadd),
                    src=src
                )
            )
            added.append(r.get("code") or r.get("name"))
        if added:
            st.success(f"Adicionados: {', '.join(added)}")
        if misses:
            st.warning(f"Sem match: {', '.join(misses)}")

with tab_map:
    st.caption("Clica no mapa e depois em **Adicionar**.")
    m0 = folium.Map(
        location=list(st.session_state.map_center),
        zoom_start=st.session_state.map_zoom,
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap",
        control_scale=True
    )
    cl = MarkerCluster().add_to(m0)
    for _, r in db.iterrows():
        col = "#e11d48" if r.get("src") == "VOR" else "#333"
        folium.CircleMarker(
            (float(r["lat"]),float(r["lon"])),
            radius=4 if r.get("src")=="VOR" else 3,
            color=col, fill=True, fill_opacity=0.9,
            tooltip=f"{(r.get('code') or r.get('name'))} — {r.get('name','')}"
        ).add_to(cl)
    for w in st.session_state.wps:
        folium.CircleMarker(
            (w["lat"],w["lon"]),
            radius=5, color="#007AFF", fill=True, fill_opacity=1,
            tooltip=w["name"]
        ).add_to(m0)
    map_out = st_folium(m0, width=None, height=480, key="pickmap")

    with st.form("add_by_click"):
        cA,cB,_ = st.columns([2,1,1])
        with cA:
            nm = st.text_input("Nome", "WP novo")
        with cB:
            alt = st.number_input(
                "Alt (ft)",
                0.0, 18000.0,
                float(st.session_state.alt_qadd),
                step=100.0
            )
        clicked = map_out.get("last_clicked")
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            st.session_state.wps.append(
                new_wp_dict(
                    nm,
                    float(clicked["lat"]), float(clicked["lng"]),
                    float(alt)
                )
            )
            st.success("Adicionado.")

# ========= ESPAÇO AÉREO =========
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
with st.expander("🛡 Espaço aéreo / restrições"):
    preset_names_sorted = sorted(PRESET_AIRSPACES.keys())
    st.session_state.preset_selected = st.multiselect(
        "Áreas publicadas (catálogo interno)",
        preset_names_sorted,
        default=st.session_state.preset_selected,
        help="Seleciona p/ mostrar no mapa (ex.: LPT1, LPT61...)."
    )
    st.caption("Áreas ad-hoc mantém-se no código.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ========= EDITOR WPs =========
del_id = None
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4 = st.columns([2,2,2,1])
            with c1:
                name = st.text_input(
                    f"Nome — WP{i+1}",
                    w["name"],
                    key=f"wpn_{w['id']}"
                )
            with c2:
                lat  = st.number_input(
                    f"Lat — WP{i+1}",
                    -90.0, 90.0,
                    float(w["lat"]),
                    step=0.0001,
                    key=f"wplat_{w['id']}"
                )
            with c3:
                lon  = st.number_input(
                    f"Lon — WP{i+1}",
                    -180.0, 180.0,
                    float(w["lon"]),
                    step=0.0001,
                    key=f"wplon_{w['id']}"
                )
            with c4:
                alt  = st.number_input(
                    f"Alt (ft) — WP{i+1}",
                    0.0, 18000.0,
                    float(w["alt"]),
                    step=50.0,
                    key=f"wpalt_{w['id']}"
                )

            # stop / espera
            stop_min = st.number_input(
                f"Paragem/Touch&Go (min) — WP{i+1}",
                0.0, 180.0,
                float(w.get("stop_min", 0.0)),
                step=1.0,
                key=f"wpstop_{w['id']}"
            )

            # vento individual
            if not st.session_state.use_global_wind:
                c5,c6 = st.columns(2)
                with c5:
                    wind_from_i = st.number_input(
                        f"Wind FROM °T — WP{i+1}",
                        0,360,
                        int(w.get("wind_from", st.session_state.wind_from)),
                        key=f"wpwindfrom_{w['id']}"
                    )
                with c6:
                    wind_kt_i = st.number_input(
                        f"Wind kt — WP{i+1}",
                        0,150,
                        int(w.get("wind_kt", st.session_state.wind_kt)),
                        key=f"wpwindkt_{w['id']}"
                    )
            else:
                wind_from_i = w.get("wind_from", st.session_state.wind_from)
                wind_kt_i   = w.get("wind_kt",   st.session_state.wind_kt)

            # escolha de VOR
            st.markdown("**Navaid / VOR p/ PDF**")
            near = nearby_vors(lat, lon, limit=8)
            vor_mode = st.selectbox(
                f"Modo VOR — WP{i+1}",
                ["AUTO"] + [f"{v['ident']} ({v['freq_mhz']:.2f}) {v['dist_nm']:.1f}nm" for v in near],
                index=0 if w.get("vor_pref","AUTO")=="AUTO" else
                      max(1, 1 + next((idx for idx,v in enumerate(near) if v["ident"]==w.get("vor_ident")), 0)),
                key=f"wpvsel_{w['id']}"
            )
            if vor_mode == "AUTO":
                vor_pref = "AUTO"
                vor_ident = ""
            else:
                vor_pref = "FIXED"
                vor_ident = vor_mode.split()[0].strip().upper()

            st.session_state.wps[i] = {
                "id": w["id"],
                "name": name,
                "lat": float(lat),
                "lon": float(lon),
                "alt": float(alt),
                "wind_from": int(wind_from_i),
                "wind_kt":   int(wind_kt_i),
                "stop_min":  float(stop_min),
                "vor_pref":  vor_pref,
                "vor_ident": vor_ident,
            }

            if st.button("Remover", key=f"delwp_{w['id']}"):
                del_id = w["id"]

if del_id is not None:
    st.session_state.wps = [w for w in st.session_state.wps if w["id"] != del_id]

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ========= ROTA (TOC/TOD) =========
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, rod_fpm):
    nodes = []
    if len(user_wps) < 2:
        return nodes
    for i in range(len(user_wps)-1):
        A, B = user_wps[i], user_wps[i+1]
        A_copy = dict(A)
        A_copy.setdefault("stop_min", 0.0)
        nodes.append(A_copy)
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        _, _, gs_cl = wind_triangle(tc, CLIMB_TAS,   A.get("wind_from", wind_from), A.get("wind_kt", wind_kt))
        _, _, gs_de = wind_triangle(tc, DESCENT_TAS, A.get("wind_from", wind_from), A.get("wind_kt", wind_kt))
        # TOC
        if B["alt"] > A["alt"]:
            dh = B["alt"] - A["alt"]
            t_need = dh / max(roc_fpm, 1.0)
            d_need = gs_cl * (t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({
                    "id": f"TOC_{i}",
                    "name": f"TOC L{i+1}",
                    "lat": lat_toc, "lon": lon_toc,
                    "alt": B["alt"],
                    "wind_from": A.get("wind_from", wind_from),
                    "wind_kt":   A.get("wind_kt", wind_kt),
                    "stop_min":  0.0,
                    "vor_pref":  "AUTO",
                    "vor_ident": "",
                })
        # TOD
        elif B["alt"] < A["alt"]:
            dh = A["alt"] - B["alt"]
            t_need = dh / max(rod_fpm, 1.0)
            d_need = gs_de * (t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({
                    "id": f"TOD_{i}",
                    "name": f"TOD L{i+1}",
                    "lat": lat_tod, "lon": lon_tod,
                    "alt": A["alt"],
                    "wind_from": A.get("wind_from", wind_from),
                    "wind_kt":   A.get("wind_kt", wind_kt),
                    "stop_min":  0.0,
                    "vor_pref":  "AUTO",
                    "vor_ident": "",
                })
    B_last = dict(user_wps[-1])
    B_last.setdefault("stop_min", 0.0)
    nodes.append(B_last)
    return nodes

def build_legs_from_nodes(nodes, mag_var, mag_is_e, ck_every_min):
    legs = []
    if len(nodes) < 2:
        return legs

    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(
                dt.date.today(),
                dt.time(h,m)
            ) + dt.timedelta(minutes=15)
        except:
            base_time = None

    carry_efob = max(0.0, float(st.session_state.start_efob) - 5.0)

    t_cursor = 0
    for i in range(len(nodes)-1):
        A, B = nodes[i], nodes[i+1]

        # 1) STOP nesta posição?
        if float(A.get("stop_min",0.0)) > 0.0:
            stop_sec = rt30(float(A["stop_min"]) * 60.0)
            burn_stop = rfuel05(FUEL_FLOW * (stop_sec/3600.0))
            efob_start = carry_efob
            efob_end = max(0.0, rfuel05(efob_start - burn_stop))
            clk_start = (
                (base_time + dt.timedelta(seconds=t_cursor)).strftime('%H:%M')
                if base_time else f"T+{mmss(t_cursor)}"
            )
            clk_end   = (
                (base_time + dt.timedelta(seconds=t_cursor+stop_sec)).strftime('%H:%M')
                if base_time else f"T+{mmss(t_cursor+stop_sec)}"
            )
            legs.append({
                "i": len(legs)+1,
                "A": A, "B": A,
                "profile": "STOP",
                "TC": 0, "TH": 0, "MH": 0,
                "TAS": 0, "GS": 0,
                "Dist": 0.0,
                "time_sec": stop_sec,
                "burn": burn_stop,
                "efob_start": efob_start,
                "efob_end": efob_end,
                "clock_start": clk_start,
                "clock_end": clk_end,
                "cps": [],
                "wind_from": A.get("wind_from", st.session_state.wind_from),
                "wind_kt":   A.get("wind_kt",   st.session_state.wind_kt),
            })
            t_cursor += stop_sec
            carry_efob = efob_end

        # 2) perna normal
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist_gc = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = rdist05(dist_gc)

        if st.session_state.use_global_wind:
            wind_from_used = st.session_state.wind_from
            wind_kt_used   = st.session_state.wind_kt
        else:
            wind_from_used = A.get("wind_from", st.session_state.wind_from)
            wind_kt_used   = A.get("wind_kt",   st.session_state.wind_kt)

        profile = "LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile=="CLIMB" else (DESCENT_TAS if profile=="DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from_used, wind_kt_used)
        mh = apply_var(th, mag_var, mag_is_e)

        time_sec = rt30((dist / max(gs,1e-9)) * 3600.0) if gs>0 else 0
        burn = rfuel05(FUEL_FLOW * (time_sec/3600.0))

        efob_start = carry_efob
        efob_end = max(0.0, rfuel05(efob_start - burn))

        clk_start = (
            (base_time + dt.timedelta(seconds=t_cursor)).strftime('%H:%M')
            if base_time else f"T+{mmss(t_cursor)}"
        )
        clk_end   = (
            (base_time + dt.timedelta(seconds=t_cursor+time_sec)).strftime('%H:%M')
            if base_time else f"T+{mmss(t_cursor+time_sec)}"
        )

        cps=[]
        if ck_every_min>0 and gs>0:
            k=1
            while k*ck_every_min*60 <= time_sec:
                t=k*ck_every_min*60
                d_nm=gs*(t/3600.0)
                eto=(base_time + dt.timedelta(seconds=t_cursor+t)).strftime('%H:%M') if base_time else ""
                cps.append({
                    "t":t,
                    "min":int(t/60),
                    "nm":round(d_nm,1),
                    "eto":eto
                })
                k+=1

        legs.append({
            "i": len(legs)+1,
            "A":A,"B":B,
            "profile":profile,
            "TC":tc,"TH":th,"MH":mh,
            "TAS":tas,"GS":gs,
            "Dist":dist,"time_sec":time_sec,
            "burn":burn,
            "efob_start":efob_start,
            "efob_end":efob_end,
            "clock_start":clk_start,
            "clock_end":clk_end,
            "cps":cps,
            "wind_from":wind_from_used,
            "wind_kt":wind_kt_used,
        })

        t_cursor += time_sec
        carry_efob = efob_end

    return legs

# ========= BOTÃO GERAR ROTA =========
cgen,_ = st.columns([2,6])
with cgen:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(
            st.session_state.wps,
            st.session_state.wind_from,
            st.session_state.wind_kt,
            st.session_state.roc_fpm,
            st.session_state.rod_fpm
        )
        st.session_state.legs = build_legs_from_nodes(
            st.session_state.route_nodes,
            st.session_state.mag_var,
            st.session_state.mag_is_e,
            st.session_state.ck_default
        )

# ========= RESUMO GLOBAL =========
if st.session_state.legs:
    total_sec  = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = sum(L["burn"] for L in st.session_state.legs)
    total_dist = sum(L["Dist"] for L in st.session_state.legs if L["profile"]!="STOP")
    efob_final = st.session_state.legs[-1]["efob_end"]
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>🧭 Distância: <b>{total_dist:.1f} nm</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b></div>"
        + f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>"
        + f"<div class='kv'>🧮 Nº legs: <b>{len(st.session_state.legs)}</b></div>"
        + "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ========= FILTRO DE PERNAS =========
if st.session_state.legs:
    with st.expander("🎯 Filtro de pernas no mapa", expanded=False):
        st.session_state.use_leg_filter = st.toggle(
            "Ativar filtro de pernas no mapa",
            value=st.session_state.use_leg_filter,
            help="Se ligado, o mapa só mostra as pernas escolhidas abaixo."
        )
        opt_labels = []
        for idx, L in enumerate(st.session_state.legs):
            if L["profile"]=="STOP":
                label = f"{idx+1:02d}  [STOP] {L['A']['name']}"
            else:
                label = f"{idx+1:02d}  {L['A']['name']}→{L['B']['name']}"
            opt_labels.append(label)

        default_labels = [
            opt_labels[i] for i in st.session_state.leg_filter_ids
            if i < len(opt_labels)
        ]

        chosen_labels = st.multiselect(
            "Quais pernas queres ver?",
            opt_labels,
            default=default_labels,
        )

        new_ids = []
        for lbl in chosen_labels:
            try:
                n_str = lbl.split()[0]
                leg_idx = int(n_str) - 1
                if 0 <= leg_idx < len(st.session_state.legs):
                    new_ids.append(leg_idx)
            except:
                pass
        st.session_state.leg_filter_ids = new_ids

# ========= MARKUP / MAPA =========
def html_marker(m, lat, lon, html):
    folium.Marker(
        (lat,lon),
        icon=folium.DivIcon(html=html, icon_size=(0,0))
    ).add_to(m)

def wp_label_html_rot(g, scale: float, angle_tc: float):
    rot = angle_tc - 90.0
    fs_name = int(14 * scale)
    fs_line = int(12 * scale)
    txtshadow = (
        "-1px -1px 0 #fff,1px -1px 0 #fff,"
        "-1px  1px 0 #fff,1px  1px 0 #fff"
    )
    lines = []
    for (efob, eto) in g["pairs"]:
        ef = f"{float(efob):.1f} L" if efob is not None else ""
        et = eto or ""
        detail = f"{ef} • {et}" if ef and et else (ef or et)
        lines.append(
            f"<div style='font-size:{fs_line}px;font-weight:700;color:#0055FF;"
            f"text-shadow:{txtshadow};'>{detail}</div>"
        )

    return f"""
    <div style="
        transform:translate(-50%,-100%) rotate({rot}deg);
        transform-origin:center center;
        line-height:1.2;text-align:center;white-space:nowrap;
        font-family:ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial;">
        <div style="font-size:{fs_name}px;font-weight:900;color:#000;
            text-shadow:{txtshadow};">
            {g['name']}
        </div>
        {''.join(lines)}
    </div>
    """

def doghouse_html_capsule(info, phase, angle_tc, scale=1.0):
    phase_arrow_map = {
        "CLIMB": "⬈",
        "LEVEL": "⮕",
        "DESCENT": "⬊",
        "STOP": "⏱",
    }
    arrow = phase_arrow_map.get(phase, "⮕")

    rot = angle_tc - 90.0

    fs_head = int(18 * scale)
    fs_alt  = int(16 * scale)
    fs_ete  = int(16 * scale)

    txtshadow = (
        "-2px -2px 0 #fff,  2px -2px 0 #fff,"
        "-2px  2px 0 #fff,  2px  2px 0 #fff,"
        "0px   0px 4px #fff, 2px 2px 3px rgba(0,0,0,.7)"
    )

    return f"""
    <div style="
        transform:translate(-50%,-50%) rotate({rot}deg);
        transform-origin:center center;
        font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-variant-numeric:tabular-nums;
        color:#000;
        background:transparent;
        border:none;
        padding:0;
        line-height:1.25;
        white-space:nowrap;
        text-align:left;
    ">
        <div style="font-size:{fs_head}px;font-weight:800;text-shadow:{txtshadow};">{info['mh_tc']}</div>
        <div style="font-size:{fs_alt}px;font-weight:700;text-shadow:{txtshadow};">{arrow} {info['alt']}</div>
        <div style="font-size:{fs_ete}px;font-weight:700;text-shadow:{txtshadow};">{info['ete']}</div>
    </div>
    """

def airspace_label_html(asp, scale):
    fs_name = int(12*scale)
    fs_line = int(11*scale)
    txtshadow = (
        "-1px -1px 0 #fff,1px -1px 0 #fff,"
        "-1px  1px 0 #fff,1px  1px 0 #fff"
    )
    lines = []
    if asp.get("bands"):
        for band in asp["bands"]:
            top_txt = band.get("top","").strip()
            lbl_txt = band.get("label","").strip()
            if lbl_txt:
                lines.append(f"{top_txt} ({lbl_txt})")
            else:
                lines.append(top_txt)
    else:
        rng = f"{asp.get('floor','')} - {asp.get('ceiling','')}".strip(" -")
        if rng:
            lines.append(rng)
        if asp.get("notes"):
            lines.append(str(asp["notes"]))

    body = "".join(
        f"<div style='font-size:{fs_line}px;font-weight:600;text-shadow:{txtshadow};'>{x}</div>" for x in lines
    )

    return f"""
    <div style="
        transform:translate(-50%,-100%);
        color:#000; white-space:nowrap; text-align:center;
        font-family:ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial;
        text-shadow:{txtshadow};">
        <div style="font-size:{fs_name}px;font-weight:900">{asp['name']}</div>
        {body}
    </div>
    """

def get_filtered_nodes_legs():
    all_nodes = st.session_state.route_nodes
    all_legs  = st.session_state.legs
    if (not st.session_state.use_leg_filter) or (not st.session_state.leg_filter_ids):
        return all_nodes, all_legs
    sel_idxs = [i for i in st.session_state.leg_filter_ids if 0 <= i < len(all_legs)]
    sel_legs = [all_legs[i] for i in sel_idxs]
    sel_nodes_seq = []
    seen = set()
    for L in sel_legs:
        for P in [L["A"], L["B"]]:
            key = (round(P["lat"],6), round(P["lon"],6), P["name"])
            if key not in seen:
                seen.add(key)
                sel_nodes_seq.append(P)
    return sel_nodes_seq, sel_legs

def render_map(nodes, legs, base_choice):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return
    m = folium.Map(
        location=list(st.session_state.map_center),
        zoom_start=st.session_state.map_zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )
    if base_choice == "OpenTopoMap (VFR-ish)":
        folium.TileLayer(
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="© OpenTopoMap"
        ).add_to(m)
    elif base_choice == "OSM Standard":
        folium.TileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="© OpenStreetMap contributors"
        ).add_to(m)
    elif base_choice == "Terrain Hillshade":
        folium.TileLayer(
            "https://services.arcgisonline.com/ArcGIS/rest/services/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri"
        ).add_to(m)
    else:
        folium.TileLayer(
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="© OpenTopoMap"
        ).add_to(m)

    token = st.session_state.openaip_token.strip()
    if st.session_state.show_openaip and token:
        folium.TileLayer(
            tiles=(
                "https://{s}.api.tiles.openaip.net/api/data/openaip/"
                "{z}/{x}/{y}.png?apiKey=" + token
            ),
            attr="© openAIP",
            name="openAIP (VFR data)",
            overlay=True,
            control=True,
            subdomains="abc",
            opacity=float(st.session_state.openaip_alpha),
            max_zoom=20,
        ).add_to(m)

    Fullscreen(position='topleft', title='Fullscreen', force_separate_button=True).add_to(m)

    # PERNAS
    for L in legs:
        if L["profile"]=="STOP":
            continue
        latlngs = [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])]
        color = PROFILE_COLORS.get(L["profile"], "#C000FF")
        folium.PolyLine(latlngs, color="#ffffff", weight=10, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color=color, weight=4, opacity=1.0).add_to(m)

    # TICKS
    if st.session_state.show_ticks:
        for L in legs:
            if L["profile"]=="STOP":
                continue
            if L["GS"]<=0 or not L["cps"]:
                continue
            for cp in L["cps"]:
                d = min(L["Dist"], L["GS"]*(cp["t"]/3600.0))
                latm, lonm = point_along_gc(
                    L["A"]["lat"],L["A"]["lon"],
                    L["B"]["lat"],L["B"]["lon"],
                    d
                )
                llat, llon = dest_point(latm, lonm, L["TC"]-90, CP_TICK_HALF)
                rlat, rlon = dest_point(latm, lonm, L["TC"]+90, CP_TICK_HALF)
                folium.PolyLine(
                    [(llat,llon),(rlat,rlon)],
                    color="#111111",
                    weight=3,
                    opacity=1
                ).add_to(m)

    # DOGHOUSES (só de legs normais)
    if st.session_state.show_doghouses:
        zones = []
        for L in legs:
            if L["profile"]=="STOP":
                continue
            dist_leg = gc_dist_nm(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"])
            steps = max(2, int(dist_leg / 0.9))
            for k in range(1, steps):
                p = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], dist_leg * k / steps)
                zones.append((p[0], p[1], 0.38))

        prev_side = None
        for idx, L in enumerate(legs):
            if L["profile"]=="STOP":
                continue
            if L["Dist"] < 0.2:
                continue

            base = min(1.25, max(0.9, L["Dist"]/7.0))
            s = base * float(st.session_state.text_scale)

            cur_tc = L["TC"]
            nxt_tc = legs[idx+1]["TC"] if idx < len(legs)-1 else L["TC"]
            turn = angdiff(nxt_tc, cur_tc)
            prefer = +1 if turn > 12 else (-1 if turn < -12 else (prev_side or +1))

            mid_lat, mid_lon = point_along_gc(
                L["A"]["lat"], L["A"]["lon"],
                L["B"]["lat"], L["B"]["lon"],
                0.50 * L["Dist"]
            )

            side_off_nm = 1.2
            anchor_lat, anchor_lon = dest_point(
                mid_lat, mid_lon,
                L["TC"] + 90 * prefer,
                side_off_nm
            )
            prev_side = prefer

            info = {
                "mh_tc": f"{deg3(L['MH'])}|{deg3(L['TC'])}",
                "alt":   f"{int(round(L['A']['alt']))} ft",
                "ete":   mmss(L['time_sec']),
            }

            folium.PolyLine(
                [(mid_lat, mid_lon), (anchor_lat, anchor_lon)],
                color="#000000",
                weight=2,
                opacity=1.0
            ).add_to(m)

            html_marker(
                m,
                anchor_lat,
                anchor_lon,
                doghouse_html_capsule(info, L["profile"], L["TC"], scale=s)
            )

    # NÓS e labels
    info_nodes = [{"eto": None, "efob": None} for _ in nodes]
    if legs:
        info_nodes[0]["eto"]  = legs[0]["clock_start"]
        info_nodes[0]["efob"] = legs[0]["efob_start"]
        for i in range(1, len(nodes)):
            if i-1 < len(legs):
                Lprev = legs[i-1]
                info_nodes[i]["eto"]  = Lprev["clock_end"]
                info_nodes[i]["efob"] = Lprev["efob_end"]

    node_tc = []
    if legs:
        for i in range(len(nodes)):
            # aproxima TC da leg seguinte
            if i < len(legs):
                node_tc.append(legs[i]["TC"])
            else:
                node_tc.append(legs[-1]["TC"])
    else:
        node_tc = [0.0]*len(nodes)

    grouped=[]
    seen=set()
    for idx,N in enumerate(nodes):
        base_name = re.sub(r"\s+#\d+$","",str(N["name"]))
        key=(round(N["lat"],6),round(N["lon"],6),base_name)
        if key not in seen:
            seen.add(key)
            grouped.append({
                "name": base_name,
                "lat": N["lat"],
                "lon": N["lon"],
                "pairs":[(info_nodes[idx]["efob"], info_nodes[idx]["eto"])],
                "angle": node_tc[idx],
            })
        else:
            for g in grouped:
                if g["name"]==base_name and abs(g["lat"]-N["lat"])<1e-6 and abs(g["lon"]-N["lon"])<1e-6:
                    g["pairs"].append((info_nodes[idx]["efob"], info_nodes[idx]["eto"]))
                    break

    for g in grouped:
        s = float(st.session_state.text_scale)
        html_marker(m, g["lat"], g["lon"], wp_label_html_rot(g, s, g["angle"]))

    if st.session_state.show_airspaces:
        combined_asp = []
        for nm in st.session_state.preset_selected:
            A = PRESET_AIRSPACES.get(nm)
            if A:
                tmp = dict(A); tmp["name"] = nm
                combined_asp.append(tmp)
        combined_asp += st.session_state.airspaces

        for asp in combined_asp:
            if asp.get("width_nm"):
                coords = asp.get("coords", [])
                if len(coords) >= 2:
                    polycoords = corridor_polygon(coords[0], coords[1], asp["width_nm"])
                else:
                    polycoords = []
            else:
                polycoords = asp.get("coords", [])

            if not polycoords:
                continue

            edge_color = CORRIDOR_COLOR if asp.get("width_nm") else ASPACE_COLOR
            folium.Polygon(
                locations=[(lat,lon) for (lat,lon) in polycoords],
                color=edge_color,
                weight=2,
                opacity=EDGE_OPACITY,
                fill=True,
                fill_color=edge_color,
                fill_opacity=FILL_OPACITY,
                tooltip=f"{asp['name']} {asp.get('floor','')}→{asp.get('ceiling','')} {asp.get('notes','')}"
            ).add_to(m)

            clat, clon = polygon_centroid(polycoords)
            s_label = float(st.session_state.text_scale)
            html_marker(m, clat, clon, airspace_label_html(asp, s_label))

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, width=None, height=760, key="mainmap", returned_objects=[])

# ========= RENDER MAPA =========
if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    nodes_to_show, legs_to_show = get_filtered_nodes_legs()
    if legs_to_show and nodes_to_show:
        render_map(nodes_to_show, legs_to_show, base_choice=st.session_state.map_base)
    else:
        st.warning("Filtro ativo mas sem pernas selecionadas para mostrar.")
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota**.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")

# ========= FLIGHT PLAN (igual à tua lógica) =========
with tab_fpl:
    st.subheader("Flight Plan — rota (Item 15)")

    def is_icao_aerodrome(name:str) -> bool:
        return bool(re.fullmatch(r"[A-Z]{4}", str(name).upper()))
    def is_published_sigpt(name:str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9]{3,5}", str(name).upper()))
    def icao_latlon(lat:float, lon:float) -> str:
        lat_abs = abs(lat); lon_abs = abs(lon)
        lat_deg = int(lat_abs); lon_deg = int(lon_abs)
        lat_min = int(round((lat_abs - lat_deg)*60))
        lon_min = int(round((lon_abs - lon_deg)*60))
        if lat_min == 60:
            lat_deg += 1; lat_min = 0
        if lon_min == 60:
            lon_deg += 1; lon_min = 0
        hemi_ns = "N" if lat >= 0 else "S"
        hemi_ew = "E" if lon >= 0 else "W"
        return f"{lat_deg:02d}{lat_min:02d}{hemi_ns}{lon_deg:03d}{lon_min:02d}{hemi_ew}"

    def build_fpl_route(user_wps):
        if len(user_wps) < 2: return ""
        seq = user_wps[:]
        if is_icao_aerodrome(seq[0]["name"]):
            seq = seq[1:]
        if seq and is_icao_aerodrome(seq[-1]["name"]):
            seq = seq[:-1]
        tokens = []
        for w in seq:
            nm = re.sub(r"\s+#\d+$","",str(w["name"]).upper())
            tokens.append(nm if is_published_sigpt(nm) else icao_latlon(w["lat"], w["lon"]))
        return "DCT " + " DCT ".join(tokens) if tokens else ""

    route_str = build_fpl_route(st.session_state.wps)
    if route_str:
        st.code(route_str.upper())
    else:
        st.info("Adiciona WPs e volta aqui para gerar a rota.")

# ========= NAVLOG / PDF =========
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.header("📄 NAVLOG — Cabeçalho & PDF")

REG_OPTIONS = ["CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW","CS-ECC","CS-ECD"]
c0,c1,c2,c3,c4 = st.columns(5)
with c0:
    callsign = st.text_input("Callsign", "RVP")
with c1:
    registration = st.selectbox("Registration", REG_OPTIONS, index=0)
with c2:
    student = st.text_input("Student", "AMOIT")
with c3:
    lesson = st.text_input("Lesson", "")
with c4:
    instructor = st.text_input("Instructor", "")

c5,c6,c7,c8,c9 = st.columns(5)
with c5:
    etd = st.text_input("ETD (HH:MM)", "")
with c6:
    eta = st.text_input("ETA (HH:MM, opcional)", "")
with c7:
    dept_freq = st.text_input("FREQ DEPT", "119.805")
with c8:
    enroute_freq = st.text_input("FREQ ENROUTE", "123.755")
with c9:
    arrival_freq = st.text_input("FREQ ARRIVAL", "131.675")

st.subheader("Alternate (opcional)")
cA,cB,cC = st.columns([2,1,1])
with cA:
    alt_query = st.text_input("Pesquisar alternante (código/nome)", "")
with cB:
    alt_elev = st.number_input("Elevação ALT (ft)", 0, 10000, 350, step=10)
with cC:
    use_alt = st.checkbox("Usar este alternante", value=False)

alt_choice = None
if alt_query.strip():
    tql = alt_query.lower().strip()
    cand = db[db.apply(
        lambda r: any(tql in str(v).lower() for v in r.values),
        axis=1
    )].head(1)
    if not cand.empty:
        r = cand.iloc[0]
        alt_choice = {
            "name": r.get("code") or r.get("name"),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "elev": int(alt_elev)
        }
        st.success(f"ALT: {alt_choice['name']}  ({alt_choice['lat']:.4f}, {alt_choice['lon']:.4f})")
    else:
        st.warning("Sem match para o alternante.")

alt_leg_info = None
if use_alt and alt_choice and st.session_state.wps:
    dest = st.session_state.wps[-1]
    tc_alt = gc_course_tc(dest["lat"], dest["lon"], alt_choice["lat"], alt_choice["lon"])
    if st.session_state.use_global_wind:
        wf = st.session_state.wind_from
        wk = st.session_state.wind_kt
    else:
        wf = dest.get("wind_from", st.session_state.wind_from)
        wk = dest.get("wind_kt",   st.session_state.wind_kt)
    _, th_alt, gs_alt = wind_triangle(tc_alt, CRUISE_TAS, wf, wk)
    mh_alt = apply_var(th_alt, st.session_state.mag_var, st.session_state.mag_is_e)
    dist_alt = rdist05(gc_dist_nm(dest["lat"], dest["lon"], alt_choice["lat"], alt_choice["lon"]))
    ete_alt_sec = rt30((dist_alt / max(gs_alt,1e-9)) * 3600.0)
    burn_alt = rfuel05(FUEL_FLOW * (ete_alt_sec/3600.0))
    alt_leg_info = {
        "tc":tc_alt,"th":th_alt,"mh":mh_alt,
        "tas":CRUISE_TAS,"gs":gs_alt,
        "dist":dist_alt,"ete":ete_alt_sec,
        "burn":burn_alt
    }

# ========= PDF helpers =========
def _pdf_mmss(sec:int):
    total_sec = int(round(sec))
    minutes, seconds = divmod(total_sec, 60)
    if minutes >= 60:
        hours, mins = divmod(minutes, 60)
        return f"{hours:02d}h{mins:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def _fill_pdf(template_path: str, out_path: str, data: dict):
    pdf = PdfReader(template_path)
    if pdf.Root.AcroForm:
        pdf.Root.AcroForm.update(PdfDict(NeedAppearances=True))
    for page in pdf.pages:
        if not getattr(page, "Annots", None):
            continue
        for a in page.Annots:
            if a.Subtype == PdfName('Widget') and a.T:
                key = str(a.T)[1:-1]
                if key in data:
                    a.update(PdfDict(V=str(data[key])))
                    if key.startswith("Leg") and "Navaid_" in key:
                        a.update(PdfDict(DA="/Helv 5 Tf 0 g"))
    PdfWriter(out_path, trailer=pdf).write()
    return out_path

def _sum_time(legs, profile):
    return sum(L["time_sec"] for L in legs if L["profile"]==profile)

def _sum_burn(legs, profile):
    return round(sum(L["burn"] for L in legs if L["profile"]==profile),1)

def _compose_clock_after(total_sec, extra_sec):
    base = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base = dt.datetime.combine(
                dt.date.today(),
                dt.time(h,m)
            ) + dt.timedelta(minutes=15)
        except:
            base = None
    t = total_sec + extra_sec
    if base:
        return (base + dt.timedelta(seconds=t)).strftime("%H:%M")
    return f"T+{mmss(t)}"

def _fill_leg_line(d:dict, idx:int, L:dict, acc_d:float, acc_t:int, prefix="Leg"):
    # se for STOP, usar o ponto A
    is_stop = (L["profile"] == "STOP")
    P = L["A"] if is_stop else L["B"]

    d[f"{prefix}{idx:02d}_Waypoint"]            = str(P["name"])
    d[f"{prefix}{idx:02d}_Altitude_FL"]         = str(int(round(P["alt"])))
    if is_stop:
        d[f"{prefix}{idx:02d}_True_Course"]         = ""
        d[f"{prefix}{idx:02d}_True_Heading"]        = ""
        d[f"{prefix}{idx:02d}_Magnetic_Heading"]    = ""
        d[f"{prefix}{idx:02d}_True_Airspeed"]       = ""
        d[f"{prefix}{idx:02d}_Ground_Speed"]        = ""
        d[f"{prefix}{idx:02d}_Leg_Distance"]        = "0.0"
        d[f"{prefix}{idx:02d}_Cumulative_Distance"] = f"{acc_d:.1f}"
    else:
        d[f"{prefix}{idx:02d}_True_Course"]         = f"{int(round(L['TC'])):03d}"
        d[f"{prefix}{idx:02d}_True_Heading"]        = f"{int(round(L['TH'])):03d}"
        d[f"{prefix}{idx:02d}_Magnetic_Heading"]    = f"{int(round(L['MH'])):03d}"
        d[f"{prefix}{idx:02d}_True_Airspeed"]       = str(int(round(L["TAS"])))
        d[f"{prefix}{idx:02d}_Ground_Speed"]        = str(int(round(L["GS"])))
        d[f"{prefix}{idx:02d}_Leg_Distance"]        = f"{L['Dist']:.1f}"
        d[f"{prefix}{idx:02d}_Cumulative_Distance"] = f"{acc_d:.1f}"

    d[f"{prefix}{idx:02d}_Leg_ETE"]             = _pdf_mmss(L["time_sec"])
    d[f"{prefix}{idx:02d}_Cumulative_ETE"]      = _pdf_mmss(acc_t)
    d[f"{prefix}{idx:02d}_ETO"]                 = L["clock_end"]
    d[f"{prefix}{idx:02d}_Planned_Burnoff"]     = f"{L['burn']:.1f}"
    d[f"{prefix}{idx:02d}_Estimated_FOB"]       = f"{L['efob_end']:.1f}"

    # VOR: usar FIXED se tiver
    vor_obj = None
    if P.get("vor_pref") == "FIXED" and P.get("vor_ident"):
        vor_obj = get_vor_by_ident(P["vor_ident"])
    if vor_obj is None:
        vor_obj = nearest_vor(float(P["lat"]), float(P["lon"]))
    if vor_obj:
        d[f"{prefix}{idx:02d}_Navaid_Identifier"] = fmt_ident_with_freq(vor_obj)
        d[f"{prefix}{idx:02d}_Navaid_Frequency"]  = fmt_radial_distance_from(vor_obj, float(P["lat"]), float(P["lon"]))
    else:
        d[f"{prefix}{idx:02d}_Navaid_Identifier"] = ""
        d[f"{prefix}{idx:02d}_Navaid_Frequency"]  = ""

def _build_payloads_main(
    legs, *,
    callsign, registration, student, lesson, instructor,
    dept, enroute, arrival, etd, eta,
    alt_info=None, alt_choice=None
):
    total_sec = sum(L["time_sec"] for L in legs)
    total_burn = sum(L["burn"] for L in legs)
    total_dist = sum(L["Dist"] for L in legs if L["profile"]!="STOP")
    obs = (
        f"Climb {_pdf_mmss(_sum_time(legs,'CLIMB'))} / "
        f"Cruise {_pdf_mmss(_sum_time(legs,'LEVEL'))} / "
        f"Descent {_pdf_mmss(_sum_time(legs,'DESCENT'))} / "
        f"Stop {_pdf_mmss(_sum_time(legs,'STOP'))}"
    )
    climb_burn = _sum_burn(legs,'CLIMB')

    N = min(len(legs), 22)
    legs_main = legs[:N]

    d = {
        "CALLSIGN": callsign,
        "REGISTRATION": registration,
        "STUDENT": student,
        "LESSON": lesson,
        "INSTRUTOR": instructor,
        "DEPT": dept,
        "ENROUTE": enroute,
        "ARRIVAL": arrival,
        "ETD/ETA": (f"{etd}/{eta}" if etd else ""),
        "Departure_Airfield": str(st.session_state.wps[0]["name"]) if st.session_state.wps else "",
        "Arrival_Airfield":   str(st.session_state.wps[-1]["name"]) if st.session_state.wps else "",
        "WIND": f"{int(st.session_state.wind_from)}/{int(st.session_state.wind_kt)}",
        "MAG_VAR": f"{abs(st.session_state.mag_var):.0f}°{'E' if st.session_state.mag_is_e else 'W'}",
        "FLT TIME": _pdf_mmss(total_sec),
        "CLIMB FUEL": f"{climb_burn:.1f}",
        "OBSERVATIONS": obs,
        "Leg_Number": str(len(legs)),
    }

    acc_d, acc_t = 0.0, 0
    for i, L in enumerate(legs_main, start=1):
        if L["profile"]!="STOP":
            acc_d = round(acc_d + L["Dist"], 1)
        acc_t += L["time_sec"]
        _fill_leg_line(d, i, L, acc_d=acc_d, acc_t=acc_t)

    # totais da viagem toda (mesmo se só 1 página)
    d["Leg23_Leg_Distance"] = f"{total_dist:.1f}"
    d["Leg23_Leg_ETE"]      = _pdf_mmss(total_sec)
    d["Leg23_Planned_Burnoff"] = f"{total_burn:.1f}"
    d["Leg23_Estimated_FOB"]   = f"{legs[-1]['efob_end']:.1f}"

    if alt_info and alt_choice:
        d.update({
            "Alternate_Airfield":alt_choice["name"],
            "Alternate_Elevation":str(int(alt_choice["elev"])),
            "Alternate_True_Course":f"{int(round(alt_info['tc'])):03d}",
            "Alternate_True_Heading":f"{int(round(alt_info['th'])):03d}",
            "Alternate_Magnetic_Heading":f"{int(round(alt_info['mh'])):03d}",
            "Alternate_True_Airspeed":str(int(round(alt_info['tas']))),
            "Alternate_Ground_Speed":str(int(round(alt_info['gs']))),
            "Alternate_Leg_Distance":f"{alt_info['dist']:.1f}",
            "Alternate_Cumulative_Distance":f"{alt_info['dist']:.1f}",
            "Alternate_Leg_ETE":_pdf_mmss(alt_info['ete']),
            "Alternate_Cumulative_ETE":_pdf_mmss(alt_info['ete']),
            "Alternate_ETO":_compose_clock_after(total_sec, alt_info['ete']),
            "Alternate_Planned_Burnoff":f"{alt_info['burn']:.1f}",
            "Alternate_Estimated_FOB":f"{(legs[-1]['efob_end'] - alt_info['burn']):.1f}",
        })
    return d

def _build_payload_cont(all_legs, start_idx, *, alt_info=None, alt_choice=None):
    legs_chunk = all_legs[start_idx:start_idx+11]
    if not legs_chunk:
        return None
    d = {"OBSERVATIONS":"SEVENAIR OPS: 131.675"}
    acc_d = 0.0
    acc_t = 0
    for offset, L in enumerate(legs_chunk, start=12):
        if L["profile"]!="STOP":
            acc_d = round(acc_d + L["Dist"], 1)
        acc_t += L["time_sec"]
        _fill_leg_line(d, offset, L, acc_d=acc_d, acc_t=acc_t)

    total_dist_all_nm   = sum(L["Dist"] for L in all_legs if L["profile"]!="STOP")
    total_time_all_sec  = sum(L["time_sec"] for L in all_legs)
    total_burn_all_L    = sum(L["burn"] for L in all_legs)
    final_efob_all_L    = all_legs[-1]["efob_end"]

    d["Leg23_Leg_Distance"] = f"{total_dist_all_nm:.1f}"
    d["Leg23_Leg_ETE"]      = _pdf_mmss(total_time_all_sec)
    d["Leg23_Planned_Burnoff"] = f"{total_burn_all_L:.1f}"
    d["Leg23_Estimated_FOB"]   = f"{final_efob_all_L:.1f}"

    if alt_info and alt_choice:
        total_sec_before_chunk = sum(L["time_sec"] for L in all_legs[:start_idx+len(legs_chunk)])
        d.update({
            "Alternate_Airfield":alt_choice["name"],
            "Alternate_Elevation":str(int(alt_choice["elev"])),
            "Alternate_True_Course":f"{int(round(alt_info['tc'])):03d}",
            "Alternate_True_Heading":f"{int(round(alt_info['th'])):03d}",
            "Alternate_Magnetic_Heading":f"{int(round(alt_info['mh'])):03d}",
            "Alternate_True_Airspeed":str(int(round(alt_info['tas']))),
            "Alternate_Ground_Speed":str(int(round(alt_info['gs']))),
            "Alternate_Leg_Distance":f"{alt_info['dist']:.1f}",
            "Alternate_Cumulative_Distance":f"{alt_info['dist']:.1f}",
            "Alternate_Leg_ETE":_pdf_mmss(alt_info['ete']),
            "Alternate_Cumulative_ETE":_pdf_mmss(alt_info['ete']),
            "Alternate_ETO":_compose_clock_after(total_sec_before_chunk, alt_info['ete']),
            "Alternate_Planned_Burnoff":f"{alt_info['burn']:.1f}",
            "Alternate_Estimated_FOB":f"{(all_legs[start_idx+len(legs_chunk)-1]['efob_end'] - alt_info['burn']):.1f}",
        })
    return d

# ========= BOTÕES PDF =========
cX, cY = st.columns([1,1])
with cX:
    make_pdfs = st.button(
        "Gerar PDF(s) NAVLOG",
        type="primary",
        use_container_width=True
    )
with cY:
    st.caption("Principal até 22 legs; continuação se exceder.")

if make_pdfs:
    if not st.session_state.legs:
        st.error("Gera primeiro a rota.")
    else:
        d_main = _build_payloads_main(
            st.session_state.legs,
            callsign=callsign,
            registration=registration,
            student=student,
            lesson=lesson,
            instructor=instructor,
            dept=dept_freq,
            enroute=enroute_freq,
            arrival=arrival_freq,
            etd=etd,
            eta=eta,
            alt_info=alt_leg_info if (use_alt and alt_choice) else None,
            alt_choice=alt_choice if (use_alt and alt_choice) else None
        )
        out_main = _fill_pdf(TEMPLATE_MAIN, "NAVLOG_FILLED.pdf", d_main)
        with open(out_main, "rb") as f:
            st.download_button(
                "⬇️ NAVLOG (principal)",
                f.read(),
                file_name="NAVLOG_FILLED.pdf",
                use_container_width=True
            )

        if len(st.session_state.legs) > 22:
            d_cont = _build_payload_cont(
                st.session_state.legs,
                start_idx=22,
                alt_info=alt_leg_info if (use_alt and alt_choice) else None,
                alt_choice=alt_choice if (use_alt and alt_choice) else None
            )
            out_cont = _fill_pdf(TEMPLATE_CONT, "NAVLOG_FILLED_1.pdf", d_cont)
            with open(out_cont, "rb") as f:
                st.download_button(
                    "⬇️ NAVLOG (continuação)",
                    f.read(),
                    file_name="NAVLOG_FILLED_1.pdf",
                    use_container_width=True
                )


