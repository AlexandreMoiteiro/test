# app.py — NAVLOG — rev25
# - ROD (ft/min) definido pelo utilizador (substitui ângulo) para TOD/descidas
# - Exporta formulários PDF preenchidos (NAVLOG_FORM.pdf + NAVLOG_FORM_1.pdf)
# - No máximo 2 PDFs: navlog_main_filled.pdf (capa) e navlog_cont_filled.pdf (continuação, se precisar)
# - Mantém o resto da app (mapa, dog houses, labels, FPL, etc.) como nas revisões anteriores

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib, io, os
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MarkerCluster
from math import degrees

# ======== TEMPLATES PDF ========
TEMPLATE_MAIN = "NAVLOG_FORM.pdf"     # Capa (campos Leg01..)
TEMPLATE_CONT = "NAVLOG_FORM_1.pdf"   # Continuação (campos Leg12..)
# Notas: os ficheiros devem estar ao lado do app.py

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0   # kt
FUEL_FLOW = 20.0                                        # L/h
EARTH_NM  = 3440.065

PROFILE_COLORS = {
    "CLIMB":   "#FF7A00",  # laranja
    "LEVEL":   "#C000FF",  # púrpura
    "DESCENT": "#00B386",  # verde-água
}

CP_TICK_HALF = 0.38

# ======== PAGE / STYLE ========
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin:8px 0;background:#fff}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:var(--line);margin:10px 0}
.leaflet-control-zoom a{font-weight:800}
.small{font-size:12px;color:#555}
.row{display:flex;gap:8px;align-items:center}
.badge{font-weight:700;border:1px solid #111;border-radius:8px;padding:2px 6px;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# ======== HELPERS NUMÉRICOS ========
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
hhmmss = lambda t: f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
wrap360 = lambda x: (x % 360 + 360) % 360
def angdiff(a, b): return (a - b + 180) % 360 - 180
def deg3(v): return f"{int(round(v))%360:03d}°"

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * math.sin(d)
    s = max(-1, min(1, cross / max(tas,1e-9)))
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

def _nm_dist(a,b): return gc_dist_nm(a[0],a[1],b[0],b[1])

# ======== STATE ========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("rod_fpm", 600)   # ROD controlado pelo utilizador
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)

ens("wps", []); ens("legs", []); ens("route_nodes", [])

ens("map_base", "OpenTopoMap (VFR-ish)")
ens("maptiler_key", "")
ens("text_scale", 1.0)          # default 1.0 (slider 0.5..1.5)
ens("show_ticks", True)
ens("show_doghouses", True)

ens("map_center", (39.7, -8.1))
ens("map_zoom", 8)

ens("db_points", None); ens("qadd", ""); ens("alt_qadd", 3000.0)
ens("search_rows", []); ens("last_q", "")

# ======== PARÂMETROS GLOBAIS ========
with st.form("globals"):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from))
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt))
    with c2:
        st.session_state.mag_var   = st.number_input("Variação magnética (±°)", -30.0, 30.0, float(st.session_state.mag_var))
        st.session_state.mag_is_e  = st.toggle("Var. é EAST (subtrai)", value=st.session_state.mag_is_e)
    with c3:
        st.session_state.roc_fpm   = st.number_input("ROC global (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10)
        st.session_state.rod_fpm   = st.number_input("ROD global (ft/min)", 200, 1500, int(st.session_state.rod_fpm), step=10)
    with c4:
        st.session_state.start_efob= st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default))
    b1,b2,b3 = st.columns([1.6,1.6,1])
    with b1:
        bases = ["OpenTopoMap (VFR-ish)","EOX Sentinel-2 (satélite)","Esri World Imagery (satélite + labels)","Esri World TopoMap (topo)","OSM Standard","MapTiler Satellite Hybrid (requer key)"]
        st.session_state.map_base = st.selectbox("Base do mapa", bases, index=bases.index(st.session_state.map_base) if st.session_state.map_base in bases else 0)
    with b2:
        st.session_state.show_doghouses = st.toggle("Mostrar dog houses", value=st.session_state.show_doghouses)
        st.session_state.show_ticks     = st.toggle("Mostrar riscas CP", value=st.session_state.show_ticks)
    with b3:
        st.session_state.text_scale  = st.slider("Tamanho do texto", 0.5, 1.5, float(st.session_state.text_scale), 0.05)
    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== CSVs locais ========
AD_CSV  = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"

def dms_to_dd(token: str, is_lon=False):
    token = str(token).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([NSEW])$", token, re.I)
    if not m: return None
    value, hemi = m.groups()
    if "." in value:
        if is_lon: deg = int(value[0:3]); minutes = int(value[3:5]); seconds = float(value[5:])
        else:      deg = int(value[0:2]); minutes = int(value[2:4]); seconds = float(value[4:])
    else:
        if is_lon: deg = int(value[0:3]); minutes = int(value[3:5]); seconds = int(value[5:])
        else:      deg = int(value[0:2]); minutes = int(value[2:4]); seconds = int(value[4:])
    dd = deg + minutes/60 + seconds/3600
    if hemi.upper() in ["S","W"]: dd = -dd
    return dd

def parse_ad_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or s.startswith(("Ident", "DEP/")): continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok = coord_toks[-2]; lon_tok = coord_toks[-1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            ident = tokens[0] if re.match(r"^[A-Z0-9]{4,}$", tokens[0]) else None
            try:    name = " ".join(tokens[1:tokens.index(coord_toks[0])]).strip()
            except: name = " ".join(tokens[1:]).strip()
            try:    lon_idx = tokens.index(lon_tok); city = " ".join(tokens[lon_idx+1:]) or None
            except: city = None
            rows.append({"src":"AD","code":ident or name, "name":name, "city":city,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or "Total de registos" in s: continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok, lon_tok = coord_toks[0], coord_toks[1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            try: lon_idx = tokens.index(lon_tok)
            except ValueError: continue
            code = tokens[lon_idx+1] if lon_idx+1 < len(tokens) else None
            sector = " ".join(tokens[lon_idx+2:]) if lon_idx+2 < len(tokens) else None
            name = " ".join(tokens[:tokens.index(lat_tok)]).strip()
            rows.append({"src":"LOC","code":code or name, "name":name, "sector":sector,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

try:
    ad_raw  = pd.read_csv(AD_CSV);  ad_df  = parse_ad_df(ad_raw)
    loc_raw = pd.read_csv(LOC_CSV); loc_df = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

if st.session_state.db_points is None:
    st.session_state.db_points = pd.concat([ad_df, loc_df]).dropna(subset=["lat","lon"]).reset_index(drop=True)
db = st.session_state.db_points

# === util WPs repetidos ===
def make_unique_name(name: str) -> str:
    names = [str(w["name"]) for w in st.session_state.wps]
    if name not in names: return name
    k=2
    while f"{name} #{k}" in names: k+=1
    return f"{name} #{k}"

def append_wp(name, lat, lon, alt):
    st.session_state.wps.append({"name": make_unique_name(str(name)), "lat": float(lat), "lon": float(lon), "alt": float(alt)})

# ======== ABAS — ADICIONAR WPs ========
tab_csv, tab_map, tab_fpl, tab_pdf = st.tabs(["🔎 Pesquisar CSV", "🗺️ Adicionar no mapa", "✈️ Flight Plan", "🧾 Formulários (PDF)"])

# ---- Pesquisa CSV + multi-termos ----
with tab_csv:
    c1, c2 = st.columns([3,1])
    with c1:
        q = st.text_input("Pesquisar único (carrega no ➕)", key="qadd",
                          placeholder="Ex: LPPT, ALPAL, ÉVORA, NISA…").strip()
    with c2:
        st.session_state.alt_qadd = st.number_input("Alt (ft) p/ novos WPs", 0.0, 18000.0,
                                                    float(st.session_state.alt_qadd), step=100.0)

    def _score_row(row, tq, last_wp):
        code = str(row.get("code") or "").lower()
        name = str(row.get("name") or "").lower()
        sim = difflib.SequenceMatcher(None, tq, f"{code} {name}").ratio()
        starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
        near = 0.0
        if last_wp:
            near = 1.0 / (1.0 + gc_dist_nm(last_wp["lat"], last_wp["lon"], row["lat"], row["lon"]))
        return starts*2 + sim + near*0.25

    def _search_points(tq):
        if not tq: return db.head(0)
        tql = tq.lower().strip()
        last = st.session_state.wps[-1] if st.session_state.wps else None
        df = db[db.apply(lambda r: any(tql in str(v).lower() for v in r.values), axis=1)].copy()
        if df.empty: return df
        df["__score"] = df.apply(lambda r: _score_row(r, tql, last), axis=1)
        return df.sort_values("__score", ascending=False)

    results = _search_points(q)
    st.session_state.search_rows = results.head(30).to_dict("records") if not results.empty else []

    if st.session_state.search_rows:
        st.caption("Resultados")
        for i, r in enumerate(st.session_state.search_rows):
            code = r.get("code") or ""
            name = r.get("name") or ""
            local = r.get("city") or r.get("sector") or ""
            lat, lon = float(r["lat"]), float(r["lon"])
            label = f"{code} — {name}"
            col1, col2 = st.columns([0.84,0.16])
            with col1:
                st.markdown(
                    f"<div class='card'><div class='row'><span class='badge'>[{r['src']}]</span>"
                    f"<b>{label}</b></div><div class='small'>{local}</div>"
                    f"<div class='small'>({lat:.4f}, {lon:.4f})</div></div>",
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("➕", key=f"csvadd_{i}", use_container_width=True):
                    append_wp(code or name, lat, lon, float(st.session_state.alt_qadd))
                    st.success("Adicionado.")
    else:
        st.info("Sem resultados.")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    multi = st.text_input("Adicionar vários (separa por espaços: ex. `LPSO VACOR VARGE`)", key="qadd_multi")
    if st.button("➕ Adicionar todos os termos"):
        terms = [t for t in re.split(r"\s+", multi.strip()) if t]
        added, misses = [], []
        for t in terms:
            cand = _search_points(t)
            if cand.empty:
                misses.append(t); continue
            r = cand.iloc[0]
            append_wp(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_qadd))
            added.append(r.get("code") or r.get("name"))
        if added: st.success(f"Adicionados: {', '.join(added)}")
        if misses: st.warning(f"Sem match: {', '.join(misses)}")

# ---- Adicionar no mapa (inclui TODOS os pontos dos CSVs) ----
with tab_map:
    st.caption("Clica no mapa e depois em **Adicionar**. Todos os WPs/Aeródromos do CSV estão visíveis.")
    m0 = folium.Map(location=list(st.session_state.map_center), zoom_start=st.session_state.map_zoom,
                    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                    attr="© OpenTopoMap", control_scale=True)
    # pontos do CSV em cluster
    cl = MarkerCluster().add_to(m0)
    for _, r in db.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        code = str(r.get("code") or r.get("name") or "")
        name = str(r.get("name") or "")
        folium.CircleMarker((lat,lon), radius=3, color="#333", fill=True, fill_opacity=0.9,
                            tooltip=f"{code} — {name}").add_to(cl)
    # WPs já adicionados
    for w in st.session_state.wps:
        folium.CircleMarker((w["lat"],w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1,
                            tooltip=w["name"]).add_to(m0)

    map_out = st_folium(m0, width=None, height=480, key="pickmap")
    with st.form("add_by_click"):
        cA,cB,_ = st.columns([2,1,1])
        with cA: nm = st.text_input("Nome", "WP novo")
        with cB: alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
        clicked = map_out.get("last_clicked")
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            lat, lon = clicked["lat"], clicked["lng"]
            append_wp(nm, float(lat), float(lon), float(alt))
            st.success("Adicionado.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== EDITOR WPs ========
del_idx = None
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4 = st.columns([2,2,2,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                del_idx = i
if del_idx is not None:
    st.session_state.wps.pop(del_idx)
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== TOC/TOD COMO WPs ========
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, rod_fpm):
    nodes = []
    if len(user_wps) < 2: return nodes
    for i in range(len(user_wps)-1):
        A, B = user_wps[i], user_wps[i+1]
        nodes.append(A)
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        _, _, gs_cl = wind_triangle(tc, CLIMB_TAS,   wind_from, wind_kt)
        _, _, gs_de = wind_triangle(tc, DESCENT_TAS, wind_from, wind_kt)
        if B["alt"] > A["alt"]:
            dh = B["alt"] - A["alt"]; t_need = dh / max(roc_fpm, 1); d_need = gs_cl * (t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:
            dh = A["alt"] - B["alt"]; t_need = dh / max(rod_fpm, 1); d_need = gs_de * (t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
    nodes.append(user_wps[-1])
    return nodes

# ======== LEGS ========
def build_legs_from_nodes(nodes, wind_from, wind_kt, mag_var, mag_is_e, ck_every_min):
    legs = []
    if len(nodes) < 2: return legs

    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None

    carry_efob = float(st.session_state.start_efob)
    t_cursor = 0
    cum_dist = 0.0
    cum_time = 0

    for i in range(len(nodes)-1):
        A, B = nodes[i], nodes[i+1]
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        profile = "LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile=="CLIMB" else (DESCENT_TAS if profile=="DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
        mh = apply_var(th, mag_var, mag_is_e)
        time_sec = rt10((dist / max(gs,1e-9)) * 3600.0) if gs>0 else 0
        burn = FUEL_FLOW * (time_sec/3600.0)
        efob_start = carry_efob; efob_end = max(0.0, r10f(efob_start - burn))
        clk_start = (base_time + dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor)}"
        clk_end   = (base_time + dt.timedelta(seconds=t_cursor+time_sec)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor+time_sec)}"

        cps=[]
        if ck_every_min>0 and gs>0:
            k=1
            while k*ck_every_min*60 <= time_sec:
                t=k*ck_every_min*60; d=gs*(t/3600.0)
                eto=(base_time + dt.timedelta(seconds=t_cursor+t)).strftime('%H:%M') if base_time else ""
                cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto})
                k+=1

        cum_dist += dist; cum_time += time_sec

        legs.append({
            "i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
            "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),
            "efob_start":r10f(efob_start),"efob_end":r10f(efob_end),
            "clock_start":clk_start,"clock_end":clk_end,"cps":cps,
            "cum_dist":r10f(cum_dist),"cum_time":cum_time
        })
        t_cursor += time_sec; carry_efob = efob_end
    return legs

# ======== GERAR ROTA ========
cgen,_ = st.columns([2,6])
with cgen:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(
            st.session_state.wps, st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.roc_fpm, st.session_state.rod_fpm
        )
        st.session_state.legs = build_legs_from_nodes(
            st.session_state.route_nodes, st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.mag_var, st.session_state.mag_is_e, st.session_state.ck_default
        )

# ======== RESUMO ========
if st.session_state.legs:
    total_sec  = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final = st.session_state.legs[-1]["efob_end"]
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b></div>"
        + f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>"
        + "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== LABEL ENGINE / DOGHOUSES (igual à rev24) ========
LABEL_MIN_NM_NORMAL = 0.2
ZONE_LABEL_BASE_R   = 1.0
LABEL_MIN_CLEAR     = 0.7

def html_marker(m, lat, lon, html):
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def doghouse_html_capsule(lines, angle_tc, scale=1.0):
    rot = angle_tc - 90.0
    fs  = int(13*scale)
    inner = "".join([f"<div style='white-space:nowrap'>{ln}</div>" for ln in lines])
    return f"""
    <div style="
        transform: translate(-50%,-50%) rotate({rot}deg);
        transform-origin:center center;
        font-size:{fs}px; color:#111; letter-spacing:.1px; line-height:1.06;
        font-variant-numeric: tabular-nums;
        background: rgba(255,255,255,0.97);
        padding:6px 8px; border-radius:999px; border:2px solid #111;
        box-shadow:0 0 0 2px rgba(255,255,255,0.97); text-align:center;">
        {inner}
    </div>
    """

def arrow_polygon(center_lat, center_lon, heading_deg, length_nm, width_nm, head_nm):
    F_lat, F_lon = dest_point(center_lat, center_lon, heading_deg,  length_nm/2.0)
    B_lat, B_lon = dest_point(center_lat, center_lon, heading_deg, -length_nm/2.0)
    neck_lat, neck_lon = dest_point(F_lat, F_lon, heading_deg, -head_nm)
    def lat_off(lat, lon, side, off_nm): return dest_point(lat, lon, heading_deg + 90*side, off_nm)
    half = width_nm/2.0
    BL = lat_off(B_lat, B_lon, -1, half)
    BR = lat_off(B_lat, B_lon, +1, half)
    NL = lat_off(neck_lat, neck_lon, -1, half*0.85)
    NR = lat_off(neck_lat, neck_lon, +1, half*0.85)
    return [BL, NL, (F_lat, F_lon), NR, BR, BL]

def dynamic_label_params(dist_nm, global_scale):
    base = min(1.25, max(0.85, dist_nm/7.0))
    s = base * float(global_scale)
    L = min(2.6, max(1.9, 2.05*s))
    W = min(0.72, max(0.46, 0.52*s))
    H = min(0.55, max(0.36, 0.42*s))
    side_off = min(2.1, max(0.9, 1.0*s))
    return s, L, W, H, side_off

class Zones:
    def __init__(self): self.z=[]
    def add(self, lat, lon, r): self.z.append((lat,lon,float(r)))
    def clearance(self, lat, lon):
        if not self.z: return 9e9
        return min(_nm_dist((lat,lon),(a,b)) - r for a,b,r in self.z)
    def add_leg_corridor(self, A, B, spacing_nm=0.9, r_nm=0.38):
        dist = _nm_dist((A["lat"],A["lon"]), (B["lat"],B["lon"]))
        if dist <= spacing_nm: return
        steps = max(2, int(dist/spacing_nm))
        for k in range(1, steps):
            p = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], dist*k/steps)
            self.add(p[0], p[1], r_nm)

def node_outside_turn_side(legs, idx_node, thr=12):
    if idx_node <= 0 or idx_node-1 >= len(legs): return None
    prev_tc = legs[idx_node-1]["TC"]
    next_tc = legs[idx_node]["TC"] if idx_node < len(legs) else prev_tc
    turn = angdiff(next_tc, prev_tc)
    if turn > thr:  return +1
    if turn < -thr: return -1
    return None

def _wp_time_fuel(nodes, legs):
    info = [{"eto": None, "efob": None} for _ in nodes]
    if not legs: return info
    info[0]["eto"]  = legs[0]["clock_start"]
    info[0]["efob"] = legs[0]["efob_start"]
    for i in range(1, len(nodes)):
        Lprev = legs[i-1]
        info[i]["eto"]  = Lprev["clock_end"]
        info[i]["efob"] = Lprev["efob_end"]
    return info

def build_fix_visit_labels(nodes, legs):
    info = _wp_time_fuel(nodes, legs)
    grouped = {}
    for idx, N in enumerate(nodes):
        base = re.sub(r"\s+#\d+$","",str(N["name"]))
        key = (round(N["lat"],6), round(N["lon"],6), base)
        if key not in grouped:
            grouped[key] = {"name": base, "lat": N["lat"], "lon": N["lon"], "pairs":[], "first_idx": idx}
        eto  = info[idx]["eto"]; efb = info[idx]["efob"]
        grouped[key]["pairs"].append((efb, eto))
    for g in grouped.values():
        i = g["first_idx"]
        g["side"] = node_outside_turn_side(legs, i)
        g["tc_ref"] = legs[i]["TC"] if i < len(legs) else legs[-1]["TC"]
    return list(grouped.values())

def fix_label_html_compact(name, pairs, scale=1.0):
    fs_name = int(14*scale)
    fs_line = int(11.5*scale)
    visits=[]
    for efb, eto in pairs:
        top = f"{(efb if efb is not None else 0):.1f} L"
        bot = f"{eto or '-'}"
        visits.append(
            f"<div style='text-align:center;margin-top:1px'>"
            f"<div style='white-space:nowrap;font-size:{fs_line}px'>{top}</div>"
            f"<div style='white-space:nowrap;font-size:{fs_line-1}px'>{bot}</div>"
            f"</div>"
        )
    body = "".join(visits)
    return f"""
    <div style="text-align:center;
                color:#111; font-weight:900; text-shadow:
                -1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;">
        <div style="font-size:{fs_name}px; white-space:nowrap">{name}</div>
        {body}
    </div>
    """

def icon_fix_html(scale=1.0):
    sz = int(10*scale); ring = int(18*scale)
    return f"""
    <div style="transform:translate(-50%,-50%);
                width:{ring}px;height:{ring}px;border:2px solid #111;border-radius:50%;
                background:#fff;display:flex;align-items:center;justify-content:center">
        <div style="width:{sz}px;height:{sz}px;border-radius:50%;background:#1D4ED8;"></div>
    </div>
    """

def icon_toc_html(scale=1.0):
    s = int(16*scale)
    return f"<div style='transform:translate(-50%,-50%) rotate(45deg); width:{s}px; height:{s}px; background:#FF7A00;border:2px solid #111;'></div>"

def icon_tod_html(scale=1.0):
    b = int(18*scale); h = int(14*scale)
    return f"""
    <div style="transform:translate(-50%,-40%); width:0; height:0;
                border-left:{b//2}px solid transparent;
                border-right:{b//2}px solid transparent;
                border-top:{h}px solid #00B386;"></div>
    """

def render_map(nodes, legs, base_choice, maptiler_key=""):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return

    m = folium.Map(location=list(st.session_state.map_center),
                   zoom_start=st.session_state.map_zoom,
                   tiles=None, control_scale=True, prefer_canvas=True)

    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
                         attr="© EOX Sentinel-2").add_to(m)
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png",
                         attr="© EOX Overlay", overlay=True).add_to(m)
    elif base_choice == "Esri World Imagery (satélite + labels)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri").add_to(m)
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", overlay=True).add_to(m)
    elif base_choice == "Esri World TopoMap (topo)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri").add_to(m)
    elif base_choice == "OpenTopoMap (VFR-ish)":
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                         attr="© OpenTopoMap (CC-BY-SA)").add_to(m)
    elif base_choice == "MapTiler Satellite Hybrid (requer key)" and maptiler_key:
        folium.TileLayer(f"https://api.maptiler.com/maps/hybrid/256/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_key}",
                         attr="© MapTiler").add_to(m)
    else:
        folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                         attr="© OpenStreetMap").add_to(m)

    Fullscreen(position='topleft', title='Fullscreen', force_separate_button=True).add_to(m)

    for L in legs:
        latlngs = [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])]
        color = PROFILE_COLORS.get(L["profile"], "#C000FF")
        folium.PolyLine(latlngs, color="#ffffff", weight=10, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color=color, weight=4, opacity=1.0).add_to(m)

    if st.session_state.show_ticks:
        for L in legs:
            if L["GS"]<=0 or not L["cps"]: continue
            for cp in L["cps"]:
                d = min(L["Dist"], L["GS"]*(cp["t"]/3600.0))
                latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
                llat, llon = dest_point(latm, lonm, L["TC"]-90, CP_TICK_HALF)
                rlat, rlon = dest_point(latm, lonm, L["TC"]+90, CP_TICK_HALF)
                folium.PolyLine([(llat,llon),(rlat,rlon)], color="#111111", weight=3, opacity=1).add_to(m)

    zones = Zones()
    for L in legs: zones.add_leg_corridor(L["A"], L["B"])

    if st.session_state.show_doghouses:
        prev_side = None
        for idx, L in enumerate(legs):
            if L["Dist"] < LABEL_MIN_NM_NORMAL:  continue
            s, Lnm, Wnm, Hnm, side_off = dynamic_label_params(L["Dist"], st.session_state.text_scale)
            cur = L["TC"]; nxt = legs[idx+1]["TC"] if idx < len(legs)-1 else L["TC"]
            turn = angdiff(nxt, cur)
            prefer = +1 if turn>12 else (-1 if turn<-12 else (prev_side or +1))
            mid = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], 0.50*L["Dist"])
            anchor = dest_point(mid[0], mid[1], L["TC"] + 90*prefer, side_off)
            if zones.clearance(anchor[0], anchor[1]) < LABEL_MIN_CLEAR:
                for extra in (0.35,0.7,1.1,1.5,1.9):
                    cand = dest_point(anchor[0], anchor[1], L["TC"] + 90*prefer, extra)
                    if zones.clearance(cand[0], cand[1]) >= LABEL_MIN_CLEAR: anchor = cand; break
            zones.add(anchor[0], anchor[1], ZONE_LABEL_BASE_R); prev_side = prefer
            lines = [
                f"{deg3(L['MH'])}|{deg3(L['TC'])}",
                f"{rint(L['GS'])}kt",
                f"{mmss(L['time_sec'])}",
                f"{L['Dist']:.1f}nm",
                f"{L['burn']:.1f}L",
            ]
            poly = arrow_polygon(anchor[0], anchor[1], L["TC"], Lnm, Wnm, Hnm)
            folium.Polygon(poly, color="#000000", weight=2, fill=True, fill_color="#FFFFFF", fill_opacity=0.96).add_to(m)
            html_marker(m, anchor[0], anchor[1], doghouse_html_capsule(lines, L["TC"], scale=s))
    else:
        for L in legs:
            count = max(2, int(L["Dist"]/6)+1)
            for k in range(1, count):
                d = L["Dist"]*k/(count)
                latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
                fs = int(22 * st.session_state.text_scale)
                html = f"""
                <div style="transform:translate(-50%,-50%) rotate({L['TC']-90}deg);
                            font-size:{fs}px;font-weight:900;color:#111;
                            text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff">➤</div>
                """
                html_marker(m, latm, lonm, html)

    for N in st.session_state.route_nodes:
        if str(N["name"]).startswith("TOC"):
            html_marker(m, N["lat"], N["lon"], icon_toc_html(scale=st.session_state.text_scale))
        elif str(N["name"]).startswith("TOD"):
            html_marker(m, N["lat"], N["lon"], icon_tod_html(scale=st.session_state.text_scale))
        else:
            html_marker(m, N["lat"], N["lon"], icon_fix_html(scale=st.session_state.text_scale))

    grouped = build_fix_visit_labels(st.session_state.route_nodes, legs)
    def place_label_near_fix(fix, zones: Zones, base_off_nm=0.22):
        side = fix.get("side"); tc = fix.get("tc_ref", 0.0)
        dirs = []
        if side in (-1,+1): dirs.extend([tc+90*side, tc+90*side+30, tc+90*side-30])
        dirs.extend([tc-90, tc+90, tc, tc+180, tc+45, tc-45])
        dists = [base_off_nm, base_off_nm+0.12, base_off_nm+0.26]
        best = None; best_clear = -9e9
        for d in dists:
            for hdg in dirs:
                p = dest_point(fix["lat"], fix["lon"], hdg, d)
                c = zones.clearance(p[0], p[1])
                if c > best_clear: best_clear = c; best = p
        zones.add(best[0], best[1], 0.55)
        return best
    for fx in grouped:
        anchor = place_label_near_fix(fx, zones, base_off_nm=0.22)
        html_marker(m, anchor[0], anchor[1], fix_label_html_compact(fx["name"], fx["pairs"], scale=float(st.session_state.text_scale)))

    st_folium(m, width=None, height=760, key="mainmap", returned_objects=[])

if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs,
               base_choice=st.session_state.map_base, maptiler_key=st.session_state.maptiler_key)
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")

# ================== ✈️ FLIGHT PLAN (Item 15) ==================
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
        if lat_min == 60: lat_deg += 1; lat_min = 0
        if lon_min == 60: lon_deg += 1; lon_min = 0
        hemi_ns = "N" if lat >= 0 else "S"
        hemi_ew = "E" if lon >= 0 else "W"
        return f"{lat_deg:02d}{lat_min:02d}{hemi_ns}{lon_deg:03d}{lon_min:02d}{hemi_ew}"
    def build_fpl_route(user_wps):
        if len(user_wps) < 2: return ""
        seq = user_wps[:]
        if is_icao_aerodrome(seq[0]["name"]):  seq = seq[1:]
        if seq and is_icao_aerodrome(seq[-1]["name"]): seq = seq[:-1]
        tokens = []
        for w in seq:
            nm = re.sub(r"\s+#\d+$","",str(w["name"]).upper())
            if is_published_sigpt(nm): tokens.append(nm)
            else: tokens.append(icao_latlon(w["lat"], w["lon"]))
        return ("DCT " + " DCT ".join(tokens)) if tokens else ""
    route_str = build_fpl_route(st.session_state.wps)
    if route_str: st.code(route_str.upper(), language=None)
    else: st.info("Adiciona WPs e volta aqui para gerar a rota.")

# ================== 🧾 FORMULÁRIOS (PDF) ==================
with tab_pdf:
    st.subheader("Gerar formulários NAVLOG (PDF)")

    # Cabeçalhos
    with st.form("pdf_headers"):
        c1,c2,c3 = st.columns(3)
        with c1:
            aircraft   = st.text_input("Aircraft", "P208")
            registration= st.text_input("Registration", "CS-XXX")
            callsign   = st.text_input("Callsign", "SVN01")
            qnh        = st.text_input("QNH", "")
        with c2:
            etd_eta    = st.text_input("ETD/ETA", "")
            takeoff    = st.text_input("Takeoff", "")
            landing    = st.text_input("Landing", "")
            shutdown   = st.text_input("Shutdown", "")
        with c3:
            lesson     = st.text_input("Lesson", "")
            instrutor  = st.text_input("Instrutor", "")
            student    = st.text_input("Student", "")
            leg_number = st.text_input("Leg Number", "")
        c4,c5,c6 = st.columns(3)
        with c4:
            dept      = st.text_input("Departure Airfield", st.session_state.wps[0]["name"] if st.session_state.wps else "")
            arrival   = st.text_input("Arrival Airfield", st.session_state.wps[-1]["name"] if st.session_state.wps else "")
        with c5:
            level_ff  = st.text_input("Level F/F", "")
            climb_fuel= st.text_input("Climb Fuel", "")
            enroute   = st.text_input("Enroute Freq", "")
        with c6:
            temp_isa  = st.text_input("TEMP/ISA DEV", "")
            mag_var_f = st.text_input("MAG VAR", f"{'-' if st.session_state.mag_is_e else '+'}{st.session_state.mag_var:.1f}°")
            wind_fld  = st.text_input("WIND", f"{int(st.session_state.wind_from)} / {int(st.session_state.wind_kt)}")
        clearances = st.text_area("Clearances/OBS", "", height=60)
        submitted = st.form_submit_button("Guardar cabeçalhos")

    # Util PDF: preencher campos
    def fill_pdf_fields(template_path: str, field_values: dict) -> bytes:
        try:
            from pypdf import PdfReader, PdfWriter
        except Exception as e:
            st.error("Precisas da biblioteca 'pypdf' (`pip install pypdf`)."); raise
        reader = PdfReader(open(template_path, "rb"))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        # atualizar campos em todas as páginas
        writer.update_page_form_field_values(writer.pages[0], field_values)
        # tentar nas restantes (alguns formulários têm campos repartidos)
        for i in range(1, len(writer.pages)):
            try: writer.update_page_form_field_values(writer.pages[i], field_values)
            except: pass
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()

    # Constrói dicionários de campos a partir das legs
    def legs_to_fields_for_main(legs):
        # pela amostra, reservamos 11 linhas na capa
        max_main = 11
        n = min(len(legs), max_main)
        data = {
            "AIRCRAFT": aircraft, "REGISTRATION": registration, "CALLSIGN": callsign,
            "ETD/ETA": etd_eta, "TAKEOFF": takeoff, "LANDING": landing, "SHUTDOWN": shutdown,
            "LESSON": lesson, "INSTRUTOR": instrutor, "STUDENT": student, "Leg_Number": leg_number,
            "LEVEL F/F": level_ff, "CLIMB FUEL": climb_fuel, "QNH": qnh, "DEPT": dept, "ARRIVAL": arrival,
            "ENROUTE": enroute, "CLEARANCES": clearances,
            "WIND": wind_fld, "MAG_VAR": mag_var_f, "TEMP/ISA_DEV": temp_isa,
            "FLIGHT_LEVEL/ALTITUDE": ""
        }
        for i in range(n):
            L = legs[i]
            idx = f"{i+1:02d}"
            data[f"Leg{idx}_Waypoint"]           = str(L["A"]["name"])
            data[f"Leg{idx}_Navaid_Identifier"]  = ""
            data[f"Leg{idx}_Navaid_Frequency"]   = ""
            data[f"Leg{idx}_Altitude_FL"]        = str(int(round(L["B"]["alt"])))  # alt alvo do segmento
            data[f"Leg{idx}_True_Course"]        = f"{int(round(L['TC']))}"
            data[f"Leg{idx}_True_Heading"]       = f"{int(round(L['TH']))}"
            data[f"Leg{idx}_Magnetic_Heading"]   = f"{int(round(L['MH']))}"
            data[f"Leg{idx}_True_Airspeed"]      = f"{int(round(L['TAS']))}"
            data[f"Leg{idx}_Ground_Speed"]       = f"{int(round(L['GS']))}"
            data[f"Leg{idx}_Leg_Distance"]       = f"{L['Dist']:.1f}"
            data[f"Leg{idx}_Leg_ETE"]            = mmss(L["time_sec"])
            data[f"Leg{idx}_ETO"]                = L["clock_end"]
            data[f"Leg{idx}_Planned_Burnoff"]    = f"{L['burn']:.1f}"
            data[f"Leg{idx}_Estimated_FOB"]      = f"{L['efob_end']:.1f}"
            data[f"Leg{idx}_Cumulative_Distance"]= f"{L['cum_dist']:.1f}"
            data[f"Leg{idx}_Cumulative_ETE"]     = mmss(L["cum_time"])
        return data, n

    def legs_to_fields_for_cont(legs, start_index):
        # continuação começa em Leg12
        start_leg_number = 12
        data = {
            "FLIGHT_LEVEL_ALTITUDE": "",
            "TEMP_ISA_DEV": temp_isa,
        }
        filled = 0
        for i in range(start_index, len(legs)):
            L = legs[i]
            field_no = start_leg_number + filled
            if field_no > 23: break  # limite usual do template
            idx = f"{field_no:02d}"
            data[f"Leg{idx}_Waypoint"]           = str(L["A"]["name"])
            data[f"Leg{idx}_Navaid_Identifier"]  = ""
            data[f"Leg{idx}_Navaid_Frequency"]   = ""
            data[f"Leg{idx}_Altitude_FL"]        = str(int(round(L["B"]["alt"])))
            data[f"Leg{idx}_True_Course"]        = f"{int(round(L['TC']))}"
            data[f"Leg{idx}_True_Heading"]       = f"{int(round(L['TH']))}"
            data[f"Leg{idx}_Magnetic_Heading"]   = f"{int(round(L['MH']))}"
            data[f"Leg{idx}_True_Airspeed"]      = f"{int(round(L['TAS']))}"
            data[f"Leg{idx}_Ground_Speed"]       = f"{int(round(L['GS']))}"
            data[f"Leg{idx}_Leg_Distance"]       = f"{L['Dist']:.1f}"
            data[f"Leg{idx}_Leg_ETE"]            = mmss(L["time_sec"])
            data[f"Leg{idx}_ETO"]                = L["clock_end"]
            data[f"Leg{idx}_Planned_Burnoff"]    = f"{L['burn']:.1f}"
            data[f"Leg{idx}_Estimated_FOB"]      = f"{L['efob_end']:.1f}"
            data[f"Leg{idx}_Cumulative_Distance"]= f"{L['cum_dist']:.1f}"
            data[f"Leg{idx}_Cumulative_ETE"]     = mmss(L["cum_time"])
            filled += 1
        return data, filled

    st.markdown("**Templates carregados?** " +
                ("✅" if (os.path.exists(TEMPLATE_MAIN) and os.path.exists(TEMPLATE_CONT)) else "⚠️ Coloca `NAVLOG_FORM.pdf` e `NAVLOG_FORM_1.pdf` ao lado do app."))

    if st.button("Gerar formulários PDF (NAVLOG)", type="primary", use_container_width=True):
        if not st.session_state.legs:
            st.warning("Gera primeiro a rota para preencher os formulários.")
        elif not (os.path.exists(TEMPLATE_MAIN) and os.path.exists(TEMPLATE_CONT)):
            st.error("Templates não encontrados. Verifica `NAVLOG_FORM.pdf` e `NAVLOG_FORM_1.pdf`.")
        else:
            # Capa
            data_main, used_main = legs_to_fields_for_main(st.session_state.legs)
            pdf_main_bytes = fill_pdf_fields(TEMPLATE_MAIN, data_main)
            st.download_button("⬇️ Download — navlog_main_filled.pdf",
                               data=pdf_main_bytes, file_name="navlog_main_filled.pdf", mime="application/pdf")

            # Continuação se necessário
            if len(st.session_state.legs) > used_main:
                data_cont, used_cont = legs_to_fields_for_cont(st.session_state.legs, used_main)
                if used_cont > 0:
                    pdf_cont_bytes = fill_pdf_fields(TEMPLATE_CONT, data_cont)
                    st.download_button("⬇️ Download — navlog_cont_filled.pdf",
                                        data=pdf_cont_bytes, file_name="navlog_cont_filled.pdf", mime="application/pdf")
                else:
                    st.info("As pernas cabem todas na capa; não foi necessário gerar continuação.")
            else:
                st.info("As pernas cabem todas na capa; não foi necessário gerar continuação.")
