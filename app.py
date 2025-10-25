# app.py — NAVLOG — rev27
# ---------------------------------------------------------------
# - Último fix agora aparece (última linha usa o B do leg final).
# - Alternate: preenche todos os campos Alternate_* + Arrival/Departure.
# - PDF principal (NAVLOG_FORM.pdf) cobre Leg01..Leg22 (2 págs).
# - Se >22 legs, gera NAVLOG_FORM_1.pdf (máx 2 págs).
# - Frequências são valores (DEPT/ENROUTE/ARRIVAL).
# - Cabeçalho: ETD/ETA, STUDENT (AMOIT), LESSON, INSTRUCTOR, CALLSIGN, REG.
# - TOD usa ROD (ft/min) do utilizador (não ângulo).
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib, os
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MarkerCluster
from math import degrees

# ======== PDF ========
from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName

TEMPLATE_MAIN = "NAVLOG_FORM.pdf"    # 2 páginas, Leg01..Leg22 + totais + Alternate
TEMPLATE_CONT = "NAVLOG_FORM_1.pdf"  # continuação (Leg12..Leg22 + Alternate)

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0   # kt
FUEL_FLOW = 20.0                                        # L/h
EARTH_NM  = 3440.065

PROFILE_COLORS = {"CLIMB":"#FF7A00","LEVEL":"#C000FF","DESCENT":"#00B386"}
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
.sep{height:1px;background:#e5e7eb;margin:10px 0}
.small{font-size:12px;color:#555}
.row{display:flex;gap:8px;align-items:center}
.badge{font-weight:700;border:1px solid #111;border-radius:8px;padding:2px 6px;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# ======== HELPERS ========
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
ens("roc_fpm", 600); ens("rod_fpm", 500)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")
ens("text_scale", 1.0); ens("show_ticks", True); ens("show_doghouses", True)
ens("map_center", (39.7, -8.1)); ens("map_zoom", 8)
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
    b1,b2,b3 = st.columns([2,2,1])
    with b1:
        bases = ["OpenTopoMap (VFR-ish)","EOX Sentinel-2 (satélite)","Esri World Imagery (satélite + labels)","Esri World TopoMap (topo)","OSM Standard"]
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
tab_csv, tab_map, tab_fpl = st.tabs(["🔎 Pesquisar CSV", "🗺️ Adicionar no mapa", "✈️ Flight Plan"])

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
    cl = MarkerCluster().add_to(m0)
    for _, r in db.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        code = str(r.get("code") or r.get("name") or "")
        name = str(r.get("name") or "")
        folium.CircleMarker((lat,lon), radius=3, color="#333", fill=True, fill_opacity=0.9,
                            tooltip=f"{code} — {name}").add_to(cl)
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
            dh = B["alt"] - A["alt"]; t_need = dh / max(roc_fpm, 1.0); d_need = gs_cl * (t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:
            dh = A["alt"] - B["alt"]; t_need = dh / max(rod_fpm, 1.0); d_need = gs_de * (t_need/60.0)
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
                cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto}); k+=1

        legs.append({
            "i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
            "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),
            "efob_start":r10f(efob_start),"efob_end":r10f(efob_end),"clock_start":clk_start,"clock_end":clk_end,"cps":cps
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
    total_dist = r10f(sum(L["Dist"] for L in st.session_state.legs))
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

# ======== MAPA ========  (idêntico ao rev26; omitido aqui por brevidade)
# ... a função render_map(...) é igual à do rev26 e é chamada abaixo:
def render_map(nodes, legs, base_choice):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**."); return
    m = folium.Map(location=list(st.session_state.map_center),
                   zoom_start=st.session_state.map_zoom, tiles=None, control_scale=True, prefer_canvas=True)
    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg", attr="© EOX").add_to(m)
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png", attr="© EOX", overlay=True).add_to(m)
    elif base_choice == "Esri World Imagery (satélite + labels)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="© Esri").add_to(m)
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", attr="© Esri", overlay=True).add_to(m)
    elif base_choice == "Esri World TopoMap (topo)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", attr="© Esri").add_to(m)
    else:
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="© OpenTopoMap (CC-BY-SA)").add_to(m)
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

    def html_marker(m, lat, lon, html):
        folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

    def fix_label(name, fuel, eto, scale=1.0):
        fs_name = int(14*scale); fs_line = int(11.5*scale)
        return f"""<div style="text-align:center;color:#111;font-weight:900;text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;">
        <div style="font-size:{fs_name}px; white-space:nowrap">{name}</div>
        <div style='font-size:{fs_line}px'>{fuel:.1f} L</div>
        <div style='font-size:{fs_line-1}px'>{eto or '-'}</div></div>"""

    for N in st.session_state.route_nodes:
        html_marker(m, N["lat"], N["lon"], f"<div style='transform:translate(-50%,-50%);width:18px;height:18px;border:2px solid #111;border-radius:50%;background:#fff'></div>")

    st_folium(m, width=None, height=760, key="mainmap", returned_objects=[])

if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs, base_choice=st.session_state.map_base)
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")

# ================== ✈️ FLIGHT PLAN (Item 15) ==================
with tab_fpl:
    st.subheader("Flight Plan — rota (Item 15)")
    def is_icao_aerodrome(name:str) -> bool: return bool(re.fullmatch(r"[A-Z]{4}", str(name).upper()))
    def is_published_sigpt(name:str) -> bool: return bool(re.fullmatch(r"[A-Z0-9]{3,5}", str(name).upper()))
    def icao_latlon(lat:float, lon:float) -> str:
        lat_abs = abs(lat); lon_abs = abs(lon)
        lat_deg = int(lat_abs); lon_deg = int(lon_abs)
        lat_min = int(round((lat_abs - lat_deg)*60)); lon_min = int(round((lon_abs - lon_deg)*60))
        if lat_min == 60: lat_deg += 1; lat_min = 0
        if lon_min == 60: lon_deg += 1; lon_min = 0
        hemi_ns = "N" if lat >= 0 else "S"; hemi_ew = "E" if lon >= 0 else "W"
        return f"{lat_deg:02d}{lat_min:02d}{hemi_ns}{lon_deg:03d}{lon_min:02d}{hemi_ew}"
    def build_fpl_route(user_wps):
        if len(user_wps) < 2: return ""
        seq = user_wps[:]
        if is_icao_aerodrome(seq[0]["name"]):  seq = seq[1:]
        if seq and is_icao_aerodrome(seq[-1]["name"]): seq = seq[:-1]
        tokens = []
        for w in seq:
            nm = re.sub(r"\s+#\d+$","",str(w["name"]).upper())
            tokens.append(nm if is_published_sigpt(nm) else icao_latlon(w["lat"], w["lon"]))
        return ("DCT " + " DCT ".join(tokens)) if tokens else ""
    route_str = build_fpl_route(st.session_state.wps)
    st.code(route_str.upper(), language=None) if route_str else st.info("Adiciona WPs e volta aqui para gerar a rota.")

# ===============================================================
#                 📄  NAVLOG — Cabeçalho & PDF
# ===============================================================

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.header("📄 NAVLOG — Cabeçalho & PDF")

REG_OPTIONS = ["CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW","CS-ECC","CS-ECD"]

c0,c1,c2,c3,c4 = st.columns(5)
with c0: callsign = st.text_input("Callsign", "RVP")
with c1: registration = st.selectbox("Registration", REG_OPTIONS, index=0)
with c2: student = st.text_input("Student", "AMOIT")
with c3: lesson = st.text_input("Lesson", "")
with c4: instructor = st.text_input("Instructor", "")

c5,c6,c7,c8,c9 = st.columns(5)
with c5: etd = st.text_input("ETD (HH:MM)", "")
with c6: eta = st.text_input("ETA (HH:MM, opcional)", "")
with c7: dept_freq = st.text_input("FREQ DEPT", "119.805")
with c8: enroute_freq = st.text_input("FREQ ENROUTE", "123.755")
with c9: arrival_freq = st.text_input("FREQ ARRIVAL", "131.675")

# Alternate
st.subheader("Alternate (opcional)")
cA,cB,cC = st.columns([2,1,1])
with cA: alt_query = st.text_input("Pesquisar alternante (código/nome)", "")
with cB: alt_elev = st.number_input("Elevação ALT (ft)", 0, 10000, 350, step=10)
with cC: use_alt = st.checkbox("Usar este alternante", value=False)

alt_choice = None
if alt_query.strip():
    tql = alt_query.lower().strip()
    cand = db[db.apply(lambda r: any(tql in str(v).lower() for v in r.values), axis=1)].head(1)
    if not cand.empty:
        r = cand.iloc[0]
        alt_choice = {"name": r.get("code") or r.get("name"), "lat": float(r["lat"]), "lon": float(r["lon"]), "elev": int(alt_elev)}
        st.success(f"ALT selecionado: {alt_choice['name']}  ({alt_choice['lat']:.4f}, {alt_choice['lon']:.4f})")
    else: st.warning("Sem match para o alternante.")

# cálculo do leg para o alternante (a partir do destino)
alt_leg_info = None
if use_alt and alt_choice and st.session_state.wps:
    dest = st.session_state.wps[-1]
    tc_alt = gc_course_tc(dest["lat"], dest["lon"], alt_choice["lat"], alt_choice["lon"])
    wca, th_alt, gs_alt = wind_triangle(tc_alt, CRUISE_TAS, st.session_state.wind_from, st.session_state.wind_kt)
    mh_alt = apply_var(th_alt, st.session_state.mag_var, st.session_state.mag_is_e)
    dist_alt = gc_dist_nm(dest["lat"], dest["lon"], alt_choice["lat"], alt_choice["lon"])
    ete_alt_sec = int(round((dist_alt / max(gs_alt,1e-9)) * 3600))
    burn_alt = FUEL_FLOW * (ete_alt_sec/3600.0)
    alt_leg_info = {"tc":tc_alt,"th":th_alt,"mh":mh_alt,"tas":CRUISE_TAS,"gs":gs_alt,"dist":dist_alt,"ete":ete_alt_sec,"burn":burn_alt}

# ============ PDF helpers ============
def _pdf_mmss(sec:int):
    m,s = divmod(int(round(sec)),60); return f"{m:02d}:{s:02d}"

def _set_need_appearances(pdf):
    if pdf.Root.AcroForm: pdf.Root.AcroForm.update(PdfDict(NeedAppearances=True))

def _fill_pdf(template_path:str, out_path:str, data:dict):
    pdf = PdfReader(template_path); _set_need_appearances(pdf)
    for page in pdf.pages:
        if not getattr(page, "Annots", None): continue
        for a in page.Annots:
            if a.Subtype==PdfName('Widget') and a.T:
                key = str(a.T)[1:-1]
                if key in data: a.update(PdfDict(V=str(data[key])))
    PdfWriter(out_path, trailer=pdf).write()
    return out_path

def _sum_time(legs, profile): return sum(L["time_sec"] for L in legs if L["profile"]==profile)
def _sum_burn(legs, profile): return round(sum(L["burn"] for L in legs if L["profile"]==profile),1)

def _compose_clock_after(total_sec, extra_sec):
    """Devolve hora 'HH:MM' se houver off-blocks; senão 'T+MM:SS'."""
    base = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base = None
    t = total_sec + extra_sec
    if base: return (base + dt.timedelta(seconds=t)).strftime("%H:%M")
    return f"T+{mmss(t)}"

def _fill_leg_line(d:dict, idx:int, L:dict, use_point:str, acc_d:float, acc_t:int, prefix="Leg"):
    """use_point: 'A' ou 'B'"""
    P = L["A"] if use_point=="A" else L["B"]
    d[f"{prefix}{idx:02d}_Waypoint"]            = str(P["name"])
    d[f"{prefix}{idx:02d}_Altitude_FL"]         = str(int(round(P["alt"])))
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

def _build_payloads_main(legs, *, callsign, registration, student, lesson, instructor,
                         dept, enroute, arrival, etd, eta, fl_hdr="", temp_hdr="",
                         alt_info=None, alt_choice=None):
    total_sec = sum(L["time_sec"] for L in legs)
    total_burn = r10f(sum(L["burn"] for L in legs))
    total_dist = r10f(sum(L["Dist"] for L in legs))
    obs = f"Climb {_pdf_mmss(_sum_time(legs,'CLIMB'))} / Cruise {_pdf_mmss(_sum_time(legs,'LEVEL'))} / Descent {_pdf_mmss(_sum_time(legs,'DESCENT'))}"
    climb_burn = _sum_burn(legs,'CLIMB')

    N = min(len(legs), 22)
    legs_main = legs[:N]
    is_last_chunk = (len(legs) <= 22)

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
        "FLIGHT_LEVEL/ALTITUDE": fl_hdr,
        "FLIGHT_LEVEL_ALTITUDE": fl_hdr,
        "TEMP/ISA_DEV": temp_hdr,
        "TEMP_ISA_DEV": temp_hdr,
        "FLT TIME": _pdf_mmss(total_sec),
        "CLIMB FUEL": f"{climb_burn:.1f}",
        "OBSERVATIONS": obs,
        "Leg_Number": str(len(legs)),
    }

    acc_d, acc_t = 0.0, 0
    for i, L in enumerate(legs_main, start=1):
        acc_d = round(acc_d + L["Dist"], 1)
        acc_t += L["time_sec"]
        use_point = "B" if (is_last_chunk and i==N) else "A"  # <- última linha do bloco final usa B
        _fill_leg_line(d, i, L, use_point, acc_d, acc_t)

    # Linha 23 (totais)
    d["Leg23_Leg_Distance"] = f"{total_dist:.1f}"
    d["Leg23_Leg_ETE"]      = _pdf_mmss(total_sec)
    d["Leg23_Planned_Burnoff"] = f"{total_burn:.1f}"
    d["Leg23_Estimated_FOB"]   = f"{legs[-1]['efob_end']:.1f}"

    # Alternate no PRINCIPAL (se fornecido)
    if alt_info and alt_choice:
        d["Alternate_Airfield"]  = alt_choice["name"]
        d["Alternate_Elevation"] = str(int(alt_choice["elev"]))
        d["Alternate_True_Course"]   = f"{int(round(alt_info['tc'])):03d}"
        d["Alternate_True_Heading"]  = f"{int(round(alt_info['th'])):03d}"
        d["Alternate_Magnetic_Heading"]= f"{int(round(alt_info['mh'])):03d}"
        d["Alternate_True_Airspeed"] = str(int(round(alt_info['tas'])))
        d["Alternate_Ground_Speed"]  = str(int(round(alt_info['gs'])))
        d["Alternate_Leg_Distance"]  = f"{alt_info['dist']:.1f}"
        d["Alternate_Cumulative_Distance"] = f"{alt_info['dist']:.1f}"
        d["Alternate_Leg_ETE"]      = _pdf_mmss(alt_info["ete"])
        d["Alternate_Cumulative_ETE"]= _pdf_mmss(alt_info["ete"])
        d["Alternate_ETO"]          = _compose_clock_after(total_sec, alt_info["ete"])
        d["Alternate_Planned_Burnoff"]= f"{r10f(alt_info['burn']):.1f}"
        d["Alternate_Estimated_FOB"]  = f"{r10f(legs[-1]['efob_end'] - alt_info['burn']):.1f}"

    return d

def _build_payload_cont(all_legs, start_idx, *, alt_info=None, alt_choice=None):
    # Folha de continuação mapeia Leg12..Leg22 (11 linhas) e pode também levar Alternate
    legs_chunk = all_legs[start_idx:start_idx+11]
    if not legs_chunk: return None
    d = {"OBSERVATIONS":"SEVENAIR OPS: 131.675"}
    N = len(all_legs)
    is_last_chunk = (start_idx + len(legs_chunk) == N)

    acc_d = 0.0; acc_t = 0
    for offset, L in enumerate(legs_chunk, start=12):
        acc_d = round(acc_d + L["Dist"], 1); acc_t += L["time_sec"]
        use_point = "B" if (is_last_chunk and offset==11 and start_idx+offset-11==N-1) else "A"
        _fill_leg_line(d, offset, L, use_point, acc_d, acc_t)

    # TOTAIS do chunk
    d["Leg23_Leg_Distance"] = f"{acc_d:.1f}"
    d["Leg23_Leg_ETE"]      = _pdf_mmss(acc_t)
    d["Leg23_Planned_Burnoff"] = f"{r10f(sum(L['burn'] for L in legs_chunk)):.1f}"
    d["Leg23_Estimated_FOB"]   = f"{legs_chunk[-1]['efob_end']:.1f}"

    # Alternate na CONTINUAÇÃO (se fornecido)
    if alt_info and alt_choice:
        total_sec_before_chunk = sum(L["time_sec"] for L in all_legs[:start_idx+len(legs_chunk)])
        d["Alternate_Airfield"]  = alt_choice["name"]
        d["Alternate_Elevation"] = str(int(alt_choice["elev"]))
        d["Alternate_True_Course"]   = f"{int(round(alt_info['tc'])):03d}"
        d["Alternate_True_Heading"]  = f"{int(round(alt_info['th'])):03d}"
        d["Alternate_Magnetic_Heading"]= f"{int(round(alt_info['mh'])):03d}"
        d["Alternate_True_Airspeed"] = str(int(round(alt_info['tas'])))
        d["Alternate_Ground_Speed"]  = str(int(round(alt_info['gs'])))
        d["Alternate_Leg_Distance"]  = f"{alt_info['dist']:.1f}"
        d["Alternate_Cumulative_Distance"] = f"{alt_info['dist']:.1f}"
        d["Alternate_Leg_ETE"]      = _pdf_mmss(alt_info["ete"])
        d["Alternate_Cumulative_ETE"]= _pdf_mmss(alt_info["ete"])
        d["Alternate_ETO"]          = _compose_clock_after(total_sec_before_chunk, alt_info["ete"])
        d["Alternate_Planned_Burnoff"]= f"{r10f(alt_info['burn']):.1f}"
        d["Alternate_Estimated_FOB"]  = f"{r10f(all_legs[start_idx+len(legs_chunk)-1]['efob_end'] - alt_info['burn']):.1f}"
    return d

# ==== Botões de exportação ====
cX, cY = st.columns([1,1])
with cX: make_pdfs = st.button("Gerar PDF(s) NAVLOG", type="primary", use_container_width=True)
with cY: st.caption("O principal tem até 22 legs. Se exceder, sai uma continuação.")

if make_pdfs:
    if not st.session_state.legs:
        st.error("Gera primeiro a rota (precisas de ter pernas calculadas).")
    else:
        # PRINCIPAL (sempre)
        d_main = _build_payloads_main(
            st.session_state.legs,
            callsign=callsign, registration=registration,
            student=student, lesson=lesson, instructor=instructor,
            dept=dept_freq, enroute=enroute_freq, arrival=arrival_freq,
            etd=etd, eta=eta, fl_hdr="", temp_hdr="",
            alt_info=alt_leg_info if (use_alt and alt_choice) else None,
            alt_choice=alt_choice if (use_alt and alt_choice) else None
        )
        out_main = _fill_pdf(TEMPLATE_MAIN, "NAVLOG_FILLED.pdf", d_main)
        with open(out_main, "rb") as f:
            st.download_button("⬇️ NAVLOG (principal)", f.read(), file_name="NAVLOG_FILLED.pdf", use_container_width=True)

        # CONTINUAÇÃO (apenas se >22 legs; pode haver mais do que uma — mas limitamos a 1 PDF adicional)
        if len(st.session_state.legs) > 22:
            d_cont = _build_payload_cont(
                st.session_state.legs, start_idx=22,
                alt_info=alt_leg_info if (use_alt and alt_choice) else None,
                alt_choice=alt_choice if (use_alt and alt_choice) else None
            )
            out_cont = _fill_pdf(TEMPLATE_CONT, "NAVLOG_FILLED_1.pdf", d_cont)
            with open(out_cont, "rb") as f:
                st.download_button("⬇️ NAVLOG (continuação)", f.read(), file_name="NAVLOG_FILLED_1.pdf", use_container_width=True)
