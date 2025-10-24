# app.py — NAVLOG — rev28
# Fluxo:
#   1) Parâmetros globais
#   2) Adicionar WPs (CSV / mapa)
#   3) Gerar rota (TOC/TOD + legs) e ver no mapa
#   4) Flight Plan (Item 15)
#   5) Gerar PDFs (NAVLOG_FORM.pdf + NAVLOG_FORM_1.pdf) — campos com nomes EXACTOS do teu template
#
# Notas:
#   • Callsign por defeito: RVP
#   • ROD em ft/min (descida) definido pelo utilizador
#   • “Climb Fuel” calculado automaticamente (soma do burn das legs CLIMB)
#   • Observações: tempos CLIMB/CRUISE/DESCENT automáticos
#   • Frequências por defeito: LPSO TWR 119.805 e LISBOA MIL 123.755
#   • Mapa lembra o zoom/centro (sem auto-fit)

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib, io, os
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from math import degrees

# ======== TEMPLATES PDF (ficheiros fornecidos por ti) ========
TEMPLATE_MAIN = "NAVLOG_FORM.pdf"     # capa (Leg01..Leg11)
TEMPLATE_CONT = "NAVLOG_FORM_1.pdf"   # continuação (Leg12..Leg23)

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0   # kt
FUEL_FLOW = 20.0                                        # L/h
EARTH_NM  = 3440.065

PROFILE_COLORS = {"CLIMB":"#FF7A00","LEVEL":"#C000FF","DESCENT":"#00B386"}
CP_TICK_HALF_NM = 0.40

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
.small{font-size:12px;color:#555}
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

def apply_var(th, var, east_is_neg=False): return wrap360(th - var if east_is_neg else th + var)

def gc_dist_nm(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2]); dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_NM * c

def gc_course_tc(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2]); dλ = λ2 - λ1
    y = math.sin(dλ)*math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.degrees(math.atan2(y, x)); return (θ + 360) % 360

def dest_point(lat, lon, bearing_deg, dist_nm):
    θ = math.radians(bearing_deg); δ = dist_nm / EARTH_NM
    φ1, λ1 = math.radians(lat), math.radians(lon)
    sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
    φ2 = math.asin(sinφ2); y = math.sin(θ)*math.sin(δ)*math.cos(φ1); x = math.cos(δ) - math.sin(φ1)*sinφ2
    λ2 = λ1 + math.atan2(y, x)
    return math.degrees(φ2), ((math.degrees(λ2)+540)%360)-180

def point_along_gc(lat1, lon1, lat2, lon2, dist_from_start_nm):
    total = gc_dist_nm(lat1, lon1, lat2, lon2)
    if total <= 0: return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)

# ======== STATE ========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("rod_fpm", 600)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)

ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_center", (39.7, -8.1)); ens("map_zoom", 8)
ens("map_base", "OpenTopoMap (VFR-ish)")
ens("text_scale", 1.0); ens("show_ticks", True)
ens("db_points", None); ens("qadd", ""); ens("alt_qadd", 3000.0)

# ======== 1) PARÂMETROS GLOBAIS ========
with st.form("globals"):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from))
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt))
    with c2:
        st.session_state.mag_var   = st.number_input("Variação magnética (±°)", -30.0, 30.0, float(st.session_state.mag_var))
        st.session_state.mag_is_e  = st.toggle("Var. é EAST (subtrai)", value=st.session_state.mag_is_e)
    with c3:
        st.session_state.roc_fpm   = st.number_input("ROC (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10)
        st.session_state.rod_fpm   = st.number_input("ROD (ft/min)", 200, 1500, int(st.session_state.rod_fpm), step=10)
    with c4:
        st.session_state.start_efob= st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP a cada (min)", 1, 10, int(st.session_state.ck_default))
    b1,b2 = st.columns([2,1])
    with b1:
        bases = ["OpenTopoMap (VFR-ish)","Esri World Imagery","Esri World TopoMap","OSM Standard"]
        st.session_state.map_base = st.selectbox("Base do mapa", bases, index=bases.index(st.session_state.map_base) if st.session_state.map_base in bases else 0)
    with b2:
        st.session_state.text_scale  = st.slider("Tamanho texto", 0.5, 1.5, float(st.session_state.text_scale), 0.05)
        st.session_state.show_ticks  = st.toggle("Mostrar riscas CP", value=st.session_state.show_ticks)
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

def make_unique_name(name: str) -> str:
    names = [str(w["name"]) for w in st.session_state.wps]
    if name not in names: return name
    k=2
    while f"{name} #{k}" in names: k+=1
    return f"{name} #{k}"

def append_wp(name, lat, lon, alt):
    st.session_state.wps.append({"name": make_unique_name(str(name)), "lat": float(lat), "lon": float(lon), "alt": float(alt)})

# ======== 2) ADICIONAR WPs ========
tab_csv, tab_map, tab_fpl, tab_pdf = st.tabs(["🔎 CSV", "🗺️ Mapa & Gerar rota", "✈️ Flight Plan", "🧾 PDFs"])

with tab_csv:
    c1, c2 = st.columns([3,1])
    with c1:
        q = st.text_input("Pesquisar (um termo)", key="qadd", placeholder="Ex.: LPSO, VACOR, ÉVORA…").strip()
    with c2:
        st.session_state.alt_qadd = st.number_input("Alt (ft) p/ novos WPs", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)

    def _score_row(row, tq, last_wp):
        code = str(row.get("code") or "").lower()
        name = str(row.get("name") or "").lower()
        sim = difflib.SequenceMatcher(None, tq, f"{code} {name}").ratio()
        starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
        near = 0.0
        if last_wp:
            d = gc_dist_nm(last_wp["lat"], last_wp["lon"], row["lat"], row["lon"])
            near = 1.0 / (1.0 + d)
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
    rows = results.head(30).to_dict("records") if not results.empty else []

    if rows:
        st.caption("Resultados")
        for i, r in enumerate(rows):
            code = r.get("code") or ""
            name = r.get("name") or ""
            local = r.get("city") or r.get("sector") or ""
            lat, lon = float(r["lat"]), float(r["lon"])
            col1, col2 = st.columns([0.84,0.16])
            with col1:
                st.markdown(
                    f"<div class='card'><b>{code} — {name}</b><div class='small'>{local}</div>"
                    f"<div class='small'>({lat:.4f}, {lon:.4f})</div></div>",
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("➕", key=f"csvadd_{i}", use_container_width=True):
                    append_wp(code or name, lat, lon, float(st.session_state.alt_qadd))
                    st.success("Adicionado.")
    else:
        st.info("Sem resultados.")

with tab_map:
    st.caption("Clica no mapa e depois em **Adicionar**. WPs do CSV visíveis.")
    # base map
    tiles, attr = ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png","© OpenTopoMap")
    if st.session_state.map_base == "Esri World Imagery":
        tiles, attr = ("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "© Esri")
    elif st.session_state.map_base == "Esri World TopoMap":
        tiles, attr = ("https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", "© Esri")
    elif st.session_state.map_base == "OSM Standard":
        tiles, attr = ("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png","© OpenStreetMap")

    m0 = folium.Map(location=list(st.session_state.map_center), zoom_start=st.session_state.map_zoom,
                    tiles=tiles, attr=attr, control_scale=True)
    cl = MarkerCluster().add_to(m0)
    for _, r in db.iterrows():
        folium.CircleMarker((float(r["lat"]), float(r["lon"])), radius=2.8, color="#333", fill=True, fill_opacity=0.9,
                            tooltip=f"{(r.get('code') or r.get('name'))} — {(r.get('name') or '')}").add_to(cl)
    # WPs atuais
    for w in st.session_state.wps:
        folium.CircleMarker((w["lat"],w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1,
                            tooltip=w["name"]).add_to(m0)

    map_out = st_folium(m0, width=None, height=480, key="pickmap", return_on_dashboard=True)
    # lembra zoom/centro
    if isinstance(map_out, dict):
        z = map_out.get("zoom"); c = map_out.get("center")
        if z: st.session_state.map_zoom = z
        if c and isinstance(c, dict) and "lat" in c and "lng" in c:
            st.session_state.map_center = (c["lat"], c["lng"])

    with st.form("add_by_click"):
        cA,cB,_ = st.columns([2,1,1])
        with cA: nm = st.text_input("Nome", "WP novo")
        with cB: alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
        clicked = map_out.get("last_clicked") if isinstance(map_out, dict) else None
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            lat, lon = clicked["lat"], clicked["lng"]
            append_wp(nm, float(lat), float(lon), float(alt))
            st.success("Adicionado.")

# ======== 3) GERAR ROTA ========
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

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
    nodes.append(user_wps[-1]); return nodes

def build_legs_from_nodes(nodes, wind_from, wind_kt, mag_var, mag_is_e, ck_every_min):
    legs = []; 
    if len(nodes) < 2: return legs
    base_time=None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None
    carry_efob = float(st.session_state.start_efob); t_cursor = 0; cum_dist=0.0; cum_time=0
    for i in range(len(nodes)-1):
        A, B = nodes[i], nodes[i+1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        profile = "LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile=="CLIMB" else (DESCENT_TAS if profile=="DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt); mh = apply_var(th, st.session_state.mag_var, st.session_state.mag_is_e)
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
        cum_dist += dist; cum_time += time_sec
        legs.append({"i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
                     "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),
                     "efob_start":r10f(efob_start),"efob_end":r10f(efob_end),
                     "clock_start":clk_start,"clock_end":clk_end,"cps":cps,
                     "cum_dist":r10f(cum_dist),"cum_time":cum_time})
        t_cursor += time_sec; carry_efob = efob_end
    return legs

cbtn,_ = st.columns([2,6])
with cbtn:
    if st.button("✅ Gerar/Atualizar rota (insere TOC/TOD)", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(st.session_state.wps, st.session_state.wind_from, st.session_state.wind_kt,
                                                         st.session_state.roc_fpm, st.session_state.rod_fpm)
        st.session_state.legs = build_legs_from_nodes(st.session_state.route_nodes, st.session_state.wind_from, st.session_state.wind_kt,
                                                      st.session_state.mag_var, st.session_state.mag_is_e, st.session_state.ck_default)

# ======== Mapa da rota ========
def render_simple_map(nodes, legs):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return
    tiles, attr = ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png","© OpenTopoMap")
    m = folium.Map(location=list(st.session_state.map_center), zoom_start=st.session_state.map_zoom, tiles=tiles, attr=attr, control_scale=True)
    # legs
    for L in legs:
        latlngs = [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])]
        color = PROFILE_COLORS.get(L["profile"], "#C000FF")
        folium.PolyLine(latlngs, color="#ffffff", weight=8, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color=color, weight=4, opacity=1.0).add_to(m)
        # CP ticks
        if st.session_state.show_ticks and L["cps"]:
            for cp in L["cps"]:
                d = min(L["Dist"], L["GS"]*(cp["t"]/3600.0))
                latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
                left  = dest_point(latm, lonm, L["TC"]-90, CP_TICK_HALF_NM)
                right = dest_point(latm, lonm, L["TC"]+90, CP_TICK_HALF_NM)
                folium.PolyLine([(left[0],left[1]),(right[0],right[1])], color="#111", weight=2.2, opacity=1).add_to(m)
    # nodes
    for idx, N in enumerate(nodes):
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#fff", weight=2, fill=True, fill_opacity=1, fill_color="#1e90ff").add_to(m)
        folium.map.Marker(
            location=(N["lat"],N["lon"]),
            icon=folium.DivIcon(html=f"<div style='transform:translate(-50%,-115%);font-weight:900;font-size:{int(16*st.session_state.text_scale)}px;text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;'>{str(N['name']).upper()}</div>")
        ).add_to(m)
    out = st_folium(m, width=None, height=600, key="routemap", return_on_dashboard=True)
    if isinstance(out, dict):
        if out.get("zoom"): st.session_state.map_zoom = out["zoom"]
        c = out.get("center")
        if c and isinstance(c, dict): st.session_state.map_center = (c["lat"], c["lng"])

if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_simple_map(st.session_state.route_nodes, st.session_state.legs)

# ======== Resumo ========
if st.session_state.legs:
    total_sec  = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final = st.session_state.legs[-1]["efob_end"]
    t_climb   = sum(L["time_sec"] for L in st.session_state.legs if L["profile"]=="CLIMB")
    t_cruise  = sum(L["time_sec"] for L in st.session_state.legs if L["profile"]=="LEVEL")
    t_descent = sum(L["time_sec"] for L in st.session_state.legs if L["profile"]=="DESCENT")
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE: <b>{hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>⛽ Burn: <b>{total_burn:.1f} L</b></div>"
        + f"<div class='kv'>🧯 EFOB final: <b>{efob_final:.1f} L</b></div>"
        + f"<div class='kv'>Perfis: CLB {mmss(t_climb)} • CRZ {mmss(t_cruise)} • DES {mmss(t_descent)}</div>"
        + "</div>", unsafe_allow_html=True
    )

# ======== 4) FLIGHT PLAN (Item 15) ========
with tab_fpl:
    st.subheader("Rota para Flight Plan (Item 15)")
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
    if route_str: st.code(route_str.upper(), language=None)
    else: st.info("Gera alguns WPs para ver aqui a rota.")

# ======== 5) PDFs — CAMPOS COM NOMES EXACTOS (dos teus templates) ========
with tab_pdf:
    st.subheader("Gerar PDFs (NAVLOG)")
    registrations = ["CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW","CS-ECC","CS-ECD"]
    with st.form("pdf_headers"):
        c1,c2,c3 = st.columns(3)
        with c1:
            aircraft     = st.text_input("Aircraft", "P208")
            registration = st.selectbox("Registration", registrations, index=0)
            callsign     = st.text_input("Callsign", "RVP")
            lesson       = st.text_input("Lesson", "")
        with c2:
            instrutor    = st.text_input("Instrutor", "")
            student      = st.text_input("Student", "")
            qnh          = st.text_input("QNH", "")
            mag_var_f    = st.text_input("MAG VAR", f"{'-' if st.session_state.mag_is_e else '+'}{st.session_state.mag_var:.1f}°")
        with c3:
            wind_fld     = st.text_input("WIND", f"{int(st.session_state.wind_from)} / {int(st.session_state.wind_kt)}")
            temp_isa     = st.text_input("TEMP/ISA DEV", "")
            etd_eta      = st.text_input("ETD/ETA", "")
        c4,c5 = st.columns(2)
        with c4:
            dep_airf     = st.text_input("Departure Airfield", st.session_state.wps[0]["name"] if st.session_state.wps else "LPSO")
            dep_freq     = st.text_input("DEP freq", "119.805")   # LPSO TWR
        with c5:
            arr_airf     = st.text_input("Arrival Airfield",  st.session_state.wps[-1]["name"] if st.session_state.wps else "LPSO")
            enroute_freq = st.text_input("ENROUTE freq", "123.755")  # Lisboa MIL
        st.form_submit_button("Guardar cabeçalhos")

    # ===== PDF helpers (EXPRESSAMENTE com nomes de campos do teu PDF) =====
    def with_cumulative(legs):
        out=[]; cd=0.0; ct=0
        for L in legs:
            cd += float(L.get("Dist",0.0)); ct += int(L.get("time_sec",0))
            LL = dict(L); LL["cum_dist"]=r10f(cd); LL["cum_time"]=ct
            out.append(LL)
        return out

    def climb_fuel_liters(legs): return r10f(sum(L.get("burn",0.0) for L in legs if L.get("profile")=="CLIMB"))

    def fill_pdf_fields_exact(template_path: str, field_values: dict) -> bytes:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(open(template_path, "rb"))
        writer = PdfWriter()
        for p in reader.pages: writer.add_page(p)
        # aplica nos 2 primeiros pages (requisito: máx 2 páginas por PDF)
        for p in range(min(2, len(writer.pages))):
            writer.update_page_form_field_values(writer.pages[p], field_values)
        out = io.BytesIO(); writer.write(out); return out.getvalue()

    # Mapeamento EXACTO — NAVLOG_FORM.pdf (capa) — Leg01..Leg11
    def build_fields_main(legs):
        legs = with_cumulative(legs)
        max_main = 11
        n = min(len(legs), max_main)
        data = {
            "AIRCRAFT": aircraft, "REGISTRATION": registration, "CALLSIGN": callsign,
            "ETD/ETA": etd_eta, "LESSON": lesson, "INSTRUTOR": instrutor, "STUDENT": student,
            "QNH": qnh, "DEPT": dep_freq, "ENROUTE": enroute_freq,
            "Departure_Airfield": dep_airf, "Arrival_Airfield": arr_airf,
            "WIND": wind_fld, "MAG_VAR": mag_var_f, "TEMP/ISA_DEV": temp_isa,
            "Leg_Number": str(len(legs)),
            "CLIMB FUEL": f"{climb_fuel_liters(legs):.1f}",
        }
        for i in range(n):
            L = legs[i]; k=f"{i+1:02d}"
            data[f"Leg{k}_Waypoint"]            = str(L["A"]["name"])
            data[f"Leg{k}_Navaid_Identifier"]   = ""
            data[f"Leg{k}_Navaid_Frequency"]    = ""
            data[f"Leg{k}_Altitude_FL"]         = f"{int(round(L['B']['alt']))}"
            data[f"Leg{k}_True_Course"]         = f"{int(round(L['TC']))}"
            data[f"Leg{k}_Ground_Speed"]        = f"{int(round(L['GS']))}"
            data[f"Leg{k}_Leg_Distance"]        = f"{L['Dist']:.1f}"
            data[f"Leg{k}_Leg_ETE"]             = mmss(L["time_sec"])
            data[f"Leg{k}_ETO"]                 = L["clock_end"]
            data[f"Leg{k}_Planned_Burnoff"]     = f"{L['burn']:.1f}"
            data[f"Leg{k}_Estimated_FOB"]       = f"{L['efob_end']:.1f}"
            data[f"Leg{k}_True_Heading"]        = f"{int(round(L['TH']))}"
            data[f"Leg{k}_Magnetic_Heading"]    = f"{int(round(L['MH']))}"
            data[f"Leg{k}_True_Airspeed"]       = f"{int(round(L['TAS']))}"
            data[f"Leg{k}_Cumulative_Distance"] = f"{L['cum_dist']:.1f}"
            data[f"Leg{k}_Cumulative_ETE"]      = mmss(L['cum_time'])
        return data, n

    # Mapeamento EXACTO — NAVLOG_FORM_1.pdf (continuação) — Leg12..Leg23
    def build_fields_cont(legs, start_idx):
        legs = with_cumulative(legs)
        data = {"TEMP_ISA_DEV": temp_isa, "FLIGHT_LEVEL_ALTITUDE": ""}  # campos do teu PDF 1
        filled = 0
        for i in range(start_idx, len(legs)):
            L = legs[i]; leg_no = i+1
            if leg_no>23: break
            k=f"{leg_no:02d}"
            data[f"Leg{k}_Waypoint"]            = str(L["A"]["name"])
            data[f"Leg{k}_Navaid_Identifier"]   = ""
            data[f"Leg{k}_Navaid_Frequency"]    = ""
            data[f"Leg{k}_Altitude_FL"]         = f"{int(round(L['B']['alt']))}"
            data[f"Leg{k}_True_Course"]         = f"{int(round(L['TC']))}"
            data[f"Leg{k}_Ground_Speed"]        = f"{int(round(L['GS']))}"
            data[f"Leg{k}_Leg_Distance"]        = f"{L['Dist']:.1f}"
            data[f"Leg{k}_Leg_ETE"]             = mmss(L["time_sec"])
            data[f"Leg{k}_ETO"]                 = L["clock_end"]
            data[f"Leg{k}_Planned_Burnoff"]     = f"{L['burn']:.1f}"
            data[f"Leg{k}_Estimated_FOB"]       = f"{L['efob_end']:.1f}"
            data[f"Leg{k}_True_Heading"]        = f"{int(round(L['TH']))}"
            data[f"Leg{k}_Magnetic_Heading"]    = f"{int(round(L['MH']))}"
            data[f"Leg{k}_True_Airspeed"]       = f"{int(round(L['TAS']))}"
            data[f"Leg{k}_Cumulative_Distance"] = f"{L['cum_dist']:.1f}"
            data[f"Leg{k}_Cumulative_ETE"]      = mmss(L['cum_time'])
            filled += 1
        return data, filled

    st.markdown("**Templates:** " +
        ("✅ encontrados" if (os.path.exists(TEMPLATE_MAIN) and os.path.exists(TEMPLATE_CONT)) else "⚠️ coloca `NAVLOG_FORM.pdf` e `NAVLOG_FORM_1.pdf` na mesma pasta do app."))

    if st.button("📄 Gerar PDFs", type="primary", use_container_width=True):
        if not st.session_state.legs:
            st.warning("Gera a rota primeiro.")
        elif not (os.path.exists(TEMPLATE_MAIN) and os.path.exists(TEMPLATE_CONT)):
            st.error("Templates não encontrados.")
        else:
            # Capa
            fields_main, used_main = build_fields_main(st.session_state.legs)
            pdf_main = fill_pdf_fields_exact(TEMPLATE_MAIN, fields_main)
            st.download_button("⬇️ navlog_main_filled.pdf", data=pdf_main, file_name="navlog_main_filled.pdf", mime="application/pdf")

            # Continuação (se necessário)
            if len(st.session_state.legs) > used_main:
                fields_cont, used_cont = build_fields_cont(st.session_state.legs, used_main)
                if used_cont>0:
                    pdf_cont = fill_pdf_fields_exact(TEMPLATE_CONT, fields_cont)
                    st.download_button("⬇️ navlog_cont_filled.pdf", data=pdf_cont, file_name="navlog_cont_filled.pdf", mime="application/pdf")
                else:
                    st.info("Todas as legs couberam no primeiro PDF.")
            else:
                st.info("Todas as legs couberam no primeiro PDF.")
