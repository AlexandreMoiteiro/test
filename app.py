# app.py — NAVLOG (Folium/Leaflet) — VFR-ish, TOC/TOD como WPs, seleção melhorada
# - TAS: 70 (climb) / 90 (cruise & descent) kt
# - FF: 20 L/h
# - TOC/TOD inseridos como novos WPs e legs partidas
# - Mapa Folium: vários estilos (satélite/topo/OSM), FULLSCREEN (para print)
# - Dog houses triangulares; MH grande (amarelo) com halo
# - Riscas a cada 2 min usando GS da leg
# - UI de pesquisa com TABELA + checkboxes (nada de multiselect confuso)
# - Sem experimental_rerun e sem RPM

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MeasureControl
from math import sin, asin, radians, degrees

# ===================== CONSTANTES =====================
CLIMB_TAS   = 70.0
CRUISE_TAS  = 90.0
DESCENT_TAS = 90.0
FUEL_FLOW   = 20.0
EARTH_NM    = 3440.065

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG — Folium (VFR) + TOC/TOD", layout="wide", initial_sidebar_state="collapsed")
CSS = """
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:var(--line);margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid var(--line);padding-bottom:8px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===================== HELPERS =====================
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
hhmmss = lambda t: f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
rang = lambda x: int(round(float(x))) % 360
wrap360 = lambda x: (x % 360 + 360) % 360
def angdiff(a, b): return (a - b + 180) % 360 - 180

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas,1e-9)))
    wca = degrees(math.asin(s))
    th  = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False):
    return wrap360(th - var if east_is_neg else th + var)

# geodesia
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

def triangle_coords(lat, lon, heading_deg, h_nm=0.95, w_nm=0.7):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [(bl_lat, bl_lon), (apex_lat, apex_lon), (br_lat, br_lon)]

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (topo VFR-ish)")
ens("maptiler_key", "")
ens("search_results", None)   # DataFrame persistente para a tabela
ens("search_q", "")

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,3,2,2])
with h1: st.title("NAVLOG — Folium (VFR) + TOC/TOD")
with h2: st.caption("TAS 70/90/90 · FF 20 L/h · Fullscreen para print")
with h3:
    if st.button("➕ Novo waypoint", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.route_nodes = []
st.markdown("</div>", unsafe_allow_html=True)

# ===================== PARÂMETROS =====================
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
        st.session_state.desc_angle= st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
    with c4:
        st.session_state.start_efob= st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default))
    b1, b2 = st.columns([2,2])
    with b1:
        st.session_state.map_base = st.selectbox(
            "Base do mapa",
            ["OpenTopoMap (topo VFR-ish)",
             "EOX Sentinel-2 (satélite)",
             "Esri World Imagery + Places",
             "Stamen Terrain (topo)",
             "Stadia Outdoors (topo)",
             "OSM Standard",
             "MapTiler Satellite Hybrid (key)"],
            index=0
        )
    with b2:
        if "MapTiler" in st.session_state.map_base:
            st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key)
    st.form_submit_button("Aplicar")
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== CSVs (locais) =====================
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
    ad_raw  = pd.read_csv(AD_CSV)
    loc_raw = pd.read_csv(LOC_CSV)
    ad_df  = parse_ad_df(ad_raw)
    loc_df = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

# ===================== PESQUISA + SELEÇÃO (TABELA COM CHECKBOX) =====================
st.subheader("Adicionar waypoints por pesquisa (CSV local)")
with st.form("search_form", clear_on_submit=False):
    c1,c2,c3 = st.columns([3,1.2,1.2])
    with c1:
        st.session_state.search_q = st.text_input("Pesquisar (ex.: LPPT, NISA, ABRANTES…)", st.session_state.search_q)
    with c2:
        alt_wp = st.number_input("Altitude para novos WPs (ft)", 0.0, 18000.0, 3000.0, step=100.0)
    with c3:
        do_search = st.form_submit_button("🔎 Procurar")
# congela resultados numa DataFrame persistente
if do_search:
    q = st.session_state.search_q.strip().lower()
    res = pd.concat([ad_df, loc_df], ignore_index=True)
    if q:
        res = res[res.apply(lambda r: any(q in str(v).lower() for v in r.values), axis=1)]
    res = res[["src","code","name","lat","lon"]].copy()
    res.insert(0, "add", False)
    st.session_state.search_results = res.reset_index(drop=True)

if st.session_state.search_results is not None:
    st.caption("Marca os registos que queres adicionar e clica em “Adicionar selecionados”.")
    edited = st.data_editor(
        st.session_state.search_results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Adicionar"),
            "src": st.column_config.TextColumn("Fonte"),
            "code": st.column_config.TextColumn("Código"),
            "name": st.column_config.TextColumn("Nome"),
            "lat": st.column_config.NumberColumn("Lat", format="%.5f"),
            "lon": st.column_config.NumberColumn("Lon", format="%.5f"),
        },
        num_rows="fixed"
    )
    st.session_state.search_results = edited
    if st.button("➕ Adicionar selecionados"):
        picks = edited[edited["add"] == True]
        for _, r in picks.iterrows():
            st.session_state.wps.append({"name": str(r["code"] or r["name"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
        st.success(f"Adicionados {len(picks)} WPs.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== EDITOR DE WPs =====================
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([2,2,2,1,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                up = st.button("↑", key=f"up{i}")
                dn = st.button("↓", key=f"dn{i}")
                if up and i>0:
                    st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
                if dn and i < len(st.session_state.wps)-1:
                    st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== TOC/TOD COMO WPs =====================
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, desc_angle_deg):
    nodes = []
    if len(user_wps) < 2: return nodes
    for i in range(len(user_wps)-1):
        A, B = user_wps[i], user_wps[i+1]
        nodes.append(A)
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        _, _, gs_cl = wind_triangle(tc, CLIMB_TAS,   wind_from, wind_kt)
        _, _, gs_de = wind_triangle(tc, DESCENT_TAS, wind_from, wind_kt)

        if B["alt"] > A["alt"]:  # CLIMB → TOC
            dh = B["alt"] - A["alt"]
            t_need = dh / max(roc_fpm, 1)             # min
            d_need = gs_cl * (t_need/60.0)            # nm
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:  # DESCENT → TOD
            rod_fpm = max(100.0, gs_de * 5.0 * (desc_angle_deg/3.0))
            dh = A["alt"] - B["alt"]
            t_need = dh / max(rod_fpm, 1)             # min
            d_need = gs_de * (t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
    nodes.append(user_wps[-1])
    return nodes

# ===================== LEGS A PARTIR DOS NODES =====================
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

        efob_start = carry_efob
        efob_end   = max(0.0, r10f(efob_start - burn))

        clk_start = (base_time + dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor)}"
        clk_end   = (base_time + dt.timedelta(seconds=t_cursor+time_sec)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor+time_sec)}"

        # CPs (texto)
        cps = []
        if ck_every_min>0 and gs>0:
            k=1
            while k*ck_every_min*60 <= time_sec:
                t=k*ck_every_min*60
                d=gs*(t/3600.0)
                eto=(base_time + dt.timedelta(seconds=t_cursor+t)).strftime('%H:%M') if base_time else ""
                efob=max(0.0, r10f(efob_start - FUEL_FLOW*(t/3600.0)))
                cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto,"efob":efob})
                k+=1

        legs.append({
            "i":i+1, "A":A, "B":B, "profile":profile,
            "TC":tc, "TH":th, "MH":mh, "TAS":tas, "GS":gs,
            "Dist":dist, "time_sec":time_sec, "burn":r10f(burn),
            "efob_start":efob_start, "efob_end":efob_end,
            "clock_start":clk_start, "clock_end":clk_end, "cps":cps
        })
        t_cursor += time_sec
        carry_efob = efob_end
    return legs

# ===================== GERAR ROTA/LEGS =====================
cg1, cg2 = st.columns([2,6])
with cg1:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(
            st.session_state.wps,
            st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.roc_fpm, st.session_state.desc_angle
        )
        st.session_state.legs = build_legs_from_nodes(
            st.session_state.route_nodes,
            st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.mag_var, st.session_state.mag_is_e,
            st.session_state.ck_default
        )

# ===================== RESUMO LEGS =====================
if st.session_state.legs:
    total_sec  = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final = st.session_state.legs[-1]["efob_end"]
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b> (20 L/h)</div>"
        + f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>"
        + "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    for L in st.session_state.legs:
        with st.expander(f"Leg {L['i']}: {L['A']['name']} → {L['B']['name']}  —  {L['profile']}", expanded=True):
            st.markdown(
                "<div class='kvrow'>"
                + f"<div class='kv'>TH/MH: <b>{rang(L['TH'])}T / {rang(L['MH'])}M</b></div>"
                + f"<div class='kv'>GS/TAS: <b>{rint(L['GS'])}/{rint(L['TAS'])} kt</b></div>"
                + f"<div class='kv'>Dist: <b>{L['Dist']:.1f} nm</b></div>"
                + f"<div class='kv'>Tempo: <b>{mmss(L['time_sec'])}</b></div>"
                + f"<div class='kv'>FF: <b>20 L/h</b> · Burn: <b>{L['burn']:.1f} L</b></div>"
                + "</div>", unsafe_allow_html=True
            )
            st.markdown(f"**Relógio** — {L['clock_start']} → {L['clock_end']}  |  **EFOB** — {L['efob_start']:.1f} → {L['efob_end']:.1f} L")

# ===================== MAPA (FOLIUM / LEAFLET) =====================
def _bounds_from_nodes(nodes):
    lats = [n["lat"] for n in nodes]; lons = [n["lon"] for n in nodes]
    sw = (min(lats), min(lons)); ne = (max(lats), max(lons))
    return [sw, ne]

def _route_latlngs(legs):
    return [ [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])] for L in legs ]

def _add_text(map_obj, lat, lon, text, size_px=22, color="#FFD700", offset_px=(0,0), bold=True, pane='labels'):
    weight = "700" if bold else "400"
    halo = "text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;"
    html = f"""
    <div style="font-size:{size_px}px;color:{color};font-weight:{weight};
                {halo}white-space:nowrap;transform:translate({offset_px[0]}px,{offset_px[1]}px);">
      {text}
    </div>"""
    icon = folium.DivIcon(html=html, icon_size=(0,0), icon_anchor=(0,0))
    folium.Marker(location=(lat,lon), icon=icon, draggable=False, pane=pane).add_to(map_obj)

def render_map_folium(nodes, legs, base_choice="OpenTopoMap (topo VFR-ish)", maptiler_key=""):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em *Gerar/Atualizar rota*.")
        return

    mean_lat = sum([n["lat"] for n in nodes])/len(nodes)
    mean_lon = sum([n["lon"] for n in nodes])/len(nodes)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=9, tiles=None, control_scale=True, prefer_canvas=True)

    # ---- BASES ----
    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer(
            tiles="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
            attr='© EOX & Sentinel-2', name="EOX Sat"
        ).add_to(m)
        folium.TileLayer(
            tiles="https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png",
            attr='© EOX overlay', name="Labels", overlay=True, opacity=1
        ).add_to(m)
    elif base_choice == "Esri World Imagery + Places":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Esri Imagery").add_to(m)
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Places", overlay=True, opacity=1).add_to(m)
    elif base_choice == "Stamen Terrain (topo)":
        folium.TileLayer("https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
                         attr="Map tiles by Stamen", name="Stamen Terrain").add_to(m)
    elif base_choice == "Stadia Outdoors (topo)":
        folium.TileLayer("https://tiles.stadiamaps.com/tiles/outdoors/{z}/{x}/{y}.png",
                         attr="© Stadia Maps © OpenMapTiles © OSM", name="Stadia Outdoors").add_to(m)
    elif base_choice == "OSM Standard":
        folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                         attr="© OpenStreetMap", name="OSM").add_to(m)
    elif base_choice == "MapTiler Satellite Hybrid (key)" and maptiler_key:
        folium.TileLayer(f"https://api.maptiler.com/maps/hybrid/256/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_key}",
                         attr="© MapTiler", name="MapTiler Hybrid").add_to(m)
    else:  # OpenTopoMap por defeito
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                         attr="© OpenTopoMap (CC-BY-SA)", name="OpenTopoMap").add_to(m)

    # ---- PANES / Z-INDEX ----
    folium.map.Pane('route',  z_index=650).add_to(m)
    folium.map.Pane('ticks',  z_index=655).add_to(m)
    folium.map.Pane('dog',    z_index=660).add_to(m)
    folium.map.Pane('labels', z_index=670).add_to(m)
    folium.map.Pane('wps',    z_index=680).add_to(m)

    # ---- ROTA COM HALO ----
    for latlngs in _route_latlngs(legs):
        folium.PolyLine(latlngs, color="#ffffff", weight=8, opacity=0.9, pane='route').add_to(m)
        folium.PolyLine(latlngs, color="#C000FF", weight=4, opacity=1.0, pane='route').add_to(m)

    # ---- RISCAS 2 MIN ----
    for L in legs:
        if L["GS"]<=0 or L["time_sec"]<=0: continue
        k, step = 1, 120
        while k*step <= L["time_sec"]:
            t = k*step
            d = min(L["Dist"], L["GS"]*(t/3600.0))
            latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
            left_lat, left_lon   = dest_point(latm, lonm, L["TC"]-90, 0.15)
            right_lat, right_lon = dest_point(latm, lonm, L["TC"]+90, 0.15)
            folium.PolyLine([(left_lat,left_lon),(right_lat,right_lon)],
                            color="#000", weight=2, opacity=1, pane='ticks').add_to(m)
            k += 1

    # ---- DOG HOUSES + RÓTULOS ----
    side = 1
    for L in legs:
        mid_lat, mid_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], L["Dist"]/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, L["TC"]+side*90, 0.38)
        tri = triangle_coords(off_lat, off_lon, L["TC"], h_nm=0.95, w_nm=0.7)
        folium.Polygon(tri, color="#000", weight=2, fill=True,
                       fill_color="#FFF", fill_opacity=0.92, pane='dog').add_to(m)

        _add_text(m, off_lat, off_lon, f"MH {rang(L['MH'])}°", size_px=28,
                  color="#FFD700", offset_px=(0,-2), pane='labels')
        _add_text(m, off_lat, off_lon,
                  f"{rang(L['TH'])}T • {rint(L['GS'])}kt • {mmss(L['time_sec'])} • {L['Dist']:.1f}nm",
                  size_px=15, color="#000000", offset_px=(side*90,16), pane='labels')
        side *= -1

    # ---- WAYPOINTS ----
    for idx, N in enumerate(nodes):
        is_tt = str(N["name"]).startswith(("TOC","TOD"))
        color = "#FF5050" if is_tt else "#007AFF"
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#FFFFFF", weight=2,
                            fill=True, fill_color=color, fill_opacity=1, pane='wps').add_to(m)
        _add_text(m, N["lat"], N["lon"], f"{idx+1}. {N['name']}",
                  size_px=14, color="#FFFFFF", offset_px=(0,-22), pane='labels')

    # ---- CONTROLOS (FULLSCREEN p/ PRINT) ----
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit='nauticalmiles').add_to(m)

    # vista
    try:
        m.fit_bounds(_bounds_from_nodes(nodes), padding=(30,30))
    except Exception:
        pass

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=720)

# ---- desenhar mapa ----
if st.session_state.wps:
    if not st.session_state.route_nodes:
        st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar legs.")
    else:
        render_map_folium(
            st.session_state.route_nodes,
            st.session_state.legs,
            base_choice=st.session_state.map_base,
            maptiler_key=st.session_state.maptiler_key
        )
else:
    st.info("Adiciona pelo menos 2 waypoints para começares.")
