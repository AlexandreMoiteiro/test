# app.py — NAVLOG v12 (VFR map + velocidades fixas + TOC/TOD = WPs)
# - TAS fixas: climb 70 kt, cruise 90 kt, descent 90 kt
# - FF fixa: 20 L/h
# - TOC/TOD inseridos como novos waypoints e legs são separadas
# - Mapa PyDeck com TileLayer OSM (estilo VFR-ish, sem Voyager)
# - Dog houses triangulares, MH grande e colorido
# - Riscas a cada 2 min pela GS
# - Pesquisa com seleção de waypoint (não adiciona tudo de uma vez)

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# ===================== CONSTANTES =====================
CLIMB_TAS   = 70.0   # kt
CRUISE_TAS  = 90.0   # kt
DESCENT_TAS = 90.0   # kt
FUEL_FLOW   = 20.0   # L/h (constante)

EARTH_NM = 3440.065

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG v12 — VFR + TOC/TOD", layout="wide", initial_sidebar_state="collapsed")
CSS = """
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.badge{background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:11px;margin-left:6px}
.sep{height:1px;background:var(--line);margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid var(--line);padding-bottom:8px}
.small{font-size:12px;color:#555}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===================== UTILS =====================
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
rang = lambda x: int(round(float(x))) % 360
wrap360 = lambda x: (x % 360 + 360) % 360

def angdiff(a, b): return (a - b + 180) % 360 - 180

def wind_triangle(tc, tas, wdir, wkt):
    # retorna (WCA, TH, GS)
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas,1e-9)))
    wca = degrees(math.asin(s))
    th  = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False):  # var: graus; E = negativo se east_is_neg=True
    return wrap360(th - var if east_is_neg else th + var)

# geodesia (NM)
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

def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600)             # ROC global default
ens("desc_angle", 3.0)          # graus
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])  # nodes = WPs + TOC/TOD gerados

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,3,2,2])
with h1: st.title("NAVLOG — v12 (VFR + TOC/TOD WPs)")
with h2:
    st.caption("Velocidades fixas · FF 20 L/h · OSM VFR map")
with h3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
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
    submitted = st.form_submit_button("Aplicar")
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== PARSE CSVs (locais) =====================
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
except:
    ad_df  = pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

# ===================== PESQUISA + SELEÇÃO =====================
cflt1, cflt2 = st.columns([3,1.5])
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, NISA…")
with cflt2: alt_wp = st.number_input("Altitude default (ft) p/ WPs adicionados", 0.0, 18000.0, 3000.0, step=100.0)

results = pd.concat([ad_df, loc_df])
if qtxt.strip():
    tq = qtxt.lower().strip()
    results = results[results.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

if not results.empty:
    options = []
    for idx, r in results.reset_index(drop=True).iterrows():
        label = f"[{r['src']}] {r.get('code','')} — {r.get('name','')} ({r['lat']:.4f}, {r['lon']:.4f})"
        options.append((idx, label))
    labels = [o[1] for o in options]
    selected = st.multiselect("Seleciona um ou mais resultados para adicionar:", labels, max_selections=None)
    if st.button("Adicionar selecionados"):
        sel_idx = [options[labels.index(s)][0] for s in selected]
        for _, r in results.iloc[sel_idx].iterrows():
            st.session_state.wps.append({"name": str(r.get("code") or r.get("name")), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
        st.success(f"Adicionados {len(sel_idx)} WPs.")
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

# ===================== CONSTRUIR NODES (TOC/TOD INSERIDOS) =====================
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, desc_angle_deg):
    nodes = []
    if len(user_wps) < 2: return nodes
    for i in range(len(user_wps)-1):
        A = user_wps[i]; B = user_wps[i+1]
        nodes.append(A)  # garante A
        tc  = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist= gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        _, th_cl, gs_cl = wind_triangle(tc, CLIMB_TAS,   wind_from, wind_kt)
        _, th_cr, gs_cr = wind_triangle(tc, CRUISE_TAS,  wind_from, wind_kt)
        _, th_de, gs_de = wind_triangle(tc, DESCENT_TAS, wind_from, wind_kt)

        if B["alt"] > A["alt"]:  # CLIMB primeiro → TOC medido a partir do início
            dh_ft = B["alt"] - A["alt"]
            t_need_min = dh_ft / max(roc_fpm,1)       # min
            d_need_nm  = gs_cl * (t_need_min/60.0)    # nm
            if d_need_nm < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need_nm)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:  # CRUISE + DESCENT no fim → TOD antes do destino
            rod_fpm = max(100.0, gs_de * 5.0 * (desc_angle_deg/3.0))
            dh_ft   = A["alt"] - B["alt"]
            t_need_min = dh_ft / max(rod_fpm,1)
            d_need_nm  = gs_de * (t_need_min/60.0)
            if d_need_nm < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need_nm)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
        # no else para LEVEL
    nodes.append(user_wps[-1])
    return nodes

# ===================== GERAR LEGS A PARTIR DOS NODES =====================
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
        A = nodes[i]; B = nodes[i+1]
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        profile = "LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile=="CLIMB" else (DESCENT_TAS if profile=="DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
        mh = apply_var(th, mag_var, st.session_state.mag_is_e)
        time_sec = rt10((dist / max(gs,1e-9)) * 3600.0) if gs>0 else 0
        burn = FUEL_FLOW * (time_sec/3600.0)
        efob_start = carry_efob
        efob_end   = max(0.0, r10f(efob_start - burn))
        clk_start = (base_time + dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor)}"
        clk_end   = (base_time + dt.timedelta(seconds=t_cursor+time_sec)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor+time_sec)}"

        # checkpoints textuais (se precisares)
        cps=[]
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

# ===================== BOTÃO: GERAR ROTA/LEGS =====================
cg1, cg2 = st.columns([2,6])
with cg1:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD e cria legs)", type="primary", use_container_width=True):
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

# ===================== MAPA VFR (OSM) =====================
def make_map(nodes, legs):
    if not nodes or not legs: 
        st.info("Adiciona pelo menos 2 WPs e carrega em *Gerar/Atualizar rota*.")
        return

    # --- data para layers ---
    # base route lines (cada leg)
    path_data = [{"path": [[L["A"]["lon"],L["A"]["lat"]],[L["B"]["lon"],L["B"]["lat"]]], "name": f"L{L['i']}"} for L in legs]

    # riscas 2 min (por leg, com GS da leg)
    tick_data = []
    for L in legs:
        if L["GS"]<=0 or L["time_sec"]<=0: continue
        interval = 120
        k=1
        while k*interval <= L["time_sec"]:
            t = k*interval
            d = min(L["Dist"], L["GS"]*(t/3600.0))
            latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
            half_nm = 0.15
            left_lat, left_lon   = dest_point(latm, lonm, L["TC"]-90, half_nm)
            right_lat, right_lon = dest_point(latm, lonm, L["TC"]+90, half_nm)
            tick_data.append({"path": [[left_lon,left_lat],[right_lon,right_lat]]})
            k+=1

    # dog houses + labels (MH grande)
    tri_data, mh_labels, info_labels = [], [], []
    for L in legs:
        mid_lat, mid_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], L["Dist"]/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, L["TC"]+90, 0.35)
        tri = triangle_coords(off_lat, off_lon, L["TC"], h_nm=0.9, w_nm=0.65)
        tri_data.append({"polygon": tri})

        # MH grande e colorido
        mh_text = f"MH {rang(L['MH'])}°"
        mh_pos  = dest_point(off_lat, off_lon, L["TC"]+90, 0.35)
        mh_labels.append({"position":[mh_pos[1], mh_pos[0]], "text": mh_text})

        # info mais pequena
        info = f"{rang(L['TH'])}T • {rint(L['GS'])}kt • {mmss(L['time_sec'])} • {L['Dist']:.1f}nm"
        info_pos = dest_point(off_lat, off_lon, L["TC"]-90, 0.15)
        info_labels.append({"position":[info_pos[1], info_pos[0]], "text": info})

    # waypoints markers (originais + TOC/TOD)
    wp_data = []
    for idx, N in enumerate(nodes):
        typ = "TOC/TOD" if str(N["name"]).startswith(("TOC","TOD")) else "WP"
        wp_data.append({"index":idx+1,"name":N["name"],"type":typ,"position":[N["lon"],N["lat"]]})

    # --- layers ---
    # Base map (OSM) — estilo VFR-ish, com nomes de localidades
    osm = pdk.Layer("TileLayer", data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0)

    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[180, 0, 255, 220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0, 0, 0, 255], width_min_pixels=2)

    tri_layer = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon",
                          get_fill_color=[255,255,255,230], get_line_color=[0,0,0,255],
                          line_width_min_pixels=2, stroked=True, filled=True)

    # MH destaque (amarelo forte, tamanho grande)
    mh_layer  = pdk.Layer("TextLayer", data=mh_labels, get_position="position", get_text="text",
                          get_size=22, get_color=[255,215,0], get_alignment_baseline="'center'")

    info_layer= pdk.Layer("TextLayer", data=info_labels, get_position="position", get_text="text",
                          get_size=14, get_color=[0,0,0], get_alignment_baseline="'center'")

    # Waypoints
    wp_layer  = pdk.Layer("ScatterplotLayer", data=wp_data, get_position="position",
                          get_radius=150, get_fill_color="[type=='TOC/TOD' ?  [255,80,80,220] : [0,122,255,220]]")
    wp_text   = pdk.Layer("TextLayer", data=[{"position":d["position"],"text":f"{d['index']}. {d['name']}"} for d in wp_data],
                          get_position="position", get_text="text", get_size=14, get_color=[20,20,20])

    mean_lat = sum([n["lat"] for n in nodes]) / len(nodes)
    mean_lon = sum([n["lon"] for n in nodes]) / len(nodes)

    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=7, pitch=0)

    deck = pdk.Deck(
        map_style=None,                # NÃO usar Voyager/Mapbox
        initial_view_state=view_state,
        layers=[osm, route_layer, ticks_layer, tri_layer, mh_layer, info_layer, wp_layer, wp_text],
        tooltip={"text":"{name}"}
    )
    st.pydeck_chart(deck)

# ---- render map ----
make_map(st.session_state.route_nodes if st.session_state.route_nodes else st.session_state.wps,
         st.session_state.legs)

