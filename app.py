# NAVLOG v12 — VFR map + TOC/TOD como WPs + TAS/FF fixas + seleção de WPs
# - TAS fixas: 70 kt (climb), 90 kt (cruise/desc)
# - FF fixa: 20 L/h
# - TOC/TOD geram waypoints e partem as legs
# - Mapa VFR-like com TileLayer (OpenTopoMap/OSM), dog houses, riscas 2min por GS
# - Pesquisa com seleção (resolve "NISA" duplicado)
# - Sem experimental_rerun; sem RPMs

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG v12 — VFR", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.badge{background:#eef6ff;border:1px solid #bfe0ff;color:#0b62ff;border-radius:999px;padding:2px 10px;font-weight:600}
.sep{height:1px;background:var(--line);margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid var(--line);padding-bottom:8px}
.tl{position:relative;margin:8px 0 18px 0;padding-bottom:46px}
.tl .bar{height:6px;background:#eef1f5;border-radius:3px}
.tl .tick{position:absolute;top:10px;width:2px;height:14px;background:#333}
.tl .cp-lbl{position:absolute;top:32px;transform:translateX(-50%);text-align:center;font-size:11px;color:#333;white-space:nowrap}
.tl .tocdot,.tl .toddot{position:absolute;top:-6px;width:14px;height:14px;border-radius:50%;transform:translateX(-50%);border:2px solid #fff;box-shadow:0 0 0 2px rgba(0,0,0,0.15)}
.tl .tocdot{background:#1f77b4}.tl .toddot{background:#d62728}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===================== CONSTANTES SIMPLES =====================
TAS_CLIMB = 70.0     # kt
TAS_CRUISE = 90.0    # kt
TAS_DESC = 90.0      # kt
FF_FIXA = 20.0       # L/h para todas as fases
DESC_ANG_GRAUS = 3.0 # pode ser ajustado na UI
EARTH_NM = 3440.065

# ===================== UTILS =====================
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang = lambda x: int(round(float(x))) % 360
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
wrap360 = lambda x: (float(x) % 360.0 + 360.0) % 360.0
def angdiff(a, b): return (a - b + 180) % 360 - 180

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc)); cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = degrees(asin(s)); th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(true_hdg, var_deg, east_is_neg=False):
    # se var é E e east_is_neg=True -> subtrai (padrão PT: E negativo)
    return wrap360(true_hdg - var_deg if east_is_neg else true_hdg + var_deg)

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
    return dest_point(lat1, lon1, tc0, min(max(dist_from_start_nm,0.0), total))

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15)
ens("mag_var", 1.0); ens("mag_is_e", True)       # E negativo por defeito em PT
ens("desc_angle", DESC_ANG_GRAUS)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("computed", [])
ens("toc_tod_points", [])   # lista de WPs auto-gerados

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,2,3,2])
with h1: st.title("NAVLOG — v12 (VFR)")
with h2: st.toggle("Mostrar TIMELINE/CPs", key="show_timeline", value=st.session_state.show_timeline)
with h3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.computed = []; st.session_state.toc_tod_points=[]
st.markdown("</div>", unsafe_allow_html=True)

# ===================== PARÂMETROS GLOBAIS =====================
with st.form("globals"):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, int(st.session_state.qnh))
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, int(st.session_state.oat))
    with p2:
        st.session_state.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
    with p3:
        st.session_state.mag_var  = st.number_input("Variação magnética (°)", -30.0, 30.0, float(st.session_state.mag_var), step=0.1)
        st.session_state.mag_is_e = st.toggle("Tratar Este (E) como negativo", value=st.session_state.mag_is_e)
    with p4:
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
        st.session_state.ck_default = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default), step=1)
    w1, w2 = st.columns(2)
    with w1: st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
    with w2: st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    st.form_submit_button("Aplicar parâmetros")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== PARSE CSVs LOCAIS =====================
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
            rows.append({"type":"AD","code":ident or name, "name":name, "city":city,"lat":lat,"lon":lon,"alt":0.0})
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
            rows.append({"type":"LOC","code":code or name, "name":name, "sector":sector,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

try:
    ad_raw  = pd.read_csv(AD_CSV)
    loc_raw = pd.read_csv(LOC_CSV)
    ad_df  = parse_ad_df(ad_raw)
    loc_df = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

def filter_df(df, q):
    if not q: return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

# ===================== PESQUISA COM SELEÇÃO (resolve duplicados) =====================
cflt1, cflt2, cbtn = st.columns([4,3,2])
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade", "", placeholder="Ex: LPPT, ABRANTES, NISA…")
with cflt2: alt_wp = st.number_input("Altitude p/ WPs (ft)", 0.0, 18000.0, 3000.0, step=100.0)
sel_opts = []

if qtxt:
    ad_f, loc_f = filter_df(ad_df, qtxt), filter_df(loc_df, qtxt)
    res = pd.concat([ad_f, loc_f], ignore_index=True)
    if len(res):
        st.caption(f"{len(res)} resultados. Seleciona quais queres adicionar:")
        # multiselect com etiqueta amigável
        for i, r in res.iterrows():
            label = f"{r['code']} — {r.get('name','')}  ({r['lat']:.4f}, {r['lon']:.4f})"
            sel_opts.append((label, float(r['lat']), float(r['lon']), str(r['code'])))
        labels = [x[0] for x in sel_opts]
        pick = st.multiselect("Escolha um ou vários", labels, default=[])
        if st.button("Adicionar selecionados"):
            selset = set(pick)
            for label, lat, lon, code in sel_opts:
                if label in selset:
                    st.session_state.wps.append({"name": code, "lat": lat, "lon": lon, "alt": float(alt_wp)})
            st.success(f"Adicionados {len(selset)} WPs.")
    else:
        st.info("Sem resultados.")

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

# ===================== BUILD LEGS (com split em TOC/TOD) =====================
def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

def build_leg(A, B, ck_min):
    """Constrói 1..2 legs entre A->B com possível TOC/TOD como novo WP + fases."""
    TC = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
    Dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
    alt0, alt1 = A["alt"], B["alt"]

    wfrom, wkt = st.session_state.wind_from, st.session_state.wind_kt
    var, varE  = st.session_state.mag_var, st.session_state.mag_is_e
    angle      = st.session_state.desc_angle

    profile = "LEVEL" if abs(alt1-alt0)<1e-6 else ("CLIMB" if alt1>alt0 else "DESCENT")

    # TAS fixas
    tas_c = TAS_CLIMB; tas_r = TAS_CRUISE; tas_d = TAS_DESC
    _, THc, GScl = wind_triangle(TC, tas_c, wfrom, wkt)
    _, THr, GScr = wind_triangle(TC, tas_r, wfrom, wkt)
    _, THd, GSde = wind_triangle(TC, tas_d, wfrom, wkt)
    MHc = apply_var(THc, var, varE); MHr = apply_var(THr, var, varE); MHd = apply_var(THd, var, varE)

    # ROD pela regra 3°: ~ GS*5 * (ângulo/3)
    ROD = max(100.0, GSde * 5.0 * (angle/3.0))
    # ROC simplificado — usa 600fpm por defeito (mais estável do que tabelas)
    ROC = 600.0

    legs_out = []
    toc_tod_wp = None

    if profile == "CLIMB":
        t_need_min = (alt1 - alt0)/ROC
        d_need = GScl * (t_need_min/60.0)
        if d_need < Dist:
            # WP TOC
            lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
            toc_tod_wp = {"name": f"TOC L{A['name']}→{B['name']}", "lat": lat_toc, "lon": lon_toc, "alt": alt1, "kind": "TOC"}
            # leg 1: A -> TOC (climb)
            legs_out.append({"TC":TC,"Dist":d_need,"Alt0":alt0,"Alt1":alt1,"phase":"CLIMB",
                             "TH":THc,"MH":MHc,"GS":GScl,"TAS":tas_c,"CK":ck_min})
            # leg 2: TOC -> B (cruise)
            legs_out.append({"TC":TC,"Dist":Dist-d_need,"Alt0":alt1,"Alt1":alt1,"phase":"CRUISE",
                             "TH":THr,"MH":MHr,"GS":GScr,"TAS":tas_r,"CK":ck_min})
        else:
            # não atinge
            legs_out.append({"TC":TC,"Dist":Dist,"Alt0":alt0,"Alt1":alt0 + ROC*(Dist/GScl)*60.0,"phase":"CLIMB",
                             "TH":THc,"MH":MHc,"GS":GScl,"TAS":tas_c,"CK":ck_min})
    elif profile == "DESCENT":
        t_need_min = (alt0 - alt1)/ROD
        d_need = GSde * (t_need_min/60.0)
        if d_need < Dist:
            # WP TOD (começa descida a d_need antes de B)
            lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], Dist - d_need)
            toc_tod_wp = {"name": f"TOD L{A['name']}→{B['name']}", "lat": lat_tod, "lon": lon_tod, "alt": alt0, "kind": "TOD"}
            # leg 1: A -> TOD (cruise)
            legs_out.append({"TC":TC,"Dist":Dist-d_need,"Alt0":alt0,"Alt1":alt0,"phase":"CRUISE",
                             "TH":THr,"MH":MHr,"GS":GScr,"TAS":tas_r,"CK":ck_min})
            # leg 2: TOD -> B (descent)
            legs_out.append({"TC":TC,"Dist":d_need,"Alt0":alt0,"Alt1":alt1,"phase":"DESCENT",
                             "TH":THd,"MH":MHd,"GS":GSde,"TAS":tas_d,"CK":ck_min})
        else:
            legs_out.append({"TC":TC,"Dist":Dist,"Alt0":alt0,"Alt1":max(0.0, alt0 - ROD*(Dist/GSde)*60.0),"phase":"DESCENT",
                             "TH":THd,"MH":MHd,"GS":GSde,"TAS":tas_d,"CK":ck_min})
    else:
        legs_out.append({"TC":TC,"Dist":Dist,"Alt0":alt0,"Alt1":alt0,"phase":"CRUISE",
                         "TH":THr,"MH":MHr,"GS":GScr,"TAS":tas_r,"CK":ck_min})

    return legs_out, toc_tod_wp

def recompute():
    st.session_state.legs = []
    st.session_state.computed = []
    st.session_state.toc_tod_points = []

    if len(st.session_state.wps) < 2: return

    # Gerar legs com split em TOC/TOD
    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        legs2, tt = build_leg(A, B, st.session_state.ck_default)
        st.session_state.legs.extend(legs2)
        if tt: st.session_state.toc_tod_points.append(tt)

    # Calcular tempos/burn e linhas de CPs
    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None

    carry_efob = float(st.session_state.start_efob)
    clock = base_time

    for leg in st.session_state.legs:
        # tempo e burn (FF fixa)
        sec = rt10((leg["Dist"] / max(leg["GS"],1e-9)) * 3600)
        burn = FF_FIXA * (sec/3600.0)
        # checkpoints
        cps=[]; t=0; cp_every=leg["CK"]
        while cp_every>0 and t+cp_every*60 <= sec:
            t += cp_every*60
            d = leg["GS"]*(t/3600.0)
            eto = (clock + dt.timedelta(seconds=t)).strftime('%H:%M') if clock else ""
            efob = max(0.0, r10f(carry_efob - FF_FIXA*(t/3600.0)))
            cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto,"efob":efob})

        start_lbl = (clock.strftime('%H:%M') if clock else f"T+{mmss(0)}")
        end_lbl   = ((clock+dt.timedelta(seconds=sec)).strftime('%H:%M') if clock else f"T+{mmss(sec)}")

        st.session_state.computed.append({
            "leg": leg,
            "sec": sec,
            "burn": r10f(burn),
            "clock_start": start_lbl,
            "clock_end": end_lbl,
            "efob_start": carry_efob,
            "efob_end": max(0.0, r10f(carry_efob - burn)),
            "cps": cps
        })

        carry_efob = max(0.0, r10f(carry_efob - burn))
        if clock: clock = clock + dt.timedelta(seconds=sec)

# ===================== GERAR / ATUALIZAR LEGS =====================
cgl1, cgl2 = st.columns([6,2])
with cgl1:
    st.caption("As legs serão geradas pela ordem dos WPs. Se existir TOC/TOD, a perna é automaticamente partida e o novo WP é marcado.")
with cgl2:
    if st.button("Gerar/Atualizar legs a partir dos WAYPOINTS", type="primary", use_container_width=True):
        recompute()

# ===================== LISTA DE LEGS + FASES =====================
if st.session_state.legs:
    total_sec_all=0; total_burn_all=0.0; efob_final=None
    for idx, leg in enumerate(st.session_state.legs):
        comp = st.session_state.computed[idx]
        with st.expander(f"Leg {idx+1} — {leg['phase']}  ({r10f(leg['Dist'])} nm)", expanded=True):
            left, right = st.columns([3,2])
            with left:
                st.markdown(
                    "<div class='kvrow'>"
                    + f"<div class='kv'>Alt: <b>{int(leg['Alt0'])}→{int(leg['Alt1'])} ft</b></div>"
                    + f"<div class='kv'>TC/MH: <b>{rang(leg['TC'])}T / {rang(leg['MH'])}M</b></div>"
                    + f"<div class='kv'>GS/TAS: <b>{rint(leg['GS'])}/{rint(leg['TAS'])} kt</b></div>"
                    + f"<div class='kv'>FF: <b>{rint(FF_FIXA)} L/h</b></div>"
                    + "</div>", unsafe_allow_html=True
                )
            with right:
                st.metric("Tempo", mmss(comp["sec"]))
                st.metric("Fuel (fase)", f"{comp['burn']:.1f} L")

            r1, r2, r3 = st.columns(3)
            with r1: st.markdown(f"**Relógio** — {comp['clock_start']} → {comp['clock_end']}")
            with r2: st.markdown(f"**EFOB** — Start {comp['efob_start']:.1f} L → End {comp['efob_end']:.1f} L")
            with r3:
                if leg["phase"]=="CLIMB": st.markdown(f"**Referência** — ROC ~600 fpm")
                elif leg["phase"]=="DESCENT": st.markdown(f"**Referência** — 3° (~5×GS)")

            if st.session_state.show_timeline and leg["GS"]>0:
                # timeline simples
                total = comp['sec']
                html = "<div class='tl'><div class='bar'></div>"
                for cp in comp['cps']:
                    pct = (cp['t']/total)*100.0
                    html += f"<div class='tick' style='left:{pct:.2f}%;'></div>"
                    html += f"<div class='cp-lbl' style='left:{pct:.2f}%;'><div>T+{cp['min']}m</div><div>{cp['nm']} nm</div><div>{cp['eto']}</div><div>EFOB {cp['efob']:.1f}</div></div>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

        total_sec_all += comp["sec"]; total_burn_all += comp["burn"]; efob_final = comp["efob_end"]

    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec_all)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{r10f(total_burn_all):.1f} L</b></div>"
        + (f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>" if efob_final is not None else "")
        + "</div>", unsafe_allow_html=True
    )
else:
    st.info("Adiciona pelo menos 2 WPs e clica em “Gerar/Atualizar legs…”")

# ===================== MAPA VFR (TileLayer) =====================
def calc_center_zoom(wps):
    if not wps: return (39.6,-8.0,7)
    lats=[w["lat"] for w in wps]; lons=[w["lon"] for w in wps]
    minlat,maxlat,minlon,maxlon = min(lats),max(lats),min(lons),max(lons)
    cenlat=(minlat+maxlat)/2; cenlon=(minlon+maxlon)/2
    # heurística simples de zoom pela extensão
    lat_span=maxlat-minlat; lon_span=maxlon-minlon
    span=max(lat_span, lon_span)
    if span<0.5: zoom=10
    elif span<1.0: zoom=9
    elif span<2.0: zoom=8
    elif span<4.0: zoom=7
    elif span<8.0: zoom=6
    else: zoom=5
    return (cenlat, cenlon, zoom)

if st.session_state.wps and st.session_state.legs:
    # tiles VFR-like
    basemap = st.selectbox("🗺️ Base map (VFR-like)", ["OpenTopoMap (default)","OpenStreetMap Standard"], index=0)
    if basemap.startswith("OpenTopoMap"):
        tile_url = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"
        max_zoom = 17
    else:
        tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        max_zoom = 19
    tile_layer = pdk.Layer(
        "TileLayer",
        data=tile_url,
        min_zoom=0,
        max_zoom=max_zoom,
        tile_size=256
    )

    # caminhos, riscas 2min, dog houses/captions
    path_data, tick_data, tri_data, mh_labels, aux_labels, wp_points = [], [], [], [], [], []

    # WPs (inclui os originais + TOC/TOD)
    for w in st.session_state.wps:
        wp_points.append({"pos":[w["lon"], w["lat"]], "name": w["name"]})
    for tt in st.session_state.toc_tod_points:
        wp_points.append({"pos":[tt["lon"], tt["lat"]], "name": tt["kind"]})

    # demarca as legs e elementos
    # criar pares de pontos para cada leg (usando progressão acumulada na rota original)
    # para path desenhamos simplesmente A->B conforme legs
    # para riscas de 2min usamos GS por leg
    # para doghouse usamos TC de cada leg e metas graficamente ao lado
    # como não guardamos o A/B da leg, reconstruímos por deslocamento ao longo da rota:
    # solução prática: usamos a distância e bearing a partir de um "start" virtual.
    # Em vez disso, calculamos a meia-distância real sobre os 2 WPs consecutivos na rota original.
    # Abordagem mais robusta: recalcular A/B de cada leg com base no encadeamento dos WPs+TOC/TOD que figuram no mapa.
    # Vamos construir uma lista "route_points" já com TOC/TOD injetados, para garantir consistência visual.

    # Reconstrói rota com TOC/TOD injetados
    route_pts = []
    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        route_pts.append(A)
        # existe TOC/TOD entre A e B?
        for tt in st.session_state.toc_tod_points:
            # verifica se o TT está entre A e B por proximidade ao grande círculo
            tc = gc_course_tc(A["lat"],A["lon"],B["lat"],B["lon"])
            dAB = gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
            dAT = gc_dist_nm(A["lat"],A["lon"],tt["lat"],tt["lon"])
            dTB = gc_dist_nm(tt["lat"],tt["lon"],B["lat"],B["lon"])
            if abs(dAT + dTB - dAB) < 0.2:  # tolerância 0.2 nm
                route_pts.append({"name":tt["kind"],"lat":tt["lat"],"lon":tt["lon"],"alt":A["alt"]})
    route_pts.append(st.session_state.wps[-1])

    # desenha paths (entre cada par consecutivo)
    for i in range(len(route_pts)-1):
        A = route_pts[i]; B = route_pts[i+1]
        path_data.append({"path": [[A["lon"],A["lat"]],[B["lon"],B["lat"]]], "name": f"{A['name']}→{B['name']}"})

    # Para riscas/doghouses/labels seguimos as legs calculadas (que já têm GS, TH, MH)
    for i, leg in enumerate(st.session_state.legs):
        # segmento geométrico: precisamos de um A->B que corresponda à leg i
        # reconstruímos avançando ao longo de route_pts pela distância da leg
        # acumula para achar posição inicial (s)
        s_prev = sum(l["Dist"] for l in st.session_state.legs[:i])
        # encontra o ponto na rota a distância s_prev e s_prev+Dist
        total_track = sum(gc_dist_nm(route_pts[j]["lat"],route_pts[j]["lon"],route_pts[j+1]["lat"],route_pts[j+1]["lon"]) for j in range(len(route_pts)-1))
        # mapeia distância ao longo do track em lat/lon
        def pos_at(snm):
            left = snm
            for j in range(len(route_pts)-1):
                A = route_pts[j]; B = route_pts[j+1]
                d = gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
                if left <= d:
                    return point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], left)
                left -= d
            return route_pts[-1]["lat"], route_pts[-1]["lon"]
        A_lat, A_lon = pos_at(s_prev)
        B_lat, B_lon = pos_at(s_prev + leg["Dist"])

        # riscas de 2 min
        if leg["GS"]>0:
            tot_sec = rt10((leg["Dist"]/leg["GS"])*3600)
            k=1; interval=120
            while k*interval <= tot_sec:
                t=k*interval; k+=1
                d=leg["GS"]*(t/3600.0)
                latm, lonm = point_along_gc(A_lat,A_lon,B_lat,B_lon,d)
                half_nm = 0.15
                left_lat,left_lon  = dest_point(latm, lonm, leg["TC"]-90, half_nm)
                right_lat,right_lon= dest_point(latm, lonm, leg["TC"]+90, half_nm)
                tick_data.append({"path":[[left_lon,left_lat],[right_lon,right_lat]]})

        # dog house deslocada para o lado direito
        mid_lat, mid_lon = point_along_gc(A_lat,A_lon,B_lat,B_lon, leg["Dist"]/2)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, leg["TC"]+90, 0.35)
        tri_data.append({"polygon": triangle_coords(off_lat, off_lon, leg["TC"])})

        # labels — MH grande (azulado) + resto em pequeno
        mh_labels.append({"position":[off_lon, off_lat], "text": f"MH {rang(leg['MH'])}°"})
        aux = f"{rang(leg['TH'])}T • {r10f(leg['Dist'])}nm • GS {rint(leg['GS'])} • ETE {mmss(rt10((leg['Dist']/leg['GS'])*3600))}"
        lab_lat, lab_lon = dest_point(off_lat, off_lon, leg["TC"]+90, 0.35)
        aux_labels.append({"position":[lab_lon, lab_lat], "text": aux})

    # LAYERS
    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[180, 0, 255, 220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0, 0, 0, 255], width_min_pixels=2)
    tri_layer   = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon",
                            get_fill_color=[255,255,255,235], get_line_color=[0,0,0,255],
                            line_width_min_pixels=2, stroked=True, filled=True)
    # MH em destaque
    mh_text  = pdk.Layer("TextLayer", data=mh_labels, get_position="position", get_text="text",
                         get_size=22, get_color=[20,110,255], get_alignment_baseline="'center'", billboard=True)
    aux_text = pdk.Layer("TextLayer", data=aux_labels, get_position="position", get_text="text",
                         get_size=14, get_color=[0,0,0], get_alignment_baseline="'center'", billboard=True)

    # WPs (pinos + nomes)
    wp_layer = pdk.Layer("ScatterplotLayer", data=wp_points, get_position="pos",
                         get_radius=1200, radius_min_pixels=5, get_fill_color=[255,90,0,220])
    wp_text  = pdk.Layer("TextLayer", data=wp_points, get_position="pos", get_text="name",
                         get_size=12, get_color=[0,0,0], get_alignment_baseline="'top'", billboard=True)

    lat0, lon0, zoom0 = calc_center_zoom(route_pts)
    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom0, pitch=0)

    deck = pdk.Deck(
        map_style=None,                 # sem mapbox / sem voyager
        initial_view_state=view_state,
        layers=[tile_layer, route_layer, ticks_layer, tri_layer, mh_text, aux_text, wp_layer, wp_text],
        tooltip={"text": "{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Gera as legs para ver a rota no mapa VFR.")
