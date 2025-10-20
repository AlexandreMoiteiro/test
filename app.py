# NAVLOG v21 — Leaflet VFR — Dog houses com texto DENTRO — TOC/TOD WPs
# TAS: 70/90/90 — FF 20 L/h — riscas 2 min — seleção única de WPs
# Melhorias:
#  • Mapa 100% Leaflet (OTM, OSM HOT, Esri Topo, Esri Satélite híbrido, MapTiler opcional)
#  • Bug fix: payload JSON é lido DEPOIS de injetado (mapa agora renderiza)
#  • Pesquisa AD/Localidade com ranking, normalização sem acentos, deduplicação e filtros por tipo
#  • ETO por sub-leg (se hora off-blocks for dada)
#  • Sem experimental_rerun (evito reruns; ao subir/descer WP apenas altero a lista)

import streamlit as st
import pandas as pd
import math, re, datetime as dt, json, unicodedata
from math import sin, asin, radians, degrees
from streamlit.components.v1 import html as st_html

# ============ PAGE / STYLE ============
st.set_page_config(page_title="NAVLOG v21 — VFR (Leaflet)", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6;--mh:#d61f69}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin:12px 0;background:#fff}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:#e5e7eb;margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
</style>
""", unsafe_allow_html=True)

# ============ UTILS ============
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang   = lambda x: int(round(float(x))) % 360
rint   = lambda x: int(round(float(x)))
r10f   = lambda x: round(float(x), 1)
clamp  = lambda v, lo, hi: max(lo, min(hi, v))

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def wrap360(x): x = math.fmod(float(x), 360.0); return x + 360 if x < 0 else x
def angdiff(a, b): return (a - b + 180) % 360 - 180
def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc)); cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = degrees(asin(s)); th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs
apply_var = lambda th, var, east_is_neg=False: wrap360(th - var if east_is_neg else th + var)

EARTH_NM = 3440.065
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
    θ = math.radians(bearing_deg); δ = dist_nm / EARTH_NM
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

# ============ FIXOS ============
TAS_CLIMB, TAS_CRUISE, TAS_DESCENT = 70.0, 90.0, 90.0
FF_CONST = 20.0
press_alt  = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0
def interp1(x, x0, x1, y0, y1): 
    if x1==x0: return y0
    t=(x-x0)/(x1-x0); return y0 + t*(y1-y0)

ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
def roc_interp(pa, temp):
    pas = sorted(ROC_ENR.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    temps = [-25,0,25,50]; t = clamp(temp, temps[0], temps[-1])
    t0,t1 = (-25,0) if t<=0 else (0,25) if t<=25 else (25,50)
    v00,v01 = ROC_ENR[p0][t0], ROC_ENR[p0][t1]
    v10,v11 = ROC_ENR[p1][t0], ROC_ENR[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01); v1 = interp1(pa_c, p0, p1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * 0.90)

# ============ STATE ============
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("sublegs", [])
ens("leaf_base", "VFR híbrido (OTM + OSM labels)")
ens("maptiler_key", "")
ens("filters", {"AD": True, "LOC": True})

# ============ CSV parsing ============
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

def load_csvs():
    try:
        ad_raw  = pd.read_csv(AD_CSV); loc_raw = pd.read_csv(LOC_CSV)
        return parse_ad_df(ad_raw), parse_loc_df(loc_raw)
    except Exception:
        st.warning("Não foi possível ler os CSVs locais (AD/LOC).")
        return pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"]), pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])

ad_df, loc_df = load_csvs()

# ============ HEADER ============
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
a,b,c,d = st.columns([3,2.3,3,2])
with a: st.title("NAVLOG — v21 (Leaflet VFR)")
with b:
    st.session_state.leaf_base = st.selectbox(
        "Base do mapa",
        ["VFR híbrido (OTM + OSM labels)","OSM HOT","OpenTopoMap","Esri Topo","Esri Satélite híbrido","MapTiler Outdoor","MapTiler Hybrid"],
        index=["VFR híbrido (OTM + OSM labels)","OSM HOT","OpenTopoMap","Esri Topo","Esri Satélite híbrido","MapTiler Outdoor","MapTiler Hybrid"].index(st.session_state.leaf_base)
    )
with c:
    st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional p/ estilos 'MapTiler')", st.session_state.maptiler_key)
with d:
    if st.button("🗑 Limpar rota/legs"):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.sublegs = []
st.markdown("</div>", unsafe_allow_html=True)

# ============ PARÂMETROS ============
with st.form("globals"):
    p1,p2,p3 = st.columns(3)
    with p1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, int(st.session_state.qnh))
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, int(st.session_state.oat))
    with p2:
        st.session_state.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
    with p3:
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
        st.session_state.ck_default = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default), step=1)
    w1,w2 = st.columns(2)
    with w1: st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from))
    with w2: st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt))
    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ============ PESQUISA/ADD WP — ranking e dedupe ============
def score_row(r, tokens):
    # campos (normalizados, sem acentos, lower)
    code = strip_accents(str(r.get("code",""))).lower()
    name = strip_accents(str(r.get("name",""))).lower()
    city = strip_accents(str(r.get("city","") or r.get("sector",""))).lower()
    s = 0
    q = " ".join(tokens)
    if q == code: s += 100
    if q == name: s += 80
    for t in tokens:
        if code.startswith(t): s += 40
        elif t in code: s += 20
        if name.startswith(t): s += 30
        elif t in name: s += 15
        if t and t in city: s += 8
    # bónus se todos os tokens surgem em (code|name|city)
    hay = f"{code} {name} {city}"
    if all(t in hay for t in tokens): s += 10
    # AD preferido em caso de empate
    if r.get("type") == "AD": s += 3
    return s

def smart_search(qtxt, want_AD=True, want_LOC=True, limit=50):
    if not qtxt.strip(): return pd.DataFrame()
    tokens = [strip_accents(t).lower() for t in qtxt.strip().split()]
    df = pd.concat([ad_df, loc_df], ignore_index=True)
    if not want_AD: df = df[df["type"]!="AD"]
    if not want_LOC: df = df[df["type"]!="LOC"]
    if df.empty: return df
    # scoring
    df = df.copy()
    df["__score"] = df.apply(lambda r: score_row(r, tokens), axis=1)
    df = df[df["__score"]>0]
    if df.empty: return df
    # dedupe por (code normalizado) + lat/lon arredondados
    df["__code_norm"] = df["code"].astype(str).map(lambda s: strip_accents(s).lower())
    df["__latr"] = (df["lat"].astype(float).round(4))
    df["__lonr"] = (df["lon"].astype(float).round(4))
    df.sort_values(["__code_norm","__latr","__lonr","__score"], ascending=[True,True,True,False], inplace=True)
    df = df.drop_duplicates(subset=["__code_norm","__latr","__lonr"], keep="first")
    df = df.sort_values("__score", ascending=False).head(limit)
    return df

sc1, sc2, sc3, sc4 = st.columns([3,1.5,1.5,2])
with sc1: qtxt = st.text_input("🔎 Procurar AD/Localidade", "", placeholder="Ex: LPPT, ABRANTES, NISA, LP0078…")
with sc2:
    st.session_state.filters["AD"]  = st.checkbox("Incluir AD", value=st.session_state.filters["AD"])
with sc3:
    st.session_state.filters["LOC"] = st.checkbox("Incluir LOC", value=st.session_state.filters["LOC"])
with sc4: alt_wp = st.number_input("Altitude para WP novo (ft)", 0.0, 18000.0, 3000.0, step=100.0)

results = smart_search(qtxt, st.session_state.filters["AD"], st.session_state.filters["LOC"], limit=60)
sel_idx = None
if not results.empty:
    st.caption(f"Resultados ({len(results)}) — escolhe um:")
    options = []
    for i, r in results.iterrows():
        extra = r.get("city") or r.get("sector") or ""
        label = f"{r['type']} • {r['name']} ({r['code']}) — {extra}  [{r['lat']:.5f}, {r['lon']:.5f}]"
        options.append((i, label))
    sel_label = st.radio("Escolha o waypoint", options=[lbl for _, lbl in options], index=0, key="sel_radio")
    sel_map = {lbl:i for i, lbl in options}; sel_idx = sel_map.get(sel_label)

if st.button("Adicionar selecionado") and sel_idx is not None:
    r = results.loc[sel_idx]
    st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
    st.success(f"Adicionado: {r['name']} ({r['code']}).")

# ============ EDITOR DE WPs ============
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    del_idx, swap_u, swap_d = None, None, None
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([2,2,2,1,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                if st.button("↑", key=f"up{i}") and i>0: swap_u = i
                if st.button("↓", key=f"dn{i}") and i < len(st.session_state.wps)-1: swap_d = i
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"): del_idx = i
    if swap_u is not None:
        i=swap_u; st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
        st.info("WP movido para cima. Carrega 'Gerar/Atualizar legs' para recomputar.")
    if swap_d is not None:
        i=swap_d; st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
        st.info("WP movido para baixo. Carrega 'Gerar/Atualizar legs' para recomputar.")
    if del_idx is not None:
        st.session_state.wps.pop(del_idx)
        st.info("WP removido. Carrega 'Gerar/Atualizar legs'.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ============ LEGS + SUBLEGS ============
def rebuild_legs_from_wps():
    st.session_state.legs = []
    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        st.session_state.legs.append(dict(
            iA=i, iB=i+1, TC=float(tc), Dist=float(dist),
            Alt0=float(A["alt"]), Alt1=float(B["alt"]),
            Wfrom=int(st.session_state.wind_from), Wkt=int(st.session_state.wind_kt),
            CK=int(st.session_state.ck_default)
        ))

colsB = st.columns([2,2,6])
with colsB[0]:
    if st.button("Gerar/Atualizar legs a partir dos WAYPOINTS", type="primary", use_container_width=True) and len(st.session_state.wps) >= 2:
        rebuild_legs_from_wps()
        st.success(f"Criadas {len(st.session_state.legs)} legs base.")

def make_sublegs():
    st.session_state.sublegs = []
    qnh = st.session_state.qnh; oat = st.session_state.oat
    mag_var = st.session_state.mag_var; mag_is_e = st.session_state.mag_is_e
    desc_angle = st.session_state.desc_angle

    for idx, L in enumerate(st.session_state.legs):
        A = st.session_state.wps[L["iA"]]; B = st.session_state.wps[L["iB"]]
        latA, lonA, latB, lonB = A["lat"], A["lon"], B["lat"], B["lon"]
        tc = L["TC"]; dist = L["Dist"]
        profile = "LEVEL" if abs(L["Alt1"] - L["Alt0"]) < 1e-6 else ("CLIMB" if L["Alt1"] > L["Alt0"] else "DESCENT")

        _, THc, GScl = wind_triangle(tc, TAS_CLIMB,  L["Wfrom"], L["Wkt"])
        _, THr, GScr = wind_triangle(tc, TAS_CRUISE, L["Wfrom"], L["Wkt"])
        _, THd, GSde = wind_triangle(tc, TAS_DESCENT, L["Wfrom"], L["Wkt"])
        MHc = apply_var(THc, mag_var, mag_is_e); MHr = apply_var(THr, mag_var, mag_is_e); MHd = apply_var(THd, mag_var, mag_is_e)

        pa0 = press_alt(L["Alt0"], qnh); ROC = roc_interp(pa0, oat)
        ROD = max(100.0, GScr * 5.0 * (desc_angle/3.0))

        def add_leg(p0, p1, alt0, alt1, phase, tas, th, mh, gs, label):
            d = gc_dist_nm(p0[0], p0[1], p1[0], p1[1])
            t = rt10((d / max(gs,1e-9)) * 3600) if gs>0 and d>0 else 0
            burn = FF_CONST * (t/3600.0)
            st.session_state.sublegs.append(dict(
                parent_idx=idx, phase=phase, label=label,
                A_lat=p0[0], A_lon=p0[1], B_lat=p1[0], B_lon=p1[1],
                Alt0=alt0, Alt1=alt1, TC=gc_course_tc(p0[0],p0[1],p1[0],p1[1]),
                TAS=tas, TH=th, MH=mh, GS=gs, Dist=d, Time=t, Burn=r10f(burn)
            ))

        if profile == "CLIMB":
            t_need = (L["Alt1"] - L["Alt0"]) / max(ROC,1e-6)
            d_need = GScl * (t_need/60.0)
            if d_need < dist:
                lat_toc, lon_toc = point_along_gc(latA, lonA, latB, lonB, d_need)
                add_leg((latA,lonA), (lat_toc,lon_toc), L["Alt0"], L["Alt1"], "CLIMB", TAS_CLIMB, THc, MHc, GScl, "Climb → TOC")
                add_leg((lat_toc,lon_toc), (latB,lonB), L["Alt1"], L["Alt1"], "CRUISE", TAS_CRUISE, THr, MHr, GScr, "Cruise (após TOC)")
                st.session_state.sublegs[-2]["TOC"] = (lat_toc, lon_toc)
            else:
                add_leg((latA,lonA), (latB,lonB), L["Alt0"], L["Alt0"]+ROC*(dist/max(GScl,1e-9))*60, "CLIMB", TAS_CLIMB, THc, MHc, GScl, "Climb (não atinge)")
        elif profile == "DESCENT":
            t_need = (L["Alt0"] - L["Alt1"]) / max(ROD,1e-6)
            d_desc = GSde * (t_need/60.0)
            if d_desc < dist:
                d_to_tod = dist - d_desc
                lat_tod, lon_tod = point_along_gc(latA, lonA, latB, lonB, d_to_tod)
                add_leg((latA,lonA), (lat_tod,lon_tod), L["Alt0"], L["Alt0"], "CRUISE", TAS_CRUISE, THr, MHr, GScr, "Cruise até TOD")
                add_leg((lat_tod,lon_tod), (latB,lonB), L["Alt0"], L["Alt1"], "DESCENT", TAS_DESCENT, THd, MHd, GSde, "Descent após TOD")
                st.session_state.sublegs[-1]["TOD"] = (lat_tod, lon_tod)
            else:
                add_leg((latA,lonA), (latB,lonB), L["Alt0"], L["Alt1"], "DESCENT", TAS_DESCENT, THd, MHd, GSde, "Descent (não atinge)")
        else:
            add_leg((latA,lonA), (latB,lonB), L["Alt0"], L["Alt0"], "CRUISE", TAS_CRUISE, THr, MHr, GScr, "Cruise")

def annotate_times():
    # Calcula ETO acumulado por subleg
    if not st.session_state.sublegs: return
    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None
    clock = base_time
    for s in st.session_state.sublegs:
        if clock is None:
            s["ETO"] = ""
        else:
            clock = clock + dt.timedelta(seconds=s["Time"])
            s["ETO"] = clock.strftime("%H:%M")

# ============ INFO RESUMO ============
def render_info_cards():
    if not st.session_state.sublegs: return
    total_t = sum(s["Time"] for s in st.session_state.sublegs)
    total_b = r10f(sum(s["Burn"] for s in st.session_state.sublegs))
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_t)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_b:.1f} L</b></div>"
        + "</div>", unsafe_allow_html=True
    )

# ============ GEOMETRIA PARA LEAFLET ============
def house_polygon(lat, lon, heading_deg, w_nm=0.72, h_nm=0.92, roof_nm=0.30):
    left_lat, left_lon   = dest_point(lat, lon, heading_deg-90.0, w_nm/2.0)
    right_lat, right_lon = dest_point(lat, lon, heading_deg+90.0, w_nm/2.0)
    top_l_lat,  top_l_lon  = dest_point(left_lat,  left_lon,  heading_deg,  h_nm/2.0)
    top_r_lat,  top_r_lon  = dest_point(right_lat, right_lon, heading_deg,  h_nm/2.0)
    bot_l_lat,  bot_l_lon  = dest_point(left_lat,  left_lon,  heading_deg, -h_nm/2.0)
    bot_r_lat,  bot_r_lon  = dest_point(right_lat, right_lon, heading_deg, -h_nm/2.0)
    roof_lat, roof_lon     = dest_point((top_l_lat+top_r_lat)/2.0, (top_l_lon+top_r_lon)/2.0, heading_deg, roof_nm)
    return [[bot_l_lat, bot_l_lon],[bot_r_lat, bot_r_lon],[top_r_lat, top_r_lon],
            [roof_lat, roof_lon],[top_l_lat, top_l_lon],[bot_l_lat, bot_l_lon]]

def chip_rect_with_center(lat, lon, heading_deg, w_nm, h_nm, forward_nm):
    c_lat, c_lon = dest_point(lat, lon, heading_deg, forward_nm)
    left_lat, left_lon   = dest_point(c_lat, c_lon, heading_deg-90.0, w_nm/2.0)
    right_lat, right_lon = dest_point(c_lat, c_lon, heading_deg+90.0, w_nm/2.0)
    top_l_lat,  top_l_lon  = dest_point(left_lat,  left_lon,  heading_deg,  h_nm/2.0)
    top_r_lat,  top_r_lon  = dest_point(right_lat, right_lon, heading_deg,  h_nm/2.0)
    bot_l_lat,  bot_l_lon  = dest_point(left_lat,  left_lon,  heading_deg, -h_nm/2.0)
    bot_r_lat,  bot_r_lon  = dest_point(right_lat, right_lon, heading_deg, -h_nm/2.0)
    poly = [[bot_l_lat, bot_l_lon],[bot_r_lat, bot_r_lon],[top_r_lat, top_r_lon],[top_l_lat, top_l_lon],[bot_l_lat, bot_l_lon]]
    return poly, (c_lat, c_lon)

def build_leaflet_payload():
    data = dict(
        base=st.session_state.leaf_base,
        maptilerKey=st.session_state.maptiler_key,
        wps=[dict(name=w["name"], lat=w["lat"], lon=w["lon"]) for w in st.session_state.wps],
        paths=[], ticks=[], houses=[], chips=[], labels=[], specials=[]
    )
    if not st.session_state.sublegs: return data

    HOUSE_OFFSET_NM = 0.65
    for s in st.session_state.sublegs:
        A=(s["A_lat"], s["A_lon"]); B=(s["B_lat"], s["B_lon"])
        data["paths"].append(dict(color="black",  weight=7, latlngs=[[A[0],A[1]],[B[0],B[1]]]))
        data["paths"].append(dict(color="#ce2bd8", weight=5, latlngs=[[A[0],A[1]],[B[0],B[1]]]))

        interval_s=120; total_t=s["Time"]; total_d=s["Dist"]; tc_here=s["TC"]
        k=1
        while k*interval_s <= total_t:
            frac=(k*interval_s)/max(total_t,1); d_here=total_d*frac
            latm, lonm = point_along_gc(A[0],A[1],B[0],B[1], d_here)
            half_nm=0.12
            left_lat,left_lon=dest_point(latm,lonm,tc_here-90,half_nm)
            right_lat,right_lon=dest_point(latm,lonm,tc_here+90,half_nm)
            data["ticks"].append([[left_lat,left_lon],[right_lat,right_lon]])
            k+=1

        mid_lat, mid_lon = point_along_gc(A[0],A[1],B[0],B[1], total_d/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, s["TC"]+90, HOUSE_OFFSET_NM)
        data["houses"].append(house_polygon(off_lat, off_lon, s["TC"]))
        poly_top,  cen_top  = chip_rect_with_center(off_lat, off_lon, s["TC"], w_nm=0.58, h_nm=0.24, forward_nm= 0.08)
        poly_bot,  cen_bot  = chip_rect_with_center(off_lat, off_lon, s["TC"], w_nm=0.58, h_nm=0.24, forward_nm=-0.13)
        data["chips"].append(poly_top); data["chips"].append(poly_bot)

        mh_text = f"{rang(s['MH'])} M"
        info = f"{rang(s['TC'])} T&nbsp;&nbsp;{r10f(s['Dist'])} nm&nbsp;&nbsp;GS {rint(s['GS'])}&nbsp;&nbsp;ETE {mmss(s['Time'])}"
        if s.get("ETO"): info += f"&nbsp;&nbsp;ETO {s['ETO']}"
        data["labels"].append(dict(lat=cen_top[0], lon=cen_top[1], cls="mh",  text=mh_text))
        data["labels"].append(dict(lat=cen_bot[0], lon=cen_bot[1], cls="info", text=info))

        if "TOC" in s:
            lt,ln = s["TOC"]; p = dest_point(lt,ln,s["TC"]+90,0.18); lab = dest_point(p[0],p[1],s["TC"]+110,0.16)
            data["specials"].append(dict(lat=p[0],lon=p[1]], label_pos=[lab[0],lab[1]], label="TOC"))
        if "TOD" in s:
            lt,ln = s["TOD"]; p = dest_point(lt,ln,s["TC"]+90,0.18); lab = dest_point(p[0],p[1],s["TC"]+110,0.16)
            data["specials"].append(dict(lat=p[0],lon=p[1]], label_pos=[lab[0],lab[1]], label="TOD"))
    return data

# ============ LEAFLET (HTML embebido) ============
LEAFLET_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map{height:100%; margin:0}
  .lbl{font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; white-space:nowrap}
  .lbl.mh{font-weight:800; font-size:22px; color:#d61f69; text-shadow:0 0 2px #fff, 0 0 6px #fff}
  .lbl.info{font-weight:600; font-size:13px; color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff}
  .lbl.wp{font-weight:700; font-size:14px; color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff}
  .lbl.tag{font-weight:800; font-size:13px; color:#ff8c00; text-shadow:0 0 2px #fff, 0 0 6px #fff}
  .toc{width:10px;height:10px;background:#ff8c00;border-radius:50%;border:2px solid #fff; box-shadow:0 0 0 2px rgba(0,0,0,.3)}
  .wpdot{width:10px;height:10px;background:#fff;border-radius:50%;border:2px solid #000}
</style>
</head>
<body>
<div id="map"></div>

<!-- payload vem ANTES do JS que o lê (bug fix) -->
<script id="payload" type="application/json">{PAYLOAD}</script>

<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);

// Map init
const map = L.map('map', { zoomControl: true });

// Base layers
const bases = {};
function mt(url){ return url.replaceAll('{key}', DATA.maptilerKey || ''); }

bases["OpenTopoMap"] = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {subdomains:'abc', maxZoom:17, opacity:1});
bases["OSM HOT"] = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {subdomains:'abc', maxZoom:19});
bases["Esri Topo"] = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {maxZoom:19});
bases["Esri Sat"]  = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:19});
bases["OSM Std"]   = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {subdomains:'abc', maxZoom:19, opacity:0.45});

// MapTiler (opcional)
if (DATA.maptilerKey){
  bases["MapTiler Outdoor"] = L.tileLayer(mt('https://api.maptiler.com/maps/outdoor/{z}/{x}/{y}.png?key={key}'), {maxZoom:19});
  bases["MapTiler Hybrid"]  = L.layerGroup([
    L.tileLayer(mt('https://api.maptiler.com/tiles/satellite/{z}/{x}/{y}.jpg?key={key}')),
    L.tileLayer(mt('https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.png?key={key}'), {opacity:0.85})
  ]);
}

// Pick default
let chosen;
switch (DATA.base){
  case "VFR híbrido (OTM + OSM labels)": chosen = L.layerGroup([bases["OpenTopoMap"], bases["OSM Std"]]); break;
  case "OSM HOT": chosen = bases["OSM HOT"]; break;
  case "OpenTopoMap": chosen = bases["OpenTopoMap"]; break;
  case "Esri Topo": chosen = bases["Esri Topo"]; break;
  case "Esri Satélite híbrido": chosen = L.layerGroup([bases["Esri Sat"], bases["OSM Std"]]); break;
  case "MapTiler Outdoor": chosen = bases["MapTiler Outdoor"] || bases["OpenTopoMap"]; break;
  case "MapTiler Hybrid": chosen = bases["MapTiler Hybrid"] || L.layerGroup([bases["Esri Sat"], bases["OSM Std"]]); break;
  default: chosen = bases["OpenTopoMap"];
}
chosen.addTo(map);

// Layers
const routeLayer = L.layerGroup().addTo(map);
const tickLayer  = L.layerGroup().addTo(map);
const houseLayer = L.layerGroup().addTo(map);
const chipLayer  = L.layerGroup().addTo(map);
const labelLayer = L.layerGroup().addTo(map);
const wpLayer    = L.layerGroup().addTo(map);
const specialLayer = L.layerGroup().addTo(map);

// Draw routes (shadow + magenta)
DATA.paths.forEach(p=>{
  L.polyline(p.latlngs, {color:p.color, weight:p.weight, opacity:1, lineCap:'round'}).addTo(routeLayer);
});

// Ticks
DATA.ticks.forEach(t=>{
  L.polyline(t,{color:'#000', weight:2}).addTo(tickLayer);
});

// WPs
DATA.wps.forEach((w,i)=>{
  L.marker([w.lat,w.lon], {icon: L.divIcon({className:'wpdot'})}).addTo(wpLayer);
  L.marker([w.lat,w.lon], {icon: L.divIcon({className:'lbl wp', html:w.name, iconAnchor:[-6,-14]})}).addTo(wpLayer);
});

// Houses + chips
DATA.houses.forEach(poly=>{
  L.polygon(poly, {color:'#000', weight:2, fill:true, fillColor:'#fff', fillOpacity:0.95}).addTo(houseLayer);
});
DATA.chips.forEach(poly=>{
  L.polygon(poly, {color:'#000', weight:2, fill:true, fillColor:'#fff', fillOpacity:1}).addTo(chipLayer);
});

// Labels (dentro das caixas)
DATA.labels.forEach(lbl=>{
  L.marker([lbl.lat,lbl.lon], {icon: L.divIcon({className:'lbl '+lbl.cls, html: lbl.text, iconAnchor:[0,12]})}).addTo(labelLayer);
});

// TOC/TOD
DATA.specials.forEach(s=>{
  L.marker([s.lat,s.lon], {icon: L.divIcon({className:'toc'})}).addTo(specialLayer);
  L.marker(s.label_pos, {icon: L.divIcon({className:'lbl tag', html:s.label, iconAnchor:[0,0]})}).addTo(specialLayer);
});

// Fit bounds
const all = [];
DATA.paths.forEach(p=>p.latlngs.forEach(ll=>all.push(ll)));
DATA.wps.forEach(w=>all.push([w.lat,w.lon]));
const b = L.latLngBounds(all);
map.fitBounds(b.pad(0.15));

L.control.scale({imperial:false}).addTo(map);
</script>
</body>
</html>
"""

# ============ RENDER ============
def render_leaflet():
    if len(st.session_state.wps) < 2:
        st.info("Adiciona pelo menos 2 waypoints e gera as legs.")
        return
    if not st.session_state.legs:
        rebuild_legs_from_wps()
    make_sublegs()
    annotate_times()
    render_info_cards()
    payload = build_leaflet_payload()
    html = LEAFLET_HTML.replace("{PAYLOAD}", json.dumps(payload))
    st_html(html, height=720, scrolling=False)

# ============ RUN ============
render_leaflet()

