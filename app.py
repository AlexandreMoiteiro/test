# NAVLOG v12 — PyDeck • Triângulos (“dog houses”) • Riscas 2 min • TOC/TOD como WPs
# Espeeds fixos: 70 kt climb, 90 kt cruise/descida. Sem RPMs.
# Mapa basemap CARTO Voyager (com estradas, rios, localidades).
# Dog houses triangulares com MH grande colorido (+ TAS e ETO).
# TOC/TOD inseridos como WPs "virtuais" e geram novas legs.

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# =============== PAGE / STYLE ===============
st.set_page_config(page_title="NAVLOG v12 — TOC/TOD + Dog Houses", layout="wide", initial_sidebar_state="collapsed")

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
.tl{position:relative;margin:8px 0 18px 0;padding-bottom:46px}
.tl .bar{height:6px;background:#eef1f5;border-radius:3px}
.tl .tick{position:absolute;top:10px;width:2px;height:14px;background:#333}
.tl .cp-lbl{position:absolute;top:32px;transform:translateX(-50%);text-align:center;font-size:11px;color:#333;white-space:nowrap}
.tl .tocdot,.tl .toddot{position:absolute;top:-6px;width:14px;height:14px;border-radius:50%;transform:translateX(-50%);border:2px solid #fff;box-shadow:0 0 0 2px rgba(0,0,0,0.15)}
.tl .tocdot{background:#1f77b4}.tl .toddot{background:#d62728}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============== UTILS ===============
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang = lambda x: int(round(float(x))) % 360
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
clamp = lambda v, lo, hi: max(lo, min(hi, v))

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

# geodesia (NM)
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

# AFM bits para ROC / ROD
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
isa_temp = lambda pa: 15.0 - 2.0*(pa/1000.0)
press_alt = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0

def interp1(x, x0, x1, y0, y1):
    if x1 == x0: return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def roc_interp(pa, oat):
    pas = sorted(ROC_ENR.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    temps = [-25,0,25,50]; t = clamp(oat, temps[0], temps[-1])
    if t <= 0: t0, t1 = -25, 0
    elif t <= 25: t0, t1 = 0, 25
    else: t0, t1 = 25, 50
    v00, v01 = ROC_ENR[p0][t0], ROC_ENR[p0][t1]
    v10, v11 = ROC_ENR[p1][t0], ROC_ENR[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01); v1 = interp1(t, t0, t1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * 0.90)

# =============== STATE ===============
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("computed", [])
# velocidades fixas
ens("spd_climb", 70.0); ens("spd_cruise", 90.0); ens("spd_desc", 90.0)
# consumos simples (ajustáveis)
ens("ff_climb", 22.0); ens("ff_cruise", 20.0); ens("ff_desc", 12.0)

# =============== TIMELINE ===============
def timeline(seg_time_s, gs_kt, ff_lph, start_label, end_label, ck_min=2, efob_start=None):
    total = max(1, int(seg_time_s))
    html = "<div class='tl'><div class='bar'></div>"
    parts = []
    if ck_min > 0:
        t=0
        while t + ck_min*60 <= total:
            t += ck_min*60
            d = gs_kt*(t/3600.0)
            burn = ff_lph*(t/3600.0)
            eto = ""  # já mostramos fora, manter limpo
            efob = (max(0.0, r10f(efob_start - burn)) if efob_start is not None else 0.0)
            pct = (t/total)*100.0
            parts += [f"<div class='tick' style='left:{pct:.2f}%;'></div>",
                      f"<div class='cp-lbl' style='left:{pct:.2f}%;'><div>T+{t//60}m</div><div>{round(d,1)} nm</div>" +
                      (f"<div>EFOB {efob:.1f}</div>" if efob_start is not None else "") + "</div>"]
    html += ''.join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"GS {rint(gs_kt)} kt · TAS {rint(gs_kt)} kt? → (TAS real: mostrado na dog house) · FF {rint(ff_lph)} L/h  |  {start_label} → {end_label}")

# =============== DOG HOUSE (triângulo + textos) ===============
def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

# =============== CSVs locais (AD/Localidades) ===============
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
    rows=[]
    for line in df.iloc[:,0].dropna().tolist():
        s=str(line).strip()
        if not s or s.startswith(("Ident","DEP/")): continue
        tokens=s.split()
        coord=[t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord)>=2:
            lat_tok, lon_tok = coord[-2], coord[-1]
            lat = dms_to_dd(lat_tok, False); lon = dms_to_dd(lon_tok, True)
            ident = tokens[0] if re.match(r"^[A-Z0-9]{4,}$", tokens[0]) else None
            try: name=" ".join(tokens[1:tokens.index(coord[0])]).strip()
            except: name=" ".join(tokens[1:]).strip()
            rows.append({"type":"AD","code":ident or name,"name":name,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for line in df.iloc[:,0].dropna().tolist():
        s=str(line).strip()
        if not s or "Total de registos" in s: continue
        tokens=s.split()
        coord=[t for t in tokens if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", t)]
        if len(coord)>=2:
            lat_tok, lon_tok = coord[0], coord[1]
            lat=dms_to_dd(lat_tok, False); lon=dms_to_dd(lon_tok, True)
            try: lon_idx=tokens.index(lon_tok)
            except: continue
            code=tokens[lon_idx+1] if lon_idx+1<len(tokens) else None
            name=" ".join(tokens[:tokens.index(lat_tok)]).strip()
            rows.append({"type":"LOC","code":code or name,"name":name,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

# =============== HEADER ===============
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,2,3,2])
with h1: st.title("NAVLOG — v12")
with h2: st.toggle("Mostrar TIMELINE/CPs", key="show_timeline", value=st.session_state.show_timeline)
with h3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.computed = []
st.markdown("</div>", unsafe_allow_html=True)

# =============== PARÂMETROS GLOBAIS ===============
with st.form("globals"):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, int(st.session_state.qnh))
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, int(st.session_state.oat))
    with p2:
        st.session_state.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
    with p3:
        st.session_state.desc_angle = st.number_input("Ângulo descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
        st.session_state.mag_var = st.number_input("Mag Var (°)", 0, 30, int(st.session_state.mag_var))
        st.session_state.mag_is_e = st.selectbox("Var E/W", ["W","E"], index=(1 if st.session_state.mag_is_e else 0)) == "E"
    with p4:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.session_state.spd_climb  = st.number_input("TAS Climb (kt)", 30.0, 140.0, float(st.session_state.spd_climb), step=1.0)
    with s2:
        st.session_state.spd_cruise = st.number_input("TAS Cruise (kt)", 30.0, 160.0, float(st.session_state.spd_cruise), step=1.0)
    with s3:
        st.session_state.spd_desc   = st.number_input("TAS Descent (kt)", 30.0, 160.0, float(st.session_state.spd_desc), step=1.0)
    f1, f2, f3 = st.columns(3)
    with f1: st.session_state.ff_climb  = st.number_input("FF Climb (L/h)", 0.0, 60.0, float(st.session_state.ff_climb), step=0.5)
    with f2: st.session_state.ff_cruise = st.number_input("FF Cruise (L/h)", 0.0, 60.0, float(st.session_state.ff_cruise), step=0.5)
    with f3: st.session_state.ff_desc   = st.number_input("FF Descent (L/h)", 0.0, 60.0, float(st.session_state.ff_desc), step=0.5)
    st.form_submit_button("Aplicar parâmetros")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# =============== LER CSVs LOCAIS + pesquisa rápida ===============
try:
    ad_raw  = pd.read_csv(AD_CSV)
    loc_raw = pd.read_csv(LOC_CSV)
    ad_df   = parse_ad_df(ad_raw)
    loc_df  = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["type","code","name","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["type","code","name","lat","lon","alt"])
    st.warning("Não consegui carregar os CSVs locais (AD/Localidades). Verifica os ficheiros.")

flt1, flt2, btn = st.columns([4,2,1.5])
with flt1: qtxt = st.text_input("🔎 Procurar AD/Localidade (dos CSVs locais)", "", placeholder="Ex: LPPT, ABRANTES, LP0078…")
with flt2: alt_wp = st.number_input("Altitude para WPs adicionados (ft)", 0.0, 18000.0, 3000.0, step=100.0)
with btn: add_sel = st.button("Adicionar resultados")

def filter_df(df,q):
    if not q: return df
    t=q.lower().strip()
    return df[df.apply(lambda r: any(t in str(v).lower() for v in r.values), axis=1)]

ad_f, loc_f = filter_df(ad_df,qtxt), filter_df(loc_df,qtxt)
if add_sel:
    for _, r in pd.concat([ad_f, loc_f]).iterrows():
        st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
    st.success(f"Adicionados {len(ad_f)+len(loc_f)} WPs.")

# =============== EDITOR DE WPs ===============
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
                up = st.button("↑", key=f"up{i}"); dn = st.button("↓", key=f"dn{i}")
                if up and i>0:
                    st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
                    st.experimental_rerun()
                if dn and i < len(st.session_state.wps)-1:
                    st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
                    st.experimental_rerun()
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i); st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# =============== GERAR ROTA COM TOC/TOD INSERIDOS ===============
def build_route_points_with_profile(wps):
    """Devolve (route_pts, profile_markers)
       route_pts inclui WPs originais + TOC/TOD (virtuais, name='TOC'/'TOD')"""
    pts = []
    markers = []  # para pôr no mapa
    for i in range(len(wps)-1):
        A, B = wps[i], wps[i+1]
        if not pts: pts.append(A)
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist_total = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])

        # velocidades TAS fixas
        tas_climb  = float(st.session_state.spd_climb)
        tas_desc   = float(st.session_state.spd_desc)
        tas_cruise = float(st.session_state.spd_cruise)

        # vento
        _, _, gs_climb  = wind_triangle(tc, tas_climb,  st.session_state.wind_from, st.session_state.wind_kt)
        _, _, gs_desc   = wind_triangle(tc, tas_desc,   st.session_state.wind_from, st.session_state.wind_kt)
        _, _, gs_cruise = wind_triangle(tc, tas_cruise, st.session_state.wind_from, st.session_state.wind_kt)

        # perfil: CLIMB, DESCENT, LEVEL
        if abs(B["alt"] - A["alt"]) < 1e-6:
            # level: não insere nada
            pts.append(B); continue

        if B["alt"] > A["alt"]:
            # CLIMB → calcula TOC
            pa0 = press_alt(A["alt"], st.session_state.qnh)
            roc = roc_interp(pa0, st.session_state.oat)  # ft/min
            t_need_min = (B["alt"] - A["alt"]) / max(roc, 1e-6)
            d_need_nm  = gs_climb * (t_need_min/60.0)
            if d_need_nm < dist_total:
                # inserir TOC
                latc, lonc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need_nm)
                TOC = {"name":"TOC","lat":latc,"lon":lonc,"alt":B["alt"], "virtual":True}
                pts.append(TOC); markers.append(dict(kind="TOC", lat=latc, lon=lonc))
                pts.append(B)
            else:
                pts.append(B)  # não atinge — sem TOC
        else:
            # DESCENT → calcula TOD (usando regra: ROD ≈ GS*5*(angle/3))
            ROD = max(100.0, gs_desc * 5.0 * (st.session_state.desc_angle/3.0))  # ft/min
            t_need_min = (A["alt"] - B["alt"]) / max(ROD, 1e-6)
            d_need_nm  = gs_desc * (t_need_min/60.0)
            if d_need_nm < dist_total:
                latd, lond = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need_nm)
                TOD = {"name":"TOD","lat":latd,"lon":lond,"alt":B["alt"], "virtual":True}
                pts.append(TOD); markers.append(dict(kind="TOD", lat=latd, lon=lond))
                pts.append(B)
            else:
                pts.append(B)  # não atinge — sem TOD
    return pts, markers

def build_legs_from_route_points(rps):
    legs=[]
    for i in range(len(rps)-1):
        A, B = rps[i], rps[i+1]
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        legs.append(dict(
            A=A, B=B, TC=tc, Dist=dist, Alt0=A["alt"], Alt1=B["alt"],
            Wfrom=st.session_state.wind_from, Wkt=st.session_state.wind_kt,
            CK=st.session_state.ck_default, HoldMin=0.0, HoldFF=0.0
        ))
    return legs

if st.button("Gerar/Atualizar LEGS (insere TOC/TOD)", type="primary"):
    if len(st.session_state.wps) < 2:
        st.warning("Precisas de pelo menos 2 WPs.")
    else:
        route_pts, markers = build_route_points_with_profile(st.session_state.wps)
        st.session_state._route_pts = route_pts      # guardo para o mapa
        st.session_state._profile_markers = markers
        st.session_state.legs = build_legs_from_route_points(route_pts)
        st.success(f"Foram criadas {len(st.session_state.legs)} legs (com TOC/TOD se aplicável).")

# =============== COMPUTE POR LEG ===============
def compute_leg(leg, base_clock=None, efob_start=None):
    tc, dist = leg["TC"], leg["Dist"]
    alt0, alt1 = leg["Alt0"], leg["Alt1"]

    # tipo e TAS/FF
    if alt1 > alt0:
        tas = float(st.session_state.spd_climb);  ff = float(st.session_state.ff_climb);  phase="Climb"
    elif alt1 < alt0:
        tas = float(st.session_state.spd_desc);   ff = float(st.session_state.ff_desc);   phase="Descent"
    else:
        tas = float(st.session_state.spd_cruise); ff = float(st.session_state.ff_cruise); phase="Cruise/Level"

    # vento → TH/MH/GS
    _, TH, GS = wind_triangle(tc, tas, leg["Wfrom"], leg["Wkt"])
    MH = apply_var(TH, st.session_state.mag_var, st.session_state.mag_is_e)

    # tempo, burn
    time_s = rt10((dist / max(GS,1e-9)) * 3600.0)
    burn   = ff * (time_s/3600.0)
    efob_end = (max(0.0, r10f(efob_start - burn)) if efob_start is not None else None)

    # relógio leg
    if base_clock:
        start = base_clock
        end   = base_clock + dt.timedelta(seconds=time_s)
        start_lbl = start.strftime("%H:%M"); end_lbl = end.strftime("%H:%M")
    else:
        start_lbl = "T+00:00"; end_lbl = f"T+{mmss(time_s)}"

    # ETO no MEIO da leg (para dog house)
    if base_clock: eto_mid = (base_clock + dt.timedelta(seconds=time_s/2)).strftime("%H:%M")
    else:          eto_mid = f"T+{mmss(int(time_s/2))}"

    return dict(
        phase=phase, TC=tc, Dist=dist, Alt0=alt0, Alt1=alt1,
        TAS=tas, GS=GS, TH=TH, MH=MH, time_s=time_s, ff=ff, burn=burn,
        start_lbl=start_lbl, end_lbl=end_lbl, eto_mid=eto_mid, efob_end=efob_end
    )

def recompute_all():
    st.session_state.computed = []
    base_clock = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_clock = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_clock = None

    carry_efob = float(st.session_state.start_efob)
    cursor = base_clock
    for leg in st.session_state.legs:
        res = compute_leg(leg, base_clock=cursor, efob_start=carry_efob)
        st.session_state.computed.append(res)
        if cursor: cursor = cursor + dt.timedelta(seconds=res["time_s"])
        if carry_efob is not None: carry_efob = res["efob_end"]

# =============== INPUTS + FASES POR LEG ===============
if st.session_state.legs:
    recompute_all()

    total_sec_all = sum(c["time_s"] for c in st.session_state.computed)
    total_burn_all = r10f(sum(c["burn"] for c in st.session_state.computed))
    efob_final = st.session_state.computed[-1]["efob_end"] if st.session_state.computed and st.session_state.computed[-1]["efob_end"] is not None else None

    for idx, (leg, comp) in enumerate(zip(st.session_state.legs, st.session_state.computed)):
        with st.expander(f"Leg {idx+1} — Inputs", expanded=True):
            i1,i2,i3,i4 = st.columns(4)
            with i1:
                TC   = st.number_input(f"True Course (°T) — L{idx+1}", 0.0, 359.9, float(leg["TC"]), step=0.1, key=f"TC_{idx}")
                Dist = st.number_input(f"Distância (nm) — L{idx+1}", 0.0, 500.0, float(leg["Dist"]), step=0.1, key=f"Dist_{idx}")
            with i2:
                Alt0 = st.number_input(f"Alt INI (ft) — L{idx+1}", 0.0, 30000.0, float(leg["Alt0"]), step=50.0, key=f"Alt0_{idx}")
                Alt1 = st.number_input(f"Alt DEST (ft) — L{idx+1}",0.0, 30000.0, float(leg["Alt1"]), step=50.0, key=f"Alt1_{idx}")
            with i3:
                Wfrom = st.number_input(f"Vento FROM (°T) — L{idx+1}", 0, 360, int(leg["Wfrom"]), step=1, key=f"Wfrom_{idx}")
                Wkt   = st.number_input(f"Vento (kt) — L{idx+1}", 0, 150, int(leg["Wkt"]), step=1, key=f"Wkt_{idx}")
            with i4:
                CK    = st.number_input(f"Checkpoints (min) — L{idx+1}", 1, 10, int(leg["CK"]), step=1, key=f"CK_{idx}")
            if st.button("Guardar leg", key=f"save_{idx}", use_container_width=True):
                leg.update(dict(TC=TC,Dist=Dist,Alt0=Alt0,Alt1=Alt1,Wfrom=Wfrom,Wkt=Wkt,CK=CK))
                recompute_all()

        # ======= FASE (logo após inputs) =======
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([3,2])
        with left:
            st.subheader(f"Fase {idx+1}: {comp['phase']}")
            st.markdown(
                "<div class='kvrow'>"
                + f"<div class='kv'>Alt: <b>{int(round(comp['Alt0']))}→{int(round(comp['Alt1']))} ft</b></div>"
                + f"<div class='kv'>TH/MH: <b>{rang(comp['TH'])}T / {rang(comp['MH'])}M</b></div>"
                + f"<div class='kv'>TAS/GS: <b>{rint(comp['TAS'])}/{rint(comp['GS'])} kt</b></div>"
                + f"<div class='kv'>FF: <b>{rint(comp['ff'])} L/h</b></div>"
                + "</div>", unsafe_allow_html=True
            )
        with right:
            st.metric("ETE desta leg", mmss(comp["time_s"]))
            st.metric("Fuel desta leg (L)", f"{r10f(comp['burn']):.1f}")
        r1,r2,r3 = st.columns(3)
        with r1: st.markdown(f"**Relógio** — {comp['start_lbl']} → {comp['end_lbl']}")
        with r2:
            if comp["efob_end"] is not None:
                start_efob = r10f(float(st.session_state.start_efob) - r10f(sum(x['burn'] for x in st.session_state.computed[:idx])))
                st.markdown(f"**EFOB** — Start {start_efob:.1f} L → End {comp['efob_end']:.1f} L")
            else:
                st.markdown("**EFOB** — —")
        with r3: st.markdown(f"**ETO (meio da leg)** — {comp['eto_mid']}")
        if st.session_state.show_timeline:
            timeline(comp["time_s"], comp["GS"], comp["ff"], comp["start_lbl"], comp["end_lbl"], ck_min=leg["CK"],
                     efob_start=(r10f(float(st.session_state.start_efob) - r10f(sum(x['burn'] for x in st.session_state.computed[:idx])))))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec_all)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn_all:.1f} L</b></div>"
        + (f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>" if efob_final is not None else "")
        + "</div>", unsafe_allow_html=True
    )

# =============== MAPA (rota + riscas + dog houses + markers TOC/TOD) ===============
if st.session_state.legs:
    # dados p/ layers
    path_data=[]; tick_data=[]; tri_data=[]; text_MH=[]; text_sub=[]; wp_markers=[]

    # usar route_pts com TOC/TOD (se existir da geração); fallback a WPs
    route_pts = st.session_state.get("_route_pts", None) or st.session_state.wps
    prof_mark = st.session_state.get("_profile_markers", [])

    # percurso principal
    for i in range(len(route_pts)-1):
        A, B = route_pts[i], route_pts[i+1]
        path_data.append({"path":[[A["lon"],A["lat"]],[B["lon"],B["lat"]]], "name":f"{A['name']}→{B['name']}"})

    # markers WPs (inclui virtuais)
    for i, P in enumerate(route_pts):
        wp_markers.append({"position":[P["lon"],P["lat"]], "label": P.get("name", f"WP{i+1}")})

    # riscas + triângulos + textos por leg
    cum_before = 0  # s, para ETO mid sem base clock
    base_clock = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_clock = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_clock=None

    for i, (leg, comp) in enumerate(zip(st.session_state.legs, st.session_state.computed)):
        A, B = leg["A"], leg["B"]
        tc = leg["TC"]

        # ---- riscas cada 2 min ----
        interval = 120
        k=1
        while k*interval <= comp["time_s"]:
            t=k*interval
            d=comp["GS"]*(t/3600.0)
            latm, lonm = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], min(d, leg["Dist"]))
            half=0.15
            llat, llon = dest_point(latm, lonm, tc-90, half)
            rlat, rlon = dest_point(latm, lonm, tc+90, half)
            tick_data.append({"path":[[llon,llat],[rlon,rlat]]})
            k+=1

        # ---- dog house (centro da leg deslocada um pouco para o lado) ----
        mid_lat, mid_lon = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], leg["Dist"]/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, tc+90, 0.35)
        tri = triangle_coords(off_lat, off_lon, tc, h_nm=0.95, w_nm=0.70)
        tri_data.append({"polygon":tri})

        # textos: MH grande colorido; sub-linha com TAS e ETO
        MH = rang(comp["MH"])
        # ETO mid
        if base_clock:
            eto_mid = (base_clock + dt.timedelta(seconds=cum_before + comp["time_s"]/2)).strftime("%H:%M")
        else:
            eto_mid = f"T+{mmss(int(cum_before + comp['time_s']/2))}"
        text_MH.append({"position":[off_lon, off_lat], "text": f"{MH}M"})
        text_sub.append({"position":[off_lon, off_lat-0.03], "text": f"TAS {rint(comp['TAS'])} • ETO {eto_mid}"})
        cum_before += comp["time_s"]

    # TOC/TOD markers (se houver)
    for m in prof_mark:
        wp_markers.append({"position":[m["lon"], m["lat"]], "label": m["kind"]})

    # LAYERS
    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[199, 0, 255, 220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0,0,0,255], width_min_pixels=2)
    tri_layer   = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon",
                            get_fill_color=[255,255,255,230], get_line_color=[0,0,0,255],
                            line_width_min_pixels=2, stroked=True, filled=True)
    # MH grande e bem visível (ciano forte)
    text_mh_layer = pdk.Layer("TextLayer", data=text_MH, get_position="position", get_text="text",
                              get_size=22, get_color=[0,120,255], get_alignment_baseline="'center'", get_text_anchor="'middle'")
    text_sub_layer = pdk.Layer("TextLayer", data=text_sub, get_position="position", get_text="text",
                               get_size=14, get_color=[0,0,0], get_alignment_baseline="'top'", get_text_anchor="'middle'")
    # WPs
    wp_layer = pdk.Layer("TextLayer", data=wp_markers, get_position="position", get_text="label",
                         get_size=12, get_color=[40,40,40], get_alignment_baseline="'bottom'", get_text_anchor="'middle'")

    # vista
    pts = route_pts if route_pts else st.session_state.wps
    mean_lat = sum(p["lat"] for p in pts)/len(pts)
    mean_lon = sum(p["lon"] for p in pts)/len(pts)

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",  # muitos nomes e detalhes
        initial_view_state=pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=7, pitch=0),
        layers=[route_layer, ticks_layer, tri_layer, text_mh_layer, text_sub_layer, wp_layer],
        tooltip={"text": "{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona WPs e gera as legs (o TOC/TOD é inserido automaticamente).")
