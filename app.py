# NAVLOG v19 — VFR híbrido real — Dog houses corrigidas (texto DENTRO) — TOC/TOD WPs
# TAS fixas: 70/90/90 kt — FF 20 L/h — riscas 2 min por GS — seleção única de WPs
# Mapas: OTM, OSM HOT, Esri Topo, Satélite híbrido (Esri+OSM), Positron. Troca de estilo força re-render.

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# ================= PAGE / STYLE =================
st.set_page_config(page_title="NAVLOG v19 — VFR", layout="wide", initial_sidebar_state="collapsed")
CSS = """
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6;--mh:#d61f69}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin:12px 0;background:#fff}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:#e5e7eb;margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
.tl{position:relative;margin:8px 0 18px 0;padding-bottom:46px}
.tl .bar{height:6px;background:#eef1f5;border-radius:3px}
.tl .tick{position:absolute;top:10px;width:2px;height:14px;background:#333}
.tl .cp-lbl{position:absolute;top:32px;transform:translateX(-50%);text-align:center;font-size:11px;color:#333;white-space:nowrap}
.tl .tocdot,.tl .toddot{position:absolute;top:-6px;width:14px;height:14px;border-radius:50%;transform:translateX(-50%);border:2px solid #fff;box-shadow:0 0 0 2px rgba(0,0,0,0.15)}
.tl .tocdot{background:#1f77b4}.tl .toddot{background:#d97706}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ================= UTILS =================
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang   = lambda x: int(round(float(x))) % 360
rint   = lambda x: int(round(float(x)))
r10f   = lambda x: round(float(x), 1)
clamp  = lambda v, lo, hi: max(lo, min(hi, v))

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

# geodesia
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

# ================= FIXOS =================
TAS_CLIMB, TAS_CRUISE, TAS_DESCENT = 70.0, 90.0, 90.0
FF_CONST = 20.0

# ================= ROC/ROD =================
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
ROC_FACTOR = 0.90
press_alt  = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0
def interp1(x, x0, x1, y0, y1):
    if x1==x0: return y0
    t=(x-x0)/(x1-x0); return y0 + t*(y1-y0)
def roc_interp(pa, temp):
    pas = sorted(ROC_ENR.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    temps = [-25,0,25,50]; t = clamp(temp, temps[0], temps[-1])
    t0,t1 = (-25,0) if t<=0 else (0,25) if t<=25 else (25,50)
    v00,v01 = ROC_ENR[p0][t0], ROC_ENR[p0][t1]
    v10,v11 = ROC_ENR[p1][t0], ROC_ENR[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01); v1 = interp1(pa_c, p0, p1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * ROC_FACTOR)

# ================= STATE =================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("sublegs", [])
ens("map_style", "VFR híbrido (OTM + OSM labels)")

# ================= TIMELINE =================
def timeline(seg, cps, start_label, end_label, toc_tod=None):
    total = max(1, int(seg['time']))
    html = "<div class='tl'><div class='bar'></div>"
    parts = []
    for cp in cps:
        pct = (cp['t']/total)*100.0
        parts += [f"<div class='tick' style='left:{pct:.2f}%;'></div>",
                  f"<div class='cp-lbl' style='left:{pct:.2f}%;'><div>T+{cp['min']}m</div><div>{cp['nm']} nm</div>" +
                  (f"<div>{cp['eto']}</div>" if cp['eto'] else "") + f"<div>EFOB {cp['efob']:.1f}</div></div>"]
    if toc_tod is not None and 0 < toc_tod['t'] < total:
        pct = (toc_tod['t']/total)*100.0
        cls = 'tocdot' if toc_tod['type'] == 'TOC' else 'toddot'
        parts.append(f"<div class='{cls}' title='{toc_tod['type']}' style='left:{pct:.2f}%;'></div>")
    html += ''.join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"GS {rint(seg['GS'])} kt · TAS {rint(seg['TAS'])} kt · FF {int(FF_CONST)} L/h  |  {start_label} → {end_label}")

# ================= PARSE CSVs =================
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
        if not s or s.startswith(("Ident","DEP/")): continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok = coord_toks[-2]; lon_tok = coord_toks[-1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            ident = tokens[0] if re.match(r"^[A-Z0-9]{4,}$", tokens[0]) else None
            try: name = " ".join(tokens[1:tokens.index(coord_toks[0])]).strip()
            except: name = " ".join(tokens[1:]).strip()
            try: lon_idx = tokens.index(lon_tok); city = " ".join(tokens[lon_idx+1:]) or None
            except: city = None
            rows.append({"type":"AD","code":ident or name,"name":name,"city":city,"lat":lat,"lon":lon,"alt":0.0})
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
            rows.append({"type":"LOC","code":code or name,"name":name,"sector":sector,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

# ================= HEADER =================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1,h2,h3,h4 = st.columns([3,2,3,2])
with h1: st.title("NAVLOG — v19 (VFR)")
with h2: st.toggle("Mostrar TIMELINE/CPs", key="show_timeline", value=st.session_state.show_timeline)
with h3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.sublegs = []
st.markdown("</div>", unsafe_allow_html=True)

# ================= PARÂMETROS =================
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
    with w1: st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
    with w2: st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    if st.form_submit_button("Aplicar parâmetros"):
        for L in st.session_state.legs:
            L["Wfrom"] = int(st.session_state.wind_from); L["Wkt"] = int(st.session_state.wind_kt)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================= PESQUISA/ADD WP (seleção única) =================
try:
    ad_raw  = pd.read_csv(AD_CSV); loc_raw = pd.read_csv(LOC_CSV)
    ad_df  = parse_ad_df(ad_raw); loc_df = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

def filter_df(df, q):
    if not q: return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

cflt1,cflt2,cbtn = st.columns([3,3,2])
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, LP0078…")
with cflt2: alt_wp = st.number_input("Altitude para WP novo (ft)", 0.0, 18000.0, 3000.0, step=100.0)
with cbtn: add_sel = st.button("Adicionar selecionado")

ad_f  = filter_df(ad_df, qtxt); loc_f = filter_df(loc_df, qtxt)
results = pd.concat([ad_f, loc_f], ignore_index=True)

sel_idx = None
if not results.empty and qtxt.strip():
    st.caption(f"Resultados para **{qtxt}** ({len(results)}) — escolhe um:")
    options = []
    for i, r in results.iterrows():
        extra = r.get("city") or r.get("sector") or ""
        label = f"{r['type']} • {r['name']} ({r['code']}) — {extra}  [{r['lat']:.5f}, {r['lon']:.5f}]"
        options.append((i, label))
    sel_label = st.radio("Escolha o waypoint", options=[lbl for _, lbl in options], index=0)
    sel_map = {lbl:i for i, lbl in options}; sel_idx = sel_map.get(sel_label)

if add_sel and sel_idx is not None:
    r = results.iloc[sel_idx]
    st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
    st.success(f"Adicionado: {r['name']} ({r['code']}).")

# ================= EDITOR DE WPs =================
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    delete_idx, swap_up, swap_down = None, None, None
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([2,2,2,1,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                if st.button("↑", key=f"up{i}") and i>0: swap_up = i
                if st.button("↓", key=f"dn{i}") and i < len(st.session_state.wps)-1: swap_down = i
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"): delete_idx = i
    if swap_up is not None:
        i=swap_up; st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
        st.rerun()
    if swap_down is not None:
        i=swap_down; st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
        st.rerun()
    if delete_idx is not None:
        st.session_state.wps.pop(delete_idx); st.rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================= LEGS BASE =================
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

# ================= SUBLEGS (TOC/TOD) =================
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
        MHc = apply_var(THc, mag_var, mag_is_e)
        MHr = apply_var(THr, mag_var, mag_is_e)
        MHd = apply_var(THd, mag_var, mag_is_e)

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

# ================= RENDER (UI + ETO/ETE) =================
def recompute_and_render():
    if not st.session_state.legs: return
    make_sublegs()

    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: pass

    carry_efob = float(st.session_state.start_efob)
    clock = base_time
    total_sec_all = 0; total_burn_all = 0.0; efob_final = None

    for idx_leg, L in enumerate(st.session_state.legs):
        with st.expander(f"Leg {idx_leg+1} — Inputs", expanded=True):
            i1,i2,i3 = st.columns(3)
            with i1:
                L['TC']   = st.number_input(f"True Course (°T) — L{idx_leg+1}", 0.0, 359.9, float(L['TC']), step=0.1, key=f"TC_{idx_leg}")
                L['Dist'] = st.number_input(f"Distância (nm) — L{idx_leg+1}", 0.0, 500.0, float(L['Dist']), step=0.1, key=f"Dist_{idx_leg}")
            with i2:
                L['Alt0'] = st.number_input(f"Altitude INI (ft) — L{idx_leg+1}", 0.0, 30000.0, float(L['Alt0']), step=50.0, key=f"Alt0_{idx_leg}")
                L['Alt1'] = st.number_input(f"Altitude DEST (ft) — L{idx_leg+1}", 0.0, 30000.0, float(L['Alt1']), step=50.0, key=f"Alt1_{idx_leg}")
            with i3:
                L['Wfrom'] = st.number_input(f"Vento FROM (°T) — L{idx_leg+1}", 0, 360, int(L['Wfrom']), step=1, key=f"Wfrom_{idx_leg}")
                L['Wkt']   = st.number_input(f"Vento (kt) — L{idx_leg+1}", 0, 150, int(L['Wkt']), step=1, key=f"Wkt_{idx_leg}")
                L['CK']    = st.number_input(f"Checkpoints (min) — L{idx_leg+1}", 1, 10, int(L['CK']), step=1, key=f"CK_{idx_leg}")

        subs = [s for s in st.session_state.sublegs if s["parent_idx"] == idx_leg]
        if not subs: continue

        for si, s in enumerate(subs):
            t = s["Time"]; burn = s["Burn"]
            if clock:
                eto_end = (clock + dt.timedelta(seconds=t)).strftime("%H:%M")
            else:
                eto_end = ""
            s["ETO"] = eto_end; s["ETE"] = t

            efob_start = carry_efob; efob_end = max(0.0, r10f(efob_start - burn))

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            left,right = st.columns([3,2])
            with left:
                fase = {"CLIMB":"Climb","CRUISE":"Cruise/Level","DESCENT":"Descent"}[s["phase"]]
                st.subheader(f"Leg {idx_leg+1}.{si+1}: {fase}")
                st.caption(s["label"])
                st.markdown(
                    "<div class='kvrow'>"
                    + f"<div class='kv'>Alt: <b>{int(round(s['Alt0']))}→{int(round(s['Alt1']))} ft</b></div>"
                    + f"<div class='kv'>TC: <b>{rang(s['TC'])}°T</b></div>"
                    + f"<div class='kv'>MH: <b>{rang(s['MH'])}°M</b></div>"
                    + f"<div class='kv'>GS/TAS: <b>{rint(s['GS'])}/{rint(s['TAS'])} kt</b></div>"
                    + f"<div class='kv'>FF: <b>{int(FF_CONST)} L/h</b></div>"
                    + f"<div class='kv'>ETE: <b>{mmss(t)}</b></div>"
                    + (f"<div class='kv'>ETO: <b>{eto_end}</b></div>" if eto_end else "")
                    + "</div>", unsafe_allow_html=True
                )
            with right:
                st.metric("Tempo", mmss(t)); st.metric("Fuel desta sub-leg (L)", f"{burn:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

            if clock: clock = clock + dt.timedelta(seconds=t)
            carry_efob = efob_end
            total_sec_all += t; total_burn_all += burn; efob_final = efob_end

        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec_all)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{r10f(total_burn_all):.1f} L</b></div>"
        + (f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>" if efob_final is not None else "")
        + "</div>", unsafe_allow_html=True
    )

# ================= MAPA (VFR) =================
def house_polygon(lat, lon, heading_deg, w_nm=0.72, h_nm=0.92, roof_nm=0.30):
    left_lat, left_lon   = dest_point(lat, lon, heading_deg-90.0, w_nm/2.0)
    right_lat, right_lon = dest_point(lat, lon, heading_deg+90.0, w_nm/2.0)
    top_l_lat,  top_l_lon  = dest_point(left_lat,  left_lon,  heading_deg,  h_nm/2.0)
    top_r_lat,  top_r_lon  = dest_point(right_lat, right_lon, heading_deg,  h_nm/2.0)
    bot_l_lat,  bot_l_lon  = dest_point(left_lat,  left_lon,  heading_deg, -h_nm/2.0)
    bot_r_lat,  bot_r_lon  = dest_point(right_lat, right_lon, heading_deg, -h_nm/2.0)
    roof_lat, roof_lon     = dest_point((top_l_lat+top_r_lat)/2.0, (top_l_lon+top_r_lon)/2.0, heading_deg, roof_nm)
    return [[bot_l_lon, bot_l_lat],[bot_r_lon, bot_r_lat],[top_r_lon, top_r_lat],
            [roof_lon, roof_lat],[top_l_lon, top_l_lat],[bot_l_lon, bot_l_lat]]

def chip_rect_with_center(lat, lon, heading_deg, w_nm, h_nm, forward_nm):
    # devolve polygon e centro do rectângulo (para colocar texto dentro)
    c_lat, c_lon = dest_point(lat, lon, heading_deg, forward_nm)
    left_lat, left_lon   = dest_point(c_lat, c_lon, heading_deg-90.0, w_nm/2.0)
    right_lat, right_lon = dest_point(c_lat, c_lon, heading_deg+90.0, w_nm/2.0)
    top_l_lat,  top_l_lon  = dest_point(left_lat,  left_lon,  heading_deg,  h_nm/2.0)
    top_r_lat,  top_r_lon  = dest_point(right_lat, right_lon, heading_deg,  h_nm/2.0)
    bot_l_lat,  bot_l_lon  = dest_point(left_lat,  left_lon,  heading_deg, -h_nm/2.0)
    bot_r_lat,  bot_r_lon  = dest_point(right_lat, right_lon, heading_deg, -h_nm/2.0)
    poly = [[bot_l_lon, bot_l_lat],[bot_r_lon, bot_r_lat],[top_r_lon, top_r_lat],[top_l_lon, top_l_lat],[bot_l_lon, bot_l_lat]]
    return poly, (c_lat, c_lon)

def render_map_vfr():
    if len(st.session_state.wps) < 2 or not st.session_state.sublegs:
        st.info("Adiciona pelo menos 2 waypoints e gera as legs para ver o mapa.")
        return

    styles = [
        "VFR híbrido (OTM + OSM labels)",
        "OSM HOT (labels grandes)",
        "OpenTopoMap",
        "Esri Topo",
        "Satélite híbrido (Esri + OSM)",
        "Positron clean"
    ]
    try:
        default_index = styles.index(st.session_state.get("map_style", styles[0]))
    except ValueError:
        default_index = 0
    style = st.selectbox("Estilo do mapa", options=styles, index=default_index, key="map_style")
    # usar a seleção como parte da key para forçar re-render sempre que muda
    chart_key = f"deck_{style}"

    HOUSE_OFFSET_NM = 0.65
    MH_SIZE, INF_SIZE = 22, 14  # tamanhos dentro das caixas

    under_paths, over_paths, tick_data = [], [], []
    houses, chips, mh_labels, info_labels = [], [], [], []
    wp_points, wp_texts, special_points, special_texts = [], [], [], []

    # WPs marcados
    for i, W in enumerate(st.session_state.wps):
        wp_points.append({"position":[W["lon"], W["lat"]], "name": f"WP{i+1} {W['name']}"})
        wp_texts.append({"position":[W["lon"], W["lat"]], "text": f"{W['name']}"})

    # Sublegs
    for s in st.session_state.sublegs:
        A = (s["A_lat"], s["A_lon"]); B = (s["B_lat"], s["B_lon"])
        under_paths.append({"path": [[A[1],A[0]], [B[1],B[0]]]})
        over_paths.append( {"path": [[A[1],A[0]], [B[1],B[0]]], "name": f"{s['phase'].title()}"})

        # riscas 2 min
        interval_s = 120; total_t = s["Time"]; total_d = s["Dist"]; tc_here = s["TC"]
        k=1
        while k*interval_s <= total_t:
            frac = (k*interval_s)/max(total_t,1)
            d_here = total_d * frac
            latm, lonm = point_along_gc(A[0], A[1], B[0], B[1], d_here)
            half_nm = 0.12
            left_lat, left_lon   = dest_point(latm, lonm, tc_here-90, half_nm)
            right_lat, right_lon = dest_point(latm, lonm, tc_here+90, half_nm)
            tick_data.append({"path": [[left_lon, left_lat], [right_lon, right_lat]]})
            k += 1

        # dog house com caixinhas e texto DENTRO
        mid_lat, mid_lon = point_along_gc(A[0], A[1], B[0], B[1], total_d/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, s["TC"]+90, HOUSE_OFFSET_NM)
        houses.append({"polygon": house_polygon(off_lat, off_lon, s["TC"])})

        poly_top,  cen_top  = chip_rect_with_center(off_lat, off_lon, s["TC"], w_nm=0.58, h_nm=0.24, forward_nm= 0.08)  # MH
        poly_bot,  cen_bot  = chip_rect_with_center(off_lat, off_lon, s["TC"], w_nm=0.58, h_nm=0.24, forward_nm=-0.13)  # INFO
        chips.append({"polygon": poly_top})
        chips.append({"polygon": poly_bot})

        # textos: MH em cima; info em baixo (multilinha)
        mh_text  = f"{rang(s['MH'])} M"
        info_lines = [
            f"{rang(s['TC'])} T",
            f"{r10f(s['Dist'])} nm",
            f"GS {rint(s['GS'])}",
            f"ETE {mmss(s['ETE'])}" + (f"  ETO {s['ETO']}" if s.get('ETO') else "")
        ]
        info_text = "  ".join(info_lines)
        mh_labels.append(  {"position":[cen_top[1], cen_top[0]],   "text": mh_text})
        info_labels.append({"position":[cen_bot[1], cen_bot[0]],   "text": info_text})

        # TOC/TOD ao lado (não em cima da rota)
        if "TOC" in s:
            lat_toc, lon_toc = s["TOC"]
            p = dest_point(lat_toc, lon_toc, s["TC"]+90, 0.18)
            special_points.append({"position":[p[1], p[0]], "name":"TOC"})
            tocl = dest_point(p[0], p[1], s["TC"]+110, 0.16)
            special_texts.append({"position":[tocl[1], tocl[0]], "text":"TOC"})
        if "TOD" in s:
            lat_tod, lon_tod = s["TOD"]
            p = dest_point(lat_tod, lon_tod, s["TC"]+90, 0.18)
            special_points.append({"position":[p[1], p[0]], "name":"TOD"})
            todl = dest_point(p[0], p[1], s["TC"]+110, 0.16)
            special_texts.append({"position":[todl[1], todl[0]], "text":"TOD"})

    # fundos: usar LISTA de URLs (sem {s}) para evitar problemas e garantir mudança
    layers = []
    if style == "VFR híbrido (OTM + OSM labels)":
        layers.append(pdk.Layer("TileLayer",
                                data=["https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
                                      "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
                                      "https://c.tile.opentopomap.org/{z}/{x}/{y}.png"],
                                min_zoom=0, max_zoom=17, tile_size=256, opacity=1.0))
        layers.append(pdk.Layer("TileLayer",
                                data=["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                      "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                      "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"],
                                min_zoom=0, max_zoom=19, tile_size=256, opacity=0.55))
    if style == "OSM HOT (labels grandes)":
        layers.append(pdk.Layer("TileLayer",
                                data=["https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
                                      "https://b.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
                                      "https://c.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png"],
                                min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0))
    if style == "OpenTopoMap":
        layers.append(pdk.Layer("TileLayer",
                                data=["https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
                                      "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
                                      "https://c.tile.opentopomap.org/{z}/{x}/{y}.png"],
                                min_zoom=0, max_zoom=17, tile_size=256, opacity=1.0))
    if style == "Esri Topo":
        layers.append(pdk.Layer("TileLayer",
                                data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                                min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0))
    if style == "Satélite híbrido (Esri + OSM)":
        layers.append(pdk.Layer("TileLayer",
                                data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0))
        layers.append(pdk.Layer("TileLayer",
                                data=["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                      "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                      "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"],
                                min_zoom=0, max_zoom=19, tile_size=256, opacity=0.35))
    if style == "Positron clean":
        layers.append(pdk.Layer("TileLayer",
                                data="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                                min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0))

    # overlays
    layers += [
        pdk.Layer("PathLayer", data=under_paths, get_path="path", get_color=[0,0,0,255],      width_min_pixels=7),
        pdk.Layer("PathLayer", data=over_paths,  get_path="path", get_color=[206,43,216,230], width_min_pixels=5),
        pdk.Layer("PathLayer", data=tick_data,   get_path="path", get_color=[0,0,0,255],      width_min_pixels=2),
        pdk.Layer("PolygonLayer", data=houses,   get_polygon="polygon",
                  get_fill_color=[255,255,255,240], get_line_color=[0,0,0,255],
                  line_width_min_pixels=2, stroked=True, filled=True),
        pdk.Layer("PolygonLayer", data=chips,    get_polygon="polygon",
                  get_fill_color=[255,255,255,255], get_line_color=[0,0,0,255],
                  line_width_min_pixels=2, stroked=True, filled=True),
        pdk.Layer("TextLayer", data=mh_labels,   get_position="position", get_text="text",
                  get_size=MH_SIZE, get_color=[214,31,105], get_alignment_baseline="'center'"),
        pdk.Layer("TextLayer", data=info_labels, get_position="position", get_text="text",
                  get_size=INF_SIZE, get_color=[0,0,0], get_alignment_baseline="'center'"),
        pdk.Layer("ScatterplotLayer", data=wp_points, get_position="position",
                  get_radius=8, radius_units="pixels",
                  get_fill_color=[255,255,255,255], stroked=True, get_line_color=[0,0,0,255], line_width_min_pixels=2),
        pdk.Layer("TextLayer", data=wp_texts,    get_position="position", get_text="text",
                  get_size=14, get_color=[0,0,0], get_alignment_baseline="'bottom'"),
        pdk.Layer("ScatterplotLayer", data=special_points, get_position="position",
                  get_radius=9, radius_units="pixels", get_fill_color=[255,140,0,240]),
        pdk.Layer("TextLayer", data=special_texts, get_position="position", get_text="text",
                  get_size=14, get_color=[255,140,0], get_alignment_baseline="'top'")
    ]

    # centro/zoom
    all_lats = [w["lat"] for w in st.session_state.wps]; all_lons = [w["lon"] for w in st.session_state.wps]
    mean_lat = sum(all_lats)/len(all_lats); mean_lon = sum(all_lons)/len(all_lons)
    lat_span = max(all_lats) - min(all_lats) if len(all_lats)>1 else 0.5
    lon_span = max(all_lons) - min(all_lons) if len(all_lons)>1 else 0.5
    span = max(lat_span, lon_span)
    zoom = 10 if span < 0.6 else (9 if span < 1 else (8 if span < 2 else 7))
    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=zoom, pitch=0, bearing=0)

    # map_style vazio; KEY depende do estilo => força re-render
    deck = pdk.Deck(map_style="", initial_view_state=view_state, layers=layers, tooltip={"text":"{name}"})
    st.pydeck_chart(deck, use_container_width=True, key=chart_key)

# ================= RUN =================
if st.session_state.wps and not st.session_state.legs and len(st.session_state.wps)>=2:
    rebuild_legs_from_wps()
if st.session_state.legs:
    recompute_and_render()
    render_map_vfr()

