# NAVLOG v12 — VFR Map + TOC/TOD como WPs + Dog Houses triangulares + riscas de 2 min
# Requisitos: streamlit>=1.33,<2 ; pydeck>=0.8 ; pandas>=2.2 ; numpy>=1.26

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt

# ==================== CONFIG / ESTILO ====================
st.set_page_config(page_title="NAVLOG v12 — VFR", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:var(--line);margin:10px 0}
.badge{background:#fef3c7;border:1px solid #fde68a;border-radius:999px;padding:2px 8px;font-size:11px;margin-left:6px}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid var(--line);padding:8px 0}
h2.section{margin:6px 0 2px 0}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ==================== UTILS ====================
EARTH_NM = 3440.065
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang   = lambda x: int(round(float(x))) % 360
rint   = lambda x: int(round(float(x)))
clamp  = lambda v, lo, hi: max(lo, min(hi, v))

def wrap360(x): x = math.fmod(float(x), 360.0); return x + 360 if x < 0 else x
def angdiff(a, b): return (a - b + 180) % 360 - 180

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * math.sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = math.degrees(math.asin(s))
    th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

apply_var = lambda th, var, east_is_neg=False: wrap360(th - var if east_is_neg else th + var)

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

# ISA + PA
press_alt = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0
isa_temp  = lambda pa: 15.0 - 2.0*(pa/1000.0)

# ROC (AFM simplificado — igual ao teu)
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
def interp1(x, x0, x1, y0, y1):
    if x1 == x0: return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def roc_interp(pa, temp):
    pas = sorted(ROC_ENR.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    temps = [-25,0,25,50]; t = clamp(temp, temps[0], temps[-1])
    if t <= 0: t0, t1 = -25, 0
    elif t <= 25: t0, t1 = 0, 25
    else: t0, t1 = 25, 50
    v00, v01 = ROC_ENR[p0][t0], ROC_ENR[p0][t1]
    v10, v11 = ROC_ENR[p1][t0], ROC_ENR[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01); v1 = interp1(t, t0, t1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * 0.90)

# ==================== ESTADO ====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("desc_angle", 3.0)
ens("wps", [])                     # [{name,lat,lon,alt, source}]
ens("legs_phase", [])              # legs depois de dividir nos TOC/TOD
ens("search_sel", None)            # id do resultado selecionado

# Velocidades fixas (kt)
TAS_CLIMB   = 70.0
TAS_CRUISE  = 90.0
TAS_DESCENT = 90.0

# ==================== PARSE CSVs locais ====================
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
            rows.append({"source":"AD","code":ident or name, "name":name, "city":city,"lat":lat,"lon":lon})
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
            rows.append({"source":"LOC","code":code or name, "name":name, "sector":sector,"lat":lat,"lon":lon})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

try:
    ad_df  = parse_ad_df(pd.read_csv(AD_CSV))
    loc_df = parse_loc_df(pd.read_csv(LOC_CSV))
except Exception:
    ad_df  = pd.DataFrame(columns=["source","code","name","city","lat","lon"])
    loc_df = pd.DataFrame(columns=["source","code","name","sector","lat","lon"])

# ==================== HEADER ====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
hl, hr = st.columns([3,2])
with hl: st.title("NAVLOG — VFR (v12)")
with hr:
    if st.button("🗑️ Limpar tudo", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs_phase = []; st.session_state.search_sel = None
st.markdown("</div>", unsafe_allow_html=True)

# ==================== PARÂMETROS GLOBAIS ====================
with st.form("globals"):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, int(st.session_state.qnh))
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, int(st.session_state.oat))
    with c2:
        st.session_state.mag_var = st.number_input("Mag Var (°)", 0, 30, int(st.session_state.mag_var))
        st.session_state.mag_is_e = st.selectbox("Var E/W", ["W","E"], index=(1 if st.session_state.mag_is_e else 0)) == "E"
    with c3:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    with c4:
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ==================== PESQUISA COM SELEÇÃO (resolve duplicados tipo 'Nisa') ====================
st.subheader("Adicionar waypoint")
qtxt = st.text_input("🔎 Procurar (código/ident/nome/cidade/sector)", "", placeholder="Ex: NISA, LPPT, ABRANTES…")
results = pd.concat([ad_df, loc_df], ignore_index=True)
if qtxt.strip():
    tq = qtxt.lower().strip()
    results = results[results.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]
# lista cartões de resultados
for idx, r in results.head(30).iterrows():
    with st.expander(f"{r.get('code','—')} — {r.get('name','')}", expanded=False):
        st.write(f"**Fonte:** {r['source']}  |  **Lat/Lon:** {round(r['lat'],5)}, {round(r['lon'],5)}")
        if r.get('city'): st.write(f"**Cidade:** {r.get('city')}")
        if r.get('sector'): st.write(f"**Sector:** {r.get('sector')}")
        alt_def = st.number_input(f"Altitude (ft) p/ este WP", 0.0, 18000.0, 3000.0, step=100.0, key=f"alt_{idx}")
        if st.button("Adicionar este WP", key=f"add_{idx}"):
            st.session_state.wps.append({"name": str(r.get('code') or r.get('name')), "lat": float(r['lat']), "lon": float(r['lon']), "alt": float(alt_def), "source": r['source']})
            st.success(f"WP adicionado: {r.get('code') or r.get('name')}")
            st.experimental_rerun()

# Também waypoint manual
with st.expander("➕ Adicionar waypoint manual", expanded=False):
    m1, m2, m3, m4 = st.columns(4)
    with m1: nm = st.text_input("Nome", "WP")
    with m2: la = st.number_input("Lat", -90.0, 90.0, 39.5, step=0.0001, format="%.5f")
    with m3: lo = st.number_input("Lon", -180.0, 180.0, -8.0, step=0.0001, format="%.5f")
    with m4: al = st.number_input("Alt (ft)", 0.0, 18000.0, 3000.0, step=50.0)
    if st.button("Adicionar manual"):
        st.session_state.wps.append({"name": nm, "lat": float(la), "lon": float(lo), "alt": float(al), "source":"MAN"})
        st.success("WP manual adicionado.")

# Editor simples de WPs (ordenar/remover/editar alt)
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([2,2,2,2,1])
            with c1:
                w["name"] = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2:
                w["lat"]  = float(st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}"))
            with c3:
                w["lon"]  = float(st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}"))
            with c4:
                w["alt"]  = float(st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}"))
            with c5:
                if st.button("Remover", key=f"delwp_{i}"):
                    st.session_state.wps.pop(i); st.experimental_rerun()
            # mover
            m1,m2,_ = st.columns([1,1,6])
            if m1.button("↑", key=f"up{i}") and i>0:
                st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]; st.experimental_rerun()
            if m2.button("↓", key=f"dn{i}") and i < len(st.session_state.wps)-1:
                st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]; st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ==================== CONSTRUIR LEGS com TOC/TOD como WPs ====================
def split_into_phase_legs(wps, qnh, oat, mag_var, mag_is_e, wind_from, wind_kt, desc_angle):
    legs = []
    gen_wps = []  # pontos gerados (TOC/TOD) apenas p/ marcação
    for i in range(len(wps)-1):
        A, B = wps[i], wps[i+1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        pa0 = press_alt(A["alt"], qnh); pa1 = press_alt(B["alt"], qnh)

        # velocidades fixas
        _, THc, GScl = wind_triangle(tc, TAS_CLIMB,   wind_from, wind_kt)
        _, THr, GScr = wind_triangle(tc, TAS_CRUISE,  wind_from, wind_kt)
        _, THd, GSde = wind_triangle(tc, TAS_DESCENT, wind_from, wind_kt)
        MHc = apply_var(THc, mag_var, mag_is_e)
        MHr = apply_var(THr, mag_var, mag_is_e)
        MHd = apply_var(THd, mag_var, mag_is_e)

        ROC = roc_interp(pa0, oat)
        ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))  # 3° => ROD ≈ GS*5

        if B["alt"] > A["alt"] + 1e-6:   # CLIMB até nível de B
            t_need = (B["alt"] - A["alt"]) / max(ROC, 1e-6)  # min
            d_need = GScl * (t_need / 60.0)
            if d_need < dist - 1e-4:
                # Leg A -> TOC (climb)
                toc_lat, toc_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                TOC = {"name":"TOC", "lat":toc_lat, "lon":toc_lon, "alt":B["alt"], "source":"SYS"}
                gen_wps.append(TOC)
                legs.append({
                    "phase":"CLIMB", "A":A, "B":TOC, "TC":tc, "TH":THc, "MH":MHc,
                    "TAS":TAS_CLIMB, "GS":GScl, "dist":d_need, "time":rt10((d_need/GScl)*3600)
                })
                # Leg TOC -> B (cruise)
                rem = dist - d_need
                legs.append({
                    "phase":"CRUISE", "A":TOC, "B":B, "TC":tc, "TH":THr, "MH":MHr,
                    "TAS":TAS_CRUISE, "GS":GScr, "dist":rem, "time":rt10((rem/GScr)*3600)
                })
            else:
                # só climb até B
                legs.append({
                    "phase":"CLIMB", "A":A, "B":B, "TC":tc, "TH":THc, "MH":MHc,
                    "TAS":TAS_CLIMB, "GS":GScl, "dist":dist, "time":rt10((dist/GScl)*3600)
                })
        elif B["alt"] < A["alt"] - 1e-6:  # DESCENT até nível de B (TOD antes do fim)
            t_need = (A["alt"] - B["alt"]) / max(ROD, 1e-6)  # min
            d_des = GSde * (t_need / 60.0)
            if d_des < dist - 1e-4:
                # cruise primeiro, TOD, depois descendes
                tod_from_start = dist - d_des
                tod_lat, tod_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], tod_from_start)
                TOD = {"name":"TOD", "lat":tod_lat, "lon":tod_lon, "alt":A["alt"], "source":"SYS"}
                gen_wps.append(TOD)
                # Leg A -> TOD (cruise)
                legs.append({
                    "phase":"CRUISE", "A":A, "B":TOD, "TC":tc, "TH":THr, "MH":MHr,
                    "TAS":TAS_CRUISE, "GS":GScr, "dist":tod_from_start, "time":rt10((tod_from_start/GScr)*3600)
                })
                # Leg TOD -> B (descent)
                legs.append({
                    "phase":"DESCENT", "A":TOD, "B":B, "TC":tc, "TH":THd, "MH":MHd,
                    "TAS":TAS_DESCENT, "GS":GSde, "dist":d_des, "time":rt10((d_des/GSde)*3600)
                })
            else:
                # só descent até B
                legs.append({
                    "phase":"DESCENT", "A":A, "B":B, "TC":tc, "TH":THd, "MH":MHd,
                    "TAS":TAS_DESCENT, "GS":GSde, "dist":dist, "time":rt10((dist/GSde)*3600)
                })
        else:
            # nível constante: cruise
            legs.append({
                "phase":"CRUISE", "A":A, "B":B, "TC":tc, "TH":THr, "MH":MHr,
                "TAS":TAS_CRUISE, "GS":GScr, "dist":dist, "time":rt10((dist/GScr)*3600)
            })
    return legs, gen_wps

# Construir legs quando clicado
if st.button("⚙️ Construir legs com TOC/TOD", type="primary", use_container_width=True):
    if len(st.session_state.wps) < 2:
        st.warning("Adiciona pelo menos 2 waypoints.")
    else:
        legs, gen_pts = split_into_phase_legs(
            st.session_state.wps,
            st.session_state.qnh, st.session_state.oat,
            st.session_state.mag_var, st.session_state.mag_is_e,
            st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.desc_angle
        )
        st.session_state.legs_phase = legs
        # junta TOC/TOD à lista “visual” (só para marcar no mapa)
        for p in gen_pts:
            st.session_state.wps.append(p)
        st.success(f"Criadas {len(legs)} pernas (fases) — incluindo TOC/TOD.")

# ==================== UI: CARTÕES DE PERNA (inputs → fase abaixo) ====================
if st.session_state.legs_phase:
    st.subheader("Pernas (já divididas nos TOC/TOD)")
    total_time = sum(L["time"] for L in st.session_state.legs_phase)
    total_dist = sum(L["dist"] for L in st.session_state.legs_phase)
    st.markdown(
        f"<div class='kvrow'><div class='kv'>ETE Total: <b>{hhmmss(total_time)}</b></div>"
        f"<div class='kv'>Distância Total: <b>{total_dist:.1f} nm</b></div></div>", unsafe_allow_html=True
    )
    for i, L in enumerate(st.session_state.legs_phase):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Leg {i+1} — {L['phase']}**  <span class='badge'>{L['A']['name']} → {L['B']['name']}</span>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("TC / TH", f"{rang(L['TC'])}°T / {rang(L['TH'])}°")
        with c2: st.metric("**MH** (destaque)", f"{rang(L['MH'])}°M")
        with c3: st.metric("GS / TAS", f"{rint(L['GS'])} / {rint(L['TAS'])} kt")
        with c4: st.metric("ETE", mmss(L["time"]))
        c5,c6 = st.columns(2)
        with c5: st.write(f"Alt A→B: **{int(L['A']['alt'])} → {int(L['B']['alt'])} ft**")
        with c6: st.write(f"Distância: **{L['dist']:.1f} nm**")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ==================== MAPA VFR (pydeck) ====================
def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

def triangle_cap(lat, lon, heading_deg, h_nm=0.28, w_nm=0.28):
    # pequeno triângulo vermelho no topo (cap)
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, 0.25)   # ligeiramente à frente do centro
    apex_lat, apex_lon     = dest_point(lat, lon, heading_deg, 0.55)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

def build_map_layers(wps, legs):
    layers = []

    # 1) Rota (magenta)
    path_data = [{"path":[[L["A"]["lon"], L["A"]["lat"]],[L["B"]["lon"], L["B"]["lat"]]]} for L in legs]
    layers.append(pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[200,0,255,220], width_min_pixels=4))

    # 2) Riscas 2 min por fase (perpendiculares)
    ticks = []
    for L in legs:
        if L["GS"] <= 0: continue
        total_t = L["time"]  # já em segundos arredondados a 10
        if total_t < 120: continue
        tc = L["TC"]
        k = 1
        while k*120 <= total_t:
            d_nm = (L["GS"] * (k*120/3600.0))
            d_nm = min(d_nm, L["dist"])
            latm, lonm = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], d_nm)
            half = 0.15
            l_lat, l_lon = dest_point(latm, lonm, tc-90, half)
            r_lat, r_lon = dest_point(latm, lonm, tc+90, half)
            ticks.append({"path":[[l_lon,l_lat],[r_lon,r_lat]]})
            k += 1
    layers.append(pdk.Layer("PathLayer", data=ticks, get_path="path", get_color=[0,0,0,255], width_min_pixels=2))

    # 3) Dog houses triangulares + texto
    tri, cap, big_mh, small_txt = [], [], [], []
    for i, L in enumerate(legs):
        # centro deslocado para o lado da perna (para não tapar)
        mid_lat, mid_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], L["dist"]/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, L["TC"]+90, 0.30)
        tri.append({"polygon": triangle_coords(off_lat, off_lon, L["TC"])})
        cap.append({"polygon": triangle_cap(off_lat, off_lon, L["TC"])})
        # MH em grande (azul forte)
        big_mh.append({"pos":[off_lon, off_lat], "text": f"{rang(L['MH'])}°M"})
        # Linhas menores (TC / GS / ETE)
        ete_txt = mmss(L["time"])
        small_txt.append({
            "pos":[off_lon, off_lat],
            "line1": f"{rang(L['TC'])}°T",
            "line2": f"GS {rint(L['GS'])}",
            "line3": f"{ete_txt}"
        })

    layers.append(pdk.Layer("PolygonLayer", data=tri, get_polygon="polygon", stroked=True, filled=True,
                            get_fill_color=[255,255,255,235], get_line_color=[0,0,0,255], line_width_min_pixels=2))
    layers.append(pdk.Layer("PolygonLayer", data=cap, get_polygon="polygon", stroked=False, filled=True,
                            get_fill_color=[220,20,60,255]))
    # MH grande (bem visível)
    layers.append(pdk.Layer("TextLayer", data=big_mh, get_position="pos", get_text="text",
                            get_size=24, get_color=[20,20,220], get_alignment_baseline="'center'"))
    # 3 linhas pequenas
    layers.append(pdk.Layer("TextLayer", data=[{"pos":d["pos"], "text":d["line1"]} for d in small_txt],
                            get_position="pos", get_text="text", get_size=12, get_color=[0,0,0], get_alignment_baseline="'top'"))
    layers.append(pdk.Layer("TextLayer", data=[{"pos":d["pos"], "text":d["line2"]} for d in small_txt],
                            get_position="pos", get_text="text", get_size=12, get_color=[0,0,0], get_alignment_baseline="'center'"))
    layers.append(pdk.Layer("TextLayer", data=[{"pos":d["pos"], "text":d["line3"]} for d in small_txt],
                            get_position="pos", get_text="text", get_size=12, get_color=[0,0,0], get_alignment_baseline="'bottom'"))

    # 4) Waypoints bem demarcados
    wp_data = [{"pos":[w["lon"], w["lat"]], "name": w["name"], "kind": w["source"]} for w in wps]
    layers.append(pdk.Layer("ScatterplotLayer", data=wp_data, get_position="pos", get_radius_pixels=8,
                            get_fill_color=[0,120,255,255]))
    layers.append(pdk.Layer("TextLayer", data=wp_data, get_position="pos", get_text="name",
                            get_size=14, get_color=[0,0,0], get_alignment_baseline="'top'"))

    return layers

if len(st.session_state.wps) >= 2 and st.session_state.legs_phase:
    mean_lat = sum([w["lat"] for w in st.session_state.wps]) / len(st.session_state.wps)
    mean_lon = sum([w["lon"] for w in st.session_state.wps]) / len(st.session_state.wps)
    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=7, pitch=0)

    layers = build_map_layers(st.session_state.wps, st.session_state.legs_phase)

    # Basemap com características e nomes (estilo VFR-like)
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",  # cidades, rios, estradas, topónimos
        initial_view_state=view_state,
        layers=layers,
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona pelo menos 2 WPs e clica **Construir legs com TOC/TOD** para ver o mapa VFR, riscas e dog houses.")

