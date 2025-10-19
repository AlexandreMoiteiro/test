# NAVLOG v12 — VFR map, TOC/TOD como WPs, velocidades fixas + FF 20 L/h
# - TOC/TOD viram novos waypoints e as legs são divididas em sub-legs.
# - Sem RPMs: TAS fixas (70 climb, 90 cruise, 90 descent).
# - Riscas de 2 min calculadas com GS real por sub-leg.
# - "Dog houses" triangulares orientadas ao TC, como no exemplo.
# - Mapa estilo VFR (CARTO Voyager) com nomes de terras e feições visíveis.
# - MH destacado no mapa (texto grande e cor forte).
# - Pesquisa de WPs com seletor único (evita adicionar duplicados como "Nisa").

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# ===================== CONFIG / ESTILO =====================
st.set_page_config(page_title="NAVLOG v12 — VFR", layout="wide", initial_sidebar_state="collapsed")

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

# ===================== CONSTANTES =====================
# velocidades fixas (kt)
SPD_CLIMB = 70.0
SPD_CRUISE = 90.0
SPD_DESC = 90.0
# consumo fixo
FF_CONST = 20.0  # L/h
# geodesia
EARTH_NM = 3440.065

# ===================== HELPERS =====================
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang   = lambda x: int(round(float(x))) % 360
rint   = lambda x: int(round(float(x)))
r10f   = lambda x: round(float(x), 1)


def wrap360(x):
    x = math.fmod(float(x), 360.0)
    return x + 360 if x < 0 else x

def angdiff(a, b):
    return (a - b + 180) % 360 - 180


def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0:
        return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = degrees(asin(s))
    th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
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
    if total <= 0:
        return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)

# ===================== AFM (ROC apenas) =====================
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
ROC_FACTOR = 0.90
VY = {0:67,2000:67,4000:67,6000:67,8000:67,10000:67,12000:67,14000:67}
press_alt  = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0
isa_temp   = lambda pa: 15.0 - 2.0*(pa/1000.0)

def interp1(x, x0, x1, y0, y1):  # linear
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def roc_interp(pa, temp):
    pas = sorted(ROC_ENR.keys())
    pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c])
    p1 = min([p for p in pas if p >= pa_c])
    temps = [-25,0,25,50]
    t = clamp(temp, temps[0], temps[-1])
    if t <= 0:
        t0, t1 = -25, 0
    elif t <= 25:
        t0, t1 = 0, 25
    else:
        t0, t1 = 25, 50
    v00, v01 = ROC_ENR[p0][t0], ROC_ENR[p0][t1]
    v10, v11 = ROC_ENR[p1][t0], ROC_ENR[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01)
    v1 = interp1(t, t0, t1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * ROC_FACTOR)

# ===================== STATE =====================
ss = st.session_state
ss.setdefault("qnh", 1013)
ss.setdefault("oat", 15)
ss.setdefault("mag_var", 1)
ss.setdefault("mag_is_e", False)
ss.setdefault("start_clock", "")
ss.setdefault("start_efob", 85.0)
ss.setdefault("ck_default", 2)
ss.setdefault("wind_from", 0)
ss.setdefault("wind_kt", 0)
ss.setdefault("wps", [])        # [{name,lat,lon,alt}]

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([3,3,3,3])
with col1:
    st.title("NAVLOG — v12 (VFR)")
with col2:
    st.caption("Velocidades fixas: Climb 70 kt • Cruise 90 kt • Descida 90 kt · FF 20 L/h")
with col3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        ss.wps.append({"name": f"WP{len(ss.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with col4:
    if st.button("🗑️ Limpar rota", use_container_width=True):
        ss.wps = []
st.markdown("</div>", unsafe_allow_html=True)

# ===================== PARÂMETROS =====================
with st.form("globals"):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        ss.qnh = st.number_input("QNH (hPa)", 900, 1050, int(ss.qnh))
        ss.oat = st.number_input("OAT (°C)", -40, 50, int(ss.oat))
    with p2:
        ss.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(ss.start_efob), step=0.5)
        ss.start_clock = st.text_input("Hora off-blocks (HH:MM)", ss.start_clock)
    with p3:
        ss.mag_var = st.number_input("Variação magnética (°)", -30.0, 30.0, float(ss.mag_var), step=0.1)
        ss.mag_is_e = st.toggle("Variação Este é negativa?", value=bool(ss.mag_is_e))
    with p4:
        ss.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(ss.wind_from), step=1)
        ss.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(ss.wind_kt), step=1)
    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== LER CSVs LOCAIS + SELETOR DE RESULTADO =====================
AD_CSV  = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"

# Conversor DMS -> DD (tokens tipo 392500N, 0081230W ou com pontos)

def dms_to_dd(token: str, is_lon=False):
    token = str(token).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([NSEW])$", token, re.I)
    if not m:
        return None
    value, hemi = m.groups()
    # aceita DDMMSS[.s]
    if is_lon:
        deg = int(value[0:3]); minutes = int(value[3:5]); seconds = float(value[5:] or 0)
    else:
        deg = int(value[0:2]); minutes = int(value[2:4]); seconds = float(value[4:] or 0)
    dd = deg + minutes/60 + seconds/3600
    if hemi.upper() in ["S","W"]:
        dd = -dd
    return dd


def parse_ad_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or s.startswith(("Ident", "DEP/")):
            continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok = coord_toks[-2]; lon_tok = coord_toks[-1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            ident = tokens[0] if re.match(r"^[A-Z0-9]{4,}$", tokens[0]) else None
            try:
                name = " ".join(tokens[1:tokens.index(coord_toks[0])]).strip()
            except:
                name = " ".join(tokens[1:]).strip()
            try:
                lon_idx = tokens.index(lon_tok); city = " ".join(tokens[lon_idx+1:]) or None
            except:
                city = None
            rows.append({"type":"AD","code":ident or name, "name":name, "city":city,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])


def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or "Total de registos" in s:
            continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok, lon_tok = coord_toks[0], coord_toks[1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            try:
                lon_idx = tokens.index(lon_tok)
            except ValueError:
                continue
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

cflt1, cflt2, cbtn = st.columns([3,3,2])
with cflt1:
    qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, LP0078…")
with cflt2:
    alt_wp = st.number_input("Altitude para WPs adicionados (ft)", 0.0, 18000.0, 3000.0, step=100.0)
with cbtn:
    st.write("")

# filtro

def filter_df(df, q):
    if not q:
        return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

ad_f  = filter_df(ad_df, qtxt)
loc_f = filter_df(loc_df, qtxt)
comb_df = pd.concat([ad_f, loc_f], ignore_index=True)

# seletor ÚNICO (resolve o caso "Nisa")
if not comb_df.empty:
    comb_df = comb_df.assign(
        label=comb_df.apply(lambda r: f"{r['type']} — {r['code']} — {r.get('name','')} {('('+str(r.get('sector',''))+')' if r.get('sector') else '')}  [lat {r['lat']:.5f}, lon {r['lon']:.5f}]", axis=1)
    )
    sel = st.selectbox("Selecionar waypoint a adicionar:", options=comb_df.index.tolist(), format_func=lambda i: comb_df.loc[i, 'label'])
    if st.button("Adicionar selecionado"):
        r = comb_df.loc[sel]
        ss.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
        st.success(f"Adicionado WP: {r['code']}")

# ===================== EDITOR DE WPs =====================
if ss.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(ss.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([2,2,2,1,1])
            with c1:
                name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2:
                lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3:
                lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4:
                alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                up = st.button("↑", key=f"up{i}"); dn = st.button("↓", key=f"dn{i}")
                if up and i>0:
                    ss.wps[i-1], ss.wps[i] = ss.wps[i], ss.wps[i-1]
                    st.experimental_rerun()
                if dn and i < len(ss.wps)-1:
                    ss.wps[i+1], ss.wps[i] = ss.wps[i], ss.wps[i+1]
                    st.experimental_rerun()
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                ss.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                ss.wps.pop(i)
                st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== CÁLCULO: SUB-LEGS COM TOC/TOD INSERIDOS =====================

def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]


def build_display_route(wps, wind_from, wind_kt, qnh, oat, mag_var, mag_is_e, ck_min=2, desc_angle_deg=3.0):
    """Devolve:
    - disp_points: lista de pontos (WPs + TOC/TOD inseridos)
    - sublegs: lista de sub-legs separadas (cada uma com TC, TH, MH, GS, tempo, etc.)
    """
    disp_points = []
    sublegs = []
    if len(wps) < 2:
        return disp_points, sublegs

    # helpers para GS e headings por TAS
    def heads(tc):
        _, THc, GSc = wind_triangle(tc, SPD_CLIMB,  wind_from, wind_kt)
        _, THr, GSr = wind_triangle(tc, SPD_CRUISE, wind_from, wind_kt)
        _, THd, GSd = wind_triangle(tc, SPD_DESC,   wind_from, wind_kt)
        MHc = apply_var(THc, mag_var, mag_is_e)
        MHr = apply_var(THr, mag_var, mag_is_e)
        MHd = apply_var(THd, mag_var, mag_is_e)
        return (THc, MHc, GSc), (THr, MHr, GSr), (THd, MHd, GSd)

    for i in range(len(wps)-1):
        A = wps[i]; B = wps[i+1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        (THc, MHc, GScl), (THr, MHr, GScr), (THd, MHd, GSde) = heads(tc)

        pa0 = press_alt(A["alt"], qnh)
        roc = roc_interp(pa0, oat)  # ft/min
        rod = max(100.0, GSde * 5.0 * (desc_angle_deg/3.0))  # ft/min

        profile = "LEVEL" if abs(B["alt"] - A["alt"]) < 1e-3 else ("CLIMB" if B["alt"] > A["alt"] else "DESCENT")

        # adicionar A ao disp_points (uma vez)
        if i == 0:
            disp_points.append({"name": A["name"], "lat": A["lat"], "lon": A["lon"], "alt": A["alt"]})

        if profile == "CLIMB":
            alt_need = B["alt"] - A["alt"]
            t_climb_min = alt_need / max(roc, 1e-6)  # minutos
            d_climb = GScl * (t_climb_min/60.0)
            if d_climb < dist:
                toc_lat, toc_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_climb)
                toc = {"name": f"TOC{i+1}", "lat": toc_lat, "lon": toc_lon, "alt": B["alt"]}
                disp_points.append(toc)
                # subleg climb
                sublegs.append({
                    "from": A, "to": toc, "dist": d_climb, "TC": tc,
                    "TH": THc, "MH": MHc, "GS": GScl, "TAS": SPD_CLIMB,
                    "label": "Climb → TOC", "time_s": rt10(t_climb_min*60),
                })
                # subleg cruise
                rem = dist - d_climb
                t_cru_s = rt10((rem/max(GScr,1e-9))*3600)
                sublegs.append({
                    "from": toc, "to": B, "dist": rem, "TC": tc,
                    "TH": THr, "MH": MHr, "GS": GScr, "TAS": SPD_CRUISE,
                    "label": "Cruise", "time_s": t_cru_s,
                })
            else:
                # sobe a viagem toda, sem TOC
                t_all = rt10((dist/max(GScl,1e-9))*3600)
                sublegs.append({
                    "from": A, "to": B, "dist": dist, "TC": tc,
                    "TH": THc, "MH": MHc, "GS": GScl, "TAS": SPD_CLIMB,
                    "label": "Climb (não atinge)", "time_s": t_all,
                })
        elif profile == "DESCENT":
            alt_drop = A["alt"] - B["alt"]
            t_desc_min = alt_drop / max(rod, 1e-6)
            d_desc = GSde * (t_desc_min/60.0)
            if d_desc < dist:
                # TOD está a dist - d_desc a partir de A
                todfar = dist - d_desc
                tod_lat, tod_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], todfar)
                tod = {"name": f"TOD{i+1}", "lat": tod_lat, "lon": tod_lon, "alt": A["alt"]}
                disp_points.append(tod)
                # primeiro cruise até TOD
                t_cru_s = rt10(((dist - d_desc)/max(GScr,1e-9))*3600)
                sublegs.append({
                    "from": A, "to": tod, "dist": dist - d_desc, "TC": tc,
                    "TH": THr, "MH": MHr, "GS": GScr, "TAS": SPD_CRUISE,
                    "label": "Cruise → TOD", "time_s": t_cru_s,
                })
                # depois descida
                sublegs.append({
                    "from": tod, "to": B, "dist": d_desc, "TC": tc,
                    "TH": THd, "MH": MHd, "GS": GSde, "TAS": SPD_DESC,
                    "label": "Descent", "time_s": rt10(t_desc_min*60),
                })
            else:
                # desce desde já, sem TOD a meio
                t_all = rt10((dist/max(GSde,1e-9))*3600)
                sublegs.append({
                    "from": A, "to": B, "dist": dist, "TC": tc,
                    "TH": THd, "MH": MHd, "GS": GSde, "TAS": SPD_DESC,
                    "label": "Descent (contínua)", "time_s": t_all,
                })
        else:
            # LEVEL
            t_cru_s = rt10((dist/max(GScr,1e-9))*3600)
            sublegs.append({
                "from": A, "to": B, "dist": dist, "TC": tc,
                "TH": THr, "MH": MHr, "GS": GScr, "TAS": SPD_CRUISE,
                "label": "Cruise/Level", "time_s": t_cru_s,
            })

        # no fim de cada par, adicionar B (último será adicionado fora do ciclo)
        disp_points.append({"name": B["name"], "lat": B["lat"], "lon": B["lon"], "alt": B["alt"]})

    # enrich: burn e checkpoints (2 min)
    for s in sublegs:
        s["burn_L"] = r10f(FF_CONST * (s["time_s"]/3600.0))
        # gera riscas de 2 min
        ticks = []
        step = 120  # s
        k = 1
        while k*step <= s["time_s"]:
            t = k*step
            d = s["GS"] * (t/3600.0)
            ticks.append({"t": t, "dist": d})
            k += 1
        s["ticks"] = ticks
    return disp_points, sublegs


# construir rota
points, sublegs = build_display_route(
    ss.wps, ss.wind_from, ss.wind_kt, ss.qnh, ss.oat, ss.mag_var, ss.mag_is_e, ck_min=ss.ck_default
)

# ===================== RESUMO TEXTO =====================
if sublegs:
    total_time = sum(s["time_s"] for s in sublegs)
    total_burn = r10f(sum(s["burn_L"] for s in sublegs))
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>Sub-legs: <b>{len(sublegs)}</b></div>"
        + f"<div class='kv'>ETE Total: <b>{hhmmss(total_time)}</b></div>"
        + f"<div class='kv'>Burn Total: <b>{total_burn:.1f} L</b></div>"
        + "</div>", unsafe_allow_html=True
    )

    # cartões por sub-leg
    efob = float(ss.start_efob)
    for idx, s in enumerate(sublegs, 1):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([3,2])
        with left:
            st.subheader(f"Leg {idx}: {s['label']}")
            st.caption(f"{s['from']['name']} → {s['to']['name']}")
            st.markdown(
                "<div class='kvrow'>"
                + f"<div class='kv'>Alt: <b>{int(round(s['from']['alt']))}→{int(round(s['to']['alt']))} ft</b></div>"
                + f"<div class='kv'>TC/TH/MH: <b>{rang(s['TC'])}T / {rang(s['TH'])}T / {rang(s['MH'])}M</b></div>"
                + f"<div class='kv'>GS/TAS: <b>{rint(s['GS'])}/{rint(s['TAS'])} kt</b></div>"
                + f"<div class='kv'>Dist: <b>{s['dist']:.1f} nm</b></div>"
                + "</div>", unsafe_allow_html=True
            )
        with right:
            st.metric("Tempo", mmss(s["time_s"]))
            st.metric("Fuel desta perna (L)", f"{s['burn_L']:.1f}")
            efob_end = max(0.0, r10f(efob - s['burn_L']))
            st.caption(f"EFOB → {efob_end:.1f} L")
            efob = efob_end
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== MAPA (VFR) =====================
if len(points) >= 2 and sublegs:
    # camadas
    path_data = []
    tick_data = []
    tri_data  = []
    mh_labels = []  # MH grande
    info_labels = []
    wp_points = []
    wp_labels = []

    # WPs (inclui TOC/TOD já inseridos)
    for p in points:
        wp_points.append({"position":[p["lon"], p["lat"]]})
        wp_labels.append({"position":[p["lon"], p["lat"]], "text": p["name"]})

    # Sub-legs, cada uma com dog house e riscas de 2 min
    for s in sublegs:
        A = s["from"]; B = s["to"]
        path_data.append({"path": [[A["lon"], A["lat"]], [B["lon"], B["lat"]]], "name": f"{A['name']} → {B['name']}"})

        # riscas a cada 2 min usando GS desta sub-leg
        for tk in s["ticks"]:
            d = min(tk["dist"], s["dist"])  # limitar à perna
            latm, lonm = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d)
            tc = s["TC"]
            half_nm = 0.15
            left_lat, left_lon   = dest_point(latm, lonm, tc-90, half_nm)
            right_lat, right_lon = dest_point(latm, lonm, tc+90, half_nm)
            tick_data.append({"path": [[left_lon, left_lat], [right_lon, right_lat]]})

        # dog house
        mid_lat, mid_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], s["dist"]/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, s["TC"]+90, 0.30)
        tri = triangle_coords(off_lat, off_lon, s["TC"], h_nm=0.9, w_nm=0.65)
        tri_data.append({"polygon": tri})

        # etiquetas — MH GRANDE + detalhe menor
        # MH destaque (cor forte, maior)
        mh_labels.append({
            "position": [off_lon, off_lat],
            "text": f"MH {rang(s['MH'])}°",
            "size": 22,
        })
        # detalhe ao lado
        lab_lat, lab_lon = dest_point(off_lat, off_lon, s["TC"]+90, 0.35)
        info = f"{rang(s['TH'])}T • {s['dist']:.1f}nm • GS {rint(s['GS'])} • ETE {mmss(s['time_s'])}"
        info_labels.append({"position":[lab_lon, lab_lat], "text": info, "size": 13})

    # Layers
    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[180, 0, 255, 220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0, 0, 0, 255], width_min_pixels=2)
    tri_layer   = pdk.Layer(
        "PolygonLayer",
        data=tri_data,
        get_polygon="polygon",
        get_fill_color=[255,255,255,230],
        get_line_color=[0,0,0,255],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
    )
    # MH em cor bem visível (azul/teal forte)
    mh_layer = pdk.Layer(
        "TextLayer",
        data=mh_labels,
        get_position="position",
        get_text="text",
        get_size="size",
        get_color=[0,120,255],
        get_alignment_baseline="'center'",
        get_pixel_offset=[0,0],
    )
    info_layer  = pdk.Layer(
        "TextLayer", data=info_labels, get_position="position", get_text="text",
        get_size="size", get_color=[0,0,0], get_alignment_baseline="'center'"
    )
    # WPs marcados + nomes
    wp_layer = pdk.Layer(
        "ScatterplotLayer",
        data=wp_points,
        get_position="position",
        get_radius_units='meters',
        get_radius=80,
        get_fill_color=[255,0,120,200],
        get_line_color=[0,0,0,255],
        line_width_min_pixels=1,
    )
    wp_text_layer = pdk.Layer(
        "TextLayer", data=wp_labels, get_position="position", get_text="text", get_size=12, get_color=[20,20,20]
    )

    # vista centrada
    mean_lat = sum([p["lat"] for p in points])/len(points)
    mean_lon = sum([p["lon"] for p in points])/len(points)

    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=7.2, pitch=0)
    # Mapa estilo VFR (CARTO Voyager traz hidrografia, estradas e nomes)
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        initial_view_state=view_state,
        layers=[route_layer, ticks_layer, tri_layer, wp_layer, wp_text_layer, mh_layer, info_layer],
        tooltip={"text": "{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona pelo menos 2 waypoints para ver o mapa, as dog houses e as riscas de 2 minutos.")
