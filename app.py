# app.py — NAVLOG v12 (AFM)
# PyDeck + Dog-houses triangulares + riscas cada 2 min + TOC/TOD viram novos WPs/legs
# Velocidades FIXAS: 70 kt (Climb), 90 kt (Cruise/Descent)
# Mapa com labels (Carto Voyager GL)

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG v12 — PyDeck + Triângulos", layout="wide", initial_sidebar_state="collapsed")

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

# ===================== CONSTANTES (velocidades fixas) =====================
TAS_CLIMB = 70.0     # kt
TAS_CRUISE = 90.0    # kt
TAS_DESCENT = 90.0   # kt

# ===================== UTILS =====================
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

# ===================== AFM simplificado (apenas FF/ROC/ROD, TAS é fixo) =====================
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
ROC_FACTOR = 0.90
CRUISE = {
    0:{1800:(82,15.3),1900:(89,17.0),2000:(95,18.7),2100:(101,20.7),2250:(110,24.6),2388:(118,27.7)},
    2000:{1800:(81,15.5),1900:(87,17.0),2000:(93,18.8),2100:(99,20.9),2250:(108,25.0)},
    4000:{1800:(79,15.2),1900:(86,16.5),2000:(92,18.1),2100:(98,19.2),2250:(106,23.9)},
    6000:{1800:(78,14.9),1900:(85,16.1),2000:(91,17.5),2100:(97,19.2),2250:(105,22.7)},
    8000:{1800:(78,14.9),1900:(84,15.7),2000:(90,17.0),2100:(96,18.5),2250:(104,21.5)},
    10000:{1800:(78,15.5),1900:(82,15.5),2000:(89,16.6),2100:(95,17.9),2250:(103,20.5)},
}
isa_temp   = lambda pa: 15.0 - 2.0*(pa/1000.0)
press_alt  = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0
def interp1(x, x0, x1, y0, y1):
    if x1 == x0: return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
def clamp(v, lo, hi): return max(lo, min(hi, v))

def cruise_lookup_ff_only(pa, rpm, oat, weight):
    """Devolve só FF; TAS é fixa e não usada."""
    rpm = min(int(rpm), 2265)
    pas = sorted(CRUISE.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    table0 = CRUISE[p0]; table1 = CRUISE[p1]
    def v(tab):
        rpms = sorted(tab.keys())
        if rpm in tab: return tab[rpm][1]
        if rpm < rpms[0]: lo, hi = rpms[0], rpms[1]
        elif rpm > rpms[-1]: lo, hi = rpms[-2], rpms[-1]
        else: lo = max([r for r in rpms if r <= rpm]); hi = min([r for r in rpms if r >= rpm])
        ff_lo, ff_hi = tab[lo][1], tab[hi][1]
        t = (rpm - lo) / (hi - lo) if hi != lo else 0
        return ff_lo + t*(ff_hi - ff_lo)
    ff0 = v(table0); ff1 = v(table1)
    ff = interp1(pa_c, p0, p1, ff0, ff1)
    # pequena correção com OAT (mantém comportamento prévio)
    if oat is not None:
        dev = float(oat) - isa_temp(pa_c)
        if dev > 0:  ff *= 1 - 0.025*(dev/15.0)
        elif dev < 0: ff *= 1 + 0.03*((-dev)/15.0)
    return max(0.0, ff)

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
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * ROC_FACTOR)

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0)
ens("rpm_climb", 2250); ens("rpm_desc", 1800)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("computed_by_leg", [])

# ===================== TIMELINE (texto) =====================
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
    st.caption(f"GS {rint(seg['GS'])} kt · TAS {rint(seg['TAS'])} kt · FF {rint(seg['ff'])} L/h  |  {start_label} → {end_label}")

def phase_label(name):
    n = name.lower()
    if "climb" in n:   return "Climb"
    if "descent" in n: return "Descent"
    if "hold" in n:    return "Hold"
    return "Cruise/Level"

# ===================== CÁLCULO POR LEG (fases) =====================
def build_segments(tc, dist, alt0, alt1, wfrom, wkt, ck_min, params, rpm_cruise_leg, hold_min=0.0, hold_ff_input=0.0):
    """TAS são FIXAS (70/90/90)."""
    qnh, oat, mag_var, mag_is_e = params['qnh'], params['oat'], params['mag_var'], params['mag_is_e']
    rpm_climb, rpm_desc, desc_angle, weight = params['rpm_climb'], params['rpm_desc'], params['desc_angle'], params['weight']

    pa0 = press_alt(alt0, qnh); pa1 = press_alt(alt1, qnh); pa_avg = (pa0 + pa1)/2.0

    ROC = roc_interp(pa0, oat)  # ft/min
    FF_climb = cruise_lookup_ff_only((pa0 + pa1)/2.0, int(rpm_climb), oat, weight)
    FF_cru   = cruise_lookup_ff_only(pa1, int(rpm_cruise_leg), oat, weight)
    FF_desc  = cruise_lookup_ff_only(pa_avg, int(rpm_desc), oat, weight)

    # TAS fixas
    TAS_climb, TAS_cru, TAS_desc = TAS_CLIMB, TAS_CRUISE, TAS_DESCENT

    _, THc, GScl = wind_triangle(tc, TAS_climb, wfrom, wkt)
    _, THr, GScr = wind_triangle(tc, TAS_cru,  wfrom, wkt)
    _, THd, GSde = wind_triangle(tc, TAS_desc, wfrom, wkt)

    MHc = apply_var(THc, mag_var, mag_is_e)
    MHr = apply_var(THr, mag_var, mag_is_e)
    MHd = apply_var(THd, mag_var, mag_is_e)

    ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))  # ft/min

    profile = "LEVEL" if abs(alt1 - alt0) < 1e-6 else ("CLIMB" if alt1 > alt0 else "DESCENT")
    segs, toc_tod_marker = [], None

    if profile == "CLIMB":
        t_need = (alt1 - alt0) / max(ROC, 1e-6)  # min
        d_need = GScl * (t_need / 60.0)
        if d_need <= dist + 1e-6:
            # como já "explodimos" a leg com TOC em WP, esta leg deverá ser só Climb
            tA = rt10((dist / max(GScl,1e-9)) * 3600)
            segs.append({"name":"Climb","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_climb,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt1})
            toc_tod_marker = {"type":"TOC","t": rt10(t_need*60)}  # informativo
        else:
            # não atinge
            tA = rt10((dist / max(GScl,1e-9)) * 3600)
            gained = ROC * (tA / 60.0)
            segs.append({"name":"Climb (não atinge)","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_climb,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0+gained})
    elif profile == "DESCENT":
        t_need = (alt0 - alt1) / max(ROD, 1e-6)
        d_need = GSde * (t_need / 60.0)
        if d_need <= dist + 1e-6:
            tA = rt10((dist / max(GSde,1e-9)) * 3600)
            segs.append({"name":"Descent","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_desc,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt1})
            toc_tod_marker = {"type":"TOD","t": rt10(t_need*60)}  # informativo
        else:
            tA = rt10((dist / max(GSde,1e-9)) * 3600)
            lost = ROD * (tA / 60.0)
            segs.append({"name":"Descent (não atinge)","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_desc,"time":tA,"dist":dist,"alt0":alt0,"alt1":max(0.0, alt0 - lost)})
    else:
        tA = rt10((dist / max(GScr,1e-9)) * 3600)
        segs.append({"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_cru,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0})

    # HOLD opcional
    hold_min = max(0.0, float(hold_min))
    if hold_min > 0.0:
        hold_ff = float(hold_ff_input)
        if hold_ff <= 0:
            hold_ff = cruise_lookup_ff_only(press_alt(alt1, qnh), int(rpm_cruise_leg), oat, weight)
        hold_sec = rt10(hold_min * 60.0)
        end_alt = segs[-1]["alt1"] if segs else alt1
        segs.append({"name":"Hold/Espera","TH":segs[-1]["TH"] if segs else tc,"MH":segs[-1]["MH"] if segs else tc,
                     "GS":0.0,"TAS":0.0,"ff":hold_ff,"time":hold_sec,"dist":0.0,"alt0":end_alt,"alt1":end_alt})

    # burn por segmento
    for s in segs:
        s["burn"] = s["ff"] * (s["time"] / 3600.0)

    tot_sec  = sum(s['time'] for s in segs)
    tot_burn = r10f(sum(s['burn'] for s in segs))

    def cps(seg, every_min, base_clk, efob_start):
        out = []; t = 0
        if every_min <= 0: return out
        while t + every_min*60 <= seg['time']:
            t += every_min*60
            d = seg['GS']*(t/3600.0)
            burn = seg['ff']*(t/3600.0)
            eto = (base_clk + dt.timedelta(seconds=t)).strftime('%H:%M') if base_clk else ""
            efob = max(0.0, r10f(efob_start - burn))
            out.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto,"efob":efob})
        return out

    return {
        "segments": segs,
        "tot_sec": tot_sec,
        "tot_burn": tot_burn,
        "roc": roc_interp(pa0, oat),
        "rod": max(100.0, wind_triangle(tc, TAS_DESCENT, 0, 0)[2] * 5.0 * (params['desc_angle']/3.0)),
        "toc_tod": toc_tod_marker,
        "ck_func": cps
    }

# ===================== CRUD / BUILD LEGS =====================
def add_leg(TC, Dist, Alt0, Alt1, Wfrom, Wkt, CK, RPMcru):
    st.session_state.legs.append(dict(
        TC=float(TC), Dist=float(Dist), Alt0=float(Alt0), Alt1=float(Alt1),
        Wfrom=int(Wfrom), Wkt=int(Wkt), CK=int(CK), HoldMin=0.0, HoldFF=0.0, RPMcru=int(RPMcru)
    ))
def update_leg(leg_ref, **vals): leg_ref.update(vals)
def delete_leg(idx): st.session_state.legs.pop(idx)

# ===================== PARSE AD/LOCALIDADES (CSV locais) =====================
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

# ===================== TOC/TOD → inserção de WAYPOINTS e criação de LEGS =====================
def explode_wps_with_toc_tod_and_build_legs(rpm_default, wind_from, wind_kt, ck_default, desc_angle):
    """Analisa cada par de WPs; insere WP TOC/TOD quando aplicável e gera legs unitárias."""
    if len(st.session_state.wps) < 2:
        st.session_state.legs = []; return

    # vamos trabalhar numa cópia, inserindo WPs toc/tod
    wps = [dict(w) for w in st.session_state.wps]
    i = 0
    while i < len(wps) - 1:
        A, B = wps[i], wps[i+1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist_total = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])

        if dist_total < 1e-4:
            i += 1
            continue

        # climb?
        if B["alt"] > A["alt"] + 1:
            _, _, GScl = wind_triangle(tc, TAS_CLIMB, wind_from, wind_kt)
            pa0 = press_alt(A["alt"], st.session_state.qnh)
            ROC = roc_interp(pa0, st.session_state.oat)
            t_need_min = (B["alt"] - A["alt"]) / max(ROC, 1e-6)
            d_need = GScl * (t_need_min / 60.0)
            if d_need < dist_total - 0.1:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                wps.insert(i+1, {"name": f"TOC{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
                continue  # reavaliar par (A → TOC) antes de avançar

        # descent?
        if B["alt"] < A["alt"] - 1:
            _, _, GSde = wind_triangle(tc, TAS_DESCENT, wind_from, wind_kt)
            ROD = max(100.0, GSde * 5.0 * (desc_angle/3.0))
            t_need_min = (A["alt"] - B["alt"]) / max(ROD, 1e-6)
            d_need = GSde * (t_need_min / 60.0)
            if d_need < dist_total - 0.1:
                # TOD a dist_total - d_need a partir de A
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], dist_total - d_need)
                wps.insert(i+1, {"name": f"TOD{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
                continue

        i += 1  # nada a inserir -> avança

    # construir legs sequenciais
    st.session_state.legs = []
    for j in range(len(wps)-1):
        A, B = wps[j], wps[j+1]
        tc = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        add_leg(tc, dist, A["alt"], B["alt"], wind_from, wind_kt, ck_default, rpm_default)

    # substituir rota pelos WPs “explodidos” (passam a incluir TOC/TOD)
    st.session_state.wps = wps

# ===================== RECOMPUTE POR LEG =====================
def recompute_all_by_leg():
    st.session_state.computed_by_leg = []
    params = dict(
        qnh=st.session_state.qnh, oat=st.session_state.oat,
        mag_var=st.session_state.mag_var, mag_is_e=st.session_state.mag_is_e,
        rpm_climb=st.session_state.rpm_climb, rpm_desc=st.session_state.rpm_desc,
        desc_angle=st.session_state.desc_angle, weight=st.session_state.weight
    )

    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None

    carry_efob = float(st.session_state.start_efob)
    clock = base_time

    for leg in st.session_state.legs:
        res = build_segments(
            tc=leg['TC'], dist=leg['Dist'], alt0=leg['Alt0'], alt1=leg['Alt1'],
            wfrom=leg['Wfrom'], wkt=leg['Wkt'], ck_min=leg['CK'],
            params=params, rpm_cruise_leg=leg['RPMcru'],
            hold_min=leg.get('HoldMin',0.0), hold_ff_input=leg.get('HoldFF',0.0)
        )

        phases = []
        t_cursor = 0
        for idx_seg, seg in enumerate(res["segments"]):
            efob_start = carry_efob
            efob_end   = max(0.0, r10f(efob_start - seg['burn']))
            if clock:
                c_start = (clock + dt.timedelta(seconds=t_cursor)).strftime('%H:%M')
                c_end   = (clock + dt.timedelta(seconds=t_cursor + seg['time'])).strftime('%H:%M')
            else:
                c_start = f"T+{mmss(t_cursor)}"; c_end = f"T+{mmss(t_cursor + seg['time'])}"
            base_k = (clock + dt.timedelta(seconds=t_cursor)) if clock else None
            cps = res["ck_func"](seg, int(leg['CK']), base_k, efob_start) if seg['GS']>0 else []

            phases.append({
                "name": seg["name"], "label": phase_label(seg["name"]),
                "TH": seg["TH"], "MH": seg["MH"], "GS": seg["GS"], "TAS": seg["TAS"], "ff": seg["ff"],
                "time": seg["time"], "dist": seg["dist"], "alt0": seg["alt0"], "alt1": seg["alt1"],
                "burn": r10f(seg["burn"]), "efob_start": efob_start, "efob_end": efob_end,
                "clock_start": c_start, "clock_end": c_end, "cps": cps,
                "toc_tod": (res["toc_tod"] if idx_seg==0 and ("Climb" in seg["name"] or "Descent" in seg["name"]) else None),
                "roc": res["roc"], "rod": res["rod"], "rpm_cruise_leg": leg["RPMcru"], "ck": leg["CK"]
            })
            t_cursor += seg['time']; carry_efob = efob_end

        if clock: clock = clock + dt.timedelta(seconds=sum(s['time'] for s in res["segments"]))
        st.session_state.computed_by_leg.append({
            "leg_ref": leg, "phases": phases,
            "tot_sec": sum(p["time"] for p in phases),
            "tot_burn": r10f(sum(p["burn"] for p in phases))
        })

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,2,3,2])
with h1: st.title("NAVLOG — v12 (AFM)")
with h2: st.toggle("Mostrar TIMELINE/CPs", key="show_timeline", value=st.session_state.show_timeline)
with h3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.computed_by_leg = []
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
        st.session_state.weight = st.number_input("Peso (kg)", 450.0, 700.0, float(st.session_state.weight), step=1.0)
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
    with p4:
        st.session_state.rpm_climb = st.number_input("Climb RPM (global)", 1800, 2265, int(st.session_state.rpm_climb), step=5)
        st.session_state.rpm_desc  = st.number_input("Descent RPM (global)",1600, 2265, int(st.session_state.rpm_desc),  step=5)
        st.session_state.ck_default = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default), step=1)
    w1, w2 = st.columns(2)
    with w1: st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
    with w2: st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    submitted = st.form_submit_button("Aplicar parâmetros")
if submitted and st.session_state.legs:
    for L in st.session_state.legs:
        L["Wfrom"] = int(st.session_state.wind_from); L["Wkt"] = int(st.session_state.wind_kt)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== LER CSVs LOCAIS + ADICIONAR WPs POR PESQUISA =====================
try:
    ad_raw  = pd.read_csv(AD_CSV);  loc_raw = pd.read_csv(LOC_CSV)
    ad_df  = parse_ad_df(ad_raw);   loc_df  = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

cflt1, cflt2, cbtn = st.columns([3,3,1.5])
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, LP0078…")
with cflt2: alt_wp = st.number_input("Altitude para WPs adicionados (ft)", 0.0, 18000.0, 3000.0, step=100.0)
with cbtn: add_sel = st.button("Adicionar resultados")

def filter_df(df, q):
    if not q: return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

ad_f  = filter_df(ad_df, qtxt)
loc_f = filter_df(loc_df, qtxt)
if add_sel:
    for _, r in pd.concat([ad_f, loc_f]).iterrows():
        st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
    st.success(f"Adicionados {len(ad_f)+len(loc_f)} WPs.")

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
                up = st.button("↑", key=f"up{i}"); dn = st.button("↓", key=f"dn{i}")
                if up and i>0: st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]; st.experimental_rerun()
                if dn and i < len(st.session_state.wps)-1: st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]; st.experimental_rerun()
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i); st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== GERAR LEGS A PARTIR DOS WPs (com TOC/TOD → novos WPs) =====================
cgl1, cgl2, _ = st.columns([2,2,6])
with cgl1: rpm_default = st.number_input("Cruise RPM default p/ legs", 1800, 2265, 2100, step=5)
with cgl2: gen = st.button("Gerar/Atualizar legs a partir dos WAYPOINTS (com TOC/TOD)", type="primary", use_container_width=True)

if gen and len(st.session_state.wps) >= 2:
    explode_wps_with_toc_tod_and_build_legs(
        rpm_default=rpm_default,
        wind_from=st.session_state.wind_from,
        wind_kt=st.session_state.wind_kt,
        ck_default=st.session_state.ck_default,
        desc_angle=st.session_state.desc_angle
    )
    st.success(f"Criadas {len(st.session_state.legs)} legs (com WPs TOC/TOD quando aplicável).")
    recompute_all_by_leg()

# ===================== INPUTS + FASES POR LEG =====================
if st.session_state.legs:
    recompute_all_by_leg()

    total_sec_all = 0; total_burn_all = 0.0; efob_final = None

    for idx_leg, leg in enumerate(st.session_state.legs):
        # === INPUTS LEG ===
        with st.expander(f"Leg {idx_leg+1} — Inputs", expanded=True):
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                TC   = st.number_input(f"True Course (°T) — L{idx_leg+1}", 0.0, 359.9, float(leg['TC']), step=0.1, key=f"TC_{idx_leg}")
                Dist = st.number_input(f"Distância (nm) — L{idx_leg+1}", 0.0, 500.0, float(leg['Dist']), step=0.1, key=f"Dist_{idx_leg}")
            with i2:
                Alt0 = st.number_input(f"Altitude INI (ft) — L{idx_leg+1}", 0.0, 30000.0, float(leg['Alt0']), step=50.0, key=f"Alt0_{idx_leg}")
                Alt1 = st.number_input(f"Altitude DEST (ft) — L{idx_leg+1}", 0.0, 30000.0, float(leg['Alt1']), step=50.0, key=f"Alt1_{idx_leg}")
            with i3:
                Wfrom = st.number_input(f"Vento FROM (°T) — L{idx_leg+1}", 0, 360, int(leg['Wfrom']), step=1, key=f"Wfrom_{idx_leg}")
                Wkt   = st.number_input(f"Vento (kt) — L{idx_leg+1}", 0, 150, int(leg['Wkt']), step=1, key=f"Wkt_{idx_leg}")
            with i4:
                CK     = st.number_input(f"Checkpoints (min) — L{idx_leg+1}", 1, 10, int(leg['CK']), step=1, key=f"CK_{idx_leg}")
                RPMcru = st.number_input(f"Cruise RPM (leg) — L{idx_leg+1}", 1800, 2265, int(leg['RPMcru']), step=5, key=f"RPMcru_{idx_leg}")
            j1, j2, j3 = st.columns([1.2,1.2,6])
            with j1: HoldMin = st.number_input(f"Espera (min) — L{idx_leg+1}", 0.0, 180.0, float(leg.get('HoldMin',0.0)), step=0.5, key=f"HoldMin_{idx_leg}")
            with j2: HoldFF  = st.number_input(f"FF espera (L/h) — L{idx_leg+1} (0=auto)", 0.0, 60.0, float(leg.get('HoldFF',0.0)), step=0.1, key=f"HoldFF_{idx_leg}")
            with j3:
                if st.button("Guardar leg", key=f"save_{idx_leg}", use_container_width=True):
                    update_leg(leg, TC=TC, Dist=Dist, Alt0=Alt0, Alt1=Alt1, Wfrom=Wfrom, Wkt=Wkt, CK=CK, RPMcru=RPMcru, HoldMin=HoldMin, HoldFF=HoldFF)
                    recompute_all_by_leg()
                if st.button("Apagar leg", key=f"del_{idx_leg}", use_container_width=True):
                    delete_leg(idx_leg); recompute_all_by_leg(); st.stop()

        # === FASES DA LEG ===
        comp_leg = st.session_state.computed_by_leg[idx_leg]
        phases = comp_leg["phases"]

        st.markdown(
            "<div class='kvrow'>"
            + f"<div class='kv'>Leg {idx_leg+1} — Fases: <b>{len(phases)}</b></div>"
            + f"<div class='kv'>ETE Leg: <b>{hhmmss(comp_leg['tot_sec'])}</b></div>"
            + f"<div class='kv'>Burn Leg: <b>{comp_leg['tot_burn']:.1f} L</b></div>"
            + "</div>", unsafe_allow_html=True
        )

        for pidx, c in enumerate(phases):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            left, right = st.columns([3,2])
            with left:
                st.subheader(f"Fase {idx_leg+1}.{pidx+1}: {c['label']}")
                st.caption(c["name"])
                st.markdown(
                    "<div class='kvrow'>"
                    + f"<div class='kv'>Alt: <b>{int(round(c['alt0']))}→{int(round(c['alt1']))} ft</b></div>"
                    + f"<div class='kv'>TH/MH: <b>{rang(c['TH'])}T / {rang(c['MH'])}M</b></div>"
                    + f"<div class='kv'>GS/TAS: <b>{rint(c['GS'])}/{rint(c['TAS'])} kt</b></div>"
                    + f"<div class='kv'>FF: <b>{rint(c['ff'])} L/h</b></div>"
                    + "</div>", unsafe_allow_html=True
                )
            with right:
                st.metric("Tempo", mmss(c["time"]))
                st.metric("Fuel desta fase (L)", f"{c['burn']:.1f}")

            r1, r2, r3 = st.columns(3)
            with r1: st.markdown(f"**Relógio** — {c['clock_start']} → {c['clock_end']}")
            with r2: st.markdown(f"**EFOB** — Start {c['efob_start']:.1f} L → End {c['efob_end']:.1f} L")
            with r3:
                if "Climb" in c["name"]:   st.markdown(f"**ROC ref.** — {rint(c['roc'])} ft/min")
                elif "Descent" in c["name"]: st.markdown(f"**ROD ref.** — {rint(c['rod'])} ft/min")
                else: st.markdown(f"**Cruise RPM (leg)** — {int(c['rpm_cruise_leg'])} RPM")

            if c["toc_tod"] is not None: st.info(f"Marcador: **{c['toc_tod']['type']}** em T+{mmss(c['toc_tod']['t'])}")
            if st.session_state.show_timeline and c["GS"] > 0:
                timeline({"GS":c["GS"],"TAS":c["TAS"],"ff":c["ff"],"time":c["time"]}, c["cps"], c["clock_start"], c["clock_end"], c["toc_tod"])

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
        total_sec_all  += comp_leg["tot_sec"]; total_burn_all += comp_leg["tot_burn"]
        if phases: efob_final = phases[-1]["efob_end"]

    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec_all)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{r10f(total_burn_all):.1f} L</b></div>"
        + (f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>" if efob_final is not None else "")
        + "</div>", unsafe_allow_html=True
    )

# ===================== MAPA (PYDECK) — rota, riscas 2min e DOG-HOUSES triangulares =====================
def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    """Triângulo isósceles orientado por 'heading_deg'."""
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

def triangle_hatching_paths(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65, step_nm=0.18):
    """Riscas internas diagonais como no exemplo (série de pequenos segmentos)."""
    # percorre ao longo da base→apex e desenha linhas inclinadas
    lines = []
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    # posição ao longo do eixo longitudinal
    s = -h_nm/2.0 + step_nm
    while s < h_nm/2.0:
        mid_lat, mid_lon = dest_point(lat, lon, heading_deg, s)
        # largura disponível na secção (linear – mais largo na base, mais estreito perto do apex)
        frac = (s + h_nm/2.0) / h_nm  # 0 na base, 1 no apex
        half_width = (1-frac) * (w_nm/2.0)
        # desenha uma “barra” curta com leve inclinação (~30°) vs. perpendicular
        left_lat, left_lon   = dest_point(mid_lat, mid_lon, heading_deg-90-30, half_width*0.85)
        right_lat, right_lon = dest_point(mid_lat, mid_lon, heading_deg+90-30, half_width*0.85)
        lines.append({"path": [[left_lon, left_lat], [right_lon, right_lat]]})
        s += step_nm
    return lines

if len(st.session_state.wps) >= 2 and st.session_state.legs and st.session_state.computed_by_leg:
    path_data, tick_data, tri_data, label_data, hatch_data, wp_labels = [], [], [], [], [], []

    # etiquetas para WPs (inclui TOC/TOD)
    for idx, W in enumerate(st.session_state.wps):
        wp_labels.append({"position":[W["lon"], W["lat"]], "text": W["name"]})

    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        leg = st.session_state.legs[i]
        phases = st.session_state.computed_by_leg[i]["phases"]

        # rota (linha)
        path_data.append({"path": [[A["lon"], A["lat"]], [B["lon"], B["lat"]]], "name": f"L{i+1}"})

        # -------- RISCA A CADA 2 MIN COM GS REAL ----------
        segments = [(p["time"], p["GS"]) for p in phases if p["GS"]>0]
        if not segments: segments = [(st.session_state.computed_by_leg[i]["tot_sec"], 0)]
        total_leg_dist = leg["Dist"]
        interval_s = 120
        k = 1
        while k*interval_s <= sum(s[0] for s in segments):
            target_t = k*interval_s
            dist_target = 0.0; t_acc = 0
            for dur, gs in segments:
                if t_acc + dur >= target_t:
                    dt_here = target_t - t_acc
                    dist_target += gs * (dt_here / 3600.0)
                    break
                else:
                    dist_target += gs * (dur / 3600.0)
                    t_acc += dur
            dist_target = min(dist_target, total_leg_dist)
            latm, lonm = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], dist_target)
            tc = leg["TC"]
            half_nm = 0.15
            left_lat, left_lon   = dest_point(latm, lonm, tc-90, half_nm)
            right_lat, right_lon = dest_point(latm, lonm, tc+90, half_nm)
            tick_data.append({"path": [[left_lon, left_lat], [right_lon, right_lat]]})
            k += 1

        # -------- DOG HOUSE TRIANGULAR + TEXTO + HATCH ----------
        mid_lat, mid_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], total_leg_dist/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, leg["TC"]+90, 0.35)
        tri = triangle_coords(off_lat, off_lon, leg["TC"], h_nm=0.9, w_nm=0.65)
        tri_data.append({"polygon": tri})
        hatch_data.extend(triangle_hatching_paths(off_lat, off_lon, leg["TC"]))

        ref = next((p for p in phases if p["GS"]>0), phases[0])
        label_text = f"{rang(ref['TH'])}T/{rang(ref['MH'])}M • {r10f(leg['Dist'])}nm • GS {rint(ref['GS'])} • ETE {mmss(st.session_state.computed_by_leg[i]['tot_sec'])}"
        label_pos = dest_point(off_lat, off_lon, leg["TC"]+90, 0.35)
        label_data.append({"position":[label_pos[1], label_pos[0]], "text": label_text})

    # LAYERS
    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[180, 0, 255, 220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0, 0, 0, 255], width_min_pixels=2)
    tri_layer   = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon",
                            get_fill_color=[255,255,255,230], get_line_color=[0,0,0,255],
                            line_width_min_pixels=2, stroked=True, filled=True)
    hatch_layer = pdk.Layer("PathLayer", data=hatch_data, get_path="path", get_color=[0,0,0,200], width_min_pixels=1)
    text_layer  = pdk.Layer("TextLayer", data=label_data, get_position="position", get_text="text",
                            get_size=14, get_color=[0,0,0], get_alignment_baseline="'center'")
    wp_text     = pdk.Layer("TextLayer", data=wp_labels, get_position="position", get_text="text",
                            get_size=12, get_color=[60,60,60], get_alignment_baseline="'top'")

    # vista (centro)
    mean_lat = sum([w["lat"] for w in st.session_state.wps])/len(st.session_state.wps)
    mean_lon = sum([w["lon"] for w in st.session_state.wps])/len(st.session_state.wps)
    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=7, pitch=0)

    # estilo com nomes de terras e POIs
    voyager = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"

    deck = pdk.Deck(
        map_style=voyager,
        initial_view_state=view_state,
        layers=[route_layer, ticks_layer, tri_layer, hatch_layer, text_layer, wp_text],
        tooltip={"text": "{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona pelo menos 2 waypoints e gera as legs (TOC/TOD incluídos) para ver o mapa e as dog-houses.")
