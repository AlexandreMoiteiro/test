# app.py — NAVLOG v10 (AFM) — Rota no Mapa + Dog Houses + TOC/TOD
# - Constrói rota por waypoints (cliques no mapa + pesquisa AD/Localidade)
# - Gera legs automaticamente (TC/Dist por geodésica) e mantém edição manual
# - Calcula fases por leg (Climb→TOC, Cruise após TOC, Descent→TOD, Cruise após TOD, Hold)
# - Desenha "dog houses" no mapa (TC/MH, Dist, ETE, GS, FF, EFOB)
# - Cruise RPM por leg; Climb/Descent RPM globais
# - Sem tabelas; cartões claros; timeline/CPs opcional

import streamlit as st
import datetime as dt
import math, re, json
import pandas as pd
import folium
from folium.plugins import MarkerCluster, Draw
from streamlit_folium import st_folium
from math import sin, asin, radians, degrees, atan2, cos

# ===================== CONFIG / STYLE =====================
st.set_page_config(page_title="NAVLOG v10 — Mapa + Dog Houses", layout="wide", initial_sidebar_state="collapsed")

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
.leaflet-container{background:#bfd7ff}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===================== UTILS (nav) =====================
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
    d = radians(angdiff(wdir, tc)); cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = degrees(asin(s)); th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

apply_var = lambda th, var, east_is_neg=False: wrap360(th - var if east_is_neg else th + var)

# Great-circle distance (NM) e rumo inicial verdadeiro (°T)
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

# ===================== AFM TABLES (Tecnam P2008 — resumo) =====================
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
VY = {0:67,2000:67,4000:67,6000:67,8000:67,10000:67,12000:67,14000:67}
ROC_FACTOR = 0.90
CRUISE = {
    0:{1800:(82,15.3),1900:(89,17.0),2000:(95,18.7),2100:(101,20.7),2250:(110,24.6),2388:(118,27.7)},
    2000:{1800:(81,15.5),1900:(87,17.0),2000:(93,18.8),2100:(99,20.9),2250:(108,25.0)},
    4000:{1800:(79,15.2),1900:(86,16.5),2000:(92,18.1),2100:(98,19.2),2250:(106,23.9)},
    6000:{1800:(78,14.9),1900:(85,16.1),2000:(91,17.5),2100:(97,19.2),2250:(105,22.7)},
    8000:{1800:(78,14.9),1900:(84,15.7),2000:(90,17.0),2100:(96,18.5),2250:(104,21.5)},
    10000:{1800:(78,15.5),1900:(82,15.5),2000:(89,16.6),2100:(95,17.9),2250:(103,20.5)},
}
isa_temp = lambda pa: 15.0 - 2.0*(pa/1000.0)
press_alt = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0

def interp1(x, x0, x1, y0, y1):
    if x1 == x0: return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def cruise_lookup(pa, rpm, oat, weight):
    rpm = min(int(rpm), 2265)
    pas = sorted(CRUISE.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    table0 = CRUISE[p0]; table1 = CRUISE[p1]

    def v(tab):
        rpms = sorted(tab.keys())
        if rpm in tab: return tab[rpm]
        if rpm < rpms[0]: lo, hi = rpms[0], rpms[1]
        elif rpm > rpms[-1]: lo, hi = rpms[-2], rpms[-1]
        else:
            lo = max([r for r in rpms if r <= rpm]); hi = min([r for r in rpms if r >= rpm])
        (tas_lo, ff_lo), (tas_hi, ff_hi) = tab[lo], tab[hi]
        t = (rpm - lo) / (hi - lo) if hi != lo else 0
        return (tas_lo + t*(tas_hi - tas_lo), ff_lo + t*(ff_hi - ff_lo))

    tas0, ff0 = v(table0); tas1, ff1 = v(table1)
    tas = interp1(pa_c, p0, p1, tas0, tas1); ff = interp1(pa_c, p0, p1, ff0, ff1)

    if oat is not None:
        dev = oat - isa_temp(pa_c)
        if dev > 0: tas *= 1 - 0.02*(dev/15.0); ff *= 1 - 0.025*(dev/15.0)
        elif dev < 0: tas *= 1 + 0.01*((-dev)/15.0); ff *= 1 + 0.03*((-dev)/15.0)

    tas *= (1.0 + 0.033*((650.0 - float(weight))/100.0))
    return max(0.0, tas), max(0.0, ff)

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

def vy_interp(pa):
    pas = sorted(VY.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    return interp1(pa_c, p0, p1, VY[p0], VY[p1])

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)

# globais
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0)
ens("rpm_climb", 2250); ens("rpm_desc", 1800)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)

# waypoints: [{name, lat, lon, alt}]
ens("wps", [])
# legs: como NAVLOG original (permitimos editar depois)
ens("legs", [])
# fases por leg (para render por leg)
ens("computed_by_leg", [])

# ===================== TIMELINE =====================
def timeline(seg, cps, start_label, end_label, toc_tod=None):
    total = max(1, int(seg['time']))
    html = "<div class='tl'><div class='bar'></div>"
    parts = []
    for cp in cps:
        pct = (cp['t']/total)*100.0
        parts.append(f"<div class='tick' style='left:{pct:.2f}%;'></div>")
        lbl = f"<div class='cp-lbl' style='left:{pct:.2f}%;'><div>T+{cp['min']}m</div><div>{cp['nm']} nm</div>" + \
              (f"<div>{cp['eto']}</div>" if cp['eto'] else "") + f"<div>EFOB {cp['efob']:.1f}</div></div>"
        parts.append(lbl)
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

# ===================== BUILD SEGMENTS (igual base anterior) =====================
def build_segments(tc, dist, alt0, alt1, wfrom, wkt, ck_min, params, rpm_cruise_leg, hold_min=0.0, hold_ff_input=0.0):
    qnh, oat, mag_var, mag_is_e = params['qnh'], params['oat'], params['mag_var'], params['mag_is_e']
    rpm_climb, rpm_desc, desc_angle, weight = params['rpm_climb'], params['rpm_desc'], params['desc_angle'], params['weight']

    pa0 = press_alt(alt0, qnh); pa1 = press_alt(alt1, qnh); pa_avg = (pa0 + pa1)/2.0
    Vy  = vy_interp(pa0)
    ROC = roc_interp(pa0, oat)
    TAS_climb = Vy
    FF_climb  = cruise_lookup((pa0 + pa1)/2.0, int(rpm_climb), oat, weight)[1]
    TAS_cru, FF_cru = cruise_lookup(pa1, int(rpm_cruise_leg), oat, weight)
    TAS_desc, FF_desc = cruise_lookup(pa_avg, int(rpm_desc), oat, weight)

    _, THc, GScl = wind_triangle(tc, TAS_climb, wfrom, wkt)
    _, THr, GScr = wind_triangle(tc, TAS_cru,  wfrom, wkt)
    _, THd, GSde = wind_triangle(tc, TAS_desc, wfrom, wkt)

    MHc = apply_var(THc, mag_var, mag_is_e)
    MHr = apply_var(THr, mag_var, mag_is_e)
    MHd = apply_var(THd, mag_var, mag_is_e)

    ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))  # ft/min

    profile = "LEVEL" if abs(alt1 - alt0) < 1e-6 else ("CLIMB" if alt1 > alt0 else "DESCENT")
    segs = []
    toc_tod_marker = None

    if profile == "CLIMB":
        t_need = (alt1 - alt0) / max(ROC, 1e-6)  # min
        d_need = GScl * (t_need / 60.0)
        if d_need <= dist:
            tA = rt10(t_need * 60)
            segs.append({"name":"Climb → TOC","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_climb,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise (após TOC)","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_cru,"time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            toc_tod_marker = {"type":"TOC","t": rt10(t_need*60)}
        else:
            tA = rt10((dist / max(GScl,1e-9)) * 3600)
            gained = ROC * (tA / 60.0)
            segs.append({"name":"Climb (não atinge)","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_climb,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0+gained})
    elif profile == "DESCENT":
        t_need = (alt0 - alt1) / max(ROD, 1e-6)
        d_need = GSde * (t_need / 60.0)
        if d_need <= dist:
            tA = rt10(t_need * 60)
            segs.append({"name":"Descent → TOD","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_desc,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise (após TOD)","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_cru,"time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            toc_tod_marker = {"type":"TOD","t": rt10(t_need*60)}
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
            _, hold_ff_auto = cruise_lookup(press_alt(alt1, qnh), int(rpm_cruise_leg), oat, weight)
            hold_ff = hold_ff_auto
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
        "rod": max(100.0, (wind_triangle(tc, cruise_lookup((pa0+pa1)/2.0, int(rpm_desc), oat, weight)[0], 0, 0)[2]) * 5.0 * (params['desc_angle']/3.0)),
        "toc_tod": toc_tod_marker,
        "ck_func": cps
    }

# ===================== RECOMPUTE POR LEG (com relógio/EFOB cumulativos) =====================
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
        except:
            base_time = None

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
                "TH": seg["TH"], "MH": seg["MH"],
                "GS": seg["GS"], "TAS": seg["TAS"], "ff": seg["ff"],
                "time": seg["time"], "dist": seg["dist"],
                "alt0": seg["alt0"], "alt1": seg["alt1"],
                "burn": r10f(seg["burn"]),
                "efob_start": efob_start, "efob_end": efob_end,
                "clock_start": c_start, "clock_end": c_end,
                "cps": cps,
                "toc_tod": (res["toc_tod"] if idx_seg==0 and ("Climb" in seg["name"] or "Descent" in seg["name"]) else None),
                "roc": res["roc"], "rod": res["rod"],
                "rpm_cruise_leg": leg["RPMcru"], "ck": leg["CK"]
            })

            t_cursor += seg["time"]
            carry_efob = efob_end

        # avança relógio pelo total desta leg
        if clock: clock = clock + dt.timedelta(seconds=sum(s['time'] for s in res["segments"]))

        st.session_state.computed_by_leg.append({
            "leg_ref": leg, "phases": phases,
            "tot_sec": sum(p["time"] for p in phases),
            "tot_burn": r10f(sum(p["burn"] for p in phases))
        })

# ===================== CRUD LEGS =====================
def add_leg_from_points(pA, pB, altA, altB, rpm_cru, wfrom, wkt, ck):
    tc = gc_course_tc(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    dist = gc_dist_nm(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    st.session_state.legs.append(dict(
        TC=float(tc), Dist=float(dist),
        Alt0=float(altA), Alt1=float(altB),
        Wfrom=int(wfrom), Wkt=int(wkt),
        CK=int(ck), HoldMin=0.0, HoldFF=0.0,
        RPMcru=int(rpm_cru)
    ))

def update_leg(leg_ref, **vals):
    leg_ref.update(vals)

def delete_leg(idx):
    st.session_state.legs.pop(idx)

# ===================== PARSE AD / LOCALIDADES (opcionais) =====================
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
            try:
                name = " ".join(tokens[1:tokens.index(coord_toks[0])]).strip()
            except ValueError:
                name = " ".join(tokens[1:]).strip()
            try:
                lon_idx = tokens.index(lon_tok); city = " ".join(tokens[lon_idx+1:]) or None
            except ValueError:
                city = None
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

# ===================== HEADER (sticky) =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,2,3,2])
with h1: st.title("NAVLOG — v10 (AFM)")
with h2:
    st.toggle("Mostrar TIMELINE/CPs", key="show_timeline", value=st.session_state.show_timeline)
with h3:
    if st.button("➕ Novo waypoint (clique no mapa)", use_container_width=True):
        st.toast("Usa a ferramenta de marcador no mapa e clica para adicionar.", icon="🗺️")
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
    # atualiza vento em todas as legs
    for L in st.session_state.legs:
        L["Wfrom"] = int(st.session_state.wind_from); L["Wkt"] = int(st.session_state.wind_kt)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== DATASETS (upload opcional) + pesquisa para adicionar ao plano =====================
ad_file = st.file_uploader("CSV AD/HEL/ULM (opcional)", type=["csv"], key="adcsv")
loc_file = st.file_uploader("CSV Localidades (opcional)", type=["csv"], key="loccsv")
ad_df = parse_ad_df(pd.read_csv(ad_file)) if ad_file else pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"])
loc_df = parse_loc_df(pd.read_csv(loc_file)) if loc_file else pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])

cflt1, cflt2, cbtn = st.columns([3,3,1.5])
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade", "", placeholder="Ex: LPPT, ABRANTES, LP0078…")
with cflt2: alt_wp = st.number_input("Altitude p/ waypoint adicionado (ft)", 0.0, 18000.0, 3000.0, step=100.0)
with cbtn:
    add_sel = st.button("Adicionar resultados ao plano")

def filter_df(df, q):
    if not q: return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

ad_f = filter_df(ad_df, qtxt)
loc_f = filter_df(loc_df, qtxt)

# ===================== MAPA (Folium) =====================
# centro: rota ou datasets
if st.session_state.wps:
    mean_lat = sum([w["lat"] for w in st.session_state.wps])/len(st.session_state.wps)
    mean_lon = sum([w["lon"] for w in st.session_state.wps])/len(st.session_state.wps)
elif len(ad_f)+len(loc_f)>0:
    mean_lat = pd.concat([ad_f["lat"], loc_f["lat"]]).mean()
    mean_lon = pd.concat([ad_f["lon"], loc_f["lon"]]).mean()
else:
    mean_lat, mean_lon = 39.5, -8.0

m = folium.Map(location=[mean_lat, mean_lon], zoom_start=7, tiles=None, control_scale=True)
folium.TileLayer(tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", name="OSM", control=False).add_to(m)

# clusters dos datasets
if not ad_f.empty:
    cl_ad = MarkerCluster(name="AD/HEL/ULM", show=True, disableClusteringAtZoom=10)
    for _, r in ad_f.iterrows():
        tip = f"<b>{r['code']}</b><br/>{r.get('name','')}"
        folium.Marker([r["lat"], r["lon"]], tooltip=tip, icon=folium.Icon(icon="plane", prefix="fa", color="gray")).add_to(cl_ad)
    cl_ad.add_to(m)
if not loc_f.empty:
    cl_loc = MarkerCluster(name="Localidades", show=True, disableClusteringAtZoom=10)
    for _, r in loc_f.iterrows():
        lbl = f"""<div style="font-size:11px;font-weight:600;color:#fff;background:rgba(0,0,0,.6);
        padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,.35);backdrop-filter:blur(1px);">{r['code']}</div>"""
        folium.Marker([r["lat"], r["lon"]], icon=folium.DivIcon(html=lbl), tooltip=r.get("name","")).add_to(cl_loc)
    cl_loc.add_to(m)

# rota atual: polyline + marcadores
if st.session_state.wps:
    coords = [(w["lat"], w["lon"]) for w in st.session_state.wps]
    folium.PolyLine(coords, color="#8a2be2", weight=4, opacity=0.9).add_to(m)
    for i, w in enumerate(st.session_state.wps):
        folium.Marker([w["lat"], w["lon"]],
                      tooltip=f"WP {i+1}: {w['name']} ({w['alt']:.0f} ft)",
                      icon=folium.Icon(color="blue", icon="flag")).add_to(m)

# dog houses (após cálculo das legs)
def add_doghouse(latA, lonA, latB, lonB, payload):
    # posição a meio da perna
    φ1, λ1, φ2, λ2 = map(math.radians, [latA, lonA, latB, lonB])
    bx = math.cos(φ2)*math.cos(λ2-λ1)
    by = math.cos(φ2)*math.sin(λ2-λ1)
    φ3 = math.atan2(math.sin(φ1)+math.sin(φ2), math.sqrt( (math.cos(φ1)+bx)**2 + by**2 ))
    λ3 = math.radians(lonA) + math.atan2(by, math.cos(φ1)+bx)
    lat_mid, lon_mid = math.degrees(φ3), ((math.degrees(λ3)+540)%360)-180

    html = f"""
    <div style="font-family:ui-sans-serif;font-size:11px;line-height:1.05;border:1px solid #111;
    background:#fff;padding:4px 6px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.35)">
      <div><b>{payload['leg_name']}</b></div>
      <div>{payload['tc']}T / {payload['mh']}M</div>
      <div>{payload['dist']} nm · GS {payload['gs']} kt</div>
      <div>ETE {payload['ete']} · FF {payload['ff']} L/h</div>
      <div>EFOB {payload['efob_s']}→{payload['efob_e']} L</div>
    </div>"""
    folium.Marker([lat_mid, lon_mid], icon=folium.DivIcon(html=html)).add_to(m)

# ferramenta de desenho p/ cliques (marcadores)
Draw(draw_options={"polyline":False,"polygon":False,"circle":False,"rectangle":False,"circlemarker":False,"marker":True},
     edit_options={"edit":False,"remove":False}).add_to(m)

# render
map_state = st_folium(m, width=None, height=650)

# processa clique (último marker desenhado)
if map_state and map_state.get("last_active_drawing"):
    last = map_state["last_active_drawing"]
    if last and last.get("geometry", {}).get("type") == "Point":
        lat, lon = last["geometry"]["coordinates"][1], last["geometry"]["coordinates"][0]
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": float(lat), "lon": float(lon), "alt": float(alt_wp)})
        st.toast(f"Waypoint adicionado: {lat:.5f}, {lon:.5f} @ {alt_wp:.0f} ft", icon="➕")

# adiciona resultados de pesquisa
if add_sel:
    for _, r in pd.concat([ad_f, loc_f]).iterrows():
        st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
    st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== EDITOR DE WAYPOINTS (sem tabelas) =====================
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2:
                lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
                lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c3:
                alt  = st.number_input(f"Altitude (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=100.0, key=f"wpalt_{i}")
            with c4:
                if st.button("Remover", key=f"delwp_{i}"):
                    st.session_state.wps.pop(i); st.experimental_rerun()
            # guardar
            if (name!=w["name"]) or (lat!=w["lat"]) or (lon!=w["lon"]) or (alt!=w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}

# ===================== GERAR LEGS AUTOMÁTICAS A PARTIR DA ROTA =====================
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
cgl1, cgl2, cgl3 = st.columns([2,2,6])
with cgl1:
    rpm_default = st.number_input("Cruise RPM default p/ legs", 1800, 2265, 2100, step=5)
with cgl2:
    gen = st.button("Gerar/Atualizar legs a partir dos WAYPOINTS", type="primary", use_container_width=True)

if gen and len(st.session_state.wps) >= 2:
    st.session_state.legs = []
    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        add_leg_from_points(A, B, A["alt"], B["alt"], rpm_default, st.session_state.wind_from, st.session_state.wind_kt, st.session_state.ck_default)
    st.success(f"Criadas {len(st.session_state.legs)} legs.")
    recompute_all_by_leg()

# ===================== INPUTS E FASES POR LEG (logo abaixo de cada leg) =====================
if st.session_state.legs:
    recompute_all_by_leg()

    total_sec_all = 0; total_burn_all = 0.0; efob_final = None

    for idx_leg, leg in enumerate(st.session_state.legs):
        # === CARTÃO INPUT LEG ===
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
            with j1:
                HoldMin = st.number_input(f"Espera (min) — L{idx_leg+1}", 0.0, 180.0, float(leg.get('HoldMin',0.0)), step=0.5, key=f"HoldMin_{idx_leg}")
            with j2:
                HoldFF  = st.number_input(f"FF espera (L/h) — L{idx_leg+1} (0=auto)", 0.0, 60.0, float(leg.get('HoldFF',0.0)), step=0.1, key=f"HoldFF_{idx_leg}")
            with j3:
                if st.button("Guardar leg", key=f"save_{idx_leg}", use_container_width=True):
                    update_leg(leg, TC=TC, Dist=Dist, Alt0=Alt0, Alt1=Alt1, Wfrom=Wfrom, Wkt=Wkt, CK=CK, RPMcru=RPMcru, HoldMin=HoldMin, HoldFF=HoldFF)
                    recompute_all_by_leg()
                if st.button("Apagar leg", key=f"del_{idx_leg}", use_container_width=True):
                    delete_leg(idx_leg); recompute_all_by_leg(); st.stop()

        # === CARTÕES FASES DA LEG ===
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

            if c["toc_tod"] is not None:
                st.info(f"Marcador: **{c['toc_tod']['type']}** em T+{mmss(c['toc_tod']['t'])}")

            if st.session_state.show_timeline and c["GS"] > 0:
                timeline(
                    {"GS":c["GS"],"TAS":c["TAS"],"ff":c["ff"],"time":c["time"]},
                    c["cps"], c["clock_start"], c["clock_end"], toc_tod=c["toc_tod"]
                )

            # avisos
            warns = []
            if c["dist"] == 0 and abs(c["alt1"]-c["alt0"])>50: warns.append("Distância 0 com variação de altitude.")
            if "não atinge" in c["name"]: warns.append("Perfil não atinge a altitude-alvo nesta fase.")
            if c["efob_end"] <= 0: warns.append("EFOB no fim desta fase é 0 (ou negativo).")
            if warns: st.warning(" | ".join(warns))

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

        # Globais acumulados
        total_sec_all  += comp_leg["tot_sec"]
        total_burn_all += comp_leg["tot_burn"]
        if phases: efob_final = phases[-1]["efob_end"]

        # DOG HOUSE no mapa para a leg (usar valores consolidados da leg — total)
        if len(st.session_state.wps) >= (idx_leg+2):
            A = st.session_state.wps[idx_leg]; B = st.session_state.wps[idx_leg+1]
            # valores “macro” da leg (usar primeira fase com GS>0 para TC/MH/GS/FF)
            ref = next((p for p in phases if p["GS"]>0), phases[0])
            payload = dict(
                leg_name=f"L{idx_leg+1}",
                tc=rang(ref["TH"]), mh=rang(ref["MH"]),
                dist=r10f(leg["Dist"]), gs=rint(ref["GS"]),
                ete=mmss(comp_leg["tot_sec"]),
                ff=rint(ref["ff"]),
                efob_s=phases[0]["efob_start"], efob_e=phases[-1]["efob_end"]
            )
            add_doghouse(A["lat"], A["lon"], B["lat"], B["lon"], payload)

    # CHIPS GLOBAIS
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec_all)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{r10f(total_burn_all):.1f} L</b></div>"
        + (f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>" if efob_final is not None else "")
        + "</div>", unsafe_allow_html=True
    )
