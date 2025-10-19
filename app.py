import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees
import copy

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG v11 — PyDeck + Triângulos", layout="wide", initial_sidebar_state="collapsed")

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

# ===================== AFM SIMPLIFICADO =====================
TAS_CLIMB = 70
TAS_CRUISE = 90
TAS_DESC = 90
FF = 20.0
ROC = 500.0
ROD_BASE = 500.0
press_alt  = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("computed_by_leg", [])
ens("effective_wps", [])

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
def build_segments(tc, dist, alt0, alt1, wfrom, wkt, ck_min, params, hold_min=0.0, hold_ff_input=0.0):
    qnh, oat, mag_var, mag_is_e = params['qnh'], params['oat'], params['mag_var'], params['mag_is_e']
    desc_angle, weight = params['desc_angle'], params['weight']

    pa0 = press_alt(alt0, qnh); pa1 = press_alt(alt1, qnh); pa_avg = (pa0 + pa1)/2.0

    FF_climb = FF
    FF_cru = FF
    FF_desc = FF

    _, THc, GScl = wind_triangle(tc, TAS_CLIMB, wfrom, wkt)
    _, THr, GScr = wind_triangle(tc, TAS_CRUISE,  wfrom, wkt)
    _, THd, GSde = wind_triangle(tc, TAS_DESC, wfrom, wkt)

    MHc = apply_var(THc, mag_var, mag_is_e)
    MHr = apply_var(THr, mag_var, mag_is_e)
    MHd = apply_var(THd, mag_var, mag_is_e)

    ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))  # ft/min

    profile = "LEVEL" if abs(alt1 - alt0) < 1e-6 else ("CLIMB" if alt1 > alt0 else "DESCENT")
    segs, toc_tod_marker = [], None

    if profile == "CLIMB":
        t_need = (alt1 - alt0) / max(ROC, 1e-6)
        d_need = GScl * (t_need / 60.0)
        if d_need <= dist:
            tA = rt10(t_need * 60)
            segs.append({"name":"Climb to TOC","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_CLIMB,"ff":FF_climb,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise after TOC","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_CRUISE,"ff":FF_cru,"time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            toc_tod_marker = {"type":"TOC","t": rt10(t_need*60), "dist": d_need}
        else:
            tA = rt10((dist / max(GScl,1e-9)) * 3600)
            gained = ROC * (tA / 60.0)
            segs.append({"name":"Climb (does not reach)","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_CLIMB,"ff":FF_climb,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0+gained})
    elif profile == "DESCENT":
        t_need = (alt0 - alt1) / max(ROD, 1e-6)
        d_need = GSde * (t_need / 60.0)
        if d_need <= dist:
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise to TOD","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_CRUISE,"ff":FF_cru,"time":tB,"dist":rem,"alt0":alt0,"alt1":alt0})
            tA = rt10(t_need * 60)
            segs.append({"name":"Descent after TOD","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_DESC,"ff":FF_desc,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            toc_tod_marker = {"type":"TOD","t": rt10(rem / max(GScr,1e-9) * 3600) if rem > 0 else 0, "dist": rem}
        else:
            tA = rt10((dist / max(GSde,1e-9)) * 3600)
            lost = ROD * (tA / 60.0)
            segs.append({"name":"Descent (does not reach)","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_DESC,"ff":FF_desc,"time":tA,"dist":dist,"alt0":alt0,"alt1":max(0.0, alt0 - lost)})
    else:
        tA = rt10((dist / max(GScr,1e-9)) * 3600)
        segs.append({"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_CRUISE,"ff":FF_cru,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0})

    # HOLD opcional
    hold_min = max(0.0, float(hold_min))
    if hold_min > 0.0:
        hold_ff = FF
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
        "roc": ROC,
        "rod": ROD,
        "toc_tod": toc_tod_marker,
        "ck_func": cps
    }

# ===================== INSERT TOC/TOD WPS =====================
def insert_toc_tod():
    new_wps = copy.deepcopy(st.session_state.wps)
    new_legs = []
    wp_idx = 0
    for leg in st.session_state.legs:
        A = new_wps[wp_idx]
        B = new_wps[wp_idx + len(new_legs) + 1]  # adjust for inserted
        params = dict(
            qnh=st.session_state.qnh, oat=st.session_state.oat,
            mag_var=st.session_state.mag_var, mag_is_e=st.session_state.mag_is_e,
            desc_angle=st.session_state.desc_angle, weight=st.session_state.weight
        )
        res = build_segments(
            tc=leg['TC'], dist=leg['Dist'], alt0=leg['Alt0'], alt1=leg['Alt1'],
            wfrom=leg['Wfrom'], wkt=leg['Wkt'], ck_min=leg['CK'],
            params=params,
            hold_min=leg.get('HoldMin',0.0), hold_ff_input=leg.get('HoldFF',0.0)
        )
        if res["toc_tod"]:
            d_marker = res["toc_tod"]["dist"]
            lat_m, lon_m = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_marker)
            m_type = res["toc_tod"]["type"]
            m_alt = leg["Alt1"] if m_type == "TOC" else leg["Alt0"]
            m_wp = {"name": m_type, "lat": lat_m, "lon": lon_m, "alt": m_alt}
            new_wps.insert(wp_idx + 1, m_wp)
            # split leg into two
            leg1 = leg.copy()
            leg1["Dist"] = d_marker
            leg1["Alt1"] = m_alt
            new_legs.append(leg1)
            leg2 = leg.copy()
            leg2["TC"] = gc_course_tc(m_wp["lat"], m_wp["lon"], B["lat"], B["lon"])
            leg2["Dist"] = leg["Dist"] - d_marker
            leg2["Alt0"] = m_alt
            leg2["Alt1"] = leg["Alt1"]
            leg2["HoldMin"] = 0.0  # hold only at end
            new_legs.append(leg2)
        else:
            new_legs.append(leg)
        wp_idx += 1 if not res["toc_tod"] else 2
    st.session_state.effective_wps = new_wps
    st.session_state.legs = new_legs  # update legs to split ones

# ===================== RECOMPUTE POR LEG =====================
def recompute_all_by_leg():
    insert_toc_tod()  # split first
    st.session_state.computed_by_leg = []
    params = dict(
        qnh=st.session_state.qnh, oat=st.session_state.oat,
        mag_var=st.session_state.mag_var, mag_is_e=st.session_state.mag_is_e,
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
            params=params,
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
                "roc": res["roc"], "rod": res["rod"], "ck": leg["CK"]
            })
            t_cursor += seg['time']; carry_efob = efob_end

        if clock: clock = clock + dt.timedelta(seconds=sum(s['time'] for s in res["segments"]))
        st.session_state.computed_by_leg.append({
            "leg_ref": leg, "phases": phases,
            "tot_sec": sum(p["time"] for p in phases),
            "tot_burn": r10f(sum(p["burn"] for p in phases))
        })

# ===================== CRUD / BUILD LEGS =====================
def add_leg_from_points(pA, pB, wfrom, wkt, ck):
    tc = gc_course_tc(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    dist = gc_dist_nm(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    st.session_state.legs.append(dict(
        TC=float(tc), Dist=float(dist), Alt0=float(pA["alt"]), Alt1=float(pB["alt"]),
        Wfrom=int(wfrom), Wkt=int(wkt), CK=int(ck), HoldMin=0.0, HoldFF=0.0
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

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([3,2,3,2])
with h1: st.title("NAVLOG — v11 (AFM)")
with h2: st.toggle("Mostrar TIMELINE/CPs", key="show_timeline", value=st.session_state.show_timeline)
with h3:
    if st.button("➕ Novo waypoint manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with h4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.computed_by_leg = []; st.session_state.effective_wps = []
st.markdown("</div>", unsafe_allow_html=True)

# ===================== PARÂMETROS GLOBAIS =====================
with st.form("globals"):
    p1, p2, p3 = st.columns(3)
    with p1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, int(st.session_state.qnh))
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, int(st.session_state.oat))
    with p2:
        st.session_state.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
    with p3:
        st.session_state.weight = st.number_input("Peso (kg)", 450.0, 700.0, float(st.session_state.weight), step=1.0)
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
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
    ad_raw  = pd.read_csv(AD_CSV)
    loc_raw = pd.read_csv(LOC_CSV)
    ad_df  = parse_ad_df(ad_raw)
    loc_df = parse_loc_df(loc_raw)
except Exception as e:
    ad_df  = pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

cflt1, cflt2 = st.columns(2)
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, LP0078…")
with cflt2: alt_wp = st.number_input("Altitude para WPs adicionados (ft)", 0.0, 18000.0, 3000.0, step=100.0)

def filter_df(df, q):
    if not q: return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

ad_f  = filter_df(ad_df, qtxt)
loc_f = filter_df(loc_df, qtxt)
filtered = pd.concat([ad_f, loc_f])
options = [f"{row['code']} - {row['name']} ({row['type']})" for _, row in filtered.iterrows()]
selected = st.multiselect("Selecionar waypoints para adicionar", options)
if st.button("Adicionar selecionados"):
    for sel in selected:
        code = sel.split(" - ")[0]
        row = filtered[filtered['code'] == code].iloc[0]
        st.session_state.wps.append({"name": str(row["code"]), "lat": float(row["lat"]), "lon": float(row["lon"]), "alt": float(alt_wp)})
    st.success(f"Adicionados {len(selected)} WPs.")

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
                if up and i>0:
                    st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
                    st.rerun()
                if dn and i < len(st.session_state.wps)-1:
                    st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
                    st.rerun()
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i); st.rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== GERAR LEGS A PARTIR DOS WPs =====================
cgl1, cgl2, _ = st.columns([2,2,6])
with cgl1: pass  # no rpm
with cgl2: gen = st.button("Gerar/Atualizar legs a partir dos WAYPOINTS", type="primary", use_container_width=True)

if gen and len(st.session_state.wps) >= 2:
    st.session_state.legs = []
    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        add_leg_from_points(A, B, st.session_state.wind_from, st.session_state.wind_kt, st.session_state.ck_default)
    st.success(f"Criadas {len(st.session_state.legs)} legs.")
    recompute_all_by_leg()

# ===================== INPUTS + FASES POR LEG =====================
if st.session_state.legs:
    recompute_all_by_leg()

    total_sec_all = 0; total_burn_all = 0.0; efob_final = None

    for idx_leg, leg in enumerate(st.session_state.legs):
        # === INPUTS LEG ===
        with st.expander(f"Leg {idx_leg+1} — Inputs", expanded=True):
            i1, i2, i3 = st.columns(3)
            with i1:
                TC   = st.number_input(f"True Course (°T) — L{idx_leg+1}", 0.0, 359.9, float(leg['TC']), step=0.1, key=f"TC_{idx_leg}")
                Dist = st.number_input(f"Distância (nm) — L{idx_leg+1}", 0.0, 500.0, float(leg['Dist']), step=0.1, key=f"Dist_{idx_leg}")
            with i2:
                Alt0 = st.number_input(f"Altitude INI (ft) — L{idx_leg+1}", 0.0, 30000.0, float(leg['Alt0']), step=50.0, key=f"Alt0_{idx_leg}")
                Alt1 = st.number_input(f"Altitude DEST (ft) — L{idx_leg+1}", 0.0, 30000.0, float(leg['Alt1']), step=50.0, key=f"Alt1_{idx_leg}")
            with i3:
                Wfrom = st.number_input(f"Vento FROM (°T) — L{idx_leg+1}", 0, 360, int(leg['Wfrom']), step=1, key=f"Wfrom_{idx_leg}")
                Wkt   = st.number_input(f"Vento (kt) — L{idx_leg+1}", 0, 150, int(leg['Wkt']), step=1, key=f"Wkt_{idx_leg}")
            j1, j2, j3, j4 = st.columns([1.2,1.2,1.2,4])
            with j1: CK     = st.number_input(f"Checkpoints (min) — L{idx_leg+1}", 1, 10, int(leg['CK']), step=1, key=f"CK_{idx_leg}")
            with j2: HoldMin = st.number_input(f"Espera (min) — L{idx_leg+1}", 0.0, 180.0, float(leg.get('HoldMin',0.0)), step=0.5, key=f"HoldMin_{idx_leg}")
            with j3: HoldFF  = st.number_input(f"FF espera (L/h) — L{idx_leg+1} (0=auto)", 0.0, 60.0, float(leg.get('HoldFF',0.0)), step=0.1, key=f"HoldFF_{idx_leg}")
            with j4:
                if st.button("Guardar leg", key=f"save_{idx_leg}", use_container_width=True):
                    update_leg(leg, TC=TC, Dist=Dist, Alt0=Alt0, Alt1=Alt1, Wfrom=Wfrom, Wkt=Wkt, CK=CK, HoldMin=HoldMin, HoldFF=HoldFF)
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
                else: pass

            if c["toc_tod"] is not None: st.info(f"Marcador: **{c['toc_tod']['type']}** em T+{mmss(c['toc_tod']['t'])}")
            if st.session_state.show_timeline and c["GS"] > 0:
                timeline({"GS":c["GS"],"TAS":c["TAS"],"ff":c["ff"],"time":c["time"]}, c["cps"], c["clock_start"], c["clock_end"], c["toc_tod"])

            warns = []
            if c["dist"] == 0 and abs(c["alt1"]-c["alt0"])>50: warns.append("Distância 0 com variação de altitude.")
            if "não atinge" in c["name"]: warns.append("Perfil não atinge a altitude-alvo nesta fase.")
            if c["efob_end"] <= 0: warns.append("EFOB no fim desta fase é 0 (ou negativo).")
            if warns: st.warning(" | ".join(warns))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
        total_sec_all  += comp_leg["tot_sec"]
        total_burn_all += comp_leg["tot_burn"]
        if phases: efob_final = phases[-1]["efob_end"]

    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec_all)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{r10f(total_burn_all):.1f} L</b></div>"
        + (f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>" if efob_final is not None else "")
        + "</div>", unsafe_allow_html=True
    )

# ===================== MAPA (PYDECK) — rota, riscas 2min, triângulos =====================
def triangle_coords(lat, lon, heading_deg, h_nm=0.8, w_nm=0.55):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

if len(st.session_state.effective_wps) >= 2 and st.session_state.legs and st.session_state.computed_by_leg:
    path_data, tick_data, tri_data, label_data, mh_label_data = [], [], [], [], []

    wps = st.session_state.effective_wps

    for i in range(len(wps)-1):
        A = wps[i]; B = wps[i+1]
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

        # -------- DOG HOUSE TRIANGULAR + TEXTO ----------
        mid_lat, mid_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], total_leg_dist/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, leg["TC"]+90, 0.3)
        tri = triangle_coords(off_lat, off_lon, leg["TC"], h_nm=0.9, w_nm=0.65)
        tri_data.append({"polygon": tri})

        ref = next((p for p in phases if p["GS"]>0), phases[0])
        label_text = f"{rang(ref['TH'])}T • {r10f(leg['Dist'])}nm • GS {rint(ref['GS'])} • ETE {mmss(st.session_state.computed_by_leg[i]['tot_sec'])}"
        label_pos = dest_point(off_lat, off_lon, leg["TC"]+90, 0.35)
        label_data.append({"position":[label_pos[1], label_pos[0]], "text": label_text})

        # MH destacado
        mh_pos = [off_lon, off_lat]
        mh_label_data.append({"position": mh_pos, "text": f"{rang(ref['MH'])}M"})

    # WAYPOINTS
    wp_data = [{"lon": w["lon"], "lat": w["lat"], "name": w["name"]} for w in wps]

    # LAYERS
    base_map = pdk.Layer(
        "TileLayer",
        data="https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        tile_size=256,
        min_zoom=0,
        max_zoom=19,
    )

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
        filled=True
    )
    text_layer  = pdk.Layer("TextLayer", data=label_data, get_position="position", get_text="text",
                            get_size=14, get_color=[0,0,0], get_alignment_baseline="'center'")
    mh_text_layer = pdk.Layer("TextLayer", data=mh_label_data, get_position="position", get_text="text",
                              get_size=20, get_color=[255,0,0], get_alignment_baseline="'center'")
    wp_layer = pdk.Layer(
        "ScatterplotLayer",
        data=wp_data,
        get_position=["lon", "lat"],
        get_radius=200,
        get_fill_color=[0, 255, 0, 200],
        pickable=True
    )

    mean_lat = sum([w["lat"] for w in wps])/len(wps)
    mean_lon = sum([w["lon"] for w in wps])/len(wps)

    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=7, pitch=0)
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[base_map, route_layer, ticks_layer, wp_layer, tri_layer, text_layer, mh_text_layer],
        tooltip={"text": "{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona pelo menos 2 waypoints e gera as legs para ver a rota, riscas e triângulos.")
