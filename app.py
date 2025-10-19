import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees
import copy

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG v12 — Enhanced VFR", layout="wide", initial_sidebar_state="collapsed")

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

# ===================== CONSTANTS =====================
TAS_CLIMB = 70
TAS_CRUISE = 90
TAS_DESC = 90
FF = 20.0
ROC = 500.0
ROD_BASE = 500.0
press_alt = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("show_timeline", False)
ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("computed_by_leg", [])
ens("effective_wps", [])

# ===================== TIMELINE =====================
def timeline(seg, cps, start_label, end_label, toc_tod=None):
    total = max(1, int(seg['time']))
    html = "<div class='tl'><div class='bar'></div>"
    parts = []
    for cp in cps:
        pct = (cp['t']/total)*100.0
        parts += [f"<div class='tick' style='left:{pct:.2f}%;'></div>",
                  f"<div class='cp-lbl' style='left:{pct:.2f}%;'><div>T+{cp['min']}m</div><div>{cp['nm']} nm</div>" +
                  (f"<div>{cp['eto']}</div>" if cp['eto'] else "") + f"<div>EFOB {cp['efob']:.1f}</div></div>"]
    if toc_tod and 0 < toc_tod['t'] < total:
        pct = (toc_tod['t']/total)*100.0
        cls = 'tocdot' if toc_tod['type'] == 'TOC' else 'toddot'
        parts.append(f"<div class='{cls}' title='{toc_tod['type']}' style='left:{pct:.2f}%;'></div>")
    html += ''.join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"GS {rint(seg['GS'])} kt · TAS {rint(seg['TAS'])} kt · FF {rint(seg['ff'])} L/h  |  {start_label} → {end_label}")

def phase_label(name):
    n = name.lower()
    if "climb" in n: return "Climb"
    if "descent" in n: return "Descent"
    if "hold" in n: return "Hold"
    return "Cruise"

# ===================== CALCULATION =====================
def build_segments(tc, dist, alt0, alt1, wfrom, wkt, ck_min, params, hold_min=0.0):
    qnh, oat, mag_var, mag_is_e, desc_angle = params['qnh'], params['oat'], params['mag_var'], params['mag_is_e'], params['desc_angle']
    pa0, pa1, pa_avg = press_alt(alt0, qnh), press_alt(alt1, qnh), (press_alt(alt0, qnh) + press_alt(alt1, qnh)) / 2.0

    _, THc, GScl = wind_triangle(tc, TAS_CLIMB, wfrom, wkt)
    _, THr, GScr = wind_triangle(tc, TAS_CRUISE, wfrom, wkt)
    _, THd, GSde = wind_triangle(tc, TAS_DESC, wfrom, wkt)

    MHc, MHr, MHd = apply_var(THc, mag_var, mag_is_e), apply_var(THr, mag_var, mag_is_e), apply_var(THd, mag_var, mag_is_e)
    ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))

    profile = "LEVEL" if abs(alt1 - alt0) < 1e-6 else ("CLIMB" if alt1 > alt0 else "DESCENT")
    segs, toc_tod_marker = [], None

    if profile == "CLIMB":
        t_need = (alt1 - alt0) / max(ROC, 1e-6)
        d_need = GScl * (t_need / 60.0)
        if d_need <= dist:
            tA = rt10(t_need * 60)
            segs.append({"name":"Climb to TOC", "TH":THc, "MH":MHc, "GS":GScl, "TAS":TAS_CLIMB, "ff":FF, "time":tA, "dist":d_need, "alt0":alt0, "alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr, 1e-9)) * 3600)
                segs.append({"name":"Cruise after TOC", "TH":THr, "MH":MHr, "GS":GScr, "TAS":TAS_CRUISE, "ff":FF, "time":tB, "dist":rem, "alt0":alt1, "alt1":alt1})
            toc_tod_marker = {"type":"TOC", "t":rt10(t_need*60), "dist":d_need}
        else:
            tA = rt10((dist / max(GScl, 1e-9)) * 3600)
            gained = ROC * (tA / 60.0)
            segs.append({"name":"Climb (does not reach)", "TH":THc, "MH":MHc, "GS":GScl, "TAS":TAS_CLIMB, "ff":FF, "time":tA, "dist":dist, "alt0":alt0, "alt1":alt0+gained})
    elif profile == "DESCENT":
        t_need = (alt0 - alt1) / max(ROD, 1e-6)
        d_need = GSde * (t_need / 60.0)
        if d_need <= dist:
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr, 1e-9)) * 3600)
                segs.append({"name":"Cruise to TOD", "TH":THr, "MH":MHr, "GS":GScr, "TAS":TAS_CRUISE, "ff":FF, "time":tB, "dist":rem, "alt0":alt0, "alt1":alt0})
            tA = rt10(t_need * 60)
            segs.append({"name":"Descent after TOD", "TH":THd, "MH":MHd, "GS":GSde, "TAS":TAS_DESC, "ff":FF, "time":tA, "dist":d_need, "alt0":alt0, "alt1":alt1})
            toc_tod_marker = {"type":"TOD", "t":rt10(rem / max(GScr, 1e-9) * 3600) if rem > 0 else 0, "dist":rem}
        else:
            tA = rt10((dist / max(GSde, 1e-9)) * 3600)
            lost = ROD * (tA / 60.0)
            segs.append({"name":"Descent (does not reach)", "TH":THd, "MH":MHd, "GS":GSde, "TAS":TAS_DESC, "ff":FF, "time":tA, "dist":dist, "alt0":alt0, "alt1":max(0.0, alt0 - lost)})
    else:
        tA = rt10((dist / max(GScr, 1e-9)) * 3600)
        segs.append({"name":"Cruise", "TH":THr, "MH":MHr, "GS":GScr, "TAS":TAS_CRUISE, "ff":FF, "time":tA, "dist":dist, "alt0":alt0, "alt1":alt0})

    if hold_min > 0.0:
        hold_sec = rt10(hold_min * 60.0)
        end_alt = segs[-1]["alt1"] if segs else alt1
        segs.append({"name":"Hold", "TH":segs[-1]["TH"] if segs else tc, "MH":segs[-1]["MH"] if segs else tc,
                     "GS":0.0, "TAS":0.0, "ff":FF, "time":hold_sec, "dist":0.0, "alt0":end_alt, "alt1":end_alt})

    for s in segs: s["burn"] = s["ff"] * (s["time"] / 3600.0)
    tot_sec = sum(s['time'] for s in segs)
    tot_burn = r10f(sum(s['burn'] for s in segs))

    def cps(seg, every_min, base_clk, efob_start):
        out = []; t = 0
        if every_min <= 0 or seg['GS'] <= 0: return out
        while t + every_min*60 <= seg['time']:
            t += every_min*60
            d = seg['GS'] * (t / 3600.0)
            burn = seg['ff'] * (t / 3600.0)
            eto = (base_clk + dt.timedelta(seconds=t)).strftime('%H:%M') if base_clk else ""
            efob = max(0.0, r10f(efob_start - burn))
            out.append({"t":t, "min":int(t/60), "nm":round(d,1), "eto":eto, "efob":efob})
        return out

    return {"segments": segs, "tot_sec": tot_sec, "tot_burn": tot_burn, "roc": ROC, "rod": ROD, "toc_tod": toc_tod_marker, "ck_func": cps}

# ===================== INSERT TOC/TOD =====================
def insert_toc_tod():
    new_wps = copy.deepcopy(st.session_state.wps)
    new_legs = []
    wp_idx = 0
    for leg in st.session_state.legs:
        A = new_wps[wp_idx]
        B = new_wps[wp_idx + len(new_legs) + 1]
        params = {"qnh": st.session_state.qnh, "oat": st.session_state.oat, "mag_var": st.session_state.mag_var,
                  "mag_is_e": st.session_state.mag_is_e, "desc_angle": st.session_state.desc_angle}
        res = build_segments(leg['TC'], leg['Dist'], leg['Alt0'], leg['Alt1'], leg['Wfrom'], leg['Wkt'], leg['CK'], params, leg.get('HoldMin', 0.0))
        if res["toc_tod"]:
            d_marker = res["toc_tod"]["dist"]
            lat_m, lon_m = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_marker)
            m_type = res["toc_tod"]["type"]
            m_alt = leg["Alt1"] if m_type == "TOC" else leg["Alt0"]
            m_wp = {"name": m_type, "lat": lat_m, "lon": lon_m, "alt": m_alt}
            new_wps.insert(wp_idx + 1, m_wp)
            leg1 = leg.copy(); leg1["Dist"] = d_marker; leg1["Alt1"] = m_alt
            new_legs.append(leg1)
            leg2 = leg.copy(); leg2["TC"] = gc_course_tc(m_wp["lat"], m_wp["lon"], B["lat"], B["lon"])
            leg2["Dist"] = leg["Dist"] - d_marker; leg2["Alt0"] = m_alt; leg2["Alt1"] = leg["Alt1"]
            leg2["HoldMin"] = 0.0
            new_legs.append(leg2)
        else:
            new_legs.append(leg)
        wp_idx += 1 if not res["toc_tod"] else 2
    st.session_state.effective_wps = new_wps
    st.session_state.legs = new_legs

# ===================== RECOMPUTE =====================
def recompute_all_by_leg():
    insert_toc_tod()
    st.session_state.computed_by_leg = []
    params = {"qnh": st.session_state.qnh, "oat": st.session_state.oat, "mag_var": st.session_state.mag_var,
              "mag_is_e": st.session_state.mag_is_e, "desc_angle": st.session_state.desc_angle}

    base_time = dt.datetime.combine(dt.date.today(), dt.time(0,0)) if st.session_state.start_clock.strip() else None
    if base_time:
        h, m = map(int, st.session_state.start_clock.split(":"))
        base_time = base_time.replace(hour=h, minute=m)

    carry_efob = float(st.session_state.start_efob)
    clock = base_time

    for leg in st.session_state.legs:
        res = build_segments(leg['TC'], leg['Dist'], leg['Alt0'], leg['Alt1'], leg['Wfrom'], leg['Wkt'], leg['CK'], params, leg.get('HoldMin', 0.0))
        phases = []
        t_cursor = 0
        for seg in res["segments"]:
            efob_start, efob_end = carry_efob, max(0.0, r10f(carry_efob - seg['burn']))
            c_start = (clock + dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if clock else f"T+{mmss(t_cursor)}"
            c_end = (clock + dt.timedelta(seconds=t_cursor + seg['time'])).strftime('%H:%M') if clock else f"T+{mmss(t_cursor + seg['time'])}"
            base_k = clock + dt.timedelta(seconds=t_cursor) if clock else None
            cps = res["ck_func"](seg, leg['CK'], base_k, efob_start) if seg['GS'] > 0 else []

            phases.append({"name": seg["name"], "label": phase_label(seg["name"]), "TH": seg["TH"], "MH": seg["MH"],
                           "GS": seg["GS"], "TAS": seg["TAS"], "ff": seg["ff"], "time": seg["time"], "dist": seg["dist"],
                           "alt0": seg["alt0"], "alt1": seg["alt1"], "burn": r10f(seg["burn"]), "efob_start": efob_start,
                           "efob_end": efob_end, "clock_start": c_start, "clock_end": c_end, "cps": cps,
                           "toc_tod": res["toc_tod"] if "Climb" in seg["name"] or "Descent" in seg["name"] else None,
                           "roc": res["roc"], "rod": res["rod"]})
            t_cursor += seg['time']; carry_efob = efob_end

        if clock: clock += dt.timedelta(seconds=sum(s['time'] for s in res["segments"]))
        st.session_state.computed_by_leg.append({"leg_ref": leg, "phases": phases, "tot_sec": res["tot_sec"], "tot_burn": res["tot_burn"]})

# ===================== CRUD LEGS =====================
def add_leg_from_points(pA, pB, wfrom, wkt, ck):
    tc = gc_course_tc(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    dist = gc_dist_nm(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    st.session_state.legs.append({"TC": float(tc), "Dist": float(dist), "Alt0": float(pA["alt"]), "Alt1": float(pB["alt"]),
                                 "Wfrom": int(wfrom), "Wkt": int(wkt), "CK": int(ck), "HoldMin": 0.0})

def update_leg(leg_ref, **vals): leg_ref.update(vals)
def delete_leg(idx): st.session_state.legs.pop(idx)

# ===================== PARSE CSV =====================
AD_CSV = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"

def dms_to_dd(token: str, is_lon=False):
    token = str(token).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([NSEW])$", token, re.I)
    if not m: return None
    value, hemi = m.groups()
    deg, minutes, seconds = (int(value[0:3]), int(value[3:5]), float(value[5:])) if is_lon and "." in value else \
                            (int(value[0:2]), int(value[2:4]), float(value[4:])) if "." in value else \
                            (int(value[0:3]), int(value[3:5]), int(value[5:])) if is_lon else \
                            (int(value[0:2]), int(value[2:4]), int(value[4:]))
    dd = deg + minutes/60 + seconds/3600
    return -dd if hemi.upper() in ["S", "W"] else dd

def parse_ad_df(df): return pd.DataFrame([{"type": "AD", "code": t[0], "name": " ".join(t[1:t.index(c[0])]), "city": " ".join(t[t.index(c[1])+1:]) if t.index(c[1])+1 < len(t) else None,
                                          "lat": dms_to_dd(c[0], False), "lon": dms_to_dd(c[1], True), "alt": 0.0}
                                         for line in df.iloc[:, 0].dropna() for t in [line.strip().split()] if t and not t[0].startswith(("Ident", "DEP/"))
                                         for c in [next((x for x in zip(t, t[1:]) if re.match(r"^\d+(?:\.\d+)?[NSEW]$", x[0]) and re.match(r"^\d+(?:\.\d+)?[NSEW]$", x[1])), None)]
                                         if c]).dropna(subset=["lat", "lon"])

def parse_loc_df(df): return pd.DataFrame([{"type": "LOC", "code": t[t.index(c[1])+1] if t.index(c[1])+1 < len(t) else t[0], "name": " ".join(t[:t.index(c[0])]),
                                           "sector": " ".join(t[t.index(c[1])+2:]) if t.index(c[1])+2 < len(t) else None,
                                           "lat": dms_to_dd(c[0], False), "lon": dms_to_dd(c[1], True), "alt": 0.0}
                                          for line in df.iloc[:, 0].dropna() for t in [line.strip().split()] if t and "Total de registos" not in t
                                          for c in [next((x for x in zip(t, t[1:]) if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", x[0]) and re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", x[1])), None)]
                                          if c]).dropna(subset=["lat", "lon"])

# ===================== HEADER =====================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 2, 2])
with col1: st.title("NAVLOG v12 — VFR Enhanced")
with col2: st.toggle("Mostrar Timeline/CPs", key="show_timeline")
with col3: 
    if st.button("➕ Adicionar WP Manual"):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
    if st.button("🗑️ Limpar Tudo"):
        st.session_state.wps = []; st.session_state.legs = []; st.session_state.computed_by_leg = []; st.session_state.effective_wps = []
st.markdown("</div>", unsafe_allow_html=True)

# ===================== GLOBAL PARAMETERS =====================
with st.sidebar:
    with st.form("global_params"):
        st.subheader("Parâmetros Globais")
        col1, col2 = st.columns(2)
        with col1: st.number_input("QNH (hPa)", 900, 1050, key="qnh", value=1013)
        with col2: st.number_input("OAT (°C)", -40, 50, key="oat", value=15)
        col3, col4 = st.columns(2)
        with col3: st.number_input("EFOB Inicial (L)", 0.0, 200.0, key="start_efob", value=85.0, step=0.5)
        with col4: st.text_input("Hora Off-Blocks (HH:MM)", key="start_clock")
        col5, col6 = st.columns(2)
        with col5: st.number_input("Peso (kg)", 450.0, 700.0, key="weight", value=650.0, step=1.0)
        with col6: st.number_input("Ângulo Descida (°)", 1.0, 6.0, key="desc_angle", value=3.0, step=0.1)
        col7, col8 = st.columns(2)
        with col7: st.number_input("Vento FROM (°T)", 0, 360, key="wind_from", value=0, step=1)
        with col8: st.number_input("Vento (kt)", 0, 150, key="wind_kt", value=0, step=1)
        st.number_input("CP Padrão (min)", 1, 10, key="ck_default", value=2, step=1)
        if st.form_submit_button("Aplicar"):
            if st.session_state.legs:
                for leg in st.session_state.legs:
                    leg["Wfrom"] = st.session_state.wind_from
                    leg["Wkt"] = st.session_state.wind_kt
            st.rerun()

# ===================== LOAD CSV & ADD WAYPOINTS =====================
try:
    ad_df = parse_ad_df(pd.read_csv(AD_CSV))
    loc_df = parse_loc_df(pd.read_csv(LOC_CSV))
except Exception as e:
    ad_df = pd.DataFrame(columns=["type", "code", "name", "city", "lat", "lon", "alt"])
    loc_df = pd.DataFrame(columns=["type", "code", "name", "sector", "lat", "lon", "alt"])
    st.warning("Erro ao carregar CSVs. Verifique os arquivos.")

st.subheader("Adicionar Waypoints")
col1, col2 = st.columns([3, 1])
with col1: query = st.text_input("Procurar AD/Localidade", placeholder="Ex: LPPT, ABRANTES...")
with col2: alt_wp = st.number_input("Altitude (ft)", 0.0, 18000.0, value=3000.0, step=100.0)
filtered = pd.concat([ad_df, loc_df]).drop_duplicates(subset=["code"])
if query:
    filtered = filtered[filtered.apply(lambda r: any(query.lower() in str(v).lower() for v in r.values), axis=1)]
options = [f"{row['code']} - {row['name']} ({row['type']})" for _, row in filtered.iterrows()]
selected = st.multiselect("Selecionar Waypoints", options, key="wp_select")
if st.button("Adicionar Selecionados"):
    for sel in selected:
        code = sel.split(" - ")[0]
        row = filtered[filtered['code'] == code].iloc[0]
        st.session_state.wps.append({"name": row["code"], "lat": float(row["lat"]), "lon": float(row["lon"]), "alt": float(alt_wp)})
    st.success(f"Adicionados {len(selected)} waypoints.")
    st.rerun()

# ===================== WAYPOINT EDITOR =====================
if st.session_state.wps:
    st.subheader("Editar Waypoints")
    for i, wp in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1}: {wp['name']}"):
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
            with col1: name = st.text_input("Nome", wp["name"], key=f"name_{i}")
            with col2: lat = st.number_input("Latitude", -90.0, 90.0, wp["lat"], step=0.0001, key=f"lat_{i}")
            with col3: lon = st.number_input("Longitude", -180.0, 180.0, wp["lon"], step=0.0001, key=f"lon_{i}")
            with col4: alt = st.number_input("Altitude (ft)", 0.0, 18000.0, wp["alt"], step=50.0, key=f"alt_{i}")
            with col5:
                if st.button("↑", key=f"up_{i}") and i > 0: st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]; st.rerun()
                if st.button("↓", key=f"down_{i}") and i < len(st.session_state.wps)-1: st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]; st.rerun()
                if st.button("Remover", key=f"rm_{i}"): st.session_state.wps.pop(i); st.rerun()
            if (name, lat, lon, alt) != (wp["name"], wp["lat"], wp["lon"], wp["alt"]):
                st.session_state.wps[i] = {"name": name, "lat": float(lat), "lon": float(lon), "alt": float(alt)}

# ===================== GENERATE LEGS =====================
if st.button("Gerar Legs a partir de Waypoints"):
    if len(st.session_state.wps) >= 2:
        st.session_state.legs = [add_leg_from_points(st.session_state.wps[i], st.session_state.wps[i+1], st.session_state.wind_from, st.session_state.wind_kt, st.session_state.ck_default)
                                for i in range(len(st.session_state.wps)-1)]
        st.success(f"Criadas {len(st.session_state.legs)} legs.")
        recompute_all_by_leg()

# ===================== LEG INPUTS & PHASES =====================
if st.session_state.legs:
    recompute_all_by_leg()
    total_sec_all, total_burn_all, efob_final = 0, 0.0, None

    for idx, leg in enumerate(st.session_state.legs):
        with st.expander(f"Leg {idx+1} - Inputs", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                tc = st.number_input("True Course (°T)", 0.0, 359.9, leg["TC"], step=0.1, key=f"tc_{idx}")
                dist = st.number_input("Distância (nm)", 0.0, 500.0, leg["Dist"], step=0.1, key=f"dist_{idx}")
            with col2:
                alt0 = st.number_input("Alt. Inicial (ft)", 0.0, 30000.0, leg["Alt0"], step=50.0, key=f"alt0_{idx}")
                alt1 = st.number_input("Alt. Final (ft)", 0.0, 30000.0, leg["Alt1"], step=50.0, key=f"alt1_{idx}")
            with col3:
                wfrom = st.number_input("Vento FROM (°T)", 0, 360, leg["Wfrom"], step=1, key=f"wfrom_{idx}")
                wkt = st.number_input("Vento (kt)", 0, 150, leg["Wkt"], step=1, key=f"wkt_{idx}")
                ck = st.number_input("Checkpoints (min)", 1, 10, leg["CK"], step=1, key=f"ck_{idx}")
            col4, col5, col6 = st.columns([1, 1, 2])
            with col4: hold_min = st.number_input("Hold (min)", 0.0, 180.0, leg["HoldMin"], step=0.5, key=f"hold_{idx}")
            with col5: 
                st.write("FF Hold fixo: 20 L/h")
            with col6:
                if st.button("Salvar", key=f"save_{idx}"):
                    update_leg(leg, TC=tc, Dist=dist, Alt0=alt0, Alt1=alt1, Wfrom=wfrom, Wkt=wkt, CK=ck, HoldMin=hold_min)
                    recompute_all_by_leg()
                if st.button("Remover", key=f"del_{idx}"):
                    delete_leg(idx); recompute_all_by_leg(); st.stop()

        comp_leg = st.session_state.computed_by_leg[idx]
        phases = comp_leg["phases"]
        st.markdown(f"<div class='kvrow'><div class='kv'>Fases: {len(phases)}</div><div class='kv'>ETE: {hhmmss(comp_leg['tot_sec'])}</div><div class='kv'>Burn: {comp_leg['tot_burn']:.1f} L</div></div>", unsafe_allow_html=True)

        for p in phases:
            with st.container():
                st.subheader(f"Fase {idx+1}.{phases.index(p)+1}: {p['label']}")
                st.caption(p["name"])
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"**Alt:** {int(p['alt0'])} → {int(p['alt1'])} ft")
                    st.markdown(f"**TH/MH:** {rang(p['TH'])}T / {rang(p['MH'])}M")
                    st.markdown(f"**GS/TAS:** {rint(p['GS'])} / {rint(p['TAS'])} kt")
                    st.markdown(f"**FF:** {rint(p['ff'])} L/h")
                with col2:
                    st.metric("Tempo", mmss(p["time"]))
                    st.metric("Combustível", f"{p['burn']:.1f} L")
                col3, col4, col5 = st.columns(3)
                with col3: st.markdown(f"**Relógio:** {p['clock_start']} → {p['clock_end']}")
                with col4: st.markdown(f"**EFOB:** {p['efob_start']:.1f} → {p['efob_end']:.1f} L")
                with col5:
                    if "Climb" in p["name"]: st.markdown(f"**ROC:** {rint(p['roc'])} ft/min")
                    elif "Descent" in p["name"]: st.markdown(f"**ROD:** {rint(p['rod'])} ft/min")
                if p["toc_tod"]: st.info(f"**{p['toc_tod']['type']}** at T+{mmss(p['toc_tod']['t'])}")
                if st.session_state.show_timeline and p["GS"] > 0:
                    timeline(p, p["cps"], p["clock_start"], p["clock_end"], p["toc_tod"])

        total_sec_all += comp_leg["tot_sec"]
        total_burn_all += comp_leg["tot_burn"]
        if phases: efob_final = phases[-1]["efob_end"]

    st.markdown(f"<div class='kvrow'><div class='kv'>ETE Total: {hhmmss(total_sec_all)}</div><div class='kv'>Burn Total: {r10f(total_burn_all):.1f} L</div>"
                f"<div class='kv'>EFOB Final: {efob_final:.1f} L</div></div>", unsafe_allow_html=True)

# ===================== MAP (VFR STYLE) =====================
def triangle_coords(lat, lon, heading_deg, h_nm=0.8, w_nm=0.55):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon = dest_point(lat, lon, heading_deg, h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

if len(st.session_state.effective_wps) >= 2 and st.session_state.legs and st.session_state.computed_by_leg:
    path_data, tick_data, tri_data, label_data, mh_data, wp_data = [], [], [], [], [], []

    wps = st.session_state.effective_wps
    for i in range(len(wps)-1):
        A, B = wps[i], wps[i+1]
        leg, phases = st.session_state.legs[i], st.session_state.computed_by_leg[i]["phases"]
        path_data.append({"path": [[A["lon"], A["lat"]], [B["lon"], B["lat"]]], "name": f"Leg {i+1}"})

        # Ticks every 2 min
        segments = [(p["time"], p["GS"]) for p in phases if p["GS"] > 0]
        if not segments: segments = [(comp_leg["tot_sec"], 0)]
        total_dist = leg["Dist"]
        k = 1
        while k * 120 <= sum(s[0] for s in segments):
            t_target = k * 120
            dist = 0.0; t_acc = 0
            for dur, gs in segments:
                if t_acc + dur >= t_target:
                    dt = t_target - t_acc
                    dist += gs * (dt / 3600.0)
                    break
                dist += gs * (dur / 3600.0)
                t_acc += dur
            dist = min(dist, total_dist)
            lat_m, lon_m = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], dist)
            tc = leg["TC"]
            half_nm = 0.15
            left_lat, left_lon = dest_point(lat_m, lon_m, tc-90, half_nm)
            right_lat, right_lon = dest_point(lat_m, lon_m, tc+90, half_nm)
            tick_data.append({"path": [[left_lon, left_lat], [right_lon, right_lat]]})
            k += 1

        # Dog House
        mid_lat, mid_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], total_dist/2.0)
        off_lat, off_lon = dest_point(mid_lat, mid_lon, leg["TC"]+90, 0.3)
        tri = triangle_coords(off_lat, off_lon, leg["TC"], h_nm=0.8, w_nm=0.55)
        tri_data.append({"polygon": tri})
        ref = next((p for p in phases if p["GS"] > 0), phases[0])
        label = f"{rang(ref['TH'])} T {r10f(leg['Dist'])} nm GS {rint(ref['GS'])} ETE {mmss(comp_leg['tot_sec'])}"
        label_pos = dest_point(off_lat, off_lon, leg["TC"]+90, 0.35)
        label_data.append({"position": [label_pos[1], label_pos[0]], "text": label})
        mh_data.append({"position": [off_lon, off_lat], "text": f"{rang(ref['MH'])}M", "size": 20, "color": [255, 0, 0]})

        # Waypoints
        wp_data.append({"lon": A["lon"], "lat": A["lat"], "name": A["name"], "size": 15})
        if i == len(wps)-2: wp_data.append({"lon": B["lon"], "lat": B["lat"], "name": B["name"], "size": 15})

    # Layers
    base_map = pdk.Layer("TileLayer", data="https://tile.opentopomap.org/{z}/{x}/{y}.png", tile_size=256, min_zoom=0, max_zoom=19)
    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[180, 0, 255, 220], width_min_pixels=4)
    tick_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0, 0, 0, 255], width_min_pixels=2)
    tri_layer = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon", get_fill_color=[255, 255, 255, 230],
                          get_line_color=[0, 0, 0, 255], line_width_min_pixels=2, stroked=True, filled=True)
    text_layer = pdk.Layer("TextLayer", data=label_data, get_position="position", get_text="text", get_size=14, get_color=[0, 0, 0])
    mh_layer = pdk.Layer("TextLayer", data=mh_data, get_position="position", get_text="text", get_size="size", get_color="color")
    wp_layer = pdk.Layer("ScatterplotLayer", data=wp_data, get_position=["lon", "lat"], get_radius=200, get_fill_color=[0, 255, 0, 200], pickable=True)

    mean_lat = sum(w["lat"] for w in wps) / len(wps)
    mean_lon = sum(w["lon"] for w in wps) / len(wps)
    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=8, pitch=0)
    deck = pdk.Deck(map_style=None, initial_view_state=view_state, layers=[base_map, route_layer, tick_layer, wp_layer, tri_layer, text_layer, mh_layer],
                    tooltip={"text": "{name}"})
    st.pydeck_chart(deck)
else:
    st.info("Adicione pelo menos 2 waypoints e gere as legs para visualizar o mapa.")
