# app.py — NAVLOG v11.2 (VFR map, TOC/TOD viram WPs, dog houses, riscas 2 min, MH destaque, seleção de WPs)
# Requisitos: streamlit, pydeck, pandas
# — Velocidades fixas: 70 (climb), 90 (cruise), 90 (descent)
# — FF fixo: 20 L/h
# — TOC/TOD viram waypoints e partem legs
# — Mapa VFR-like com OpenTopoMap / OSM Standard via TileLayer (sem Voyager)
# — MH destacado no mapa

import streamlit as st
import pydeck as pdk
import pandas as pd
import math, re, datetime as dt
from math import sin, asin, radians, degrees

# ===================== PAGE / STYLE =====================
st.set_page_config(page_title="NAVLOG v11.2 — VFR + TOC/TOD WPs", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.badge{background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:11px;margin-left:6px}
.sep{height:1px;background:var(--line);margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid #ddd;padding-bottom:8px}
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
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = degrees(asin(s))
    th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False):
    # se var for Este e east_is_neg=True, subtrai; caso contrário, soma
    return wrap360(th - var if east_is_neg else th + var)

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

# ===================== CONSTANTES SIMPLES (sem AFM) =====================
CLIMB_TAS_KT   = 70.0
CRUISE_TAS_KT  = 90.0
DESC_TAS_KT    = 90.0
FF_LPH         = 20.0   # L/h
ROC_FTPM_DEF   = 700.0  # ROC default global (ajustável no UI)

# ===================== STATE =====================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("mag_var", 1.0); ens("mag_is_e", False)
ens("weight", 650.0)  # não usado diretamente, mantido para futuro
ens("desc_angle", 3.0)
ens("roc_global", ROC_FTPM_DEF)
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

# ===================== CÁLCULO POR LEG (com TOC/TOD como novos WPs) =====================
def build_segments(tc, dist, alt0, alt1, wfrom, wkt, ck_min, params, hold_min=0.0, hold_ff_input=0.0):
    mag_var, mag_is_e = params['mag_var'], params['mag_is_e']
    desc_angle = params['desc_angle']; roc_global = params['roc_global']

    # TAS fixas
    TAS_climb = CLIMB_TAS_KT
    TAS_cru   = CRUISE_TAS_KT
    TAS_desc  = DESC_TAS_KT

    # Ventos → TH/MH/GS
    _, THc, GScl = wind_triangle(tc, TAS_climb, wfrom, wkt)
    _, THr, GScr = wind_triangle(tc, TAS_cru,   wfrom, wkt)
    _, THd, GSde = wind_triangle(tc, TAS_desc,  wfrom, wkt)

    MHc = apply_var(THc, mag_var, mag_is_e)
    MHr = apply_var(THr, mag_var, mag_is_e)
    MHd = apply_var(THd, mag_var, mag_is_e)

    # Ângulo de descida → ROD aprox (regra dos 5 * GS para 3°)
    ROD = max(100.0, GScr * 5.0 * (desc_angle / 3.0))

    profile = "LEVEL" if abs(alt1 - alt0) < 1e-6 else ("CLIMB" if alt1 > alt0 else "DESCENT")
    segs, toc_tod_marker = [], None
    # TOC/TOD info para criação de WP virtual (distância a partir do início da leg)
    vt_wp = None  # dict: {"type":"TOC"/"TOD","at_nm":float}

    if profile == "CLIMB":
        t_need_min = (alt1 - alt0) / max(roc_global, 1e-6)     # minutos
        d_need = GScl * (t_need_min / 60.0)
        if d_need < dist:
            # Climb até TOC → depois Cruise
            tA = rt10(t_need_min * 60)
            segs.append({"name":"Climb → TOC","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_LPH,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise (após TOC)","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_LPH,"time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            toc_tod_marker = {"type":"TOC","t": rt10(t_need_min*60)}
            vt_wp = {"type":"TOC","at_nm": d_need}
        else:
            # não atinge altitude desejada
            tA = rt10((dist / max(GScl,1e-9)) * 3600)
            gained = roc_global * (tA / 60.0)
            segs.append({"name":"Climb (não atinge)","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_LPH,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0+gained})
    elif profile == "DESCENT":
        # Primeiro Cruise até TOD, depois Descent
        t_need_min = (alt0 - alt1) / max(ROD, 1e-6)   # minutos
        d_need = GSde * (t_need_min / 60.0)
        if d_need < dist:
            # Cruise até TOD
            cruise_before = dist - d_need
            tCru = rt10((cruise_before / max(GScr,1e-9)) * 3600)
            if cruise_before > 0:
                segs.append({"name":"Cruise (até TOD)","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_LPH,"time":tCru,"dist":cruise_before,"alt0":alt0,"alt1":alt0})
            # Descida após TOD
            tDes = rt10(t_need_min * 60)
            segs.append({"name":"Descent após TOD","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_LPH,"time":tDes,"dist":d_need,"alt0":alt0,"alt1":alt1})
            toc_tod_marker = {"type":"TOD","t": rt10(tCru)}
            vt_wp = {"type":"TOD","at_nm": cruise_before}
        else:
            tA = rt10((dist / max(GSde,1e-9)) * 3600)
            lost = ROD * (tA / 60.0)
            segs.append({"name":"Descent (não atinge)","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_LPH,"time":tA,"dist":dist,"alt0":alt0,"alt1":max(0.0, alt0 - lost)})
    else:
        tA = rt10((dist / max(GScr,1e-9)) * 3600)
        segs.append({"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_LPH,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0})

    # HOLD opcional (FF = 20 L/h)
    hold_min = max(0.0, float(hold_min))
    if hold_min > 0.0:
        hold_ff = float(hold_ff_input) if float(hold_ff_input) > 0 else FF_LPH
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
        "roc": roc_global,
        "rod": ROD,
        "toc_tod": toc_tod_marker,
        "vt_wp": vt_wp,
        "ck_func": cps
    }

# ===================== RECOMPUTE POR LEG + construir legs partidas em TOC/TOD =====================
def recompute_all_by_leg():
    st.session_state.computed_by_leg = []
    params = dict(
        mag_var=st.session_state.mag_var,
        mag_is_e=st.session_state.mag_is_e,
        desc_angle=st.session_state.desc_angle,
        roc_global=st.session_state.roc_global
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
            params=params, hold_min=leg.get('HoldMin',0.0), hold_ff_input=leg.get('HoldFF',0.0)
        )

        # Calcular WPs virtuais (TOC/TOD) e "legs visuais" partidas
        A = leg['A']; B = leg['B']  # (lat,lon) dos endpoints originais
        visual_legs = []
        phases = []
        t_cursor = 0
        efob_local = carry_efob

        # Ponto virtual (se existir)
        vt_coord = None
        if res["vt_wp"] is not None:
            at_nm = res["vt_wp"]["at_nm"]
            vt_lat, vt_lon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], at_nm)
            vt_coord = dict(lat=vt_lat, lon=vt_lon, name=res["vt_wp"]["type"])
        # Construir legs partidas conforme segmentos com GS>0
        prev_point = dict(lat=A["lat"], lon=A["lon"], name=A["name"])
        consumed_nm_along = 0.0

        for idx_seg, seg in enumerate(res["segments"]):
            # Determinar endpoint desta sub-leg (pode ser VT WP ou final B)
            seg_dist = seg["dist"]
            consumed_nm_next = consumed_nm_along + seg_dist
            if vt_coord and abs(consumed_nm_next - res["vt_wp"]["at_nm"]) < 1e-3:
                end_point = vt_coord
            elif vt_coord and consumed_nm_along < res["vt_wp"]["at_nm"] < consumed_nm_next:
                # raríssimo com arredondamentos; força endpoint ser o VT
                end_point = vt_coord
                seg_dist = res["vt_wp"]["at_nm"] - consumed_nm_along
                consumed_nm_next = res["vt_wp"]["at_nm"]
            elif consumed_nm_next >= leg['Dist'] - 1e-6:
                end_point = dict(lat=B["lat"], lon=B["lon"], name=B["name"])
            else:
                # ponto ao longo
                elat, elon = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], consumed_nm_next)
                end_point = dict(lat=elat, lon=elon, name=f"L{leg['idx']}.{idx_seg+1}")

            # Book-keeping fase/cartão
            efob_start = efob_local
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
                "time": seg["time"], "dist": seg_dist, "alt0": seg["alt0"], "alt1": seg["alt1"],
                "burn": r10f(seg["burn"]), "efob_start": efob_start, "efob_end": efob_end,
                "clock_start": c_start, "clock_end": c_end, "cps": cps,
                "toc_tod": (res["toc_tod"] if idx_seg==0 and ("Climb" in seg["name"] or "Descent" in seg["name"]) else None),
                "ck": leg["CK"]
            })
            # Leg visual (para mapa, riscas e doghouse), só se houver deslocamento
            if seg["GS"] > 0 and seg_dist > 0:
                visual_legs.append({
                    "from": prev_point, "to": end_point, "TC": leg["TC"], "MH": seg["MH"],
                    "GS": seg["GS"], "dist": seg_dist, "time": seg["time"], "label_name": seg["name"]
                })

            # Avançar cursores
            t_cursor += seg['time']; efob_local = efob_end
            consumed_nm_along = consumed_nm_next
            prev_point = end_point

        if clock: clock = clock + dt.timedelta(seconds=sum(s['time'] for s in res["segments"]))
        st.session_state.computed_by_leg.append({
            "leg_ref": leg, "phases": phases, "visual_legs": visual_legs,
            "vt_wp": vt_coord,
            "tot_sec": sum(p["time"] for p in phases),
            "tot_burn": r10f(sum(p["burn"] for p in phases))
        })

# ===================== CRUD / BUILD LEGS =====================
def add_leg_from_points(pA, pB, altA, altB, wfrom, wkt, ck):
    tc = gc_course_tc(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    dist = gc_dist_nm(pA["lat"], pA["lon"], pB["lat"], pB["lon"])
    st.session_state.legs.append(dict(
        idx=len(st.session_state.legs)+1,
        TC=float(tc), Dist=float(dist), Alt0=float(altA), Alt1=float(altB),
        Wfrom=int(wfrom), Wkt=int(wkt), CK=int(ck), HoldMin=0.0, HoldFF=0.0,
        A=dict(lat=pA["lat"], lon=pA["lon"], name=pA["name"]),
        B=dict(lat=pB["lat"], lon=pB["lon"], name=pB["name"])
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
with h1: st.title("NAVLOG — v11.2 (VFR + TOC/TOD WPs)")
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
        st.session_state.mag_var = st.number_input("Variação Magnética (°)", -50.0, 50.0, float(st.session_state.mag_var), step=0.1)
        st.session_state.mag_is_e = st.toggle("Var E é negativa?", value=st.session_state.mag_is_e, help="Se ligado: MH = TH − Var")
    with p2:
        st.session_state.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
    with p3:
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.5)
        st.session_state.roc_global = st.number_input("ROC global (ft/min)", 200.0, 1500.0, float(st.session_state.roc_global), step=25.0)
    with p4:
        st.session_state.ck_default = st.number_input("Checkpoints (min)", 1, 10, int(st.session_state.ck_default), step=1)
    w1, w2 = st.columns(2)
    with w1: st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
    with w2: st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    submitted = st.form_submit_button("Aplicar parâmetros")
if submitted and st.session_state.legs:
    for L in st.session_state.legs:
        L["Wfrom"] = int(st.session_state.wind_from); L["Wkt"] = int(st.session_state.wind_kt)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== LER CSVs LOCAIS + PESQUISA COM SELEÇÃO =====================
try:
    ad_raw  = pd.read_csv(AD_CSV)
    loc_raw = pd.read_csv(LOC_CSV)
    ad_df  = parse_ad_df(ad_raw)
    loc_df = parse_loc_df(loc_raw)
except Exception as e:
    ad_df  = pd.DataFrame(columns=["type","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["type","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

def filter_df(df, q):
    if not q: return df
    tq = q.lower().strip()
    return df[df.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)]

cflt1, cflt2, cbtn = st.columns([3,3,1.5])
with cflt1: qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, NISA…")
with cflt2: alt_wp = st.number_input("Altitude para WPs adicionados (ft)", 0.0, 18000.0, 3000.0, step=100.0)
st.caption("Seleciona **apenas** os resultados que queres adicionar (resolve o caso de 'Nisa' com múltiplos).")

ad_f  = filter_df(ad_df, qtxt)
loc_f = filter_df(loc_df, qtxt)
sel_col1, sel_col2 = st.columns(2)
with sel_col1:
    st.subheader("AD/HEL/ULM")
    for i, r in ad_f.reset_index(drop=True).iterrows():
        key = f"sel_ad_{i}_{r['lat']}_{r['lon']}"
        txt = f"{str(r['code'])} — {str(r['name'])} ({r['lat']:.5f}, {r['lon']:.5f})"
        st.checkbox(txt, key=key)
with sel_col2:
    st.subheader("Localidades")
    for i, r in loc_f.reset_index(drop=True).iterrows():
        key = f"sel_loc_{i}_{r['lat']}_{r['lon']}"
        txt = f"{str(r['code'])} — {str(r['name'])} ({r['lat']:.5f}, {r['lon']:.5f})"
        st.checkbox(txt, key=key)

if st.button("➕ Adicionar selecionados"):
    added = 0
    # ADs
    for i, r in ad_f.reset_index(drop=True).iterrows():
        key = f"sel_ad_{i}_{r['lat']}_{r['lon']}"
        if st.session_state.get(key, False):
            st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
            added += 1
    # Locs
    for i, r in loc_f.reset_index(drop=True).iterrows():
        key = f"sel_loc_{i}_{r['lat']}_{r['lon']}"
        if st.session_state.get(key, False):
            st.session_state.wps.append({"name": str(r["code"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(alt_wp)})
            added += 1
    st.success(f"Adicionados {added} WPs.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

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
                    st.experimental_rerun()
                if dn and i < len(st.session_state.wps)-1:
                    st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
                    st.experimental_rerun()
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i); st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ===================== GERAR LEGS A PARTIR DOS WPs =====================
cgl1, cgl2, _ = st.columns([2,2,6])
with cgl1: pass
with cgl2: gen = st.button("Gerar/Atualizar legs a partir dos WAYPOINTS", type="primary", use_container_width=True)

if gen and len(st.session_state.wps) >= 2:
    st.session_state.legs = []
    for i in range(len(st.session_state.wps)-1):
        A = st.session_state.wps[i]; B = st.session_state.wps[i+1]
        add_leg_from_points(A, B, A["alt"], B["alt"], st.session_state.wind_from, st.session_state.wind_kt, st.session_state.ck_default)
    st.success(f"Criadas {len(st.session_state.legs)} legs.")
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
            j1, j2, j3 = st.columns([1.2,1.2,6])
            with j1: HoldMin = st.number_input(f"Espera (min) — L{idx_leg+1}", 0.0, 180.0, float(leg.get('HoldMin',0.0)), step=0.5, key=f"HoldMin_{idx_leg}")
            with j2: HoldFF  = st.number_input(f"FF espera (L/h) — L{idx_leg+1} (0=20)", 0.0, 60.0, float(leg.get('HoldFF',0.0)), step=0.1, key=f"HoldFF_{idx_leg}")
            with j3:
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
                    + f"<div class='kv'>TH/MH: <b>{rang(c['TH'])}T / <span style='color:#d00;font-size:14px'>{rang(c['MH'])}M</span></b></div>"
                    + f"<div class='kv'>GS/TAS: <b>{rint(c['GS'])}/{rint(c['TAS'])} kt</b></div>"
                    + f"<div class='kv'>FF: <b>{rint(FF_LPH)} L/h</b></div>"
                    + "</div>", unsafe_allow_html=True
                )
            with right:
                st.metric("Tempo", mmss(c["time"]))
                st.metric("Fuel desta fase (L)", f"{c['burn']:.1f}")

            r1, r2, r3 = st.columns(3)
            with r1: st.markdown(f"**Relógio** — {c['clock_start']} → {c['clock_end']}")
            with r2: st.markdown(f"**EFOB** — Start {c['efob_start']:.1f} L → End {c['efob_end']:.1f} L")
            with r3:
                if "Descent" in c["name"]: st.markdown(f"**ROD ref.** — {rint(st.session_state.desc_angle/3.0 * 5 * c['GS'])} ft/min")
                else: st.markdown(f"**ROC ref.** — {rint(st.session_state.roc_global)} ft/min")

            if c["toc_tod"] is not None: st.info(f"Marcador: **{c['toc_tod']['type']}** em T+{mmss(c['toc_tod']['t'])}")
            if st.session_state.show_timeline and c["GS"] > 0:
                timeline({"GS":c["GS"],"TAS":c["TAS"],"ff":FF_LPH,"time":c["time"]}, c["cps"], c["clock_start"], c["clock_end"], c["toc_tod"])

            warns = []
            if c["dist"] == 0 and abs(c["alt1"]-c["alt0"])>50: warns.append("Distância 0 com variação de altitude.")
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

# ===================== MAPA (VFR-like) — TileLayer OSM/OpenTopo + rota, riscas, dog houses, WPs =====================
def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [[bl_lon, bl_lat], [apex_lon, apex_lat], [br_lon, br_lat], [bl_lon, bl_lat]]

def bounds_from_points(points):
    lats = [p["lat"] for p in points]; lons = [p["lon"] for p in points]
    return (min(lats), max(lats), min(lons), max(lons))

def approx_zoom(min_lat, max_lat, min_lon, max_lon):
    span_lat = max(0.0001, max_lat - min_lat)
    span_lon = max(0.0001, max_lon - min_lon)
    zl = math.log2(180.0 / span_lat)
    zo = math.log2(360.0 / span_lon)
    z  = max(4.0, min(12.0, min(zl, zo)))  # clamp
    return float(z)

if len(st.session_state.wps) >= 2 and st.session_state.legs and st.session_state.computed_by_leg:
    # Base VFR-like (OpenTopoMap por defeito; opção para OSM Standard)
    base_choice = st.radio("Base do mapa (VFR-like)", ["OpenTopoMap (relevo/terreno)", "OSM Standard (estradas/locais)"], index=0, horizontal=True)
    base_url = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png" if base_choice.startswith("OpenTopo") else "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    tile_layer = pdk.Layer(
        "TileLayer",
        data=base_url,
        min_zoom=0, max_zoom=18, tile_size=256,
        render_sub_layers=pdk.DeckGLJSExpression("""
            props => new deck.BitmapLayer(props, {
              data: null,
              image: props.data,
              bounds: [props.tile.bbox[0], props.tile.bbox[1], props.tile.bbox[2], props.tile.bbox[3]]
            })
        """)
    )

    path_data, tick_data, tri_data, mh_labels, info_labels, wp_points, wp_labels = [], [], [], [], [], [], []

    # WPs originais + virtuais (TOC/TOD)
    for i in range(len(st.session_state.wps)-1):
        comp = st.session_state.computed_by_leg[i]
        leg = st.session_state.legs[i]
        phases = comp["phases"]
        vlegs = comp["visual_legs"]

        # Rota partida (cada sub-leg)
        for v in vlegs:
            path_data.append({"path": [[v["from"]["lon"], v["from"]["lat"]], [v["to"]["lon"], v["to"]["lat"]]], "name": v["label_name"]})

        # Riscas a cada 2 minutos com GS real por sub-leg
        for v in vlegs:
            segments = [(v["time"], v["GS"])]  # cada v é já um segmento
            total_leg_dist = v["dist"]
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
                latm, lonm = point_along_gc(v["from"]["lat"], v["from"]["lon"], v["to"]["lat"], v["to"]["lon"], dist_target)
                tc = leg["TC"]
                half_nm = 0.15
                left_lat, left_lon   = dest_point(latm, lonm, tc-90, half_nm)
                right_lat, right_lon = dest_point(latm, lonm, tc+90, half_nm)
                tick_data.append({"path": [[left_lon, left_lat], [right_lon, right_lat]]})
                k += 1

        # Dog house por sub-leg (triângulo + labels)
        for v in vlegs:
            mid_lat, mid_lon = point_along_gc(v["from"]["lat"], v["from"]["lon"], v["to"]["lat"], v["to"]["lon"], v["dist"]/2.0)
            off_lat, off_lon = dest_point(mid_lat, mid_lon, leg["TC"]+90, 0.3)
            tri = triangle_coords(off_lat, off_lon, leg["TC"], h_nm=0.9, w_nm=0.65)
            tri_data.append({"polygon": tri})
            # Label MH (grande e visível) + info (menor)
            mh_text = f"MH {rang(v['MH'])}°"
            info_text = f"TC {rang(leg['TC'])}° • {r10f(v['dist'])} nm • GS {rint(v['GS'])} • ETE {mmss(v['time'])}"
            label_pos = dest_point(off_lat, off_lon, leg["TC"]+90, 0.35)
            mh_labels.append({"position":[label_pos[1], label_pos[0]], "text": mh_text})
            info_labels.append({"position":[label_pos[1], label_pos[0]], "text": info_text})

        # WPs: início, TOC/TOD (se existir), fim
        wp_points.append({"position":[leg["A"]["lon"], leg["A"]["lat"]], "color":[0,120,255], "radius":80, "name":leg["A"]["name"]})
        wp_labels.append({"position":[leg["A"]["lon"], leg["A"]["lat"]], "text": leg["A"]["name"]})
        if comp["vt_wp"] is not None:
            wp_points.append({"position":[comp["vt_wp"]["lon"], comp["vt_wp"]["lat"]], "color":[255,120,0], "radius":70, "name":comp["vt_wp"]["name"]})
            wp_labels.append({"position":[comp["vt_wp"]["lon"], comp["vt_wp"]["lat"]], "text": comp["vt_wp"]["name"]})
        wp_points.append({"position":[leg["B"]["lon"], leg["B"]["lat"]], "color":[0,120,255], "radius":80, "name":leg["B"]["name"]})
        wp_labels.append({"position":[leg["B"]["lon"], leg["B"]["lat"]], "text": leg["B"]["name"]})

    # LAYERS
    route_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_color=[180, 0, 255, 220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0, 0, 0, 255], width_min_pixels=2)
    tri_layer   = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon",
                            get_fill_color=[255,255,255,230], get_line_color=[0,0,0,255],
                            line_width_min_pixels=2, stroked=True, filled=True)
    # MH grande (bem visível)
    text_layer_mh  = pdk.Layer("TextLayer", data=mh_labels, get_position="position", get_text="text",
                               get_size=20, get_color=[210,0,0], get_alignment_baseline="'center'",
                               get_pixel_offset=[0, -14])
    # Info menor
    text_layer_info = pdk.Layer("TextLayer", data=info_labels, get_position="position", get_text="text",
                                get_size=13, get_color=[0,0,0], get_alignment_baseline="'center'",
                                get_pixel_offset=[0, 10])
    # WPs marcados + labels
    wp_layer = pdk.Layer("ScatterplotLayer", data=wp_points, get_position="position", get_radius="radius",
                         get_fill_color="color", stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1)
    wp_text  = pdk.Layer("TextLayer", data=wp_labels, get_position="position", get_text="text",
                         get_size=14, get_color=[0,0,0], get_alignment_baseline="'top'", get_pixel_offset=[0, 14])

    # View — centrado à VFR (fit aprox aos WPs)
    pts = []
    for L in st.session_state.legs:
        pts.extend([{"lat":L["A"]["lat"],"lon":L["A"]["lon"]},{"lat":L["B"]["lat"],"lon":L["B"]["lon"]}])
    minlat, maxlat, minlon, maxlon = bounds_from_points(pts)
    center_lat = (minlat+maxlat)/2.0; center_lon = (minlon+maxlon)/2.0
    zoom = approx_zoom(minlat, maxlat, minlon, maxlon)

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)
    deck = pdk.Deck(
        map_style=None,  # sem Voyager / sem Mapbox style
        initial_view_state=view_state,
        layers=[tile_layer, route_layer, ticks_layer, tri_layer, text_layer_mh, text_layer_info, wp_layer, wp_text],
        tooltip={"text": "{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona pelo menos 2 waypoints e gera as legs para ver a rota, riscas e dog houses.")
