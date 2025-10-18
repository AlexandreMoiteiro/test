# app.py — NAVLOG v10 (AFM) — UI Otimizada & Intuitiva
# Objetivo: Interface moderna, fluida e focada no usuário. Inputs agrupados logicamente,
# navegação por tabs para legs, cartões interativos por fase, visualizações aprimoradas,
# validações e feedback em tempo real. Mantém todas as funcionalidades originais.

import streamlit as st
import datetime as dt
import math
from math import sin, asin, radians, degrees

# ====== CONFIG ======
st.set_page_config(page_title="NAVLOG v10 (AFM) — UI Otimizada", layout="wide", initial_sidebar_state="collapsed")

# ====== UTILS ======
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang   = lambda x: int(round(float(x))) % 360
rint   = lambda x: int(round(float(x)))
r10f   = lambda x: round(float(x), 1)

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

# ====== AFM (resumo Tecnam P2008) ======
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

isa_temp   = lambda pa: 15.0 - 2.0*(pa/1000.0)
press_alt  = lambda alt, qnh: float(alt) + (1013.0 - float(qnh)) * 30.0
clamp      = lambda v, lo, hi: max(lo, min(hi, v))

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
        if dev > 0:
            tas *= 1 - 0.02*(dev/15.0); ff *= 1 - 0.025*(dev/15.0)
        elif dev < 0:
            tas *= 1 + 0.01*((-dev)/15.0); ff *= 1 + 0.03*((-dev)/15.0)

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

# ====== STATE ======
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0)
ens("rpm_climb", 2250); ens("rpm_desc", 1800)
ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("legs", [])  # cada leg: {TC, Dist, Alt0, Alt1, Wfrom, Wkt, CK, HoldMin, HoldFF, RPMcru}
ens("computed", [])
ens("ck_default", 2)
ens("show_timeline", True)  # Ativado por default para melhor UX
ens("selected_leg", None)  # Para edição

# ====== ESTILO APRIMORADO (mais moderno, com cores e transições) ======
CSS = """
<style>
:root {
    --primary: #1f77b4; /* Azul principal */
    --secondary: #d62728; /* Vermelho para destaques */
    --bg: #f9fafb; /* Fundo claro */
    --card-bg: #ffffff; /* Fundo cartão */
    --line: #d1d5db; /* Bordas */
    --text: #111827; /* Texto principal */
    --muted: #6b7280; /* Texto muted */
    --chip: #f3f4f6; /* Chips */
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
body { background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial; }
.card { background: var(--card-bg); border: 1px solid var(--line); border-radius: 16px; padding: 20px; margin: 16px 0; box-shadow: var(--shadow); transition: box-shadow 0.3s; }
.card:hover { box-shadow: 0 6px 12px -2px rgba(0,0,0,0.15); }
.kvrow { display: flex; gap: 12px; flex-wrap: wrap; }
.kv { background: var(--chip); border: 1px solid var(--line); border-radius: 12px; padding: 8px 12px; font-size: 13px; font-weight: 500; }
.badge { background: var(--chip); border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; font-size: 12px; margin-left: 8px; }
.sep { height: 1px; background: var(--line); margin: 16px 0; }
.sticky { position: sticky; top: 0; background: #ffffffee; backdrop-filter: saturate(150%) blur(8px); z-index: 50; border-bottom: 1px solid var(--line); padding: 12px 0; }
.tl { position: relative; margin: 12px 0 24px 0; padding-bottom: 50px; }
.tl .bar { height: 8px; background: #e5e7eb; border-radius: 4px; }
.tl .tick { position: absolute; top: 12px; width: 2px; height: 16px; background: var(--muted); }
.tl .cp-lbl { position: absolute; top: 36px; transform: translateX(-50%); text-align: center; font-size: 12px; color: var(--text); white-space: nowrap; }
.tl .tocdot, .tl .toddot { position: absolute; top: -8px; width: 16px; height: 16px; border-radius: 50%; transform: translateX(-50%); border: 3px solid #fff; box-shadow: 0 0 0 2px rgba(0,0,0,0.2); transition: transform 0.3s; }
.tl .tocdot { background: var(--primary); }
.tl .toddot { background: var(--secondary); }
.tl .tocdot:hover, .tl .toddot:hover { transform: translateX(-50%) scale(1.2); }
.stButton button { border-radius: 8px; transition: background 0.3s; }
.stButton button:hover { background: var(--primary); color: white; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ====== TIMELINE APRIMORADA (com hover e mais info) ======
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
        title = f"{toc_tod['type']} em T+{mmss(toc_tod['t'])}"
        parts.append(f"<div class='{cls}' title='{title}' style='left:{pct:.2f}%;'></div>")
    html += ''.join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"GS {rint(seg['GS'])} kt · TAS {rint(seg['TAS'])} kt · FF {rint(seg['ff'])} L/h  |  {start_label} → {end_label}")

# ====== PERFIL LABEL ======
def phase_label(name):
    n = name.lower()
    if "climb" in n:   return "Climb"
    if "descent" in n: return "Descent"
    if "hold" in n:    return "Hold"
    return "Cruise/Level"

# ====== CÁLCULO (mantido, com validações extras) ======
def build_segments(tc, dist, alt0, alt1, wfrom, wkt, ck_min, params, rpm_cruise_leg, hold_min=0.0, hold_ff_input=0.0):
    qnh, oat, mag_var, mag_is_e = params['qnh'], params['oat'], params['mag_var'], params['mag_is_e']
    rpm_climb, rpm_desc, desc_angle, weight = params['rpm_climb'], params['rpm_desc'], params['desc_angle'], params['weight']

    # Validações
    if dist < 0: dist = 0
    if alt0 < 0: alt0 = 0
    if alt1 < 0: alt1 = 0
    tc = wrap360(tc)
    wfrom = wrap360(wfrom)

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

    ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))

    profile = "LEVEL" if abs(alt1 - alt0) < 1e-6 else ("CLIMB" if alt1 > alt0 else "DESCENT")
    segs = []
    toc_tod_marker = None

    if profile == "CLIMB":
        t_need = (alt1 - alt0) / max(ROC, 1e-6)
        d_need = GScl * (t_need / 60.0)
        if d_need <= dist:
            tA = rt10(t_need * 60)
            segs.append({"name":"Climb → TOC","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_climb,
                         "time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise (após TOC)","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_cru,
                             "time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            toc_tod_marker = {"type":"TOC","t": rt10(t_need*60)}
        else:
            tA = rt10((dist / max(GScl,1e-9)) * 3600)
            gained = ROC * (tA / 60.0)
            segs.append({"name":"Climb (não atinge)","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_climb,"ff":FF_climb,
                         "time":tA,"dist":dist,"alt0":alt0,"alt1":alt0+gained})
    elif profile == "DESCENT":
        t_need = (alt0 - alt1) / max(ROD, 1e-6)
        d_need = GSde * (t_need / 60.0)
        if d_need <= dist:
            tA = rt10(t_need * 60)
            segs.append({"name":"Descent → TOD","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_desc,
                         "time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem = dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr,1e-9)) * 3600)
                segs.append({"name":"Cruise (após TOD)","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_cru,
                             "time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            toc_tod_marker = {"type":"TOD","t": rt10(t_need*60)}
        else:
            tA = rt10((dist / max(GSde,1e-9)) * 3600)
            lost = ROD * (tA / 60.0)
            segs.append({"name":"Descent (não atinge)","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_desc,"ff":FF_desc,
                         "time":tA,"dist":dist,"alt0":alt0,"alt1":max(0.0, alt0 - lost)})
    else:
        tA = rt10((dist / max(GScr,1e-9)) * 3600)
        segs.append({"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_cru,"ff":FF_cru,
                     "time":tA,"dist":dist,"alt0":alt0,"alt1":alt0})

    hold_min = max(0.0, float(hold_min))
    if hold_min > 0.0:
        hold_ff = float(hold_ff_input)
        if hold_ff <= 0:
            _, hold_ff_auto = cruise_lookup(pa1, int(rpm_cruise_leg), oat, weight)
            hold_ff = hold_ff_auto
        hold_sec = rt10(hold_min * 60.0)
        end_alt = segs[-1]["alt1"] if segs else alt1
        segs.append({"name":"Hold/Espera","TH":segs[-1]["TH"] if segs else tc,"MH":segs[-1]["MH"] if segs else tc,
                     "GS":0.0,"TAS":0.0,"ff":hold_ff,"time":hold_sec,"dist":0.0,"alt0":end_alt,"alt1":end_alt})

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
        "roc_info": {"pa0": pa0, "roc": roc_interp(pa0, oat)},
        "rod_info": {"rod": max(100.0, (wind_triangle(tc, cruise_lookup(pa_avg, int(rpm_desc), oat, weight)[0], 0, 0)[2]) * 5.0 * (params['desc_angle']/3.0))},
        "toc_tod": toc_tod_marker,
        "ck_func": cps
    }

# ====== RECOMPUTE (com feedback de erros) ======
def recompute_all():
    st.session_state.computed = []
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
        except ValueError:
            st.error("Hora off-blocks inválida. Use formato HH:MM.")
            return

    carry_efob = float(st.session_state.start_efob)
    clock = base_time
    cum_sec = 0; cum_burn = 0.0

    for leg in st.session_state.legs:
        res = build_segments(
            tc=leg['TC'], dist=leg['Dist'], alt0=leg['Alt0'], alt1=leg['Alt1'],
            wfrom=leg['Wfrom'], wkt=leg['Wkt'], ck_min=leg['CK'], params=params,
            rpm_cruise_leg=leg['RPMcru'], hold_min=leg.get('HoldMin',0.0), hold_ff_input=leg.get('HoldFF',0.0)
        )

        segs = res["segments"]
        t_cursor = 0
        for idx_seg, seg in enumerate(segs):
            seg_efob_start = carry_efob
            seg_efob_end   = max(0.0, r10f(seg_efob_start - seg['burn']))
            if clock: 
                s_start = (clock + dt.timedelta(seconds=t_cursor)).strftime('%H:%M')
                s_end   = (clock + dt.timedelta(seconds=t_cursor + seg['time'])).strftime('%H:%M')
            else:
                s_start = f"T+{mmss(t_cursor)}"; s_end = f"T+{mmss(t_cursor + seg['time'])}"

            base_k = (clock + dt.timedelta(seconds=t_cursor)) if clock else None
            cps = res["ck_func"](seg, int(leg['CK']), base_k, seg_efob_start) if seg['GS']>0 else []

            st.session_state.computed.append({
                "phase_name": seg["name"],
                "phase_label": phase_label(seg["name"]),
                "TH": seg["TH"], "MH": seg["MH"],
                "GS": seg["GS"], "TAS": seg["TAS"],
                "ff": seg["ff"], "time": seg["time"], "dist": seg["dist"],
                "alt0": seg["alt0"], "alt1": seg["alt1"],
                "burn": r10f(seg["burn"]),
                "efob_start": seg_efob_start, "efob_end": seg_efob_end,
                "clock_start": s_start, "clock_end": s_end,
                "cps": cps,
                "toc_tod": (res["toc_tod"] if idx_seg==0 and ("Climb" in seg["name"] or "Descent" in seg["name"]) else None),
                "roc": res["roc_info"]["roc"], "rod": res["rod_info"]["rod"],
                "rpm_cruise_leg": leg["RPMcru"], "ck": leg["CK"],
                "leg_ref": leg
            })

            t_cursor += seg["time"]
            carry_efob = seg_efob_end

        if clock: clock = clock + dt.timedelta(seconds=sum(s['time'] for s in segs))
        cum_sec += sum(s['time'] for s in segs)
        cum_burn = r10f(cum_burn + sum(s['burn'] for s in segs))

# ====== CRUD APRIMORADO ======
def add_leg():
    d = dict(TC=0.0, Dist=0.0, Alt0=0.0, Alt1=0.0, Wfrom=0, Wkt=0,
             CK=int(st.session_state.ck_default),
             HoldMin=0.0, HoldFF=0.0,
             RPMcru=2100)
    st.session_state.legs.append(d)
    st.session_state.selected_leg = len(st.session_state.legs) - 1
    recompute_all()

def update_leg(idx, **vals):
    st.session_state.legs[idx].update(vals)
    recompute_all()

def delete_leg(idx):
    del st.session_state.legs[idx]
    if st.session_state.selected_leg == idx:
        st.session_state.selected_leg = None
    recompute_all()

# ====== HEADER STICKY (com toggle e botões) ======
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
h1, h2, h3 = st.columns([4, 2, 2])
with h1:
    st.title("NAVLOG v10 — AFM Otimizado")
with h2:
    st.toggle("Mostrar Timeline/CPs", key="show_timeline")
with h3:
    if st.button("➕ Nova Leg", type="primary", use_container_width=True):
        add_leg()
st.markdown("</div>", unsafe_allow_html=True)

# ====== TABS PRINCIPAIS: Parâmetros | Legs | Resultados ======
tab1, tab2, tab3 = st.tabs(["⚙️ Parâmetros Globais", "🛫 Legs", "📊 Resultados"])

with tab1:
    with st.form("params_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("QNH (hPa)", 900, 1050, key="qnh", help="Pressão altimétrica")
            st.number_input("OAT (°C)", -40, 50, key="oat", help="Temperatura exterior")
            st.number_input("Peso (kg)", 450.0, 700.0, key="weight", step=1.0, help="Peso total")
        with col2:
            st.number_input("EFOB Inicial (L)", 0.0, 200.0, key="start_efob", step=0.5, help="Estimated Fuel On Board inicial")
            st.text_input("Hora Off-Blocks (HH:MM)", key="start_clock", help="Hora de partida")
            st.number_input("Ângulo de Descida (°)", 1.0, 6.0, key="desc_angle", step=0.1)
        with col3:
            st.number_input("Climb RPM (Global)", 1800, 2265, key="rpm_climb", step=5)
            st.number_input("Descent RPM (Global)", 1600, 2265, key="rpm_desc", step=5)
            st.number_input("Mag Var (°)", 0, 30, key="mag_var")
            st.selectbox("Var E/W", ["W", "E"], key="mag_is_e", index=1 if st.session_state.mag_is_e else 0)
        if st.form_submit_button("Aplicar Parâmetros"):
            recompute_all()
            st.success("Parâmetros aplicados com sucesso!")

with tab2:
    if not st.session_state.legs:
        st.info("Adicione uma nova leg usando o botão acima para começar.")
    else:
        # Lista de legs à esquerda, editor à direita
        left, right = st.columns([1, 3])
        with left:
            st.subheader("Lista de Legs")
            for i, leg in enumerate(st.session_state.legs):
                if st.button(f"Leg {i+1}: TC {leg['TC']}° / Dist {leg['Dist']} nm", key=f"select_leg_{i}", use_container_width=True):
                    st.session_state.selected_leg = i
            if st.button("🗑️ Limpar Todas as Legs", use_container_width=True):
                st.session_state.legs = []
                st.session_state.selected_leg = None
                recompute_all()

        with right:
            if st.session_state.selected_leg is not None:
                i = st.session_state.selected_leg
                leg = st.session_state.legs[i]
                st.subheader(f"Editando Leg {i+1}")
                with st.form(f"leg_form_{i}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        TC = st.number_input("True Course (°T)", 0.0, 359.9, float(leg['TC']), step=0.1)
                        Dist = st.number_input("Distância (nm)", 0.0, 500.0, float(leg['Dist']), step=0.1)
                        Alt0 = st.number_input("Altitude INI (ft)", 0.0, 30000.0, float(leg['Alt0']), step=50.0)
                        Alt1 = st.number_input("Altitude DEST (ft)", 0.0, 30000.0, float(leg['Alt1']), step=50.0)
                    with c2:
                        Wfrom = st.number_input("Vento FROM (°T)", 0, 360, int(leg['Wfrom']), step=1)
                        Wkt = st.number_input("Vento (kt)", 0, 150, int(leg['Wkt']), step=1)
                        CK = st.number_input("Checkpoints (min)", 1, 10, int(leg['CK']), step=1)
                        RPMcru = st.number_input("Cruise RPM (leg)", 1800, 2265, int(leg['RPMcru']), step=5)
                    with c3:
                        HoldMin = st.number_input("Espera (min)", 0.0, 180.0, float(leg.get('HoldMin',0.0)), step=0.5)
                        HoldFF = st.number_input("FF Espera (L/h) (0=auto)", 0.0, 60.0, float(leg.get('HoldFF',0.0)), step=0.1)
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.form_submit_button("Salvar Alterações"):
                            update_leg(i, TC=TC, Dist=Dist, Alt0=Alt0, Alt1=Alt1, Wfrom=Wfrom, Wkt=Wkt,
                                       CK=CK, RPMcru=RPMcru, HoldMin=HoldMin, HoldFF=HoldFF)
                            st.success("Leg atualizada!")
                    with b2:
                        if st.button("Apagar esta Leg"):
                            delete_leg(i)
                            st.success("Leg apagada!")

with tab3:
    if not st.session_state.computed:
        st.info("Configure parâmetros e adicione legs para ver os resultados.")
    else:
        # Resumo global
        total_sec = sum(c["time"] for c in st.session_state.computed)
        total_burn = r10f(sum(c["burn"] for c in st.session_state.computed))
        efob_end = st.session_state.computed[-1]["efob_end"]
        st.markdown(
            "<div class='kvrow'>"
            + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
            + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b></div>"
            + f"<div class='kv'>🧯 EFOB Final: <b>{efob_end:.1f} L</b></div>"
            + "</div>", unsafe_allow_html=True
        )
        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

        # Cartões de fase
        for idx, c in enumerate(st.session_state.computed):
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                left, right = st.columns([3, 1])
                with left:
                    st.subheader(f"Fase {idx+1}: {c['phase_label']}")
                    st.caption(c["phase_name"])
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
                    st.metric("Burn (L)", f"{c['burn']:.1f}")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f"**Relógio:** {c['clock_start']} → {c['clock_end']}")
                with r2:
                    st.markdown(f"**EFOB:** {c['efob_start']:.1f} L → {c['efob_end']:.1f} L")
                with r3:
                    if "Climb" in c["phase_name"]:
                        st.markdown(f"**ROC:** {rint(c['roc'])} ft/min")
                    elif "Descent" in c["phase_name"]:
                        st.markdown(f"**ROD:** {rint(c['rod'])} ft/min")
                    else:
                        st.markdown(f"**RPM Cruise:** {int(c['rpm_cruise_leg'])}")

                if c["toc_tod"] is not None:
                    st.info(f"{c['toc_tod']['type']} em T+{mmss(c['toc_tod']['t'])}")

                if st.session_state.show_timeline and c["GS"] > 0:
                    timeline(
                        {"GS":c["GS"],"TAS":c["TAS"],"ff":c["ff"],"time":c["time"]},
                        c["cps"],
                        c["clock_start"], c["clock_end"],
                        toc_tod=c["toc_tod"]
                    )

                warns = []
                if c["dist"] == 0 and abs(c["alt1"]-c["alt0"])>50: warns.append("Distância 0 com variação de altitude.")
                if "não atinge" in c["phase_name"]: warns.append("Perfil não atinge altitude-alvo.")
                if c["efob_end"] <= 0: warns.append("EFOB final ≤ 0.")
                if warns: st.warning(" | ".join(warns))

                st.markdown("</div>", unsafe_allow_html=True)
