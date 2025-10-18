import streamlit as st
import datetime as dt
import math
from math import sin, asin, radians, degrees
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from uuid import uuid4
import csv
import io

# ====== CONFIGURAÇÃO GERAL ======
st.set_page_config(page_title="NAVLOG v10 (AFM) — UI Focado", layout="wide", initial_sidebar_state="collapsed")

# ====== TIPOS E CONSTANTES ======
@dataclass
class Leg:
    TC: float
    Dist: float
    Alt0: float
    Alt1: float
    Wfrom: int
    Wkt: int
    CK: int
    HoldMin: float
    HoldFF: float
    RPMcru: int
    id: str = ""  # UUID para identificação única

@dataclass
class Segment:
    name: str
    TH: float
    MH: float
    GS: float
    TAS: float
    ff: float
    time: int
    dist: float
    alt0: float
    alt1: float
    burn: float

@dataclass
class ComputedCard:
    phase_name: str
    phase_label: str
    TH: float
    MH: float
    GS: float
    TAS: float
    ff: float
    time: int
    dist: float
    alt0: float
    alt1: float
    burn: float
    efob_start: float
    efob_end: float
    clock_start: str
    clock_end: str
    cps: List[Dict]
    toc_tod: Optional[Dict]
    roc: float
    rod: float
    rpm_cruise_leg: int
    ck: int
    leg_ref: Leg

# Dados de performance (Tecnam P2008)
ROC_ENR = {
    0: {-25: 981, 0: 835, 25: 704, 50: 586}, 2000: {-25: 870, 0: 726, 25: 597, 50: 481},
    4000: {-25: 759, 0: 617, 25: 491, 50: 377}, 6000: {-25: 648, 0: 509, 25: 385, 50: 273},
    8000: {-25: 538, 0: 401, 25: 279, 50: 170}, 10000: {-25: 428, 0: 294, 25: 174, 50: 66},
    12000: {-25: 319, 0: 187, 25: 69, 50: -37}, 14000: {-25: 210, 0: 80, 25: -35, 50: -139}
}
VY = {0: 67, 2000: 67, 4000: 67, 6000: 67, 8000: 67, 10000: 67, 12000: 67, 14000: 67}
ROC_FACTOR = 0.90
CRUISE = {
    0: {1800: (82, 15.3), 1900: (89, 17.0), 2000: (95, 18.7), 2100: (101, 20.7), 2250: (110, 24.6), 2388: (118, 27.7)},
    2000: {1800: (81, 15.5), 1900: (87, 17.0), 2000: (93, 18.8), 2100: (99, 20.9), 2250: (108, 25.0)},
    4000: {1800: (79, 15.2), 1900: (86, 16.5), 2000: (92, 18.1), 2100: (98, 19.2), 2250: (106, 23.9)},
    6000: {1800: (78, 14.9), 1900: (85, 16.1), 2000: (91, 17.5), 2100: (97, 19.2), 2250: (105, 22.7)},
    8000: {1800: (78, 14.9), 1900: (84, 15.7), 2000: (90, 17.0), 2100: (96, 18.5), 2250: (104, 21.5)},
    10000: {1800: (78, 15.5), 1900: (82, 15.5), 2000: (89, 16.6), 2100: (95, 17.9), 2250: (103, 20.5)},
}

# ====== UTILITÁRIOS ======
def rt10(s: float) -> int:
    return max(10, int(round(s / 10.0) * 10)) if s > 0 else 0

def mmss(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"

def hhmmss(t: int) -> str:
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

def rang(x: float) -> int:
    return int(round(float(x))) % 360

def r10f(x: float) -> float:
    return round(float(x), 1)

def wrap360(x: float) -> float:
    x = math.fmod(float(x), 360.0)
    return x + 360 if x < 0 else x

def angdiff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180

def wind_triangle(tc: float, tas: float, wdir: int, wkt: int) -> Tuple[float, float, float]:
    if tas <= 0:
        return 0.0, wrap360(tc), 0.0
    d = radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = degrees(asin(s))
    th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th: float, var: int, east_is_neg: bool = False) -> float:
    return wrap360(th - var if east_is_neg else th + var)

def isa_temp(pa: float) -> float:
    return 15.0 - 2.0 * (pa / 1000.0)

def press_alt(alt: float, qnh: int) -> float:
    return float(alt) + (1013.0 - float(qnh)) * 30.0

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def interp1(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def cruise_lookup(pa: float, rpm: int, oat: Optional[float], weight: float) -> Tuple[float, float]:
    rpm = min(rpm, 2265)
    pas = sorted(CRUISE.keys())
    pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c])
    p1 = min([p for p in pas if p >= pa_c])
    table0 = CRUISE[p0]
    table1 = CRUISE[p1]

    def v(tab: Dict) -> Tuple[float, float]:
        rpms = sorted(tab.keys())
        if rpm in tab:
            return tab[rpm]
        if rpm < rpms[0]:
            lo, hi = rpms[0], rpms[1]
        elif rpm > rpms[-1]:
            lo, hi = rpms[-2], rpms[-1]
        else:
            lo = max([r for r in rpms if r <= rpm])
            hi = min([r for r in rpms if r >= rpm])
        (tas_lo, ff_lo), (tas_hi, ff_hi) = tab[lo], tab[hi]
        t = (rpm - lo) / (hi - lo) if hi != lo else 0
        return (tas_lo + t * (tas_hi - tas_lo), ff_lo + t * (ff_hi - ff_lo))

    tas0, ff0 = v(table0)
    tas1, ff1 = v(table1)
    tas = interp1(pa_c, p0, p1, tas0, tas1)
    ff = interp1(pa_c, p0, p1, ff0, ff1)

    if oat is not None:
        dev = oat - isa_temp(pa_c)
        if dev > 0:
            tas *= 1 - 0.02 * (dev / 15.0)
            ff *= 1 - 0.025 * (dev / 15.0)
        elif dev < 0:
            tas *= 1 + 0.01 * ((-dev) / 15.0)
            ff *= 1 + 0.03 * ((-dev) / 15.0)

    tas *= (1.0 + 0.033 * ((650.0 - float(weight)) / 100.0))
    return max(0.0, tas), max(0.0, ff)

def roc_interp(pa: float, temp: float) -> float:
    pas = sorted(ROC_ENR.keys())
    pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c])
    p1 = min([p for p in pas if p >= pa_c])
    temps = [-25, 0, 25, 50]
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

def vy_interp(pa: float) -> float:
    pas = sorted(VY.keys())
    pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c])
    p1 = min([p for p in pas if p >= pa_c])
    return interp1(pa_c, p0, p1, VY[p0], VY[p1])

# ====== ESTADO INICIAL ======
def init_session_state():
    defaults = {
        "qnh": 1013,
        "oat": 15,
        "mag_var": 1,
        "mag_is_e": False,
        "weight": 650.0,
        "rpm_climb": 2250,
        "rpm_desc": 1800,
        "desc_angle": 3.0,
        "start_clock": "",
        "start_efob": 85.0,
        "legs": [],
        "computed": [],
        "ck_default": 2,
        "show_timeline": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session_state()

# ====== ESTILO CSS ======
CSS = """
<style>
:root {
    --line: #e5e7eb;
    --chip: #f3f4f6;
    --muted: #6b7280;
    --primary: #1f77b4;
    --danger: #d62728;
}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card {border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin: 12px 0; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.05)}
.kvrow {display: flex; gap: 8px; flex-wrap: wrap}
.kv {background: var(--chip); border: 1px solid var(--line); border-radius: 10px; padding: 6px 10px; font-size: 13px}
.badge {background: var(--chip); border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; font-size: 12px; margin-left: 8px}
.sep {height: 1px; background: var(--line); margin: 12px 0}
.sticky {position: sticky; top: 0; background: #ffffffee; backdrop-filter: saturate(150%) blur(4px); z-index: 50; border-bottom: 1px solid var(--line); padding: 12px 0}
.tl {position: relative; margin: 10px 0 20px 0; padding-bottom: 50px}
.tl .bar {height: 8px; background: #eef1f5; border-radius: 4px}
.tl .tick {position: absolute; top: 12px; width: 2px; height: 16px; background: #333}
.tl .cp-lbl {position: absolute; top: 34px; transform: translateX(-50%); text-align: center; font-size: 12px; color: #333; white-space: nowrap}
.tl .tocdot, .tl .toddot {position: absolute; top: -6px; width: 16px; height: 16px; border-radius: 50%; transform: translateX(-50%); border: 2px solid #fff; box-shadow: 0 0 0 2px rgba(0,0,0,0.15)}
.tl .tocdot {background: var(--primary)}
.tl .toddot {background: var(--danger)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ====== TIMELINE ======
def timeline(seg: Segment, cps: List[Dict], start_label: str, end_label: str, toc_tod: Optional[Dict] = None):
    total = max(1, seg.time)
    html = "<div class='tl'><div class='bar'></div>"
    parts = []
    for cp in cps:
        pct = (cp['t'] / total) * 100.0
        parts.append(f"<div class='tick' style='left:{pct:.2f}%;'></div>")
        lbl = (
            f"<div class='cp-lbl' style='left:{pct:.2f}%;'>"
            f"<div>T+{cp['min']}m</div><div>{cp['nm']} nm</div>"
            f"<div>{cp['eto']}</div><div>EFOB {cp['efob']:.1f}</div></div>"
        ) if cp['eto'] else (
            f"<div class='cp-lbl' style='left:{pct:.2f}%;'>"
            f"<div>T+{cp['min']}m</div><div>{cp['nm']} nm</div><div>EFOB {cp['efob']:.1f}</div></div>"
        )
        parts.append(lbl)
    if toc_tod and 0 < toc_tod['t'] < total:
        pct = (toc_tod['t'] / total) * 100.0
        cls = 'tocdot' if toc_tod['type'] == 'TOC' else 'toddot'
        parts.append(f"<div class='{cls}' title='{toc_tod['type']}' style='left:{pct:.2f}%;'></div>")
    html += ''.join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"GS {rint(seg.GS)} kt · TAS {rint(seg.TAS)} kt · FF {rint(seg.ff)} L/h  |  {start_label} → {end_label}")

# ====== CÁLCULOS ======
def phase_label(name: str) -> str:
    n = name.lower()
    if "climb" in n:
        return "Climb"
    if "descent" in n:
        return "Descent"
    if "hold" in n:
        return "Hold"
    return "Cruise/Level"

def build_segments(
    leg: Leg,
    params: Dict,
) -> Dict:
    qnh = params['qnh']
    oat = params['oat']
    mag_var = params['mag_var']
    mag_is_e = params['mag_is_e']
    rpm_climb = params['rpm_climb']
    rpm_desc = params['rpm_desc']
    desc_angle = params['desc_angle']
    weight = params['weight']

    pa0 = press_alt(leg.Alt0, qnh)
    pa1 = press_alt(leg.Alt1, qnh)
    pa_avg = (pa0 + pa1) / 2.0
    Vy = vy_interp(pa0)
    ROC = roc_interp(pa0, oat)
    TAS_climb = Vy
    FF_climb = cruise_lookup((pa0 + pa1) / 2.0, rpm_climb, oat, weight)[1]
    TAS_cru, FF_cru = cruise_lookup(pa1, leg.RPMcru, oat, weight)
    TAS_desc, FF_desc = cruise_lookup(pa_avg, rpm_desc, oat, weight)

    _, THc, GScl = wind_triangle(leg.TC, TAS_climb, leg.Wfrom, leg.Wkt)
    _, THr, GScr = wind_triangle(leg.TC, TAS_cru, leg.Wfrom, leg.Wkt)
    _, THd, GSde = wind_triangle(leg.TC, TAS_desc, leg.Wfrom, leg.Wkt)

    MHc = apply_var(THc, mag_var, mag_is_e)
    MHr = apply_var(THr, mag_var, mag_is_e)
    MHd = apply_var(THd, mag_var, mag_is_e)

    ROD = max(100.0, GSde * 5.0 * (desc_angle / 3.0))

    profile = "LEVEL" if abs(leg.Alt1 - leg.Alt0) < 1e-6 else ("CLIMB" if leg.Alt1 > leg.Alt0 else "DESCENT")
    segs = []
    toc_tod_marker = None

    if profile == "CLIMB":
        t_need = (leg.Alt1 - leg.Alt0) / max(ROC, 1e-6)
        d_need = GScl * (t_need / 60.0)
        if d_need <= leg.Dist:
            segs.append(Segment(
                name="Climb → TOC", TH=THc, MH=MHc, GS=GScl, TAS=TAS_climb, ff=FF_climb,
                time=rt10(t_need * 60), dist=d_need, alt0=leg.Alt0, alt1=leg.Alt1, burn=0.0
            ))
            rem = leg.Dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr, 1e-9)) * 3600)
                segs.append(Segment(
                    name="Cruise (após TOC)", TH=THr, MH=MHr, GS=GScr, TAS=TAS_cru, ff=FF_cru,
                    time=tB, dist=rem, alt0=leg.Alt1, alt1=leg.Alt1, burn=0.0
                ))
            toc_tod_marker = {"type": "TOC", "t": rt10(t_need * 60)}
        else:
            tA = rt10((leg.Dist / max(GScl, 1e-9)) * 3600)
            gained = ROC * (tA / 60.0)
            segs.append(Segment(
                name="Climb (não atinge)", TH=THc, MH=MHc, GS=GScl, TAS=TAS_climb, ff=FF_climb,
                time=tA, dist=leg.Dist, alt0=leg.Alt0, alt1=leg.Alt0 + gained, burn=0.0
            ))
    elif profile == "DESCENT":
        t_need = (leg.Alt0 - leg.Alt1) / max(ROD, 1e-6)
        d_need = GSde * (t_need / 60.0)
        if d_need <= leg.Dist:
            segs.append(Segment(
                name="Descent → TOD", TH=THd, MH=MHd, GS=GSde, TAS=TAS_desc, ff=FF_desc,
                time=rt10(t_need * 60), dist=d_need, alt0=leg.Alt0, alt1=leg.Alt1, burn=0.0
            ))
            rem = leg.Dist - d_need
            if rem > 0:
                tB = rt10((rem / max(GScr, 1e-9)) * 3600)
                segs.append(Segment(
                    name="Cruise (após TOD)", TH=THr, MH=MHr, GS=GScr, TAS=TAS_cru, ff=FF_cru,
                    time=tB, dist=rem, alt0=leg.Alt1, alt1=leg.Alt1, burn=0.0
                ))
            toc_tod_marker = {"type": "TOD", "t": rt10(t_need * 60)}
        else:
            tA = rt10((leg.Dist / max(GSde, 1e-9)) * 3600)
            lost = ROD * (tA / 60.0)
            segs.append(Segment(
                name="Descent (não atinge)", TH=THd, MH=MHd, GS=GSde, TAS=TAS_desc, ff=FF_desc,
                time=tA, dist=leg.Dist, alt0=leg.Alt0, alt1=max(0.0, leg.Alt0 - lost), burn=0.0
            ))
    else:
        tA = rt10((leg.Dist / max(GScr, 1e-9)) * 3600)
        segs.append(Segment(
            name="Cruise", TH=THr, MH=MHr, GS=GScr, TAS=TAS_cru, ff=FF_cru,
            time=tA, dist=leg.Dist, alt0=leg.Alt0, alt1=leg.Alt0, burn=0.0
        ))

    hold_min = max(0.0, leg.HoldMin)
    if hold_min > 0.0:
        hold_ff = leg.HoldFF
        if hold_ff <= 0:
            _, hold_ff = cruise_lookup(pa1, leg.RPMcru, oat, weight)
        hold_sec = rt10(hold_min * 60.0)
        end_alt = segs[-1].alt1 if segs else leg.Alt1
        segs.append(Segment(
            name="Hold/Espera", TH=segs[-1].TH if segs else leg.TC, MH=segs[-1].MH if segs else leg.TC,
            GS=0.0, TAS=0.0, ff=hold_ff, time=hold_sec, dist=0.0, alt0=end_alt, alt1=end_alt, burn=0.0
        ))

    for s in segs:
        s.burn = r10f(s.ff * (s.time / 3600.0))

    tot_sec = sum(s.time for s in segs)
    tot_burn = r10f(sum(s.burn for s in segs))

    def cps(seg: Segment, every_min: int, base_clk: Optional[dt.datetime], efob_start: float) -> List[Dict]:
        out = []
        if every_min <= 0 or seg.GS <= 0:
            return out
        t = 0
        while t + every_min * 60 <= seg.time:
            t += every_min * 60
            d = seg.GS * (t / 3600.0)
            burn = seg.ff * (t / 3600.0)
            eto = (base_clk + dt.timedelta(seconds=t)).strftime('%H:%M') if base_clk else ""
            efob = max(0.0, r10f(efob_start - burn))
            out.append({"t": t, "min": int(t / 60), "nm": round(d, 1), "eto": eto, "efob": efob})
        return out

    return {
        "segments": segs,
        "tot_sec": tot_sec,
        "tot_burn": tot_burn,
        "roc_info": {"pa0": pa0, "roc": roc_interp(pa0, oat)},
        "rod_info": {"rod": max(100.0, (wind_triangle(leg.TC, cruise_lookup(pa_avg, rpm_desc, oat, weight)[0], 0, 0)[2]) * 5.0 * (desc_angle / 3.0))},
        "toc_tod": toc_tod_marker,
        "ck_func": cps
    }

def recompute_all():
    st.session_state.computed = []
    params = {
        "qnh": st.session_state.qnh,
        "oat": st.session_state.oat,
        "mag_var": st.session_state.mag_var,
        "mag_is_e": st.session_state.mag_is_e,
        "rpm_climb": st.session_state.rpm_climb,
        "rpm_desc": st.session_state.rpm_desc,
        "desc_angle": st.session_state.desc_angle,
        "weight": st.session_state.weight
    }

    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h, m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h, m))
        except ValueError:
            st.error("Formato de hora inválido (use HH:MM).")
            return

    carry_efob = float(st.session_state.start_efob)
    clock = base_time
    cum_sec = 0

    for leg in st.session_state.legs:
        res = build_segments(leg, params)

        segs = res["segments"]
        t_cursor = 0
        for idx_seg, seg in enumerate(segs):
            seg_efob_start = carry_efob
            seg_efob_end = max(0.0, r10f(seg_efob_start - seg.burn))
            s_start = (clock + dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if clock else f"T+{mmss(t_cursor)}"
            s_end = (clock + dt.timedelta(seconds=t_cursor + seg.time)).strftime('%H:%M') if clock else f"T+{mmss(t_cursor + seg.time)}"

            base_k = (clock + dt.timedelta(seconds=t_cursor)) if clock else None
            cps = res["ck_func"](seg, leg.CK, base_k, seg_efob_start) if seg.GS > 0 else []

            st.session_state.computed.append(ComputedCard(
                phase_name=seg.name,
                phase_label=phase_label(seg.name),
                TH=seg.TH, MH=seg.MH, GS=seg.GS, TAS=seg.TAS, ff=seg.ff,
                time=seg.time, dist=seg.dist, alt0=seg.alt0, alt1=seg.alt1, burn=seg.burn,
                efob_start=seg_efob_start, efob_end=seg_efob_end,
                clock_start=s_start, clock_end=s_end, cps=cps,
                toc_tod=(res["toc_tod"] if idx_seg == 0 and ("Climb" in seg.name or "Descent" in seg.name) else None),
                roc=res["roc_info"]["roc"], rod=res["rod_info"]["rod"],
                rpm_cruise_leg=leg.RPMcru, ck=leg.CK, leg_ref=leg
            ))

            t_cursor += seg.time
            carry_efob = seg_efob_end

        if clock:
            clock += dt.timedelta(seconds=sum(s.time for s in segs))
        cum_sec += sum(s.time for s in segs)

# ====== CRUD ======
def add_leg():
    leg = Leg(
        TC=0.0, Dist=0.0, Alt0=0.0, Alt1=0.0, Wfrom=0, Wkt=0,
        CK=st.session_state.ck_default, HoldMin=0.0, HoldFF=0.0, RPMcru=2100,
        id=str(uuid4())
    )
    st.session_state.legs.append(leg)
    recompute_all()

def duplicate_leg(leg: Leg):
    new_leg = Leg(
        TC=leg.TC, Dist=leg.Dist, Alt0=leg.Alt0, Alt1=leg.Alt1,
        Wfrom=leg.Wfrom, Wkt=leg.Wkt, CK=leg.CK,
        HoldMin=leg.HoldMin, HoldFF=leg.HoldFF, RPMcru=leg.RPMcru,
        id=str(uuid4())
    )
    st.session_state.legs.append(new_leg)
    recompute_all()

def update_leg(leg: Leg, **vals):
    for k, v in vals.items():
        setattr(leg, k, v)
    recompute_all()

def delete_leg(leg_id: str):
    st.session_state.legs = [leg for leg in st.session_state.legs if leg.id != leg_id]
    recompute_all()

# ====== EXPORTAÇÃO ======
def export_to_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Phase", "Start Alt (ft)", "End Alt (ft)", "TH (°T)", "MH (°M)", "GS (kt)", "TAS (kt)", "FF (L/h)", "Time (s)", "Dist (nm)", "Burn (L)", "EFOB Start (L)", "EFOB End (L)", "Clock Start", "Clock End"])
    for c in st.session_state.computed:
        writer.writerow([
            c.phase_name, c.alt0, c.alt1, c.TH, c.MH, c.GS, c.TAS, c.ff, c.time, c.dist,
            c.burn, c.efob_start, c.efob_end, c.clock_start, c.clock_end
        ])
    return output.getvalue()

# ====== INTERFACE ======
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
with c1:
    st.title("NAVLOG v10 (AFM)")
with c2:
    st.toggle("Mostrar Timeline/CPs", key="show_timeline")
with c3:
    if st.button("➕ Nova Leg", type="primary", use_container_width=True):
        add_leg()
with c4:
    if st.button("🗑️ Limpar Legs", use_container_width=True) and st.session_state.legs:
        st.session_state.legs = []
        st.session_state.computed = []
with c5:
    if st.session_state.computed:
        csv_data = export_to_csv()
        st.download_button("📥 Exportar CSV", csv_data, "navlog_export.csv", "text/csv", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.form("params"):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, st.session_state.qnh, step=1)
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, st.session_state.oat, step=1)
    with p2:
        st.session_state.start_efob = st.number_input("EFOB Inicial (L)", 0.0, 200.0, st.session_state.start_efob, step=0.5)
        st.session_state.start_clock = st.text_input("Hora Off-Blocks (HH:MM)", st.session_state.start_clock, help="Formato: HH:MM")
    with p3:
        st.session_state.weight = st.number_input("Peso (kg)", 450.0, 700.0, st.session_state.weight, step=1.0)
        st.session_state.desc_angle = st.number_input("Ângulo de Descida (°)", 1.0, 6.0, st.session_state.desc_angle, step=0.1)
    with p4:
        st.session_state.rpm_climb = st.number_input("Climb RPM (global)", 1800, 2265, st.session_state.rpm_climb, step=5)
        st.session_state.rpm_desc = st.number_input("Descent RPM (global)", 1600, 2265, st.session_state.rpm_desc, step=5)
    with st.expander("Magnético / Avançado"):
        a1, a2 = st.columns(2)
        with a1:
            st.session_state.mag_var = st.number_input("Mag Var (°)", 0, 30, st.session_state.mag_var, step=1)
        with a2:
            st.session_state.mag_is_e = st.selectbox("Var E/W", ["W", "E"], index=(1 if st.session_state.mag_is_e else 0)) == "E"
    if st.form_submit_button("Aplicar Parâmetros"):
        recompute_all()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

if not st.session_state.legs:
    st.info("Clique em **Nova Leg** para começar. Preencha os campos e veja os cartões de fase gerados.")
else:
    for i, leg in enumerate(st.session_state.legs):
        with st.expander(f"Leg {i+1} — Inputs", expanded=True):
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                TC = st.number_input(f"True Course (°T) — L{i+1}", 0.0, 359.9, leg.TC, step=0.1, key=f"TC_{leg.id}")
                Dist = st.number_input(f"Distância (nm) — L{i+1}", 0.0, 500.0, leg.Dist, step=0.1, key=f"Dist_{leg.id}")
            with i2:
                Alt0 = st.number_input(f"Altitude INI (ft) — L{i+1}", 0.0, 30000.0, leg.Alt0, step=50.0, key=f"Alt0_{leg.id}")
                Alt1 = st.number_input(f"Altitude DEST (ft) — L{i+1}", 0.0, 30000.0, leg.Alt1, step=50.0, key=f"Alt1_{leg.id}")
            with i3:
                Wfrom = st.number_input(f"Vento FROM (°T) — L{i+1}", 0, 360, leg.Wfrom, step=1, key=f"Wfrom_{leg.id}")
                Wkt = st.number_input(f"Vento (kt) — L{i+1}", 0, 150, leg.Wkt, step=1, key=f"Wkt_{leg.id}")
            with i4:
                CK = st.number_input(f"Checkpoints (min) — L{i+1}", 1, 10, leg.CK, step=1, key=f"CK_{leg.id}")
                RPMcru = st.number_input(f"Cruise RPM (leg) — L{i+1}", 1800, 2265, leg.RPMcru, step=5, key=f"RPMcru_{leg.id}")

            j1, j2, j3 = st.columns([1.2, 1.2, 6])
            with j1:
                HoldMin = st.number_input(f"Espera (min) — L{i+1}", 0.0, 180.0, leg.HoldMin, step=0.5, key=f"HoldMin_{leg.id}")
            with j2:
                HoldFF = st.number_input(f"FF Espera (L/h) — L{i+1} (0=auto)", 0.0, 60.0, leg.HoldFF, step=0.1, key=f"HoldFF_{leg.id}")
            with j3:
                st.caption("Cruise RPM é por leg. Climb/Descent RPM são globais. 'Espera' vira um cartão de fase próprio.")

            b1, b2, b3, _ = st.columns([1, 1, 1, 7])
            with b1:
                if st.button("Guardar", key=f"save_{leg.id}", use_container_width=True):
                    update_leg(leg, TC=TC, Dist=Dist, Alt0=Alt0, Alt1=Alt1, Wfrom=Wfrom, Wkt=Wkt, CK=CK, RPMcru=RPMcru, HoldMin=HoldMin, HoldFF=HoldFF)
            with b2:
                if st.button("Apagar", key=f"del_{leg.id}", use_container_width=True):
                    delete_leg(leg.id)
                    st.experimental_rerun()
            with b3:
                if st.button("Duplicar", key=f"dup_{leg.id}", use_container_width=True):
                    duplicate_leg(leg)

    recompute_all()

if st.session_state.computed:
    total_sec = sum(c.time for c in st.session_state.computed)
    total_burn = r10f(sum(c.burn for c in st.session_state.computed))
    efob_end = st.session_state.computed[-1].efob_end
    st.markdown(
        "<div class='kvrow'>"
        f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
        f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b></div>"
        f"<div class='kv'>🧯 EFOB Final: <b>{efob_end:.1f} L</b></div>"
        "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    for idx, c in enumerate(st.session_state.computed):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            st.subheader(f"Fase {idx+1}: {c.phase_label}")
            st.caption(c.phase_name)
            st.markdown(
                "<div class='kvrow'>"
                f"<div class='kv'>Alt: <b>{int(round(c.alt0))}→{int(round(c.alt1))} ft</b></div>"
                f"<div class='kv'>TH/MH: <b>{rang(c.TH)}T / {rang(c.MH)}M</b></div>"
                f"<div class='kv'>GS/TAS: <b>{rint(c.GS)}/{rint(c.TAS)} kt</b></div>"
                f"<div class='kv'>FF: <b>{rint(c.ff)} L/h</b></div>"
                "</div>", unsafe_allow_html=True
            )
        with right:
            st.metric("Tempo", mmss(c.time))
            st.metric("Fuel desta fase (L)", f"{c.burn:.1f}")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"**Relógio** — {c.clock_start} → {c.clock_end}")
        with r2:
            st.markdown(f"**EFOB** — Start {c.efob_start:.1f} L → End {c.efob_end:.1f} L")
        with r3:
            if "Climb" in c.phase_name:
                st.markdown(f"**ROC ref.** — {rint(c.roc)} ft/min")
            elif "Descent" in c.phase_name:
                st.markdown(f"**ROD ref.** — {rint(c.rod)} ft/min")
            else:
                st.markdown(f"**Cruise RPM (leg)** — {c.rpm_cruise_leg} RPM")

        if c.toc_tod:
            st.info(f"Marcador: **{c.toc_tod['type']}** em T+{mmss(c.toc_tod['t'])}")

        if st.session_state.show_timeline and c.GS > 0:
            timeline(
                Segment(name=c.phase_name, TH=c.TH, MH=c.MH, GS=c.GS, TAS=c.TAS, ff=c.ff,
                        time=c.time, dist=c.dist, alt0=c.alt0, alt1=c.alt1, burn=c.burn),
                c.cps, c.clock_start, c.clock_end, c.toc_tod
            )

        warns = []
        if c.dist == 0 and abs(c.alt1 - c.alt0) > 50:
            warns.append("Distância 0 com variação de altitude.")
        if "não atinge" in c.phase_name:
            warns.append("Perfil não atinge a altitude-alvo nesta fase.")
        if c.efob_end <= 0:
            warns.append("EFOB no fim desta fase é 0 (ou negativo).")
        if warns:
            st.warning(" | ".join(warns))

        st.markdown("</div>", unsafe_allow_html=True)

