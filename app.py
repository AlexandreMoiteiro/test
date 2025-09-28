
# app.py — NAVLOG (PDF + Relatório) para NAVLOG_FORM.pdf
# UI limpa; PDF tab = só gerar PDF; ETE ↑10 s; dist 0.1 nm; fuel 0.5 L; TAS/GS/ângulos/FF inteiros;
# Altitudes <1000→50 ft; ≥1000→100 ft; Vento FROM; TOC/TOD dentro do leg; NAVAIDs opcionais.
# Reqs: streamlit, pypdf, reportlab, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod

# --------------- App Config ---------------
st.set_page_config(page_title="NAVLOG (PDF + Relatório)", layout="wide", initial_sidebar_state="collapsed")
PDF_TEMPLATE = "NAVLOG_FORM.pdf"   # usa o PDF novo (mesma pasta)

# --------------- Optional imports ---------------
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, PageBreak
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# --------------- Formatting / Rounding ---------------
def _round_alt(x: float) -> int:
    if x is None: return 0
    v = abs(float(x))
    base = 50 if v < 1000 else 100
    return int(round(float(x)/base) * base)

def _round_unit(x: float) -> int:
    if x is None: return 0
    return int(round(float(x)))

def _round_half(x: float) -> float:
    if x is None: return 0.0
    return round(float(x)*2.0)/2.0

def _round_angle(x: float) -> int:
    if x is None: return 0
    return int(round(float(x))) % 360

def ceil_to_10s(sec: float) -> int:
    if sec <= 0: return 0
    s = int(math.ceil(sec / 10.0) * 10)
    return max(s, 10)

def hhmmss(tsec: int) -> str:
    h = tsec // 3600
    m = (tsec % 3600) // 60
    s = tsec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def mmss(tsec: int) -> str:
    m = tsec // 60
    s = tsec % 60
    return f"{m:02d}:{s:02d}"

def fmt(x: float, kind: str) -> str:
    if kind == "dist":   return f"{round(float(x or 0),1):.1f}"
    if kind == "fuel":   return f"{_round_half(x):.1f}"
    if kind == "ff":     return str(_round_unit(x))
    if kind == "speed":  return str(_round_unit(x))
    if kind == "angle":  return str(_round_angle(x))
    if kind == "alt":    return str(_round_alt(x))
    return str(x)

def ascii_safe(x: str) -> str:
    return unicodedata.normalize("NFKD", str(x or "")).encode("ascii","ignore").decode("ascii")

def parse_hhmm(s:str):
    s=(s or "").strip()
    for fmt in ("%H:%M:%S","%H:%M","%H%M"):
        try:
            return dt.datetime.strptime(s,fmt).time()
        except:
            pass
    return None

def add_seconds(t:dt.time, s:int):
    if not t: return None
    today=dt.date.today(); base=dt.datetime.combine(today,t)
    return (base + dt.timedelta(seconds=int(s))).time()

# --------------- Perf tables (exemplo) ---------------
ROC_ENROUTE = {
    0:{-25:981,0:835,25:704,50:586},  2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377},6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170},10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37},14000:{-25:210,0:80,25:-35,50:-139},
}
ROC_FACTOR = 0.90
VY_ENROUTE = {0:67,2000:67,4000:67,6000:67,8000:67,10000:67,12000:67,14000:67}
CRUISE={
    0:{1800:(82,15.3),1900:(89,17.0),2000:(95,18.7),2100:(101,20.7),2250:(110,24.6),2388:(118,26.9)},
    2000:{1800:(82,15.3),1900:(88,16.6),2000:(94,17.5),2100:(100,19.9),2250:(109,23.5)},
    4000:{1800:(81,15.1),1900:(88,16.2),2000:(94,17.5),2100:(100,19.2),2250:(108,22.4)},
    6000:{1800:(81,14.9),1900:(87,15.9),2000:(93,17.1),2100:(99,18.5),2250:(108,21.3)},
    8000:{1800:(81,14.9),1900:(86,15.6),2000:(92,16.7),2100:(98,18.0),2250:(107,20.4)},
    10000:{1800:(85,15.4),1900:(91,16.4),2000:(91,16.4),2100:(97,17.5),2250:(106,19.7)},
}
def clamp(v,lo,hi): return max(lo,min(hi,v))
def interp1(x,x0,x1,y0,y1):
    if x1==x0: return y0
    t=(x-x0)/(x1-x0); return y0+t*(y1-y0)
def isa_temp(pa_ft): return 15.0 - 2.0*(pa_ft/1000.0)

def cruise_lookup(pa_ft: float, rpm: int, oat_c: Optional[float]) -> Tuple[float,float]:
    pas=sorted(CRUISE.keys()); pa_c=clamp(pa_ft,pas[0],pas[-1])
    p0=max([p for p in pas if p<=pa_c]); p1=min([p for p in pas if p>=pa_c])
    def val(pa):
        table=CRUISE[pa]
        if rpm in table: return table[rpm]
        rpms=sorted(table.keys())
        if rpm<rpms[0]: lo,hi=rpms[0],rpms[1]
        elif rpm>rpms[-1]: lo,hi=rpms[-2],rpms[-1]
        else:
            lo=max([r for r in rpms if r<=rpm]); hi=min([r for r in rpms if r>=rpm])
        (tas_lo,ff_lo),(tas_hi,ff_hi)=table[lo],table[hi]
        t=(rpm-lo)/(hi-lo) if hi!=lo else 0.0
        return (tas_lo + t*(tas_hi-tas_lo), ff_lo + t*(ff_hi-ff_lo))
    tas0,ff0=val(p0); tas1,ff1=val(p1)
    tas=interp1(pa_c,p0,p1,tas0,tas1); ff=interp1(pa_c,p0,p1,ff0,ff1)
    if oat_c is not None:
        dev=oat_c - isa_temp(pa_c)
        if dev>0: tas*=1-0.02*(dev/15); ff*=1-0.025*(dev/15)
        elif dev<0: tas*=1+0.01*((-dev)/15); ff*=1+0.03*((-dev)/15)
    return max(0.0,tas), max(0.0,ff)

def roc_interp_enroute(pa, temp_c):
    pas=sorted(ROC_ENROUTE.keys()); pa_c=clamp(pa,pas[0],pas[-1])
    p0=max([p for p in pas if p<=pa_c]); p1=min([p for p in pas if p>=pa_c])
    temps=[-25,0,25,50]; t=clamp(temp_c,temps[0],temps[-1])
    if t<=0: t0,t1=-25,0
    elif t<=25: t0,t1=0,25
    else: t0,t1=25,50
    v00, v01 = ROC_ENROUTE[p0][t0], ROC_ENROUTE[p0][t1]
    v10, v11 = ROC_ENROUTE[p1][t0], ROC_ENROUTE[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01); v1 = interp1(pa_c, p0, p1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1) * ROC_FACTOR)

def vy_interp_enroute(pa):
    pas=sorted(VY_ENROUTE.keys()); pa_c=clamp(pa,pas[0],pas[-1])
    p0=max([p for p in pas if p<=pa_c]); p1=min([p for p in pas if p>=pa_c])
    return interp1(pa_c, p0, p1, VY_ENROUTE[p0], VY_ENROUTE[p1])

# --------------- Wind / Variation ---------------
def wrap360(x): x=fmod(x,360.0); return x+360 if x<0 else x
def angle_diff(a,b): return (a-b+180)%360-180

def wind_triangle(tc_deg: float, tas_kt: float, wind_from_deg: float, wind_kt: float):
    if tas_kt <= 0: return 0.0, wrap360(tc_deg), 0.0
    delta = radians(angle_diff(wind_from_deg, tc_deg))
    cross = wind_kt * sin(delta)
    s = max(-1.0, min(1.0, cross/max(tas_kt,1e-9)))
    wca = degrees(asin(s))
    th  = wrap360(tc_deg + wca)
    gs  = max(0.0, tas_kt*math.cos(radians(wca)) - wind_kt*math.cos(delta))
    return wca, th, gs

def apply_var(true_deg,var_deg,east_is_negative=False):
    return wrap360(true_deg - var_deg if east_is_negative else true_deg + var_deg)

# --------------- Aeródromos (exemplo) ---------------
AEROS={
 "LPSO":{"elev":390,"freq":"119.805"},
 "LPEV":{"elev":807,"freq":"122.705"},
 "LPCB":{"elev":1251,"freq":"122.300"},
 "LPCO":{"elev":587,"freq":"118.405"},
 "LPVZ":{"elev":2060,"freq":"118.305"},
}
def aero_elev(icao): return int(AEROS.get(icao,{}).get("elev",0))
def aero_freq(icao): return AEROS.get(icao,{}).get("freq","")

# --------------- PDF helpers ---------------
def read_pdf_bytes(path: str) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.read_bytes()

def get_fields(template_bytes: bytes):
    reader = PdfReader(io.BytesIO(template_bytes))
    fd = reader.get_fields() or {}
    names = set(fd.keys())
    maxlens = {}
    for k,v in fd.items():
        ml = v.get("/MaxLen")
        if ml: maxlens[k] = int(ml)
    return names, maxlens

def fill_pdf(template_bytes: bytes, fields: dict) -> bytes:
    reader = PdfReader(io.BytesIO(template_bytes))
    writer = PdfWriter()
    for p in reader.pages: writer.add_page(p)
    root = reader.trailer["/Root"]
    if "/AcroForm" not in root: raise RuntimeError("Template has no AcroForm")
    writer._root_object.update({NameObject("/AcroForm"): root["/AcroForm"]})
    try:
        writer._root_object["/AcroForm"].update({
            NameObject("/NeedAppearances"): True,
            NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g")
        })
    except: pass
    str_fields = {k:(str(v) if v is not None else "") for k,v in fields.items()}
    for page in writer.pages:
        writer.update_page_form_field_values(page, str_fields)
    bio = io.BytesIO(); writer.write(bio); return bio.getvalue()

def put(out: dict, fieldset: set, key: str, value: str, maxlens: Dict[str,int]):
    if key in fieldset:
        s = "" if value is None else str(value)
        if key in maxlens and len(s) > maxlens[key]:
            s = s[:maxlens[key]]
        out[key] = s

# --------------- UI ---------------
st.title("Navigation Plan & Inflight Log — Tecnam P2008")

# Cabeçalho (Form evita perder inputs)
with st.form("hdr"):
    c1,c2,c3 = st.columns(3)
    with c1:
        aircraft = st.text_input("Aircraft", "P208")
        registration = st.selectbox("Registration", ["CS-ECC","CS-ECD","CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW"], index=0)
        callsign = st.text_input("Callsign", "RVP")
    with c2:
        student = st.text_input("Student", "AMOIT")
        lesson  = st.text_input("Lesson (ex: 12)", "")
        instrutor = st.text_input("Instrutor","")
    with c3:
        dept = st.selectbox("Departure", list(AEROS.keys()), index=0)
        arr  = st.selectbox("Arrival",  list(AEROS.keys()), index=1)
        altn = st.selectbox("Alternate",list(AEROS.keys()), index=2)
    startup_str = st.text_input("Startup (HH:MM ou HH:MM:SS)", "")

    with st.expander("Atmosfera / Performance / Opções", expanded=False):
        c4,c5,c6 = st.columns(3)
        with c4:
            qnh = st.number_input("QNH (hPa)", 900, 1050, 1013, step=1)
            cruise_alt = st.number_input("Cruise Altitude (ft)", 0, 14000, 4000, step=100)
        with c5:
            temp_c = st.number_input("OAT (°C)", -40, 50, 15, step=1)
            var_deg = st.number_input("Mag Variation (°)", 0, 30, 1, step=1)
            var_is_e = (st.selectbox("Variação E/W", ["W","E"], index=0)=="E")
        with c6:
            wind_from = st.number_input("Wind FROM (°TRUE)", 0, 360, 0, step=1)
            wind_kt   = st.number_input("Wind (kt)", 0, 120, 17, step=1)

        c7,c8,c9 = st.columns(3)
        with c7:
            rpm_climb  = st.number_input("Climb RPM (AFM)", 1800, 2388, 2250, step=10)
            rpm_cruise = st.number_input("Cruise RPM (AFM)", 1800, 2388, 2000, step=10)
        with c8:
            descent_ff = st.number_input("Descent FF (L/h)", 0.0, 30.0, 15.0, step=0.1)
        with c9:
            rod_fpm = st.number_input("ROD (ft/min)", 200, 1500, 700, step=10)
            start_fuel = st.number_input("Fuel inicial (EFOB_START) [L]", 0.0, 1000.0, 85.0, step=0.1)

        st.markdown("---")
        use_navaids_pdf = st.checkbox("Usar NAVAIDs no PDF (preencher IDENT/FREQ quando marcado por linha)", value=False)
        cruise_ref_kt  = st.number_input("Cruise speed (kt)", 40, 140, 90, step=1)
        descent_ref_kt = st.number_input("Descent speed (kt)", 40, 120, 65, step=1)

    # Rota (texto simples)
    st.markdown("##### Rota")
    default_route = f"{dept} {arr}"
    route_text = st.text_area("Pontos (separados por espaço, vírgulas ou '->')",
                              value=(st.session_state.get("route_text") or default_route))
    hdr_submit = st.form_submit_button("Aplicar cabeçalho & rota", type="primary")

def parse_route_text(txt:str) -> List[str]:
    tokens = re.split(r"[,\s→\-]+", (txt or "").strip())
    return [t for t in tokens if t]

if "points" not in st.session_state:
    st.session_state.points = parse_route_text(route_text) if route_text else [dept,arr]
if hdr_submit:
    st.session_state["route_text"] = route_text
    pts = parse_route_text(route_text) or [dept,arr]
    if not pts: pts=[dept,arr]
    pts[0]=dept
    if len(pts)>=2: pts[-1]=arr
    st.session_state.points = pts
    st.session_state["use_navaids_pdf"] = use_navaids_pdf

points = st.session_state.points
use_navaids_pdf = st.session_state.get("use_navaids_pdf", False)

# --------------- Planeamento (tabela única) ---------------
def make_default_rows(points: List[str], cruise_alt:int) -> List[dict]:
    rows=[]
    arr_elev=_round_alt(aero_elev(points[-1]))
    for i in range(1,len(points)):
        to_is_last = (i == len(points)-1)
        rows.append({
            "From": points[i-1],
            "To": points[i],
            "TC": 0.0,
            "Dist": 0.0,
            "ALT_to_ft": float(arr_elev if to_is_last else _round_alt(cruise_alt)),
            "UseNavaid": False,
            "Navaid_IDENT": "",
            "Navaid_FREQ": "",
        })
    return rows

def preserve_merge(old: List[dict], new_points: List[str], cruise_alt:int) -> List[dict]:
    idx = {(r.get("From"), r.get("To")): r for r in old}
    merged=[]
    arr_elev=_round_alt(aero_elev(new_points[-1]))
    for i in range(1,len(new_points)):
        key=(new_points[i-1], new_points[i])
        base=idx.get(key, {})
        to_is_last=(i==len(new_points)-1)
        merged.append({
            "From": key[0], "To": key[1],
            "TC": float(base.get("TC", 0.0)),
            "Dist": float(base.get("Dist", 0.0)),
            "ALT_to_ft": float(base.get("ALT_to_ft", (arr_elev if to_is_last else _round_alt(cruise_alt)))),
            "UseNavaid": bool(base.get("UseNavaid", False)),
            "Navaid_IDENT": base.get("Navaid_IDENT",""),
            "Navaid_FREQ":  base.get("Navaid_FREQ",""),
        })
    return merged

if "plan_rows" not in st.session_state or hdr_submit:
    if "plan_rows" in st.session_state:
        st.session_state.plan_rows = preserve_merge(st.session_state.plan_rows, points, cruise_alt)
    else:
        st.session_state.plan_rows = make_default_rows(points, cruise_alt)

tab_plan, tab_results, tab_pdf, tab_report = st.tabs(["📝 Planeamento", "📊 Resultados", "📄 PDF", "📑 Relatório"])

with tab_plan:
    st.markdown("### Planeamento")
    column_config = {
        "From": st.column_config.TextColumn("From", disabled=True),
        "To":   st.column_config.TextColumn("To", disabled=True),
        "TC":   st.column_config.NumberColumn("TC (°T)", step=0.1, min_value=0.0, max_value=359.9),
        "Dist": st.column_config.NumberColumn("Dist (nm)", step=0.1, min_value=0.0),
        "ALT_to_ft": st.column_config.NumberColumn("ALT alvo no To (ft)", step=50, min_value=0.0),
        "UseNavaid": st.column_config.CheckboxColumn("Usar navaid?"),
        "Navaid_IDENT": st.column_config.TextColumn("Navaid IDENT"),
        "Navaid_FREQ":  st.column_config.TextColumn("Navaid FREQ"),
    }
    with st.form("plan_form"):
        plan_edit = st.data_editor(
            st.session_state.plan_rows,
            key="plan_table",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config=column_config,
            column_order=list(column_config.keys())
        )
        plan_submit = st.form_submit_button("Guardar planeamento", type="primary")
    if plan_submit:
        st.session_state.plan_rows = plan_edit

# --------------- Cálculos comuns ---------------
legs = st.session_state.plan_rows
N = len(legs)

def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0
dep_elev = _round_alt(aero_elev(points[0]))
arr_elev = _round_alt(aero_elev(points[-1]))
altn_elev = _round_alt(aero_elev(altn))
start_alt = float(dep_elev); end_alt = float(arr_elev)

pa_start  = pressure_alt(start_alt, qnh)
pa_cruise = pressure_alt(cruise_alt, qnh)
vy_kt = vy_interp_enroute(pa_start)
tas_climb, tas_cruise, tas_descent = vy_kt, float(cruise_ref_kt), float(descent_ref_kt)
roc = roc_interp_enroute(pa_start, temp_c)

pa_mid_climb = start_alt + 0.5*max(0.0, cruise_alt - start_alt)
pa_mid_desc  = end_alt   + 0.5*max(0.0, cruise_alt - end_alt)
_, ff_climb  = cruise_lookup(pa_mid_climb, int(rpm_climb),  temp_c)
_, ff_cruise = cruise_lookup(pa_cruise,   int(rpm_cruise),  temp_c)
ff_descent   = float(descent_ff)

dist = [float(legs[i]["Dist"] or 0.0) for i in range(N)]
tcs  = [float(legs[i]["TC"]   or 0.0) for i in range(N)]
alt_targets = [start_alt] + [float(legs[i].get("ALT_to_ft", arr_elev if i==N-1 else _round_alt(cruise_alt))) for i in range(N)]

front_used_dist = [0.0]*N
back_used_dist  = [0.0]*N
climb_time_alloc   = [0.0]*N
descent_time_alloc = [0.0]*N
impossible_notes = []
toc_markers = []; tod_markers = []

def gs_for_leg(i:int, phase:str) -> float:
    wdir, wkt = int(wind_from), int(wind_kt)
    tas = vy_kt if phase=="CLIMB" else (cruise_ref_kt if phase=="CRUISE" else descent_ref_kt)
    _,_,gs = wind_triangle(tcs[i], tas, wdir, wkt)
    return max(gs,1e-6)

def alloc_front(i, time_min):
    gs = gs_for_leg(i,"CLIMB")
    max_time = max(0.0, (dist[i] - front_used_dist[i] - back_used_dist[i]) * 60.0 / gs)
    use = min(time_min, max_time)
    d = gs * use / 60.0
    front_used_dist[i] += d
    climb_time_alloc[i] += use
    return use

def alloc_back(i, time_min):
    gs = gs_for_leg(i,"DESCENT")
    max_time = max(0.0, (dist[i] - front_used_dist[i] - back_used_dist[i]) * 60.0 / gs)
    use = min(time_min, max_time)
    d = gs * use / 60.0
    back_used_dist[i] += d
    descent_time_alloc[i] += use
    return use

# Cumprir alvos de altitude por "To"
for j in range(1, len(alt_targets)):
    delta = alt_targets[j] - alt_targets[j-1]
    if delta > 0:
        need = delta / max(roc,1e-6)
        rem = need
        for k in range(j-1, -1, -1):
            if rem <= 1e-9: break
            rem -= alloc_front(k, rem)
        if rem > 1e-6:
            impossible_notes.append(f"Impossível atingir {int(alt_targets[j])} ft em {legs[j-1]['To']} (climb insuficiente). Falta {rem:.1f} min.")
    elif delta < 0:
        need = (-delta) / max(rod_fpm,1e-6)
        rem = need
        for k in range(j-1, -1, -1):
            if rem <= 1e-9: break
            rem -= alloc_back(k, rem)
        if rem > 1e-6:
            impossible_notes.append(f"Impossível atingir {int(alt_targets[j])} ft em {legs[j-1]['To']} (descent insuficiente). Falta {rem:.1f} min.")

startup = parse_hhmm(startup_str)
takeoff = add_seconds(startup, 15*60) if startup else None
clock = takeoff

rows=[]; seq_points=[]; calc_rows=[]; calc_details=[]
PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓"}
efob=float(start_fuel)

# Linha 0: DEP (chegada ao DEP “virtual”)
seq_points.append({"name": points[0], "alt": _round_alt(alt_targets[0]),
                   "tc":"", "th":"", "mc":"", "mh":"", "tas":"", "gs":"", "dist":"",
                   "ete_sec":0, "eto": (takeoff.strftime("%H:%M") if takeoff else ""),
                   "burn":"", "efob": float(start_fuel), "leg_idx": None})

def add_segment(phase, frm, to, i_leg, d_nm, tas, ff_lph, alt_start_ft, rate_fpm, wdir, wkt):
    global clock, efob
    if d_nm <= 1e-9: return alt_start_ft
    tc = float(tcs[i_leg])
    wca, th, gs = wind_triangle(tc, tas, wdir, wkt)
    mc = apply_var(tc, var_deg, var_is_e); mh = apply_var(th, var_deg, var_is_e)

    # Tempo
    ete_sec_raw = (60.0 * d_nm / max(gs,1e-6)) * 60.0
    ete_sec = ceil_to_10s(ete_sec_raw)
    ete_disp = mmss(int(ete_sec))

    # Fuel
    burn_raw = ff_lph * (ete_sec_raw / 3600.0)
    burn = _round_half(burn_raw)

    # Altitude
    if phase == "CLIMB":   alt_end_ft = alt_start_ft + rate_fpm * (ete_sec_raw/60.0)
    elif phase == "DESCENT": alt_end_ft = alt_start_ft - rate_fpm * (ete_sec_raw/60.0)
    else: alt_end_ft = alt_start_ft

    # ETO
    eto = ""
    if clock:
        clock = add_seconds(clock, int(ete_sec))
        eto = clock.strftime("%H:%M")

    efob = max(0.0, _round_half(efob - burn_raw))

    # Tabela da app
    rows.append({
        "Fase": PH_ICON[phase], "Leg/Marker": f"{frm}→{to}",
        "ALT (ft)": f"{fmt(alt_start_ft,'alt')}→{fmt(alt_end_ft,'alt')}",
        "TC (°T)": _round_angle(tc), "TH (°T)": _round_angle(th),
        "MC (°M)": _round_angle(mc), "MH (°M)": _round_angle(mh),
        "TAS (kt)": _round_unit(tas), "GS (kt)": _round_unit(gs),
        "FF (L/h)": _round_unit(ff_lph),
        "Dist (nm)": fmt(d_nm,'dist'), "ETE (mm:ss)": ete_disp, "ETO": eto,
        "Burn (L)": fmt(burn,'fuel'), "EFOB (L)": fmt(efob,'fuel')
    })

    # Para PDF
    seq_points.append({
        "name": to, "alt": _round_alt(alt_end_ft),
        "tc": _round_angle(tc), "th": _round_angle(th),
        "mc": _round_angle(mc), "mh": _round_angle(mh),
        "tas": _round_unit(tas), "gs": _round_unit(gs),
        "dist": float(f"{d_nm:.3f}"),
        "ete_sec": int(ete_sec), "eto": eto,
        "burn": float(burn_raw), "efob": float(efob),
        "leg_idx": int(i_leg)
    })

    # Relatório: fórmulas
    delta = angle_diff(wdir, tc)
    calc_rows.append([f"{frm}→{to}", phase,
                      f"{_round_angle(tc)}°", f"{_round_angle(mc)}°", f"{_round_angle(th)}°", f"{_round_angle(mh)}°",
                      _round_unit(tas), _round_unit(gs),
                      fmt(d_nm,'dist'), ete_disp, eto or "—",
                      fmt(burn,'fuel'), fmt(efob,'fuel'),
                      f"{fmt(alt_start_ft,'alt')}→{fmt(alt_end_ft,'alt')}"])
    calc_details.append(
        "• {src}->{dst} [{ph}]  Δ=(W_from−TC)={dl:.1f}°;  WCA=asin((W/TAS)·sinΔ)={wca:.2f}°;  "
        "TH=TC+WCA={th:.2f}° → MH=TH±Var={mh:.2f}°;  "
        "GS=TAS·cos(WCA)−W·cosΔ={gs:.2f} kt;  Dist={d:.2f} nm;  "
        "ETE_raw={sec:.1f} s → ETE={eteds} (ceil 10 s);  "
        "Burn_raw=FF·(ETE_raw/3600)={br:.2f} → Burn={burn:.1f} L (0.5 L);  "
        "ALT {h0:.0f}→{h1:.0f} ft (50/100)."
        .format(src=frm, dst=to, ph=phase, dl=delta, wca=wca, th=th, mh=mh, gs=gs, d=d_nm,
                sec=ete_sec_raw, eteds=ete_disp, br=burn_raw, burn=_round_half(burn_raw), h0=alt_start_ft, h1=alt_end_ft)
    )
    return alt_end_ft

# Construção com cortes TOC/TOD dentro de legs
cur_alt = alt_targets[0]
for i in range(N):
    frm, to = legs[i]["From"], legs[i]["To"]
    d_total  = dist[i]
    d_cl = front_used_dist[i]
    d_ds = back_used_dist[i]
    d_cr = max(0.0, d_total - d_cl - d_ds)

    wdir_leg, wkt_leg = int(wind_from), int(wind_kt)

    if d_cl > 1e-9:
        toc_name = "TOC" if (d_cr + d_ds) > 0 else to
        cur_alt = add_segment("CLIMB", frm, toc_name, i, d_cl, vy_kt, ff_climb, cur_alt, roc, wdir_leg, wkt_leg)
        frm = toc_name
    if d_cr > 1e-9:
        tod_name = "TOD" if d_ds > 1e-9 else to
        cur_alt = add_segment("CRUISE", frm, tod_name, i, d_cr, float(cruise_ref_kt), ff_cruise, cur_alt, 0.0, wdir_leg, wkt_leg)
        frm = tod_name
    if d_ds > 1e-9:
        cur_alt = add_segment("DESCENT", frm, to, i, d_ds, float(descent_ref_kt), ff_descent, cur_alt, rod_fpm, wdir_leg, wkt_leg)

eta = clock
shutdown = add_seconds(eta, 5*60) if eta else None

# --------------- Resultados ---------------
with tab_results:
    st.markdown("### Resultados")
    cA,cB,cC = st.columns(3)
    with cA:
        st.metric("Vy (kt)", _round_unit(vy_kt))
        st.metric("ROC @ DEP (ft/min)", _round_unit(roc))
        st.metric("ROD (ft/min)", _round_unit(rod_fpm))
    with cB:
        st.metric("TAS climb / cruise / descent", f"{_round_unit(vy_kt)} / {_round_unit(cruise_ref_kt)} / {_round_unit(descent_ref_kt)} kt")
        st.metric("FF climb / cruise / descent", f"{_round_unit(ff_climb)} / {_round_unit(ff_cruise)} / {_round_unit(ff_descent)} L/h")
    with cC:
        isa_dev = temp_c - isa_temp(pressure_alt(dep_elev, qnh))
        st.metric("ISA dev @ DEP (°C)", int(round(isa_dev)))

    st.markdown("### Flight plan (linhas)")
    st.dataframe(rows, use_container_width=True)

    tot_ete_sec = sum(int(p.get('ete_sec',0)) for p in seq_points if isinstance(p.get('ete_sec'), (int,float)))
    tot_nm  = sum(float(p['dist']) for p in seq_points if isinstance(p.get('dist'), (int,float)))
    tot_bo_raw = sum(float(p['burn']) for p in seq_points if isinstance(p.get('burn'), (int,float)))
    tot_bo = _round_half(tot_bo_raw)
    line = f"**Totais** — Dist {fmt(tot_nm,'dist')} nm • ETE {hhmm(tot_ete_sec)} • Burn {fmt(tot_bo,'fuel')} L • EFOB {fmt(seq_points[-1]['efob'] if seq_points else start_fuel,'fuel')} L"
    st.markdown(line)
    if impossible_notes:
        st.warning(" / ".join(impossible_notes))

# --------------- PDF (só PDF nesta aba) ---------------
with tab_pdf:
    st.markdown("### Gerar PDF (formulário NAVLOG)")
    if not PYPDF_OK:
        st.error("pypdf não está disponível no ambiente.")
    else:
        try:
            template_bytes = read_pdf_bytes(PDF_TEMPLATE)
            fieldset, maxlens = get_fields(template_bytes)
        except Exception as e:
            template_bytes=None; fieldset=set(); maxlens={}
            st.error(f"Não foi possível ler o PDF: {e}")

        named: Dict[str,str] = {}
        def P(key, value): put(named, fieldset, key, value, maxlens)

        if template_bytes:
            etd = (add_seconds(parse_hhmm(startup_str), 15*60).strftime("%H:%M") if startup_str else "")
            tot_ete_sec = sum(int(p.get('ete_sec',0)) for p in seq_points if isinstance(p.get('ete_sec'), (int,float)))
            isa_dev = int(round(temp_c - isa_temp(pressure_alt(dep_elev, qnh))))
            mag_str = f"{int(round(var_deg))}{'E' if var_is_e else 'W'}"

            # Cabeçalho (nomes exatos)
            P("AIRCRAFT", aircraft)
            P("REGISTRATION", registration)
            P("CALLSIGN", callsign)
            P("ETD/ETA", f"{etd} / {eta.strftime('%H:%M') if eta else ''}")
            P("STARTUP", startup_str)
            P("TAKEOFF", etd)
            P("LANDING", eta.strftime("%H:%M") if eta else "")
            P("SHUTDOWN", shutdown.strftime("%H:%M") if shutdown else "")
            P("LESSON", lesson)
            P("INSTRUTOR", instrutor)
            P("STUDENT", student)
            P("FLT TIME", f"{(tot_ete_sec//3600):02d}:{((tot_ete_sec%3600)//60):02d}")
            P("LEVEL F/F", fmt(cruise_alt,'alt'))
            P("FLIGHT_LEVEL/ALTITUDE", fmt(cruise_alt,'alt'))
            P("CLIMB FUEL", fmt((sum(climb_time_alloc)/60.0*ff_climb),'fuel'))
            P("QNH", str(int(round(qnh))))
            P("DEPT", aero_freq(points[0]))
            P("ENROUTE", "123.755")
            P("ARRIVAL", aero_freq(points[-1]))
            P("WIND", f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}")
            P("MAG_VAR", mag_str)
            P("TEMP/ISA_DEV", f"{int(round(temp_c))} / {isa_dev}")
            P("Departure_Airfield", points[0])
            P("Arrival_Airfield", points[-1])
            P("Alternate_Airfield", altn)
            P("Alternate_Elevation", fmt(altn_elev,'alt'))
            P("Leg_Number", str(len(seq_points)))

            # Linhas (Leg01..Leg22)
            acc_dist = 0.0
            acc_sec  = 0
            for idx, p in enumerate(seq_points, start=1):
                if idx > 22: break
                tag = f"Leg{idx:02d}_"
                is_seg = (idx>1)

                P(tag+"Waypoint", p["name"])
                P(tag+"Altitude_FL", (fmt(p["alt"], 'alt') if p["alt"]!="" else ""))

                if is_seg and use_navaids_pdf:
                    leg_idx = p.get("leg_idx", None)
                    if isinstance(leg_idx, int) and 0 <= leg_idx < len(legs):
                        leg = legs[leg_idx]
                        if leg.get("UseNavaid", False):
                            P(tag+"Navaid_Identifier", leg.get("Navaid_IDENT",""))
                            P(tag+"Navaid_Frequency",  leg.get("Navaid_FREQ",""))

                if is_seg:
                    acc_dist += float(p["dist"] or 0.0)
                    acc_sec  += int(p.get("ete_sec",0) or 0)
                    P(tag+"True_Course",      fmt(p["tc"], 'angle'))
                    P(tag+"True_Heading",     fmt(p["th"], 'angle'))
                    P(tag+"Magnetic_Heading", fmt(p["mh"], 'angle'))
                    P(tag+"True_Airspeed",    fmt(p["tas"], 'speed'))
                    P(tag+"Ground_Speed",     fmt(p["gs"], 'speed'))
                    P(tag+"Leg_Distance",     fmt(p["dist"], 'dist'))
                    P(tag+"Leg_ETE",          mmss(int(p.get("ete_sec",0))))
                    P(tag+"ETO",              p["eto"])
                    P(tag+"Planned_Burnoff",  fmt(p["burn"], 'fuel'))
                    P(tag+"Estimated_FOB",    fmt(p["efob"], 'fuel'))
                    P(tag+"Cumulative_Distance", fmt(acc_dist,'dist'))
                    P(tag+"Cumulative_ETE",      mmss(acc_sec))
                else:
                    P(tag+"ETO", p["eto"])
                    P(tag+"Estimated_FOB", fmt(p["efob"], 'fuel'))

            # Linha 23 (se existir na segunda página, é opcional)
            if "Leg23_Leg_Distance" in fieldset:
                P("Leg23_Leg_Distance", fmt(acc_dist,'dist'))
                P("Leg23_Leg_ETE",      mmss(acc_sec))
                # Totais de combustível com a mesma política
                tot_bo_raw = sum(float(pp['burn']) for pp in seq_points if isinstance(pp.get('burn'), (int,float,float)))
                P("Leg23_Planned_Burnoff", fmt(tot_bo_raw,'fuel'))
                P("Leg23_Estimated_FOB",   fmt(seq_points[-1]['efob'],'fuel'))

            st.button("Gerar PDF NAVLOG", type="primary", key="genpdf")
            if st.session_state.get("genpdf"):
                try:
                    out = fill_pdf(template_bytes, named)
                    m = re.search(r'(\d+)', lesson or "")
                    lesson_num = m.group(1) if m else "00"
                    safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
                    filename = f"{safe_date}_LESSON-{lesson_num}_NAVLOG.pdf"
                    st.download_button("📄 Download PDF", data=out, file_name=filename, mime="application/pdf", key="dlpdf")
                    st.success("PDF gerado (planeado).")
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {e}")

# --------------- Relatório (A4 landscape, legível) ---------------
with tab_report:
    def build_report(calc_rows: List[List], details: List[str], params: Dict[str,str]) -> bytes:
        if not REPORTLAB_OK: raise RuntimeError("reportlab missing")
        bio = io.BytesIO()
        doc = SimpleDocTemplate(bio, pagesize=landscape(A4),
                                leftMargin=16*mm, rightMargin=16*mm,
                                topMargin=12*mm, bottomMargin=12*mm)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=11))
        H1 = styles["Heading1"]; H2 = styles["Heading2"]; P = styles["Small"]

        story=[]
        story.append(Paragraph("NAVLOG — Relatório (Planeado)", H1))
        story.append(Spacer(1,6))

        resume = [
            ["Aeronave", params.get("aircraft","—")],
            ["Matrícula", params.get("registration","—")],
            ["Callsign", params.get("callsign","—")],
            ["Lição", params.get("lesson","—")],
            ["DEP / ARR / ALTN", f"{params.get('dept','—')} / {params.get('arr','—')} / {params.get('altn','—')}"],
            ["Elev DEP / ARR / ALTN", f"{fmt(params.get('elev_dep',0),'alt')} / {fmt(params.get('elev_arr',0),'alt')} / {fmt(params.get('elev_altn',0),'alt')} ft"],
            ["Cruise Alt", fmt(params.get("cruise_alt",0),'alt')+" ft"],
            ["QNH", params.get("qnh","—")],
            ["Vento (global FROM)", params.get("wind","—")],
            ["Var. Magn.", params.get("var","—")],
            ["OAT / ISA dev", params.get("temp_isa","—")],
            ["Startup / ETD", f"{params.get('startup','—')} / {params.get('etd','—')}"],
            ["ETA / Shutdown", f"{params.get('eta','—')} / {params.get('shutdown','—')}"],
            ["Tempo total (PLN)", params.get("flt_time","—")],
            ["Fuel inicial", params.get("start_fuel","—")+" L"],
            ["Notas", params.get("notes","—")],
        ]
        t1 = LongTable(resume, colWidths=[60*mm, None], hAlign="LEFT")
        t1.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
            ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("FONTSIZE",(0,0),(-1,-1),9),
        ]))
        story.append(t1)
        story.append(Spacer(1,6))

        story.append(Paragraph("Segmentos (resumo)", H2))
        data = [["From→To","Fase","TC°","MC°","TH°","MH°","TAS","GS","Dist(nm)","ETE","ETO","Burn(L)","EFOB(L)","ALT ini→fim"]]
        data += calc_rows
        t3 = LongTable(data,
                       colWidths=[54*mm, 16*mm, 12*mm, 12*mm, 12*mm, 12*mm, 16*mm, 16*mm, 24*mm, 18*mm, 24*mm, 16*mm, 16*mm, 40*mm],
                       repeatRows=1, hAlign="LEFT", splitByRow=1)
        t3.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("ALIGN",(2,1),(7,-1),"RIGHT"),
            ("ALIGN",(8,1),(12,-1),"RIGHT"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("FONTSIZE",(0,0),(-1,-1),8.0),
        ]))
        story.append(t3)
        story.append(PageBreak())

        story.append(Paragraph("Cálculos por segmento (fórmulas + aproximações)", H2))
        story.append(Paragraph(
            "Ângulos/Vento (FROM): Δ=(W_from−TC); WCA=asin((W/TAS)·sinΔ); TH=TC+WCA; MH=TH±Var. "
            "GS=TAS·cos(WCA)−W·cosΔ. Tempo: ETE_raw (s) → ETE arredondado ↑ a 10 s (mm:ss). "
            "Comb.: Burn_raw=FF·(ETE_raw/3600) → 0.5 L. Alt: ALT_end=ALT_ini±rate·(ETE_raw/60) → "
            "(<1000→50; ≥1000→100).", P))
        for s in calc_details:
            story.append(Paragraph(s, P))

        doc.build(story)
        return bio.getvalue()

    btn = st.button("Gerar Relatório (PDF legível)")
    if btn:
        try:
            tot_ete_sec = sum(int(p.get('ete_sec',0)) for p in seq_points if isinstance(p.get('ete_sec'), (int,float)))
            params = {
                "aircraft": aircraft, "registration": registration, "callsign": callsign,
                "lesson": lesson, "dept": points[0], "arr": points[-1], "altn": altn,
                "elev_dep": dep_elev, "elev_arr": arr_elev, "elev_altn": altn_elev,
                "qnh": str(int(round(qnh))),
                "wind": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
                "var": f"{int(round(var_deg))}{'E' if var_is_e else 'W'}",
                "cruise_alt": _round_alt(cruise_alt),
                "temp_isa": f"{int(round(temp_c))} / {int(round(temp_c - isa_temp(pressure_alt(dep_elev, qnh))))}",
                "startup": startup_str,
                "etd": (add_seconds(parse_hhmm(startup_str),15*60).strftime("%H:%M") if startup_str else ""),
                "eta": (eta.strftime("%H:%M") if eta else ""), "shutdown": (shutdown.strftime("%H:%M") if shutdown else ""),
                "flt_time": f"{(tot_ete_sec//3600):02d}:{((tot_ete_sec%3600)//60):02d}",
                "start_fuel": fmt(start_fuel,'fuel'),
                "notes": " ; ".join(impossible_notes) if impossible_notes else "—",
            }
            rep = build_report(calc_rows, calc_details, params)
            m = re.search(r'(\d+)', lesson or "")
            lesson_num = m.group(1) if m else "00"
            safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
            filename = f"{safe_date}_LESSON-{lesson_num}_NAVLOG_RELATORIO.pdf"
            st.download_button("📑 Download Relatório (PDF)", data=rep, file_name=filename, mime="application/pdf")
            st.success("Relatório gerado.")
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {e}")
