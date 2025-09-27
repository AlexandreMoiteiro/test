# app.py — NAVLOG (PDF + Relatório) com:
#   • UI limpa em tabs + forms (inputs não desaparecem)
#   • UMA tabela única de planeamento (From, To, TC, Dist, ALT_to, vento por waypoint, Navaid opcional)
#   • TOC/TOD corretos dentro das pernas (cumpre alvos por waypoint)
#   • NAVAIDs só quando marcado
#   • Altitude do alternante no PDF
#   • Export/Import da rota (JSON)
#   • Métricas de performance visíveis (ROC, Vy, TAS/FF, ROD)
#   • Relatório com explicação de cada conta + arredondamento aplicado (LongTable)
#
# Reqs: streamlit, pypdf, reportlab, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod

# ---------------- Config ----------------
st.set_page_config(page_title="NAVLOG (PDF + Relatório)", layout="wide", initial_sidebar_state="collapsed")
PDF_TEMPLATE_PATHS = ["NAVLOG_FORM.pdf"]  # coloca este ficheiro ao lado do app.py

# ---------------- Imports dinâmicos ----------------
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, PageBreak
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------------- Arredondamentos ----------------
def _round_alt(x: float) -> int:
    if x is None: return 0
    v = abs(float(x)); base = 5 if v < 1000 else 100
    return int(round(float(x)/base) * base)

def _round_unit(x: float) -> int:
    if x is None: return 0
    return int(round(float(x)))

def _round_half(x: float) -> float:
    if x is None: return 0.0
    return round(float(x)*2.0)/2.0  # 0.5 em 0.5

def _round_angle(x: float) -> int:
    if x is None: return 0
    return int(round(float(x))) % 360

def fmt(x: float, kind: str) -> str:
    if kind == "dist":   return f"{round(float(x or 0),1):.1f}"
    if kind == "mins":   return str(_round_unit(x))
    if kind == "fuel":   return f"{_round_half(x):.1f}"
    if kind == "ff":     return str(_round_unit(x))
    if kind == "speed":  return str(_round_unit(x))
    if kind == "angle":  return str(_round_angle(x))
    if kind == "alt":    return str(_round_alt(x))
    return str(x)

# ---------------- Helpers ----------------
def ascii_safe(x: str) -> str:
    return unicodedata.normalize("NFKD", str(x or "")).encode("ascii","ignore").decode("ascii")

def parse_hhmm(s:str):
    s=(s or "").strip()
    for fmt in ("%H:%M","%H%M"):
        try: return dt.datetime.strptime(s,fmt).time()
        except: pass
    return None

def add_minutes(t:dt.time,m:int):
    if not t: return None
    today=dt.date.today(); base=dt.datetime.combine(today,t)
    return (base+dt.timedelta(minutes=m)).time()

def clamp(v,lo,hi): return max(lo,min(hi,v))
def interp1(x,x0,x1,y0,y1):
    if x1==x0: return y0
    t=(x-x0)/(x1-x0); return y0+t*(y1-y0)

def wrap360(x): x=fmod(x,360.0); return x+360 if x<0 else x
def angle_diff(a,b): return (a-b+180)%360-180

# ---------------- AFM & tabelas ----------------
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

# ---------------- Vento/var ----------------
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

# ---------------- Aeródromos (exemplo) ----------------
AEROS={
 "LPSO":{"elev":390,"freq":"119.805"},
 "LPEV":{"elev":807,"freq":"122.705"},
 "LPCB":{"elev":1251,"freq":"122.300"},
 "LPCO":{"elev":587,"freq":"118.405"},
 "LPVZ":{"elev":2060,"freq":"118.305"},
}
def aero_elev(icao): return int(AEROS.get(icao,{}).get("elev",0))
def aero_freq(icao): return AEROS.get(icao,{}).get("freq","")

# ---------------- PDF helpers ----------------
def read_pdf_bytes(paths: List[str]) -> bytes:
    for p in paths:
        if Path(p).exists():
            return Path(p).read_bytes()
    raise FileNotFoundError(paths)

def get_form_fields(template_bytes: bytes):
    reader = PdfReader(io.BytesIO(template_bytes))
    fd = reader.get_fields() or {}
    field_names = set(fd.keys())
    maxlens = {}
    for k,v in fd.items():
        ml = v.get("/MaxLen")
        if ml: maxlens[k] = int(ml)
    return field_names, maxlens

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

# ---------------- UI ----------------
st.title("Navigation Plan & Inflight Log — Tecnam P2008")

DEFAULT_STUDENT="AMOIT"; DEFAULT_AIRCRAFT="P208"; DEFAULT_CALLSIGN="RVP"
REGS=["CS-ECC","CS-ECD","CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW"]

tab_plan, tab_perf, tab_results, tab_io = st.tabs(["🧭 Rota & Planeamento", "🌤️ Atmosfera & Perf.", "📋 Resultados", "💾 Exportar / Importar"])

with tab_perf:
    c1,c2,c3=st.columns(3)
    with c1:
        aircraft=st.text_input("Aircraft",DEFAULT_AIRCRAFT, key="aircraft")
        registration=st.selectbox("Registration",REGS,index=0, key="registration")
        callsign=st.text_input("Callsign",DEFAULT_CALLSIGN, key="callsign")
    with c2:
        student=st.text_input("Student",DEFAULT_STUDENT, key="student")
        lesson = st.text_input("Lesson (ex: 12)", "", key="lesson")
        instrutor = st.text_input("Instrutor","", key="instrutor")
    with c3:
        dept=st.selectbox("Departure",list(AEROS.keys()),index=0, key="dept")
        arr =st.selectbox("Arrival", list(AEROS.keys()),index=1, key="arr")
        altn=st.selectbox("Alternate",list(AEROS.keys()),index=2, key="altn")
    startup_str=st.text_input("Startup (HH:MM)","", key="startup")

    c4,c5,c6=st.columns(3)
    with c4:
        qnh=st.number_input("QNH (hPa)",900,1050,1013,step=1, key="qnh")
        cruise_alt=st.number_input("Cruise Altitude (ft)",0,14000,4000,step=100, key="cruise_alt")
    with c5:
        temp_c=st.number_input("OAT (°C)",-40,50,15,step=1, key="temp_c")
        var_deg=st.number_input("Mag Variation (°)",0,30,1,step=1, key="var_deg")
        var_is_e=(st.selectbox("E/W",["W","E"],index=0, key="var_is")=="E")
    with c6:
        wind_from_global=st.number_input("Wind FROM (°TRUE)",0,360,0,step=1, key="wind_dir_g")
        wind_kt_global=st.number_input("Wind (kt)",0,120,17,step=1, key="wind_kt_g")

    c7,c8,c9=st.columns(3)
    with c7:
        rpm_climb  = st.number_input("Climb RPM (AFM)",1800,2388,2250,step=10, key="rpm_climb")
        rpm_cruise = st.number_input("Cruise RPM (AFM)",1800,2388,2000,step=10, key="rpm_cruise")
    with c8:
        descent_ff = st.number_input("Descent FF (L/h)", 0.0, 30.0, 15.0, step=0.1, key="descent_ff")
    with c9:
        rod_fpm=st.number_input("ROD (ft/min)",200,1500,700,step=10, key="rod_fpm")
        start_fuel=st.number_input("Fuel inicial (EFOB_START) [L]",0.0,1000.0,85.0,step=0.1, key="start_fuel")

    cruise_ref_kt = st.number_input("Cruise speed (kt)", 40, 140, 90, step=1, key="cruise_ref")
    descent_ref_kt= st.number_input("Descent speed (kt)", 40, 120, 65, step=1, key="descent_ref")

    # ---- Métricas (preview) ----
    def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0
    dep_elev = aero_elev(st.session_state.dept); arr_elev = aero_elev(st.session_state.arr)
    pa_start = pressure_alt(dep_elev, st.session_state.qnh)
    vy_kt = vy_interp_enroute(pa_start)
    roc = roc_interp_enroute(pa_start, st.session_state.temp_c)
    pa_cruise = pressure_alt(st.session_state.cruise_alt, st.session_state.qnh)
    _, ff_climb  = cruise_lookup(dep_elev + 0.5*max(0.0, st.session_state.cruise_alt - dep_elev), int(st.session_state.rpm_climb),  st.session_state.temp_c)
    _, ff_cruise = cruise_lookup(pa_cruise,   int(st.session_state.rpm_cruise), st.session_state.temp_c)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Vy (kt)", _round_unit(vy_kt))
    m2.metric("ROC (ft/min)", _round_unit(roc))
    m3.metric("TAS cruise (kt)", _round_unit(st.session_state.cruise_ref))
    m4.metric("FF cruise (L/h)", _round_unit(ff_cruise))

with tab_plan:
    # -------- Rota & Planeamento dentro de um FORM (para não perder inputs) --------
    if "points" not in st.session_state:
        st.session_state.points = [st.session_state.dept, st.session_state.arr]

    # aplica rota por form
    with st.form("form_route"):
        st.subheader("Rota (DEP … ARR)")
        default_route = f"{st.session_state.dept} {st.session_state.arr}"
        route_text = st.text_area("Pontos (separados por espaço, vírgulas ou '->')",
                                  value=st.session_state.get("route_text", default_route),
                                  key="route_text_box")
        apply_route = st.form_submit_button("Aplicar rota")

    def parse_route_text(txt:str) -> List[str]:
        tokens = re.split(r"[,\s→\-]+", (txt or "").strip())
        return [t for t in tokens if t]

    if apply_route:
        pts = parse_route_text(st.session_state.route_text_box)
        if len(pts) < 2: pts = [st.session_state.dept, st.session_state.arr]
        pts[0] = st.session_state.dept
        pts[-1] = st.session_state.arr
        st.session_state.points = pts
        st.session_state.route_text = " ".join(pts)

    points = st.session_state.points
    if points:
        points[0] = st.session_state.dept
        if len(points)>=2: points[-1] = st.session_state.arr

    # tabela única (linhas = pernas) com ALT_to e vento por waypoint
    def make_default_plan_rows(points: List[str]) -> List[dict]:
        rows=[]
        dep_elev=aero_elev(points[0]); arr_elev=aero_elev(points[-1])
        for i in range(1,len(points)):
            to_is_last = (i == len(points)-1)
            rows.append({
                "From": points[i-1],
                "To": points[i],
                "TC": 0.0,
                "Dist": 0.0,
                "ALT_to_ft": float(arr_elev if to_is_last else st.session_state.cruise_alt),
                "Wind_DIR_to": int(st.session_state.wind_dir_g),
                "Wind_KT_to": int(st.session_state.wind_kt_g),
                "UseNavaid": False,
                "Navaid_IDENT": "",
                "Navaid_FREQ": "",
            })
        return rows

    def preserve_merge(old: List[dict], new_points: List[str]) -> List[dict]:
        idx = {(r.get("From"), r.get("To")): r for r in old}
        merged=[]
        dep_elev=aero_elev(new_points[0]); arr_elev=aero_elev(new_points[-1])
        for i in range(1,len(new_points)):
            key=(new_points[i-1], new_points[i])
            base=idx.get(key, {})
            to_is_last = (i == len(new_points)-1)
            merged.append({
                "From": key[0],
                "To": key[1],
                "TC": float(base.get("TC", 0.0)),
                "Dist": float(base.get("Dist", 0.0)),
                "ALT_to_ft": float(base.get("ALT_to_ft", (arr_elev if to_is_last else st.session_state.cruise_alt))),
                "Wind_DIR_to": int(base.get("Wind_DIR_to", st.session_state.wind_dir_g)),
                "Wind_KT_to": int(base.get("Wind_KT_to", st.session_state.wind_kt_g)),
                "UseNavaid": bool(base.get("UseNavaid", False)),
                "Navaid_IDENT": base.get("Navaid_IDENT",""),
                "Navaid_FREQ": base.get("Navaid_FREQ",""),
            })
        return merged

    if "plan_rows" not in st.session_state:
        st.session_state.plan_rows = make_default_plan_rows(points)
    else:
        current_pairs = [(r["From"], r["To"]) for r in st.session_state.plan_rows]
        new_pairs = [(points[i-1], points[i]) for i in range(1,len(points))]
        if current_pairs != new_pairs:
            st.session_state.plan_rows = preserve_merge(st.session_state.plan_rows, points)

    with st.form("form_plan"):
        st.subheader("Planeamento (edita TC, Dist, ALT_to, Vento (To), Navaid)")
        plan_cfg = {
            "From": st.column_config.TextColumn("From", disabled=True),
            "To":   st.column_config.TextColumn("To", disabled=True),
            "TC":   st.column_config.NumberColumn("TC (°T)", step=0.1, min_value=0.0, max_value=359.9),
            "Dist": st.column_config.NumberColumn("Dist (nm)", step=0.1, min_value=0.0),
            "ALT_to_ft": st.column_config.NumberColumn("ALT alvo no To (ft)", step=5, min_value=0.0),
            "Wind_DIR_to": st.column_config.NumberColumn("Wind FROM To (°T)", step=1, min_value=0, max_value=360),
            "Wind_KT_to":  st.column_config.NumberColumn("Wind To (kt)", step=1, min_value=0, max_value=200),
            "UseNavaid": st.column_config.CheckboxColumn("Usar navaid?"),
            "Navaid_IDENT": st.column_config.TextColumn("Navaid IDENT"),
            "Navaid_FREQ":  st.column_config.TextColumn("Navaid FREQ"),
        }
        plan_edit = st.data_editor(
            st.session_state.plan_rows, key="plan_table",
            hide_index=True, use_container_width=True, num_rows="fixed",
            column_config=plan_cfg
        )
        saved = st.form_submit_button("Guardar planeamento")
        if saved:
            st.session_state.plan_rows = plan_edit
            st.success("Planeamento guardado.")

with tab_results:
    # ------------- Cálculo vertical + cortes (usa plan_rows) -------------
    if "plan_rows" not in st.session_state or len(st.session_state.plan_rows)==0:
        st.info("Define primeiro a rota e o planeamento.")
        st.stop()

    points = st.session_state.points
    legs = st.session_state.plan_rows
    N = len(legs)

    # Perf
    def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0
    dep_elev = aero_elev(points[0]); arr_elev = aero_elev(points[-1])
    start_alt = float(dep_elev); end_alt = float(arr_elev)
    pa_start  = pressure_alt(start_alt, st.session_state.qnh)
    vy_kt = vy_interp_enroute(pa_start)
    roc = roc_interp_enroute(pa_start, st.session_state.temp_c)
    pa_cruise = pressure_alt(st.session_state.cruise_alt, st.session_state.qnh)
    _, ff_climb  = cruise_lookup(start_alt + 0.5*max(0.0, st.session_state.cruise_alt - start_alt), int(st.session_state.rpm_climb),  st.session_state.temp_c)
    _, ff_cruise = cruise_lookup(pa_cruise,   int(st.session_state.rpm_cruise), st.session_state.temp_c)
    ff_descent   = float(st.session_state.descent_ff)
    tas_climb, tas_cruise, tas_descent = vy_kt, float(st.session_state.cruise_ref), float(st.session_state.descent_ref)

    # Arrays por perna (inclui vento por perna)
    dist = [float(legs[i]["Dist"] or 0.0) for i in range(N)]
    tcs  = [float(legs[i]["TC"]   or 0.0) for i in range(N)]
    wind_dir_leg = [int(legs[i].get("Wind_DIR_to", st.session_state.wind_dir_g)) for i in range(N)]
    wind_kt_leg  = [int(legs[i].get("Wind_KT_to",  st.session_state.wind_kt_g))  for i in range(N)]

    def gs_for(i, tas):
        return wind_triangle(float(tcs[i]), float(tas), float(wind_dir_leg[i]), float(wind_kt_leg[i]))[2]
    gs_climb   = [gs_for(i, tas_climb)   for i in range(N)]
    gs_cruise  = [gs_for(i, tas_cruise)  for i in range(N)]
    gs_descent = [gs_for(i, tas_descent) for i in range(N)]

    # Alvos por waypoint (N+1): [DEP] + ALT_to
    alt_targets = [start_alt] + [float(legs[i].get("ALT_to_ft") or (arr_elev if i==N-1 else st.session_state.cruise_alt)) for i in range(N)]

    # Alocação climb/desc (para trás)
    front_used_dist = [0.0]*N   # climb no início
    back_used_dist  = [0.0]*N   # descent no fim
    climb_time_alloc   = [0.0]*N
    descent_time_alloc = [0.0]*N
    impossible_notes = []

    def alloc_front(i, time_min, gs):
        max_time = max(0.0, (dist[i] - front_used_dist[i] - back_used_dist[i]) * 60.0 / max(gs,1e-6))
        use = min(time_min, max_time)
        d = gs * use / 60.0
        front_used_dist[i] += d
        climb_time_alloc[i] += use
        return use

    def alloc_back(i, time_min, gs):
        max_time = max(0.0, (dist[i] - front_used_dist[i] - back_used_dist[i]) * 60.0 / max(gs,1e-6))
        use = min(time_min, max_time)
        d = gs * use / 60.0
        back_used_dist[i] += d
        descent_time_alloc[i] += use
        return use

    for j in range(1, len(alt_targets)):
        delta = alt_targets[j] - alt_targets[j-1]
        if delta > 0:
            need = delta / max(roc,1e-6)
            rem = need
            for k in range(j-1, -1, -1):
                if rem <= 1e-9: break
                rem -= alloc_front(k, rem, max(gs_climb[k],1e-6))
            if rem > 1e-6:
                impossible_notes.append(f"Impossível atingir {int(alt_targets[j])} ft em {legs[j-1]['To']} (climb insuficiente). Falta {rem:.1f} min.")
        elif delta < 0:
            need = (-delta) / max(st.session_state.rod_fpm,1e-6)
            rem = need
            for k in range(j-1, -1, -1):
                if rem <= 1e-9: break
                rem -= alloc_back(k, rem, max(gs_descent[k],1e-6))
            if rem > 1e-6:
                impossible_notes.append(f"Impossível atingir {int(alt_targets[j])} ft em {legs[j-1]['To']} (descent insuficiente). Falta {rem:.1f} min.")

    # --------- Construção de segmentos ---------
    startup = parse_hhmm(st.session_state.startup)
    takeoff = add_minutes(startup,15) if startup else None
    clock = takeoff
    def ceil_pos_minutes(x): return max(1, int(math.ceil(x - 1e-9))) if x > 0 else 0

    rows=[]; seq_points=[]
    calc_rows=[]; calc_details=[]
    PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓"}
    efob=float(st.session_state.start_fuel)

    # DEP
    seq_points.append({"name": points[0], "alt": _round_alt(alt_targets[0]),
                       "tc":"", "th":"", "mc":"", "mh":"", "tas":"", "gs":"", "dist":"",
                       "ete":"", "eto": (takeoff.strftime("%H:%M") if takeoff else ""),
                       "burn":"", "efob": float(st.session_state.start_fuel)})

    def add_seg(phase, frm, to, i_leg, d_nm, tas, ff_lph, alt_start_ft, rate_fpm):
        nonlocal clock, efob
        if d_nm <= 1e-9: return alt_start_ft
        tc = float(tcs[i_leg])
        wca, th, gs = wind_triangle(tc, tas, float(wind_dir_leg[i_leg]), float(wind_kt_leg[i_leg]))
        mc = apply_var(tc, st.session_state.var_deg, var_is_e)
        mh = apply_var(th, st.session_state.var_deg, var_is_e)

        ete_raw = 60.0 * d_nm / max(gs,1e-6); ete = ceil_pos_minutes(ete_raw)
        burn_raw = ff_lph * (ete_raw/60.0); burn = _round_half(burn_raw)

        if phase == "CLIMB":   alt_end_ft = alt_start_ft + rate_fpm * ete_raw
        elif phase == "DESCENT": alt_end_ft = alt_start_ft - rate_fpm * ete_raw
        else: alt_end_ft = alt_start_ft

        eto = ""
        if clock: clock = add_minutes(clock, ete); eto = clock.strftime("%H:%M")
        efob = max(0.0, _round_half(efob - burn_raw))

        rows.append({
            "Fase": PH_ICON[phase], "Leg/Marker": f"{frm}→{to}",
            "ALT (ft)": f"{fmt(alt_start_ft,'alt')}→{fmt(alt_end_ft,'alt')}",
            "TC (°T)": _round_angle(tc), "TH (°T)": _round_angle(th),
            "MC (°M)": _round_angle(mc), "MH (°M)": _round_angle(mh),
            "TAS (kt)": _round_unit(tas), "GS (kt)": _round_unit(gs),
            "FF (L/h)": _round_unit(ff_lph),
            "Dist (nm)": fmt(d_nm,'dist'), "ETE (min)": int(ete), "ETO": eto,
            "Burn (L)": fmt(burn,'fuel'), "EFOB (L)": fmt(efob,'fuel')
        })

        seq_points.append({
            "name": to, "alt": _round_alt(alt_end_ft),
            "tc": _round_angle(tc), "th": _round_angle(th),
            "mc": _round_angle(mc), "mh": _round_angle(mh),
            "tas": _round_unit(tas), "gs": _round_unit(gs),
            "dist": float(f"{d_nm:.3f}"),
            "ete": int(ete), "eto": eto,
            "burn": float(burn_raw), "efob": float(efob)
        })

        calc_rows.append([f"{frm}→{to}", phase,
                          f"{_round_angle(tc)}°", f"{_round_angle(mc)}°", f"{_round_angle(th)}°", f"{_round_angle(mh)}°",
                          _round_unit(tas), _round_unit(gs),
                          fmt(d_nm,'dist'), int(ete), eto or "—",
                          fmt(burn,'fuel'), fmt(efob,'fuel'),
                          f"{fmt(alt_start_ft,'alt')}→{fmt(alt_end_ft,'alt')}"])
        delta = angle_diff(float(wind_dir_leg[i_leg]), tc)

        # Explicação detalhada + arredondamentos usados
        calc_details.append(
            "• {src}->{dst} [{ph}] "
            "TC={tc:.1f}°, Var={var:.1f}{EW} ⇒ MC={mc:.1f}°; "
            "Δ(wind−TC)={dl:.1f}°; WCA=asin((W/TAS)·sinΔ)={wca:.2f}° ⇒ "
            "TH=TC+WCA={th:.2f}° ⇒ MH=TH±Var={mh:.2f}°; "
            "GS_raw=TAS·cos(WCA)−W·cosΔ={gsr:.2f} kt ⇒ GS={gs_rounded} kt; "
            "ETE_raw=60·D/GS={eter:.2f} min ⇒ ETE={ete} min (ceil); "
            "Burn_raw=FF·(ETE_raw/60)={br:.2f} L ⇒ Burn={burn:.1f} L (0.5L step); "
            "ALT {h0:.0f}→{h1raw:.0f} ft ⇒ {h1r} ft (alt rounding)."
            .format(src=frm, dst=to, ph=phase, tc=tc, var=st.session_state.var_deg, EW=("E" if var_is_e else "W"),
                    mc=mc, dl=delta, wca=wca, th=th, mh=mh, gsr=gs, gs_rounded=_round_unit(gs),
                    d=d_nm, eter=ete_raw, ete=ete, br=burn_raw, burn=_round_half(burn_raw),
                    h0=alt_start_ft, h1raw=alt_end_ft, h1r=fmt(alt_end_ft,'alt'))
        )
        return alt_end_ft

    # construir perna a perna
    cur_alt = alt_targets[0]
    for i in range(N):
        frm, to = legs[i]["From"], legs[i]["To"]
        d_total  = dist[i]
        d_cl = front_used_dist[i]
        d_ds = back_used_dist[i]
        d_cr = max(0.0, d_total - d_cl - d_ds)

        if d_cl > 1e-9:
            toc_name = "TOC" if (d_cr + d_ds) > 0 else to
            cur_alt = add_seg("CLIMB", frm, toc_name, i, d_cl, vy_kt, ff_climb, cur_alt, roc)
            frm = toc_name
        if d_cr > 1e-9:
            tod_name = "TOD" if d_ds > 1e-9 else to
            cur_alt = add_seg("CRUISE", frm, tod_name, i, d_cr, float(st.session_state.cruise_ref), ff_cruise, cur_alt, 0.0)
            frm = tod_name
        if d_ds > 1e-9:
            cur_alt = add_seg("DESCENT", frm, to, i, d_ds, float(st.session_state.descent_ref), ff_descent, cur_alt, st.session_state.rod_fpm)

    eta = clock
    shutdown = add_minutes(eta,5) if eta else None

    # ---------- Tabela & Totais ----------
    st.subheader("Segmentos calculados")
    st.dataframe(rows, use_container_width=True)

    tot_ete_m = int(sum(int(r['ETE (min)']) for r in rows))
    tot_nm  = sum(float(p['dist']) for p in seq_points if isinstance(p.get('dist'), (int,float)))
    tot_bo_raw = sum(float(p['burn']) for p in seq_points if isinstance(p.get('burn'), (int,float)))
    tot_bo = _round_half(tot_bo_raw)
    line = f"**Totais** — Dist {fmt(tot_nm,'dist')} nm • ETE {tot_ete_m//60:02d}:{tot_ete_m%60:02d} • Burn {fmt(tot_bo,'fuel')} L • EFOB {fmt(seq_points[-1]['efob'] if seq_points else st.session_state.start_fuel,'fuel')} L"
    if eta: line += f" • **ETA {eta.strftime('%H:%M')}** • **Shutdown {shutdown.strftime('%H:%M')}**"
    st.markdown(line)
    if impossible_notes:
        st.warning(" / ".join(impossible_notes))

    # ---------- Export PDF ----------
    st.subheader("Exportar para PDF (NAVLOG_FORM.pdf)")
    try:
        template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)
        fieldset, maxlens = get_form_fields(template_bytes) if PYPDF_OK else (set(), {})
    except Exception as e:
        template_bytes=None; fieldset=set(); maxlens={}
        st.error(f"Não foi possível ler o PDF: {e}")

    named: Dict[str,str] = {}
    def put_h(k, v):
        if fieldset:
            put(named, fieldset, k, v, maxlens)

    if fieldset:
        etd = (add_minutes(parse_hhmm(st.session_state.startup),15).strftime("%H:%M") if st.session_state.startup else "")
        put_h("AIRCRAFT", st.session_state.aircraft)
        put_h("REGISTRATION", st.session_state.registration)
        put_h("CALLSIGN", st.session_state.callsign)
        put_h("ETD/ETA", f"{etd} / {eta.strftime('%H:%M') if eta else ''}")
        put_h("STARTUP", st.session_state.startup)
        put_h("TAKEOFF", etd)
        put_h("LANDING", eta.strftime("%H:%M") if eta else "")
        put_h("SHUTDOWN", shutdown.strftime("%H:%M") if shutdown else "")
        put_h("LESSON", st.session_state.lesson)
        put_h("INSTRUTOR", st.session_state.instrutor)
        put_h("STUDENT", st.session_state.student)
        put_h("FLT TIME", f"{tot_ete_m//60:02d}:{tot_ete_m%60:02d}")
        put_h("LEVEL F/F", fmt(st.session_state.cruise_alt,'alt'))
        put_h("CLIMB FUEL", fmt((sum([t for t in climb_time_alloc])/60.0*ff_climb),'fuel'))
        put_h("QNH", fmt(st.session_state.qnh,'alt'))
        put_h("DEPT", aero_freq(points[0]))
        put_h("ENROUTE", "123.755")
        put_h("ARRIVAL", aero_freq(points[-1]))
        put_h("CLEARANCES", "")
        put_h("Departure_Airfield", points[0])
        put_h("Arrival_Airfield", points[-1])
        put_h("Alternate_Airfield", st.session_state.altn)
        # ---- Altitude do alternante (tolerância a vários nomes) ----
        for k_alt in ["Alt_Alternate","ALTN_ALT","Alternate_Altitude","ALTERNATE ALT","ALT ALTN"]:
            put_h(k_alt, str(_round_alt(aero_elev(st.session_state.altn))))

        put_h("Leg_Number", str(len(seq_points)))
        for fld in ["FLIGHT LEVEL / ALTITUDE","FLIGHT LEVEL","LEVEL / ALTITUDE"]:
            put_h(fld, fmt(st.session_state.cruise_alt,'alt'))
        put_h("WIND", f"{int(round(st.session_state.wind_dir_g)):03d}/{int(round(st.session_state.wind_kt_g)):02d}")
        put_h("MAG  VAR", f"{int(round(st.session_state.var_deg))}{'E' if var_is_e else 'W'}")
        put_h("TEMP / ISA DEV", f"{fmt(st.session_state.temp_c,'alt')} / {fmt(st.session_state.temp_c - isa_temp(pressure_alt(aero_elev(points[0]), st.session_state.qnh)),'alt')}")

        # Linhas LegNN_*
        acc_dist = 0.0
        acc_time = 0
        for idx, p in enumerate(seq_points, start=1):
            tag = f"Leg{idx:02d}_"
            is_seg = (idx>1)
            put_h(tag+"Waypoint", p["name"])
            # navaid do To da perna anterior (idx-2)
            if is_seg:
                leg = legs[idx-2]
                if leg.get("UseNavaid", False):
                    put_h(tag+"Navaid_Identifier", leg.get("Navaid_IDENT",""))
                    put_h(tag+"Navaid_Frequency",  leg.get("Navaid_FREQ",""))
            put_h(tag+"Altitude_FL", (fmt(p["alt"], 'alt') if p["alt"]!="" else ""))

            if is_seg:
                acc_dist += float(p["dist"] or 0.0)
                acc_time += int(p["ete"] or 0)
                put_h(tag+"True_Course",      fmt(p["tc"], 'angle'))
                put_h(tag+"Magnetic_Course",  fmt(p["mc"], 'angle'))
                put_h(tag+"Ground_Speed",     fmt(p["gs"], 'speed'))
                put_h(tag+"Leg_Distance",     fmt(p["dist"], 'dist'))
                put_h(tag+"Leg_ETE",          fmt(p["ete"], 'mins'))
                put_h(tag+"ETO",              p["eto"])
                put_h(tag+"Planned_Burnoff",  fmt(p["burn"], 'fuel'))
                put_h(tag+"Estimated_FOB",    fmt(p["efob"], 'fuel'))
                put_h(tag+"True_Heading",     fmt(p["th"], 'angle'))
                put_h(tag+"Magnetic_Heading", fmt(p["mh"], 'angle'))
                put_h(tag+"True_Airspeed",    fmt(p["tas"], 'speed'))
                put_h(tag+"Cumulative_Distance", fmt(acc_dist,'dist'))
                put_h(tag+"Cumulative_ETE",      fmt(acc_time,'mins'))
            else:
                put_h(tag+"ETO", p["eto"])
                put_h(tag+"Estimated_FOB", fmt(p["efob"], 'fuel'))

        # Botão PDF
        if st.button("Gerar PDF NAVLOG (planeado)", type="primary"):
            try:
                out = fill_pdf(template_bytes, named)
                m = re.search(r'(\d+)', st.session_state.lesson or "")
                lesson_num = m.group(1) if m else "00"
                safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
                filename = f"{safe_date}_LESSON-{lesson_num}_NAVLOG.pdf"
                st.download_button("📄 Download PDF", data=out, file_name=filename, mime="application/pdf")
                st.success("PDF gerado (planeado).")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

    # -------- Relatório --------
    st.subheader("Relatório (PDF legível)")
    def build_report_pdf(calc_rows: List[List], details: List[str], params: Dict[str,str]) -> bytes:
        if not REPORTLAB_OK: raise RuntimeError("reportlab missing")
        bio = io.BytesIO()
        doc = SimpleDocTemplate(bio, pagesize=A4,
                                leftMargin=16*mm, rightMargin=16*mm,
                                topMargin=14*mm, bottomMargin=14*mm)
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
            ["Partida", params.get("dept","—")],
            ["Chegada", params.get("arr","—")],
            ["Alternante (ICAO / ALT)", params.get("altn","—")+" / "+params.get("altn_alt","—")+" ft"],
            ["Cruise Alt", params.get("cruise_alt","—")+" ft"],
            ["QNH", params.get("qnh","—")],
            ["Vento (global)", params.get("wind","—")],
            ["Var. Magn.", params.get("var","—")],
            ["OAT / ISA dev", params.get("temp_isa","—")],
            ["Startup / ETD", f"{params.get('startup','—')} / {params.get('etd','—')}"],
            ["ETA / Shutdown", f"{params.get('eta','—')} / {params.get('shutdown','—')}"],
            ["Tempo total (PLN)", params.get("flt_time","—")],
            ["Fuel inicial", params.get("start_fuel","—")+" L"],
            ["Arredondamentos", params.get("rounding","—")],
            ["Notas", params.get("notes","—")],
        ]
        t1 = LongTable(resume, colWidths=[60*mm, None], repeatRows=0, hAlign="LEFT")
        t1.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
            ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("FONTSIZE",(0,0),(-1,-1),8.2),
        ]))
        story.append(t1)
        story.append(Spacer(1,6))

        story.append(Paragraph("Segmentos (resumo)", H2))
        data = [["From→To","Fase","TC°","MC°","TH°","MH°","TAS","GS","Dist(nm)","ETE","ETO","Burn(L)","EFOB(L)","ALT ini→fim"]]
        data += calc_rows
        t3 = LongTable(data,
                       colWidths=[40*mm, 12*mm, 10*mm, 10*mm, 10*mm, 10*mm, 12*mm, 12*mm, 18*mm, 12*mm, 20*mm, 14*mm, 14*mm, 32*mm],
                       repeatRows=1, hAlign="LEFT")
        t3.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("ALIGN",(2,1),(7,-1),"RIGHT"),
            ("ALIGN",(8,1),(12,-1),"RIGHT"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("FONTSIZE",(0,0),(-1,-1),7.9),
        ]))
        story.append(t3)
        story.append(PageBreak())

        story.append(Paragraph("Cálculos por segmento (passo-a-passo + aproximações)", H2))
        for s in details:
            story.append(Paragraph(s, P))

        doc.build(story)
        return bio.getvalue()

    if st.button("Gerar Relatório (PDF legível)"):
        try:
            params = {
                "aircraft": st.session_state.aircraft, "registration": st.session_state.registration, "callsign": st.session_state.callsign,
                "lesson": st.session_state.lesson, "dept": points[0], "arr": points[-1], "altn": st.session_state.altn,
                "altn_alt": str(_round_alt(aero_elev(st.session_state.altn))),
                "qnh": fmt(st.session_state.qnh,'alt'),
                "wind": f"{int(round(st.session_state.wind_dir_g)):03d}/{int(round(st.session_state.wind_kt_g)):02d}",
                "var": f"{int(round(st.session_state.var_deg))}{'E' if var_is_e else 'W'}",
                "cruise_alt": fmt(st.session_state.cruise_alt,'alt'),
                "temp_isa": f"{fmt(st.session_state.temp_c,'alt')} / {fmt(st.session_state.temp_c - isa_temp(pressure_alt(aero_elev(points[0]), st.session_state.qnh)),'alt')}",
                "startup": st.session_state.startup,
                "etd": (add_minutes(parse_hhmm(st.session_state.startup),15).strftime("%H:%M") if st.session_state.startup else ""),
                "eta": (eta.strftime("%H:%M") if eta else ""), "shutdown": (shutdown.strftime("%H:%M") if shutdown else ""),
                "flt_time": f"{tot_ete_m//60:02d}:{tot_ete_m%60:02d}",
                "start_fuel": fmt(st.session_state.start_fuel,'fuel'),
                "rounding": "Dist 0.1 nm; Tempo min inteiro (ceil); Fuel 0.5 L; Vel/TAS/GS 1 kt; Ângulos 1°; Alt <1000→5 / ≥1000→100.",
                "notes": " ; ".join(impossible_notes) if impossible_notes else "—",
            }
            report_bytes_out = build_report_pdf(calc_rows, calc_details, params)
            m = re.search(r'(\d+)', st.session_state.lesson or "")
            lesson_num = m.group(1) if m else "00"
            safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
            filename_rep = f"{safe_date}_LESSON-{lesson_num}_NAVLOG_RELATORIO.pdf"
            st.download_button("📑 Download Relatório (PDF)", data=report_bytes_out, file_name=filename_rep, mime="application/pdf")
            st.success("Relatório gerado.")
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {e}")

with tab_io:
    st.subheader("Exportar / Importar rota")
    # snapshot
    def snapshot() -> dict:
        return {
            "meta": {
                "aircraft": st.session_state.aircraft,
                "registration": st.session_state.registration,
                "callsign": st.session_state.callsign,
                "student": st.session_state.student,
                "lesson": st.session_state.lesson,
                "instrutor": st.session_state.instrutor,
                "dept": st.session_state.dept,
                "arr": st.session_state.arr,
                "altn": st.session_state.altn,
                "startup": st.session_state.startup,
                "qnh": st.session_state.qnh,
                "cruise_alt": st.session_state.cruise_alt,
                "temp_c": st.session_state.temp_c,
                "var_deg": st.session_state.var_deg,
                "var_is_e": (st.session_state.var_is == "E") if isinstance(st.session_state.var_is, str) else False,
                "wind_global": {"dir": st.session_state.wind_dir_g, "kt": st.session_state.wind_kt_g},
                "perf": {
                    "rpm_climb": st.session_state.rpm_climb,
                    "rpm_cruise": st.session_state.rpm_cruise,
                    "descent_ff": st.session_state.descent_ff,
                    "rod_fpm": st.session_state.rod_fpm,
                    "cruise_ref_kt": st.session_state.cruise_ref,
                    "descent_ref_kt": st.session_state.descent_ref,
                    "start_fuel": st.session_state.start_fuel,
                }
            },
            "points": st.session_state.points,
            "legs": st.session_state.plan_rows
        }
    st.download_button("💾 Download rota (JSON)",
                       data=json.dumps(snapshot(), ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"route_{ascii_safe(st.session_state.lesson or 'lesson')}.json",
                       mime="application/json")

    uploaded = st.file_uploader("📤 Seleciona rota (JSON)", type=["json"])
    if st.button("Usar rota do ficheiro") and uploaded is not None:
        try:
            data = json.loads(uploaded.read().decode("utf-8"))
            st.session_state.points = list(data.get("points") or [st.session_state.dept, st.session_state.arr])
            st.session_state.plan_rows = list(data.get("legs") or [])
            meta = data.get("meta") or {}
            # opcional: repor alguns meta
            for k in ("aircraft","registration","callsign","student","lesson","instrutor",
                      "dept","arr","altn","startup","qnh","cruise_alt","temp_c","var_deg"):
                if k in meta: st.session_state[k] = meta[k]
            if "wind_global" in meta:
                st.session_state.wind_dir_g = meta["wind_global"].get("dir", st.session_state.wind_dir_g)
                st.session_state.wind_kt_g  = meta["wind_global"].get("kt",  st.session_state.wind_kt_g)
            if "perf" in meta:
                pf = meta["perf"]
                for k in ("rpm_climb","rpm_cruise","descent_ff","rod_fpm","cruise_ref_kt","descent_ref_kt","start_fuel"):
                    if k in pf: st.session_state[k if k!="cruise_ref_kt" and k!="descent_ref_kt" else ("cruise_ref" if "cruise" in k else "descent_ref")] = pf[k]
            st.success("Rota carregada do JSON.")
        except Exception as e:
            st.error(f"Falha a carregar JSON: {e}")
