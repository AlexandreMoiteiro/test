# app.py — NAVLOG (PDF + Relatório) com:
# - altitude alvo por waypoint (default = cruise), com distribuição de CLIMB/DESCENT e TOC/TOD no sítio certo
# - NAVAIDs opcionais por waypoint (apenas preenche quando marcado)
# - arredondamentos: Dist 0.1 nm; Tempo em min (ceil, min 1); Fuel 0.5 L; TAS/GS/ângulos à unidade; Alt <1000→5, ≥1000→100
# - relatório com LongTable (não corta) e cabeçalho repetido
#
# Reqs: streamlit, pypdf, reportlab, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod

# ---------------- UI & Template ----------------
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
    v = abs(float(x))
    base = 5 if v < 1000 else 100
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
    field_names, maxlens = set(), {}
    try:
        fd = reader.get_fields() or {}
        field_names |= set(fd.keys())
        for k,v in fd.items():
            ml = v.get("/MaxLen")
            if ml: maxlens[k] = int(ml)
    except: pass
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
st.title("Navigation Plan & Inflight Log — Tecnam P2008 (PDF + Relatório)")

DEFAULT_STUDENT="AMOIT"; DEFAULT_AIRCRAFT="P208"; DEFAULT_CALLSIGN="RVP"
REGS=["CS-ECC","CS-ECD","CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW"]

# Header
c1,c2,c3=st.columns(3)
with c1:
    aircraft=st.text_input("Aircraft",DEFAULT_AIRCRAFT)
    registration=st.selectbox("Registration",REGS,index=0)
    callsign=st.text_input("Callsign",DEFAULT_CALLSIGN)
with c2:
    student=st.text_input("Student",DEFAULT_STUDENT)
    lesson = st.text_input("Lesson (ex: 12)", "")
    instrutor = st.text_input("Instrutor","")
with c3:
    dept=st.selectbox("Departure",list(AEROS.keys()),index=0)
    arr =st.selectbox("Arrival", list(AEROS.keys()),index=1)
    altn=st.selectbox("Alternate",list(AEROS.keys()),index=2)
startup_str=st.text_input("Startup (HH:MM)","")

# Atmosfera / navegação
c4,c5,c6=st.columns(3)
with c4:
    qnh=st.number_input("QNH (hPa)",900,1050,1013,step=1)
    cruise_alt=st.number_input("Cruise Altitude (ft)",0,14000,4000,step=100)
with c5:
    temp_c=st.number_input("OAT (°C)",-40,50,15,step=1)
    var_deg=st.number_input("Mag Variation (°)",0,30,1,step=1)
    var_is_e=(st.selectbox("E/W",["W","E"],index=0)=="E")
with c6:
    wind_from=st.number_input("Wind FROM (°TRUE)",0,360,0,step=1)
    wind_kt=st.number_input("Wind (kt)",0,120,17,step=1)

# Perf / consumos
c7,c8,c9=st.columns(3)
with c7:
    rpm_climb  = st.number_input("Climb RPM (AFM)",1800,2388,2250,step=10)
    rpm_cruise = st.number_input("Cruise RPM (AFM)",1800,2388,2000,step=10)
with c8:
    idle_mode  = st.checkbox("Descent at idle", value=False)
    descent_ff = st.number_input("Descent FF (L/h)", 0.0, 30.0, 15.0, step=0.1)
with c9:
    rod_fpm=st.number_input("ROD (ft/min)",200,1500,700,step=10)
    idle_ff=st.number_input("Idle FF (L/h)", 0.0, 20.0, 5.0, step=0.1)
    start_fuel=st.number_input("Fuel inicial (EFOB_START) [L]",0.0,1000.0,85.0,step=0.1)

# Velocidades ref
cruise_ref_kt = st.number_input("Cruise speed (kt)", 40, 140, 90, step=1)
descent_ref_kt= st.number_input("Descent speed (kt)", 40, 120, 65, step=1)

# ===== ROUTE =====
def parse_route_text(txt:str) -> List[str]:
    tokens = re.split(r"[,\s→\-]+", (txt or "").strip())
    return [t for t in tokens if t]

st.markdown("### Route (DEP … ARR)")
default_route = f"{dept} {arr}"
route_text = st.text_area("Pontos (separados por espaço, vírgulas ou '->')",
                          value=st.session_state.get("route_text", default_route))
apply_route = st.button("Aplicar rota")

if "points" not in st.session_state:
    st.session_state.points = [dept, arr]
if apply_route:
    pts = parse_route_text(route_text)
    if len(pts) < 2: pts = [dept, arr]
    st.session_state.points = pts
    st.session_state.route_text = " ".join(pts)
points = st.session_state.points
if points: points[0]=dept
if len(points)>=2: points[-1]=arr

# LEGS (entrada de TC/Dist)
def blank_leg(): return {"From":"","To":"","TC":0.0,"Dist":0.0}
if "legs" not in st.session_state: st.session_state.legs = []
target_legs = max(0, len(points)-1)
legs = st.session_state.legs
if len(legs) < target_legs: legs += [blank_leg() for _ in range(target_legs - len(legs))]
elif len(legs) > target_legs: legs = legs[:target_legs]
for i in range(target_legs):
    legs[i]["From"]=points[i]; legs[i]["To"]=points[i+1]
st.session_state.legs = legs

st.markdown("### Legs (distância do ponto anterior)")
legs_cfg = {
    "From": st.column_config.TextColumn("From", disabled=True),
    "To":   st.column_config.TextColumn("To", disabled=True),
    "TC":   st.column_config.NumberColumn("TC (°T)", step=0.1, min_value=0.0, max_value=359.9),
    "Dist": st.column_config.NumberColumn("Dist (nm)", step=0.1, min_value=0.0),
}
legs_view = st.data_editor(legs, hide_index=True, use_container_width=True,
                           column_config=legs_cfg, num_rows="fixed", key="legs_table")
for i,row in enumerate(legs_view):
    legs[i]["TC"]  = float(row.get("TC") or 0.0)
    legs[i]["Dist"]= float(row.get("Dist") or 0.0)
N = len(legs)

# ===== Alvos de altitude por waypoint =====
st.markdown("### Altitude alvo por Waypoint (default = cruise)")
if "alt_targets" not in st.session_state:
    # 1 por waypoint
    st.session_state.alt_targets = [{"Waypoint": points[i], "ALT_target_ft": None} for i in range(len(points))]
# actualiza nomes
alt_rows = st.session_state.alt_targets
if len(alt_rows) != len(points):
    alt_rows = [{"Waypoint": points[i], "ALT_target_ft": (alt_rows[i]["ALT_target_ft"] if i < len(alt_rows) else None)}
                for i in range(len(points))]
    st.session_state.alt_targets = alt_rows
for i in range(len(points)):
    alt_rows[i]["Waypoint"] = points[i]
alt_view = st.data_editor(
    alt_rows, hide_index=True, use_container_width=True,
    column_config={
        "Waypoint": st.column_config.TextColumn("Waypoint", disabled=True),
        "ALT_target_ft": st.column_config.NumberColumn("ALT alvo (ft) — deixar vazio = cruise", step=5, min_value=0),
    },
    key="alt_table"
)
st.session_state.alt_targets = alt_view

# ===== NAVAIDs opcionais por waypoint =====
st.markdown("### NAVAIDs (opcional por waypoint)")
if "navaids" not in st.session_state:
    st.session_state.navaids = [{"Waypoint": points[i], "Use": False, "IDENT":"", "FREQ":""} for i in range(len(points))]
nav_rows = st.session_state.navaids
if len(nav_rows) != len(points):
    nav_rows = [{"Waypoint": points[i],
                 "Use": (nav_rows[i]["Use"] if i < len(nav_rows) else False),
                 "IDENT": (nav_rows[i]["IDENT"] if i < len(nav_rows) else ""),
                 "FREQ": (nav_rows[i]["FREQ"] if i < len(nav_rows) else "")}
                for i in range(len(points))]
for i in range(len(points)):
    nav_rows[i]["Waypoint"] = points[i]
nav_view = st.data_editor(
    nav_rows, hide_index=True, use_container_width=True,
    column_config={
        "Waypoint": st.column_config.TextColumn("Waypoint", disabled=True),
        "Use": st.column_config.CheckboxColumn("Usar navaid?"),
        "IDENT": st.column_config.TextColumn("IDENT"),
        "FREQ": st.column_config.TextColumn("FREQ"),
    },
    key="nav_table"
)
st.session_state.navaids = nav_view

# ---------------- Cálculo vertical com alvos por waypoint ----------------
def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0
dep_elev = aero_elev(dept); arr_elev = aero_elev(arr)

start_alt = float(dep_elev)
end_alt   = float(arr_elev)

pa_start  = pressure_alt(start_alt, qnh)
pa_cruise = pressure_alt(cruise_alt, qnh)
vy_kt = vy_interp_enroute(pa_start)
tas_climb, tas_cruise, tas_descent = vy_kt, float(cruise_ref_kt), float(descent_ref_kt)

roc = roc_interp_enroute(pa_start, temp_c)                 # ft/min
t_climb_total = max(0.0, (cruise_alt - start_alt) / max(roc,1e-6))
t_desc_total  = max(0.0, (cruise_alt - end_alt)  / max(rod_fpm,1e-6))

pa_mid_climb = start_alt + 0.5*max(0.0, cruise_alt - start_alt)
pa_mid_desc  = end_alt   + 0.5*max(0.0, cruise_alt - end_alt)
_, ff_climb  = cruise_lookup(pa_mid_climb, int(rpm_climb),  temp_c)
_, ff_cruise = cruise_lookup(pa_cruise,   int(rpm_cruise),  temp_c)
ff_descent   = float(idle_ff) if idle_mode else float(descent_ff)

def gs_for(tc, tas): return wind_triangle(float(tc), float(tas), wind_from, wind_kt)[2]
dist = [float(l["Dist"] or 0.0) for l in legs]
gs_climb   = [gs_for(legs[i]["TC"], tas_climb)   for i in range(N)]
gs_cruise  = [gs_for(legs[i]["TC"], tas_cruise)  for i in range(N)]
gs_descent = [gs_for(legs[i]["TC"], tas_descent) for i in range(N)]

# Alvos de altitude (por waypoint)
alt_targets: List[Optional[float]] = []
for i in range(len(points)):
    v = st.session_state.alt_targets[i].get("ALT_target_ft")
    if v is None or v == "":
        v = cruise_alt if (i>0 and i<len(points)-1) else (start_alt if i==0 else end_alt)
    alt_targets.append(float(v))

# Distribuição backward de CLIMB/DESCENT para cumprir alvos
front_used_dist = [0.0]*N   # usado no início de cada perna (climb)
back_used_dist  = [0.0]*N   # usado no fim de cada perna (descent)
climb_time_alloc   = [0.0]*N
descent_time_alloc = [0.0]*N

impossible_notes = []

def alloc_on_leg_front(i, time_min, gs):
    """Aloca 'time_min' de subida no início da perna i, respeitando distância remanescente."""
    max_time_by_dist = max(0.0, (dist[i] - front_used_dist[i] - back_used_dist[i]) * 60.0 / max(gs,1e-6))
    use = min(time_min, max_time_by_dist)
    d = gs * use / 60.0
    front_used_dist[i] += d
    climb_time_alloc[i] += use
    return use

def alloc_on_leg_back(i, time_min, gs):
    """Aloca 'time_min' de descida no fim da perna i, respeitando distância remanescente."""
    max_time_by_dist = max(0.0, (dist[i] - front_used_dist[i] - back_used_dist[i]) * 60.0 / max(gs,1e-6))
    use = min(time_min, max_time_by_dist)
    d = gs * use / 60.0
    back_used_dist[i] += d
    descent_time_alloc[i] += use
    return use

# Para cada waypoint j (1..M-1), assegurar altitude alvo em j
for j in range(1, len(points)):
    delta = alt_targets[j] - alt_targets[j-1]
    if delta > 0:  # necessidade de subir até j
        need = delta / max(roc,1e-6)  # minutos
        rem = need
        for k in range(j-1, -1, -1):  # aloca para trás
            if rem <= 1e-9: break
            rem -= alloc_on_leg_front(k, rem, max(gs_climb[k],1e-6))
        if rem > 1e-6:
            impossible_notes.append(f"Impossível atingir {int(alt_targets[j])} ft em {points[j]} (climb insuficiente). Falta {rem:.1f} min.")
    elif delta < 0:  # necessidade de descer até j
        need = (-delta) / max(rod_fpm,1e-6)
        rem = need
        for k in range(j-1, -1, -1):  # aloca para trás (fim de cada perna)
            if rem <= 1e-9: break
            rem -= alloc_on_leg_back(k, rem, max(gs_descent[k],1e-6))
        if rem > 1e-6:
            impossible_notes.append(f"Impossível atingir {int(alt_targets[j])} ft em {points[j]} (descent insuficiente). Falta {rem:.1f} min.")

# Construção dos segmentos com TOC/TOD no local certo
startup = parse_hhmm(startup_str)
takeoff = add_minutes(startup,15) if startup else None
clock = takeoff
def ceil_pos_minutes(x):
    return max(1, int(math.ceil(x - 1e-9))) if x > 0 else 0

rows=[]; seq_points=[]
calc_rows=[]; calc_details=[]
PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓"}
efob=float(start_fuel)

# DEP (linha 1 no PDF)
seq_points.append({"name": points[0], "alt": _round_alt(alt_targets[0]),
                   "tc":"", "th":"", "mc":"", "mh":"", "tas":"", "gs":"", "dist":"",
                   "ete":"", "eto": (takeoff.strftime("%H:%M") if takeoff else ""),
                   "burn":"", "efob": float(start_fuel)})

def add_seg(phase, from_nm, to_nm, i_leg, d_nm, tas, ff_lph, alt_start_ft, rate_fpm):
    """Cria segmento, avança relógio/burn/alt e adiciona pontos finais (to_nm)"""
    global clock, efob
    if d_nm <= 1e-9: return alt_start_ft
    tc = float(legs[i_leg]["TC"])
    wca, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
    mc = apply_var(tc, var_deg, var_is_e)
    mh = apply_var(th, var_deg, var_is_e)

    ete_raw = 60.0 * d_nm / max(gs,1e-6)
    ete = ceil_pos_minutes(ete_raw)
    burn_raw = ff_lph * (ete_raw/60.0)
    burn = _round_half(burn_raw)

    # altitude no fim do segmento
    if phase == "CLIMB":
        alt_end_ft = alt_start_ft + rate_fpm * ete_raw
    elif phase == "DESCENT":
        alt_end_ft = alt_start_ft - rate_fpm * ete_raw
    else:
        alt_end_ft = alt_start_ft

    eto = ""
    if clock:
        clock = add_minutes(clock, ete); eto = clock.strftime("%H:%M")
    efob = max(0.0, _round_half(efob - burn_raw))

    rows.append({
        "Fase": PH_ICON[phase], "Leg/Marker": f"{from_nm}→{to_nm}",
        "ALT (ft)": f"{fmt(alt_start_ft,'alt')}→{fmt(alt_end_ft,'alt')}",
        "TC (°T)": _round_angle(tc), "TH (°T)": _round_angle(th),
        "MC (°M)": _round_angle(mc), "MH (°M)": _round_angle(mh),
        "TAS (kt)": _round_unit(tas), "GS (kt)": _round_unit(gs),
        "FF (L/h)": _round_unit(ff_lph),
        "Dist (nm)": fmt(d_nm,'dist'), "ETE (min)": int(ete), "ETO": eto,
        "Burn (L)": fmt(burn,'fuel'), "EFOB (L)": fmt(efob,'fuel')
    })

    seq_points.append({
        "name": to_nm, "alt": _round_alt(alt_end_ft),
        "tc": _round_angle(tc), "th": _round_angle(th),
        "mc": _round_angle(mc), "mh": _round_angle(mh),
        "tas": _round_unit(tas), "gs": _round_unit(gs),
        "dist": float(f"{d_nm:.3f}"),
        "ete": int(ete), "eto": eto,
        "burn": float(burn_raw), "efob": float(efob)
    })

    calc_rows.append([f"{from_nm}→{to_nm}", phase,
                      f"{_round_angle(tc)}°", f"{_round_angle(mc)}°", f"{_round_angle(th)}°", f"{_round_angle(mh)}°",
                      _round_unit(tas), _round_unit(gs),
                      fmt(d_nm,'dist'), int(ete), eto or "—",
                      fmt(burn,'fuel'), fmt(efob,'fuel'),
                      f"{fmt(alt_start_ft,'alt')}→{fmt(alt_end_ft,'alt')}"])

    delta = angle_diff(wind_from, tc)
    calc_details.append(
        "• {src}->{dst} [{ph}] TC={tc:.1f}°, Var={var:.1f}{EW} → MC={mc:.1f}°; "
        "Δ(wind−TC)={dl:.1f}°; WCA={wca:.2f}°; TH={th:.2f}° → MH={mh:.2f}°; "
        "GS={gs:.2f} kt; Dist={d:.2f} nm; ETE_raw={eter:.2f} min → ETE={ete} min; "
        "Burn_raw={br:.2f} L → Burn={burn:.1f} L; ALT {h0:.0f}→{h1:.0f} ft."
        .format(src=from_nm, dst=to_nm, ph=phase, tc=tc, var=var_deg, EW=("E" if var_is_e else "W"),
                mc=mc, dl=delta, wca=wca, th=th, mh=mh, gs=gs, d=d_nm, eter=ete_raw, ete=ete,
                br=burn_raw, burn=_round_half(burn_raw), h0=alt_start_ft, h1=alt_end_ft)
    )
    return alt_end_ft

# construir perna a perna com alocações
cur_alt = alt_targets[0]
for i in range(N):
    from_nm, to_nm = legs[i]["From"], legs[i]["To"]
    d_total = dist[i]
    d_cl = front_used_dist[i]
    d_ds = back_used_dist[i]
    d_cr = max(0.0, d_total - d_cl - d_ds)

    # CLIMB (início da perna)
    if d_cl > 1e-9:
        toc_name = "TOC" if (d_cr + d_ds) > 0 else to_nm
        cur_alt = add_seg("CLIMB", from_nm, toc_name, i, d_cl, vy_kt, ff_climb, cur_alt, roc)
        from_nm = toc_name

    # CRUISE (meio da perna)
    if d_cr > 1e-9:
        tod_name = "TOD" if d_ds > 1e-9 else to_nm
        cur_alt = add_seg("CRUISE", from_nm, tod_name, i, d_cr, float(cruise_ref_kt), ff_cruise, cur_alt, 0.0)
        from_nm = tod_name

    # DESCENT (fim da perna)
    if d_ds > 1e-9:
        cur_alt = add_seg("DESCENT", from_nm, to_nm, i, d_ds, float(descent_ref_kt), ff_descent, cur_alt, rod_fpm)

eta = clock
landing = eta
shutdown = add_minutes(eta,5) if eta else None

# --------- Tabela resumo na App ---------
st.markdown("### Flight plan (App)")
st.dataframe(rows, use_container_width=True)

tot_ete_m = int(sum(int(r['ETE (min)']) for r in rows))
tot_nm  = sum(float(p['dist']) for p in seq_points if isinstance(p.get('dist'), (int,float)))
tot_bo_raw = sum(float(p['burn']) for p in seq_points if isinstance(p.get('burn'), (int,float)))
tot_bo = _round_half(tot_bo_raw)

line = f"**Totais** — Dist {fmt(tot_nm,'dist')} nm • ETE {tot_ete_m//60:02d}:{tot_ete_m%60:02d} • Burn {fmt(tot_bo,'fuel')} L • EFOB {fmt(efob,'fuel')} L"
if eta:
    line += f" • **ETA {eta.strftime('%H:%M')}** • **Shutdown {shutdown.strftime('%H:%M')}**"
st.markdown(line)
if impossible_notes:
    st.warning(" / ".join(impossible_notes))

# ---------------- Export PDF planeado ----------------
# Lê template
try:
    template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)
    fieldset, maxlens = get_form_fields(template_bytes) if PYPDF_OK else (set(), {})
except Exception as e:
    template_bytes=None; fieldset=set(); maxlens={}
    st.error(f"Não foi possível ler o PDF: {e}")

def put_h(k, v, named): 
    if fieldset: put(named, fieldset, k, v, maxlens)

named: Dict[str,str] = {}
if fieldset:
    # Cabeçalho
    etd = (add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else "")
    put_h("AIRCRAFT", aircraft, named)
    put_h("REGISTRATION", registration, named)
    put_h("CALLSIGN", callsign, named)
    put_h("ETD/ETA", f"{etd} / {eta.strftime('%H:%M') if eta else ''}", named)
    put_h("STARTUP", startup_str, named)
    put_h("TAKEOFF", etd, named)
    put_h("LANDING", eta.strftime("%H:%M") if eta else "", named)
    put_h("SHUTDOWN", shutdown.strftime("%H:%M") if shutdown else "", named)
    put_h("LESSON", lesson, named)
    put_h("INSTRUTOR", instrutor, named)
    put_h("STUDENT", student, named)
    put_h("FLT TIME", f"{tot_ete_m//60:02d}:{tot_ete_m%60:02d}", named)
    put_h("LEVEL F/F", fmt(cruise_alt,'alt'), named)
    put_h("CLIMB FUEL", fmt((ff_climb*(max(sum(climb_time_alloc),0)/60.0)),'fuel'), named)
    put_h("QNH", fmt(qnh,'alt'), named)
    put_h("DEPT", aero_freq(dept), named)
    put_h("ENROUTE", "123.755", named)
    put_h("ARRIVAL", aero_freq(arr), named)
    put_h("CLEARANCES", "", named)
    put_h("Departure_Airfield", points[0], named)
    put_h("Arrival_Airfield", points[-1], named)
    put_h("Alternate_Airfield", altn, named)
    put_h("Leg_Number", str(len(seq_points)), named)
    put_h("FLIGHT LEVEL / ALTITUDE", fmt(cruise_alt,'alt'), named)
    put_h("WIND", f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}", named)
    put_h("MAG  VAR", f"{int(round(var_deg))}{'E' if var_is_e else 'W'}", named)
    put_h("TEMP / ISA DEV", f"{fmt(temp_c,'alt')} / {fmt(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)),'alt')}", named)

    # Linhas LegNN_* (1..)
    acc_dist = 0.0
    acc_time = 0
    # seq_points[0] já é o DEP
    for idx, p in enumerate(seq_points, start=1):
        tag = f"Leg{idx:02d}_"
        is_seg = (idx>1)
        if is_seg and isinstance(p["dist"], (int,float)):
            acc_dist += float(p["dist"])
            acc_time += int(p["ete"] or 0)

        # Waypoint e alt alvo (a do ponto, não a da perna)
        put_h(tag+"Waypoint", p["name"], named)
        # NAVAID (opcional)
        if idx-1 < len(st.session_state.navaids) and st.session_state.navaids[idx-1]["Use"]:
            put_h(tag+"Navaid_Identifier", st.session_state.navaids[idx-1]["IDENT"], named)
            put_h(tag+"Navaid_Frequency", st.session_state.navaids[idx-1]["FREQ"], named)
        else:
            # deixa vazio se não usar
            pass

        put_h(tag+"Altitude_FL", (fmt(p["alt"], 'alt') if p["alt"]!="" else ""), named)
        # Métricas da perna (só se is_seg)
        if is_seg:
            put_h(tag+"True_Course",      fmt(p["tc"], 'angle'), named)
            put_h(tag+"Magnetic_Course",  fmt(p["mc"], 'angle'), named)
            put_h(tag+"Ground_Speed",     fmt(p["gs"], 'speed'), named)
            put_h(tag+"Leg_Distance",     fmt(p["dist"], 'dist'), named)
            put_h(tag+"Leg_ETE",          fmt(p["ete"], 'mins'), named)
            put_h(tag+"ETO",              p["eto"], named)
            put_h(tag+"Planned_Burnoff",  fmt(p["burn"], 'fuel'), named)
            put_h(tag+"Estimated_FOB",    fmt(p["efob"], 'fuel'), named)
            put_h(tag+"True_Heading",     fmt(p["th"], 'angle'), named)
            put_h(tag+"Magnetic_Heading", fmt(p["mh"], 'angle'), named)
            put_h(tag+"True_Airspeed",    fmt(p["tas"], 'speed'), named)
            put_h(tag+"Cumulative_Distance", fmt(acc_dist,'dist'), named)
            put_h(tag+"Cumulative_ETE",      fmt(acc_time,'mins'), named)
        else:
            # DEP: ETO (takeoff) e EFOB inicial
            put_h(tag+"ETO", p["eto"], named)
            put_h(tag+"Estimated_FOB", fmt(p["efob"], 'fuel'), named)

# Botão: gerar PDF planeado
if fieldset and st.button("Gerar PDF NAVLOG (planeado)", type="primary"):
    try:
        pdf_bytes_out = fill_pdf(template_bytes, named)
        m = re.search(r'(\d+)', lesson or "")
        lesson_num = m.group(1) if m else "00"
        safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
        filename_pdf = f"{safe_date}_LESSON-{lesson_num}_NAVLOG.pdf"
        st.download_button("📄 Download PDF", data=pdf_bytes_out, file_name=filename_pdf, mime="application/pdf")
        st.success("PDF gerado (planeado).")
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")

# ---------------- Relatório (não corta) ----------------
def build_report_pdf(calc_rows: List[List], details: List[str], params: Dict[str,str]) -> bytes:
    if not REPORTLAB_OK: raise RuntimeError("reportlab missing")
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    H1 = styles["Heading1"]; H2 = styles["Heading2"]; P = styles["Small"]

    story=[]
    story.append(Paragraph("NAVLOG — Relatório (Planeado)", H1))
    story.append(Spacer(1,6))

    # Resumo
    resume = [
        ["Aeronave", params.get("aircraft","—")],
        ["Matrícula", params.get("registration","—")],
        ["Callsign", params.get("callsign","—")],
        ["Lição", params.get("lesson","—")],
        ["Partida", params.get("dept","—")],
        ["Chegada", params.get("arr","—")],
        ["Alternante", params.get("altn","—")],
        ["Cruise Alt", params.get("cruise_alt","—")+" ft"],
        ["QNH", params.get("qnh","—")],
        ["Vento", params.get("wind","—")],
        ["Var. Magn.", params.get("var","—")],
        ["OAT / ISA dev", params.get("temp_isa","—")],
        ["Startup / ETD", f"{params.get('startup','—')} / {params.get('etd','—')}"],
        ["ETA / Shutdown", f"{params.get('eta','—')} / {params.get('shutdown','—')}"],
        ["Tempo total (PLN)", params.get("flt_time","—")],
        ["Fuel inicial", params.get("start_fuel","—")+" L"],
        ["Notas", params.get("notes","—")],
    ]
    t1 = LongTable(resume, colWidths=[45*mm, None], repeatRows=0, hAlign="LEFT")
    t1.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(t1)
    story.append(Spacer(1,6))

    story.append(Paragraph("Rates e ajustes usados", H2))
    rates = [
        ["Vy (kt)", params.get("vy","—")],
        ["ROC (ft/min)", params.get("roc","—")],
        ["ROD (ft/min)", params.get("rod","—")],
        ["TAS climb / cruise / descent (kt)", params.get("tases","—")],
        ["FF climb / cruise / descent (L/h)", params.get("ffs","—")],
        ["Arredondamentos", params.get("rounding","—")],
        ["Distribuição TOC/TOD", params.get("toc_tod","—")],
    ]
    t2 = LongTable(rates, colWidths=[60*mm, None], repeatRows=0, hAlign="LEFT")
    t2.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(t2)
    story.append(Spacer(1,6))

    story.append(Paragraph("Segmentos (resumo)", H2))
    data = [["From→To","Fase","TC°","MC°","TH°","MH°","TAS","GS","Dist(nm)","ETE","ETO","Burn(L)","EFOB(L)","ALT ini→fim"]]
    data += calc_rows
    t3 = LongTable(data,
                   colWidths=[36*mm, 12*mm, 10*mm, 10*mm, 10*mm, 10*mm, 12*mm, 12*mm, 18*mm, 10*mm, 18*mm, 14*mm, 14*mm, 28*mm],
                   repeatRows=1, hAlign="LEFT")
    t3.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("ALIGN",(2,1),(7,-1),"RIGHT"),
        ("ALIGN",(8,1),(12,-1),"RIGHT"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),8.2),
    ]))
    story.append(t3)
    story.append(PageBreak())

    story.append(Paragraph("Cálculos por segmento (passo-a-passo)", H2))
    for s in details:
        story.append(Paragraph(s, P))

    doc.build(story)
    return bio.getvalue()

if st.button("Gerar Relatório (PDF legível)"):
    try:
        params = {
            "aircraft": aircraft, "registration": registration, "callsign": callsign,
            "lesson": lesson, "dept": points[0], "arr": points[-1], "altn": altn,
            "qnh": fmt(qnh,'alt'),
            "wind": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
            "var": f"{int(round(var_deg))}{'E' if var_is_e else 'W'}",
            "cruise_alt": fmt(cruise_alt,'alt'),
            "temp_isa": f"{fmt(temp_c,'alt')} / {fmt(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)),'alt')}",
            "startup": startup_str,
            "etd": (add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else ""),
            "eta": (eta.strftime("%H:%M") if eta else ""), "shutdown": (shutdown.strftime("%H:%M") if shutdown else ""),
            "flt_time": f"{tot_ete_m//60:02d}:{tot_ete_m%60:02d}",
            "start_fuel": fmt(start_fuel,'fuel'),
            "vy": str(_round_unit(vy_kt)),
            "roc": str(_round_unit(roc)),
            "rod": str(_round_unit(rod_fpm)),
            "tases": f"{_round_unit(vy_kt)} / {_round_unit(cruise_ref_kt)} / {_round_unit(descent_ref_kt)}",
            "ffs": f"{fmt(ff_climb,'ff')} / {fmt(ff_cruise,'ff')} / {fmt(ff_descent,'ff')}",
            "rounding": "Dist 0.1 nm; Tempo min inteiro (ceil); Fuel 0.5 L; Vel/TAS/GS 1 kt; Ângulos 1°; Alt <1000→5 / ≥1000→100.",
            "toc_tod": f"Alocação por perna — início: climb (TOC se parcial); fim: descent (TOD se parcial).",
            "notes": " ; ".join(impossible_notes) if impossible_notes else "—",
        }
        report_bytes_out = build_report_pdf(calc_rows, calc_details, params)
        m = re.search(r'(\d+)', lesson or "")
        lesson_num = m.group(1) if m else "00"
        safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
        filename_rep = f"{safe_date}_LESSON-{lesson_num}_NAVLOG_RELATORIO.pdf"
        st.download_button("📑 Download Relatório (PDF)", data=report_bytes_out, file_name=filename_rep, mime="application/pdf")
        st.success("Relatório gerado.")
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {e}")

