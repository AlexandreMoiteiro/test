
# app.py — NAVLOG com TOC/TOD dentro do leg + PDF dinâmico + HOLD em FIX
# Reqs: streamlit, pypdf, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod
from dataclasses import dataclass, asdict

# ========================= GERAL =========================
APP_TZ = pytz.timezone("Europe/Lisbon")

def now_str():
    return dt.datetime.now(APP_TZ).strftime("%Y-%m-%d")

def ascii_safe(x: str) -> str:
    return unicodedata.normalize("NFKD", str(x or "")).encode("ascii","ignore").decode("ascii")

# ========================= PDF helpers =========================
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

def read_pdf_bytes(paths: List[str]) -> Optional[bytes]:
    for p in paths:
        pth = Path(p)
        if pth.exists():
            return pth.read_bytes()
    return None

def get_fields_and_meta(template_bytes: bytes):
    reader = PdfReader(io.BytesIO(template_bytes))
    field_names, maxlens = set(), {}
    try:
        fd = reader.get_fields() or {}
        field_names |= set(fd.keys())
        for k,v in fd.items():
            ml = v.get("/MaxLen")
            if ml: maxlens[k] = int(ml)
    except: pass
    try:
        for page in reader.pages:
            if "/Annots" in page:
                for a in page["/Annots"]:
                    obj = a.get_object()
                    if obj.get("/T"):
                        nm = str(obj["/T"]); field_names.add(nm)
                        ml = obj.get("/MaxLen")
                        if ml: maxlens[nm] = int(ml)
    except: pass
    return field_names, maxlens

def fill_pdf(template_bytes: bytes, fields: dict) -> bytes:
    if not PYPDF_OK: raise RuntimeError("pypdf missing")
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

# ========================= MODELO =========================
@dataclass
class Leg:
    frm: str
    to: str
    tc_true: float = 0.0
    dist_nm: float = 0.0

@dataclass
class Segment:
    phase: str  # "CLIMB" | "CRUISE" | "DESCENT" | "HOLD"
    leg_idx: int
    from_nm: str
    to_nm: str
    dist_nm: float
    tc_true: float
    th_true: float
    mh_mag: float
    tas_kt: float
    gs_kt: float
    ete_min_raw: float
    ete_min_ui: int
    alt_start_ft: float
    alt_end_ft: float
    ff_lph: float
    burn_l_raw: float
    burn_l_ui: float
    eto: str
    efob_l: float

# ========================= PERF & HELPERS =========================
def wrap360(x): x=fmod(x,360.0); return x+360 if x<0 else x
def angle_diff(a,b): return (a-b+180)%360-180

# Triângulo do vento (vento-para = from+180)
def wind_triangle(tc_deg: float, tas_kt: float, wind_from_deg: float, wind_kt: float):
    if tas_kt <= 0:
        return 0.0, wrap360(tc_deg), 0.0
    wind_to = wrap360(wind_from_deg + 180.0)
    beta = radians(angle_diff(wind_to, tc_deg))
    cross = wind_kt * sin(beta)              # +vento da esquerda
    head  = wind_kt * math.cos(beta)         # +tailwind / −headwind
    s = max(-1.0, min(1.0, cross/max(tas_kt,1e-9)))
    wca = degrees(asin(s))
    th  = wrap360(tc_deg + wca)
    gs  = max(0.0, tas_kt*math.cos(radians(wca)) + head)
    return wca, th, gs

def isa_temp(pa_ft): return 15.0 - 2.0*(pa_ft/1000.0)

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

# ========================= DADOS LOCAIS =========================
AEROS={
    "LPSO":{"elev":390,"freq":"119.805"},
    "LPEV":{"elev":807,"freq":"122.705"},
    "LPCB":{"elev":1251,"freq":"122.300"},
    "LPCO":{"elev":587,"freq":"118.405"},
    "LPVZ":{"elev":2060,"freq":"118.305"},
}
def aero_elev(icao): return int(AEROS.get(icao,{}).get("elev",0))
def aero_freq(icao): return AEROS.get(icao,{}).get("freq","")

# ========================= TEMPO =========================
def parse_hhmm(s:str):
    s=(s or "").strip()
    for fmt in ("%H:%M","%H%M"):
        try: return dt.datetime.strptime(s,fmt).time()
        except: pass
    return None

def add_minutes(t:Optional[dt.time],m:int):
    if not t: return None
    today=dt.date.today(); base=dt.datetime.combine(today,t)
    return (base+dt.timedelta(minutes=m)).time()

# ========================= UI =========================
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.title("Navigation Plan & Inflight Log — Tecnam P2008")

DEFAULT_STUDENT="AMOIT"; DEFAULT_AIRCRAFT="P208"; DEFAULT_CALLSIGN="RVP"
REGS=["CS-ECC","CS-ECD","CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW"]
PDF_TEMPLATE_PATHS=["/mnt/data/NAVLOG_FORM.pdf", "NAVLOG_FORM.pdf", "NAVLOG - FORM.pdf"]

# Header
c1,c2,c3=st.columns(3)
with c1:
    aircraft=st.text_input("Aircraft",DEFAULT_AIRCRAFT)
    registration=st.selectbox("Registration",REGS,index=0)
    callsign=st.text_input("Callsign",DEFAULT_CALLSIGN)
with c2:
    student=st.text_input("Student",DEFAULT_STUDENT)
    lesson = st.text_input("Lesson","")
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
    rpm_descent= st.number_input("Descent RPM (se NÃO idle)",1700,2300,1800,step=10)
    idle_mode  = st.checkbox("Descent mostly IDLE", value=True)
with c9:
    rod_fpm=st.number_input("ROD (ft/min)",200,1500,700,step=10)
    idle_ff=st.number_input("Idle FF (L/h)", 0.0, 20.0, 5.0, step=0.1)
    start_fuel=st.number_input("Fuel inicial (EFOB_START) [L]",0.0,1000.0,85.0,step=0.1)

# Velocidades ref
cruise_ref_kt = st.number_input("Cruise speed (kt)", 40, 140, 80, step=1)
descent_ref_kt= st.number_input("Descent speed (kt)", 40, 120, 65, step=1)

# HOLD options
st.markdown("### Tempo em FIX (hold)")
hold_ff_lph = st.number_input("Hold FF (L/h)", 0.0, 40.0, 18.0, step=0.1)
st.caption("Define quantos minutos queres permanecer em cada ponto de chegada (DIST=0; consome combustível; avança relógio).")

# ===== ROUTE (textarea) + JSON =====
def parse_route_text(txt:str) -> List[str]:
    tokens = re.split(r"[,\s→\-]+", (txt or "").strip())
    return [t for t in tokens if t]

st.markdown("### Route (DEP … ARR)")
default_route = f"{dept} {arr}"
route_text = st.text_area("Pontos (separados por espaço, vírgulas ou '->')",
                          value=st.session_state.get("route_text", default_route))
c_ra, c_rb = st.columns([1,1])
with c_ra:
    apply_route = st.button("Aplicar rota")
with c_rb:
    def snapshot_route() -> dict:
        return {
            "route_points": st.session_state.get("points", [dept, arr]),
            "legs": [{"TC":l["tc_true"], "Dist":l["dist_nm"]} for l in st.session_state.get("legs_raw", [])]
        }
    st.download_button("💾 Download rota (JSON)",
                       data=json.dumps(snapshot_route(), ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"route_{ascii_safe(registration)}.json",
                       mime="application/json")

uploaded = st.file_uploader("📤 Seleciona rota (JSON)", type=["json"])
use_uploaded = st.button("Usar rota do ficheiro")
if use_uploaded and uploaded is not None:
    try:
        data = json.loads(uploaded.read().decode("utf-8"))
        st.session_state.points = list(data.get("route_points") or [dept, arr])
        tgt = max(0,len(st.session_state.points)-1)
        src_legs = data.get("legs") or []
        legs = []
        for i in range(tgt):
            tc = float(src_legs[i]["TC"]) if i < len(src_legs) and "TC" in src_legs[i] else 0.0
            di = float(src_legs[i]["Dist"]) if i < len(src_legs) and "Dist" in src_legs[i] else 0.0
            legs.append({"frm":st.session_state.points[i],
                         "to":st.session_state.points[i+1],
                         "tc_true":tc,"dist_nm":di})
        st.session_state.legs_raw = legs
        st.session_state["route_text"] = " ".join(st.session_state.points)
        st.success("Rota carregada do JSON.")
    except Exception as e:
        st.error(f"Falha a carregar JSON: {e}")

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

# LEGS
def blank_leg(): return {"frm":"","to":"","tc_true":0.0,"dist_nm":0.0}
if "legs_raw" not in st.session_state: st.session_state.legs_raw = []
target_legs = max(0, len(points)-1)
legs_raw = st.session_state.legs_raw
if len(legs_raw) < target_legs: legs_raw += [blank_leg() for _ in range(target_legs - len(legs_raw))]
elif len(legs_raw) > target_legs: legs_raw = legs_raw[:target_legs]
for i in range(target_legs):
    legs_raw[i]["frm"]=points[i]; legs_raw[i]["to"]=points[i+1]
st.session_state.legs_raw = legs_raw

st.markdown("### Legs (distância do ponto anterior)")
legs_cfg = {
    "frm": st.column_config.TextColumn("From", disabled=True),
    "to":   st.column_config.TextColumn("To", disabled=True),
    "tc_true":   st.column_config.NumberColumn("TC (°T)", step=0.1, min_value=0.0, max_value=359.9),
    "dist_nm": st.column_config.NumberColumn("Dist (nm)", step=0.1, min_value=0.0),
}
legs_view = st.data_editor(legs_raw, hide_index=True, use_container_width=True,
                           column_config=legs_cfg, num_rows="fixed", key="legs_table")
for i,row in enumerate(legs_view):
    legs_raw[i]["tc_true"]  = float(row.get("tc_true") or 0.0)
    legs_raw[i]["dist_nm"]= float(row.get("dist_nm") or 0.0)

legs: List[Leg] = [Leg(frm=l["frm"], to=l["to"], tc_true=l["tc_true"], dist_nm=l["dist_nm"]) for l in legs_raw]
N = len(legs)

# ===== HOLD table (minutos por fix de chegada)
hold_points = [{"Fix": legs[i].to, "Hold_min": 0} for i in range(N)] if N>0 else []
# guarda estado editável
hold_points = st.data_editor(
    hold_points, hide_index=True, use_container_width=True,
    column_config={
        "Fix": st.column_config.TextColumn("Fix (ponto de chegada)"),
        "Hold_min": st.column_config.NumberColumn("Hold (min)", min_value=0, step=1),
    },
    key="hold_table"
)
hold_minutes: Dict[str,int] = {row["Fix"]: int(row.get("Hold_min") or 0) for row in hold_points if row.get("Fix")}

# ===== Cálculo (perfil vertical, cortes dentro do leg) =====
def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0

dep_elev = aero_elev(dept); arr_elev = aero_elev(arr)
start_alt = float(dep_elev)
end_alt   = float(arr_elev)

pa_start  = pressure_alt(start_alt, qnh)
pa_cruise = pressure_alt(cruise_alt, qnh)
vy_kt = vy_interp_enroute(pa_start)
tas_climb, tas_cruise, tas_descent = vy_kt, float(cruise_ref_kt), float(descent_ref_kt)

roc = roc_interp_enroute(pa_start, temp_c)                 # ft/min
delta_climb = max(0.0, cruise_alt - start_alt)
delta_desc  = max(0.0, cruise_alt - end_alt)
t_climb_total = delta_climb / max(roc,1e-6)
t_desc_total  = delta_desc  / max(rod_fpm,1e-6)

# FFs (AFM)
pa_mid_climb = start_alt + 0.5*delta_climb
pa_mid_desc  = end_alt   + 0.5*delta_desc
_, ff_climb  = cruise_lookup(pa_mid_climb, int(rpm_climb),  temp_c)
_, ff_cruise = cruise_lookup(pa_cruise,   int(rpm_cruise),  temp_c)
ff_descent   = float(idle_ff) if idle_mode else cruise_lookup(pa_mid_desc, int(rpm_descent), temp_c)[1]

def apply_var(true_deg,var_deg,east_is_negative=False):
    return wrap360(true_deg - var_deg if east_is_negative else true_deg + var_deg)

def gs_for(tc, tas): return wind_triangle(float(tc), float(tas), wind_from, wind_kt)[2]

dist = [float(l.dist_nm or 0.0) for l in legs]
gs_climb   = [gs_for(legs[i].tc_true, tas_climb)   for i in range(N)]
gs_cruise  = [gs_for(legs[i].tc_true, tas_cruise)  for i in range(N)]
gs_descent = [gs_for(legs[i].tc_true, tas_descent) for i in range(N)]

# ---- Distribuir CLIMB para a frente
climb_nm   = [0.0]*N
idx_toc = None
rem_t = float(t_climb_total)
for i in range(N):
    if rem_t <= 1e-9: break
    gs = max(gs_climb[i], 1e-6)
    t_full = 60.0 * dist[i] / gs
    use_t = min(rem_t, t_full)
    climb_nm[i] = min(dist[i], gs * use_t / 60.0)
    rem_t -= use_t
    if rem_t <= 1e-9:
        idx_toc = i
        break

# ---- Distribuir DESCENT para trás
descent_nm = [0.0]*N
idx_tod = None
rem_t = float(t_desc_total)
for j in range(N-1, -1, -1):
    if rem_t <= 1e-9: break
    gs = max(gs_descent[j], 1e-6)
    t_full = 60.0 * dist[j] / gs
    use_t = min(rem_t, t_full)
    descent_nm[j] = min(dist[j], gs * use_t / 60.0)
    rem_t -= use_t
    if rem_t <= 1e-9:
        idx_tod = j
        break

# ===== Segmentação & linhas
startup = parse_hhmm(startup_str)
takeoff = add_minutes(startup,15) if startup else None
clock = takeoff

PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓", "HOLD":"⟳"}

alt_cursor = float(start_alt)
efob=float(start_fuel)
segments: List[Segment] = []

# Ponto inicial (DEP) para o PDF-seq
seq_points=[]
seq_points.append({"name": dept, "alt": int(round(start_alt)),
                   "tc":"", "th":"", "mh":"", "tas":"", "gs":"", "dist":"", "ete":"", "eto": (takeoff.strftime("%H:%M") if takeoff else ""), "burn":"", "efob": efob})

def ceil_pos_minutes(x):  # arredonda ↑ e garante 1 min quando >0
    return max(1, int(math.ceil(x - 1e-9))) if x > 0 else 0

def add_segment(phase:str, from_nm:str, to_nm:str, i_leg:int, d_nm:float, tas:float, ff_lph:float, ete_override_min: Optional[int] = None, speed_for_pdf: Optional[float] = None):
    """Acrescenta segmento (inclui HOLD se ete_override_min for dado). speed_for_pdf permite SPEED no PDF diferir do GS (ex.: HOLD)."""
    nonlocal clock, alt_cursor, efob
    if d_nm <= 1e-9 and not ete_override_min:
        return
    leg_tc = (legs[i_leg].tc_true if 0 <= i_leg < len(legs) else 0.0)

    if phase == "HOLD":
        # hold: sem deslocação, TC/TH irrelevantes, GS=0
        ete_raw = float(ete_override_min or 0)
        gs = 0.0
        th = leg_tc
        alt_start = alt_cursor
        alt_end = alt_cursor
    else:
        _, th, gs = wind_triangle(leg_tc, tas, wind_from, wind_kt)
        ete_raw = 60.0 * d_nm / max(gs,1e-6)
        alt_start = alt_cursor
        if phase == "CLIMB":
            alt_end = min(cruise_alt, alt_start + roc * ete_raw)
        elif phase == "DESCENT":
            alt_end = max(end_alt,   alt_start - rod_fpm * ete_raw)
        else:
            alt_end = alt_start

    ete_ui = ceil_pos_minutes(ete_raw) if phase != "HOLD" else int(ete_raw)
    burn_raw = ff_lph * (ete_raw/60.0)
    burn_ui = round(burn_raw, 1)

    eto = ""
    if clock:
        clock = add_minutes(clock, ete_ui); eto = clock.strftime("%H:%M")
    efob = max(0.0, efob - burn_raw)
    alt_cursor = alt_end

    mh = apply_var(th, var_deg, var_is_e) if phase != "HOLD" else apply_var(leg_tc, var_deg, var_is_e)
    spd_pdf = speed_for_pdf if speed_for_pdf is not None else (gs if phase!="HOLD" else "")

    seg = Segment(
        phase=phase, leg_idx=i_leg, from_nm=from_nm, to_nm=to_nm, dist_nm=(0.0 if phase=="HOLD" else d_nm),
        tc_true=(leg_tc if phase!="HOLD" else leg_tc), th_true=th, mh_mag=mh,
        tas_kt=(tas if phase!="HOLD" else 0.0), gs_kt=(gs if phase!="HOLD" else 0.0),
        ete_min_raw=ete_raw, ete_min_ui=ete_ui,
        alt_start_ft=alt_start, alt_end_ft=alt_end,
        ff_lph=ff_lph, burn_l_raw=burn_raw, burn_l_ui=burn_ui,
        eto=eto, efob_l=efob
    )
    segments.append(seg)

    # Ponto PDF (chegada ao marker do segmento)
    name_for_pdf = (to_nm if phase!="HOLD" else f"{to_nm} HOLD")
    seq_points.append({
        "name": name_for_pdf,
        "alt": int(round(alt_end)),
        "tc": int(round(leg_tc)) if phase!="HOLD" else int(round(leg_tc)),
        "th": int(round(th)) if phase!="HOLD" else "",
        "mh": int(round(mh)) if phase!="HOLD" else int(round(apply_var(leg_tc, var_deg, var_is_e))),
        "tas": int(round(tas)) if phase!="HOLD" else "",
        "gs": int(round(gs)) if phase!="HOLD" else "",
        "dist": (d_nm if phase!="HOLD" else 0.0),
        "ete": ete_ui, "eto": eto, "burn": burn_raw, "efob": efob
    })

# Percorrer legs e inserir cortes + holds
for i in range(N):
    leg_from, leg_to = legs[i].frm, legs[i].to
    d_total  = dist[i]
    d_cl = min(climb_nm[i], d_total)
    d_ds = min(descent_nm[i], d_total - d_cl)
    d_cr = max(0.0, d_total - d_cl - d_ds)

    cur_from = leg_from

    if d_cl > 0:
        to_name = "TOC" if (idx_toc == i and d_cl < d_total) else leg_to
        add_segment("CLIMB", cur_from, to_name, i, d_cl, vy_kt, ff_climb)
        cur_from = to_name

    if d_cr > 0:
        to_name = "TOD" if (idx_tod == i and d_ds > 0) else leg_to
        add_segment("CRUISE", cur_from, to_name, i, d_cr, float(cruise_ref_kt), ff_cruise)
        cur_from = to_name

    if d_ds > 0:
        add_segment("DESCENT", cur_from, leg_to, i, d_ds, float(descent_ref_kt), ff_descent)
        cur_from = leg_to

    # HOLD ao chegar ao FIX (se definido)
    hm = hold_minutes.get(leg_to, 0)
    if hm and hm > 0:
        # fase HOLD: sem deslocação; usa ff=hold_ff_lph; TC do leg para referência, mas sem TH/GS/TAS
        add_segment("HOLD", cur_from, leg_to, i, 0.0, 0.0, hold_ff_lph, ete_override_min=int(hm), speed_for_pdf="")

eta = clock
landing = eta
shutdown = add_minutes(eta,5) if eta else None

# ===== Tabela da APP =====
st.markdown("### Flight plan — cortes dentro do leg (App)")
rows=[]
for s in segments:
    rows.append({
        "Fase": PH_ICON[s.phase],
        "Leg/Marker": f"{s.from_nm}→{s.to_nm}" if s.phase!="HOLD" else f"{s.to_nm} (HOLD)",
        "To (Name)": s.to_nm,
        "ALT (ft)": f"{int(round(s.alt_start_ft))}→{int(round(s.alt_end_ft))}",
        "Detalhe": f"{s.phase.capitalize()} {s.dist_nm:.1f} nm" if s.phase!="HOLD" else f"HOLD {s.ete_min_ui} min",
        "TC (°T)": round(s.tc_true,0),
        "TH (°T)": (round(s.th_true,0) if s.phase!="HOLD" else ""),
        "MH (°M)": round(s.mh_mag,0),
        "TAS (kt)": (round(s.tas_kt,0) if s.phase!="HOLD" else ""),
        "GS (kt)": (round(s.gs_kt,0) if s.phase!="HOLD" else ""),
        "FF (L/h)": round(s.ff_lph,1),
        "Dist (nm)": round(s.dist_nm,1),
        "ETE (min)": int(s.ete_min_ui), "ETO": s.eto,
        "Burn (L)": round(s.burn_l_raw,1), "EFOB (L)": round(s.efob_l,1)
    })

cfg={
    "Fase":      st.column_config.TextColumn("Fase"),
    "Leg/Marker": st.column_config.TextColumn("Leg / Marker"),
    "To (Name)":  st.column_config.TextColumn("To (Name)", disabled=True),
    "ALT (ft)":   st.column_config.TextColumn("ALT (ft)"),
    "Detalhe":    st.column_config.TextColumn("Detalhe"),
    "TC (°T)":    st.column_config.NumberColumn("TC (°T)", disabled=True),
    "TH (°T)":    st.column_config.NumberColumn("TH (°T)", disabled=True),
    "MH (°M)":    st.column_config.NumberColumn("MH (°M)", disabled=True),
    "TAS (kt)":   st.column_config.NumberColumn("TAS (kt)", disabled=True),
    "GS (kt)":    st.column_config.NumberColumn("GS (kt)", disabled=True),
    "FF (L/h)":   st.column_config.NumberColumn("FF (L/h)", disabled=True),
    "Dist (nm)":  st.column_config.NumberColumn("Dist (nm)", disabled=True),
    "ETE (min)":  st.column_config.NumberColumn("ETE (min)", disabled=True),
    "ETO":        st.column_config.TextColumn("ETO", disabled=True),
    "Burn (L)":   st.column_config.NumberColumn("Burn (L)", disabled=True),
    "EFOB (L)":   st.column_config.NumberColumn("EFOB (L)", disabled=True),
}
st.data_editor(rows, hide_index=True, use_container_width=True, num_rows="fixed", column_config=cfg, key="fp_table")

# Totais a partir dos raws (coerência)
tot_nm  = sum(s.dist_nm for s in segments)
tot_min_raw = sum(s.ete_min_raw for s in segments)
tot_burn_raw = sum(s.burn_l_raw for s in segments)
tot_min_ui = int(round(tot_min_raw))
tot_line = f"**Totais** — Dist {tot_nm:.1f} nm • ETE {tot_min_ui//60:02d}:{tot_min_ui%60:02d} • Burn {tot_burn_raw:.1f} L • EFOB {efob:.1f} L"
if eta:
    tot_line += f" • **ETA {eta.strftime('%H:%M')}** • **Landing {landing.strftime('%H:%M')}** • **Shutdown {shutdown.strftime('%H:%M')}**"
st.markdown(tot_line)

# ===== Exportações básicas =====
st.markdown("### Exportações")
plan = {
    "header": {
        "aircraft": aircraft, "registration": registration, "callsign": callsign,
        "student": student, "lesson": lesson, "instrutor": instrutor,
        "dept": dept, "arr": arr, "altn": altn, "startup": startup_str
    },
    "environment": {
        "qnh": qnh, "cruise_alt": cruise_alt, "temp_c": temp_c,
        "var_deg": var_deg, "var_is_e": var_is_e,
        "wind_from": wind_from, "wind_kt": wind_kt
    },
    "performance": {
        "rpm_climb": rpm_climb, "rpm_cruise": rpm_cruise, "rpm_descent": rpm_descent,
        "idle_mode": idle_mode, "rod_fpm": rod_fpm, "idle_ff": idle_ff,
        "start_fuel": start_fuel, "cruise_ref_kt": cruise_ref_kt, "descent_ref_kt": descent_ref_kt,
        "hold_ff_lph": hold_ff_lph
    },
    "route": {
        "points": points,
        "legs": [asdict(l) for l in legs],
        "holds": hold_minutes
    },
    "segments": [asdict(s) for s in segments],
    "totals": {
        "dist_nm": tot_nm, "ete_min_raw": tot_min_raw, "burn_l_raw": tot_burn_raw,
        "ete_min_ui": tot_min_ui,
        "eta": (eta.strftime('%H:%M') if eta else ""),
        "landing": (landing.strftime('%H:%M') if landing else ""),
        "shutdown": (shutdown.strftime('%H:%M') if shutdown else ""),
        "efob_end_l": efob
    }
}
st.download_button(
    "💾 Exportar plano completo (JSON)",
    data=json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"plan_{ascii_safe(registration)}_{now_str()}.json",
    mime="application/json"
)

# ===== PDF export (NAVLOG_FORM.pdf dinâmico) =====
st.markdown("### PDF export")
show_fields = st.checkbox("Mostrar nomes de campos do PDF (debug)")
uploaded_template = st.file_uploader("📄 Template PDF (opcional)", type=["pdf"], key="tpl")
if uploaded_template:
    template_bytes = uploaded_template.read()
else:
    template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)

def build_pdf_items_from_points(points):
    """Cada item é o ponto de chegada (inclui HOLD como ponto distinto). O primeiro é o DEP sem métricas."""
    items = []
    for idx, p in enumerate(points):
        it = {
            "FIX": p["name"],
            "ALT": str(int(round(p["alt"]))),
            "TC":  (p["tc"]  if idx>0 else ""),
            # M CRS será calculado a partir do TC + VAR
            "TAS": (p["tas"] if idx>0 else ""),
            "GS":  (p["gs"]  if idx>0 else ""),
            "DIST": (f"{p['dist']:.1f}" if idx>0 and isinstance(p["dist"], (int,float)) else ""),
            "ETE":  (str(p["ete"]) if idx>0 else ""),
            "ETO":  (p["eto"] if idx>0 else (p["eto"] or "")),
            "BURN": (f"{p['burn']:.1f}" if idx>0 and isinstance(p["burn"], (int,float)) else ""),
            "EFOB": (f"{p['efob']:.1f}" if idx>0 and isinstance(p["efob"], (int,float)) else f"{p['efob']:.1f}" if idx==0 else "")
        }
        items.append(it)
    return items

def normalize_base(fname: str) -> str:
    return re.sub(r"_[0-9]+$", "", fname).strip()

def suffix_index(fname: str) -> int:
    m = re.search(r"_([0-9]+)$", fname)
    return int(m.group(1)) if m else 0

if not template_bytes:
    st.error("Não foi possível ler um template PDF (carrega o teu 'NAVLOG_FORM.pdf' ou usa o uploader).")
else:
    fieldset, maxlens = get_fields_and_meta(template_bytes)
    if show_fields:
        st.code("\n".join(sorted(fieldset)))

    try:
        named: Dict[str,str] = {}
        header_values = {
            "AIRCRAFT": aircraft,
            "REGISTRATION": registration,
            "CALLSIGN": callsign,
            "STARTUP": startup_str,
            "TAKEOFF": add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else "",
            "LANDING": (eta.strftime("%H:%M") if eta else ""),
            "SHUTDOWN": (shutdown.strftime("%H:%M") if shutdown else ""),
            "LESSON": lesson,
            "INSTRUTOR": instrutor,
            "STUDENT": student,
            "INSTRUMENT": "",
            "LANDINGS": "",
            "FLT TIME": f"{(int(round(tot_min_raw))//60):02d}:{(int(round(tot_min_raw))%60):02d}",
            "LEVEL F/F": f"{int(round(cruise_alt))}",
            "CLIMB FUEL": f"{(ff_climb*(max(t_climb_total,0.0)/60.0)):.1f}",
            "QNH": f"{int(round(qnh))}",
            "CLEARANCES": "",
            "DEPT": aero_freq(dept),
            "ENROUTE": "123.755",
            "ARRIVAL": aero_freq(arr),
            "Departure_Airfield": dept,
            "Arrival_Airfield": arr,
            "Alternate_Airfield": altn,
            "FLIGHT LEVEL / ALTITUDE": f"{int(round(cruise_alt))}",
            "WIND": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
            "MAG  VAR": f"{var_deg:.1f}{'E' if var_is_e else 'W'}",
            "TEMP / ISA DEV": f"{int(round(temp_c))} / {round(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)))}",
            "ETD/ETA": f"{(add_minutes(parse_hhmm(startup_str),15).strftime('%H:%M') if startup_str else '')} / {(eta.strftime('%H:%M') if eta else '')}",
            "Leg_Number": str(len(legs)) if len(legs)>0 else "0",
            "OBSERVATIONS": "",
        }
        for k,v in header_values.items():
            put(named, fieldset, k, v, maxlens)

        # Construir linhas (inclui HOLD como ponto)
        pdf_items = build_pdf_items_from_points(seq_points)

        # Agrupar campos por base e ordenar por sufixo para preenchimento dinâmico
        by_base: Dict[str, List[str]] = {}
        for fname in fieldset:
            base = normalize_base(fname)
            by_base.setdefault(base, []).append(fname)
        for base in by_base:
            by_base[base].sort(key=lambda x: suffix_index(x))

        def fill_series(base_name: str, values: List[str]):
            names = by_base.get(base_name, [])
            for i, val in enumerate(values):
                if i >= len(names): break
                put(named, fieldset, names[i], val, maxlens)

        vals_FIX   = [it["FIX"] for it in pdf_items]
        vals_ALT   = [it["ALT"] for it in pdf_items]
        vals_TCRS  = [str(it["TC"])   if it["TC"]   != "" else "" for it in pdf_items]
        # M CRS: usa TC +/- Var
        vals_MCRS  = []
        for it in pdf_items:
            if it["TC"] == "":
                vals_MCRS.append("")
            else:
                try:
                    mcrs = apply_var(float(it["TC"]), var_deg, var_is_e)
                    vals_MCRS.append(str(int(round(mcrs))))
                except:
                    vals_MCRS.append("")
        vals_SPEED = [str(it["GS"])   if it["GS"]   != "" else "" for it in pdf_items]  # SPEED = GS
        vals_TAS   = [str(it["TAS"])  if it["TAS"]  != "" else "" for it in pdf_items]
        vals_DIST  = [it["DIST"] for it in pdf_items]
        vals_ETE   = [it["ETE"]  for it in pdf_items]
        vals_ETO   = [it["ETO"]  for it in pdf_items]
        vals_ATO   = ["" for _ in pdf_items]  # preenchido em voo
        vals_PBO   = [it["BURN"] for it in pdf_items]  # Pl B/O
        vals_EFOB  = [it["EFOB"] for it in pdf_items]

        # NAVAIDS: preenche primeira coluna com a freq do FIX conhecido; TOC/TOD/HOLD ficam vazios
        def navaid_for(name: str) -> str:
            nm = str(name or "")
            if nm in AEROS: return aero_freq(nm)
            # nomes "XYZ HOLD", "TOC", "TOD" -> ""
            return ""
        navaid_col1 = [navaid_for(x) for x in vals_FIX]
        navaid_col2 = ["" for _ in pdf_items]

        # Disparo das séries
        fill_series("FIX",    vals_FIX)
        fill_series("ALT",    vals_ALT)
        fill_series("T CRS",  vals_TCRS)
        fill_series("M CRS",  vals_MCRS)
        fill_series("SPEED",  vals_SPEED)
        fill_series("TAS",    vals_TAS)
        fill_series("DIST",   vals_DIST)
        fill_series("ETE",    vals_ETE)
        fill_series("ETO",    vals_ETO)
        fill_series("ATO",    vals_ATO)
        fill_series("Pl B/O", vals_PBO)
        fill_series("EFOB",   vals_EFOB)

        # NAVAIDS em duas colunas (NAVAIDS, NAVAIDS_1, NAVAIDS_2, ... alternando)
        navaids_names = by_base.get("NAVAIDS", [])
        for i in range(len(pdf_items)):
            if i*2 < len(navaids_names):
                put(named, fieldset, navaids_names[i*2], navaid_col1[i], maxlens)
            if i*2+1 < len(navaids_names):
                put(named, fieldset, navaids_names[i*2+1], navaid_col2[i], maxlens)

        if show_fields:
            unused = sorted(set(fieldset) - set(named.keys()))
            if unused:
                st.caption(f"Campos não usados no preenchimento: {', '.join(unused)}")

        if st.button("Gerar PDF preenchido", type="primary"):
            try:
                out = fill_pdf(template_bytes, named)
                safe_reg = ascii_safe(registration)
                filename = f"{now_str()}_{safe_reg}_NAVLOG.pdf"
                st.download_button("Download PDF", data=out, file_name=filename, mime="application/pdf", key="pdf_dl")
                st.success("PDF gerado. Revê antes do voo.")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")
    except Exception as e:
        st.error(f"Erro ao preparar/gerar PDF: {e}")

# ===== Alertas simples =====
with st.expander("Reservas & alertas"):
    reserve_min = st.number_input("Reserva (min)", 0, 180, 30, step=5)
    reserve_fuel_l = st.number_input("Reserva combustível (L)", 0.0, 200.0, 10.0, step=0.5)
    if start_fuel - tot_burn_raw < reserve_fuel_l:
        st.warning("EFOB final abaixo da reserva definida.")
    if idx_toc is None and cruise_alt > start_alt:
        st.info("Rota curta para atingir cruzeiro antes do destino.")
    if idx_tod is None and cruise_alt > end_alt and tot_min_ui>0:
        st.info("Sem espaço suficiente para a descida planeada (TOD no destino).")
