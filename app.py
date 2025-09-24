# app.py — NAVLOG planeado (TOC/TOD), preenche NAVLOG_FORM.pdf e gera Relatório legível
# Reqs: streamlit, pypdf, reportlab, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod

# ============== Config ==============
st.set_page_config(page_title="NAVLOG (PDF + Relatório)", layout="wide", initial_sidebar_state="collapsed")
PDF_TEMPLATE_PATHS = ["NAVLOG_FORM.pdf"]   # teu template

# ============== Imports opcionais ==============
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ============== Arredondamento / Formatação ==============
def _round_default(x: float) -> int:
    """Default: <1000 de 5 em 5; >=1000 à centena."""
    if x is None: return 0
    v = float(x)
    base = 5 if abs(v) < 1000 else 100
    return int(round(v / base) * base)

def _round_unit(x: float) -> int:
    """Parâmetros sensíveis (fuel, velocidades, minutos): à unidade."""
    if x is None: return 0
    return int(round(float(x)))

def _round_angle(x: float) -> int:
    """Ângulos ao grau [0..359]."""
    if x is None: return 0
    return int(round(float(x))) % 360

def fmt(x: float, kind: str = "default") -> str:
    """
    kind:
      - "default" (altitudes, distâncias): <1000->5; >=1000->100
      - "fuel"     (EFOB, Burn, FF): unidade
      - "speed"    (TAS/GS): unidade
      - "mins"     (ETE/ACC): unidade (já inteiro a montante)
      - "angle"    (TC/TH/MH): grau
    """
    if kind == "fuel":  return str(_round_unit(x))
    if kind == "speed": return str(_round_unit(x))
    if kind == "mins":  return str(_round_unit(x))
    if kind == "angle": return str(_round_angle(x))
    return str(_round_default(x))

# ============== Helpers ==============
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

# ============== AFM (igual ao teu) ==============
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

# ============== Vento & variação ==============
def wind_triangle(tc_deg: float, tas_kt: float, wind_from_deg: float, wind_kt: float):
    if tas_kt <= 0:
        return 0.0, wrap360(tc_deg), 0.0
    delta = radians(angle_diff(wind_from_deg, tc_deg))
    cross = wind_kt * sin(delta)           # + se vento pela esquerda
    s = max(-1.0, min(1.0, cross/max(tas_kt,1e-9)))
    wca = degrees(asin(s))
    th  = wrap360(tc_deg + wca)
    gs  = max(0.0, tas_kt*math.cos(radians(wca)) - wind_kt*math.cos(delta))
    return wca, th, gs

def apply_var(true_deg,var_deg,east_is_negative=False):
    return wrap360(true_deg - var_deg if east_is_negative else true_deg + var_deg)

# ============== Aeródromos (exemplo) ==============
AEROS={
 "LPSO":{"elev":390,"freq":"119.805"},
 "LPEV":{"elev":807,"freq":"122.705"},
 "LPCB":{"elev":1251,"freq":"122.300"},
 "LPCO":{"elev":587,"freq":"118.405"},
 "LPVZ":{"elev":2060,"freq":"118.305"},
}
def aero_elev(icao): return int(AEROS.get(icao,{}).get("elev",0))
def aero_freq(icao): return AEROS.get(icao,{}).get("freq","")

# ============== PDF helpers ==============
def read_pdf_bytes(paths: List[str]) -> bytes:
    for p in paths:
        if Path(p).exists():
            return Path(p).read_bytes()
    raise FileNotFoundError(paths)

def get_form_fields(template_bytes: bytes):
    """Devolve (fieldset, maxlens, fields_pos) onde fields_pos = [{name,page,(x0,y0,x1,y1)}]."""
    if not PYPDF_OK:
        raise RuntimeError("Dependência ausente: pypdf. Instala com: pip install pypdf")
    reader = PdfReader(io.BytesIO(template_bytes))
    field_names, maxlens = set(), {}
    fields_pos = []
    try:
        fd = reader.get_fields() or {}
        field_names |= set(fd.keys())
        for k,v in fd.items():
            ml = v.get("/MaxLen")
            if ml: maxlens[k] = int(ml)
    except: pass
    try:
        for p_idx, page in enumerate(reader.pages):
            if "/Annots" in page:
                for a in page["/Annots"]:
                    obj = a.get_object()
                    nm = obj.get("/T")
                    rc = obj.get("/Rect")
                    if nm:
                        name = str(nm)
                        field_names.add(name)
                        if rc and len(rc)==4:
                            x0,y0,x1,y1 = [float(z) for z in rc]
                            fields_pos.append({"name":name, "page":p_idx, "rect":(x0,y0,x1,y1)})
                        ml = obj.get("/MaxLen")
                        if ml: maxlens[name] = int(ml)
    except: pass
    return field_names, maxlens, fields_pos

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

# ============== Posicionamento por linha (corrige “fora do sítio”) ==============
def canon(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', (s or '').upper())

GRID_BASES = [
    "FIX","NAME","NAVAIDS","ALT","TCRS","MCRS","T CRS","M CRS",
    "GS","TAS","SPEED","DIST","ETE","ETO","ACC",
    "PL B/O","Pl B/O","PL_BO","PL BO","EFOB",
    "T HDG","M HDG"  # se existirem
]
def base_of(name:str) -> str:
    c = (name or "").upper()
    # normalizar nomes comuns
    c = c.replace("  ", " ")
    c = c.replace("TRUE COURSE","T CRS").replace("MAG COURSE","M CRS")
    c = c.replace("COURSE TRUE","T CRS").replace("COURSE MAG","M CRS")
    return c

def fields_to_micro_rows(fields_pos: List[dict]) -> List[List[dict]]:
    """Agrupa por página, ordena por Y desc e ‘clusteriza’ por linhas micro (top/bottom)."""
    # filtra só campos da grelha
    def is_grid_field(nm:str)->bool:
        b = base_of(nm)
        return any(b.startswith(x) for x in GRID_BASES)
    grid = [f for f in fields_pos if is_grid_field(f["name"])]
    micro_rows_all=[]
    for pg in sorted(set(f["page"] for f in grid)):
        page_fields = [f for f in grid if f["page"]==pg]
        # ordenar por Y desc
        for f in page_fields:
            x0,y0,x1,y1 = f["rect"]; f["xc"]=(x0+x1)/2.0; f["yc"]=(y0+y1)/2.0
        page_fields.sort(key=lambda a: (-a["yc"], a["xc"]))
        # cluster por Y (pequeno threshold para separar top/bottom da mesma linha)
        y_tol = 6.0
        cur=[]; last_y=None
        for f in page_fields:
            if last_y is None or abs(f["yc"]-last_y) <= y_tol:
                cur.append(f); last_y = f["yc"] if last_y is None else (last_y+f["yc"])/2.0
            else:
                micro_rows_all.append(sorted(cur, key=lambda z: z["xc"]))
                cur=[f]; last_y=f["yc"]
        if cur:
            micro_rows_all.append(sorted(cur, key=lambda z: z["xc"]))
    return micro_rows_all  # sequência: topo->baixo, por página

def find_first_in_row(micro_row: List[dict], starts_with: List[str]) -> Optional[str]:
    for f in micro_row:
        nm = base_of(f["name"])
        for s in starts_with:
            if nm.startswith(s):
                return f["name"]
    return None

def find_all_in_row(micro_row: List[dict], starts_with: List[str]) -> List[str]:
    out=[]
    for f in micro_row:
        nm = base_of(f["name"])
        for s in starts_with:
            if nm.startswith(s):
                out.append(f["name"]); break
    return out

# ============== UI ==============
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
c_ra, c_rb = st.columns([1,1])
with c_ra:
    apply_route = st.button("Aplicar rota")
with c_rb:
    def snapshot_route() -> dict:
        return {
            "route_points": st.session_state.get("points", [dept, arr]),
            "legs": [{"TC":l["TC"], "Dist":l["Dist"]} for l in st.session_state.get("legs", [])]
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
        st.session_state.legs = []
        for i in range(tgt):
            tc = float(src_legs[i]["TC"]) if i < len(src_legs) and "TC" in src_legs[i] else 0.0
            di = float(src_legs[i]["Dist"]) if i < len(src_legs) and "Dist" in src_legs[i] else 0.0
            st.session_state.legs.append({"From":st.session_state.points[i],
                                          "To":st.session_state.points[i+1],
                                          "TC":tc,"Dist":di})
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

# ===== NAVAIDS por fix =====
st.markdown("### NAVAIDS por fix (IDENT / FREQ)")
if "navaids" not in st.session_state:
    st.session_state.navaids = [{"IDENT":"", "FREQ":""} for _ in range(max(1, N+1))]
nav_view = st.data_editor(
    st.session_state.navaids, hide_index=True, use_container_width=True,
    column_config={"IDENT": st.column_config.TextColumn("IDENT"),
                   "FREQ":  st.column_config.TextColumn("FREQ")}
)
st.session_state.navaids = nav_view

# ============== Cálculo vertical + cortes ==============
def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0
dep_elev = aero_elev(dept); arr_elev = aero_elev(arr)
start_alt = float(dep_elev); end_alt = float(arr_elev)
pa_start = pressure_alt(start_alt, qnh)
pa_cruise = pressure_alt(cruise_alt, qnh)
vy_kt = vy_interp_enroute(pa_start)
tas_climb, tas_cruise, tas_descent = vy_kt, float(cruise_ref_kt), float(descent_ref_kt)

roc = roc_interp_enroute(pa_start, temp_c)
delta_climb = max(0.0, cruise_alt - start_alt)
delta_desc  = max(0.0, cruise_alt - end_alt)
t_climb_total = delta_climb / max(roc,1e-6)
t_desc_total  = delta_desc  / max(rod_fpm,1e-6)

pa_mid_climb = start_alt + 0.5*delta_climb
pa_mid_desc  = end_alt   + 0.5*delta_desc
_, ff_climb  = cruise_lookup(pa_mid_climb, int(rpm_climb),  temp_c)
_, ff_cruise = cruise_lookup(pa_cruise,   int(rpm_cruise),  temp_c)
ff_descent   = float(idle_ff) if idle_mode else float(descent_ff)

def gs_for(tc, tas): return wind_triangle(float(tc), float(tas), wind_from, wind_kt)[2]
dist = [float(l["Dist"] or 0.0) for l in legs]
gs_climb   = [gs_for(legs[i]["TC"], tas_climb)   for i in range(N)]
gs_cruise  = [gs_for(legs[i]["TC"], tas_cruise)  for i in range(N)]
gs_descent = [gs_for(legs[i]["TC"], tas_descent) for i in range(N)]

climb_nm   = [0.0]*N; descent_nm = [0.0]*N
idx_toc = None; idx_tod = None

rem_t = float(t_climb_total)
for i in range(N):
    if rem_t <= 1e-9: break
    gs = max(gs_climb[i], 1e-6)
    t_full = 60.0 * dist[i] / gs
    use_t = min(rem_t, t_full)
    climb_nm[i] = min(dist[i], gs * use_t / 60.0)
    rem_t -= use_t
    if rem_t <= 1e-9: idx_toc = i; break

rem_t = float(t_desc_total)
for j in range(N-1, -1, -1):
    if rem_t <= 1e-9: break
    gs = max(gs_descent[j], 1e-6)
    t_full = 60.0 * dist[j] / gs
    use_t = min(rem_t, t_full)
    descent_nm[j] = min(dist[j], gs * use_t / 60.0)
    rem_t -= use_t
    if rem_t <= 1e-9: idx_tod = j; break

startup = parse_hhmm(startup_str)
takeoff = add_minutes(startup,15) if startup else None
clock = takeoff
def ceil_pos_minutes(x):  # arredonda ↑ e garante 1 min quando >0
    return max(1, int(math.ceil(x - 1e-9))) if x > 0 else 0

rows=[]; seq_points=[]; calc_steps=[]
PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓"}
alt_cursor = float(start_alt); total_ete = total_burn = 0.0
efob=float(start_fuel)

def add_segment(phase:str, from_nm:str, to_nm:str, i_leg:int, d_nm:float, tas:float, ff_lph:float):
    global clock, total_ete, total_burn, efob, alt_cursor
    if d_nm <= 1e-9: return
    tc = float(legs[i_leg]["TC"])
    wca, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
    ete_raw = 60.0 * d_nm / max(gs,1e-6)
    ete = ceil_pos_minutes(ete_raw)
    burn = ff_lph * (ete_raw/60.0)

    alt_start = alt_cursor
    if phase == "CLIMB":
        alt_end = min(cruise_alt, alt_start + roc * ete_raw)
    elif phase == "DESCENT":
        alt_end = max(end_alt,   alt_start - rod_fpm * ete_raw)
    else:
        alt_end = alt_start

    eto = ""
    if clock:
        clock = add_minutes(clock, ete); eto = clock.strftime("%H:%M")
    total_ete += ete; total_burn += burn
    efob = max(0.0, efob - burn); alt_cursor = alt_end

    rows.append({
        "Fase": PH_ICON[phase], "Leg/Marker": f"{from_nm}→{to_nm}",
        "ALT (ft)": f"{fmt(alt_start,'default')}→{fmt(alt_end,'default')}",
        "TC (°T)": _round_angle(tc), "TH (°T)": _round_angle(th),
        "MH (°M)": _round_angle(apply_var(th, var_deg, var_is_e)),
        "TAS (kt)": _round_unit(tas), "GS (kt)": _round_unit(gs),
        "FF (L/h)": _round_unit(ff_lph), "Dist (nm)": _round_default(d_nm),
        "ETE (min)": int(ete), "ETO": eto, "Burn (L)": _round_unit(burn),
        "EFOB (L)": _round_unit(efob)
    })

    seq_points.append({
        "name": to_nm, "alt": _round_default(alt_end),
        "tc": _round_angle(tc), "th": _round_angle(th),
        "mh": _round_angle(apply_var(th, var_deg, var_is_e)),
        "tas": _round_unit(tas), "gs": _round_unit(gs),
        "dist": d_nm, "ete": int(ete), "eto": eto, "burn": burn, "efob": efob
    })

    # Passos (texto curto) p/ relatório
    calc_steps.append([
        from_nm, to_nm, phase,
        f"{_round_angle(tc)}°", f"{_round_angle(apply_var(th, var_deg, var_is_e))}°",
        _round_unit(tas), _round_unit(gs),
        _round_default(d_nm), int(ete), eto or "—",
        _round_unit(burn), _round_unit(efob),
        f"{fmt(alt_start,'default')}→{fmt(alt_end,'default')}"
    ])

# DEP para a 1ª linha sem métrica
seq_points.append({"name": dept, "alt": _round_default(start_alt),
                   "tc":"", "th":"", "mh":"", "tas":"", "gs":"", "dist":"", "ete":"", "eto": (takeoff.strftime("%H:%M") if takeoff else ""), "burn":"", "efob": efob})

for i in range(N):
    leg_from, leg_to = legs[i]["From"], legs[i]["To"]
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

eta = clock; landing = eta; shutdown = add_minutes(eta,5) if eta else None

# ===== Tabela rápida na App =====
st.markdown("### Flight plan — cortes dentro do leg (App)")
st.dataframe(rows, use_container_width=True)

tot_ete_m = int(sum(int(r['ETE (min)']) for r in rows))
tot_nm  = sum(float(p['dist']) for p in seq_points if isinstance(p.get('dist'), (int,float)))
tot_bo  = sum(float(r['Burn (L)']) for r in rows)
st.markdown(
    f"**Totais** — Dist {fmt(tot_nm,'default')} nm • "
    f"ETE {tot_ete_m//60:02d}:{tot_ete_m%60:02d} • "
    f"Burn {fmt(tot_bo,'fuel')} L • EFOB {fmt(efob,'fuel')} L"
)
if eta:
    st.markdown(f"**ETA {eta.strftime('%H:%M')}** • **Landing {landing.strftime('%H:%M')}** • **Shutdown {shutdown.strftime('%H:%M')}**")

# Ajustar NAVAIDS ao nº real de linhas (DEP + chegadas, incluindo TOC/TOD)
if len(st.session_state.navaids) < len(seq_points):
    st.session_state.navaids += [{"IDENT":"", "FREQ":""} for _ in range(len(seq_points)-len(st.session_state.navaids))]
elif len(st.session_state.navaids) > len(seq_points):
    st.session_state.navaids = st.session_state.navaids[:len(seq_points)]

# ============== PDF export ==============
st.markdown("### PDF export (apenas PDF)")
show_debug = st.checkbox("Mostrar debug de linhas (posições)")

try:
    template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)
except Exception as e:
    template_bytes = None
    st.error(f"Não foi possível ler o PDF de template: {e}")

pdf_bytes_out = None
report_bytes_out = None

if not template_bytes:
    st.info("Coloca o ficheiro NAVLOG_FORM.pdf no diretório da app.")
elif not PYPDF_OK:
    st.error("A biblioteca **pypdf** não está instalada. Instala com: `pip install pypdf`.")
else:
    try:
        fieldset, maxlens, fields_pos = get_form_fields(template_bytes)
    except Exception as e:
        st.error(f"Falha a ler os campos do PDF (pypdf): {e}")
        fieldset, maxlens, fields_pos = set(), {}, []

    if show_debug and fields_pos:
        st.write(f"N.º total de campos: {len(fieldset)} | com posições: {len(fields_pos)}")

    if fieldset and fields_pos:
        # Detectar micro-rows (top/bottom alternadas), por página
        micro_rows = fields_to_micro_rows(fields_pos)
        if show_debug:
            st.write(f"Micro-rows detectadas: {len(micro_rows)} (cada linha da grelha deverá ter 2 micro-rows)")

        # Mapeamento por micro-rows (top/bottom)
        named: Dict[str,str] = {}

        # ===== Cabeçalho (se existirem) =====
        header_candidates = {
            "FLIGHT LEVEL / ALTITUDE": fmt(cruise_alt,'default'),
            "WIND": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
            "MAG VAR": f"{int(round(var_deg))}{'E' if var_is_e else 'W'}",
            "TEMP / ISA DEV.": f"{fmt(temp_c,'default')} / {fmt(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)),'default')}",
            "Dept_Airfield": dept, "Arrival_Airfield": arr,
            "Alternate_Airfield": altn, "Alt_Alternate": fmt(aero_elev(altn),'default'),
            "Startup": startup_str,
            "Takeoff": (add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else "")
        }
        for k,v in header_candidates.items():
            put(named, fieldset, k, v, maxlens)

        # Constrói itens para escrita (DEP + chegadas)
        # Cada linha lógica = 2 micro-rows: top (leg metrics) + bottom (second line)
        def write_row(r_idx: int, top: List[dict], bottom: List[dict], item: dict):
            # FIX / ALT
            f_fix = find_first_in_row(top, ["FIX","NAME"])
            if f_fix: put(named, fieldset, f_fix, item["Name"], maxlens)
            f_alt = find_first_in_row(top, ["ALT"])
            if f_alt: put(named, fieldset, f_alt, item["Alt"], maxlens)

            # NAVAIDS IDENT (top) / FREQ (bottom)
            f_nav_top  = find_first_in_row(top, ["NAVAIDS"])
            f_nav_bot  = find_first_in_row(bottom, ["NAVAIDS"])
            ident = (st.session_state.navaids[r_idx]["IDENT"] if r_idx < len(st.session_state.navaids) else "")
            freq  = (st.session_state.navaids[r_idx]["FREQ"]  if r_idx < len(st.session_state.navaids) else "")
            if f_nav_top: put(named, fieldset, f_nav_top, ident, maxlens)
            if f_nav_bot: put(named, fieldset, f_nav_bot, freq, maxlens)

            # DIRECTION
            f_tcrs = find_first_in_row(top, ["T CRS","TCRS"])
            f_mcrs = find_first_in_row(bottom, ["M CRS","MCRS"])
            if item["TC"]!="" and f_tcrs: put(named, fieldset, f_tcrs, item["TC"], maxlens)
            if item["MH"]!="" and f_mcrs: put(named, fieldset, f_mcrs, item["MH"], maxlens)

            # SPEED: GS (top), TAS (bottom)
            f_gs_top = find_first_in_row(top, ["GS"]) or find_first_in_row(top, ["SPEED"])
            f_tas_bot= find_first_in_row(bottom, ["TAS"]) or find_first_in_row(bottom, ["SPEED"])
            if item["GS"]!="" and f_gs_top: put(named, fieldset, f_gs_top, item["GS"], maxlens)
            if item["TAS"]!="" and f_tas_bot: put(named, fieldset, f_tas_bot, item["TAS"], maxlens)

            # DIST: LEG (top), ACC (bottom)
            f_dist_top = find_first_in_row(top, ["DIST"])
            f_dist_acc = find_first_in_row(bottom, ["DIST","ACC"])  # alguns templates usam ACC
            if item["Dist_leg"]!="" and f_dist_top: put(named, fieldset, f_dist_top, item["Dist_leg"], maxlens)
            if item["Dist_acc"]!="" and f_dist_acc: put(named, fieldset, f_dist_acc, item["Dist_acc"], maxlens)

            # TIME: ETE & ETO (top), ACC (bottom)
            f_ete = find_first_in_row(top, ["ETE"])
            f_eto = find_first_in_row(top, ["ETO"])
            f_acc = find_first_in_row(bottom, ["ACC"])
            if item["ETE"]!="" and f_ete: put(named, fieldset, f_ete, item["ETE"], maxlens)
            if item["ETO"]!="" and f_eto: put(named, fieldset, f_eto, item["ETO"], maxlens)
            if item["ACC_time"]!="" and f_acc: put(named, fieldset, f_acc, item["ACC_time"], maxlens)
            # ATO/RETO/DIFF ficam vazios

            # FUEL: Pl B/O + EFOB (top)
            f_pbo = find_first_in_row(top, ["PL B/O","Pl B/O","PL_BO","PL BO"])
            f_efob= find_first_in_row(top, ["EFOB"])
            if item["Burn"]!="" and f_pbo:  put(named, fieldset, f_pbo,  item["Burn"], maxlens)
            if item["EFOB"]!="" and f_efob: put(named, fieldset, f_efob, item["EFOB"], maxlens)
            # Act B/O, AFOB (bottom) — não preencher

        # Preparar itens formatados (DEP + segmentos)
        def build_pdf_items(seq_points):
            items=[]
            acc_dist = 0.0
            acc_time = 0
            for idx, p in enumerate(seq_points):
                is_seg = (idx>0)
                if is_seg and isinstance(p["dist"], (int,float)): acc_dist += float(p["dist"])
                if is_seg: acc_time += int(p["ete"] or 0)
                it = {
                    "Name": p["name"],
                    "Alt": fmt(p["alt"], 'default') if p["alt"]!="" else "",
                    "TC":  (fmt(p["tc"], 'angle') if is_seg else ""),
                    "MH":  (fmt(p["mh"], 'angle') if is_seg else ""),
                    "GS":  (fmt(p["gs"], 'speed') if is_seg else ""),
                    "TAS": (fmt(p["tas"], 'speed') if is_seg else ""),
                    "Dist_leg": (fmt(p["dist"], 'default') if is_seg and isinstance(p["dist"], (int,float)) else ""),
                    "Dist_acc": (fmt(acc_dist, 'default') if is_seg else ""),
                    "ETE":  (fmt(p["ete"], 'mins') if is_seg else ""),
                    "ETO":  (p["eto"] if is_seg else (p["eto"] or "")),
                    "ACC_time": (fmt(acc_time, 'mins') if is_seg else ""),
                    "Burn": (fmt(p["burn"], 'fuel') if is_seg and isinstance(p["burn"], (int,float)) else ""),
                    "EFOB": (fmt(p["efob"], 'fuel') if isinstance(p["efob"], (int,float)) else fmt(p["efob"],'fuel') if idx==0 else "")
                }
                items.append(it)
            return items

        pdf_items = build_pdf_items(seq_points)

        # Escrever por pares de micro-rows
        total_pairs = len(micro_rows)//2
        use_rows = min(len(pdf_items), total_pairs)
        if show_debug:
            st.write(f"Linhas lógicas disponíveis: {total_pairs} | a escrever: {use_rows}")

        for i in range(use_rows):
            top_row = micro_rows[2*i]
            bot_row = micro_rows[2*i+1]
            write_row(i, top_row, bot_row, pdf_items[i])

        # Totais e tempos finais (se existirem campos)
        last_eto = pdf_items[-1]["ETO"] if pdf_items else ""
        put(named, fieldset, "LANDING", last_eto, maxlens)
        put(named, fieldset, "SHUTDOWN", (add_minutes(parse_hhmm(last_eto),5).strftime("%H:%M") if last_eto else ""), maxlens)
        put(named, fieldset, "ETD/ETA", f"{(add_minutes(parse_hhmm(startup_str),15).strftime('%H:%M') if startup_str else '')} / {last_eto}", maxlens)
        tot_min = sum(int(it["ETE"] or "0") for it in pdf_items)
        put(named, fieldset, "FLT TIME", f"{tot_min//60:02d}:{tot_min%60:02d}", maxlens)
        put(named, fieldset, "CLIMB FUEL", fmt((ff_climb*(t_climb_total/60.0)),'fuel'), maxlens)

        # ===== Gerar PDF =====
        if st.button("Gerar PDF preenchido", type="primary"):
            try:
                pdf_bytes_out = fill_pdf(template_bytes, named)
                # Nome do ficheiro — pelo número da Lesson (não pela matrícula)
                # Se capturarmos dígitos, usamos só o número; senão o texto da Lesson sanitizado.
                m = re.search(r'(\d+)', lesson or "")
                lesson_id = m.group(1) if m else ascii_safe(lesson or "LESSON")
                safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
                filename_pdf = f"{safe_date}_LESSON-{lesson_id}_NAVLOG.pdf"
                st.download_button("📄 Download PDF", data=pdf_bytes_out, file_name=filename_pdf, mime="application/pdf")
                st.success("PDF gerado (apenas planeado). Revê antes do voo.")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

# ============== Relatório legível (PDF) ==============
def build_report_pdf_human(calc_rows: List[List], params: Dict[str,str]) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab missing")
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]; H2 = styles["Heading2"]; H3 = styles["Heading3"]; P = styles["BodyText"]

    story=[]
    story.append(Paragraph("NAVLOG — Relatório de Cálculos (Planeado)", H1))
    story.append(Spacer(1,6))

    # Resumo do voo
    resume = [
        ["Aeronave", params.get("aircraft","—")],
        ["Matrícula", params.get("registration","—")],
        ["Callsign", params.get("callsign","—")],
        ["Lição", params.get("lesson","—")],
        ["Partida", params.get("dept","—")],
        ["Chegada", params.get("arr","—")],
        ["Alternante", params.get("altn","—")],
        ["Nível/Alt Cruzeiro", params.get("cruise_alt","—") + " ft"],
        ["QNH", params.get("qnh","—")],
        ["Vento", params.get("wind","—")],
        ["Var. Magn.", params.get("var","—")],
        ["OAT / ISA dev", params.get("temp_isa","—")],
        ["Startup", params.get("startup","—")],
        ["ETD", params.get("etd","—")],
        ["ETA", params.get("eta","—")],
        ["Shutdown", params.get("shutdown","—")],
        ["Tempo total (PLN)", params.get("flt_time","—")],
        ["Fuel inicial", params.get("start_fuel","—")+" L"],
    ]
    t = Table(resume, hAlign="LEFT", colWidths=[45*mm, None])
    t.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("LEFTPADDING",(0,0),(-1,-1),4),
        ("RIGHTPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(t)
    story.append(Spacer(1,8))

    story.append(Paragraph("Fórmulas utilizadas (vento FROM):", H2))
    formulae = [
        "WCA = asin( (W/TAS) · sin(θwind_from − θcourse_true) )",
        "TH  = TC + WCA",
        "GS  = TAS·cos(WCA) − W·cos(θwind_from − θcourse_true)",
        "ETE (min) = ceil( 60 · Dist / GS ), com mínimo 1 se Dist > 0",
        "Burn (L) = FF (L/h) · (ETE_real/60)  [usamos ETE_real antes do arredondamento]",
    ]
    for f in formulae:
        story.append(Paragraph(f"• {f}", P))
    story.append(Spacer(1,6))

    story.append(Paragraph("Segmentos (por ordem):", H2))
    # Tabela de segmentos legível
    data = [["From→To","Fase","TC°","MH°","TAS","GS","Dist","ETE","ETO","Burn","EFOB","ALT ini→fim"]]
    for r in calc_rows:
        data.append(r)
    ts = TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("ALIGN",(2,1),(5,-1),"RIGHT"),
        ("ALIGN",(6,1),(10,-1),"RIGHT"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),8.7),
        ("LEFTPADDING",(0,0),(-1,-1),3),
        ("RIGHTPADDING",(0,0),(-1,-1),3),
    ])
    tbl = Table(data, hAlign="LEFT",
                colWidths=[34*mm, 12*mm, 10*mm, 10*mm, 10*mm, 10*mm, 12*mm, 10*mm, 14*mm, 12*mm, 12*mm, 24*mm])
    tbl.setStyle(ts)
    story.append(tbl)

    doc.build(story)
    return bio.getvalue()

if st.button("Gerar Relatório (PDF legível)"):
    try:
        params = {
            "aircraft": aircraft, "registration": registration, "callsign": callsign,
            "lesson": lesson,
            "dept": dept, "arr": arr, "altn": altn,
            "qnh": fmt(qnh,'default'),
            "wind": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
            "var": f"{int(round(var_deg))}{'E' if var_is_e else 'W'}",
            "cruise_alt": fmt(cruise_alt,'default'),
            "temp_isa": f"{fmt(temp_c,'default')} / {fmt(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)),'default')}",
            "startup": startup_str,
            "etd": (add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else ""),
            "eta": (eta.strftime("%H:%M") if eta else ""), "shutdown": (shutdown.strftime("%H:%M") if shutdown else ""),
            "flt_time": f"{tot_ete_m//60:02d}:{tot_ete_m%60:02d}",
            "start_fuel": fmt(start_fuel,'fuel')
        }
        # Transforma calc_steps (lista de listas) no formato esperado: [From→To, Fase, ...]
        calc_rows = []
        for r in calc_steps:
            # r = [from, to, phase, TC°, MH°, TAS, GS, Dist, ETE, ETO, Burn, EFOB, ALTini→fim]
            calc_rows.append([f"{r[0]}→{r[1]}", r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12]])
        report_bytes_out = build_report_pdf_human(calc_rows, params)
        # Nomeia relatório também pela Lesson
        m = re.search(r'(\d+)', lesson or "")
        lesson_id = m.group(1) if m else ascii_safe(lesson or "LESSON")
        safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
        filename_rep = f"{safe_date}_LESSON-{lesson_id}_NAVLOG_RELATORIO.pdf"
        st.download_button("📑 Download Relatório (PDF)", data=report_bytes_out, file_name=filename_rep, mime="application/pdf")
        st.success("Relatório gerado.")
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {e}")

