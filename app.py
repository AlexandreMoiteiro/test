# app.py — NAVLOG planeado (TOC/TOD), preenche PDF, exporta JPEG e relatório de cálculos
# Reqs: streamlit, pypdf, pymupdf, reportlab, pytz
import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod

# =================== Config ===================
st.set_page_config(page_title="NAVLOG (PDF→JPEG + Relatório)", layout="wide", initial_sidebar_state="collapsed")
PDF_TEMPLATE_PATHS = ["NAVLOG_FORM.pdf"]   # nome do teu template

# =================== Imports opcionais ===================
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import mm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# =================== Arredondamento / Formatação ===================
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

# =================== Helpers ===================
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

# =================== AFM (igual ao teu) ===================
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

# =================== Vento & variação ===================
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

# =================== Aeródromos (exemplo) ===================
AEROS={
 "LPSO":{"elev":390,"freq":"119.805"},
 "LPEV":{"elev":807,"freq":"122.705"},
 "LPCB":{"elev":1251,"freq":"122.300"},
 "LPCO":{"elev":587,"freq":"118.405"},
 "LPVZ":{"elev":2060,"freq":"118.305"},
}
def aero_elev(icao): return int(AEROS.get(icao,{}).get("elev",0))
def aero_freq(icao): return AEROS.get(icao,{}).get("freq","")

# =================== PDF helpers ===================
def read_pdf_bytes(paths: List[str]) -> bytes:
    for p in paths:
        if Path(p).exists():
            return Path(p).read_bytes()
    raise FileNotFoundError(paths)

def get_fields_and_meta(template_bytes: bytes):
    """Lê nomes e MaxLen dos campos do formulário do PDF."""
    if not PYPDF_OK:
        raise RuntimeError("Dependência ausente: pypdf. Instala com: pip install pypdf")
    reader = PdfReader(io.BytesIO(template_bytes))
    field_names, maxlens = set(), {}
    try:
        fd = reader.get_fields() or {}
        field_names |= set(fd.keys())
        for k, v in fd.items():
            ml = v.get("/MaxLen")
            if ml:
                maxlens[k] = int(ml)
    except Exception:
        pass
    try:
        for page in reader.pages:
            if "/Annots" in page:
                for a in page["/Annots"]:
                    obj = a.get_object()
                    if obj.get("/T"):
                        nm = str(obj["/T"])
                        field_names.add(nm)
                        ml = obj.get("/MaxLen")
                        if ml:
                            maxlens[nm] = int(ml)
    except Exception:
        pass
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

def pdf_to_jpeg_bytes(pdf_bytes: bytes, page_index: int = 0, dpi: int = 220) -> bytes:
    if not PYMUPDF_OK: raise RuntimeError("pymupdf (fitz) missing")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("jpg")  # bytes JPG

# util: escreve se existir (respeita MaxLen)
def put(out: dict, fieldset: set, key: str, value: str, maxlens: Dict[str,int]):
    if key in fieldset:
        s = "" if value is None else str(value)
        if key in maxlens and len(s) > maxlens[key]:
            s = s[:maxlens[key]]
        out[key] = s

# =================== Field matching robusto ===================
def canon(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', (s or '').upper())

def series(fieldset: set, base_names: List[str]) -> List[str]:
    """Devolve todos os nomes que começam por algum base (canon) ordenados por sufixo numérico (""->0)."""
    bases = [canon(b) for b in base_names]
    pairs=[]
    for k in fieldset:
        ck = canon(k)
        for b in bases:
            if ck.startswith(b):
                m = re.search(r'(\d+)$', k)  # sufixo numérico (ou nada)
                n = int(m.group(1)) if m else 0
                pairs.append((n, k))
                break
    pairs.sort(key=lambda x: x[0])
    return [k for _,k in pairs]

def get_row_field(fields_list: List[str], row_index: int, subindex: int = 0, per_row: int = 1) -> Optional[str]:
    """Campo na 'row_index' considerando 'per_row' entradas por linha e 'subindex' (0..per_row-1)."""
    pos = row_index*per_row + subindex
    if pos < len(fields_list):
        return fields_list[pos]
    return None

# =================== UI ===================
st.title("Navigation Plan & Inflight Log — Tecnam P2008 (PDF→JPEG + Relatório)")

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
# navaids para [DEP] + cada chegada (inclui TOC/TOD quando houver cortes)
if "navaids" not in st.session_state:
    st.session_state.navaids = [{"IDENT":"", "FREQ":""} for _ in range(max(1, N+1))]
nav_view = st.data_editor(
    st.session_state.navaids, hide_index=True, use_container_width=True,
    column_config={"IDENT": st.column_config.TextColumn("IDENT"),
                   "FREQ":  st.column_config.TextColumn("FREQ")}
)
st.session_state.navaids = nav_view

# =================== Cálculo vertical + cortes ===================
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

    # Para relatório (com valores crús + formatados)
    calc_steps.append(
        f"[{from_nm}→{to_nm} / {phase}] "
        f"TC={tc:.1f}°, WCA=asin((W/TAS)*sin Δ)={wca:.2f}°, TH=TC+WCA={th:.2f}°, "
        f"GS=TAS*cos(WCA)-W*cos Δ={gs:.2f} kt; Dist={d_nm:.2f} nm → "
        f"ETE_raw=60*D/GS={ete_raw:.2f} min → ETE={ete} min; "
        f"Burn={ff_lph:.2f}*(ETE_raw/60)={burn:.2f} L; "
        f"ALT {alt_start:.0f}->{alt_end:.0f} ft; ETO={eto}; EFOB={efob:.2f} L"
    )

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

# ===== PDF export =====
st.markdown("### PDF export")
show_fields = st.checkbox("Mostrar nomes de campos do PDF (debug)")

# carregar template
try:
    template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)
except Exception as e:
    template_bytes = None
    st.error(f"Não foi possível ler o PDF de template: {e}")

pdf_bytes_out = None
jpeg_bytes_out = None
report_bytes_out = None

if not template_bytes:
    st.info("Coloca o ficheiro NAVLOG_FORM.pdf no diretório da app.")
elif not PYPDF_OK:
    st.error("A biblioteca **pypdf** não está instalada. Instala com: `pip install pypdf`.")
else:
    try:
        fieldset, maxlens = get_fields_and_meta(template_bytes)
    except Exception as e:
        st.error(f"Falha a ler os campos do PDF (pypdf): {e}")
        fieldset, maxlens = set(), {}

    if show_fields and fieldset:
        st.code("\n".join(sorted(fieldset)))

    if fieldset:
        # coleções por coluna (detecção robusta)
        FIX_FIELDS    = series(fieldset, ["FIX","NAME"])
        NAVAIDS_ALL   = series(fieldset, ["NAVAIDS"])
        ALT_FIELDS    = series(fieldset, ["ALT"])

        # DIRECTION: preferir listas separadas; fallback para base única com 2 col/linha
        TCRS_FIELDS   = series(fieldset, ["T CRS","TCRS","TRUE CRS","TRUE COURSE"])
        MCRS_FIELDS   = series(fieldset, ["M CRS","MCRS","MAG CRS","MAG COURSE"])
        DIR_FIELDS    = series(fieldset, ["CRS","COURSE","DIRECTION"])  # fallback 2 por linha: True (par/1ª), Mag (ímpar/2ª)

        SPEED_FIELDS  = series(fieldset, ["SPEED"])   # se houver 2/linha (GS/TAS)
        GS_FIELDS     = series(fieldset, ["GS"])
        TAS_FIELDS    = series(fieldset, ["TAS"])

        DIST_FIELDS   = series(fieldset, ["DIST"])    # 2/linha: LEG (top), ACC (bottom)
        ETE_FIELDS    = series(fieldset, ["ETE"])
        ETO_FIELDS    = series(fieldset, ["ETO"])
        ACC_FIELDS    = series(fieldset, ["ACC"])     # tempo acumulado

        PBO_FIELDS    = series(fieldset, ["PL B/O","Pl B/O","PL_BO","PL BO"])
        EFOB_FIELDS   = series(fieldset, ["EFOB"])
        # Campos Actual (deixar vazio)
        # ATO/RETO/DIFF/Act B/O/AFOB intencionalmente não escritos

        def write_field(named, key, value):
            put(named, fieldset, key, value, maxlens)

        # ===== Cabeçalho básico (se existirem) =====
        named: Dict[str,str] = {}
        # Info de topo (alguns templates têm estes nomes)
        for k, v in {
            "FLIGHT LEVEL / ALTITUDE": fmt(cruise_alt,'default'),
            "WIND": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
            "MAG VAR": f"{int(round(var_deg))}{'E' if var_is_e else 'W'}",
            "TEMP / ISA DEV.": f"{fmt(temp_c,'default')} / {fmt(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)),'default')}",
            "Dept_Airfield": dept, "Arrival_Airfield": arr,
            "Alternate_Airfield": altn, "Alt_Alternate": fmt(aero_elev(altn),'default'),
            "Startup": startup_str,
            "Takeoff": (add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else "")
        }.items():
            write_field(named, k, v)

        # ===== Linhas (DEP + segmentos) =====
        def build_pdf_items_from_points(points):
            """Cada item é o ponto de chegada; idx 0 = DEP."""
            items = []
            acc_dist = 0.0
            acc_time = 0
            for idx, p in enumerate(points):  # 0 = DEP
                is_seg = (idx > 0)
                if is_seg and isinstance(p["dist"], (int,float)):
                    acc_dist += float(p["dist"])
                if is_seg:
                    acc_time += int(p["ete"] or 0)
                it = {
                    "Name": p["name"],
                    "Alt": fmt(p["alt"], 'default') if p["alt"] != "" else "",
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

        pdf_items = build_pdf_items_from_points(seq_points)
        rows_count = len(pdf_items)  # DEP + chegadas

        for r in range(rows_count):
            it = pdf_items[r]
            # ---- FIX / ALT
            if r < len(FIX_FIELDS):  write_field(named, FIX_FIELDS[r], it["Name"])
            if r < len(ALT_FIELDS):  write_field(named, ALT_FIELDS[r], it["Alt"])

            # ---- NAVAIDS: 2 por linha (IDENT / FREQ) vindos da UI
            ident = (st.session_state.navaids[r]["IDENT"] if r < len(st.session_state.navaids) else "")
            freq  = (st.session_state.navaids[r]["FREQ"]  if r < len(st.session_state.navaids) else "")
            ident_field = get_row_field(NAVAIDS_ALL, r, 0, per_row=2)
            freq_field  = get_row_field(NAVAIDS_ALL, r, 1, per_row=2)
            if ident_field: write_field(named, ident_field, ident)
            if freq_field:  write_field(named, freq_field,  freq)

            # ---- DIRECTION (T CRS / M CRS)
            if r < len(TCRS_FIELDS) and it["TC"]!="":
                write_field(named, TCRS_FIELDS[r], it["TC"])
            if r < len(MCRS_FIELDS) and it["MH"]!="":
                write_field(named, MCRS_FIELDS[r], it["MH"])
            # Fallback: uma única base (2 col/linha): True na 1ª, Mag na 2ª
            if len(TCRS_FIELDS)==0 and len(MCRS_FIELDS)==0 and len(DIR_FIELDS) >= 2*rows_count:
                dir_true = get_row_field(DIR_FIELDS, r, 0, per_row=2)
                dir_mag  = get_row_field(DIR_FIELDS, r, 1, per_row=2)
                if dir_true and it["TC"]!="": write_field(named, dir_true, it["TC"])
                if dir_mag  and it["MH"]!="": write_field(named, dir_mag,  it["MH"])

            # ---- SPEED: GS/TAS (2 por linha se só houver "SPEED_*"; ou listas GS_/TAS_)
            if len(GS_FIELDS) >= rows_count and r < len(GS_FIELDS):
                write_field(named, GS_FIELDS[r], it["GS"])
            if len(TAS_FIELDS) >= rows_count and r < len(TAS_FIELDS):
                write_field(named, TAS_FIELDS[r], it["TAS"])
            if len(SPEED_FIELDS) >= 2*rows_count:
                sp_gs = get_row_field(SPEED_FIELDS, r, 0, per_row=2)
                sp_tas= get_row_field(SPEED_FIELDS, r, 1, per_row=2)
                if sp_gs:  write_field(named, sp_gs,  it["GS"])
                if sp_tas: write_field(named, sp_tas, it["TAS"])

            # ---- DIST: LEG (alto) + ACC (baixo)
            if len(DIST_FIELDS) >= 2*rows_count:
                d_leg = get_row_field(DIST_FIELDS, r, 0, per_row=2)
                d_acc = get_row_field(DIST_FIELDS, r, 1, per_row=2)
                if d_leg: write_field(named, d_leg, it["Dist_leg"])
                if d_acc: write_field(named, d_acc, it["Dist_acc"])

            # ---- TIME: ETE / ETO (top), ACC (bottom). ATO/RETO/DIFF ficam vazios.
            if r < len(ETE_FIELDS): write_field(named, ETE_FIELDS[r], it["ETE"])
            if r < len(ETO_FIELDS): write_field(named, ETO_FIELDS[r], it["ETO"])
            if r < len(ACC_FIELDS): write_field(named, ACC_FIELDS[r], it["ACC_time"])

            # ---- FUEL: Pl B/O + EFOB (top). Act B/O + AFOB ficam vazios.
            if r < len(PBO_FIELDS):  write_field(named, PBO_FIELDS[r], it["Burn"])
            if r < len(EFOB_FIELDS): write_field(named, EFOB_FIELDS[r], it["EFOB"])

        # ===== Totais e tempos finais =====
        last_eto = pdf_items[-1]["ETO"] if pdf_items else ""
        put(named, fieldset, "LANDING", last_eto, maxlens)
        put(named, fieldset, "SHUTDOWN", (add_minutes(parse_hhmm(last_eto),5).strftime("%H:%M") if last_eto else ""), maxlens)
        put(named, fieldset, "ETD/ETA", f"{(add_minutes(parse_hhmm(startup_str),15).strftime('%H:%M') if startup_str else '')} / {last_eto}", maxlens)

        # Totais planeados
        tot_min = sum(int(it["ETE"] or "0") for it in pdf_items)
        put(named, fieldset, "FLT TIME", f"{tot_min//60:02d}:{tot_min%60:02d}", maxlens)
        put(named, fieldset, "CLIMB FUEL", fmt((ff_climb*(t_climb_total/60.0)),'fuel'), maxlens)

        # ===== Botão: gerar PDF + JPEG =====
        if st.button("Gerar PDF preenchido + JPEG", type="primary"):
            try:
                pdf_bytes_out = fill_pdf(template_bytes, named)
                safe_reg = ascii_safe(registration)
                safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
                filename_pdf = f"{safe_date}_{safe_reg}_NAVLOG.pdf"
                st.download_button("📄 Download PDF", data=pdf_bytes_out, file_name=filename_pdf, mime="application/pdf")

                if not PYMUPDF_OK:
                    st.warning("PyMuPDF não instalado — sem JPEG. Instala com: `pip install pymupdf`.")
                else:
                    jpeg_bytes_out = pdf_to_jpeg_bytes(pdf_bytes_out, page_index=0, dpi=220)
                    filename_jpg = f"{safe_date}_{safe_reg}_NAVLOG.jpg"
                    st.image(jpeg_bytes_out, caption="Pré-visualização NAVLOG (JPEG)", use_column_width=True)
                    st.download_button("🖼️ Download JPEG", data=jpeg_bytes_out, file_name=filename_jpg, mime="image/jpeg")
            except Exception as e:
                st.error(f"Erro ao gerar PDF/JPEG: {e}")

# ===== Relatório PDF dos cálculos =====
def build_report_pdf(calc_steps: List[str], rows, seq_points, params: Dict[str,str]) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab missing")
    bio = io.BytesIO()
    c = rl_canvas.Canvas(bio, pagesize=A4)
    W, H = A4
    x, y = 20*mm, H-20*mm
    def line(txt, size=10, lead=12):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(x, y, txt)
        y -= lead
        if y < 20*mm:
            c.showPage()
            y = H-20*mm

    c.setTitle("NAVLOG — Relatório de Cálculos")
    c.setAuthor("NAVLOG App")

    # Cabeçalho
    line("NAVLOG — Relatório de Cálculos (Planeado)", 14, 18)
    for k in ["aircraft","registration","callsign","dept","arr","altn","qnh","temp","wind","var","cruise_alt","startup","etd","eta","shutdown","flt_time","start_fuel"]:
        if k in params:
            line(f"- {k}: {params[k]}")

    line("", lead=8)
    line("Fórmulas:", 12, 16)
    line(" • WCA = asin( (W/TAS) * sin(θw_from − θc) )")
    line(" • TH  = TC + WCA")
    line(" • GS  = TAS*cos(WCA) − W*cos(θw_from − θc)")
    line(" • ETE (min) = ceil( 60 * Dist / GS ), min 1 se Dist>0")
    line(" • Burn (L) = FF (L/h) * (ETE_real/60)  [usamos ETE_real antes do ceil]")

    line("", lead=8)
    line("Segmentos:", 12, 16)
    for s in calc_steps:
        for chunk in re.findall('.{1,110}(?:\\s+|$)', s):
            line(chunk.strip())

    line("", lead=8)
    line("Resumo por ponto:", 12, 16)
    acc_dist = 0.0; acc_time = 0
    for i, p in enumerate(seq_points):
        if i==0:
            line(f"[DEP {p['name']}] ALT={fmt(p['alt'],'default')} ft, ETO={p['eto']}, EFOB={fmt(p['efob'],'fuel')} L")
        else:
            acc_dist += float(p["dist"])
            acc_time += int(p["ete"])
            line(f"{p['name']}: ALT={fmt(p['alt'],'default')} ft, TC={fmt(p['tc'],'angle')}°, MH={fmt(p['mh'],'angle')}°, "
                 f"TAS={fmt(p['tas'],'speed')} kt, GS={fmt(p['gs'],'speed')} kt, "
                 f"LEG={fmt(p['dist'],'default')} nm, ACC={fmt(acc_dist,'default')} nm, "
                 f"ETE={fmt(p['ete'],'mins')} min, ETO={p['eto']}, Burn={fmt(p['burn'],'fuel')} L, "
                 f"EFOB={fmt(p['efob'],'fuel')} L")

    c.showPage(); c.save()
    return bio.getvalue()

if st.button("Gerar Relatório de Cálculos (PDF)"):
    try:
        params = {
            "aircraft": aircraft, "registration": registration, "callsign": callsign,
            "dept": dept, "arr": arr, "altn": altn, "qnh": fmt(qnh,'default'),
            "temp": fmt(temp_c,'default'), "wind": f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}",
            "var": f"{int(round(var_deg))}{'E' if var_is_e else 'W'}",
            "cruise_alt": fmt(cruise_alt,'default'), "startup": startup_str,
            "etd": (add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else ""),
            "eta": (eta.strftime("%H:%M") if eta else ""), "shutdown": (shutdown.strftime("%H:%M") if shutdown else ""),
            "flt_time": f"{tot_ete_m//60:02d}:{tot_ete_m%60:02d}",
            "start_fuel": fmt(start_fuel,'fuel')
        }
        report_bytes_out = build_report_pdf(calc_steps, rows, seq_points, params)
        safe_reg = ascii_safe(registration)
        safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
        filename_rep = f"{safe_date}_{safe_reg}_NAVLOG_RELATORIO.pdf"
        st.download_button("📑 Download Relatório (PDF)", data=report_bytes_out, file_name=filename_rep, mime="application/pdf")
        st.success("Relatório gerado.")
    except Exception as e:
        st.error(f"Erro ao gerar relatório: {e}")
