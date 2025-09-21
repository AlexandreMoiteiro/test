# app.py — NAVLOG (lê AD-HEL-ULM.csv e Localidades-Nova-versao-230223.csv — formatos reais)
# MAIÚSCULAS, TC/Dist auto, dropdowns de LPxxxx, PDF NAVLOG_FORM.pdf bloqueado
# Reqs: streamlit, pypdf, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod, atan2, cos

# ======================= PDF helpers =======================
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject, NumberObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

def ascii_safe(x: str) -> str:
    return unicodedata.normalize("NFKD", str(x or "")).encode("ascii","ignore").decode("ascii")

def read_pdf_bytes(paths_or_bytes):
    if isinstance(paths_or_bytes, (bytes, bytearray)): return bytes(paths_or_bytes)
    for p in (paths_or_bytes or []):
        if Path(p).exists(): return Path(p).read_bytes()
    raise FileNotFoundError(paths_or_bytes)

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
                        nm = str(obj["/T"])
                        field_names.add(nm)
                        ml = obj.get("/MaxLen")
                        if ml: maxlens[nm] = int(ml)
    except: pass
    return field_names, maxlens

def _ensure_appearances(writer):
    try:
        acro = writer._root_object.get("/AcroForm")
        if acro is not None:
            acro.update({
                NameObject("/NeedAppearances"): True,
                NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g")
            })
    except: pass

def fill_pdf(template_bytes: bytes, fields: dict, make_readonly=True) -> bytes:
    if not PYPDF_OK: raise RuntimeError("pypdf missing")
    reader = PdfReader(io.BytesIO(template_bytes))
    writer = PdfWriter()
    for p in reader.pages: writer.add_page(p)
    root = reader.trailer["/Root"]
    if "/AcroForm" not in root: raise RuntimeError("Template has no AcroForm")
    writer._root_object.update({NameObject("/AcroForm"): root["/AcroForm"]})
    _ensure_appearances(writer)
    str_fields = {k:(str(v) if v is not None else "") for k,v in fields.items()}
    for page in writer.pages:
        writer.update_page_form_field_values(page, str_fields)
    if make_readonly:
        try:
            for page in writer.pages:
                if "/Annots" in page:
                    for a in page["/Annots"]:
                        obj = a.get_object()
                        if obj.get("/T"):
                            ff = int(obj.get("/Ff", 0))
                            obj.update({NameObject("/Ff"): NumberObject(ff | 1)})
        except Exception:
            pass
    bio = io.BytesIO(); writer.write(bio); return bio.getvalue()

def put(out: dict, fieldset: set, key: str, value: str, maxlens: Dict[str,int]):
    if key in fieldset:
        s = "" if value is None else str(value)
        if key in maxlens and len(s) > maxlens[key]: s = s[:maxlens[key]]
        out[key] = s

def pick(fieldset:set, *aliases) -> Optional[str]:
    for nm in aliases:
        if nm in fieldset: return nm
    return None

# ======================= Utilidades numéricas / vento =======================
def wrap360(x): x=fmod(x,360.0); return x+360 if x<0 else x
def angle_diff(a,b): return (a-b+180)%360-180
def clamp(v,lo,hi): return max(lo,min(hi,v))
def interp1(x,x0,x1,y0,y1):
    if x1==x0: return y0
    t=(x-x0)/(x1-x0); return y0+t*(y1-y0)

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

def wind_triangle(tc_deg: float, tas_kt: float, wind_from_deg: float, wind_kt: float):
    if tas_kt <= 0: return 0.0, wrap360(tc_deg), 0.0
    wind_to = wrap360(wind_from_deg + 180.0)
    beta = radians(angle_diff(wind_to, tc_deg))
    cross = wind_kt * sin(beta)
    head  = wind_kt * math.cos(beta)
    s = max(-1.0, min(1.0, cross/max(tas_kt,1e-9)))
    wca = degrees(asin(s))
    th  = wrap360(tc_deg + wca)
    gs  = max(0.0, tas_kt*math.cos(radians(wca)) + head)
    return wca, th, gs

def apply_var(true_deg,var_deg,east_is_negative=False):
    return wrap360(true_deg - var_deg if east_is_negative else true_deg + var_deg)

# ======================= AFM (650 kg) =======================
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

# ======================= Helpers de coordenadas =======================
def dms_to_dd(s: str) -> Optional[float]:
    """Aceita '372755.90N' ou '392747N' e '0084414.21W' etc."""
    s = (s or "").strip().upper()
    m = re.fullmatch(r"(\d{2,3})(\d{2})(\d{2}(?:\.\d+)?)\s*([NSEW])", s)
    if not m: return None
    deg, mn, sec, hemi = m.groups()
    deg = float(deg); mn = float(mn); sec = float(sec)
    val = deg + mn/60.0 + sec/3600.0
    if hemi in ("S","W"): val = -val
    return val

# ======================= Leitura dedicada dos teus ficheiros =======================
def load_ad_hel_ulm(path: Path) -> List[dict]:
    """Parseia linhas tipo:
    LP0078   ALENTEJO AIR PARK ULM   3728N00844W   372755.90N  0084414.21W  SAO TEOTONIO
    """
    out=[]
    if not path.exists(): return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip().strip('"')
        if not s or not s.startswith("LP"): continue
        m = re.search(r'^(LP[0-9A-Z]{4,})\s+(.+?)\s+\d{4,5}[NS]\d{5}[EW]\s+([0-9\.NnSs]+)\s+([0-9\.EeWw]+)\s+(.*)$', s)
        if not m:
            # fallback: tenta apanhar os 2 últimos tokens DMS
            toks = s.split()
            if len(toks)>=5 and re.fullmatch(r'[NS\dn\.]+', toks[-3]) and re.fullmatch(r'[EW\dn\.]+', toks[-2]):
                code=toks[0]; name=" ".join(toks[1:-3]); latd=toks[-3]; lond=toks[-2]
            else:
                continue
        else:
            code, name, latd, lond, _city = m.groups()
        la = dms_to_dd(latd); lo = dms_to_dd(lond)
        if la is None or lo is None: continue
        name = name.upper().strip()
        out.append({
            "Name": code.upper(),          # usar LPxxxx como chave principal
            "Alias": name,                 # alias com nome descritivo
            "Type": ("AERO" if "ULM" in name or "HEL" in name or "HELI" in name or "AIR" in name else "AERO"),
            "Freq": "",                    # ficheiro não traz VHF de forma fiável
            "lat": la, "lon": lo,
            "elev": None
        })
    return out

def load_localidades(path: Path) -> List[dict]:
    """Linhas tipo:
    ABRANTES  392747N 0081159W  ABRAN  LC
    """
    out=[]
    if not path.exists(): return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip().strip('"')
        if not s or s.startswith("LOCALIDADE") or "Total de registos" in s: continue
        # nome (coluna larga) + lat + lon + código (5 letras)
        m = re.match(r'^([A-Z0-9ÁÂÃÀÉÍÓÚÇ\-\s\(\)\/\.]+?)\s+(\d{6,7}[NS])\s+(\d{7,8}[EW])\s+([A-Z0-9]{4,6})', s)
        if not m: continue
        nome, latd, lond, code = m.groups()
        la = dms_to_dd(latd); lo = dms_to_dd(lond)
        if la is None or lo is None: continue
        out.append({
            "Name": code.upper(),          # ex.: PTSOR
            "Alias": nome.upper().strip(), # nome por extenso
            "Type": "WPT",
            "Freq": "",
            "lat": la, "lon": lo,
            "elev": None
        })
    return out

def build_index(records: List[dict]) -> Dict[str, dict]:
    idx={}
    def norm(s): return ascii_safe((s or "").upper()).strip()
    for r in records:
        keys = {norm(r["Name"]), norm(r.get("Alias",""))}
        for k in list(keys):
            if k: idx[k]=r
    return idx

# carregar da pasta do script e também de /mnt/data (caso corra noutro ambiente)
HERE = Path(__file__).parent
AD_FILE = (HERE/"AD-HEL-ULM.csv") if (HERE/"AD-HEL-ULM.csv").exists() else Path("/mnt/data/AD-HEL-ULM.csv")
LOC_FILE = (HERE/"Localidades-Nova-versao-230223.csv") if (HERE/"Localidades-Nova-versao-230223.csv").exists() else Path("/mnt/data/Localidades-Nova-versao-230223.csv")

AD_ROWS  = load_ad_hel_ulm(AD_FILE)
LOC_ROWS = load_localidades(LOC_FILE)
DB_ROWS  = AD_ROWS + LOC_ROWS
POINTS_DB = build_index(DB_ROWS)

# aeródromos para dropdown: todos LPxxxx do AD_FILE
AERO_LIST = sorted([r["Name"] for r in AD_ROWS if re.fullmatch(r"LP[A-Z0-9]{3}", r["Name"])])

def point_lookup(name: str) -> Optional[dict]:
    nm = ascii_safe((name or "").upper()).strip()
    return POINTS_DB.get(nm)

def point_latlon(name:str):
    rec = point_lookup(name) or {}
    return (rec.get("lat"), rec.get("lon"))

def point_elev(name:str) -> int:
    ev = (point_lookup(name) or {}).get("elev")
    try: return int(ev) if ev is not None else 0
    except: return 0

# ======================= TC/Dist =======================
def nm_haversine(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2-φ1; dλ = λ2-λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R_nm * c

def initial_true_course(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    y = sin(λ2-λ1)*cos(φ2)
    x = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(λ2-λ1)
    brng = degrees(atan2(y, x))
    return wrap360(brng)

# ======================= UI =======================
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.title("Navigation Plan & Inflight Log — Tecnam P2008")
st.caption(f"Pontos carregados: {len(DB_ROWS)}  •  Aeródromos (LP…): {len(AERO_LIST)}")

# Header
DEFAULT_STUDENT="AMOIT"; DEFAULT_AIRCRAFT="P208"; DEFAULT_CALLSIGN="RVP"
REGS=["CS-ECC","CS-ECD","CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW"]
PDF_TEMPLATE_PATHS=["NAVLOG_FORM.pdf", str(HERE/"NAVLOG_FORM.pdf")]

c1,c2,c3=st.columns(3)
with c1:
    aircraft=st.text_input("Aircraft",DEFAULT_AIRCRAFT).upper()
    registration=st.selectbox("Registration",REGS,index=0)
    callsign=st.text_input("Callsign",DEFAULT_CALLSIGN).upper()
with c2:
    student=st.text_input("Student",DEFAULT_STUDENT).upper()
    lesson = st.text_input("Lesson","").upper()
    instrutor = st.text_input("Instrutor","").upper()
with c3:
    if AERO_LIST:
        dept=st.selectbox("Departure (LP… da BD)", AERO_LIST, index=0).upper()
        arr =st.selectbox("Arrival (LP… da BD)",   AERO_LIST, index=min(1,len(AERO_LIST)-1)).upper()
        altn=st.selectbox("Alternate (LP… da BD)", AERO_LIST, index=min(2,len(AERO_LIST)-1)).upper()
    else:
        st.warning("Não encontrei LPxxxx no AD-HEL-ULM.csv — verifica o ficheiro.")
        dept=st.text_input("Departure").upper()
        arr =st.text_input("Arrival").upper()
        altn=st.text_input("Alternate").upper()
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

# Perf / consumos (descent usa FF de cruise por defeito)
c7,c8,c9=st.columns(3)
with c7:
    rpm_climb  = st.number_input("Climb RPM (AFM)",1800,2388,2250,step=10)
    rpm_cruise = st.number_input("Cruise RPM (AFM)",1800,2388,2000,step=10)
with c8:
    rod_fpm=st.number_input("ROD (ft/min)",200,1500,700,step=10)
    start_fuel=st.number_input("Fuel inicial (EFOB_START) [L]",0.0,1000.0,85.0,step=0.1)
with c9:
    cruise_ref_kt = st.number_input("Cruise speed (kt)", 40, 140, 80, step=1)
    descent_ref_kt= st.number_input("Descent speed (kt)", 40, 120, 65, step=1)

# ===== ROUTE =====
ROUTE_TYPES = ["AERO", "WPT", "NAVAID"]

def parse_route_text(txt:str) -> List[str]:
    tokens = re.split(r"[,\s→\-]+", (txt or "").strip())
    return [t.upper() for t in tokens if t]

def parse_route_tokens(tokens: List[str]) -> List[dict]:
    pts=[]
    for tok in tokens:
        if tok in ("->","→","-","--"): continue
        t=tok.strip().upper()
        if not t: continue
        freq=""
        if "@" in t:
            name, maybe = t.split("@",1)
            t = name.strip().upper()
            freq = maybe.strip()
        typ="WPT"; nm=t
        if re.fullmatch(r"LP[A-Z0-9]{3}", nm): typ="AERO"
        m = re.match(r"(NAVAID|VOR|NDB|DME|VORTAC)[:\-]?(.*)$", nm, re.I)
        if m:
            typ="NAVAID"; nm=(m.group(2) or "").strip().upper()
        if freq: typ="NAVAID"
        pts.append({"Name": nm, "Type": typ, "Freq": freq})
    return pts

st.markdown("### Route (DEP … ARR)")
default_route = f"{dept} {arr}"
route_text = st.text_area("Pontos (separados por espaço, vírgulas ou '->') — **MAIÚSCULA** (eu forço se não estiver)",
                          value=st.session_state.get("route_text", default_route))
colA, colB, colC = st.columns([1,1,1])
with colA:
    apply_route = st.button("Aplicar rota")
with colB:
    auto_calc = st.checkbox("Auto calcular TC/Dist por coordenadas", value=True)
with colC:
    skip_navaids = st.checkbox("NAVAID NÃO conta como leg", value=True)

if "route_rows" not in st.session_state:
    st.session_state.route_rows = [{"Name": dept, "Type":"AERO", "Freq":""},
                                   {"Name": arr,  "Type":"AERO", "Freq":""}]

if apply_route:
    pts = parse_route_tokens(parse_route_text(route_text))
    if len(pts)<2:
        pts=[{"Name": dept, "Type":"AERO", "Freq":""},{"Name": arr, "Type":"AERO", "Freq":""}]
    st.session_state.route_rows = pts
    st.session_state.route_text = " ".join([p["Name"] for p in pts])

# sincronizar DEP/ARR + maiúsculas
rows = st.session_state.route_rows
if rows:
    rows[0]["Name"]=dept.upper(); rows[0]["Type"]="AERO"
if len(rows)>=2:
    rows[-1]["Name"]=arr.upper(); rows[-1]["Type"]="AERO"
rows = [{**r, "Name": (r.get("Name","").upper()), "Type": (r.get("Type","WPT").upper()), "Freq": (r.get("Freq",""))} for r in rows]

st.markdown("### Tabela de rota (Name/Type/Freq). TC e Dist aparecem abaixo.")
rcfg = {
    "Name": st.column_config.TextColumn("Name"),
    "Type": st.column_config.SelectboxColumn("Type", options=ROUTE_TYPES),
    "Freq": st.column_config.TextColumn("Freq (ex: 113.40)"),
}
route_edited = st.data_editor(rows, hide_index=True, use_container_width=True,
                              column_config=rcfg, num_rows="dynamic", key="route_table")
route_edited = [ {"Name": r.get("Name","").upper(), "Type": r.get("Type","WPT").upper(), "Freq": r.get("Freq","")}
                 for r in route_edited if (r.get("Name") or "").strip() ]

# LEGS (exclui NAVAIDs se marcado)
leg_points = [r for r in route_edited if (r["Type"]!="NAVAID" or not skip_navaids)]

legs=[]; missing=set()
for i in range(len(leg_points)-1):
    a = leg_points[i]["Name"]; b = leg_points[i+1]["Name"]
    la1, lo1 = point_latlon(a); la2, lo2 = point_latlon(b)
    if auto_calc and None not in (la1,lo1,la2,lo2):
        di = round(nm_haversine(la1, lo1, la2, lo2), 1)
        tc = round(initial_true_course(la1, lo1, la2, lo2), 1)
    else:
        di, tc = 0.0, 0.0
        if la1 is None or lo1 is None: missing.add(a)
        if la2 is None or lo2 is None: missing.add(b)
    dst_row = next((r for r in route_edited if r["Name"]==b), {"Type":"WPT","Freq":""})
    legs.append({"From": a, "To": b, "Dist": di, "TC": tc,
                 "ToType": dst_row["Type"], "ToFreq": (dst_row["Freq"] or "").strip()})

seg_preview = [{"SEGMENTO": f"{l['From']}→{l['To']}", "TC (°T)": l["TC"], "Dist (nm)": l["Dist"]} for l in legs]
st.dataframe(seg_preview, use_container_width=True)
if missing:
    st.warning("Sem coordenadas na BD para: " + ", ".join(sorted(missing)))

# ======================= Perfil vertical / cortes =======================
def pressure_alt(alt_ft, qnh_hpa): return float(alt_ft) + (1013.0 - float(qnh_hpa))*30.0

dep_elev = point_elev(dept); arr_elev = point_elev(arr)
start_alt = float(dep_elev); end_alt = float(arr_elev)

pa_start  = pressure_alt(start_alt, qnh)
pa_cruise = pressure_alt(cruise_alt, qnh)
vy_kt = vy_interp_enroute(pa_start)
tas_climb, tas_cruise, tas_descent = vy_kt, float(cruise_ref_kt), float(descent_ref_kt)

roc = roc_interp_enroute(pa_start, temp_c)
delta_climb = max(0.0, cruise_alt - start_alt)
delta_desc  = max(0.0, cruise_alt - end_alt)
t_climb_total = delta_climb / max(roc,1e-6)
t_desc_total  = delta_desc  / max(rod_fpm,1e-6)

# FFs — descent usa SEMPRE cruise FF
pa_mid_climb = start_alt + 0.5*delta_climb
_, ff_climb  = cruise_lookup(pa_mid_climb, int(rpm_climb),  temp_c)
_, ff_cruise = cruise_lookup(pa_cruise,   int(rpm_cruise),  temp_c)
ff_descent   = float(ff_cruise)

def gs_for(tc, tas): return wind_triangle(float(tc), float(tas), wind_from, wind_kt)[2]

dist = [float(l["Dist"] or 0.0) for l in legs]
N = len(legs)
gs_climb   = [gs_for(legs[i]["TC"], tas_climb)   for i in range(N)]
gs_cruise  = [gs_for(legs[i]["TC"], tas_cruise)  for i in range(N)]
gs_descent = [gs_for(legs[i]["TC"], tas_descent) for i in range(N)]

climb_nm   = [0.0]*N; idx_toc = None; rem_t = float(t_climb_total)
for i in range(N):
    if rem_t <= 1e-9: break
    gs = max(gs_climb[i], 1e-6)
    use_t = min(rem_t, 60.0 * dist[i] / gs)
    climb_nm[i] = min(dist[i], gs * use_t / 60.0); rem_t -= use_t
    if rem_t <= 1e-9: idx_toc = i; break

descent_nm = [0.0]*N; idx_tod = None; rem_t = float(t_desc_total)
for j in range(N-1, -1, -1):
    if rem_t <= 1e-9: break
    gs = max(gs_descent[j], 1e-6)
    use_t = min(rem_t, 60.0 * dist[j] / gs)
    descent_nm[j] = min(dist[j], gs * use_t / 60.0); rem_t -= use_t
    if rem_t <= 1e-9: idx_tod = j; break

startup = parse_hhmm(startup_str); takeoff = add_minutes(startup,15) if startup else None; clock = takeoff
def ceil_pos_minutes(x):  return max(1, int(math.ceil(x - 1e-9))) if x > 0 else 0

rows_fp=[]; seq_points=[]
PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓"}
alt_cursor = float(start_alt); efob=float(start_fuel)

def add_segment(phase, from_nm, to_nm, i_leg, d_nm, tas, ff_lph):
    global clock, efob, alt_cursor
    if d_nm <= 1e-9: return
    tc = float(legs[i_leg]["TC"]); _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
    ete_raw = 60.0 * d_nm / max(gs,1e-6); ete = ceil_pos_minutes(ete_raw)
    burn = ff_lph * (ete_raw/60.0)
    alt_start = alt_cursor
    if phase == "CLIMB":   alt_end = min(cruise_alt, alt_start + roc * ete_raw)
    elif phase == "DESCENT": alt_end = max(end_alt,   alt_start - rod_fpm * ete_raw)
    else:                   alt_end = alt_start
    eto = ""
    if clock: clock = add_minutes(clock, ete); eto = clock.strftime("%H:%M")
    efob = max(0.0, efob - burn); alt_cursor = alt_end
    rows_fp.append({
        "Fase": PH_ICON[phase], "Leg/Marker": f"{from_nm}→{to_nm}", "To (Name)": to_nm,
        "ALT (ft)": f"{int(round(alt_start))}→{int(round(alt_end))}",
        "Detalhe": f"{phase.capitalize()} {d_nm:.1f} nm",
        "TC (°T)": round(tc,0), "TH (°T)": round(th,0),
        "MH (°M)": round(apply_var(th, var_deg, var_is_e),0),
        "TAS (kt)": round(tas,0), "GS (kt)": round(gs,0), "FF (L/h)": round(ff_lph,1),
        "Dist (nm)": round(d_nm,1), "ETE (min)": ete, "ETO": eto,
        "Burn (L)": round(burn,1), "EFOB (L)": round(efob,1)
    })
    seq_points.append({
        "name": to_nm, "alt": int(round(alt_end)),
        "tc": int(round(tc)), "th": int(round(th)),
        "mh": int(round(apply_var(th, var_deg, var_is_e))),
        "tas": int(round(tas)), "gs": int(round(gs)),
        "dist": d_nm, "ete": ete, "eto": eto, "burn": burn, "efob": efob
    })

seq_points.append({"name": dept, "alt": int(round(start_alt)),
                   "tc":"", "th":"", "mh":"", "tas":"", "gs":"", "dist":"", "ete":"", "eto": (takeoff.strftime("%H:%M") if takeoff else ""), "burn":"", "efob": efob})

for i in range(N):
    leg_from, leg_to = legs[i]["From"], legs[i]["To"]
    d_total  = dist[i]; d_cl = min(climb_nm[i], d_total); d_ds = min(descent_nm[i], d_total - d_cl)
    d_cr = max(0.0, d_total - d_cl - d_ds); cur_from = leg_from
    if d_cl > 0:
        to_name = "TOC" if (idx_toc == i and d_cl < d_total) else leg_to
        add_segment("CLIMB", cur_from, to_name, i, d_cl, vy_kt, ff_climb); cur_from = to_name
    if d_cr > 0:
        to_name = "TOD" if (idx_tod == i and d_ds > 0) else leg_to
        add_segment("CRUISE", cur_from, to_name, i, d_cr, float(cruise_ref_kt), ff_cruise); cur_from = to_name
    if d_ds > 0:
        add_segment("DESCENT", cur_from, leg_to, i, d_ds, float(descent_ref_kt), ff_descent)

eta = clock; landing = eta; shutdown = add_minutes(eta,5) if eta else None

# ===== Tabela da APP =====
st.markdown("### Flight plan — cortes dentro do leg (App)")
cfg={
    "Fase": st.column_config.TextColumn("Fase"),
    "Leg/Marker": st.column_config.TextColumn("Leg / Marker"),
    "To (Name)": st.column_config.TextColumn("To (Name)", disabled=True),
    "ALT (ft)": st.column_config.TextColumn("ALT (ft)"),
    "Detalhe": st.column_config.TextColumn("Detalhe"),
    "TC (°T)": st.column_config.NumberColumn("TC (°T)", disabled=True),
    "TH (°T)": st.column_config.NumberColumn("TH (°T)", disabled=True),
    "MH (°M)": st.column_config.NumberColumn("MH (°M)", disabled=True),
    "TAS (kt)": st.column_config.NumberColumn("TAS (kt)", disabled=True),
    "GS (kt)": st.column_config.NumberColumn("GS (kt)", disabled=True),
    "FF (L/h)": st.column_config.NumberColumn("FF (L/h)", disabled=True),
    "Dist (nm)": st.column_config.NumberColumn("Dist (nm)", disabled=True),
    "ETE (min)": st.column_config.NumberColumn("ETE (min)", disabled=True),
    "ETO": st.column_config.TextColumn("ETO", disabled=True),
    "Burn (L)": st.column_config.NumberColumn("Burn (L)", disabled=True),
    "EFOB (L)": st.column_config.NumberColumn("EFOB (L)", disabled=True),
}
st.data_editor([], key="dummy")  # evita erro quando vazio
st.data_editor(_, disabled=True) if False else None
st.data_editor(rows_fp, hide_index=True, use_container_width=True, num_rows="fixed", column_config=cfg, key="fp_table")

tot_ete_m = int(sum(int(r['ETE (min)']) for r in rows_fp)) if rows_fp else 0
tot_line = f"**Totais** — Dist {sum(float(r['Dist (nm)']) for r in rows_fp):.1f} nm • ETE {tot_ete_m//60:02d}:{tot_ete_m%60:02d} • Burn {sum(float(r['Burn (L)']) for r in rows_fp):.1f} L • EFOB {efob:.1f} L"
if eta:
    tot_line += f" • **ETA {eta.strftime('%H:%M')}** • **Landing {landing.strftime('%H:%M')}** • **Shutdown {shutdown.strftime('%H:%M')}**"
st.markdown(tot_line)

# ======================= PDF export =======================
st.markdown("### PDF export")
show_fields = st.checkbox("Mostrar nomes de campos do PDF (debug)")
try:
    template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)
except Exception as e:
    template_bytes = None
    st.error(f"Não foi possível ler 'NAVLOG_FORM.pdf': {e}")

def build_pdf_items_from_points(points):
    items = []
    for idx, p in enumerate(points, start=1):
        it = {
            "Name": p["name"], "Alt": str(int(round(p["alt"]))),
            "TC":  (str(p["tc"]) if idx>1 else ""), "TH": (str(p["th"]) if idx>1 else ""),
            "MH":  (str(p["mh"]) if idx>1 else ""), "TAS": (str(p["tas"]) if idx>1 else ""),
            "GS":  (str(p["gs"]) if idx>1 else ""), "Dist": (f"{p['dist']:.1f}" if idx>1 and isinstance(p["dist"], (int,float)) else ""),
            "ETE": (str(p["ete"]) if idx>1 else ""), "ETO": (p["eto"] if idx>1 else (p["eto"] or "")),
            "Burn": (f"{p['burn']:.1f}" if idx>1 and isinstance(p["burn"], (int,float)) else ""),
            "EFOB": (f"{p['efob']:.1f}" if idx>1 and isinstance(p["efob"], (int,float)) else f"{p['efob']:.1f}" if idx==1 else ""),
            "Freq": ""
        }
        items.append(it)
    return items

if template_bytes:
    fieldset, maxlens = get_fields_and_meta(template_bytes)
    if show_fields: st.code("\n".join(sorted(fieldset)))
    try:
        named: Dict[str,str] = {}
        def put_alias(value, *aliases):
            key = pick(fieldset, *aliases)
            if key: put(named, fieldset, key, value, maxlens)

        takeoff_str = add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else ""
        temp_dev = round(temp_c - isa_temp(point_elev(dept) + (1013.0 - qnh)*30.0))
        put_alias(aircraft,"Aircraft","Aeronave","ACFT")
        put_alias(registration,"Registration","Matricula","REG")
        put_alias(callsign,"Callsign","Indicativo","CS")
        put_alias(student,"Student","Aluno")
        put_alias(lesson,"Lesson","Aula")
        put_alias(instrutor,"Instrutor","Instructor","INSTR")
        put_alias(dept,"Dept_Airfield","Departure","DEP")
        put_alias(arr,"Arrival_Airfield","Arrival","ARR")
        put_alias(altn,"Alternate","ALTN")
        put_alias(str(point_elev(altn)),"Alt_Alternate","ALTN_ELEV","ALTN ELEV")
        put_alias("","Dept_Comm","DEP_FREQ","DEP FREQ")
        put_alias("","Arrival_comm","ARR_FREQ","ARR FREQ")
        put_alias("123.755","Enroute_comm","ENR_FREQ","ENR FREQ")
        put_alias(f"{int(round(qnh))}","QNH","ALT SET","QNH(hPa)")
        put_alias(f"{int(round(temp_c))} / {temp_dev}","temp_isa_dev","OAT/DEV","OAT ISA DEV")
        put_alias(f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}","wind","WIND")
        put_alias(f"{var_deg:.1f}{'E' if var_is_e else 'W'}","mag_var","VAR")
        put_alias(f"{int(round(cruise_alt))}","flt_lvl_altitude","CRZ_ALT","FL/ALT")
        put_alias(startup_str,"Startup","STARTUP","ETD-15")
        put_alias(takeoff_str,"Takeoff","ETD","OFF BLOCKS")

        pdf_items = build_pdf_items_from_points(seq_points)
        last_eto = pdf_items[-1]["ETO"] if pdf_items else ""
        put_alias(last_eto,"Landing","ETA","ON BLOCKS")
        put_alias((add_minutes(parse_hhmm(last_eto),5).strftime("%H:%M") if last_eto else ""),"Shutdown","SHUTDOWN")
        put_alias(f"{takeoff_str} / {last_eto}","ETD/ETA","ETD ETA")

        tot_min = sum(int(it["ETE"] or "0") for it in pdf_items)
        tot_nm  = sum(float(it["Dist"] or 0.0) for it in pdf_items)
        tot_bo  = sum(float(it["Burn"] or 0.0) for it in pdf_items)
        last_efob = pdf_items[-1]["EFOB"] if pdf_items else ""
        put_alias(f"{tot_min//60:02d}:{tot_min%60:02d}","FLT TIME","BLOCK TIME","TOTAL TIME")
        for key in ("LEVEL F/F","LEVEL_FF","Level_FF","Level F/F","CRZ_LEVEL"):
            put(named, fieldset, key, f"{int(round(cruise_alt))}", maxlens)
        pa_mid_climb_hdr = start_alt + 0.5*max(0.0, cruise_alt - start_alt)
        _, ff_climb_hdr = cruise_lookup(pa_mid_climb_hdr, int(rpm_climb), temp_c)
        climb_minutes = (max(0.0, cruise_alt - start_alt) / max(roc_interp_enroute(pressure_alt(start_alt,qnh), temp_c),1e-6))
        put_alias(f"{ff_climb_hdr*(climb_minutes/60.0):.1f}","CLIMB FUEL","FUEL CLIMB")
        put_alias(f"{tot_min}","ETE_Total","ETE TOTAL","TOTAL ETE")
        put_alias(f"{tot_nm:.1f}","Dist_Total","TOTAL DIST")
        put_alias(f"{tot_bo:.1f}","PL_BO_TOTAL","BURN TOTAL","TOTAL BURN")
        put_alias(last_efob,"EFOB_TOTAL","AFOB_TOTAL","FOB END")

        def field_aliases(idx, base):
            s=str(idx); return (f"{base}{s}", f"{base}_{s}", f"{base} {s}", f"{base}{int(s):02d}")
        for i, r in enumerate(pdf_items[:11], start=1):
            for base, val in [("Name",r["Name"]),("Alt",r["Alt"]),("TCRS",r["TC"]),("THDG",r["TH"]),("MHDG",r["MH"]),
                              ("TAS",r["TAS"]),("GS",r["GS"]),("Dist",r["Dist"]),("ETE",r["ETE"]),("ETO",r["ETO"]),
                              ("PL_BO",r["Burn"]),("EFOB",r["EFOB"]),("AFOB",r["EFOB"]),("FREQ",r["Freq"])]:
                if val!="":
                    key = pick(fieldset, *field_aliases(i, base))
                    if key: put(named, fieldset, key, val, maxlens)

        if st.button("Gerar PDF preenchido", type="primary"):
            out = fill_pdf(template_bytes, named, make_readonly=True)
            safe_reg = ascii_safe(registration)
            safe_date = dt.datetime.now(pytz.timezone("Europe/Lisbon")).strftime("%Y-%m-%d")
            filename = f"{safe_date}_{safe_reg}_NAVLOG.pdf"
            st.download_button("Download PDF", data=out, file_name=filename, mime="application/pdf")
            st.success("PDF gerado (campos bloqueados). Revê antes do voo.")
    except Exception as e:
        st.error(f"Erro ao preparar/gerar PDF: {e}")

