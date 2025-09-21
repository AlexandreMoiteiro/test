# app.py — NAVLOG com cortes dentro do leg (TOC/TOD), rota por coordenadas e PDF bloqueado
# Reqs: streamlit, pypdf, pytz

import streamlit as st
import datetime as dt
import pytz, io, json, unicodedata, re, math, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sin, asin, radians, degrees, fmod, atan2, cos

# ======================= PDF helpers (melhorado) =======================
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, TextStringObject, NumberObject
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

def ascii_safe(x: str) -> str:
    return unicodedata.normalize("NFKD", str(x or "")).encode("ascii","ignore").decode("ascii")

def read_pdf_bytes(paths_or_bytes):
    """Aceita lista de paths OU bytes diretos (ex. de upload)."""
    if isinstance(paths_or_bytes, (bytes, bytearray)):
        return bytes(paths_or_bytes)
    for p in (paths_or_bytes or []):
        if Path(p).exists():
            return Path(p).read_bytes()
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
    """Preenche e (opcional) torna os campos read-only (na prática, não editáveis)."""
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
                            obj.update({NameObject("/Ff"): NumberObject(ff | 1)})  # bit 1 = ReadOnly
        except Exception:
            pass

    bio = io.BytesIO(); writer.write(bio); return bio.getvalue()

def put(out: dict, fieldset: set, key: str, value: str, maxlens: Dict[str,int]):
    if key in fieldset:
        s = "" if value is None else str(value)
        if key in maxlens and len(s) > maxlens[key]:
            s = s[:maxlens[key]]
        out[key] = s

def pick(fieldset:set, *aliases) -> Optional[str]:
    """Devolve o primeiro nome de campo existente no PDF dado um conjunto de apelidos."""
    for nm in aliases:
        if nm in fieldset:
            return nm
    return None

# ======================= Utilidades numéricas =======================
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

# ======================= Vento / rumo =======================
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

def apply_var(true_deg,var_deg,east_is_negative=False):
    return wrap360(true_deg - var_deg if east_is_negative else true_deg + var_deg)

# ======================= Aviónica/perf (AFM 650 kg) =======================
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

# ======================= Aeródromos simples (fallback) =======================
AEROS={
 "LPSO":{"elev":390,"freq":"119.805","lat":40.149,"lon":-8.47},
 "LPEV":{"elev":807,"freq":"122.705","lat":41.275,"lon":-7.717},
 "LPCB":{"elev":1251,"freq":"122.300","lat":40.367,"lon":-7.776},
 "LPCO":{"elev":587,"freq":"118.405","lat":40.286,"lon":-7.512},
 "LPVZ":{"elev":2060,"freq":"118.305","lat":40.723,"lon":-7.889},
}
def aero_elev(icao): return int(AEROS.get(icao,{}).get("elev",0))
def aero_freq(icao): return AEROS.get(icao,{}).get("freq","")

# ======================= Coordenadas e TC/Dist =======================
def nm_haversine(lat1, lon1, lat2, lon2):
    """Distância em NM."""
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

def _parse_float(s):
    if s is None: return None
    if isinstance(s, (int,float)): return float(s)
    s=str(s).strip().replace(",", ".")
    try: return float(s)
    except: return None

def load_points_from_csv(path: Path) -> List[dict]:
    rows=[]
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # heurísticas de nomes de coluna
            name = r.get("Name") or r.get("NOME") or r.get("Nome") or r.get("DESIGNACAO") or r.get("Designacao") or r.get("LOCAL") or r.get("Local") or ""
            lat  = r.get("lat") or r.get("LAT") or r.get("Latitude") or r.get("LATITUDE") or r.get("Y") or r.get("Lat") or r.get("coord_y")
            lon  = r.get("lon") or r.get("LON") or r.get("Longitude") or r.get("LONGITUDE") or r.get("X") or r.get("Lon") or r.get("coord_x")
            typ  = r.get("Type") or r.get("TIPO") or r.get("Tipo") or r.get("Categoria") or ""
            freq = r.get("Freq") or r.get("FREQ") or r.get("Frequency") or r.get("Frequencia") or r.get("Frequência") or ""
            name = str(name).strip()
            la = _parse_float(lat); lo = _parse_float(lon)
            if name and la is not None and lo is not None:
                rows.append({"Name": name.upper(), "Type": (str(typ).upper() if typ else ""), "Freq": str(freq).strip(), "lat": la, "lon": lo})
    return rows

def build_point_index(rows: List[dict]) -> Dict[str, dict]:
    d={}
    for r in rows:
        key = r["Name"].upper()
        if key not in d:
            d[key]=r
    return d

# ======================= APP UI =======================
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.title("Navigation Plan & Inflight Log — Tecnam P2008")

DEFAULT_STUDENT="AMOIT"; DEFAULT_AIRCRAFT="P208"; DEFAULT_CALLSIGN="RVP"
REGS=["CS-ECC","CS-ECD","CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW"]
PDF_TEMPLATE_PATHS=["NAVLOG - FORM.pdf"]

# Tentar carregar CSVs por omissão
default_csv_paths = [
    Path("/mnt/data/AD-HEL-ULM.csv"),
    Path("/mnt/data/Localidades-Nova-versao-230223.csv"),
]

with st.expander("Bases de dados de pontos (CSV)"):
    st.write("Por omissão, tenta ler os CSVs em /mnt/data. Podes também carregar ficheiros adicionais.")
    up_csvs = st.file_uploader("Carregar CSV(s) com colunas Name/Lat/Lon (freq opcional)", type=["csv"], accept_multiple_files=True)

db_rows=[]
for p in default_csv_paths:
    try:
        db_rows += load_points_from_csv(p)
    except Exception:
        pass
if up_csvs:
    for uf in up_csvs:
        try:
            db_rows += load_points_from_csv(Path(uf.name))  # só para DictReader - conteúdo já foi lido pelo Streamlit
        except Exception:
            # fallback: ler do buffer
            try:
                text = uf.read().decode("utf-8")
                f = io.StringIO(text)
                reader = csv.DictReader(f)
                for r in reader:
                    name = r.get("Name") or r.get("NOME") or r.get("Nome") or r.get("DESIGNACAO") or r.get("Designacao") or r.get("LOCAL") or r.get("Local") or ""
                    lat  = r.get("lat") or r.get("LAT") or r.get("Latitude") or r.get("LATITUDE") or r.get("Y") or r.get("Lat") or r.get("coord_y")
                    lon  = r.get("lon") or r.get("LON") or r.get("Longitude") or r.get("LONGITUDE") or r.get("X") or r.get("Lon") or r.get("coord_x")
                    typ  = r.get("Type") or r.get("TIPO") or r.get("Tipo") or r.get("Categoria") or ""
                    freq = r.get("Freq") or r.get("FREQ") or r.get("Frequency") or r.get("Frequencia") or r.get("Frequência") or ""
                    la = _parse_float(lat); lo = _parse_float(lon)
                    if name and la is not None and lo is not None:
                        db_rows.append({"Name": str(name).upper().strip(), "Type": (str(typ).upper() if typ else ""), "Freq": str(freq).strip(), "lat": la, "lon": lo})
            except Exception:
                pass

# adicionar aeródromos fallback à BD (com lat/lon se houver)
for icao,info in AEROS.items():
    if "lat" in info and "lon" in info:
        db_rows.append({"Name": icao, "Type":"AERO", "Freq": info.get("freq",""), "lat": info["lat"], "lon": info["lon"]})

POINTS_DB = build_point_index(db_rows)

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
    dept=st.text_input("Departure (ex.: LPSO)", value="LPSO")
    arr = st.text_input("Arrival (ex.: LPCB)", value="LPCB")
    altn=st.text_input("Alternate (ex.: LPCO)", value="LPCO")
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
    rod_fpm=st.number_input("ROD (ft/min)",200,1500,700,step=10)
    start_fuel=st.number_input("Fuel inicial (EFOB_START) [L]",0.0,1000.0,85.0,step=0.1)
with c9:
    cruise_ref_kt = st.number_input("Cruise speed (kt)", 40, 140, 80, step=1)
    descent_ref_kt= st.number_input("Descent speed (kt)", 40, 120, 65, step=1)

# ===== ROUTE (textarea) + Tabela + Auto cálculo por coordenadas =====
def parse_route_text(txt:str) -> List[str]:
    tokens = re.split(r"[,\s→\-]+", (txt or "").strip())
    return [t for t in tokens if t]

ROUTE_TYPES = ["AERO", "WPT", "NAVAID"]

def parse_route_tokens(tokens: List[str]) -> List[dict]:
    pts = []
    for tok in tokens:
        if tok in ("->", "→", "-", "--"): 
            continue
        t = tok.strip()
        if not t: 
            continue
        freq = ""
        if "@" in t:
            name, maybe = t.split("@",1)
            t = name.strip()
            freq = maybe.strip()
        typ = "WPT"; nm = t.upper()
        if re.fullmatch(r"[A-Z]{4}", nm) and nm.startswith("LP"):
            typ = "AERO"
        m = re.match(r"(NAVAID|VOR|NDB|DME|VORTAC)[:\-]?(.*)$", nm, re.I)
        if m:
            typ = "NAVAID"
            nm = (m.group(2) or "").strip().upper()
        if freq: typ="NAVAID"
        pts.append({"Name": nm, "Type": typ, "Freq": freq, "TC": 0.0, "Dist": 0.0, "From":"", "To":""})
    return pts

st.markdown("### Route (DEP … ARR)")
default_route = f"{dept} {arr}"
route_text = st.text_area("Pontos (separados por espaço, vírgulas ou '->')",
                          value=st.session_state.get("route_text", default_route))

c_ra, c_rb, c_rc = st.columns([1,1,1])
with c_ra:
    apply_route = st.button("Aplicar rota")
with c_rb:
    auto_calc = st.checkbox("Auto calcular TC/Dist via coordenadas", value=True)
with c_rc:
    def snapshot_route() -> dict:
        return {
            "route_points": [r["Name"] for r in st.session_state.get("route_rows", [])],
            "legs": [{"TC":r.get("TC",0.0), "Dist":r.get("Dist",0.0)} for r in st.session_state.get("route_rows", []) if r.get("To")]
        }
    st.download_button("💾 Download rota (JSON)",
                       data=json.dumps(snapshot_route(), ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"route_{ascii_safe(registration)}.json",
                       mime="application/json")

uploaded = st.file_uploader("📤 Seleciona rota (JSON)", type=["json"], key="routefile")
use_uploaded = st.button("Usar rota do ficheiro")

if "route_rows" not in st.session_state:
    st.session_state.route_rows = [{"Name": dept, "Type":"AERO", "Freq":"", "TC":0.0, "Dist":0.0, "From":"", "To":""},
                                   {"Name": arr,  "Type":"AERO", "Freq":"", "TC":0.0, "Dist":0.0, "From":"", "To":""}]

if use_uploaded and uploaded is not None:
    try:
        data = json.loads(uploaded.read().decode("utf-8"))
        names = list(data.get("route_points") or [dept, arr])
        if not names: names = [dept, arr]
        rows=[]
        for nm in names:
            typ = "AERO" if re.fullmatch(r"[A-Z]{4}", nm or "") else "WPT"
            rows.append({"Name": nm.upper(), "Type": typ, "Freq":"", "TC":0.0, "Dist":0.0, "From":"", "To":""})
        st.session_state.route_rows = rows
        st.session_state["route_text"] = " ".join(names)
        st.success("Rota carregada do JSON.")
    except Exception as e:
        st.error(f"Falha a carregar JSON: {e}")

if apply_route:
    tokens = re.split(r"[,\s→\-]+", (route_text or "").strip())
    pts = parse_route_tokens(tokens)
    if len(pts) < 2:
        pts = [{"Name": dept, "Type":"AERO", "Freq":"", "TC":0.0, "Dist":0.0, "From":"", "To":""},
               {"Name": arr,  "Type":"AERO", "Freq":"", "TC":0.0, "Dist":0.0, "From":"", "To":""}]
    st.session_state.route_rows = [{**r, "From":"", "To":""} for r in pts]
    st.session_state.route_text = " ".join([r["Name"] for r in pts])

# garantir DEP/ARR atuais
rows = st.session_state.route_rows
if rows:
    rows[0]["Name"] = dept.upper(); rows[0]["Type"]="AERO"
if len(rows)>=2:
    rows[-1]["Name"] = arr.upper(); rows[-1]["Type"]="AERO"

st.markdown("### Legs / Route (edita aqui; Freq só se for NAVAID)")

rcfg = {
    "Name": st.column_config.TextColumn("Name"),
    "Type": st.column_config.SelectboxColumn("Type", options=ROUTE_TYPES),
    "Freq": st.column_config.TextColumn("Freq (ex: 113.40)"),
    "TC":   st.column_config.NumberColumn("TC (°T)", step=0.1, min_value=0.0, max_value=359.9),
    "Dist": st.column_config.NumberColumn("Dist (nm)", step=0.1, min_value=0.0),
}
route_edited = st.data_editor(rows, hide_index=True, use_container_width=True,
                              column_config=rcfg, num_rows="dynamic", key="route_table")

# reconstruir legs a partir da tabela e calcular TC/Dist pelas coordenadas se possível
points = [r["Name"].upper() for r in route_edited if (r.get("Name") or "").strip()]
N = max(0, len(points)-1)
legs = []
for i in range(N):
    src = route_edited[i]; dst = route_edited[i+1]
    from_nm, to_nm = src["Name"].upper(), dst["Name"].upper()
    tc = float(dst.get("TC") or 0.0); di = float(dst.get("Dist") or 0.0)
    # auto cálculo
    if auto_calc:
        a = POINTS_DB.get(from_nm) or {}
        b = POINTS_DB.get(to_nm) or {}
        if "lat" in a and "lon" in a and "lat" in b and "lon" in b:
            di = nm_haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            tc = initial_true_course(a["lat"], a["lon"], b["lat"], b["lon"])
    legs.append({
        "From": from_nm, "To": to_nm,
        "TC": round(tc,1), "Dist": round(di,1),
        "ToType": dst.get("Type","WPT"),
        "ToFreq": (dst.get("Freq") or "").strip()
    })
st.session_state.legs = legs

# ======================= Cálculo (perfil vertical, cortes) =======================
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

# FFs (AFM) — descent usa SEMPRE cruise FF (idle removido)
pa_mid_climb = start_alt + 0.5*delta_climb
pa_mid_desc  = end_alt   + 0.5*delta_desc
_, ff_climb  = cruise_lookup(pa_mid_climb, int(rpm_climb),  temp_c)
_, ff_cruise = cruise_lookup(pa_cruise,   int(rpm_cruise),  temp_c)
ff_descent   = float(ff_cruise)

def gs_for(tc, tas): return wind_triangle(float(tc), float(tas), wind_from, wind_kt)[2]

dist = [float(l["Dist"] or 0.0) for l in legs]
gs_climb   = [gs_for(legs[i]["TC"], tas_climb)   for i in range(N)]
gs_cruise  = [gs_for(legs[i]["TC"], tas_cruise)  for i in range(N)]
gs_descent = [gs_for(legs[i]["TC"], tas_descent) for i in range(N)]

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

# ===== APP: linhas por SEGMENTO =====
startup = parse_hhmm(startup_str)
takeoff = add_minutes(startup,15) if startup else None
clock = takeoff

def ceil_pos_minutes(x):  # arredonda ↑ e garante 1 min quando >0
    return max(1, int(math.ceil(x - 1e-9))) if x > 0 else 0

rows_fp=[]; seq_points=[]  # para o PDF (nome-a-nome, na ordem dos cortes)
PH_ICON = {"CLIMB":"↑","CRUISE":"→","DESCENT":"↓"}

alt_cursor = float(start_alt)
total_ete = total_burn = 0.0
efob=float(start_fuel)  # EFOB planeado

def add_segment(phase:str, from_nm:str, to_nm:str, i_leg:int, d_nm:float, tas:float, ff_lph:float):
    """Acrescenta um segmento; atualiza relógio, ALT, EFOB; regista o ponto 'to_nm' para o PDF na ORDEM CORRETA."""
    global clock, total_ete, total_burn, efob, alt_cursor
    if d_nm <= 1e-9: return

    tc = float(legs[i_leg]["TC"])
    _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)

    ete_raw = 60.0 * d_nm / max(gs,1e-6)  # minutos reais
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
    total_ete += ete
    total_burn += burn
    efob = max(0.0, efob - burn)
    alt_cursor = alt_end

    rows_fp.append({
        "Fase": PH_ICON[phase],
        "Leg/Marker": f"{from_nm}→{to_nm}",
        "To (Name)": to_nm,
        "ALT (ft)": f"{int(round(alt_start))}→{int(round(alt_end))}",
        "Detalhe": f"{phase.capitalize()} {d_nm:.1f} nm",
        "TC (°T)": round(tc,0),
        "TH (°T)": round(th,0),
        "MH (°M)": round(apply_var(th, var_deg, var_is_e),0),
        "TAS (kt)": round(tas,0), "GS (kt)": round(gs,0),
        "FF (L/h)": round(ff_lph,1),
        "Dist (nm)": round(d_nm,1), "ETE (min)": ete, "ETO": eto,
        "Burn (L)": round(burn,1), "EFOB (L)": round(efob,1)
    })

    # ponto de chegada (to_nm) para o PDF — ORDEM EXATA dos cortes
    seq_points.append({
        "name": to_nm, "alt": int(round(alt_end)),
        "tc": int(round(tc)), "th": int(round(th)),
        "mh": int(round(apply_var(th, var_deg, var_is_e))),
        "tas": int(round(tas)), "gs": int(round(gs)),
        "dist": d_nm, "ete": ete, "eto": eto, "burn": burn, "efob": efob
    })

# Ponto inicial (DEP) para o PDF
seq_points.append({"name": dept.upper(), "alt": int(round(start_alt)),
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

eta = clock
landing = eta
shutdown = add_minutes(eta,5) if eta else None

# ===== Tabela da APP =====
st.markdown("### Flight plan — cortes dentro do leg (App)")
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
st.data_editor(rows_fp, hide_index=True, use_container_width=True, num_rows="fixed", column_config=cfg, key="fp_table")

tot_ete_m = int(sum(int(r['ETE (min)']) for r in rows_fp))
tot_line = f"**Totais** — Dist {sum(float(r['Dist (nm)']) for r in rows_fp):.1f} nm • ETE {tot_ete_m//60:02d}:{tot_ete_m%60:02d} • Burn {sum(float(r['Burn (L)']) for r in rows_fp):.1f} L • EFOB {efob:.1f} L"
if eta:
    tot_line += f" • **ETA {eta.strftime('%H:%M')}** • **Landing {landing.strftime('%H:%M')}** • **Shutdown {shutdown.strftime('%H:%M')}**"
st.markdown(tot_line)

# ======================= PDF export =======================
st.markdown("### PDF template")
uploaded_pdf = st.file_uploader("📄 Carrega o formulário PDF (novo modelo)", type=["pdf"], key="pdftpl")

st.markdown("### PDF export")
show_fields = st.checkbox("Mostrar nomes de campos do PDF (debug)")

def build_pdf_items_from_points(points):
    """Cada item é o ponto de chegada; idx=1 é o DEP (sem métricas do segmento)."""
    items = []
    for idx, p in enumerate(points, start=1):
        it = {
            "Name": p["name"],
            "Alt": str(int(round(p["alt"]))),
            "TC":  (str(p["tc"]) if idx>1 else ""),
            "TH":  (str(p["th"]) if idx>1 else ""),
            "MH":  (str(p["mh"]) if idx>1 else ""),
            "TAS": (str(p["tas"]) if idx>1 else ""),
            "GS":  (str(p["gs"])  if idx>1 else ""),
            "Dist": (f"{p['dist']:.1f}" if idx>1 and isinstance(p["dist"], (int,float)) else ""),
            "ETE":  (str(p["ete"]) if idx>1 else ""),
            "ETO":  (p["eto"] if idx>1 else (p["eto"] or "")),
            "Burn": (f"{p['burn']:.1f}" if idx>1 and isinstance(p["burn"], (int,float)) else ""),
            "EFOB": (f"{p['efob']:.1f}" if idx>1 and isinstance(p["efob"], (int,float)) else f"{p['efob']:.1f}" if idx==1 else ""),
            "Freq": str(p.get("freq") or "")
        }
        items.append(it)
    return items

try:
    template_bytes = None
    if uploaded_pdf is not None:
        template_bytes = uploaded_pdf.read()
    else:
        template_bytes = read_pdf_bytes(PDF_TEMPLATE_PATHS)
except Exception as e:
    template_bytes = None
    st.error(f"Não foi possível ler o PDF: {e}")

if template_bytes:
    fieldset, maxlens = get_fields_and_meta(template_bytes)
    if show_fields:
        st.code("\n".join(sorted(fieldset)))

    try:
        named: Dict[str,str] = {}

        # Cabeçalho – apelidos
        def put_alias(value, *aliases):
            key = pick(fieldset, *aliases)
            if key: put(named, fieldset, key, value, maxlens)

        takeoff_str = add_minutes(parse_hhmm(startup_str),15).strftime("%H:%M") if startup_str else ""
        temp_dev = round(temp_c - isa_temp(pressure_alt(aero_elev(dept), qnh)))

        put_alias(aircraft,      "Aircraft","Aeronave","ACFT")
        put_alias(registration,  "Registration","Matricula","REG")
        put_alias(callsign,      "Callsign","Indicativo","CS")
        put_alias(student,       "Student","Aluno")
        put_alias(lesson,        "Lesson","Aula")
        put_alias(instrutor,     "Instrutor","Instructor","INSTR")
        put_alias(dept.upper(),  "Dept_Airfield","Departure","DEP")
        put_alias(arr.upper(),   "Arrival_Airfield","Arrival","ARR")
        put_alias(altn.upper(),  "Alternate","ALTN")
        put_alias(str(aero_elev(altn)), "Alt_Alternate","ALTN_ELEV","ALTN ELEV")
        put_alias(aero_freq(dept), "Dept_Comm","DEP_FREQ","DEP FREQ")
        put_alias(aero_freq(arr),  "Arrival_comm","ARR_FREQ","ARR FREQ")
        put_alias("123.755",       "Enroute_comm","ENR_FREQ","ENR FREQ")
        put_alias(f"{int(round(qnh))}", "QNH","ALT SET","QNH(hPa)")
        put_alias(f"{int(round(temp_c))} / {temp_dev}", "temp_isa_dev","OAT/DEV","OAT ISA DEV")
        put_alias(f"{int(round(wind_from)):03d}/{int(round(wind_kt)):02d}", "wind","WIND")
        put_alias(f"{var_deg:.1f}{'E' if var_is_e else 'W'}", "mag_var","VAR")
        put_alias(f"{int(round(cruise_alt))}", "flt_lvl_altitude","CRZ_ALT","FL/ALT")
        put_alias(startup_str, "Startup","STARTUP","ETD-15")
        put_alias(takeoff_str, "Takeoff","ETD","OFF BLOCKS")

        # enriquecer pontos com Freq (apenas NAVAID e com freq)
        def enrich_points_with_freq(spoints, legs):
            byname = {}
            for l in legs:
                byname[l["To"]] = (l.get("ToType","WPT"), (l.get("ToFreq") or "").strip())
            out=[]
            for p in spoints:
                typ,freq = byname.get(p["name"].upper(), ("WPT",""))
                p2 = dict(p)
                p2["freq"] = freq if (typ=="NAVAID" and freq) else ""
                out.append(p2)
            return out

        pdf_points = enrich_points_with_freq(seq_points, st.session_state.legs)
        pdf_items = build_pdf_items_from_points(pdf_points)

        # ETA/Shutdown (planeado)
        last_eto = pdf_items[-1]["ETO"] if pdf_items else ""
        put_alias(last_eto, "Landing","ETA","ON BLOCKS")
        put_alias((add_minutes(parse_hhmm(last_eto),5).strftime("%H:%M") if last_eto else ""), "Shutdown","SHUTDOWN")
        put_alias(f"{takeoff_str} / {last_eto}", "ETD/ETA","ETD ETA")

        # Totais
        tot_min = sum(int(it["ETE"] or "0") for it in pdf_items)
        tot_nm  = sum(float(it["Dist"] or 0.0) for it in pdf_items)
        tot_bo  = sum(float(it["Burn"] or 0.0) for it in pdf_items)
        last_efob = pdf_items[-1]["EFOB"] if pdf_items else ""
        put_alias(f"{tot_min//60:02d}:{tot_min%60:02d}", "FLT TIME","BLOCK TIME","TOTAL TIME")
        for key in ("LEVEL F/F","LEVEL_FF","Level_FF","Level F/F","CRZ_LEVEL"):
            put(named, fieldset, key, f"{int(round(cruise_alt))}", maxlens)
        put_alias(f"{ff_climb*(t_climb_total/60.0):.1f}", "CLIMB FUEL","FUEL CLIMB")
        put_alias(f"{tot_min}", "ETE_Total","ETE TOTAL","TOTAL ETE")
        put_alias(f"{tot_nm:.1f}", "Dist_Total","TOTAL DIST")
        put_alias(f"{tot_bo:.1f}", "PL_BO_TOTAL","BURN TOTAL","TOTAL BURN")
        put_alias(last_efob, "EFOB_TOTAL","AFOB_TOTAL","FOB END")

        # Linhas (até 11) – mapeia por posição usando apelidos por coluna
        def field_aliases(idx, base):
            s=str(idx)
            return (f"{base}{s}", f"{base}_{s}", f"{base} {s}", f"{base}{int(s):02d}")

        for i, r in enumerate(pdf_items[:11], start=1):
            for base, val in [
                ("Name",  r["Name"]),
                ("Alt",   r["Alt"]),
                ("TCRS",  r["TC"]),
                ("THDG",  r["TH"]),
                ("MHDG",  r["MH"]),
                ("TAS",   r["TAS"]),
                ("GS",    r["GS"]),
                ("Dist",  r["Dist"]),
                ("ETE",   r["ETE"]),
                ("ETO",   r["ETO"]),
                ("PL_BO", r["Burn"]),
                ("EFOB",  r["EFOB"]),
                ("AFOB",  r["EFOB"]),
                ("FREQ",  r["Freq"]),  # só virá preenchido se NAVAID c/ freq
            ]:
                if val != "":
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
