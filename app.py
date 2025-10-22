# app.py — NAVLOG — rev17 (fix legibilidade das pílulas)
# - Texto das pílulas SEM rotação (mantém-se horizontal)
# - Escala das fontes = zoom Leaflet (--navlog-z) × slider do utilizador (--navlog-user)
# - Mantida a tua lógica de posicionamento, TOC/TOD, CPs, etc.

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib
from streamlit_folium import st_folium
from folium.plugins import Fullscreen
from math import degrees
from branca.element import MacroElement
from jinja2 import Template

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0   # kt
FUEL_FLOW = 20.0                                        # L/h
EARTH_NM  = 3440.065

PROFILE_COLORS = {"CLIMB":"#FF7A00","LEVEL":"#C000FF","DESCENT":"#00B386"}

# ======== PAGE / STYLE ========
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{
  --line:#e5e7eb;
  --chip:#f3f4f6;
  --navlog-z: 1;      /* escala pelo zoom do mapa (JS) */
  --navlog-user: 1;   /* escala pelo slider do utilizador (Python) */
}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin:8px 0;background:#fff}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:var(--line);margin:10px 0}
.leaflet-control-zoom a{font-weight:800}
.small{font-size:12px;color:#555}
.row{display:flex;gap:8px;align-items:center}
.badge{font-weight:700;border:1px solid #111;border-radius:8px;padding:2px 6px;margin-right:6px}

/* === tipografia e visual das pílulas — SEM rotação do texto === */
.nav-pill{ 
  font-weight:900; color:#111; background:rgba(255,255,255,0.96);
  padding:4px 6px; border-radius:10px; border:2px solid #111;
  box-shadow:0 0 0 2px rgba(255,255,255,0.96);
  line-height:1.1; letter-spacing:.2px; text-align:center; white-space:nowrap;
  /* tamanho base: reage a zoom e slider do utilizador */
  font-size: calc(14px * var(--navlog-z, 1) * var(--navlog-user, 1));
  transform-origin:center center;
  will-change: transform;
}

.nav-wpname{
  font-size: calc(16px * var(--navlog-z, 1) * var(--navlog-user, 1));
  color:#111; font-weight:900;
  text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;
  white-space:nowrap;
}
.nav-wpinfo{
  font-size: calc(12px * var(--navlog-z, 1) * var(--navlog-user, 1));
  font-weight:800; color:#111;
  background:rgba(255,255,255,0.96); border:2px solid #111; border-radius:8px;
  padding:1px 5px; white-space:nowrap;
}
</style>
""", unsafe_allow_html=True)

# ======== HELPERS ========
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
hhmmss = lambda t: f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
wrap360 = lambda x: (x % 360 + 360) % 360
def angdiff(a, b): return (a - b + 180) % 360 - 180
def deg3(v): return f"{int(round(v))%360:03d}°"

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * math.sin(d)
    s = max(-1, min(1, cross / max(tas,1e-9)))
    wca = degrees(math.asin(s)); th = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False): return wrap360(th - var if east_is_neg else th + var)

def gc_dist_nm(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_NM * c

def gc_course_tc(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dλ = λ2 - λ1
    y = math.sin(dλ)*math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360) % 360

def dest_point(lat, lon, bearing_deg, dist_nm):
    θ = math.radians(bearing_deg); δ = dist_nm / EARTH_NM
    φ1, λ1 = math.radians(lat), math.radians(lon)
    sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
    φ2 = math.asin(sinφ2)
    y = math.sin(θ)*math.sin(δ)*math.cos(φ1); x = math.cos(δ) - math.sin(φ1)*sinφ2
    λ2 = λ1 + math.atan2(y, x)
    return math.degrees(φ2), ((math.degrees(λ2)+540)%360)-180

def point_along_gc(lat1, lon1, lat2, lon2, dist_from_start_nm):
    total = gc_dist_nm(lat1, lon1, lat2, lon2)
    if total <= 0: return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)

def _nm_dist(a,b): return gc_dist_nm(a[0],a[1],b[0],b[1])

# ======== STATE ========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("show_legend", False)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")
ens("maptiler_key", "")
ens("show_labels", True); ens("show_ticks", True); ens("text_scale", 1.25)
ens("db_points", None); ens("qadd", ""); ens("alt_qadd", 3000.0)
ens("search_rows", []); ens("last_q", "")

# ======== PARÂMETROS GLOBAIS ========
with st.form("globals"):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from))
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt))
    with c2:
        st.session_state.mag_var   = st.number_input("Variação magnética (±°)", -30.0, 30.0, float(st.session_state.mag_var))
        st.session_state.mag_is_e  = st.toggle("Var. é EAST (subtrai)", value=st.session_state.mag_is_e)
    with c3:
        st.session_state.roc_fpm   = st.number_input("ROC global (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10)
        st.session_state.desc_angle= st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
    with c4:
        st.session_state.start_efob= st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default))
    b1,b2,b3,b4 = st.columns([1.4,1,1,1])
    with b1:
        bases = ["OpenTopoMap (VFR-ish)","EOX Sentinel-2 (satélite)","Esri World Imagery (satélite + labels)","Esri World TopoMap (topo)","OSM Standard","MapTiler Satellite Hybrid (requer key)"]
        st.session_state.map_base = st.selectbox("Base do mapa", bases, index=bases.index(st.session_state.map_base) if st.session_state.map_base in bases else 0)
    with b2:
        st.session_state.show_labels = st.toggle("Mostrar pílulas", value=st.session_state.show_labels)
        st.session_state.show_ticks  = st.toggle("Mostrar riscas CP", value=st.session_state.show_ticks)
    with b3:
        st.session_state.show_legend = st.toggle("Mostrar legenda perfis", value=st.session_state.show_legend)
    with b4:
        st.session_state.text_scale  = st.slider("Tamanho base do texto", 0.9, 1.8, float(st.session_state.text_scale), 0.05)
    st.form_submit_button("Aplicar")

# -> Injeta o valor do slider na variável CSS global (--navlog-user)
st.markdown(
    f"<style>:root{{--navlog-user:{float(st.session_state.text_scale)};}}</style>",
    unsafe_allow_html=True
)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== CSVs locais ========
AD_CSV  = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"

def dms_to_dd(token: str, is_lon=False):
    token = str(token).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([NSEW])$", token, re.I)
    if not m: return None
    value, hemi = m.groups()
    if "." in value:
        if is_lon: deg = int(value[0:3]); minutes = int(value[3:5]); seconds = float(value[5:])
        else:      deg = int(value[0:2]); minutes = int(value[2:4]); seconds = float(value[4:])
    else:
        if is_lon: deg = int(value[0:3]); minutes = int(value[3:5]); seconds = int(value[5:])
        else:      deg = int(value[0:2]); minutes = int(value[2:4]); seconds = int(value[4:])
    dd = deg + minutes/60 + seconds/3600
    if hemi.upper() in ["S","W"]: dd = -dd
    return dd

def parse_ad_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or s.startswith(("Ident", "DEP/")): continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok = coord_toks[-2]; lon_tok = coord_toks[-1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            ident = tokens[0] if re.match(r"^[A-Z0-9]{4,}$", tokens[0]) else None
            try:    name = " ".join(tokens[1:tokens.index(coord_toks[0])]).strip()
            except: name = " ".join(tokens[1:]).strip()
            try:    lon_idx = tokens.index(lon_tok); city = " ".join(tokens[lon_idx+1:]) or None
            except: city = None
            rows.append({"src":"AD","code":ident or name, "name":name, "city":city,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in df.iloc[:,0].dropna().tolist():
        s = str(line).strip()
        if not s or "Total de registos" in s: continue
        tokens = s.split()
        coord_toks = [t for t in tokens if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", t)]
        if len(coord_toks) >= 2:
            lat_tok, lon_tok = coord_toks[0], coord_toks[1]
            lat = dms_to_dd(lat_tok, is_lon=False); lon = dms_to_dd(lon_tok, is_lon=True)
            try: lon_idx = tokens.index(lon_tok)
            except ValueError: continue
            code = tokens[lon_idx+1] if lon_idx+1 < len(tokens) else None
            sector = " ".join(tokens[lon_idx+2:]) if lon_idx+2 < len(tokens) else None
            name = " ".join(tokens[:tokens.index(lat_tok)]).strip()
            rows.append({"src":"LOC","code":code or name, "name":name, "sector":sector,"lat":lat,"lon":lon,"alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

try:
    ad_raw  = pd.read_csv(AD_CSV);  ad_df  = parse_ad_df(ad_raw)
    loc_raw = pd.read_csv(LOC_CSV); loc_df = parse_loc_df(loc_raw)
except Exception:
    ad_df  = pd.DataFrame(columns=["src","code","name","city","lat","lon","alt"])
    loc_df = pd.DataFrame(columns=["src","code","name","sector","lat","lon","alt"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

if st.session_state.db_points is None:
    st.session_state.db_points = pd.concat([ad_df, loc_df]).dropna(subset=["lat","lon"]).reset_index(drop=True)
db = st.session_state.db_points

# ======== util WPs repetidos ========
def make_unique_name(name: str) -> str:
    names = [str(w["name"]) for w in st.session_state.wps]
    if name not in names:
        return name
    k = 2
    while f"{name} #{k}" in names:
        k += 1
    return f"{name} #{k}"

def append_wp(name: str, lat: float, lon: float, alt: float) -> None:
    nm = make_unique_name(str(name).strip())
    st.session_state.wps.append({"name": nm, "lat": float(lat), "lon": float(lon), "alt": float(alt)})

# ======== ABAS — ADICIONAR WPs ========
tab_csv, tab_map, tab_paste = st.tabs(["🔎 Pesquisar CSV", "🗺️ Adicionar no mapa", "📋 Colar lista"])

with tab_csv:
    c1, c2 = st.columns([3,1])
    with c1:
        q = st.text_input("Pesquisar (marca os cards ou usa ➕ para adicionar)", key="qadd",
                          placeholder="Ex: LPPT, ALPAL, ÉVORA, NISA…").strip()
    with c2:
        st.session_state.alt_qadd = st.number_input("Alt (ft) p/ novos WPs", 0.0, 18000.0,
                                                    float(st.session_state.alt_qadd), step=100.0)
    if q != st.session_state.last_q:
        for k in list(st.session_state.keys()):
            if k.startswith("csvpick_"): del st.session_state[k]
        st.session_state.last_q = q

    def _score_row(row, tq, last_wp):
        code = str(row.get("code") or "").lower()
        name = str(row.get("name") or "").lower()
        sim = difflib.SequenceMatcher(None, tq, f"{code} {name}").ratio()
        starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
        near = 0.0
        if last_wp: near = 1.0/(1.0+gc_dist_nm(last_wp["lat"], last_wp["lon"], row["lat"], row["lon"]))
        return starts*2 + sim + near*0.25

    def _search_points(tq):
        if not tq: return db.head(0)
        tql = tq.lower().strip()
        last = st.session_state.wps[-1] if st.session_state.wps else None
        df = db[db.apply(lambda r: any(tql in str(v).lower() for v in r.values), axis=1)].copy()
        if df.empty: return df
        df["__score"] = df.apply(lambda r: _score_row(r, tql, last), axis=1)
        return df.sort_values("__score", ascending=False)

    results = _search_points(q)
    st.session_state.search_rows = results.head(30).to_dict("records") if not results.empty else []

    left, right = st.columns([2,1])
    with left:
        if st.session_state.search_rows:
            st.caption("Resultados")
            picked = []
            for i, r in enumerate(st.session_state.search_rows):
                code = r.get("code") or ""
                name = r.get("name") or ""
                local = r.get("city") or r.get("sector") or ""
                lat, lon = float(r["lat"]), float(r["lon"])
                label = f"{code} — {name}"
                col0, col1, col2 = st.columns([0.08,0.76,0.16])
                with col0:
                    sel = st.checkbox("", key=f"csvpick_{i}")
                with col1:
                    st.markdown(
                        f"<div class='card'><div class='row'><span class='badge'>[{r['src']}]</span>"
                        f"<b>{label}</b></div><div class='small'>{local}</div>"
                        f"<div class='small'>({lat:.4f}, {lon:.4f})</div></div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    if st.button("➕", key=f"csvadd_{i}", use_container_width=True):
                        append_wp(code or name, lat, lon, float(st.session_state.alt_qadd))
                        st.success("Adicionado.")
                if sel: picked.append((code or name, lat, lon))
            st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
            if st.button("➕ Adicionar selecionados", use_container_width=True, disabled=(len(picked)==0)):
                for nm, la, lo in picked: append_wp(nm, la, lo, float(st.session_state.alt_qadd))
                st.success(f"Adicionados {len(picked)} WPs.")
        else:
            st.info("Sem resultados.")
    with right:
        st.caption("Pré-visualização")
        sel = []
        for i, r in enumerate(st.session_state.search_rows):
            if st.session_state.get(f"csvpick_{i}", False): sel.append((float(r["lat"]), float(r["lon"])))
        if sel:
            mprev = folium.Map(location=list(sel[0]), zoom_start=8,
                               tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                               attr="© OpenTopoMap", control_scale=True)
            for (la,lo) in sel:
                folium.CircleMarker((la, lo), radius=6, color="#FF8800",
                                    weight=2, fill=True, fill_color="#FF8800", fill_opacity=0.9).add_to(mprev)
            st_folium(mprev, width=None, height=260, key="preview_map")
        else:
            st.info("Marca alguns cards para ver no mini-mapa.")

with tab_map:
    st.caption("Clica no mapa e depois em **Adicionar**.")
    m0 = folium.Map(location=[39.7, -8.1], zoom_start=7,
                    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                    attr="© OpenTopoMap", control_scale=True)
    for w in st.session_state.wps:
        folium.CircleMarker((w["lat"],w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1).add_to(m0)
    map_out = st_folium(m0, width=None, height=420, key="pickmap")
    with st.form("add_by_click"):
        cA,cB,_ = st.columns([2,1,1])
        with cA: nm = st.text_input("Nome", "WP novo")
        with cB: alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
        clicked = map_out.get("last_clicked")
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            lat, lon = clicked["lat"], clicked["lng"]
            append_wp(nm, float(lat), float(lon), float(alt))
            st.success("Adicionado.")

with tab_paste:
    st.caption("Formato por linha: `NOME; LAT; LON; ALT` — aceita DD ou DMS (`390712N 0083155W`). ALT opcional.")
    txt = st.text_area("Lista", height=120,
                       placeholder="ABRANTES; 39.4667; -8.2; 3000\nPONTO X; 390712N; 0083155W; 2500")
    alt_def = st.number_input("Alt (ft) se faltar", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0, key="alt_def_paste")
    if st.button("Adicionar da lista"):
        n=0
        for line in txt.splitlines():
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 3: continue
            name = parts[0]
            lat  = dms_to_dd(parts[1], is_lon=False) if re.search(r"[NnSs]$", parts[1]) else float(parts[1].replace(",","."))
            lon  = dms_to_dd(parts[2], is_lon=True ) if re.search(r"[EeWw]$", parts[2]) else float(parts[2].replace(",","."))
            alt  = float(parts[3]) if len(parts)>=4 and parts[3] else alt_def
            if lat is None or lon is None: continue
            append_wp(name, float(lat), float(lon), float(alt)); n+=1
        st.success(f"Adicionados {n} WPs.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== EDITOR WPs ========
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4 = st.columns([2,2,2,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i); st.experimental_rerun()
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== TOC/TOD COMO WPs ========
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, desc_angle_deg):
    nodes = []
    if len(user_wps) < 2: return nodes
    for i in range(len(user_wps)-1):
        A, B = user_wps[i], user_wps[i+1]
        nodes.append(A)
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        _, _, gs_cl = wind_triangle(tc, CLIMB_TAS,   wind_from, wind_kt)
        _, _, gs_de = wind_triangle(tc, DESCENT_TAS, wind_from, wind_kt)
        if B["alt"] > A["alt"]:
            dh = B["alt"] - A["alt"]; t_need = dh / max(roc_fpm, 1); d_need = gs_cl * (t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:
            rod_fpm = max(100.0, gs_de * 5.0 * (desc_angle_deg/3.0))
            dh = A["alt"] - B["alt"]; t_need = dh / max(rod_fpm, 1); d_need = gs_de * (t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
    nodes.append(user_wps[-1]); return nodes

# ======== LEGS ========
def build_legs_from_nodes(nodes, wind_from, wind_kt, mag_var, mag_is_e, ck_every_min):
    legs = []
    if len(nodes) < 2: return legs
    base_time=None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time=None
    carry_efob = float(st.session_state.start_efob); t_cursor=0

    for i in range(len(nodes)-1):
        A,B = nodes[i], nodes[i+1]
        tc   = gc_course_tc(A["lat"], A["lon"], B["lat"], B["lon"])
        dist = gc_dist_nm(A["lat"], A["lon"], B["lat"], B["lon"])
        profile = "LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile=="CLIMB" else (DESCENT_TAS if profile=="DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt)
        mh = apply_var(th, mag_var, mag_is_e)
        time_sec = rt10((dist / max(gs,1e-9)) * 3600.0) if gs>0 else 0
        burn = FUEL_FLOW * (time_sec/3600.0)
        efob_start = carry_efob; efob_end = max(0.0, r10f(efob_start - burn))
        clk_start = (base_time + dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor)}"
        clk_end   = (base_time + dt.timedelta(seconds=t_cursor+time_sec)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor+time_sec)}"
        cps=[]
        if ck_every_min>0 and gs>0:
            k=1
            while k*ck_every_min*60 <= time_sec:
                t=k*ck_every_min*60; d=gs*(t/3600.0)
                eto=(base_time + dt.timedelta(seconds=t_cursor+t)).strftime('%H:%M') if base_time else ""
                cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto}); k+=1
        legs.append({"i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
                     "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),"efob_start":r10f(efob_start),
                     "efob_end":r10f(efob_end),"clock_start":clk_start,"clock_end":clk_end,"cps":cps})
        t_cursor += time_sec; carry_efob = efob_end
    return legs

# ======== GERAR ROTA ========
cgen,_ = st.columns([2,6])
with cgen:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(
            st.session_state.wps, st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.roc_fpm, st.session_state.desc_angle
        )
        st.session_state.legs = build_legs_from_nodes(
            st.session_state.route_nodes, st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.mag_var, st.session_state.mag_is_e, st.session_state.ck_default
        )

# ======== RESUMO ========
if st.session_state.legs:
    total_sec  = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final = st.session_state.legs[-1]["efob_end"]
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b></div>"
        + f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>"
        + "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== LABEL ENGINE (sem setas, só “placas”) ========
LABEL_MIN_NM_NORMAL = 0.3
ZONE_WP_R, ZONE_LABEL_BASE_R = 0.85, 1.0
LABEL_MIN_CLEAR = 0.7

def html_marker(m, lat, lon, html):
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def pill_html(line1, line2, angle_deg, scale=1.0):
    # Texto SEM rotação; aplicamos apenas translate + scale local da pílula
    return f"""
    <div class="nav-pill" style="transform: translate(-50%,-50%) scale({scale});">
        <div>{line1}</div>
        <div>{line2}</div>
    </div>
    """

class Zones:
    def __init__(self): self.z=[]
    def add(self, lat, lon, r): self.z.append((lat,lon,float(r)))
    def clearance(self, lat, lon):
        if not self.z: return 9e9
        return min(_nm_dist((lat,lon),(a,b)) - r for a,b,r in self.z)
    def fit_anchor(self, base_lat, base_lon, normal_bearing, step_nm=0.22, max_iter=8):
        lat, lon = base_lat, base_lon; i=0
        while self.clearance(lat,lon) < LABEL_MIN_CLEAR and i<max_iter:
            lat, lon = dest_point(lat, lon, normal_bearing, step_nm); i+=1
        return (lat,lon), i
    def add_leg_corridor(self, A, B, spacing_nm=0.9, r_nm=0.38):
        dist = _nm_dist((A["lat"],A["lon"]), (B["lat"],B["lon"])); 
        if dist <= spacing_nm: return
        steps = max(2, int(dist/spacing_nm))
        for k in range(1, steps):
            p = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], dist*k/steps)
            self.add(p[0], p[1], r_nm)

def label_candidates(L, side_off):
    cands=[]
    fracs = (0.22,0.34,0.46,0.58,0.70,0.82) if L["Dist"] >= 6.0 else (0.30,0.50,0.70)
    lateral = (side_off, side_off+0.35, side_off+0.7, side_off+1.1, side_off+1.5)
    for frac in fracs:
        base_d = max(0.35, min(L["Dist"]-0.35, L["Dist"]*frac))
        base = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], base_d)
        for side in (-1, +1):
            for off in lateral:
                anchor = dest_point(base[0], base[1], L["TC"]+90*side, off)
                cands.append((anchor, side, off))
    return cands

def preferred_side_outside_turn(legs, idx_leg, thr=12):
    if idx_leg < 0 or idx_leg >= len(legs)-1: return None
    cur, nxt = legs[idx_leg]["TC"], legs[idx_leg+1]["TC"]
    turn = angdiff(nxt, cur)
    if turn > thr:  return +1
    if turn < -thr: return -1
    return None

def choose_anchor_teimoso(L, zones: Zones, side_off, label_radius_nm, prefer=None):
    sides_order = [prefer, -prefer] if prefer in (-1,+1) else [+1, -1]
    base_cands = label_candidates(L, side_off)

    def try_place(anchor_lat, anchor_lon, side, max_out_nm=6.0):
        normal = L["TC"] + 90*side
        lat,lon = anchor_lat, anchor_lon
        steps = int(max_out_nm/0.25)
        for _ in range(steps):
            if zones.clearance(lat,lon) >= LABEL_MIN_CLEAR:
                zones.add(lat,lon,label_radius_nm); return (lat,lon), side, True
            lat,lon = dest_point(lat,lon, normal, 0.25)
        # desliza ao longo da perna e repete
        for along in (0.4,0.8,1.2,1.6,2.2):
            for dirb in (L["TC"], wrap360(L["TC"]+180)):
                p = dest_point(anchor_lat, anchor_lon, dirb, along)
                lat,lon = p
                for _ in range(steps):
                    if zones.clearance(lat,lon) >= LABEL_MIN_CLEAR:
                        zones.add(lat,lon,label_radius_nm); return (lat,lon), side, True
                    lat,lon = dest_point(lat,lon, normal, 0.25)
        return (anchor_lat, anchor_lon), side, False

    for side in sides_order:
        cands_side = [c for c in base_cands if c[1]==side] + [c for c in base_cands if c[1]!=side]
        cands_side.sort(key=lambda c: zones.clearance(c[0][0], c[0][1]), reverse=True)
        for (anchor, s, _off) in cands_side:
            (lat,lon), s2, ok = try_place(anchor[0], anchor[1], side)
            if ok: return (lat,lon), s2

    # último recurso
    mid = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], 0.5*L["Dist"])
    anchor = dest_point(mid[0], mid[1], L["TC"]+90, side_off+6.0)
    zones.add(anchor[0], anchor[1], max(label_radius_nm, 1.4))
    return (anchor[0], anchor[1]), +1

# ======== MAPA ========
def _bounds_from_nodes(nodes):
    lats = [n["lat"] for n in nodes]; lons = [n["lon"] for n in nodes]
    return [(min(lats),min(lons)), (max(lats),max(lons))]

def _wp_time_fuel(nodes, legs):
    info = [{"eto": None, "efob": None} for _ in nodes]
    if not legs: return info
    info[0]["eto"]  = legs[0]["clock_start"]; info[0]["efob"] = legs[0]["efob_start"]
    for i in range(1, len(nodes)):
        Lprev = legs[i-1]
        info[i]["eto"]  = Lprev["clock_end"]; info[i]["efob"] = Lprev["efob_end"]
    return info

class ZoomScaler(MacroElement):
    _template = Template("""
    {% macro script(this, kwargs) %}
    var map = {{this._parent.get_name()}};
    function navlogSetScale(){
      var z = map.getZoom();
      var s = Math.max(0.6, Math.min(1.6, 0.6 + (z-6)*0.14)); // z=6→0.6, z=12→1.44
      document.documentElement.style.setProperty('--navlog-z', s);
    }
    map.on('zoomend', navlogSetScale); navlogSetScale();
    {% endmacro %}
    """)

def render_map(nodes, legs, base_choice, maptiler_key=""):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return

    mean_lat = sum(n["lat"] for n in nodes)/len(nodes)
    mean_lon = sum(n["lon"] for n in nodes)/len(nodes)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8, tiles=None, control_scale=True, prefer_canvas=True)

    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
                         attr="© EOX Sentinel-2", name="Sentinel-2", overlay=False).add_to(m)
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png",
                         attr="© EOX Overlay", name="Labels", overlay=True).add_to(m)
    elif base_choice == "Esri World Imagery (satélite + labels)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Esri Imagery", overlay=False).add_to(m)
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Labels", overlay=True).add_to(m)
    elif base_choice == "Esri World TopoMap (topo)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Esri Topo", overlay=False).add_to(m)
    elif base_choice == "OpenTopoMap (VFR-ish)":
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                         attr="© OpenTopoMap (CC-BY-SA)", name="OpenTopoMap", overlay=False).add_to(m)
    elif base_choice == "MapTiler Satellite Hybrid (requer key)" and maptiler_key:
        folium.TileLayer(f"https://api.maptiler.com/maps/hybrid/256/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_key}",
                         attr="© MapTiler", name="MapTiler Hybrid", overlay=False).add_to(m)
    else:
        folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png",
                         attr="© OpenStreetMap", name="OSM", overlay=False).add_to(m)

    # halo + cor por perfil
    for L in legs:
        latlngs = [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])]
        color = PROFILE_COLORS.get(L["profile"], "#C000FF")
        folium.PolyLine(latlngs, color="#ffffff", weight=10, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color=color, weight=4, opacity=1.0).add_to(m)

    if st.session_state.show_ticks:
        for L in legs:
            if L["GS"]<=0 or not L["cps"]: continue
            for cp in L["cps"]:
                d = min(L["Dist"], L["GS"]*(cp["t"]/3600.0))
                latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
                llat, llon = dest_point(latm, lonm, L["TC"]-90, 0.38)
                rlat, rlon = dest_point(latm, lonm, L["TC"]+90, 0.38)
                folium.PolyLine([(llat,llon),(rlat,rlon)], color="#111111", weight=2, opacity=1).add_to(m)

    # Legenda (opcional)
    if st.session_state.show_legend:
        legend_html = f"""
        <div style="position:absolute;bottom:14px;left:14px;background:#fff;border:2px solid #111;border-radius:10px;padding:6px 8px;font-weight:800">
          <div style='display:flex;align-items:center;gap:6px;margin:2px 0'><div style='width:22px;height:4px;border-radius:3px;background:{PROFILE_COLORS["CLIMB"]}'></div>CLIMB</div>
          <div style='display:flex;align-items:center;gap:6px;margin:2px 0'><div style='width:22px;height:4px;border-radius:3px;background:{PROFILE_COLORS["LEVEL"]}'></div>CRUISE</div>
          <div style='display:flex;align-items:center;gap:6px;margin:2px 0'><div style='width:22px;height:4px;border-radius:3px;background:{PROFILE_COLORS["DESCENT"]}'></div>DESCENT</div>
        </div>
        """
        html_marker(m, mean_lat, mean_lon, legend_html)

    # zonas
    zones = Zones()
    for N in nodes: zones.add(N["lat"], N["lon"], ZONE_WP_R)
    for L in legs:  zones.add_leg_corridor(L["A"], L["B"])

    # pílulas (placas) — SEM setas; texto horizontal
    if st.session_state.show_labels:
        prev_side=None
        for idx, L in enumerate(legs):
            if L["Dist"] < LABEL_MIN_NM_NORMAL: continue
            # tamanho relativo à distância e ao slider (aplica-se via scale() local)
            base = min(1.25, max(0.85, L["Dist"]/7.0))
            s = base  # o slider é multiplicado via --navlog-user no CSS
            side_off = min(2.2, max(1.1, 1.10*s))
            label_r = ZONE_LABEL_BASE_R + 0.30*(s-1.0)
            prefer = preferred_side_outside_turn(legs, idx) or prev_side
            anchor, side = choose_anchor_teimoso(L, zones, side_off, label_r, prefer=prefer)
            prev_side = side
            line1 = f"{deg3(L['MH'])}M / {deg3(L['TC'])}T"
            line2 = f"{rint(L['GS'])} kt · {mmss(L['time_sec'])} · {L['Dist']:.1f} nm · {L['burn']:.1f} L"
            html_marker(m, anchor[0], anchor[1], pill_html(line1, line2, L["TC"], scale=s))

    # nomes e info por baixo do fix (sem líder)
    info = _wp_time_fuel(nodes, legs)
    for idx, N in enumerate(nodes):
        is_toc_tod = str(N["name"]).startswith(("TOC","TOD"))
        color = "#FF5050" if is_toc_tod else "#007AFF"
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#FFFFFF",
                            weight=2, fill=True, fill_color=color, fill_opacity=1).add_to(m)
        # nome (logo acima do ponto)
        html_marker(m, N["lat"], N["lon"],
                    f"<div class='nav-wpname' style='transform:translate(-50%,-140%);'>{idx+1}. {N['name']}</div>")
        # info (logo abaixo do ponto)
        eto = info[idx]["eto"] or "-"
        efb = info[idx]["efob"]; efb_txt = f"{efb:.1f}L" if efb is not None else "-"
        html_marker(m, N["lat"], N["lon"],
                    f"<div class='nav-wpinfo' style='transform:translate(-50%, 40%);'> {eto} · {efb_txt} </div>")

    # ajustar bounds e injetar escalador de zoom
    try: m.fit_bounds(_bounds_from_nodes(nodes), padding=(30,30))
    except: pass
    m.add_child(ZoomScaler())
    st_folium(m, width=None, height=760)

# ---- render ----
if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs,
               base_choice=st.session_state.map_base, maptiler_key=st.session_state.maptiler_key)
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")

