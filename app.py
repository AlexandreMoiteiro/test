# app.py — NAVLOG (Folium VFR + PDF) — rev5
# - OpenTopoMap por defeito
# - Seleção de WPs: pesquisa -> tabela com checkboxes (Adicionar selecionados), botões ➕ e "Adicionar 1.º"
# - Pesquisa ordenada por melhor correspondência e proximidade ao último WP
# - Mapa: rótulos anti-sobreposição (pílula + MH deslocado), riscas 2 min, halo forte na rota
# - FULLSCREEN real (folium.plugins.Fullscreen) + Export PDF/PNG

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
from folium.plugins import Fullscreen
from math import sin, radians, degrees

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0  # kt
FUEL_FLOW = 20.0  # L/h
EARTH_NM  = 3440.065

# ======== PAGE / STYLE ========
st.set_page_config(page_title="NAVLOG — Folium VFR + PDF", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:var(--line);margin:10px 0}
.sticky{position:sticky;top:0;background:#ffffffee;backdrop-filter:saturate(150%) blur(4px);z-index:50;border-bottom:1px solid var(--line);padding-bottom:8px}
</style>
""", unsafe_allow_html=True)

# ======== HELPERS ========
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
hhmmss = lambda t: f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
rint   = lambda x: int(round(float(x)))
r10f   = lambda x: round(float(x), 1)
rang   = lambda x: int(round(float(x))) % 360
wrap360= lambda x: (x % 360 + 360) % 360
def angdiff(a, b): return (a - b + 180) % 360 - 180

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    s = max(-1, min(1, (wkt * math.sin(d)) / max(tas,1e-9)))
    wca = degrees(math.asin(s))
    th  = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False):
    return wrap360(th - var if east_is_neg else th + var)

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
    y = math.sin(θ)*math.sin(δ)*math.cos(φ1)
    x = math.cos(δ) - math.sin(φ1)*sinφ2
    λ2 = λ1 + math.atan2(y, x)
    return math.degrees(φ2), ((math.degrees(λ2)+540)%360)-180

def point_along_gc(lat1, lon1, lat2, lon2, dist_from_start_nm):
    total = gc_dist_nm(lat1, lon1, lat2, lon2)
    if total <= 0: return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)

# ======== LABELS (anti-sobreposição) ========
LABEL_MIN_NM   = 3.5
LABEL_SIDE_NM  = [0.9, 1.1, 1.3]  # tentativas
LABEL_ALONG_FR = [0.30, 0.50, 0.70]
LABEL_FWD_NM   = [0.00, 0.20, -0.20]

def _nm_to_deg_lat(nm): return nm/60.0
def _nm_to_deg_lon(nm, lat): return nm/(60.0*max(0.2, math.cos(math.radians(lat))))  # evita divisão por ~0

def _bbox(anchor_lat, anchor_lon, w_nm=1.2, h_nm=0.4):
    """BBox aproximado em graus para uma pílula de texto."""
    dy = _nm_to_deg_lat(h_nm/2.0)
    dx = _nm_to_deg_lon(w_nm/2.0, anchor_lat)
    return (anchor_lat-dy, anchor_lon-dx, anchor_lat+dy, anchor_lon+dx)

def _bboxes_overlap(a, b):
    return not (a[2]<b[0] or a[0]>b[2] or a[3]<b[1] or a[1]>b[3])

def add_text_marker(m, lat, lon, text, size_px=16, color="#111", halo=True, weight="900"):
    shadow = "text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;" if halo else ""
    html=f"""<div style="font-size:{size_px}px;color:{color};font-weight:{weight};{shadow};white-space:nowrap;">{text}</div>"""
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def add_pill(m, lat, lon, text):
    html = f"""
    <div style="font-size:13px;background:#fff;border:1px solid #000;border-radius:14px;
                padding:4px 8px;box-shadow:0 1px 2px rgba(0,0,0,.25);white-space:nowrap;">{text}</div>
    """
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def add_mh_big(m, lat, lon, text):
    html = f"""
    <div style="font-size:26px;color:#FFD700;font-weight:900;
                text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;
                white-space:nowrap;">{text}</div>
    """
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def find_label_slot(L, used_bboxes):
    """Tenta várias posições (lado, avanço, afastamento) e aceita a 1.ª sem overlap."""
    for fr in LABEL_ALONG_FR:
        along = max(0.6, min(L["Dist"]-0.6, L["Dist"]*fr))
        base_lat, base_lon = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], along)
        for side in (-1, +1):
            for fwd in LABEL_FWD_NM:
                bx_lat, bx_lon = dest_point(base_lat, base_lon, L["TC"], fwd)
                for off in LABEL_SIDE_NM:
                    anc_lat, anc_lon = dest_point(bx_lat, bx_lon, L["TC"]+90*side, off)
                    bb = _bbox(anc_lat, anc_lon, w_nm=1.4, h_nm=0.45)
                    if all(not _bboxes_overlap(bb, ub) for ub in used_bboxes):
                        return (anc_lat, anc_lon), (base_lat, base_lon), side, bb
    # fallback: coloca perto do meio, mesmo lado, pode sobrepor ligeiro
    mid_lat, mid_lon = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], L["Dist"]/2)
    anc_lat, anc_lon = dest_point(mid_lat, mid_lon, L["TC"]+90, 1.0)
    bb = _bbox(anc_lat, anc_lon, w_nm=1.4, h_nm=0.45)
    return (anc_lat, anc_lon), (mid_lat, mid_lon), +1, bb

# ======== STATE ========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")  # padrão
ens("maptiler_key", "")
# pesquisa
ens("db_points", None)
ens("search_q", ""); ens("alt_new", 3000.0)
ens("search_results", pd.DataFrame())

# ======== HEADER ========
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
a,b,c,d = st.columns([3,3,2,2])
with a: st.title("NAVLOG — Folium VFR + PDF")
with b: st.caption("TAS 70/90/90 · FF 20 L/h · offsets NM · pronto a imprimir")
with c:
    if st.button("➕ WP", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with d:
    if st.button("🗑️ Limpar", use_container_width=True):
        for k in ["wps","legs","route_nodes"]: st.session_state[k] = []
st.markdown("</div>", unsafe_allow_html=True)

# ======== FORM GLOBAL ========
with st.form("globals"):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        ens("wind_from", st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from)))
        ens("wind_kt",   st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt)))
    with c2:
        ens("mag_var",  st.number_input("Variação magnética (±°)", -30.0, 30.0, float(st.session_state.mag_var)))
        ens("mag_is_e", st.toggle("Var. é EAST (subtrai)", value=st.session_state.mag_is_e))
    with c3:
        ens("roc_fpm",    st.number_input("ROC global (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10))
        ens("desc_angle", st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1))
    with c4:
        ens("start_efob",   st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5))
        ens("start_clock",  st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock))
        ens("ck_default",   st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default)))

    b1, b2 = st.columns([2,2])
    with b1:
        bases = ["OpenTopoMap (VFR-ish)","EOX Sentinel-2 (satélite)","Esri World Imagery (satélite + labels)",
                 "Esri World TopoMap (topo)","OSM Standard","MapTiler Satellite Hybrid (requer key)"]
        cur = st.session_state.get("map_base", bases[0])
        ens("map_base", st.selectbox("Base do mapa", bases, index=bases.index(cur) if cur in bases else 0))
    with b2:
        if "MapTiler" in st.session_state.map_base:
            ens("maptiler_key", st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key))
    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== CSVs ========
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

# ======== ADICIONAR WPs — Pesquisa com seleção (checkbox) ========
st.subheader("Adicionar waypoints")

def add_wp_unique(name, lat, lon, alt):
    for w in st.session_state.wps:
        if str(w["name"]).strip().lower() == str(name).strip().lower():
            if gc_dist_nm(w["lat"], w["lon"], lat, lon) <= 0.2:
                return False
    st.session_state.wps.append({"name": str(name), "lat": float(lat), "lon": float(lon), "alt": float(alt)})
    return True

def _score_row(row, tq, last_wp):
    code = str(row.get("code") or "").lower()
    name = str(row.get("name") or "").lower()
    text = f"{code} {name}"
    sim = difflib.SequenceMatcher(None, tq, text).ratio()
    starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
    near = 0.0
    if last_wp:
        near = 1.0 / (1.0 + gc_dist_nm(last_wp["lat"], last_wp["lon"], row["lat"], row["lon"]))
    return starts*2 + sim + near*0.3

with st.form("search_add"):
    c1,c2,c3 = st.columns([3,1,1])
    with c1:
        st.session_state.search_q = st.text_input("Pesquisar (nome/código)", st.session_state.search_q,
                                                  placeholder="Ex: LPPT, ALPAL, ÉVORA, NISA…")
    with c2:
        st.session_state.alt_new = st.number_input("Alt (ft) novos WPs", 0.0, 18000.0, float(st.session_state.alt_new), step=100.0)
    with c3:
        submit = st.form_submit_button("Procurar / Atualizar", use_container_width=True)

if submit or st.session_state.search_results.empty and st.session_state.search_q.strip():
    tq = st.session_state.search_q.strip().lower()
    last = st.session_state.wps[-1] if st.session_state.wps else None
    res = db[db.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)].copy() if tq else db.head(0).copy()
    if not res.empty:
        res["__score"] = res.apply(lambda r: _score_row(r, tq, last), axis=1)
        res = res.sort_values("__score", ascending=False)
    res = res.drop(columns=["__score"], errors="ignore").reset_index(drop=True)
    if not res.empty:
        res = res[["src","code","name","city","lat","lon"]]
    st.session_state.search_results = res

res = st.session_state.search_results

# ações rápidas
colA,colB = st.columns([1.2,1])
with colA:
    if st.button("➕ Adicionar 1.º resultado", disabled=res.empty, use_container_width=True):
        r = res.iloc[0]
        ok = add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_new))
        (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")
with colB:
    if st.button("➕➕ Adicionar todos visíveis (máx 50)", disabled=res.empty, use_container_width=True):
        n=0
        for _, r in res.head(50).iterrows():
            if add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_new)): n+=1
        st.success(f"Adicionados {n} WPs.")

# tabela com seleção
if not res.empty:
    st.caption("Seleciona o(s) ponto(s) correto(s) e clica **Adicionar selecionados**.")
    view = res.head(50).copy()
    view.insert(0, "✓", False)
    edited = st.data_editor(view, hide_index=True, num_rows="fixed",
                            column_config={"✓": st.column_config.CheckboxColumn("✓", help="Selecionar")})
    if st.button("Adicionar selecionados", use_container_width=True):
        sel = edited[edited["✓"]==True]
        n=0
        for _, r in sel.iterrows():
            if add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_new)): n+=1
        st.success(f"Adicionados {n} WPs.")
else:
    st.info("Sem resultados. Escreve algo acima.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== EDITOR WPs ========
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5 = st.columns([2,2,2,1,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                up = st.button("↑", key=f"up{i}"); dn = st.button("↓", key=f"dn{i}")
                if up and i>0: st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
                if dn and i < len(st.session_state.wps)-1: st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i] = {"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"): st.session_state.wps.pop(i)
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== TOC/TOD AS WPs ========
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
    nodes.append(user_wps[-1])
    return nodes

# ======== LEGS ========
def build_legs_from_nodes(nodes, wind_from, wind_kt, mag_var, mag_is_e, ck_every_min):
    legs = []
    if len(nodes) < 2: return legs

    base_time = None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None

    carry_efob = float(st.session_state.start_efob)
    t_cursor = 0

    for i in range(len(nodes)-1):
        A, B = nodes[i], nodes[i+1]
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
                cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto})
                k+=1

        legs.append({"i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
                     "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),"efob_start":efob_start,
                     "efob_end":efob_end,"clock_start":clk_start,"clock_end":clk_end,"cps":cps})
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
        + f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b> (20 L/h)</div>"
        + f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div>"
        + "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== MAPA ========
def _bounds_from_nodes(nodes):
    lats = [n["lat"] for n in nodes]; lons = [n["lon"] for n in nodes]
    return [(min(lats),min(lons)), (max(lats),max(lons))]

def _route_latlngs(legs):
    return [[(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])] for L in legs]

def add_print_and_fullscreen(m):
    # Browser Print
    m.get_root().header.add_child(folium.Element(
        '<script src="https://unpkg.com/leaflet.browser.print/dist/leaflet.browser.print.min.js"></script>'
    ))
    tpl = """
    {% macro script(this, kwargs) %}
      const mp = {{this._parent.get_name()}};
      L.control.browserPrint({position:'topleft', title:'Export PDF/PNG',
                              printModes:['Portrait','Landscape','Auto','Custom']}).addTo(mp);
    {% endmacro %}
    """
    macro = MacroElement(); macro._template = Template(tpl)
    m.get_root().script.add_child(macro)
    # Fullscreen (plugin do folium)
    Fullscreen(position='topleft', title='Fullscreen', force_separate_button=True).add_to(m)

def render_map(nodes, legs, base_choice, maptiler_key=""):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return

    mean_lat = sum(n["lat"] for n in nodes)/len(nodes)
    mean_lon = sum(n["lon"] for n in nodes)/len(nodes)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8, tiles=None, control_scale=True, prefer_canvas=True)

    # bases
    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
                         attr="© EOX Sentinel-2", name="Sentinel-2").add_to(m)
        folium.TileLayer("https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png",
                         attr="© EOX Overlay", name="Labels", overlay=True).add_to(m)
    elif base_choice == "Esri World Imagery (satélite + labels)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Esri Imagery").add_to(m)
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Labels", overlay=True).add_to(m)
    elif base_choice == "Esri World TopoMap (topo)":
        folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                         attr="© Esri", name="Esri Topo").add_to(m)
    elif base_choice == "OpenTopoMap (VFR-ish)":
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                         attr="© OpenTopoMap (CC-BY-SA)", name="OpenTopoMap").add_to(m)
    elif base_choice == "MapTiler Satellite Hybrid (requer key)" and maptiler_key:
        folium.TileLayer(f"https://api.maptiler.com/maps/hybrid/256/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_key}",
                         attr="© MapTiler", name="MapTiler Hybrid").add_to(m)
    else:
        folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                         attr="© OpenStreetMap", name="OSM").add_to(m)

    # rota com HALO forte
    for latlngs in _route_latlngs(legs):
        folium.PolyLine(latlngs, color="#ffffff", weight=10, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color="#C000FF", weight=4, opacity=1.0).add_to(m)

    # riscas a cada 2 min
    for L in legs:
        if L["GS"]<=0 or L["time_sec"]<120: continue
        t = 120
        while t <= L["time_sec"]:
            d = min(L["Dist"], L["GS"]*(t/3600.0))
            latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
            llat, llon = dest_point(latm, lonm, L["TC"]-90, 0.18)
            rlat, rlon = dest_point(latm, lonm, L["TC"]+90, 0.18)
            folium.PolyLine([(llat,llon),(rlat,rlon)], color="#111111", weight=2, opacity=1).add_to(m)
            t += 120

    # labels: pílulas + MH grande com algoritmo anti-sobreposição
    used_boxes = []
    for L in legs:
        if L["Dist"] < LABEL_MIN_NM or L["GS"]<=0 or L["time_sec"]<=0: continue
        (anc_lat, anc_lon), (base_lat, base_lon), side, bb = find_label_slot(L, used_boxes)
        used_boxes.append(bb)

        # leader line
        mid_lat, mid_lon = dest_point(base_lat, base_lon, L["TC"] + 90*side, 0.35)
        folium.PolyLine([(base_lat,base_lon), (mid_lat,mid_lon)], color="#000000", weight=2, opacity=1).add_to(m)

        # info
        info_txt = f"{rang(L['TH'])}T • {rint(L['GS'])}kt • {mmss(L['time_sec'])} • {L['Dist']:.1f}nm"
        add_pill(m, anc_lat, anc_lon, info_txt)

        # MH grande: mais para fora e um pouco "à frente" para não tapar a pílula
        mh_lat, mh_lon = dest_point(anc_lat, anc_lon, L["TC"] + 90*side, 0.22)
        mh_lat, mh_lon = dest_point(mh_lat, mh_lon, L["TC"], 0.12)
        add_mh_big(m, mh_lat, mh_lon, f"MH {rang(L['MH'])}°")

    # waypoints
    for idx, N in enumerate(nodes):
        is_toc_tod = str(N["name"]).startswith(("TOC","TOD"))
        color = "#FF5050" if is_toc_tod else "#007AFF"
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#FFFFFF",
                            weight=2, fill=True, fill_color=color, fill_opacity=1).add_to(m)
        add_text_marker(m, N["lat"], N["lon"], f"{idx+1}. {N['name']}", size_px=14, color="#111", halo=True)

    try: m.fit_bounds(_bounds_from_nodes(nodes), padding=(30,30))
    except: pass
    add_print_and_fullscreen(m)  # inclui FULLSCREEN
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=760)

# ---- render ----
if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs,
               base_choice=st.session_state.map_base, maptiler_key=st.session_state.maptiler_key)
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")



