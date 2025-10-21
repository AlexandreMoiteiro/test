# app.py — NAVLOG (Folium VFR + PDF) — rev3
# - OpenTopoMap por defeito
# - FULLSCREEN no mapa (Leaflet.Fullscreen)
# - Seleção de WPs ultra simples:
#     • Caixa de pesquisa -> ENTER adiciona logo o 1.º resultado (1 ação)
#     • Sugestões (top 8) clicáveis (1 clique)
#     • Clique no mapa e Colar lista continuam disponíveis
# - Rótulos de legs limpos (pílulas) + “MH” grande com leader line (anti-sobreposição)
# - Riscas exatamente a cada 2 min pela GS (sem confusão visual)
# - Nomes de WPs a PRETO com halo (visível em qualquer base)

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
from math import sin, asin, radians, degrees

# ================== CONSTANTES ==================
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0  # kt
FUEL_FLOW = 20.0  # L/h
EARTH_NM  = 3440.065
TICK_EVERY_MIN = 2  # riscas sempre de 2 min

# ================== PAGE / STYLE ==================
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

# ================== HELPERS ==================
rt10 = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
hhmmss = lambda t: f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
rint = lambda x: int(round(float(x)))
r10f = lambda x: round(float(x), 1)
rang = lambda x: int(round(float(x))) % 360
wrap360 = lambda x: (x % 360 + 360) % 360
def angdiff(a, b): return (a - b + 180) % 360 - 180

def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas,1e-9)))
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
    θ = math.radians(bearing_deg)
    δ = dist_nm / EARTH_NM
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

# ======== LABELS (sem sobreposição) ========
LABEL_MIN_NM    = 4.0
LABEL_SIDE_OFF  = 0.65
LABEL_LINE_OFF  = 0.32

def _nm_dist(a,b):
    return gc_dist_nm(a[0],a[1],b[0],b[1])

def add_text_marker(map_obj, lat, lon, text, size_px=16, color="#111111", halo=True, weight="800"):
    shadow = "text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;" if halo else ""
    html=f"""<div style="font-size:{size_px}px;color:{color};font-weight:{weight};{shadow};white-space:nowrap;">{text}</div>"""
    folium.Marker(location=(lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(map_obj)

def add_label_box(m, lat, lon, text, big=False):
    if big:
        html = f"""
        <div style="font-size:26px;color:#FFD700;font-weight:900;
                    text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;
                    white-space:nowrap;">{text}</div>
        """
    else:
        html = f"""
        <div style="font-size:13px; background:#fff; border:1px solid #000;
                    border-radius:14px; padding:4px 8px; box-shadow:0 1px 2px rgba(0,0,0,.25);
                    white-space:nowrap;">{text}</div>
        """
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def best_label_anchor(L, used_points):
    candidates = []
    for frac in (0.33, 0.67):
        along = max(0.6, min(L["Dist"]-0.6, L["Dist"]*frac))
        base_lat, base_lon = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], along)
        for side in (-1, +1):
            lab_lat, lab_lon = dest_point(base_lat, base_lon, L["TC"]+90*side, LABEL_SIDE_OFF)
            dists = [ _nm_dist((lab_lat,lab_lon), (p[0],p[1])) for p in used_points ]
            d_wpA = _nm_dist((lab_lat,lab_lon), (L["A"]["lat"],L["A"]["lon"]))
            d_wpB = _nm_dist((lab_lat,lab_lon), (L["B"]["lat"],L["B"]["lon"]))
            score = min(dists+[d_wpA, d_wpB, 999])
            candidates.append((score, (lab_lat,lab_lon), (base_lat,base_lon), side))
    _, anchor, base, side = max(candidates, key=lambda x: x[0])
    return anchor, base, side

# ================== STATE ==================
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")     # 👉 padrão OpenTopo
ens("maptiler_key", "")
ens("qadd", ""); ens("alt_qadd", 3000.0)
ens("db_points", None)

# ================== HEADER ==================
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
a,b,c,d = st.columns([3,3,2,2])
with a: st.title("NAVLOG — Folium VFR + PDF")
with b: st.caption("TAS 70/90/90 · FF 20 L/h · offsets em NM · pronto a imprimir")
with c:
    if st.button("➕ WP", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with d:
    if st.button("🗑️ Limpar", use_container_width=True):
        for k in ["wps","legs","route_nodes"]: st.session_state[k] = []
st.markdown("</div>", unsafe_allow_html=True)

# ================== PARÂMETROS ==================
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

    b1, b2 = st.columns([2,2])
    with b1:
        bases = [
            "OpenTopoMap (VFR-ish)",
            "EOX Sentinel-2 (satélite)",
            "Esri World Imagery (satélite + labels)",
            "Esri World TopoMap (topo)",
            "OSM Standard",
            "MapTiler Satellite Hybrid (requer key)"
        ]
        cur = st.session_state.get("map_base", bases[0])
        idx = bases.index(cur) if cur in bases else 0
        st.session_state.map_base = st.selectbox("Base do mapa", bases, index=idx, key="map_base_choice")
    with b2:
        if "MapTiler" in st.session_state.map_base:
            st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key)

    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================== CSVs LOCAIS ==================
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

# base de pesquisa única em memória
if st.session_state.db_points is None:
    db = pd.concat([ad_df, loc_df]).dropna(subset=["lat","lon"]).reset_index(drop=True)
    st.session_state.db_points = db
else:
    db = st.session_state.db_points

# ================== ADICIONAR WPs — ULTRA SIMPLES ==================
st.subheader("Adicionar waypoints (rápido)")

def add_wp_unique(name, lat, lon, alt):
    for w in st.session_state.wps:
        if str(w["name"]).strip().lower() == str(name).strip().lower():
            if gc_dist_nm(w["lat"], w["lon"], lat, lon) <= 0.2:
                return False
    st.session_state.wps.append({"name": str(name), "lat": float(lat), "lon": float(lon), "alt": float(alt)})
    return True

def auto_add_first_result():
    q = st.session_state.qadd.strip().lower()
    alt = float(st.session_state.alt_qadd)
    if not q: return
    res = db[db.apply(lambda r: any(q in str(v).lower() for v in r.values), axis=1)]
    if res.empty: return
    r = res.iloc[0]
    ok = add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), alt)
    st.session_state.qadd = ""  # limpa a caixa
    st.session_state._last_msg = ("success", f"Adicionado: {r.get('code') or r.get('name')}") if ok else ("warning", "Já existia perto com o mesmo nome.")

c1, c2 = st.columns([3,1])
with c1:
    st.text_input("Escreve e carrega ENTER para adicionar o 1.º resultado",
                  key="qadd", on_change=auto_add_first_result,
                  placeholder="Ex: LPPT, ÉVORA, NISA, ABRANTES…")
with c2:
    st.number_input("Alt (ft) para novos WPs", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0, key="alt_qadd")

# mostra mensagem do auto-add (sem DeltaGenerator no output)
if "_last_msg" in st.session_state:
    typ, msg = st.session_state._last_msg
    (st.success if typ=="success" else st.warning)(msg)
    del st.session_state._last_msg

# sugestões clicáveis (top 8) conforme se escreve
q_preview = st.session_state.qadd.strip().lower()
if q_preview:
    res_prev = db[db.apply(lambda r: any(q_preview in str(v).lower() for v in r.values), axis=1)].head(8)
    if not res_prev.empty:
        st.caption("Sugestões (clique para adicionar):")
        for i, r in res_prev.iterrows():
            label = f"[{r['src']}] {r.get('code','')} — {r.get('name','')}  ({r['lat']:.4f}, {r['lon']:.4f})"
            if st.button(label, key=f"sug_{i}"):
                ok = add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_qadd))
                (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")

# ======== Alternativas: Mapa (clique) e Colar lista ========
tabs = st.tabs(["🗺️ Adicionar no mapa", "📋 Colar lista"])
with tabs[0]:
    st.caption("Clica no mapa e depois em **Adicionar**.")
    m0 = folium.Map(location=[39.7, -8.1], zoom_start=7,
                    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                    attr="© OpenTopoMap", control_scale=True)
    for w in st.session_state.wps:
        folium.CircleMarker((w["lat"],w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1).add_to(m0)
    map_out = st_folium(m0, width=None, height=420, key="pickmap")
    with st.form("add_by_click"):
        cA,cB,cC = st.columns([2,1,1])
        with cA: nm = st.text_input("Nome", "WP novo")
        with cB: alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
        clicked = map_out.get("last_clicked")
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            ok = add_wp_unique(nm, clicked["lat"], clicked["lng"], alt)
            (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")

with tabs[1]:
    st.caption("Formato por linha: `NOME; LAT; LON; ALT` — aceita DD ou DMS compactas (`390712N 0083155W`). ALT opcional.")
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
            n += 1 if add_wp_unique(name, lat, lon, alt) else 0
        st.success(f"Adicionados {n} WPs.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ================== EDITOR WPs ==================
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

# ================== TOC/TOD AS WPs ==================
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

# ================== LEGS ==================
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

# ================== GERAR ROTA/LEGS ==================
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

# ================== RESUMO ==================
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

# ================== MAPA (FOLIUM) ==================
def _bounds_from_nodes(nodes):
    lats = [n["lat"] for n in nodes]; lons = [n["lon"] for n in nodes]
    return [(min(lats),min(lons)), (max(lats),max(lons))]

def _route_latlngs(legs):
    return [[(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])] for L in legs]

def add_print_and_fullscreen(m):
    # Print
    m.get_root().header.add_child(folium.Element(
        '<script src="https://unpkg.com/leaflet.browser.print/dist/leaflet.browser.print.min.js"></script>'
    ))
    # Fullscreen
    m.get_root().header.add_child(folium.Element(
        '<link rel="stylesheet" href="https://unpkg.com/leaflet.fullscreen/Control.FullScreen.css" />'
    ))
    m.get_root().header.add_child(folium.Element(
        '<script src="https://unpkg.com/leaflet.fullscreen/Control.FullScreen.js"></script>'
    ))
    tpl = """
    {% macro script(this, kwargs) %}
      const mp = {{this._parent.get_name()}};
      L.control.browserPrint({position:'topleft', title:'Export PDF/PNG',
                              printModes:['Portrait','Landscape','Auto','Custom']}).addTo(mp);
      if (L.control.fullscreen) { L.control.fullscreen({position:'topleft', title:'Fullscreen'}).addTo(mp); }
    {% endmacro %}
    """
    macro = MacroElement(); macro._template = Template(tpl)
    m.get_root().script.add_child(macro)

def render_map(nodes, legs, base_choice, maptiler_key=""):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return

    mean_lat = sum(n["lat"] for n in nodes)/len(nodes)
    mean_lon = sum(n["lon"] for n in nodes)/len(nodes)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8, tiles=None, control_scale=True, prefer_canvas=True)

    # bases
    if base_choice == "EOX Sentinel-2 (satélite)":
        folium.TileLayer(
            tiles="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
            attr="© EOX Sentinel-2", name="Sentinel-2", overlay=False
        ).add_to(m)
        folium.TileLayer(
            tiles="https://tiles.maps.eox.at/wmts/1.0.0/overlay_bright/GoogleMapsCompatible/{z}/{y}/{x}.png",
            attr="© EOX Overlay", name="Labels", overlay=True
        ).add_to(m)
    elif base_choice == "Esri World Imagery (satélite + labels)":
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri", name="Esri Imagery", overlay=False
        ).add_to(m)
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri", name="Labels", overlay=True
        ).add_to(m)
    elif base_choice == "Esri World TopoMap (topo)":
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="© Esri", name="Esri Topo", overlay=False
        ).add_to(m)
    elif base_choice == "OpenTopoMap (VFR-ish)":
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="© OpenTopoMap (CC-BY-SA)", name="OpenTopoMap", overlay=False
        ).add_to(m)
    elif base_choice == "MapTiler Satellite Hybrid (requer key)" and maptiler_key:
        folium.TileLayer(
            tiles=f"https://api.maptiler.com/maps/hybrid/256/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_key}",
            attr="© MapTiler", name="MapTiler Hybrid", overlay=False
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="© OpenStreetMap", name="OSM", overlay=False
        ).add_to(m)

    # rota com HALO mais forte
    for latlngs in _route_latlngs(legs):
        folium.PolyLine(latlngs, color="#ffffff", weight=9, opacity=1.0).add_to(m)
        folium.PolyLine(latlngs, color="#C000FF", weight=4, opacity=1.0).add_to(m)

    # riscas — exatamente a cada 2 min pela GS
    for L in legs:
        if L["GS"]<=0 or L["time_sec"]<120: continue
        t = 120
        while t <= L["time_sec"]:
            d = min(L["Dist"], L["GS"]*(t/3600.0))
            latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
            llat, llon = dest_point(latm, lonm, L["TC"]-90, 0.17)
            rlat, rlon = dest_point(latm, lonm, L["TC"]+90, 0.17)
            folium.PolyLine([(llat,llon),(rlat,rlon)], color="#111111", weight=2, opacity=1).add_to(m)
            t += 120

    # rótulos (pílulas) + MH grande
    used = []
    for L in legs:
        if L["Dist"] < LABEL_MIN_NM or L["GS"] <= 0 or L["time_sec"] <= 0:
            continue

        anchor, base, side = best_label_anchor(L, used)
        used.append(anchor)

        # leader line
        mid_pt = dest_point(base[0], base[1], L["TC"] + 90 * side, min(LABEL_LINE_OFF, LABEL_SIDE_OFF-0.05))
        folium.PolyLine([(base[0],base[1]), (mid_pt[0], mid_pt[1])], color="#000000", weight=2, opacity=1).add_to(m)

        info_txt = f"{rang(L['TH'])}T • {rint(L['GS'])}kt • {mmss(L['time_sec'])} • {L['Dist']:.1f}nm"
        add_label_box(m, anchor[0], anchor[1], info_txt, big=False)

        mh_lat, mh_lon = dest_point(anchor[0], anchor[1], L["TC"], 0.22)
        add_label_box(m, mh_lat, mh_lon, f"MH {rang(L['MH'])}°", big=True)

    # WPs (ponto + nome a preto com halo)
    for idx, N in enumerate(nodes):
        is_toc_tod = str(N["name"]).startswith(("TOC","TOD"))
        color = "#FF5050" if is_toc_tod else "#007AFF"
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#FFFFFF",
                            weight=2, fill=True, fill_color=color, fill_opacity=1).add_to(m)
        add_text_marker(m, N["lat"], N["lon"], f"{idx+1}. {N['name']}", size_px=14, color="#111111", halo=True)

    try: m.fit_bounds(_bounds_from_nodes(nodes), padding=(30,30))
    except: pass
    add_print_and_fullscreen(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=740)

# ---- render ----
if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs,
               base_choice=st.session_state.map_base, maptiler_key=st.session_state.maptiler_key)
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")
