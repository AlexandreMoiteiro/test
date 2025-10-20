
import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
from math import sin, asin, radians, degrees

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
.smallcaps{font-variant:all-small-caps;letter-spacing:.04em;color:#4b5563}
</style>
""", unsafe_allow_html=True)

# ======== HELPERS ========
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

def triangle_coords(lat, lon, heading_deg, h_nm=0.95, w_nm=0.7):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return [(bl_lat, bl_lon), (apex_lat, apex_lon), (br_lat, br_lon)]

# ======== STATE ========

def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")   # 👉 agora garantidamente por defeito
ens("maptiler_key", "")  # opcional
# Aparência/antioverlap
ens("show_dogs", True); ens("show_ticks", True)
ens("label_min_sep_nm", 3.0)  # separação mínima entre etiquetas/doghouses
ens("label_size_mh", 26); ens("label_size_info", 14)
# Picker / inserção
ens("picker_default_alt", 3000.0)
ens("insert_after", 0)  # 0 = antes do primeiro; len = ao fim

# ======== HEADER ========
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

# ======== PARÂMETROS (form corrigido) ========
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
            "EOX Sentinel-2 (satélite)",
            "Esri World Imagery (satélite + labels)",
            "Esri World TopoMap (topo)",
            "OpenTopoMap (VFR-ish)",
            "OSM Standard",
            "MapTiler Satellite Hybrid (requer key)"
        ]
        # 👉 Escolha robusta: se a base guardada não existir, cai para OpenTopoMap
        cur = st.session_state.get("map_base", "OpenTopoMap (VFR-ish)")
        if cur not in bases:
            cur = "OpenTopoMap (VFR-ish)"
            st.session_state.map_base = cur
        idx = next((i for i, v in enumerate(bases) if v == cur), bases.index("OpenTopoMap (VFR-ish)"))
        choice = st.selectbox("Base do mapa", bases, index=idx, key="map_base_choice")
        st.session_state.map_base = choice
    with b2:
        if "MapTiler" in st.session_state.map_base:
            st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key)

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    e1,e2,e3,e4 = st.columns(4)
    with e1:
        st.session_state.show_dogs = st.toggle("Mostrar dog houses", value=st.session_state.show_dogs)
    with e2:
        st.session_state.show_ticks = st.toggle("Riscas de 2 min", value=st.session_state.show_ticks)
    with e3:
        st.session_state.label_min_sep_nm = st.slider("Separação mínima etiquetas (NM)", 0.5, 8.0, float(st.session_state.label_min_sep_nm), 0.5)
    with e4:
        st.session_state.label_size_mh = st.slider("Tamanho MH", 14, 40, int(st.session_state.label_size_mh), 1)

    submitted = st.form_submit_button("Aplicar")  # <-- garante submit

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== CSVs locais (AD/LOC) ========
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

# ======== CATÁLOGO + PESQUISA (melhorado) ========

# Unifica e elimina duplicados por coordenada aproximada
CAT = pd.concat([ad_df, loc_df], ignore_index=True)
if not CAT.empty:
    CAT["key"] = CAT.apply(lambda r: f"{round(float(r['lat']),4)}|{round(float(r['lon']),4)}", axis=1)
    CAT = CAT.sort_values(["src","name","code"]).drop_duplicates("key")

cflt1, cflt2, cflt3 = st.columns([2.5,1.3,1.2])
with cflt1:
    qtxt = st.text_input("🔎 Procurar AD/Localidade (CSV local)", "", placeholder="Ex: LPPT, ABRANTES, NISA…")
with cflt2:
    st.session_state.picker_default_alt = st.number_input("Altitude p/ novos WPs (ft)", 0.0, 18000.0, float(st.session_state.picker_default_alt), step=100.0)
with cflt3:
    st.session_state.insert_after = st.number_input("Inserir depois de # (0=início)", 0, max(0, len(st.session_state.wps)), int(st.session_state.insert_after))

if not CAT.empty:
    results = CAT.copy()
    if qtxt.strip():
        tq = qtxt.lower().strip()
        results = results[results.apply(lambda r: any(tq in str(v).lower() for v in [r.get("code"), r.get("name"), r.get("city"), r.get("sector")]), axis=1)]

    with st.expander("Selecionar via pesquisa", expanded=bool(qtxt)):
        if results.empty:
            st.info("Sem resultados.")
        else:
            view = results[["src","code","name","city","sector","lat","lon"]].reset_index(drop=True)
            view["add"] = False
            edited = st.data_editor(view, use_container_width=True, num_rows="dynamic", key="search_editor")
            to_add = edited[edited["add"] == True]
            if st.button("Adicionar selecionados (sem duplicar)"):
                count = 0
                ins = int(st.session_state.insert_after)
                for _, r in to_add.iterrows():
                    # evita duplicados por coordenada
                    exists = any(abs(w["lat"]-float(r["lat"]))<1e-5 and abs(w["lon"]-float(r["lon"]))<1e-5 for w in st.session_state.wps)
                    if exists: continue
                    wp = {"name": str(r.get("code") or r.get("name")), "lat": float(r["lat"]), "lon": float(r["lon"]), "alt": float(st.session_state.picker_default_alt)}
                    st.session_state.wps.insert(min(ins, len(st.session_state.wps)), wp)
                    ins += 1; count += 1
                st.success(f"Adicionados {count} WPs.")

# ======== PICKER por mapa (clique) ========

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.subheader("Escolher WPs por clique no mapa")

# Helpers de base para não duplicar código

def add_print_button(m):
    m.get_root().header.add_child(folium.Element(
        '<script src="https://unpkg.com/leaflet.browser.print/dist/leaflet.browser.print.min.js"></script>'
    ))
    tpl = """
    {% macro script(this, kwargs) %}
      L.control.browserPrint({
        position:'topleft',
        title: 'Export PDF/PNG',
        printModes: ['Portrait','Landscape','Auto','Custom']
      }).addTo({{this._parent.get_name()}});
    {% endmacro %}
    """
    macro = MacroElement(); macro._template = Template(tpl)
    m.get_root().script.add_child(macro)


def add_base_layers_to_map(m, base_choice, maptiler_key=""):
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


# Mapa pequeno para picking
center = [39.6, -8.2]
picker_map = folium.Map(location=center, zoom_start=7, tiles=None, control_scale=True)
add_base_layers_to_map(picker_map, st.session_state.map_base, st.session_state.maptiler_key)
folium.LayerControl(collapsed=False).add_to(picker_map)
add_print_button(picker_map)

out_pick = st_folium(picker_map, width=None, height=420, key="picker_map")
if out_pick and out_pick.get("last_clicked"):
    lat = float(out_pick["last_clicked"]["lat"]); lon = float(out_pick["last_clicked"]["lng"]) 
    with st.form("add_from_click", clear_on_submit=True):
        st.markdown("**Último clique:** "+f"{lat:.5f}, {lon:.5f}")
        name = st.text_input("Nome do WP", value=f"DROP {len(st.session_state.wps)+1}")
        alt = st.number_input("Altitude (ft)", 0.0, 18000.0, float(st.session_state.picker_default_alt), step=50.0)
        ins = st.number_input("Inserir depois de # (0=início)", 0, max(0, len(st.session_state.wps)), int(st.session_state.insert_after))
        ok = st.form_submit_button("Adicionar WP do clique")
        if ok:
            st.session_state.wps.insert(min(int(ins), len(st.session_state.wps)), {"name":name,"lat":lat,"lon":lon,"alt":float(alt)})
            st.success("WP adicionado.")

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
else:
    st.info("Adiciona WPs pela pesquisa, clique no mapa, ou botão ➕ WP.")

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

        # CPs (texto)
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

# ======== GERAR ROTA/LEGS ========
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

# ======== MAPA (FOLIUM) ========

def _bounds_from_nodes(nodes):
    lats = [n["lat"] for n in nodes]; lons = [n["lon"] for n in nodes]
    return [(min(lats),min(lons)), (max(lats),max(lons))]


def _route_latlngs(legs):
    return [[(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])] for L in legs]


def _add_text(map_obj, lat, lon, text, size_px=22, color="#FFD700", halo=True, weight="700"):
    shadow = "text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;" if halo else ""
    html=f"""<div style=\"font-size:{size_px}px;color:{color};font-weight:{weight};{shadow};white-space:nowrap;\">{text}</div>"""
    folium.Marker(location=(lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(map_obj)


# ---------- Colocação inteligente (anti-sobreposição) ----------

def best_label_anchor_for_leg(L, placed_points, min_sep_nm=3.0):
    """Encontra um ponto para etiqueta/doghouse ao longo da perna L, afastado dos já colocados.
    Estratégia: testar posições a 35%, 50% e 65% do comprimento, em ambos os lados e com offsets perpendiculares crescentes."""
    along_fracs = [0.35, 0.5, 0.65]
    perp_offsets = [0.4, 0.6, 0.85, 1.2, 1.6]  # NM
    tc = L["TC"]; dist = L["Dist"]

    def ok(lat, lon):
        return all(gc_dist_nm(lat, lon, p[0], p[1]) >= min_sep_nm for p in placed_points)

    # tenta candidatos
    for f in along_fracs:
        along_nm = max(0.2, min(dist-0.2, dist * f))
        base_lat, base_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], along_nm)
        for side in (+1, -1):
            for off in perp_offsets:
                lat, lon = dest_point(base_lat, base_lon, tc + side*90, off)
                if ok(lat, lon):
                    return lat, lon, side
    # fallback: se tudo falhar, usar meio da perna com pequeno offset
    mid_lat, mid_lon = point_along_gc(L["A"]["lat"], L["A"]["lon"], L["B"]["lat"], L["B"]["lon"], dist/2)
    lat, lon = dest_point(mid_lat, mid_lon, tc + 90, 0.5)
    return lat, lon, +1


def render_map(nodes, legs, base_choice, maptiler_key=""):
    if not nodes or not legs:
        st.info("Adiciona pelo menos 2 WPs e carrega em **Gerar/Atualizar rota**.")
        return

    mean_lat = sum(n["lat"] for n in nodes)/len(nodes)
    mean_lon = sum(n["lon"] for n in nodes)/len(nodes)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8, tiles=None, control_scale=True, prefer_canvas=True)

    add_base_layers_to_map(m, base_choice, maptiler_key)

    # rota com HALO
    for latlngs in _route_latlngs(legs):
        folium.PolyLine(latlngs, color="#ffffff", weight=8, opacity=0.9).add_to(m)
        folium.PolyLine(latlngs, color="#C000FF", weight=4, opacity=1.0).add_to(m)

    # riscas 2 min (opcional)
    if st.session_state.show_ticks:
        for L in legs:
            if L["GS"]<=0 or L["time_sec"]<=0: continue
            k, step = 1, 120
            while k*step <= L["time_sec"]:
                t = k*step
                d = min(L["Dist"], L["GS"]*(t/3600.0))
                latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
                llat, llon = dest_point(latm, lonm, L["TC"]-90, 0.16)
                rlat, rlon = dest_point(latm, lonm, L["TC"]+90, 0.16)
                folium.PolyLine([(llat,llon),(rlat,rlon)], color="#000000", weight=2, opacity=1).add_to(m)
                k += 1

    # dog houses + etiquetas (anti-sobreposição)
    placed = []  # âncoras já usadas
    if st.session_state.show_dogs:
        for L in legs:
            latc, lonc, side = best_label_anchor_for_leg(L, placed, st.session_state.label_min_sep_nm)
            # Triângulo
            tri = triangle_coords(latc, lonc, L["TC"], h_nm=1.0, w_nm=0.72)
            folium.Polygon(tri, color="#000000", weight=2, fill=True, fill_color="#FFFFFF", fill_opacity=0.92).add_to(m)
            # MH grande
            _add_text(m, latc, lonc, f"MH {rang(L['MH'])}°", size_px=int(st.session_state.label_size_mh), color="#FFD700")
            # Linha/etiqueta informativa mais ao exterior
            info_lat, info_lon = dest_point(latc, lonc, L["TC"]+side*90, 0.22)
            _add_text(m, info_lat, info_lon,
                      f"{rang(L['TH'])}T • {rint(L['GS'])}kt • {mmss(L['time_sec'])} • {L['Dist']:.1f}nm",
                      size_px=int(st.session_state.label_size_info), color="#000000", halo=True, weight="600")
            placed.append((latc, lonc))
            placed.append((info_lat, info_lon))

    # waypoints
    for idx, N in enumerate(nodes):
        is_toc_tod = str(N["name"]).startswith(("TOC","TOD"))
        color = "#FF5050" if is_toc_tod else "#007AFF"
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#FFFFFF",
                            weight=2, fill=True, fill_color=color, fill_opacity=1).add_to(m)
        _add_text(m, N["lat"], N["lon"], f"{idx+1}. {N['name']}", size_px=14, color="#FFFFFF")

    try: m.fit_bounds(_bounds_from_nodes(nodes), padding=(30,30))
    except: pass
    add_print_button(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=720, key="main_map")

# ---- render ----
if st.session_state.wps and st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs,
               base_choice=st.session_state.map_base, maptiler_key=st.session_state.maptiler_key)
elif st.session_state.wps:
    st.info("Carrega em **Gerar/Atualizar rota** para inserir TOC/TOD e criar as legs.")
else:
    st.info("Adiciona pelo menos 2 waypoints.")
