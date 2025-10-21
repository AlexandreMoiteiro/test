
# app.py — NAVLOG (Folium VFR + PDF) — rev5
# • OpenTopoMap por defeito
# • Pesquisa de WPs com DESAMBIGUAÇÃO clara:
#    - Caixa de pesquisa -> mostra resultados
#    - Tabela editável com coluna "Add" (vários de uma vez) + rádio para escolher 1 e botão "Adicionar"
#    - Ordenação por semelhança e proximidade ao último WP
# • Mapa com FULLSCREEN (folium.plugins.Fullscreen) e impressão (Leaflet.Browser.Print)
# • Render das legs refeito: rótulos anti-sobreposição, pílulas limpas, MH grande deslocado,
#   riscas a cada 2 min pela GS, nomes dos WPs a preto com halo.

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib
from streamlit_folium import st_folium
from branca.element import Template, MacroElement
from folium.plugins import Fullscreen
from math import sin, asin, radians, degrees

# ========= CONSTANTES =========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0  # kt
FUEL_FLOW = 20.0  # L/h
EARTH_NM  = 3440.065

# ========= PAGE / STYLE =========
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

# ========= HELPERS =========
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

# ========= LABELS / ANTI-SOBREPOSIÇÃO =========
LABEL_MIN_NM    = 4.0
LABEL_SIDE_NM   = [0.9, 1.2, 1.5]        # candidatos de afastamento lateral
LABEL_ALONG_FR  = [0.25, 0.45, 0.60, 0.75]
LABEL_LINE_OFF  = 0.45
LABEL_MIN_SEP   = 0.75                    # separação mínima entre âncoras (NM)

def _nm_dist(a,b): return gc_dist_nm(a[0],a[1],b[0],b[1])

def add_div_text(m, lat, lon, html):
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)

def add_wp_name(m, lat, lon, text):
    html=f"""<div style="font-size:14px;color:#111;font-weight:900;
             text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;
             white-space:nowrap;">{text}</div>"""
    add_div_text(m, lat, lon, html)

def add_pill(m, lat, lon, text):
    html=f"""<div style="font-size:13px;background:#fff;border:1px solid #000;
             border-radius:14px;padding:4px 8px;box-shadow:0 1px 2px rgba(0,0,0,.25);
             white-space:nowrap;">{text}</div>"""
    add_div_text(m, lat, lon, html)

def add_mh_big(m, lat, lon, text):
    html=f"""<div style="font-size:26px;color:#FFD700;font-weight:900;
             text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;
             white-space:nowrap;">{text}</div>"""
    add_div_text(m, lat, lon, html)

def place_leg_labels(L, occupied_pts, crowd_pts):
    """Escolhe a melhor âncora evitando sobreposição: tenta várias posições
       (frações ao longo da perna, dois lados e vários afastamentos)."""
    best = None
    for frac in LABEL_ALONG_FR:
        along = max(0.8, min(L["Dist"]-0.8, L["Dist"]*frac))
        base_lat, base_lon = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], along)
        for side in (-1,+1):
            for off in LABEL_SIDE_NM:
                lat, lon = dest_point(base_lat, base_lon, L["TC"]+90*side, off)
                # separação a tudo o que já existe
                dlist = [_nm_dist((lat,lon), p) for p in (occupied_pts+crowd_pts+[(L["A"]["lat"],L["A"]["lon"]),(L["B"]["lat"],L["B"]["lon"])])]
                sep = min(dlist) if dlist else 999
                score = sep   # maior é melhor
                cand = (score, (lat,lon), (base_lat,base_lon), side, off)
                if best is None or score > best[0]: best = cand
                if sep >= LABEL_MIN_SEP:   # bom o suficiente, aceita já
                    return (lat,lon), (base_lat,base_lon), side, off
    # se nenhuma cumprir separação mínima, usa a melhor pontuação
    _, anchor, base, side, off = best
    return anchor, base, side, off

# ========= STATE =========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("wps", []); ens("legs", []); ens("route_nodes", [])
ens("map_base", "OpenTopoMap (VFR-ish)")    # padrão
ens("maptiler_key", "")
# pesquisa
ens("q", ""); ens("alt_new", 3000.0)
ens("db", None); ens("results_df", pd.DataFrame()); ens("radio_pick", None)

# ========= HEADER =========
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
a,b,c,d = st.columns([3,3,2,2])
with a: st.title("NAVLOG — Folium VFR + PDF")
with b: st.caption("TAS 70/90/90 · FF 20 L/h · offsets em NM · pronto a imprimir")
with c:
    if st.button("➕ WP manual", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with d:
    if st.button("🗑️ Limpar", use_container_width=True):
        for k in ["wps","legs","route_nodes"]: st.session_state[k] = []
st.markdown("</div>", unsafe_allow_html=True)

# ========= PARÂMETROS =========
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
        st.session_state.map_base = st.selectbox("Base do mapa", bases, index=bases.index(cur))
    with b2:
        if "MapTiler" in st.session_state.map_base:
            st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key)

    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ========= CSVs LOCAIS =========
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

# base de pesquisa
if st.session_state.db is None:
    st.session_state.db = pd.concat([ad_df, loc_df]).dropna(subset=["lat","lon"]).reset_index(drop=True)
db = st.session_state.db

# ========= ADICIONAR WPs — DESAMBIGUAÇÃO =========
st.subheader("Adicionar waypoints (pesquisa com seleção)")

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

def find_results(q):
    tq = q.strip().lower()
    if not tq: return db.head(0)
    last = st.session_state.wps[-1] if st.session_state.wps else None
    df = db[db.apply(lambda r: any(tq in str(v).lower() for v in r.values), axis=1)].copy()
    if df.empty: return df
    df["score"] = df.apply(lambda r: _score_row(r, tq, last), axis=1)
    df = df.sort_values("score", ascending=False)
    df = df[["src","code","name","city","sector","lat","lon"]]
    return df

c1, c2 = st.columns([3,1])
with c1:
    st.session_state.q = st.text_input("Pesquisar (nome/código) — escolhe e clica em 'Adicionar selecionados'",
                                       value=st.session_state.q, placeholder="Ex: LPPT, ALPAL, ÉVORA, NISA…")
with c2:
    st.session_state.alt_new = st.number_input("Alt (ft) p/ novos WPs", 0.0, 18000.0, float(st.session_state.alt_new), step=100.0)

if st.session_state.q is not None:
    res = find_results(st.session_state.q)
    if not res.empty:
        # rádio para escolha única
        labels = [f"[{r['src']}] {str(r['code'] or '')} — {r['name']} ({r['lat']:.4f}, {r['lon']:.4f})"
                  for _, r in res.head(10).iterrows()]
        st.session_state.radio_pick = st.radio("Escolha rápida (1 item)", options=["—"]+labels, horizontal=False, index=0, key="rad")
        if st.button("Adicionar escolhido (rápido)") and st.session_state.radio_pick != "—":
            idx = labels.index(st.session_state.radio_pick)
            r = res.head(10).iloc[idx]
            ok = add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_new))
            (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")

        st.write("")  # espaçamento

        # tabela com checkboxes (múltiplos)
        tbl = res.head(40).copy()
        tbl.insert(0, "Add", False)
        edited = st.data_editor(
            tbl,
            column_config={
                "Add": st.column_config.CheckboxColumn(help="Seleciona para adicionar"),
                "lat": st.column_config.NumberColumn(format="%.5f"),
                "lon": st.column_config.NumberColumn(format="%.5f"),
            },
            hide_index=True,
            use_container_width=True,
            height=360
        )
        if st.button("Adicionar selecionados"):
            sel = edited[edited["Add"]==True]
            n=0
            for _, r in sel.iterrows():
                if add_wp_unique(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_new)):
                    n+=1
            st.success(f"Adicionados {n} WPs.")
    else:
        st.info("Sem resultados.")

# ---- Alternativas: clique no mapa / colar lista
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
        cA,cB,_ = st.columns([2,1,1])
        with cA: nm = st.text_input("Nome", "WP novo")
        with cB: alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_new), step=100.0)
        clicked = map_out.get("last_clicked")
        st.write("Último clique:", clicked if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            ok = add_wp_unique(nm, clicked["lat"], clicked["lng"], alt)
            (st.success if ok else st.warning)("Adicionado." if ok else "Já existia perto com o mesmo nome.")

with tabs[1]:
    st.caption("Formato: `NOME; LAT; LON; ALT` — aceita DD ou DMS compactas (`390712N 0083155W`). ALT opcional.")
    txt = st.text_area("Lista", height=120,
                       placeholder="ABRANTES; 39.4667; -8.2; 3000\nPONTO X; 390712N; 0083155W; 2500")
    alt_def = st.number_input("Alt (ft) se faltar", 0.0, 18000.0, float(st.session_state.alt_new), step=100.0, key="alt_def_paste")
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

# ========= EDITOR WPs =========
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

# ========= TOC/TOD =========
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
        if B["alt"] > A["alt"] and dist>0:
            dh = B["alt"] - A["alt"]; t_need = dh / max(roc_fpm, 1); d_need = gs_cl * (t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"] and dist>0:
            rod_fpm = max(100.0, gs_de * 5.0 * (desc_angle_deg/3.0))
            dh = A["alt"] - B["alt"]; t_need = dh / max(rod_fpm, 1); d_need = gs_de * (t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
    nodes.append(user_wps[-1])
    return nodes

# ========= LEGS =========
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
                     "Dist":dist,"time_sec":time_sec,"
