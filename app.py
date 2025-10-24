# app.py — NAVLOG — rev28
# - Fluxo lógico corrigido: Globais → Adicionar WPs → Gerar Rota + Mapa → FPL → PDFs
# - Mapa sem auto-zoom (memoriza centro/zoom)
# - PDF preenche com NOMES EXATOS dos campos (NAVLOG_FORM.pdf e NAVLOG_FORM_1.pdf)
# - UI limpa (sem campos supérfluos); callsign padrão "RVP"
# - ROD (ft/min) configurável; Climb Fuel auto; Observações com tempos por perfil
# - "Adicionar no mapa" mostra TODOS os pontos dos CSV
# - Flight Plan (Item 15) em caixa "DCT …"

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib, io, os
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from math import degrees

# ======== TEMPLATES PDF ========
TEMPLATE_MAIN = "NAVLOG_FORM.pdf"     # capa (Leg01–Leg11)
TEMPLATE_CONT = "NAVLOG_FORM_1.pdf"   # continuação (Leg12–Leg23)

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0   # kt
FUEL_FLOW = 20.0                                        # L/h
EARTH_NM  = 3440.065
PROFILE_COLORS = {"CLIMB":"#FF7A00","LEVEL":"#C000FF","DESCENT":"#00B386"}
CP_TICK_HALF_NM = 0.40

# ======== ESTILO ========
st.set_page_config(page_title="NAVLOG", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--line:#e5e7eb;--chip:#f3f4f6}
*{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin:8px 0;background:#fff}
.kvrow{display:flex;gap:8px;flex-wrap:wrap}
.kv{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:6px 8px;font-size:12px}
.sep{height:1px;background:var(--line);margin:10px 0}
.small{font-size:12px;color:#555}
.badge{font-weight:700;border:1px solid #111;border-radius:8px;padding:2px 6px;margin-right:6px}
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
    wca = degrees(math.asin(s))
    th  = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs

def apply_var(th, var, east_is_neg=False):
    return wrap360(th - var if east_is_neg else th + var)

def gc_dist_nm(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2]); dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_NM * c

def gc_course_tc(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2]); dλ = λ2 - λ1
    y = math.sin(dλ)*math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.degrees(math.atan2(y, x)); return (θ + 360) % 360

def dest_point(lat, lon, bearing_deg, dist_nm):
    θ = math.radians(bearing_deg); δ = dist_nm / EARTH_NM
    φ1, λ1 = math.radians(lat), math.radians(lon)
    sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
    φ2 = math.asin(sinφ2); y = math.sin(θ)*math.sin(δ)*math.cos(φ1); x = math.cos(δ) - math.sin(φ1)*sinφ2
    λ2 = λ1 + math.atan2(y, x)
    return math.degrees(φ2), ((math.degrees(λ2)+540)%360)-180

def point_along_gc(lat1, lon1, lat2, lon2, dist_from_start_nm):
    total = gc_dist_nm(lat1, lon1, lat2, lon2)
    if total <= 0: return lat1, lon1
    tc0 = gc_course_tc(lat1, lon1, lat2, lon2)
    return dest_point(lat1, lon1, tc0, dist_from_start_nm)

# ======== STATE ========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 0); ens("wind_kt", 0)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600); ens("rod_fpm", 600)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)
ens("text_scale", 1.0)
ens("map_center", (39.7, -8.1)); ens("map_zoom", 8)
ens("wps", []); ens("route_nodes", []); ens("legs", [])
ens("db_points", None); ens("alt_qadd", 3000.0)

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
        st.session_state.roc_fpm   = st.number_input("ROC (ft/min)", 200, 1500, int(st.session_state.roc_fpm), step=10)
        st.session_state.rod_fpm   = st.number_input("ROD (ft/min)", 200, 1500, int(st.session_state.rod_fpm), step=10)
    with c4:
        st.session_state.start_efob= st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP (min)", 1, 10, int(st.session_state.ck_default))
        st.session_state.text_scale  = st.slider("Escala texto mapa", 0.5, 1.5, float(st.session_state.text_scale), 0.05)
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
    rows=[]
    for line in df.iloc[:,0].dropna().tolist():
        s=str(line).strip()
        if not s or s.startswith(("Ident","DEP/")): continue
        tokens=s.split()
        coord=[t for t in tokens if re.match(r"^\d+(?:\.\d+)?[NSEW]$", t)]
        if len(coord)>=2:
            lat_tok, lon_tok = coord[-2], coord[-1]
            lat = dms_to_dd(lat_tok, False); lon = dms_to_dd(lon_tok, True)
            ident = tokens[0] if re.match(r"^[A-Z0-9]{4,}$", tokens[0]) else None
            try: name=" ".join(tokens[1:tokens.index(coord[0])]).strip()
            except: name=" ".join(tokens[1:]).strip()
            rows.append({"src":"AD","code":ident or name, "name":name, "lat":lat, "lon":lon, "alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

def parse_loc_df(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for line in df.iloc[:,0].dropna().tolist():
        s=str(line).strip()
        if not s or "Total de registos" in s: continue
        tokens=s.split()
        coord=[t for t in tokens if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$", t)]
        if len(coord)>=2:
            lat_tok, lon_tok = coord[0], coord[1]
            lat = dms_to_dd(lat_tok, False); lon = dms_to_dd(lon_tok, True)
            try: lon_idx = tokens.index(lon_tok)
            except ValueError: continue
            code = tokens[lon_idx+1] if lon_idx+1 < len(tokens) else None
            name = " ".join(tokens[:tokens.index(lat_tok)]).strip()
            rows.append({"src":"LOC","code":code or name, "name":name, "lat":lat, "lon":lon, "alt":0.0})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

try:
    ad_raw  = pd.read_csv(AD_CSV);  ad_df  = parse_ad_df(ad_raw)
    loc_raw = pd.read_csv(LOC_CSV); loc_df = parse_loc_df(loc_raw)
except Exception:
    ad_df = pd.DataFrame(columns=["src","code","name","lat","lon","alt"])
    loc_df= pd.DataFrame(columns=["src","code","name","lat","lon","alt"])
    st.warning("Falha a ler CSVs locais.")

if st.session_state.db_points is None:
    st.session_state.db_points = pd.concat([ad_df, loc_df]).dropna(subset=["lat","lon"]).reset_index(drop=True)
db = st.session_state.db_points

def make_unique_name(name: str) -> str:
    names=[str(w["name"]) for w in st.session_state.wps]
    if name not in names: return name
    k=2
    while f"{name} #{k}" in names: k+=1
    return f"{name} #{k}"

def append_wp(name, lat, lon, alt):
    st.session_state.wps.append({"name":make_unique_name(str(name)), "lat":float(lat), "lon":float(lon), "alt":float(alt)})

# ======== ABAS: CSV, MAPA, FPL, PDFs ========
tab_csv, tab_map, tab_fpl, tab_pdf = st.tabs(["🔎 Pesquisar CSV", "🗺️ Adicionar no mapa", "✈️ Flight Plan", "🧾 Formulários"])

with tab_csv:
    c1,c2 = st.columns([3,1])
    with c1:
        q = st.text_input("Pesquisar (1 termo) e usar ➕", placeholder="Ex: LPSO, VACOR, ÉVORA…").strip()
    with c2:
        st.session_state.alt_qadd = st.number_input("Alt (ft) p/ novos WPs", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
    def _score_row(row, tq, last_wp):
        code=str(row.get("code") or "").lower(); name=str(row.get("name") or "").lower()
        sim = difflib.SequenceMatcher(None, tq, f"{code} {name}").ratio()
        starts = 1.0 if code.startswith(tq) or name.startswith(tq) else 0.0
        near = 0.0
        if last_wp: near = 1.0/(1.0+gc_dist_nm(last_wp["lat"], last_wp["lon"], row["lat"], row["lon"]))
        return starts*2 + sim + 0.25*near
    def _search_points(tq):
        if not tq: return db.head(0)
        tql=tq.lower().strip(); last=st.session_state.wps[-1] if st.session_state.wps else None
        df=db[db.apply(lambda r: any(tql in str(v).lower() for v in r.values), axis=1)].copy()
        if df.empty: return df
        df["__score"]=df.apply(lambda r:_score_row(r,tql,last),axis=1); return df.sort_values("__score",ascending=False)
    res=_search_points(q)
    if not res.empty:
        st.caption("Resultados")
        for i, r in res.head(30).iterrows():
            code=r.get("code") or ""; name=r.get("name") or ""; lat,lon=float(r["lat"]),float(r["lon"])
            col1,col2=st.columns([0.84,0.16])
            with col1:
                st.markdown(f"<div class='card'><div><span class='badge'>[{r['src']}]</span><b>{code} — {name}</b></div><div class='small'>({lat:.4f}, {lon:.4f})</div></div>", unsafe_allow_html=True)
            with col2:
                if st.button("➕", key=f"add_{i}", use_container_width=True):
                    append_wp(code or name, lat, lon, float(st.session_state.alt_qadd)); st.success("Adicionado.")
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    multi = st.text_input("Adicionar vários (separa por espaços — ex.: LPSO VACOR VARGE)", key="multi_terms")
    if st.button("➕ Adicionar todos os termos"):
        added,misses=[],[]
        for t in [x for x in re.split(r"\s+", multi.strip()) if x]:
            df=_search_points(t)
            if df.empty: misses.append(t); continue
            r=df.iloc[0]; append_wp(r.get("code") or r.get("name"), float(r["lat"]), float(r["lon"]), float(st.session_state.alt_qadd))
            added.append(r.get("code") or r.get("name"))
        if added: st.success("Adicionados: " + ", ".join(added))
        if misses: st.warning("Sem match: " + ", ".join(misses))

with tab_map:
    st.caption("Todos os pontos dos CSV aparecem em cinzento; os teus WPs em azul. Clica para criar novo WP.")
    m = folium.Map(location=list(st.session_state.map_center), zoom_start=st.session_state.map_zoom,
                   tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="© OpenTopoMap", control_scale=True)
    cl = MarkerCluster().add_to(m)
    for _, r in db.iterrows():
        folium.CircleMarker((float(r["lat"]),float(r["lon"])), radius=3, color="#333", fill=True, fill_opacity=0.9,
                            tooltip=f"{r.get('code') or ''} — {r.get('name') or ''}").add_to(cl)
    for w in st.session_state.wps:
        folium.CircleMarker((w["lat"],w["lon"]), radius=5, color="#007AFF", fill=True, fill_opacity=1, tooltip=w["name"]).add_to(m)
    map_out = st_folium(m, width=None, height=520, key="map_pick")
    if map_out and map_out.get("center") and map_out.get("zoom"):
        st.session_state.map_center = (map_out["center"]["lat"], map_out["center"]["lng"])
        st.session_state.map_zoom   = map_out["zoom"]
    with st.form("add_click"):
        c1,c2,c3 = st.columns([2,1,1])
        with c1: nm = st.text_input("Nome", "WP novo")
        with c2: alt = st.number_input("Alt (ft)", 0.0, 18000.0, float(st.session_state.alt_qadd), step=100.0)
        clicked = map_out.get("last_clicked") if map_out else None
        with c3:
            st.write("Último clique:", f"{clicked['lat']:.5f}, {clicked['lng']:.5f}" if clicked else "—")
        if st.form_submit_button("Adicionar do clique") and clicked:
            append_wp(nm, float(clicked["lat"]), float(clicked["lng"]), float(alt))
            st.success("Adicionado.")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== EDITOR WPs ========
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    remove_idx = None
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}"):
            c1,c2,c3,c4 = st.columns([2,2,2,1])
            with c1: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            if (name,lat,lon,alt) != (w["name"],w["lat"],w["lon"]):
                st.session_state.wps[i]={"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"del_{i}"): remove_idx = i
    if remove_idx is not None: st.session_state.wps.pop(remove_idx)
else:
    st.info("Adiciona pelo menos 2 waypoints.")

# ======== TOC/TOD → nós ========
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, rod_fpm):
    nodes=[]
    if len(user_wps) < 2: return nodes
    for i in range(len(user_wps)-1):
        A,B=user_wps[i], user_wps[i+1]
        nodes.append(A)
        tc   = gc_course_tc(A["lat"],A["lon"],B["lat"],B["lon"])
        dist = gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
        _,_,gs_cl = wind_triangle(tc, CLIMB_TAS,   wind_from, wind_kt)
        _,_,gs_de = wind_triangle(tc, DESCENT_TAS, wind_from, wind_kt)
        if B["alt"] > A["alt"]:
            dh=B["alt"]-A["alt"]; t_need=dh/max(roc_fpm,1); d_need=gs_cl*(t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc,lon_toc = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], d_need)
                nodes.append({"name":f"TOC L{i+1}","lat":lat_toc,"lon":lon_toc,"alt":B["alt"]})
        elif B["alt"] < A["alt"]:
            dh=A["alt"]-B["alt"]; t_need=dh/max(rod_fpm,1); d_need=gs_de*(t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod,lon_tod = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], pos_from_start)
                nodes.append({"name":f"TOD L{i+1}","lat":lat_tod,"lon":lon_tod,"alt":A["alt"]})
    nodes.append(user_wps[-1]); return nodes

# ======== LEGS ========
def build_legs_from_nodes(nodes, wind_from, wind_kt, mag_var, mag_is_e, ck_every_min):
    legs=[]
    if len(nodes)<2: return legs
    base_time=None
    if st.session_state.start_clock.strip():
        try:
            h,m = map(int, st.session_state.start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time=None
    carry_efob=float(st.session_state.start_efob); t_cursor=0; cum_d=0.0; cum_t=0
    for i in range(len(nodes)-1):
        A,B=nodes[i],nodes[i+1]
        tc   = gc_course_tc(A["lat"],A["lon"],B["lat"],B["lon"])
        dist = gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
        profile = "LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas = CLIMB_TAS if profile=="CLIMB" else (DESCENT_TAS if profile=="DESCENT" else CRUISE_TAS)
        _, th, gs = wind_triangle(tc, tas, wind_from, wind_kt); mh = apply_var(th, st.session_state.mag_var, st.session_state.mag_is_e)
        time_sec = rt10((dist/max(gs,1e-9))*3600.0) if gs>0 else 0
        burn = FUEL_FLOW*(time_sec/3600.0)
        efob_start=carry_efob; efob_end=max(0.0, r10f(efob_start-burn))
        clk_start=(base_time+dt.timedelta(seconds=t_cursor)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor)}"
        clk_end  =(base_time+dt.timedelta(seconds=t_cursor+time_sec)).strftime('%H:%M') if base_time else f"T+{mmss(t_cursor+time_sec)}"
        cps=[]
        if ck_every_min>0 and gs>0:
            k=1
            while k*ck_every_min*60 <= time_sec:
                t=k*ck_every_min*60; d=gs*(t/3600.0)
                cps.append({"t":t,"nm":round(d,1)}); k+=1
        cum_d+=dist; cum_t+=time_sec; t_cursor+=time_sec; carry_efob=efob_end
        legs.append({"i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
                     "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),
                     "efob_start":r10f(efob_start),"efob_end":r10f(efob_end),
                     "clock_start":clk_start,"clock_end":clk_end,"cps":cps,
                     "cum_dist":r10f(cum_d),"cum_time":cum_t})
    return legs

# ======== BOTÃO: GERAR ROTA ========
cgen,_ = st.columns([2,6])
with cgen:
    if st.button("Gerar / Atualizar rota ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(st.session_state.wps, st.session_state.wind_from, st.session_state.wind_kt, st.session_state.roc_fpm, st.session_state.rod_fpm)
        st.session_state.legs = build_legs_from_nodes(st.session_state.route_nodes, st.session_state.wind_from, st.session_state.wind_kt, st.session_state.mag_var, st.session_state.mag_is_e, st.session_state.ck_default)

# ======== MAPA ========
def render_map(nodes, legs):
    if not nodes or not legs: 
        st.info("Gera a rota para ver o mapa."); return
    m = folium.Map(location=list(st.session_state.map_center), zoom_start=st.session_state.map_zoom,
                   tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="© OpenTopoMap", control_scale=True)
    # Pernas (halo branco + cor por perfil)
    for L in legs:
        latlngs = [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])]
        color = PROFILE_COLORS.get(L["profile"], "#C000FF")
        folium.PolyLine(latlngs, color="#FFFFFF", weight=10, opacity=1).add_to(m)
        folium.PolyLine(latlngs, color=color, weight=4, opacity=1).add_to(m)
        # ticks CP
        if L["GS"]>0 and L["cps"]:
            for cp in L["cps"]:
                d=min(L["Dist"], L["GS"]*(cp["t"]/3600.0))
                latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
                llat,llon = dest_point(latm,lonm, L["TC"]-90, CP_TICK_HALF_NM)
                rlat,rlon = dest_point(latm,lonm, L["TC"]+90, CP_TICK_HALF_NM)
                folium.PolyLine([(llat,llon),(rlat,rlon)], color="#000", weight=3, opacity=1).add_to(m)
    # WPs com etiqueta compacta (nome + EFOB na linha abaixo + hora)
    def wp_label(nm, liters, clk):
        fs = int(14*float(st.session_state.text_scale))
        sub = int(max(10, fs-3))
        return f"""<div style="transform:translate(-50%,-50%);font-weight:900;color:#111;text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;text-align:left;line-height:1.05">
        <div style="font-size:{fs}px">{nm}</div>
        <div style="font-size:{sub}px">{liters:.1f} L<br><span style="font-weight:700">{clk}</span></div></div>"""
    info=[{"eto":legs[0]["clock_start"],"efob":legs[0]["efob_start"]}] if legs else []
    for i in range(1,len(nodes)):
        info.append({"eto":legs[i-1]["clock_end"],"efob":legs[i-1]["efob_end"]})
    for idx,N in enumerate(nodes):
        folium.CircleMarker((N["lat"],N["lon"]), radius=6, color="#FFF", weight=2, fill=True, fill_color="#1f78ff", fill_opacity=1).add_to(m)
        ef=info[idx]["efob"] if idx<len(info) else None; tm=info[idx]["eto"] if idx<len(info) else "-"
        html = wp_label(str(N["name"]).upper(), ef if ef is not None else 0.0, tm)
        folium.Marker((N["lat"],N["lon"]), icon=folium.DivIcon(html=html, icon_size=(0,0))).add_to(m)
    out = st_folium(m, width=None, height=560, key="route_map")
    if out and out.get("center") and out.get("zoom"):
        st.session_state.map_center=(out["center"]["lat"], out["center"]["lng"])
        st.session_state.map_zoom=out["zoom"]

if st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs)

# ======== RESUMO ========
if st.session_state.legs:
    total_sec  = sum(L["time_sec"] for L in st.session_state.legs)
    total_burn = r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final = st.session_state.legs[-1]["efob_end"]
    t_climb   = sum(L["time_sec"] for L in st.session_state.legs if L["profile"]=="CLIMB")
    t_level   = sum(L["time_sec"] for L in st.session_state.legs if L["profile"]=="LEVEL")
    t_descent = sum(L["time_sec"] for L in st.session_state.legs if L["profile"]=="DESCENT")
    st.markdown(
        "<div class='kvrow'>"
        + f"<div class='kv'>⏱️ Total: <b>{hhmmss(total_sec)}</b></div>"
        + f"<div class='kv'>⛽ Burn: <b>{total_burn:.1f} L</b></div>"
        + f"<div class='kv'>🧯 EFOB final: <b>{efob_final:.1f} L</b></div>"
        + f"<div class='kv'>CLIMB <b>{mmss(t_climb)}</b> • CRUISE <b>{mmss(t_level)}</b> • DESC <b>{mmss(t_descent)}</b></div>"
        + "</div>", unsafe_allow_html=True
    )

# ================== ✈️ Flight Plan (Item 15) ==================
with tab_fpl:
    st.subheader("Flight Plan — Rota (Item 15)")
    def is_icao_aerodrome(nm:str)->bool: return bool(re.fullmatch(r"[A-Z]{4}", nm))
    def is_pub_pt(nm:str)->bool: return bool(re.fullmatch(r"[A-Z0-9]{3,5}", nm))
    def icao_latlon(lat,lon)->str:
        la,lo=abs(lat),abs(lon)
        lad, lod = int(la), int(lo)
        lam, lom = int(round((la-lad)*60)), int(round((lo-lod)*60))
        if lam==60: lad+=1; lam=0
        if lom==60: lod+=1; lom=0
        return f"{lad:02d}{lam:02d}{'N' if lat>=0 else 'S'}{lod:03d}{lom:02d}{'E' if lon>=0 else 'W'}"
    def build_route(wps):
        if len(wps)<2: return ""
        seq = wps[:]
        if is_icao_aerodrome(seq[0]['name'].upper()):  seq=seq[1:]
        if seq and is_icao_aerodrome(seq[-1]['name'].upper()): seq=seq[:-1]
        toks=[]
        for w in seq:
            nm=re.sub(r"\s+#\d+$","",str(w['name']).upper())
            toks.append(nm if is_pub_pt(nm) else icao_latlon(w['lat'], w['lon']))
        return ("DCT " + " DCT ".join(toks)) if toks else ""
    route_str = build_route(st.session_state.wps)
    st.code(route_str or "—", language=None)

# ================== 🧾 FORMULÁRIOS (PDF) ==================
with tab_pdf:
    st.subheader("Gerar NAVLOG (PDF)")
    registrations = ["CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW","CS-ECC","CS-ECD"]
    with st.form("pdf_hdr"):
        c1,c2,c3 = st.columns(3)
        with c1:
            aircraft     = st.text_input("Aircraft", "P208")
            registration = st.selectbox("Registration", registrations, index=0)
            callsign     = st.text_input("Callsign", "RVP")
        with c2:
            etd_eta      = st.text_input("ETD/ETA", "")
            qnh          = st.text_input("QNH", "")
            lesson       = st.text_input("Lesson", "")
        with c3:
            instrutor    = st.text_input("Instrutor", "")
            student      = st.text_input("Student", "")
            mag_var_f    = st.text_input("MAG VAR", f"{'-' if st.session_state.mag_is_e else '+'}{st.session_state.mag_var:.1f}°")
        c4,c5 = st.columns(2)
        with c4:
            dep_airf     = st.text_input("Departure Airfield", st.session_state.wps[0]["name"] if st.session_state.wps else "LPSO")
            arr_airf     = st.text_input("Arrival Airfield",  st.session_state.wps[-1]["name"] if st.session_state.wps else "LPSO")
            dep_freq     = st.text_input("DEP freq", "119.805")         # LPSO TWR padrão
        with c5:
            enroute_freq = st.text_input("ENROUTE freq", "123.755")     # Lisboa MIL padrão
            wind_fld     = st.text_input("WIND", f"{int(st.session_state.wind_from)} / {int(st.session_state.wind_kt)}")
            temp_isa     = st.text_input("TEMP/ISA DEV", "")
        st.form_submit_button("Guardar")

    # util PDF
    def fill_pdf_fields_exact(template_path: str, field_values: dict) -> bytes:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(open(template_path, "rb")); writer = PdfWriter()
        for page in reader.pages: writer.add_page(page)
        # aplica valores a TODAS as páginas do PDF (cada template tem no máx. 2)
        for p in range(len(writer.pages)):
            writer.update_page_form_field_values(writer.pages[p], field_values)
        out = io.BytesIO(); writer.write(out); return out.getvalue()

    def climb_fuel_liters(legs): return r10f(sum(L["burn"] for L in legs if L["profile"]=="CLIMB"))

    # ---- Mapeamento exato — NAVLOG_FORM.pdf (capa) ----
    def fields_main(legs):
        # cabeçalhos
        data = {
            "AIRCRAFT": aircraft,
            "REGISTRATION": registration,
            "CALLSIGN": callsign,
            "ETD/ETA": etd_eta,
            "LESSON": lesson,
            "INSTRUTOR": instrutor,
            "STUDENT": student,
            "QNH": qnh,
            "DEPT": dep_freq,
            "ENROUTE": enroute_freq,
            "Departure_Airfield": dep_airf,
            "Arrival_Airfield":  arr_airf,
            "WIND": wind_fld,
            "MAG_VAR": mag_var_f,
            "TEMP/ISA_DEV": temp_isa,
            "CLIMB FUEL": f"{climb_fuel_liters(legs):.1f}",
            "Leg_Number": str(len(legs)),
            # Observações também aparecem na capa no teu PDF
            "OBSERVATIONS": f"CLIMB {mmss(sum(L['time_sec'] for L in legs if L['profile']=='CLIMB'))} — "
                            f"CRUISE {mmss(sum(L['time_sec'] for L in legs if L['profile']=='LEVEL'))} — "
                            f"DESCENT {mmss(sum(L['time_sec'] for L in legs if L['profile']=='DESCENT'))}"
        }
        # Legs 01–11
        upto = min(len(legs), 11)
        for i in range(upto):
            L = legs[i]; idx=f"{i+1:02d}"
            data.update({
                f"Leg{idx}_Waypoint":           str(L["A"]["name"]),
                f"Leg{idx}_Navaid_Identifier":  "",
                f"Leg{idx}_Navaid_Frequency":   "",
                f"Leg{idx}_Altitude_FL":        f"{int(round(L['B']['alt']))}",
                f"Leg{idx}_True_Course":        f"{int(round(L['TC']))}",
                f"Leg{idx}_True_Heading":       f"{int(round(L['TH']))}",
                f"Leg{idx}_Magnetic_Heading":   f"{int(round(L['MH']))}",
                f"Leg{idx}_True_Airspeed":      f"{int(round(L['TAS']))}",
                f"Leg{idx}_Ground_Speed":       f"{int(round(L['GS']))}",
                f"Leg{idx}_Leg_Distance":       f"{L['Dist']:.1f}",
                f"Leg{idx}_Leg_ETE":            mmss(L["time_sec"]),
                f"Leg{idx}_ETO":                L["clock_end"],
                f"Leg{idx}_Planned_Burnoff":    f"{L['burn']:.1f}",
                f"Leg{idx}_Estimated_FOB":      f"{L['efob_end']:.1f}",
                f"Leg{idx}_Cumulative_Distance":f"{L['cum_dist']:.1f}",
                f"Leg{idx}_Cumulative_ETE":     mmss(L["cum_time"]),
            })
        return data, upto

    # ---- Mapeamento exato — NAVLOG_FORM_1.pdf (continuação) ----
    def fields_cont(legs, start_idx):
        data = {
            "FLIGHT_LEVEL_ALTITUDE": "",
            "TEMP_ISA_DEV": temp_isa,
            "Alternate_Airfield": "",
            "Alternate_Elevation": "",
            # Observações também existe no form_1
            "OBSERVATIONS": f"CLIMB {mmss(sum(L['time_sec'] for L in legs if L['profile']=='CLIMB'))} — "
                            f"CRUISE {mmss(sum(L['time_sec'] for L in legs if L['profile']=='LEVEL'))} — "
                            f"DESCENT {mmss(sum(L['time_sec'] for L in legs if L['profile']=='DESCENT'))}"
        }
        filled=0
        for i in range(start_idx, len(legs)):
            field_no = 12 + filled
            if field_no > 23: break
            L = legs[i]; idx=f"{field_no:02d}"
            data.update({
                f"Leg{idx}_Waypoint":           str(L["A"]["name"]),
                f"Leg{idx}_Navaid_Identifier":  "",
                f"Leg{idx}_Navaid_Frequency":   "",
                f"Leg{idx}_Altitude_FL":        f"{int(round(L['B']['alt']))}",
                f"Leg{idx}_True_Course":        f"{int(round(L['TC']))}",
                f"Leg{idx}_True_Heading":       f"{int(round(L['TH']))}",
                f"Leg{idx}_Magnetic_Heading":   f"{int(round(L['MH']))}",
                f"Leg{idx}_True_Airspeed":      f"{int(round(L['TAS']))}",
                f"Leg{idx}_Ground_Speed":       f"{int(round(L['GS']))}",
                f"Leg{idx}_Leg_Distance":       f"{L['Dist']:.1f}",
                f"Leg{idx}_Leg_ETE":            mmss(L["time_sec"]),
                f"Leg{idx}_ETO":                L["clock_end"],
                f"Leg{idx}_Planned_Burnoff":    f"{L['burn']:.1f}",
                f"Leg{idx}_Estimated_FOB":      f"{L['efob_end']:.1f}",
                f"Leg{idx}_Cumulative_Distance":f"{L['cum_dist']:.1f}",
                f"Leg{idx}_Cumulative_ETE":     mmss(L["cum_time"]),
            })
            filled += 1
        return data, filled

    st.markdown("**Templates:** " + ("✅ encontrados" if os.path.exists(TEMPLATE_MAIN) and os.path.exists(TEMPLATE_CONT) else "⚠️ coloca `NAVLOG_FORM.pdf` e `NAVLOG_FORM_1.pdf` ao lado do app."))

    if st.button("Gerar PDFs", type="primary", use_container_width=True):
        if not st.session_state.legs:
            st.warning("Gera primeiro a rota.")
        elif not (os.path.exists(TEMPLATE_MAIN) and os.path.exists(TEMPLATE_CONT)):
            st.error("Templates não encontrados.")
        else:
            data_main, used = fields_main(st.session_state.legs)
            pdf_main = fill_pdf_fields_exact(TEMPLATE_MAIN, data_main)
            st.download_button("⬇️ navlog_main_filled.pdf", data=pdf_main, file_name="navlog_main_filled.pdf", mime="application/pdf")
            if len(st.session_state.legs) > used:
                data_cont, used_cont = fields_cont(st.session_state.legs, used)
                if used_cont>0:
                    pdf_cont = fill_pdf_fields_exact(TEMPLATE_CONT, data_cont)
                    st.download_button("⬇️ navlog_cont_filled.pdf", data=pdf_cont, file_name="navlog_cont_filled.pdf", mime="application/pdf")
                else:
                    st.info("Sem necessidade de continuação (cabe tudo na capa).")
