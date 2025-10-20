# app.py — NAVLOG VFR (Folium) — TAS fixas + TOC/TOD como WPs + impressão
# Motor Leaflet/Folium, várias bases, botão Imprimir/Exportar, riscas 2 min por GS,
# dog houses, MH grande, seleção de WPs bonita. Corrigido: mapa renderiza mesmo
# que route_nodes esteja vazio (usa nós derivados das legs).

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt
from streamlit_folium import st_folium
from math import sin, asin, radians, degrees
from jinja2 import Template
from folium import MacroElement

# -------------------- CONSTANTES --------------------
CLIMB_TAS   = 70.0
CRUISE_TAS  = 90.0
DESCENT_TAS = 90.0
FUEL_FLOW   = 20.0  # L/h
EARTH_NM    = 3440.065

# -------------------- PAGE / STYLE --------------------
st.set_page_config(page_title="NAVLOG — VFR (Folium) + TOC/TOD", layout="wide", initial_sidebar_state="collapsed")
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

# -------------------- HELPERS --------------------
rt10  = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss  = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
hhmmss= lambda t: f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
rint  = lambda x: int(round(float(x)))
r10f  = lambda x: round(float(x), 1)
rang  = lambda x: int(round(float(x))) % 360
wrap360 = lambda x: (x % 360 + 360) % 360
def angdiff(a,b): return (a - b + 180) % 360 - 180
def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * sin(d)
    s = max(-1, min(1, cross / max(tas,1e-9)))
    wca = degrees(math.asin(s)); th = wrap360(tc + wca)
    gs  = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs
def apply_var(th, var, east_is_neg=False): return wrap360(th - var if east_is_neg else th + var)

def gc_dist_nm(lat1,lon1,lat2,lon2):
    φ1, λ1, φ2, λ2 = map(math.radians,[lat1,lon1,lat2,lon2])
    dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_NM * c
def gc_course_tc(lat1,lon1,lat2,lon2):
    φ1, λ1, φ2, λ2 = map(math.radians,[lat1,lon1,lat2,lon2])
    dλ = λ2-λ1
    y = math.sin(dλ)*math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    θ = math.degrees(math.atan2(y,x))
    return (θ+360)%360
def dest_point(lat,lon,bearing_deg,dist_nm):
    θ = math.radians(bearing_deg); δ = dist_nm/EARTH_NM
    φ1, λ1 = math.radians(lat), math.radians(lon)
    sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
    φ2 = math.asin(sinφ2)
    y = math.sin(θ)*math.sin(δ)*math.cos(φ1); x = math.cos(δ) - math.sin(φ1)*sinφ2
    λ2 = λ1 + math.atan2(y,x)
    return math.degrees(φ2), ((math.degrees(λ2)+540)%360)-180
def point_along_gc(lat1,lon1,lat2,lon2,dist_nm):
    tc = gc_course_tc(lat1,lon1,lat2,lon2)
    return dest_point(lat1,lon1,tc,dist_nm)
def triangle_coords(lat,lon,hdg,h_nm=1.00,w_nm=0.72):
    base_c_lat, base_c_lon = dest_point(lat, lon, hdg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, hdg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, hdg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, hdg+90.0, w_nm/2.0)
    return [(bl_lat, bl_lon), (apex_lat, apex_lon), (br_lat, br_lon)]

# -------------------- STATE --------------------
def ens(k,v): return st.session_state.setdefault(k,v)
ens("wind_from",0); ens("wind_kt",0)
ens("mag_var",1.0); ens("mag_is_e",False)
ens("roc_fpm",600); ens("desc_angle",3.0)
ens("start_clock",""); ens("start_efob",85.0)
ens("ck_default",2)
ens("wps",[]); ens("legs",[]); ens("route_nodes",[])
ens("map_base","OpenTopoMap (topo VFR-ish)"); ens("maptiler_key","")

# -------------------- HEADER --------------------
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns([3,3,2,2])
with c1: st.title("NAVLOG — VFR (Folium) + TOC/TOD")
with c2: st.caption("TAS 70/90/90 · FF 20 L/h · impressão integrada")
with c3:
    if st.button("➕ Novo waypoint", use_container_width=True):
        st.session_state.wps.append({"name":f"WP{len(st.session_state.wps)+1}","lat":39.5,"lon":-8.0,"alt":3000.0})
with c4:
    if st.button("🗑️ Limpar rota", use_container_width=True):
        st.session_state.wps=[]; st.session_state.legs=[]; st.session_state.route_nodes=[]
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- PARÂMETROS --------------------
with st.form("globals"):
    g1,g2,g3,g4 = st.columns(4)
    with g1:
        st.session_state.wind_from = st.number_input("Vento FROM (°T)",0,360,int(st.session_state.wind_from))
        st.session_state.wind_kt   = st.number_input("Vento (kt)",0,150,int(st.session_state.wind_kt))
    with g2:
        st.session_state.mag_var  = st.number_input("Variação magnética (±°)",-30.0,30.0,float(st.session_state.mag_var))
        st.session_state.mag_is_e = st.toggle("Var. é EAST (subtrai)", value=st.session_state.mag_is_e)
    with g3:
        st.session_state.roc_fpm   = st.number_input("ROC global (ft/min)",200,1500,int(st.session_state.roc_fpm),step=10)
        st.session_state.desc_angle= st.number_input("Ângulo de descida (°)",1.0,6.0,float(st.session_state.desc_angle),step=0.1)
    with g4:
        st.session_state.start_efob   = st.number_input("EFOB inicial (L)",0.0,200.0,float(st.session_state.start_efob),step=0.5)
        st.session_state.start_clock  = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default   = st.number_input("CP por defeito (min)",1,10,int(st.session_state.ck_default))
    b1,b2 = st.columns([2,2])
    with b1:
        st.session_state.map_base = st.selectbox(
            "Base do mapa",
            ["OpenTopoMap (topo VFR-ish)","EOX Sentinel-2 (satélite)","ESRI World Imagery (satélite)",
             "ESRI WorldTopoMap (topo)","Carto Positron (clean)","OSM Standard","MapTiler Satellite Hybrid (requer key)"],
            index=["OpenTopoMap (topo VFR-ish)","EOX Sentinel-2 (satélite)","ESRI World Imagery (satélite)",
                   "ESRI WorldTopoMap (topo)","Carto Positron (clean)","OSM Standard","MapTiler Satellite Hybrid (requer key)"].index(st.session_state.map_base)
        )
    with b2:
        if "MapTiler" in st.session_state.map_base:
            st.session_state.maptiler_key = st.text_input("MapTiler API key (opcional)", st.session_state.maptiler_key)
    st.form_submit_button("Aplicar")
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# -------------------- LER CSVs --------------------
AD_CSV  = "AD-HEL-ULM.csv"
LOC_CSV = "Localidades-Nova-versao-230223.csv"

def dms_to_dd(token,is_lon=False):
    token=str(token).strip()
    m=re.match(r"^(\d+(?:\.\d+)?)([NSEW])$",token,re.I)
    if not m: return None
    value,hemi=m.groups()
    if "." in value:
        deg = int(value[0:3] if is_lon else value[0:2]); minutes=int(value[3:5] if is_lon else value[2:4]); seconds=float(value[5:])
    else:
        deg = int(value[0:3] if is_lon else value[0:2]); minutes=int(value[3:5] if is_lon else value[2:4]); seconds=int(value[5:])
    dd = deg + minutes/60 + seconds/3600
    return -dd if hemi.upper() in ["S","W"] else dd

def parse_ad_df(df):
    rows=[]
    for line in df.iloc[:,0].dropna().tolist():
        s=str(line).strip()
        if not s or s.startswith(("Ident","DEP/")): continue
        toks=s.split(); coords=[t for t in toks if re.match(r"^\d+(?:\.\d+)?[NSEW]$",t)]
        if len(coords)>=2:
            lat_tok, lon_tok = coords[-2], coords[-1]
            lat,lon=dms_to_dd(lat_tok), dms_to_dd(lon_tok,True)
            ident=toks[0] if re.match(r"^[A-Z0-9]{4,}$",toks[0]) else None
            try: name=" ".join(toks[1:toks.index(coords[0])]).strip()
            except: name=" ".join(toks[1:]).strip()
            rows.append({"tipo":"AD","code":ident or name,"name":name,"lat":lat,"lon":lon})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

def parse_loc_df(df):
    rows=[]
    for line in df.iloc[:,0].dropna().tolist():
        s=str(line).strip()
        if not s or "Total de registos" in s: continue
        toks=s.split(); coords=[t for t in toks if re.match(r"^\d{6,7}(?:\.\d+)?[NSEW]$",t)]
        if len(coords)>=2:
            lat_tok, lon_tok = coords[0], coords[1]
            lat,lon=dms_to_dd(lat_tok), dms_to_dd(lon_tok,True)
            try: lon_idx=toks.index(lon_tok)
            except ValueError: continue
            code=toks[lon_idx+1] if lon_idx+1<len(toks) else None
            name=" ".join(toks[:toks.index(lat_tok)]).strip()
            rows.append({"tipo":"LOC","code":code or name,"name":name,"lat":lat,"lon":lon})
    return pd.DataFrame(rows).dropna(subset=["lat","lon"])

try:
    ad_raw=pd.read_csv(AD_CSV); loc_raw=pd.read_csv(LOC_CSV)
    ad_df=parse_ad_df(ad_raw);  loc_df=parse_loc_df(loc_raw)
except Exception:
    ad_df=pd.DataFrame(columns=["tipo","code","name","lat","lon"])
    loc_df=pd.DataFrame(columns=["tipo","code","name","lat","lon"])
    st.warning("Não foi possível ler os CSVs locais. Verifica os nomes de ficheiro.")

# -------------------- PESQUISA + SELEÇÃO (tabela bonita) --------------------
st.subheader("Adicionar waypoints por pesquisa")
col1,col2=st.columns([3,1.2])
with col1: qtxt=st.text_input("🔎 Procurar (AD/Localidades)", "", placeholder="Ex: LPPT, ABRANTES, NISA…")
with col2: alt_wp=st.number_input("Altitude p/ WPs (ft)", 0.0, 18000.0, 3000.0, step=100.0)

results=pd.concat([ad_df,loc_df], ignore_index=True)
if qtxt.strip():
    tq=qtxt.lower().strip()
    results=results[results.apply(lambda r: any(tq in str(v).lower() for v in [r.get("code",""), r.get("name","")]), axis=1)]

if not results.empty:
    table=results[["tipo","code","name","lat","lon"]].rename(columns={"tipo":"Tipo","code":"Código","name":"Nome"})
    table.insert(0,"Adicionar",False)
    edited=st.data_editor(table, hide_index=True, use_container_width=True,
                          column_config={"Adicionar": st.column_config.CheckboxColumn(),
                                         "lat": st.column_config.NumberColumn("lat", disabled=True),
                                         "lon": st.column_config.NumberColumn("lon", disabled=True)})
    cA,cB=st.columns([1.2,1.2])
    with cA:
        if st.button("➕ Adicionar selecionados"):
            sel=edited[edited["Adicionar"]==True]
            for _,r in sel.iterrows():
                st.session_state.wps.append({"name":str(r["Código"] or r["Nome"]), "lat":float(r["lat"]), "lon":float(r["lon"]), "alt":float(alt_wp)})
            st.success(f"Adicionados {len(sel)} WPs.")
    with cB:
        if st.button("👀 Pré-visualizar"):
            sel=edited[edited["Adicionar"]==True]
            if not sel.empty:
                m=folium.Map(location=[sel["lat"].mean(), sel["lon"].mean()], zoom_start=9)
                for _,r in sel.iterrows(): folium.Marker((r["lat"],r["lon"]), tooltip=f"{r['Código'] or r['Nome']}").add_to(m)
                st_folium(m, height=300, width=None)
            else:
                st.info("Nada marcado para pré-visualizar.")
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# -------------------- EDITOR DE WPs --------------------
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i,w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            c1,c2,c3,c4,c5=st.columns([2,2,2,1,1])
            with c1: name=st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with c2: lat =st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c3: lon =st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with c4: alt =st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with c5:
                up=st.button("↑", key=f"up{i}"); dn=st.button("↓", key=f"dn{i}")
                if up and i>0: st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]
                if dn and i<len(st.session_state.wps)-1: st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]
            if (name,lat,lon,alt)!=(w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i]={"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"): st.session_state.wps.pop(i)
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# -------------------- TOC/TOD COMO WPs --------------------
def build_route_nodes(user_wps, wfrom, wkt, roc_fpm, desc_angle_deg):
    nodes=[]; 
    if len(user_wps)<2: return nodes
    for i in range(len(user_wps)-1):
        A,B=user_wps[i],user_wps[i+1]; nodes.append(A)
        tc=gc_course_tc(A["lat"],A["lon"],B["lat"],B["lon"]); dist=gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
        _,_,gs_cl=wind_triangle(tc, CLIMB_TAS, wfrom, wkt)
        _,_,gs_de=wind_triangle(tc, DESCENT_TAS, wfrom, wkt)
        if B["alt"]>A["alt"]:
            dh=B["alt"]-A["alt"]; t_need=dh/max(roc_fpm,1); d_need=gs_cl*(t_need/60.0)
            if d_need < dist-0.05:
                lt,ln=point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], d_need)
                nodes.append({"name":f"TOC L{i+1}","lat":lt,"lon":ln,"alt":B["alt"]})
        elif B["alt"]<A["alt"]:
            rod=max(100.0, gs_de*5.0*(desc_angle_deg/3.0))
            dh=A["alt"]-B["alt"]; t_need=dh/max(rod,1); d_need=gs_de*(t_need/60.0)
            if d_need < dist-0.05:
                pos=max(0.0, dist-d_need); lt,ln=point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"], pos)
                nodes.append({"name":f"TOD L{i+1}","lat":lt,"lon":ln,"alt":A["alt"]})
    nodes.append(user_wps[-1]); return nodes

# -------------------- LEGS A PARTIR DOS NODES --------------------
def build_legs_from_nodes(nodes, wfrom, wkt, mag_var, mag_is_e, ck_every_min):
    legs=[]; 
    if len(nodes)<2: return legs
    base_time=None
    if st.session_state.start_clock.strip():
        try: h,m=map(int,st.session_state.start_clock.split(":")); base_time=dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time=None
    carry=float(st.session_state.start_efob); tcur=0
    for i in range(len(nodes)-1):
        A,B=nodes[i],nodes[i+1]
        tc=gc_course_tc(A["lat"],A["lon"],B["lat"],B["lon"]); dist=gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
        prof="LEVEL" if abs(B["alt"]-A["alt"])<1e-6 else ("CLIMB" if B["alt"]>A["alt"] else "DESCENT")
        tas=CLIMB_TAS if prof=="CLIMB" else (DESCENT_TAS if prof=="DESCENT" else CRUISE_TAS)
        _,th,gs=wind_triangle(tc,tas,wfrom,wkt); mh=apply_var(th,mag_var,mag_is_e)
        time_sec=rt10((dist/max(gs,1e-9))*3600.0) if gs>0 else 0; burn=FUEL_FLOW*(time_sec/3600.0)
        efob_start=carry; efob_end=max(0.0, r10f(efob_start - burn))
        clk_start=(base_time+dt.timedelta(seconds=tcur)).strftime('%H:%M') if base_time else f"T+{mmss(tcur)}"
        clk_end  =(base_time+dt.timedelta(seconds=tcur+time_sec)).strftime('%H:%M") if base_time else f"T+{mmss(tcur+time_sec)}"
        cps=[]
        if ck_every_min>0 and gs>0:
            k=1
            while k*ck_every_min*60 <= time_sec:
                t=k*ck_every_min*60; d=gs*(t/3600.0)
                eto=(base_time+dt.timedelta(seconds=tcur+t)).strftime('%H:%M') if base_time else ""
                efob=max(0.0, r10f(efob_start - FUEL_FLOW*(t/3600.0)))
                cps.append({"t":t,"min":int(t/60),"nm":round(d,1),"eto":eto,"efob":efob}); k+=1
        legs.append({"i":i+1,"A":A,"B":B,"profile":prof,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
                     "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),"efob_start":efob_start,"efob_end":efob_end,
                     "clock_start":clk_start,"clock_end":clk_end,"cps":cps})
        tcur+=time_sec; carry=efob_end
    return legs

# -------------------- GERAR --------------------
cc1,cc2=st.columns([2,6])
with cc1:
    if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
        st.session_state.route_nodes = build_route_nodes(st.session_state.wps,
            st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.roc_fpm, st.session_state.desc_angle)
        st.session_state.legs = build_legs_from_nodes(st.session_state.route_nodes,
            st.session_state.wind_from, st.session_state.wind_kt,
            st.session_state.mag_var, st.session_state.mag_is_e,
            st.session_state.ck_default)

# -------------------- RESUMO --------------------
if st.session_state.legs:
    total_sec=sum(L["time_sec"] for L in st.session_state.legs)
    total_burn=r10f(sum(L["burn"] for L in st.session_state.legs))
    efob_final=st.session_state.legs[-1]["efob_end"]
    st.markdown(f"<div class='kvrow'><div class='kv'>⏱️ ETE Total: <b>{hhmmss(total_sec)}</b></div>"
                f"<div class='kv'>⛽ Burn Total: <b>{total_burn:.1f} L</b></div>"
                f"<div class='kv'>🧯 EFOB Final: <b>{efob_final:.1f} L</b></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# -------------------- IMPRESSÃO --------------------
class BrowserPrintPlugin(MacroElement):
    _template = Template("""
        {% macro header(this, kwargs) %}
            <link rel="stylesheet" href="https://unpkg.com/leaflet.browser.print/dist/leaflet.browser.print.css"/>
        {% endmacro %}
        {% macro script(this, kwargs) %}
            L.control.browserPrint({
                title:'Imprimir/Exportar', position:'topleft',
                printModes:[L.BrowserPrint.Mode.Landscape(), L.BrowserPrint.Mode.Portrait(), 'Auto', 'Custom']
            }).addTo({{this._parent.get_name()}});
        {% endmacro %}
    """)
def enable_print(m):
    folium.Element('<script src="https://unpkg.com/leaflet.browser.print/dist/leaflet.browser.print.min.js"></script>').add_to(m)
    m.add_child(BrowserPrintPlugin())

# -------------------- MAPA --------------------
def _bounds_from_nodes(nodes):
    lats=[n["lat"] for n in nodes]; lons=[n["lon"] for n in nodes]
    return [(min(lats),min(lons)),(max(lats),max(lons))]
def _add_text(m,lat,lon,text,size_px=22,color="#FFD700",offset_px=(0,0),bold=True,halo=True):
    weight="700" if bold else "400"
    shadow="text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;" if halo else ""
    html=f"<div style='font-size:{size_px}px;color:{color};font-weight:{weight};{shadow};white-space:nowrap;transform:translate({offset_px[0]}px,{offset_px[1]}px);'>{text}</div>"
    folium.Marker((lat,lon), icon=folium.DivIcon(html=html, icon_size=(0,0), icon_anchor=(0,0))).

