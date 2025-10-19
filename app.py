# app.py — NAVLOG v12.4 (fixo FF=20 L/h, TOC/TOD visíveis, dog houses triangulares em “slots”, mapa VFR)
# Speeds fixas: 70 (climb), 90 (cruise), 90 (descent). Riscas a cada 2 min. MH destacado.

import streamlit as st
import pydeck as pdk
import math, datetime as dt

# -------------------- CONFIG + CSS --------------------
st.set_page_config(page_title="NAVLOG v12.4 — VFR + Dog Houses", layout="wide")
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

# -------------------- CONSTANTES --------------------
TAS_CLIMB, TAS_CRUISE, TAS_DESC = 70.0, 90.0, 90.0
FF_CONST = 20.0  # 20 L/h em todas as fases
EARTH_NM = 3440.065

# -------------------- HELPERS --------------------
rt10   = lambda s: max(10, int(round(s/10.0)*10)) if s>0 else 0
mmss   = lambda t: f"{t//60:02d}:{t%60:02d}"
hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
rang   = lambda x: int(round(float(x))) % 360
rint   = lambda x: int(round(float(x)))
r10f   = lambda x: round(float(x), 1)
def wrap360(x): x = math.fmod(float(x), 360.0); return x + 360 if x < 0 else x
def angdiff(a, b): return (a - b + 180) % 360 - 180
def wind_triangle(tc, tas, wdir, wkt):
    if tas <= 0: return 0.0, wrap360(tc), 0.0
    d = math.radians(angdiff(wdir, tc))
    cross = wkt * math.sin(d)
    s = max(-1, min(1, cross / max(tas, 1e-9)))
    wca = math.degrees(math.asin(s))
    th = wrap360(tc + wca)
    gs = max(0.0, tas * math.cos(math.radians(wca)) - wkt * math.cos(d))
    return wca, th, gs
def apply_var(th, var, east_is_neg=False): return wrap360(th - var if east_is_neg else th + var)

def gc_dist_nm(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2]); dφ, dλ = φ2-φ1, λ2-λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a)); return EARTH_NM * c
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

# ROC aproximado (AFM simplificado que já tinhas)
ROC_ENR = {
    0:{-25:981,0:835,25:704,50:586}, 2000:{-25:870,0:726,25:597,50:481},
    4000:{-25:759,0:617,25:491,50:377}, 6000:{-25:648,0:509,25:385,50:273},
    8000:{-25:538,0:401,25:279,50:170}, 10000:{-25:428,0:294,25:174,50:66},
    12000:{-25:319,0:187,25:69,50:-37}, 14000:{-25:210,0:80,25:-35,50:-139}
}
def isa_temp(pa): return 15.0 - 2.0*(pa/1000.0)
def press_alt(alt, qnh): return float(alt) + (1013.0 - float(qnh)) * 30.0
def interp1(x, x0, x1, y0, y1):
    if x1 == x0: return y0
    t=(x-x0)/(x1-x0); return y0 + t*(y1-y0)
def clamp(v, lo, hi): return max(lo, min(hi, v))
def roc_interp(pa, temp):
    pas = sorted(ROC_ENR.keys()); pa_c = clamp(pa, pas[0], pas[-1])
    p0 = max([p for p in pas if p <= pa_c]); p1 = min([p for p in pas if p >= pa_c])
    temps = [-25,0,25,50]; t = clamp(temp, temps[0], temps[-1])
    t0,t1 = (-25,0) if t<=0 else ((0,25) if t<=25 else (25,50))
    v00,v01 = ROC_ENR[p0][t0], ROC_ENR[p0][t1]
    v10,v11 = ROC_ENR[p1][t0], ROC_ENR[p1][t1]
    v0 = interp1(t, t0, t1, v00, v01); v1 = interp1(t, t0, t1, v10, v11)
    return max(1.0, interp1(pa_c, p0, p1, v0, v1)*0.90)

# -------------------- STATE --------------------
def ens(k, v): return st.session_state.setdefault(k, v)
ens("qnh", 1013); ens("oat", 15); ens("mag_var", 1); ens("mag_is_e", False)
ens("weight", 650.0); ens("desc_angle", 3.0)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2); ens("wind_from", 0); ens("wind_kt", 0)
ens("wps", []); ens("legs", []); ens("route_points", []); ens("computed_by_leg", [])
ens("map_style", "CARTO Voyager")

# -------------------- HEADER --------------------
st.markdown("<div class='sticky'>", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns([3,3,2,2])
with c1: st.title("NAVLOG — v12.4")
with c2: st.selectbox("🗺️ Estilo do mapa", ["CARTO Voyager","CARTO Positron","CARTO Dark"], key="map_style")
with c3:
    if st.button("➕ Novo waypoint", use_container_width=True):
        st.session_state.wps.append({"name": f"WP{len(st.session_state.wps)+1}", "lat": 39.5, "lon": -8.0, "alt": 3000.0})
with c4:
    if st.button("🗑️ Limpar rota/legs", use_container_width=True):
        st.session_state.wps=[]; st.session_state.legs=[]; st.session_state.route_points=[]; st.session_state.computed_by_leg=[]
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- PARÂMETROS --------------------
with st.form("globals"):
    p1,p2,p3,p4 = st.columns(4)
    with p1:
        st.session_state.qnh = st.number_input("QNH (hPa)", 900, 1050, int(st.session_state.qnh))
        st.session_state.oat = st.number_input("OAT (°C)", -40, 50, int(st.session_state.oat))
    with p2:
        st.session_state.start_efob = st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
    with p3:
        st.session_state.weight = st.number_input("Peso (kg)", 450.0, 700.0, float(st.session_state.weight), step=1.0)
        st.session_state.desc_angle = st.number_input("Ângulo de descida (°)", 1.0, 6.0, float(st.session_state.desc_angle), step=0.1)
    with p4:
        st.session_state.ck_default = st.number_input("Checkpoints (min)", 1, 10, int(st.session_state.ck_default), step=1)
        st.session_state.wind_from = st.number_input("Vento FROM (°T)", 0, 360, int(st.session_state.wind_from), step=1)
        st.session_state.wind_kt   = st.number_input("Vento (kt)", 0, 150, int(st.session_state.wind_kt), step=1)
    st.form_submit_button("Aplicar parâmetros")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# -------------------- EDITOR DE WPs --------------------
if st.session_state.wps:
    st.subheader("Rota (Waypoints)")
    for i, w in enumerate(st.session_state.wps):
        with st.expander(f"WP {i+1} — {w['name']}", expanded=False):
            a,b,c,d,e = st.columns([2,2,2,1,1])
            with a: name = st.text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
            with b: lat  = st.number_input(f"Lat — WP{i+1}", -90.0, 90.0, float(w["lat"]), step=0.0001, key=f"wplat_{i}")
            with c: lon  = st.number_input(f"Lon — WP{i+1}", -180.0, 180.0, float(w["lon"]), step=0.0001, key=f"wplon_{i}")
            with d: alt  = st.number_input(f"Alt (ft) — WP{i+1}", 0.0, 18000.0, float(w["alt"]), step=50.0, key=f"wpalt_{i}")
            with e:
                up=st.button("↑", key=f"up{i}"); dn=st.button("↓", key=f"dn{i}")
                if up and i>0: st.session_state.wps[i-1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i-1]; st.experimental_rerun()
                if dn and i < len(st.session_state.wps)-1: st.session_state.wps[i+1], st.session_state.wps[i] = st.session_state.wps[i], st.session_state.wps[i+1]; st.experimental_rerun()
            if (name,lat,lon,alt)!=(w["name"],w["lat"],w["lon"],w["alt"]):
                st.session_state.wps[i]={"name":name,"lat":float(lat),"lon":float(lon),"alt":float(alt)}
            if st.button("Remover", key=f"delwp_{i}"):
                st.session_state.wps.pop(i); st.experimental_rerun()
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# -------------------- SPLIT COM TOC/TOD --------------------
def split_legs_from_wps_with_toc_tod(wps, wind_from, wind_kt, qnh, oat, mag_var, mag_is_e, desc_angle):
    if not isinstance(wps, (list,tuple)) or len(wps)<2: return [], []
    W=[]
    for i,w in enumerate(wps):
        if isinstance(w,dict): W.append({"name":str(w.get("name",f"WP{i+1}")),"lat":float(w["lat"]),"lon":float(w["lon"]),"alt":float(w.get("alt",0.0))})
        else: raise ValueError("WP inválido")
    legs=[]; rps=[W[0]]

    for i in range(len(W)-1):
        A,B=W[i],W[i+1]
        tc=gc_course_tc(A["lat"],A["lon"],B["lat"],B["lon"])
        dist=gc_dist_nm(A["lat"],A["lon"],B["lat"],B["lon"])
        _,_,GScl=wind_triangle(tc, TAS_CLIMB,  wind_from, wind_kt)
        _,_,GScr=wind_triangle(tc, TAS_CRUISE, wind_from, wind_kt)
        _,_,GSde=wind_triangle(tc, TAS_DESC,   wind_from, wind_kt)
        ROC = roc_interp(press_alt(A["alt"],qnh), oat)
        ROD = max(100.0, GSde * 5.0 * (desc_angle/3.0))

        if B["alt"]>A["alt"]:   # CLIMB -> TOC
            t_need=(B["alt"]-A["alt"])/max(ROC,1e-6); d_need=GScl*(t_need/60.0)
            if d_need<=dist:     # <= para garantir TOC também no fim
                lat_toc,lon_toc=point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"],d_need)
                wp_toc={"name":"TOC","lat":lat_toc,"lon":lon_toc,"alt":B["alt"]}
                legs.append({"TC":tc,"Dist":d_need,"Alt0":A["alt"],"Alt1":B["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
                rps.append(wp_toc)
                rem=dist-d_need
                if rem>0:
                    legs.append({"TC":tc,"Dist":rem,"Alt0":B["alt"],"Alt1":B["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
                rps.append(B)
            else:
                legs.append({"TC":tc,"Dist":dist,"Alt0":A["alt"],"Alt1":B["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
                rps.append(B)

        elif B["alt"]<A["alt"]: # DESCENT -> TOD
            t_need=(A["alt"]-B["alt"])/max(ROD,1e-6); d_need=GSde*(t_need/60.0)
            if d_need<=dist:
                d_cru=dist-d_need
                lat_tod,lon_tod=point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"],d_cru)
                wp_tod={"name":"TOD","lat":lat_tod,"lon":lon_tod,"alt":A["alt"]}
                if d_cru>0:
                    legs.append({"TC":tc,"Dist":d_cru,"Alt0":A["alt"],"Alt1":A["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
                rps.append(wp_tod)
                legs.append({"TC":tc,"Dist":d_need,"Alt0":A["alt"],"Alt1":B["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
                rps.append(B)
            else:
                legs.append({"TC":tc,"Dist":dist,"Alt0":A["alt"],"Alt1":B["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
                rps.append(B)

        else:                   # LEVEL
            legs.append({"TC":tc,"Dist":dist,"Alt0":A["alt"],"Alt1":A["alt"],"Wfrom":wind_from,"Wkt":wind_kt,"CK":st.session_state.ck_default,"HoldMin":0.0,"HoldFF":0.0})
            rps.append(B)

    return legs, rps

# -------------------- FASES + BURN (FF=20) --------------------
def build_leg(tc, dist, alt0, alt1, wfrom, wkt, ck_min, qnh, oat, var, var_is_e, desc_angle, hold_min=0.0, hold_ff=0.0):
    pa0 = press_alt(alt0,qnh); pa1 = press_alt(alt1,qnh); pa_avg=(pa0+pa1)/2.0
    ROC = roc_interp(pa0, oat)
    _, THc, GScl = wind_triangle(tc, TAS_CLIMB,  wfrom, wkt)
    _, THr, GScr = wind_triangle(tc, TAS_CRUISE, wfrom, wkt)
    _, THd, GSde = wind_triangle(tc, TAS_DESC,   wfrom, wkt)
    ROD = max(100.0, GSde*5.0*(desc_angle/3.0))
    MHc = apply_var(THc, var, var_is_e); MHr = apply_var(THr, var, var_is_e); MHd = apply_var(THd, var, var_is_e)
    segs=[]; marker=None

    if alt1>alt0:  # climb
        t_need=(alt1-alt0)/max(ROC,1e-6); d_need=GScl*(t_need/60.0)
        if d_need<=dist:
            tA=rt10(t_need*60)
            segs.append({"name":"Climb→TOC","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_CLIMB,"ff":FF_CONST,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem=dist-d_need
            if rem>0:
                tB=rt10((rem/max(GScr,1e-9))*3600)
                segs.append({"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_CRUISE,"ff":FF_CONST,"time":tB,"dist":rem,"alt0":alt1,"alt1":alt1})
            marker={"type":"TOC","t":rt10(t_need*60)}
        else:
            tA=rt10((dist/max(GScl,1e-9))*3600); gained=ROC*(tA/60.0)
            segs.append({"name":"Climb (não atinge)","TH":THc,"MH":MHc,"GS":GScl,"TAS":TAS_CLIMB,"ff":FF_CONST,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0+gained})

    elif alt1<alt0: # descent
        t_need=(alt0-alt1)/max(ROD,1e-6); d_need=GSde*(t_need/60.0)
        if d_need<=dist:
            tA=rt10(t_need*60)
            segs.append({"name":"Descent→TOD","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_DESC,"ff":FF_CONST,"time":tA,"dist":d_need,"alt0":alt0,"alt1":alt1})
            rem=dist-d_need
            if rem>0:
                tB=rt10((rem/max(GScr,1e-9))*3600)
                segs.insert(0,{"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_CRUISE,"ff":FF_CONST,"time":tB,"dist":rem,"alt0":alt0,"alt1":alt0})
            marker={"type":"TOD","t":rt10((rem/GScr)*3600) if rem>0 else 0}
        else:
            tA=rt10((dist/max(GSde,1e-9))*3600); lost=ROD*(tA/60.0)
            segs.append({"name":"Descent (não atinge)","TH":THd,"MH":MHd,"GS":GSde,"TAS":TAS_DESC,"ff":FF_CONST,"time":tA,"dist":dist,"alt0":alt0,"alt1":max(0.0,alt0-lost)})

    else: # level
        tA=rt10((dist/max(GScr,1e-9))*3600)
        segs.append({"name":"Cruise","TH":THr,"MH":MHr,"GS":GScr,"TAS":TAS_CRUISE,"ff":FF_CONST,"time":tA,"dist":dist,"alt0":alt0,"alt1":alt0})

    # hold
    hold_min=max(0.0,float(hold_min))
    if hold_min>0:
        segs.append({"name":"Hold","TH":segs[-1]["TH"],"MH":segs[-1]["MH"],"GS":0.0,"TAS":0.0,"ff":(hold_ff if hold_ff>0 else FF_CONST),
                     "time":rt10(hold_min*60.0),"dist":0.0,"alt0":segs[-1]["alt1"],"alt1":segs[-1]["alt1"]})

    for s in segs: s["burn"]=s["ff"]*(s["time"]/3600.0)
    return {"segments":segs,"tot_sec":sum(s['time'] for s in segs),"tot_burn":sum(s['burn'] for s in segs),"marker":marker}

# -------------------- RECOMPUTE --------------------
def recompute_all_by_leg():
    st.session_state.computed_by_leg=[]
    base=None
    if st.session_state.start_clock.strip():
        try:
            h,m=map(int, st.session_state.start_clock.split(":")); base=dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base=None
    efob=float(st.session_state.start_efob); clk=base
    for leg in st.session_state.legs:
        res=build_leg(leg['TC'],leg['Dist'],leg['Alt0'],leg['Alt1'],leg['Wfrom'],leg['Wkt'],leg['CK'],
                      st.session_state.qnh,st.session_state.oat,st.session_state.mag_var,st.session_state.mag_is_e,
                      st.session_state.desc_angle, leg.get('HoldMin',0.0), leg.get('HoldFF',0.0))
        phases=[]; tcur=0
        for seg in res["segments"]:
            e_start=efob; e_end=max(0.0, round(e_start - seg['burn'],1))
            if clk:
                c0=(clk+dt.timedelta(seconds=tcur)).strftime('%H:%M'); c1=(clk+dt.timedelta(seconds=tcur+seg['time'])).strftime('%H:%M')
            else:
                c0=f"T+{mmss(tcur)}"; c1=f"T+{mmss(tcur+seg['time'])}"
            phases.append({"seg":seg, "efob_start":e_start, "efob_end":e_end, "c0":c0, "c1":c1})
            tcur+=seg['time']; efob=e_end
        if clk: clk = clk + dt.timedelta(seconds=sum(s['time'] for s in res["segments"]))
        st.session_state.computed_by_leg.append({"leg":leg,"phases":phases,"tot_sec":sum(p["seg"]["time"] for p in phases),"tot_burn":round(sum(p["seg"]["burn"] for p in phases),1),"marker":res["marker"]})

# -------------------- CRIAR LEGS A PARTIR DE WPS --------------------
btn,_=st.columns([2,6])
with btn:
    if st.button("Gerar/Atualizar legs (com TOC/TOD)", type="primary", use_container_width=True):
        if len(st.session_state.wps)<2: st.warning("Precisas de pelo menos 2 waypoints.")
        else:
            legs, rps = split_legs_from_wps_with_toc_tod(
                st.session_state.wps, st.session_state.wind_from, st.session_state.wind_kt,
                st.session_state.qnh, st.session_state.oat, st.session_state.mag_var, st.session_state.mag_is_e,
                st.session_state.desc_angle
            )
            st.session_state.legs=legs; st.session_state.route_points=rps
            recompute_all_by_leg()
            st.success(f"{len(legs)} legs criadas.")

# -------------------- MOSTRAR LEGS --------------------
if st.session_state.legs:
    recompute_all_by_leg()
    total_t=sum(c["tot_sec"] for c in st.session_state.computed_by_leg)
    total_f=round(sum(c["tot_burn"] for c in st.session_state.computed_by_leg),1)
    st.markdown(f"<div class='kvrow'><div class='kv'>ETE Total: <b>{hhmmss(total_t)}</b></div><div class='kv'>Burn Total: <b>{total_f:.1f} L</b></div></div>", unsafe_allow_html=True)

# -------------------- MAPA (rota + riscas + dog houses + TOC/TOD) --------------------
def triangle_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    base_c_lat, base_c_lon = dest_point(lat, lon, heading_deg, -h_nm/2.0)
    apex_lat, apex_lon      = dest_point(lat, lon, heading_deg,  h_nm/2.0)
    bl_lat, bl_lon = dest_point(base_c_lat, base_c_lon, heading_deg-90.0, w_nm/2.0)
    br_lat, br_lon = dest_point(base_c_lat, base_c_lon, heading_deg+90.0, w_nm/2.0)
    return (bl_lat, bl_lon), (apex_lat, apex_lon), (br_lat, br_lon)
def triangle_poly_coords(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65):
    bl, ap, br = triangle_coords(lat, lon, heading_deg, h_nm, w_nm)
    return [[bl[1], bl[0]], [ap[1], ap[0]], [br[1], br[0]], [bl[1], bl[0]]]
def triangle_hatch_paths(lat, lon, heading_deg, h_nm=0.9, w_nm=0.65, lines=3):
    bl, ap, br = triangle_coords(lat, lon, heading_deg, h_nm, w_nm); paths=[]
    for t in [i/(lines+1.0) for i in range(1, lines+1)]:
        L_lat = bl[0] + t*(ap[0]-bl[0]); L_lon = bl[1] + t*(ap[1]-bl[1])
        R_lat = br[0] + (1-t)*(ap[0]-br[0]); R_lon = br[1] + (1-t)*(ap[1]-br[1])
        paths.append({"path":[[L_lon,L_lat],[R_lon,R_lat]]})
    return paths

if len(st.session_state.route_points)>=2 and st.session_state.legs and st.session_state.computed_by_leg:
    rps = st.session_state.route_points
    # dados layers
    route_data, tick_data, tri_data, tri_hatch = [], [], [], []
    mh_text, th_text, dist_text, gs_text, ete_text = [], [], [], [], []
    fix_pts, fix_lbls, toc_pts, tod_pts = [], [], [], []

    # estilo VFR
    style_map = {
        "CARTO Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "CARTO Positron": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "CARTO Dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    }[st.session_state.map_style]

    leg_idx = 0
    for i in range(len(rps)-1):
        A,B=rps[i],rps[i+1]
        route_data.append({"path":[[A["lon"],A["lat"]],[B["lon"],B["lat"]]],"name":f"{A['name']}→{B['name']}"})
        # fixes
        for P in [A,B] if i==0 else [B]:
            label=str(P["name"])
            if label.startswith("TOC"): toc_pts.append({"position":[P["lon"],P["lat"]]})
            elif label.startswith("TOD"): tod_pts.append({"position":[P["lon"],P["lat"]]})
            else:
                fix_pts.append({"position":[P["lon"],P["lat"]]})
                fix_lbls.append({"position":[P["lon"],P["lat"]], "text": label})

        # Ticks e doghouse por leg
        if leg_idx < len(st.session_state.legs):
            leg = st.session_state.legs[leg_idx]
            comp = st.session_state.computed_by_leg[leg_idx]
            # usar primeira fase com GS>0 para a leg
            phase = next((p for p in comp["phases"] if p["seg"]["GS"]>0), comp["phases"][0])
            gs = phase["seg"]["GS"]; total_t = comp["tot_sec"]; dist_leg = leg["Dist"]; tc = leg["TC"]
            # riscas 2 min
            k=1
            while k*120 <= total_t:
                d=min(gs*(k*120/3600.0), dist_leg)
                latm,lonm = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"],d)
                llat,llon = dest_point(latm,lonm,tc-90,0.15)
                rlat,rlon = dest_point(latm,lonm,tc+90,0.15)
                tick_data.append({"path":[[llon,llat],[rlon,rlat]]}); k+=1
            # doghouse
            mid_lat,mid_lon = point_along_gc(A["lat"],A["lon"],B["lat"],B["lon"],dist_leg/2.0)
            off_lat,off_lon = dest_point(mid_lat,mid_lon,tc+90,0.30)
            tri_data.append({"polygon": triangle_poly_coords(off_lat,off_lon,tc)})
            tri_hatch += triangle_hatch_paths(off_lat,off_lon,tc,lines=3)
            # slots de texto (MH grande + TH, Dist, GS, ETE)
            mh = rang(phase["seg"]["MH"]); th = rang(phase["seg"]["TH"])
            pos_mh_lat,pos_mh_lon = dest_point(off_lat,off_lon,tc,0.55)
            pos_th_lat,pos_th_lon = dest_point(off_lat,off_lon,tc,-0.25)
            pos_d_lat,pos_d_lon   = dest_point(off_lat,off_lon,tc,0.25)
            pos_gs_lat,pos_gs_lon = dest_point(off_lat,off_lon,tc,0.45)
            pos_et_lat,pos_et_lon = dest_point(off_lat,off_lon,tc,0.65)
            mh_text.append({"position":[pos_mh_lon,pos_mh_lat],"text":f"{mh}M"})
            th_text.append({"position":[pos_th_lon,pos_th_lat],"text":f"{th}T"})
            dist_text.append({"position":[pos_d_lon,pos_d_lat],"text":f"{r10f(dist_leg)}nm"})
            gs_text.append({"position":[pos_gs_lon,pos_gs_lat],"text":f"GS {rint(gs)}"})
            ete_text.append({"position":[pos_et_lon,pos_et_lat],"text": mmss(total_t)})
            leg_idx += 1

    # desloca TOC/TOD se coincidir com fix para ficarem visíveis
    def offset_if_same(pts, others, bearing=90.0):
        out=[]
        for p in pts:
            same = any(abs(p["position"][0]-q["position"][0])<1e-5 and abs(p["position"][1]-q["position"][1])<1e-5 for q in others)
            if same:
                lat_off, lon_off = dest_point(p["position"][1], p["position"][0], bearing, 0.2)
                out.append({"position":[lon_off,lat_off]})
            else: out.append(p)
        return out
    toc_pts = offset_if_same(toc_pts, fix_pts)
    tod_pts = offset_if_same(tod_pts, fix_pts)

    # layers
    route_layer = pdk.Layer("PathLayer", data=route_data, get_path="path", get_color=[180,0,255,220], width_min_pixels=4)
    ticks_layer = pdk.Layer("PathLayer", data=tick_data, get_path="path", get_color=[0,0,0,255], width_min_pixels=2)
    tri_layer   = pdk.Layer("PolygonLayer", data=tri_data, get_polygon="polygon",
                            get_fill_color=[255,255,255,230], get_line_color=[0,0,0,255],
                            line_width_min_pixels=2, stroked=True, filled=True)
    tri_hatch   = pdk.Layer("PathLayer", data=tri_hatch, get_path="path", get_color=[0,0,0,200], width_min_pixels=1)

    text_mh = pdk.Layer("TextLayer", data=mh_text, get_position="position", get_text="text", get_size=22, get_color=[225,29,72])
    text_th = pdk.Layer("TextLayer", data=th_text, get_position="position", get_text="text", get_size=14, get_color=[0,0,0])
    text_d  = pdk.Layer("TextLayer", data=dist_text, get_position="position", get_text="text", get_size=14, get_color=[0,0,0])
    text_gs = pdk.Layer("TextLayer", data=gs_text, get_position="position", get_text="text", get_size=14, get_color=[0,0,0])
    text_et = pdk.Layer("TextLayer", data=ete_text, get_position="position", get_text="text", get_size=14, get_color=[0,0,0])

    fix_layer = pdk.Layer("ScatterplotLayer", data=fix_pts, get_position="position",
                          get_radius=60, get_fill_color=[255,255,255,230], get_line_color=[0,0,0,255], line_width_min_pixels=2)
    fix_text  = pdk.Layer("TextLayer", data=fix_lbls, get_position="position", get_text="text", get_size=14, get_color=[0,0,0])

    toc_layer = pdk.Layer("ScatterplotLayer", data=toc_pts, get_position="position",
                          get_radius=70, get_fill_color=[37,99,235,230], get_line_color=[255,255,255,255], line_width_min_pixels=2)
    tod_layer = pdk.Layer("ScatterplotLayer", data=tod_pts, get_position="position",
                          get_radius=70, get_fill_color=[225,29,72,230], get_line_color=[255,255,255,255], line_width_min_pixels=2)
    toc_text  = pdk.Layer("TextLayer", data=[{"position":p["position"],"text":"TOC"} for p in toc_pts], get_position="position", get_text="text", get_size=14, get_color=[37,99,235])
    tod_text  = pdk.Layer("TextLayer", data=[{"position":p["position"],"text":"TOD"} for p in tod_pts], get_position="position", get_text="text", get_size=14, get_color=[225,29,72])

    mean_lat = sum([w["lat"] for w in rps])/len(rps); mean_lon = sum([w["lon"] for w in rps])/len(rps)
    deck = pdk.Deck(
        map_style=style_map,  # CARTO Voyager: feições e nomes de terras (VFR-like)
        initial_view_state=pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=8, pitch=0),
        layers=[route_layer, ticks_layer, tri_layer, tri_hatch,
                text_mh, text_th, text_d, text_gs, text_et,
                fix_layer, fix_text, toc_layer, tod_layer, toc_text, tod_text],
        tooltip={"text":"{name}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Adiciona WPs e carrega «Gerar/Atualizar legs» para ver a rota, TOC/TOD e as dog houses.")

