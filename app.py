# app.py — NAVLOG — rev26
# ---------------------------------------------------------------
# - Mapa estável (sem reruns contínuos) com rota/markers/ticks.
# - TOD usa ROD (ft/min).
# - PDF principal (NAVLOG_FORM.pdf) tem 2 páginas (capacidade 22 legs).
# - PDF de continuação (NAVLOG_FORM_1.pdf) só se >22 legs (capacidade +11).
# - Frequências: Departure / Enroute / Arrival (MHz; arrival default 131.675).
# - Alternate: cálculo de distância/ETE/queima e preenchimento básico.
# - Leg01 mostra o AERÓDROMO DE PARTIDA (não salta para TOC).
# - Header extra: ETD/ETA, STUDENT (AMOIT), LESSON, INSTRUCTOR, AIRCRAFT, REG, CALLSIGN.
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import folium, math, re, datetime as dt, difflib, os
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MarkerCluster
from math import degrees

from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName

# ======== CONSTANTES ========
CLIMB_TAS, CRUISE_TAS, DESCENT_TAS = 70.0, 90.0, 90.0   # kt
FUEL_FLOW = 20.0                                        # L/h
EARTH_NM  = 3440.065

CP_TICK_HALF = 0.38
PROFILE_COLOR = dict(CLIMB="#FF7A00", LEVEL="#8B5CF6", DESCENT="#10B981")

REG_OPTIONS = ["CS-DHS","CS-DHT","CS-DHU","CS-DHV","CS-DHW","CS-ECC","CS-ECD"]
ACFT_OPTIONS = ["P208", "C172", "PA28"]

# ======== PAGE / STYLE ========
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

# ======== STATE ========
def ens(k, v): return st.session_state.setdefault(k, v)
ens("wind_from", 250); ens("wind_kt", 10)
ens("mag_var", 1.0); ens("mag_is_e", False)
ens("roc_fpm", 600)
ens("rod_fpm", 500)
ens("start_clock", ""); ens("start_efob", 85.0)
ens("ck_default", 2)

ens("wps", []); ens("legs", []); ens("route_nodes", [])

ens("map_base", "OpenTopoMap (VFR-ish)")
ens("text_scale", 1.0)
ens("map_center", (39.7, -8.1)); ens("map_zoom", 7)

ens("depart_freq", "119.805")
ens("enroute_freq", "123.755")
ens("arrival_freq", "131.675")

ens("aircraft", "P208")
ens("registration", "CS-DHS")
ens("callsign", "RVP")
ens("student", "AMOIT")
ens("lesson", "")
ens("instructor", "")
ens("etdeta", "")   # ex. "14:00/17:30"

# ======== HEADER / GLOBAIS ========
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
        st.session_state.rod_fpm   = st.number_input("ROD global (ft/min)", 200, 1500, int(st.session_state.rod_fpm), step=10)
    with c4:
        st.session_state.start_efob= st.number_input("EFOB inicial (L)", 0.0, 200.0, float(st.session_state.start_efob), step=0.5)
        st.session_state.start_clock = st.text_input("Hora off-blocks (HH:MM)", st.session_state.start_clock)
        st.session_state.ck_default  = st.number_input("CP por defeito (min)", 1, 10, int(st.session_state.ck_default))
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    c5,c6,c7,c8 = st.columns(4)
    with c5:
        st.session_state.aircraft = st.selectbox("Aircraft", ACFT_OPTIONS, index=ACFT_OPTIONS.index(st.session_state.aircraft))
        st.session_state.registration = st.selectbox("Registration", REG_OPTIONS, index=max(0, REG_OPTIONS.index(st.session_state.registration) if st.session_state.registration in REG_OPTIONS else 0))
    with c6:
        st.session_state.callsign = st.text_input("Callsign", st.session_state.callsign)
        st.session_state.student  = st.text_input("Student", st.session_state.student)
    with c7:
        st.session_state.lesson   = st.text_input("Lesson", st.session_state.lesson)
        st.session_state.instructor = st.text_input("Instructor", st.session_state.instructor)
    with c8:
        st.session_state.etdeta   = st.text_input("ETD/ETA (HH:MM/HH:MM)", st.session_state.etdeta)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    c9,c10,c11,_ = st.columns(4)
    with c9:
        st.session_state.depart_freq  = st.text_input("Departure Freq (MHz)", st.session_state.depart_freq)
    with c10:
        st.session_state.enroute_freq = st.text_input("Enroute Freq (MHz)",  st.session_state.enroute_freq)
    with c11:
        st.session_state.arrival_freq = st.text_input("Arrival Freq (MHz)",  st.session_state.arrival_freq)
    st.form_submit_button("Aplicar")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== WPs “mínimos” (podes manter a tua camada CSV; aqui só mapa & rota) ========
def append_wp(name, lat, lon, alt):
    st.session_state.wps.append({"name": str(name), "lat": float(lat), "lon": float(lon), "alt": float(alt)})

st.subheader("Waypoints")
with st.expander("Adicionar manualmente", expanded=False):
    c1,c2,c3,c4 = st.columns(4)
    nm = c1.text_input("Nome", value="LPSO")
    lat = c2.number_input("Lat", -90.0, 90.0, 39.4667, step=0.0001, format="%.6f")
    lon = c3.number_input("Lon", -180.0, 180.0, -8.3670, step=0.0001, format="%.6f")
    alt = c4.number_input("Alt (ft)", 0.0, 18000.0, 400.0, step=50.0)
    if st.button("➕ Adicionar WP"):
        append_wp(nm, lat, lon, alt)

if st.session_state.wps:
    for i, w in enumerate(st.session_state.wps):
        cols = st.columns([4,2,2,2,1])
        w["name"] = cols[0].text_input(f"Nome — WP{i+1}", w["name"], key=f"wpn_{i}")
        w["lat"]  = float(cols[1].number_input(f"Lat — WP{i+1}", -90.0,90.0,w["lat"], step=0.0001, format="%.6f", key=f"wplat_{i}"))
        w["lon"]  = float(cols[2].number_input(f"Lon — WP{i+1}", -180.0,180.0,w["lon"], step=0.0001, format="%.6f", key=f"wplon_{i}"))
        w["alt"]  = float(cols[3].number_input(f"Alt — WP{i+1}", 0.0, 18000.0, w["alt"], step=50.0, key=f"wpalt_{i}"))
        if cols[4].button("🗑️", key=f"del_{i}"):
            st.session_state.wps.pop(i)
            st.experimental_rerun()

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== TOC/TOD E LEGS ========
def build_route_nodes(user_wps, wind_from, wind_kt, roc_fpm, rod_fpm):
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
            dh = B["alt"] - A["alt"]
            t_need = dh / max(roc_fpm, 1.0)
            d_need = gs_cl * (t_need/60.0)
            if d_need < dist - 0.05:
                lat_toc, lon_toc = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], d_need)
                nodes.append({"name": f"TOC L{i+1}", "lat": lat_toc, "lon": lon_toc, "alt": B["alt"]})
        elif B["alt"] < A["alt"]:
            dh = A["alt"] - B["alt"]
            t_need = dh / max(rod_fpm, 1.0)
            d_need = gs_de * (t_need/60.0)
            if d_need < dist - 0.05:
                pos_from_start = max(0.0, dist - d_need)
                lat_tod, lon_tod = point_along_gc(A["lat"], A["lon"], B["lat"], B["lon"], pos_from_start)
                nodes.append({"name": f"TOD L{i+1}", "lat": lat_tod, "lon": lon_tod, "alt": A["alt"]})
    nodes.append(user_wps[-1])
    return nodes

def build_legs_from_nodes(nodes, wind_from, wind_kt, mag_var, mag_is_e, ck_every_min, start_efob, start_clock):
    legs = []
    if len(nodes) < 2: return legs

    base_time = None
    if start_clock.strip():
        try:
            h,m = map(int, start_clock.split(":"))
            base_time = dt.datetime.combine(dt.date.today(), dt.time(h,m))
        except: base_time = None

    carry_efob = float(start_efob)
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

        legs.append({
            "i":i+1,"A":A,"B":B,"profile":profile,"TC":tc,"TH":th,"MH":mh,"TAS":tas,"GS":gs,
            "Dist":dist,"time_sec":time_sec,"burn":r10f(burn),
            "efob_start":r10f(efob_start),"efob_end":r10f(efob_end),
            "clock_start":clk_start,"clock_end":clk_end,"cps":cps
        })
        t_cursor += time_sec; carry_efob = efob_end
    return legs

if st.button("Gerar/Atualizar rota (insere TOC/TOD) ✅", type="primary", use_container_width=True):
    st.session_state.route_nodes = build_route_nodes(
        st.session_state.wps, st.session_state.wind_from, st.session_state.wind_kt,
        st.session_state.roc_fpm, st.session_state.rod_fpm
    )
    st.session_state.legs = build_legs_from_nodes(
        st.session_state.route_nodes, st.session_state.wind_from, st.session_state.wind_kt,
        st.session_state.mag_var, st.session_state.mag_is_e, st.session_state.ck_default,
        st.session_state.start_efob, st.session_state.start_clock
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
        + f"<div class='kv'>🧮 Nº legs: <b>{len(st.session_state.legs)}</b></div>"
        + "</div>", unsafe_allow_html=True
    )
    # ETD/ETA auto (se off-blocks definido)
    if not st.session_state.etdeta and st.session_state.start_clock.strip():
        start = st.session_state.start_clock
        eta   = (dt.datetime.strptime(start,"%H:%M")+dt.timedelta(seconds=total_sec)).strftime("%H:%M")
        st.info(f"ETD/ETA sugerido: {start}/{eta} (podes editar no cabeçalho).")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# ======== MAPA (estável: sem mutações do session_state durante render) ========
def render_map(nodes, legs, base="OpenTopoMap (VFR-ish)"):
    if not nodes or not legs:
        st.info("Adiciona WPs e gera a rota para ver o mapa.")
        return
    m = folium.Map(location=[nodes[0]["lat"], nodes[0]["lon"]], zoom_start=8,
                   tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                   attr="© OpenTopoMap", control_scale=True)
    Fullscreen().add_to(m)

    # Polylines
    for L in legs:
        pts = [(L["A"]["lat"],L["A"]["lon"]), (L["B"]["lat"],L["B"]["lon"])]
        folium.PolyLine(pts, color="#ffffff", weight=8, opacity=1).add_to(m)
        folium.PolyLine(pts, color=PROFILE_COLOR.get(L["profile"], "#8B5CF6"), weight=4, opacity=1).add_to(m)

    # Ticks a cada CK minutos
    for L in legs:
        if L["GS"]<=0 or not L["cps"]: continue
        for cp in L["cps"]:
            d = min(L["Dist"], L["GS"]*(cp["t"]/3600.0))
            latm, lonm = point_along_gc(L["A"]["lat"],L["A"]["lon"],L["B"]["lat"],L["B"]["lon"], d)
            llat, llon = dest_point(latm, lonm, L["TC"]-90, CP_TICK_HALF)
            rlat, rlon = dest_point(latm, lonm, L["TC"]+90, CP_TICK_HALF)
            folium.PolyLine([(llat,llon),(rlat,rlon)], color="#111111", weight=3, opacity=1).add_to(m)

    # WPs
    for N in nodes:
        folium.CircleMarker((N["lat"],N["lon"]), radius=5, color="#1D4ED8", fill=True, fill_opacity=1,
                            tooltip=N["name"]).add_to(m)

    st_folium(m, width=None, height=520, key="mainmap", returned_objects=[])

if st.session_state.route_nodes and st.session_state.legs:
    render_map(st.session_state.route_nodes, st.session_state.legs)
else:
    st.info("Mapa será mostrado após gerares a rota.")

# ===============================================================
#                📄  NAVLOG — PDFs (2 págs por PDF)
# ===============================================================
TEMPLATE_MAIN = "NAVLOG_FORM.pdf"     # 2 páginas (legs 01..22 + totais 23)
TEMPLATE_CONT = "NAVLOG_FORM_1.pdf"   # 2 páginas (legs 12..22)

def _set_need_appearances(pdf):
    if pdf.Root.AcroForm:
        pdf.Root.AcroForm.update(PdfDict(NeedAppearances=True))

def _fill_pdf(template_path:str, out_path:str, data:dict):
    pdf = PdfReader(template_path)
    _set_need_appearances(pdf)
    for page in pdf.pages:
        if not getattr(page, "Annots", None): continue
        for a in page.Annots:
            if a.Subtype==PdfName('Widget') and a.T:
                key = str(a.T)[1:-1]
                if key in data: a.update(PdfDict(V=str(data[key])))
    PdfWriter(out_path, trailer=pdf).write()
    return out_path

def _list_leg_slots(pdf_path):
    pdf = PdfReader(pdf_path)
    slots = []
    for page in pdf.pages:
        if not getattr(page, "Annots", None): continue
        for a in page.Annots:
            if a.Subtype==PdfName('Widget') and a.T:
                k = str(a.T)[1:-1]
                m = re.match(r"Leg(\d{2})_Waypoint$", k)
                if m: slots.append(int(m.group(1)))
    slots = sorted(set(slots))
    return slots

# detectar capacidades reais (não “hardcode”)
MAIN_SLOTS = _list_leg_slots(TEMPLATE_MAIN)     # p.ex. [1..22]
CONT_SLOTS = _list_leg_slots(TEMPLATE_CONT)     # p.ex. [12..22]
MAIN_CAP   = len(MAIN_SLOTS)
CONT_CAP   = len(CONT_SLOTS)

def mmss_pdf(sec:int):
    m,s = divmod(int(round(sec)),60); return f"{m:02d}:{s:02d}"

def sum_time(legs, profile): return sum(L["time_sec"] for L in legs if L["profile"]==profile)
def sum_burn(legs, profile): return round(sum(L["burn"] for L in legs if L["profile"]==profile),1)

def build_payload_main(legs):
    """Preenche NAVLOG_FORM.pdf (duas páginas)."""
    total_sec = sum(L["time_sec"] for L in legs)
    obs_times = f"Climb {mmss_pdf(sum_time(legs,'CLIMB'))} / Cruise {mmss_pdf(sum_time(legs,'LEVEL'))} / Descent {mmss_pdf(sum_time(legs,'DESCENT'))}"

    d = {
        "AIRCRAFT": st.session_state.aircraft,
        "REGISTRATION": st.session_state.registration,
        "CALLSIGN": st.session_state.callsign,
        "ETD/ETA": st.session_state.etdeta,
        "LESSON": st.session_state.lesson,
        "INSTRUTOR": st.session_state.instructor,
        "STUDENT": st.session_state.student,
        "DEPT": st.session_state.depart_freq,
        "ENROUTE": st.session_state.enroute_freq,
        "ARRIVAL": st.session_state.arrival_freq,
        "WIND": f"{int(st.session_state.wind_from)}/{int(st.session_state.wind_kt)}",
        "MAG_VAR": f"{abs(st.session_state.mag_var):.0f}°{'E' if st.session_state.mag_is_e else 'W'}",
        "FLT TIME": mmss_pdf(total_sec),
        "CLIMB FUEL": f"{sum_burn(legs,'CLIMB'):.1f}",
        "OBSERVATIONS": f"{obs_times}",
        "Leg_Number": str(len(legs)),
        # Alternate básicos (se definido)
        "Alternate_Airfield": st.session_state.get("altn_name",""),
        "Alternate_Elevation": st.session_state.get("altn_elev",""),
    }

    # Se houver cálculo de ALTN, acrescenta resumo nas observações
    if st.session_state.get("altn_summary"):
        d["OBSERVATIONS"] += f" | ALTN {st.session_state['altn_summary']}"

    # Leg slots do template principal
    slots = MAIN_SLOTS
    acc_d, acc_t = 0.0, 0

    for row_idx, leg_idx in enumerate(slots):
        if row_idx >= len(legs): break
        L = legs[row_idx]
        # ——— Primeira linha mostra o AERÓDROMO DE PARTIDA ———
        # (Leg01_Waypoint usa origem; demais usam destino)
        fix_name = L["A"]["name"] if leg_idx == min(slots) else L["B"]["name"]

        d[f"Leg{leg_idx:02d}_Waypoint"]            = str(fix_name)
        d[f"Leg{leg_idx:02d}_Altitude_FL"]         = str(int(round(L["B"]["alt"])))
        d[f"Leg{leg_idx:02d}_True_Course"]         = f"{int(round(L['TC'])):03d}"
        d[f"Leg{leg_idx:02d}_True_Heading"]        = f"{int(round(L['TH'])):03d}"
        d[f"Leg{leg_idx:02d}_Magnetic_Heading"]    = f"{int(round(L['MH'])):03d}"
        d[f"Leg{leg_idx:02d}_True_Airspeed"]       = str(int(round(L["TAS"])))
        d[f"Leg{leg_idx:02d}_Ground_Speed"]        = str(int(round(L["GS"])))
        d[f"Leg{leg_idx:02d}_Leg_Distance"]        = f"{L['Dist']:.1f}"
        acc_d = round(acc_d + L["Dist"], 1)
        d[f"Leg{leg_idx:02d}_Cumulative_Distance"] = f"{acc_d:.1f}"
        d[f"Leg{leg_idx:02d}_Leg_ETE"]             = mmss_pdf(L["time_sec"])
        acc_t += L["time_sec"]
        d[f"Leg{leg_idx:02d}_Cumulative_ETE"]      = mmss_pdf(acc_t)
        d[f"Leg{leg_idx:02d}_ETO"]                 = L["clock_end"]
        d[f"Leg{leg_idx:02d}_Planned_Burnoff"]     = f"{L['burn']:.1f}"
        d[f"Leg{leg_idx:02d}_Estimated_FOB"]       = f"{L['efob_end']:.1f}"

    # Totais (linha 23 do template)
    if len(legs) > 0:
        d["Leg23_Leg_Distance"] = f"{acc_d:.1f}"
        d["Leg23_Leg_ETE"]      = mmss_pdf(acc_t)
        d["Leg23_Planned_Burnoff"] = f"{sum(L['burn'] for L in legs):.1f}"
        d["Leg23_Estimated_FOB"]   = f"{legs[-1]['efob_end']:.1f}"

    return d

def build_payload_cont(legs_overflow):
    """Preenche NAVLOG_FORM_1.pdf (duas páginas) se necessário."""
    if not legs_overflow: return None
    d = {
        "FLIGHT_LEVEL_ALTITUDE": "",
        "TEMP_ISA_DEV": "",
        "OBSERVATIONS": "",
        "Alternate_Airfield": st.session_state.get("altn_name",""),
        "Alternate_Elevation": st.session_state.get("altn_elev",""),
    }
    slots = CONT_SLOTS
    for row_idx, leg_idx in enumerate(slots):
        if row_idx >= len(legs_overflow): break
        L = legs_overflow[row_idx]
        # aqui não há necessidade de “origem na primeira”, já é continuação
        fix_name = L["B"]["name"]
        d[f"Leg{leg_idx:02d}_Waypoint"]        = str(fix_name)
        d[f"Leg{leg_idx:02d}_Altitude_FL"]     = str(int(round(L["B"]["alt"])))
        d[f"Leg{leg_idx:02d}_True_Course"]     = f"{int(round(L['TC'])):03d}"
        d[f"Leg{leg_idx:02d}_True_Heading"]    = f"{int(round(L['TH'])):03d}"
        d[f"Leg{leg_idx:02d}_Magnetic_Heading"]= f"{int(round(L['MH'])):03d}"
        d[f"Leg{leg_idx:02d}_True_Airspeed"]   = str(int(round(L["TAS"])))
        d[f"Leg{leg_idx:02d}_Ground_Speed"]    = str(int(round(L["GS"])))
        d[f"Leg{leg_idx:02d}_Leg_Distance"]    = f"{L['Dist']:.1f}"
        d[f"Leg{leg_idx:02d}_Leg_ETE"]         = mmss_pdf(L["time_sec"])
        d[f"Leg{leg_idx:02d}_ETO"]             = L["clock_end"]
        d[f"Leg{leg_idx:02d}_Planned_Burnoff"] = f"{L['burn']:.1f}"
        d[f"Leg{leg_idx:02d}_Estimated_FOB"]   = f"{L['efob_end']:.1f}"
    return d

st.subheader("Gerar PDF(s)")
if st.button("📄 Gerar NAVLOG PDF(s)", type="primary", use_container_width=True):
    if not st.session_state.legs:
        st.error("Gera a rota primeiro.")
    else:
        legs_all = st.session_state.legs[:]
        # Principal leva até MAIN_CAP (22)
        main_legs = legs_all[:MAIN_CAP]
        overflow  = legs_all[MAIN_CAP:]
        data_main = build_payload_main(main_legs)
        path_main = _fill_pdf(TEMPLATE_MAIN, "NAVLOG_FILLED.pdf", data_main)
        with open(path_main, "rb") as f:
            st.download_button("⬇️ NAVLOG (principal — 2 págs)", f.read(), file_name="NAVLOG_FILLED.pdf", use_container_width=True)

        # Continuação só se houver overflow (até CONT_CAP = 11)
        if overflow:
            cont_legs = overflow[:CONT_CAP]
            data_cont = build_payload_cont(cont_legs)
            path_cont = _fill_pdf(TEMPLATE_CONT, "NAVLOG_FILLED_1.pdf", data_cont)
            with open(path_cont, "rb") as f:
                st.download_button("⬇️ NAVLOG (continuação — 2 págs)", f.read(), file_name="NAVLOG_FILLED_1.pdf", use_container_width=True)
            if len(overflow) > CONT_CAP:
                st.warning(f"Aviso: tens {len(overflow)-CONT_CAP} pernas extra que não cabem (limite total: {MAIN_CAP+CONT_CAP}).")

# ===============================================================
#                  🛬  ALTERNATE (cálculo rápido)
# ===============================================================
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.subheader("Alternate — cálculo rápido")
alt_col = st.columns([3,1,1,1,1])
with alt_col[0]:
    altn_name = st.text_input("Alternate (nome/ident)", value=st.session_state.get("altn_name",""))
with alt_col[1]:
    altn_lat  = st.number_input("Lat ALTN", -90.0, 90.0, value=st.session_state.get("altn_lat", 0.0), step=0.0001, format="%.6f")
with alt_col[2]:
    altn_lon  = st.number_input("Lon ALTN", -180.0, 180.0, value=st.session_state.get("altn_lon", 0.0), step=0.0001, format="%.6f")
with alt_col[3]:
    altn_elev = st.text_input("Elevation ALTN (ft)", value=st.session_state.get("altn_elev",""))
with alt_col[4]:
    do_altn = st.button("Calcular ALTN", use_container_width=True)

if do_altn and st.session_state.legs:
    # usa último ponto da rota
    dest = st.session_state.route_nodes[-1]
    dist = gc_dist_nm(dest["lat"], dest["lon"], altn_lat, altn_lon)
    tc   = gc_course_tc(dest["lat"], dest["lon"], altn_lat, altn_lon)
    _, _, gs = wind_triangle(tc, CRUISE_TAS, st.session_state.wind_from, st.session_state.wind_kt)
    ete_sec = (dist/max(gs,1e-9))*3600
    burn    = FUEL_FLOW*(ete_sec/3600.0)

    # guardar no estado para o PDF
    st.session_state.altn_name = altn_name
    st.session_state.altn_lat  = altn_lat
    st.session_state.altn_lon  = altn_lon
    st.session_state.altn_elev = altn_elev
    st.session_state.altn_summary = f"{dist:.1f} nm / {mmss_pdf(ete_sec)} / {burn:.1f} L"

    st.success(f"ALTN ➜ {dist:.1f} nm | ETE {mmss_pdf(ete_sec)} | queima {burn:.1f} L")
elif do_altn and not st.session_state.legs:
    st.error("Gera a rota primeiro para calcular a partir do destino.")

st.caption("Observação: o PDF principal já inclui os campos Alternate_Airfield e Alternate_Elevation; o resumo do ALTN é adicionado às Observações.")



