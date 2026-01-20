import time
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "ok counter"

# -----------------------------
# Utils
# -----------------------------
def now() -> float:
    return time.time()

def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

def effective_elapsed(t0: float | None, finished_at: float | None, paused_total: float, paused: bool, pause_started_at: float | None) -> float:
    if not t0:
        return 0.0
    end = finished_at if finished_at else now()
    total = end - t0
    extra_pause = 0.0
    if paused and pause_started_at and not finished_at:
        extra_pause = end - pause_started_at
    return max(0.0, total - (paused_total + extra_pause))

def build_df(t0: float | None, ok_times: list[float]) -> pd.DataFrame:
    if not t0 or not ok_times:
        return pd.DataFrame(columns=["t", "dt", "elapsed_s"])
    ts = sorted(ok_times)
    df = pd.DataFrame({"t": ts})
    df["dt"] = pd.to_datetime(df["t"], unit="s")
    df["elapsed_s"] = df["t"] - t0
    return df

def per_minute_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["minute", "oks"])
    minute = (df["elapsed_s"] // 60).astype(int)
    s = minute.value_counts().sort_index()
    return pd.DataFrame({"minute": s.index.values, "oks": s.values})

def sliding_window_peak(times: np.ndarray, window_s: float) -> int:
    """Max OKs within any window window_s (two pointers)."""
    if times.size == 0:
        return 0
    i = 0
    best = 1
    for j in range(times.size):
        while times[j] - times[i] > window_s:
            i += 1
        best = max(best, j - i + 1)
    return int(best)

def max_streak_by_gap(times: np.ndarray, gap_s: float) -> int:
    """Max streak where consecutive OKs are <= gap_s apart."""
    if times.size == 0:
        return 0
    if times.size == 1:
        return 1
    diffs = np.diff(times)
    streak = 1
    best = 1
    for d in diffs:
        if d <= gap_s:
            streak += 1
        else:
            best = max(best, streak)
            streak = 1
    return int(max(best, streak))


# -----------------------------
# State
# -----------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("started", False)
    ss.setdefault("finished", False)
    ss.setdefault("paused", False)

    ss.setdefault("t0", None)
    ss.setdefault("finished_at", None)

    ss.setdefault("pause_started_at", None)
    ss.setdefault("paused_total", 0.0)

    ss.setdefault("ok_times", [])
    ss.setdefault("report", None)  # computed only when Finish

init_state()
ss = st.session_state


# -----------------------------
# Page + CSS (UI simples/bonita)
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
<style>
.block-container { max-width: 1000px; padding-top: 1.4rem; padding-bottom: 2.2rem; }
h1 { margin-bottom: 0.2rem; }
.small { opacity: 0.75; }

.topbar {
  padding: 14px 16px; border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  margin-bottom: 16px;
}

.kpiwrap { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 900px) { .kpiwrap { grid-template-columns: repeat(2, 1fr); } }

.kpi {
  padding: 14px 14px; border-radius: 16px;
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.10);
}
.kpi .label { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
.kpi .value { font-size: 1.55rem; font-weight: 800; line-height: 1.1; }
.kpi .sub { font-size: 0.85rem; opacity: 0.75; margin-top: 6px; }

.okcard {
  padding: 18px;
  border-radius: 22px;
  background: radial-gradient(80% 120% at 20% 0%, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
  margin: 16px 0;
}

div.stButton > button {
  width: 100%;
  height: 120px;
  border-radius: 26px;
  font-size: 46px;
  font-weight: 900;
  border: 1px solid rgba(255,255,255,0.18);
}

.controls div.stButton > button {
  height: 44px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  border-radius: 14px !important;
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1.1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.title(APP_TITLE)
st.caption("Clica ✅ OK sempre que alguém disser “ok”. No fim, carrega Finish para gerar o report.")

# -----------------------------
# Controls (top, no sidebar)
# -----------------------------
def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def do_start():
    ss.started = True
    ss.finished = False
    ss.paused = False
    ss.t0 = now()
    ss.finished_at = None
    ss.pause_started_at = None
    ss.paused_total = 0.0
    ss.ok_times = []
    ss.report = None

def do_pause():
    ss.paused = True
    ss.pause_started_at = now()

def do_resume():
    if ss.pause_started_at:
        ss.paused_total += now() - ss.pause_started_at
    ss.paused = False
    ss.pause_started_at = None

def do_finish():
    # close pause if needed
    if ss.paused and ss.pause_started_at:
        ss.paused_total += now() - ss.pause_started_at
        ss.paused = False
        ss.pause_started_at = None
    ss.finished = True
    ss.finished_at = now()

controls = st.container()
with controls:
    st.markdown("<div class='topbar'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1], vertical_alignment="center")
    with c1:
        if st.button("Start", use_container_width=True, disabled=ss.started and not ss.finished):
            do_start()
    with c2:
        if st.button("Pause", use_container_width=True, disabled=(not ss.started) or ss.finished or ss.paused):
            do_pause()
    with c3:
        if st.button("Resume", use_container_width=True, disabled=(not ss.started) or ss.finished or (not ss.paused)):
            do_resume()
    with c4:
        if st.button("Finish", use_container_width=True, disabled=(not ss.started) or ss.finished):
            do_finish()
    with c5:
        if st.button("Reset", use_container_width=True):
            reset_all()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Settings (simple, inline)
# -----------------------------
with st.expander("Opções do report (simples)"):
    streak_gap_s = st.slider("Streak: gap máximo entre OKs (s)", 1, 12, 3)
    window_s = st.select_slider("Pico: janela deslizante (s)", options=[5, 10, 15, 20, 30, 45, 60], value=10)

# -----------------------------
# Live KPIs (light)
# -----------------------------
elapsed = effective_elapsed(ss.t0, ss.finished_at, ss.paused_total, ss.paused, ss.pause_started_at)
oks = len(ss.ok_times)
rate = (oks / elapsed * 60) if elapsed > 0 else 0.0
state = "Pronto"
if ss.started and not ss.finished and not ss.paused:
    state = "A contar"
elif ss.paused and not ss.finished:
    state = "Pausado"
elif ss.finished:
    state = "Finalizado"

last_ok = "-"
if ss.ok_times:
    last_ok = pd.to_datetime(ss.ok_times[-1], unit="s").strftime("%H:%M:%S")

st.markdown(
    f"""
<div class="kpiwrap">
  <div class="kpi"><div class="label">Estado</div><div class="value">{state}</div></div>
  <div class="kpi"><div class="label">Tempo (sem pausas)</div><div class="value">{fmt_duration(elapsed)}</div></div>
  <div class="kpi"><div class="label">Total OK</div><div class="value">{oks}</div><div class="sub">Média: {rate:.2f} OK/min</div></div>
  <div class="kpi"><div class="label">Último OK</div><div class="value">{last_ok}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Big OK button
# -----------------------------
ok_disabled = (not ss.started) or ss.finished or ss.paused

center = st.columns([1, 2, 1])[1]
with center:
    st.markdown("<div class='okcard'>", unsafe_allow_html=True)
    pressed = st.button("✅ OK", disabled=ok_disabled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if pressed:
    ss.ok_times.append(now())

# Quick undo (optional, still simple)
u1, u2, u3 = st.columns([1, 1, 1])
with u2:
    undo_disabled = (not ss.started) or (len(ss.ok_times) == 0) or ss.finished
    if st.button("↩️ desfazer último OK", disabled=undo_disabled, use_container_width=True):
        if ss.ok_times:
            ss.ok_times.pop()

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Report (only after finish; compute once)
# -----------------------------
def compute_report():
    times = np.array(sorted(ss.ok_times), dtype=float)
    df = build_df(ss.t0, ss.ok_times)
    duration_s = effective_elapsed(ss.t0, ss.finished_at, ss.paused_total, False, None)  # finished

    total = int(times.size)
    avg_per_min = (total / duration_s * 60) if duration_s > 0 else 0.0

    intervals = np.diff(times) if times.size >= 2 else np.array([])
    median_gap = float(np.median(intervals)) if intervals.size else np.nan
    p90_gap = float(np.percentile(intervals, 90)) if intervals.size else np.nan

    pm = per_minute_counts(df)
    peak_per_min = int(pm["oks"].max()) if not pm.empty else 0

    peak_window = sliding_window_peak(times, float(window_s))
    max_streak = max_streak_by_gap(times, float(streak_gap_s))

    return {
        "df": df,
        "pm": pm,
        "intervals": intervals,
        "total": total,
        "duration_s": float(duration_s),
        "avg_per_min": float(avg_per_min),
        "median_gap": median_gap,
        "p90_gap": p90_gap,
        "peak_per_min": peak_per_min,
        "peak_window": peak_window,
        "max_streak": max_streak,
    }

st.subheader("Report")

if not ss.started:
    st.info("Carrega **Start** para começares.")
elif not ss.finished:
    st.info("Quando carregares **Finish**, aparece aqui o report com gráficos.")
else:
    if ss.report is None:
        ss.report = compute_report()
    r = ss.report

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total OK", r["total"])
    m2.metric("Tempo", fmt_duration(r["duration_s"]))
    m3.metric("Média", f"{r['avg_per_min']:.2f} OK/min")
    m4.metric("Pico por minuto", r["peak_per_min"])

    n1, n2, n3 = st.columns(3)
    n1.metric(f"Pico em {window_s}s", r["peak_window"])
    n2.metric(f"Max streak (<= {streak_gap_s}s)", r["max_streak"])
    n3.metric("Mediana intervalo", "-" if np.isnan(r["median_gap"]) else f"{r['median_gap']:.2f}s")

    tabs = st.tabs(["📈 Cumulativo", "📊 Por minuto", "⏱️ Intervalos", "🧾 Exportar"])

    with tabs[0]:
        df = r["df"]
        if df.empty:
            st.info("Sem dados.")
        else:
            cum = df.copy()
            cum["count"] = np.arange(1, len(cum) + 1)
            st.line_chart(cum.set_index("dt")[["count"]])

    with tabs[1]:
        pm = r["pm"]
        if pm.empty:
            st.info("Sem dados.")
        else:
            st.bar_chart(pm.set_index("minute")[["oks"]])

    with tabs[2]:
        intervals = r["intervals"]
        if intervals.size < 1:
            st.info("Precisas de pelo menos 2 OKs.")
        else:
            bins = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233], dtype=float)
            b = pd.cut(intervals, bins=bins, include_lowest=True)
            hist = b.value_counts().sort_index()
            hist_df = pd.DataFrame({"intervalo (s)": hist.index.astype(str), "contagem": hist.values}).set_index("intervalo (s)")
            st.bar_chart(hist_df)

    with tabs[3]:
        df = r["df"]
        if df.empty:
            st.info("Sem dados para exportar.")
        else:
            export_df = df.copy()
            export_df["timestamp_iso"] = export_df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
            export_df = export_df[["timestamp_iso", "t", "elapsed_s"]]
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="ok_counter_events.csv", mime="text/csv")
