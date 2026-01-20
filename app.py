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


def sliding_window_peak(times: np.ndarray, window_s: float) -> int:
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


def build_df(t0: float, ok_times: list[float]) -> pd.DataFrame:
    if not ok_times or not t0:
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


def effective_elapsed(t0: float, finished_at: float | None, paused_total: float, paused: bool, pause_started_at: float | None) -> float:
    if not t0:
        return 0.0
    end = finished_at if finished_at else now()
    total = end - t0

    extra_pause = 0.0
    if paused and pause_started_at and not finished_at:
        extra_pause = end - pause_started_at

    return max(0.0, total - (paused_total + extra_pause))


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

    # Report computed only when finishing
    ss.setdefault("report", None)

init_state()
ss = st.session_state


# -----------------------------
# Page config + CSS (UI bonita)
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
<style>
/* layout */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
h1 { margin-bottom: 0.25rem; }

/* top badge */
.badge {
  display:inline-block; padding:6px 10px; border-radius:999px;
  background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
  font-size: 0.9rem;
}

/* KPI cards */
.kpi {
  padding: 16px 16px;
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
}
.kpi .label { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
.kpi .value { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
.kpi .sub { font-size: 0.85rem; opacity: 0.75; margin-top: 6px; }

/* BIG OK button */
div.stButton > button {
  width: 100%;
  height: 110px;
  border-radius: 24px;
  font-size: 40px;
  font-weight: 800;
  border: 1px solid rgba(255,255,255,0.18);
}
.okwrap {
  padding: 18px;
  border-radius: 22px;
  background: radial-gradient(80% 120% at 20% 0%, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
}

/* subtle separators */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
left, right = st.columns([3, 2], vertical_alignment="center")
with left:
    st.title(APP_TITLE)
    st.caption("Clica no botão sempre que alguém disser “ok”. No fim, gera um report com estatísticas e gráficos.")
with right:
    # show state badge
    state = "Pronto"
    if ss.started and not ss.finished and not ss.paused:
        state = "A contar"
    elif ss.paused and not ss.finished:
        state = "Pausado"
    elif ss.finished:
        state = "Finalizado"
    st.markdown(f"<span class='badge'>Estado: <b>{state}</b></span>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Sessão")

    if not ss.started:
        if st.button("▶️ Start", use_container_width=True):
            ss.started = True
            ss.finished = False
            ss.paused = False
            ss.t0 = now()
            ss.finished_at = None
            ss.pause_started_at = None
            ss.paused_total = 0.0
            ss.ok_times = []
            ss.report = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            if not ss.finished:
                if not ss.paused:
                    if st.button("⏸️ Pause", use_container_width=True):
                        ss.paused = True
                        ss.pause_started_at = now()
                else:
                    if st.button("▶️ Resume", use_container_width=True):
                        if ss.pause_started_at:
                            ss.paused_total += now() - ss.pause_started_at
                        ss.paused = False
                        ss.pause_started_at = None
        with c2:
            if not ss.finished:
                if st.button("⏹️ Finish", use_container_width=True):
                    # close pause if needed
                    if ss.paused and ss.pause_started_at:
                        ss.paused_total += now() - ss.pause_started_at
                        ss.paused = False
                        ss.pause_started_at = None
                    ss.finished = True
                    ss.finished_at = now()

    st.divider()
    st.header("Métricas (report)")
    streak_gap_s = st.slider("Streak (gap máx em s)", 1, 12, 3)
    window_s = st.select_slider("Pico (janela em s)", options=[5, 10, 15, 20, 30, 45, 60], value=10)

    st.divider()
    if st.button("🧹 Reset total", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# -----------------------------
# Main live KPIs (lightweight)
# -----------------------------
elapsed = effective_elapsed(ss.t0, ss.finished_at, ss.paused_total, ss.paused, ss.pause_started_at)
oks = len(ss.ok_times)
rate = (oks / elapsed * 60) if elapsed > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi'><div class='label'>Tempo (sem pausas)</div><div class='value'>{fmt_duration(elapsed)}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>Total OK</div><div class='value'>{oks}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Média</div><div class='value'>{rate:.2f}</div><div class='sub'>OK/min</div></div>", unsafe_allow_html=True)
with k4:
    last = "-"
    if ss.ok_times:
        last = pd.to_datetime(ss.ok_times[-1], unit="s").strftime("%H:%M:%S")
    st.markdown(f"<div class='kpi'><div class='label'>Último OK</div><div class='value'>{last}</div></div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


# -----------------------------
# BIG OK button (no st.rerun)
# -----------------------------
ok_disabled = (not ss.started) or ss.finished or ss.paused

center = st.columns([1, 2, 1])[1]
with center:
    st.markdown("<div class='okwrap'>", unsafe_allow_html=True)
    pressed = st.button("✅ OK", disabled=ok_disabled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if pressed:
    ss.ok_times.append(now())
    # Sem st.rerun(): o Streamlit vai reexecutar sozinho (inevitável),
    # mas o script é leve e não recalcula gráficos pesados em cada clique.


# -----------------------------
# Report (compute ONLY after Finish, once)
# -----------------------------
def compute_report():
    times = np.array(sorted(ss.ok_times), dtype=float)
    duration_s = effective_elapsed(ss.t0, ss.finished_at, ss.paused_total, False, None)  # finished: no live pause
    total = int(times.size)
    avg_per_min = (total / duration_s * 60) if duration_s > 0 else 0.0

    df = build_df(ss.t0, ss.ok_times)

    intervals = np.diff(times) if times.size >= 2 else np.array([])
    median_gap = float(np.median(intervals)) if intervals.size else np.nan
    p90_gap = float(np.percentile(intervals, 90)) if intervals.size else np.nan
    min_gap = float(np.min(intervals)) if intervals.size else np.nan

    pm = per_minute_counts(df)
    peak_per_min = int(pm["oks"].max()) if not pm.empty else 0
    peak_minute = int(pm.loc[pm["oks"].idxmax(), "minute"]) if not pm.empty else None

    peak_window = sliding_window_peak(times, float(window_s))
    max_streak = max_streak_by_gap(times, float(streak_gap_s))

    # extra: peak in 10s buckets
    peak_10s = 0
    if not df.empty:
        b10 = (df["elapsed_s"] // 10).astype(int)
        c10 = b10.value_counts()
        peak_10s = int(c10.max()) if not c10.empty else 0

    return {
        "total": total,
        "duration_s": float(duration_s),
        "avg_per_min": float(avg_per_min),
        "median_gap": median_gap,
        "p90_gap": p90_gap,
        "min_gap": min_gap,
        "peak_per_min": peak_per_min,
        "peak_minute": peak_minute,
        "peak_window": peak_window,
        "max_streak": max_streak,
        "peak_10s": peak_10s,
        "df": df,
        "pm": pm,
        "intervals": intervals,
    }


st.subheader("Report")

if not ss.started:
    st.info("Carrega **Start** para começar.")
elif oks == 0 and not ss.finished:
    st.warning("Ainda não há OKs registados.")
elif ss.finished:
    # compute once
    if ss.report is None:
        ss.report = compute_report()

    r = ss.report
    df = r["df"]
    pm = r["pm"]
    intervals = r["intervals"]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total OK", r["total"])
    a2.metric("Tempo", fmt_duration(r["duration_s"]))
    a3.metric("Média", f"{r['avg_per_min']:.2f} OK/min")
    if r["peak_minute"] is None:
        a4.metric("Pico por minuto", r["peak_per_min"])
    else:
        a4.metric("Pico por minuto", r["peak_per_min"], help=f"Minuto #{r['peak_minute']} desde o início")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric(f"Pico em {window_s}s", r["peak_window"])
    b2.metric(f"Max streak (≤ {streak_gap_s}s)", r["max_streak"])
    b3.metric("Mediana intervalo", "-" if np.isnan(r["median_gap"]) else f"{r['median_gap']:.2f}s")
    b4.metric("P90 intervalo", "-" if np.isnan(r["p90_gap"]) else f"{r['p90_gap']:.2f}s")

    st.caption(
        f"Extra: maior densidade em blocos de 10s = **{r['peak_10s']} OKs**. "
        "A streak considera OKs consecutivos com intervalo <= o gap escolhido."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Cumulativo", "📊 Por minuto", "⏱️ Intervalos", "🧾 Exportar"])

    with tab1:
        if df.empty:
            st.info("Sem dados.")
        else:
            cum = df.copy()
            cum["count"] = np.arange(1, len(cum) + 1)
            st.line_chart(cum.set_index("dt")[["count"]])

    with tab2:
        if pm.empty:
            st.info("Sem dados.")
        else:
            st.bar_chart(pm.set_index("minute")[["oks"]])

    with tab3:
        if intervals.size < 1:
            st.info("Precisas de pelo menos 2 OKs.")
        else:
            bins = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233], dtype=float)
            b = pd.cut(intervals, bins=bins, include_lowest=True)
            hist = b.value_counts().sort_index()
            hist_df = pd.DataFrame({"intervalo (s)": hist.index.astype(str), "contagem": hist.values}).set_index("intervalo (s)")
            st.bar_chart(hist_df)

    with tab4:
        if df.empty:
            st.info("Sem dados para exportar.")
        else:
            export_df = df.copy()
            export_df["timestamp_iso"] = export_df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
            export_df = export_df[["timestamp_iso", "t", "elapsed_s"]]
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="ok_counter_events.csv", mime="text/csv")

else:
    st.info("Quando carregares **Finish**, o report aparece aqui (com gráficos e estatísticas).")
