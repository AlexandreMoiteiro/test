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


def effective_elapsed(t0: float | None, finished_at: float | None, paused_total: float,
                     paused: bool, pause_started_at: float | None) -> float:
    if not t0:
        return 0.0
    end = finished_at if finished_at else now()
    total = end - t0

    extra_pause = 0.0
    if paused and pause_started_at and not finished_at:
        extra_pause = end - pause_started_at

    return max(0.0, total - (paused_total + extra_pause))


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


def compute_report(t0: float, finished_at: float, paused_total: float, ok_times: list[float],
                   window_s: float, streak_gap_s: float):
    times = np.array(sorted(ok_times), dtype=float)
    duration_s = max(0.0, (finished_at - t0) - paused_total)
    total = int(times.size)
    avg_per_min = (total / duration_s * 60) if duration_s > 0 else 0.0

    df = build_df(t0, ok_times)

    intervals = np.diff(times) if times.size >= 2 else np.array([])
    median_gap = float(np.median(intervals)) if intervals.size else np.nan
    p90_gap = float(np.percentile(intervals, 90)) if intervals.size else np.nan

    pm = per_minute_counts(df)
    peak_per_min = int(pm["oks"].max()) if not pm.empty else 0
    peak_minute = int(pm.loc[pm["oks"].idxmax(), "minute"]) if not pm.empty else None

    peak_window = sliding_window_peak(times, float(window_s))
    max_streak = max_streak_by_gap(times, float(streak_gap_s))

    # Extra: densidade por bloco de 10s
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
        "peak_per_min": peak_per_min,
        "peak_minute": peak_minute,
        "peak_window": peak_window,
        "max_streak": max_streak,
        "peak_10s": peak_10s,
        "df": df,
        "pm": pm,
        "intervals": intervals,
        "window_s": float(window_s),
        "streak_gap_s": float(streak_gap_s),
    }


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
    ss.setdefault("report", None)

init_state()
ss = st.session_state


# -----------------------------
# Page + Style
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
<style>
.block-container { max-width: 1100px; padding-top: 1.2rem; }
h1 { margin-bottom: 0.2rem; }
.smallcap { opacity: 0.8; }

/* KPI cards */
.kpi {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
}
.kpi .label { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
.kpi .value { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }

/* BIG OK button */
.okwrap {
  padding: 18px;
  border-radius: 22px;
  background: radial-gradient(80% 120% at 20% 0%, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
}
div.stButton > button {
  width: 100%;
  height: 140px;
  border-radius: 28px;
  font-size: 52px;
  font-weight: 900;
  border: 1px solid rgba(255,255,255,0.18);
}

/* Make small buttons look consistent */
div[data-testid="stHorizontalBlock"] div.stButton > button {
  height: 44px;
  font-size: 16px;
  font-weight: 700;
  border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
st.title(APP_TITLE)
st.caption("Clica no botão sempre que alguém disser “ok”. Carrega Finish para gerar o report.")

# Controls row (simple)
c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1.2, 1.2])

if c1.button("▶️ Start", use_container_width=True, disabled=ss.started and not ss.finished):
    ss.started = True
    ss.finished = False
    ss.paused = False
    ss.t0 = now()
    ss.finished_at = None
    ss.pause_started_at = None
    ss.paused_total = 0.0
    ss.ok_times = []
    ss.report = None

pause_disabled = (not ss.started) or ss.finished or ss.paused
if c2.button("⏸️ Pause", use_container_width=True, disabled=pause_disabled):
    ss.paused = True
    ss.pause_started_at = now()

resume_disabled = (not ss.started) or ss.finished or (not ss.paused)
if c3.button("▶️ Resume", use_container_width=True, disabled=resume_disabled):
    if ss.pause_started_at:
        ss.paused_total += now() - ss.pause_started_at
    ss.paused = False
    ss.pause_started_at = None

finish_disabled = (not ss.started) or ss.finished
if c4.button("⏹️ Finish", use_container_width=True, disabled=finish_disabled):
    # close pause
    if ss.paused and ss.pause_started_at:
        ss.paused_total += now() - ss.pause_started_at
        ss.paused = False
        ss.pause_started_at = None
    ss.finished = True
    ss.finished_at = now()

# Settings right side (simple, inline)
window_s = c5.selectbox("Pico (janela s)", [5, 10, 15, 20, 30, 45, 60], index=1)
streak_gap_s = c6.slider("Streak gap (s)", 1, 12, 3)

# Reset at bottom of controls
if st.button("🧹 Reset", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.divider()


# -----------------------------
# Live KPIs (lightweight)
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

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi'><div class='label'>Estado</div><div class='value'>{state}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'><div class='label'>Tempo (sem pausas)</div><div class='value'>{fmt_duration(elapsed)}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'><div class='label'>Total OK</div><div class='value'>{oks}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi'><div class='label'>Média</div><div class='value'>{rate:.2f} OK/min</div></div>", unsafe_allow_html=True)

st.divider()


# -----------------------------
# BIG OK Button (centered)
# -----------------------------
ok_disabled = (not ss.started) or ss.finished or ss.paused

center = st.columns([1, 2.2, 1])[1]
with center:
    st.markdown("<div class='okwrap'>", unsafe_allow_html=True)
    pressed = st.button("✅ OK", use_container_width=True, disabled=ok_disabled)
    st.markdown("</div>", unsafe_allow_html=True)

if pressed:
    ss.ok_times.append(now())
    # Sem st.rerun() — re-execução acontece de qualquer forma,
    # mas o app continua leve e sem gráficos pesados durante a contagem.

# Small helper line
if ss.ok_times:
    last = pd.to_datetime(ss.ok_times[-1], unit="s").strftime("%H:%M:%S")
    st.caption(f"Último OK: {last}")


# -----------------------------
# Report (only after Finish; compute once)
# -----------------------------
st.subheader("Report")

if not ss.started:
    st.info("Carrega **Start** para começar.")
elif not ss.finished:
    st.info("Quando carregares **Finish**, o report aparece aqui.")
else:
    if ss.report is None:
        ss.report = compute_report(
            t0=ss.t0,
            finished_at=ss.finished_at,
            paused_total=ss.paused_total,
            ok_times=ss.ok_times,
            window_s=float(window_s),
            streak_gap_s=float(streak_gap_s),
        )

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
    b1.metric(f"Pico em {int(r['window_s'])}s", r["peak_window"])
    b2.metric(f"Max streak (≤ {int(r['streak_gap_s'])}s)", r["max_streak"])
    b3.metric("Mediana intervalo", "-" if np.isnan(r["median_gap"]) else f"{r['median_gap']:.2f}s")
    b4.metric("P90 intervalo", "-" if np.isnan(r["p90_gap"]) else f"{r['p90_gap']:.2f}s")

    st.caption(f"Extra: maior densidade em blocos de 10s = **{r['peak_10s']} OKs**.")

    tabs = st.tabs(["📈 Cumulativo", "📊 Por minuto", "⏱️ Intervalos", "🧾 Exportar"])

    with tabs[0]:
        if df.empty:
            st.info("Sem dados.")
        else:
            cum = df.copy()
            cum["count"] = np.arange(1, len(cum) + 1)
            st.line_chart(cum.set_index("dt")[["count"]])

    with tabs[1]:
        if pm.empty:
            st.info("Sem dados.")
        else:
            st.bar_chart(pm.set_index("minute")[["oks"]])

    with tabs[2]:
        if intervals.size < 1:
            st.info("Precisas de pelo menos 2 OKs.")
        else:
            bins = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233], dtype=float)
            b = pd.cut(intervals, bins=bins, include_lowest=True)
            hist = b.value_counts().sort_index()
            hist_df = pd.DataFrame(
                {"intervalo (s)": hist.index.astype(str), "contagem": hist.values}
            ).set_index("intervalo (s)")
            st.bar_chart(hist_df)

    with tabs[3]:
        if df.empty:
            st.info("Sem dados para exportar.")
        else:
            export_df = df.copy()
            export_df["timestamp_iso"] = export_df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
            export_df = export_df[["timestamp_iso", "t", "elapsed_s"]]
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="ok_counter_events.csv", mime="text/csv")
