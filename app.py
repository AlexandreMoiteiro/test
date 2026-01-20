import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Config
# -------------------------
APP_TITLE = "ok counter"


# -------------------------
# Utils
# -------------------------
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
    """Max events inside any window of length window_s (two pointers)."""
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


def per_minute_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per minute from start (raw, not pause-adjusted per-event)."""
    if df.empty:
        return pd.DataFrame(columns=["minute", "oks"])
    minute = (df["elapsed_s"] // 60).astype(int)
    s = minute.value_counts().sort_index()
    out = pd.DataFrame({"minute": s.index.values, "oks": s.values})
    return out


# -------------------------
# State
# -------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("started", False)
    ss.setdefault("finished", False)
    ss.setdefault("paused", False)
    ss.setdefault("t0", None)  # epoch
    ss.setdefault("t_finish", None)
    ss.setdefault("pause_started_at", None)
    ss.setdefault("paused_total", 0.0)
    ss.setdefault("ok_times", [])  # list[float]


def effective_elapsed(end_ts: float) -> float:
    """Elapsed excluding pauses (session-level)."""
    ss = st.session_state
    if not ss.t0:
        return 0.0
    end = ss.t_finish if ss.finished else end_ts
    elapsed = end - ss.t0

    paused_total = ss.paused_total
    if ss.paused and ss.pause_started_at and not ss.finished:
        paused_total += (end_ts - ss.pause_started_at)
    return max(0.0, elapsed - paused_total)


def build_df() -> pd.DataFrame:
    ss = st.session_state
    if not ss.ok_times or not ss.t0:
        return pd.DataFrame(columns=["t", "dt", "elapsed_s"])
    df = pd.DataFrame({"t": sorted(ss.ok_times)})
    df["dt"] = pd.to_datetime(df["t"], unit="s")
    df["elapsed_s"] = df["t"] - ss.t0
    return df


def hard_reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()
ss = st.session_state

st.title(APP_TITLE)

with st.sidebar:
    st.header("Controlo da sessão")

    if not ss.started:
        if st.button("▶️ Start", use_container_width=True):
            ss.started = True
            ss.finished = False
            ss.paused = False
            ss.t0 = now()
            ss.t_finish = None
            ss.pause_started_at = None
            ss.paused_total = 0.0
            ss.ok_times = []
            st.rerun()
    else:
        col1, col2 = st.columns(2)

        with col1:
            if not ss.finished:
                if not ss.paused:
                    if st.button("⏸️ Pause", use_container_width=True):
                        ss.paused = True
                        ss.pause_started_at = now()
                        st.rerun()
                else:
                    if st.button("▶️ Resume", use_container_width=True):
                        if ss.pause_started_at:
                            ss.paused_total += now() - ss.pause_started_at
                        ss.paused = False
                        ss.pause_started_at = None
                        st.rerun()

        with col2:
            if not ss.finished:
                if st.button("⏹️ Finish", use_container_width=True):
                    # close pause if any
                    if ss.paused and ss.pause_started_at:
                        ss.paused_total += now() - ss.pause_started_at
                        ss.paused = False
                        ss.pause_started_at = None
                    ss.finished = True
                    ss.t_finish = now()
                    st.rerun()

    st.divider()
    st.header("Métricas avançadas")

    streak_gap_s = st.slider(
        "Streak: OKs a no máximo X segundos",
        min_value=1,
        max_value=12,
        value=3,
        help="Se o intervalo entre OKs for <= X segundos, conta como parte da mesma streak.",
    )

    window_s = st.select_slider(
        "Pico: janela deslizante (segundos)",
        options=[5, 10, 15, 20, 30, 45, 60],
        value=10,
        help="Máximo de OKs em qualquer intervalo contínuo desta duração.",
    )

    st.divider()
    if st.button("🧹 Reset total", use_container_width=True):
        hard_reset()

# Top metrics
t_now = now()
elapsed = effective_elapsed(t_now)
oks = len(ss.ok_times)
rate = (oks / elapsed * 60) if elapsed > 0 else 0.0

state_label = "Pronto"
if ss.started and not ss.finished and not ss.paused:
    state_label = "A contar"
elif ss.paused and not ss.finished:
    state_label = "Pausado"
elif ss.finished:
    state_label = "Finalizado"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Estado", state_label)
m2.metric("Tempo (sem pausas)", fmt_duration(elapsed))
m3.metric("Total OKs", f"{oks}")
m4.metric("Média (OK/min)", f"{rate:.2f}")

st.divider()

# OK button
ok_disabled = (not ss.started) or ss.finished or ss.paused
left, right = st.columns([1, 2])
with left:
    if st.button("✅ OK", use_container_width=True, disabled=ok_disabled):
        ss.ok_times.append(now())
        st.rerun()
with right:
    st.write(
        "Clica **OK** sempre que alguém disser “ok”. "
        "Usa **Pause** para parar sem terminar a sessão e **Finish** para gerar o report final."
    )

# Optional undo
undo_disabled = (not ss.started) or (oks == 0) or ss.finished
if st.button("↩️ Desfazer último OK", disabled=undo_disabled):
    if ss.ok_times:
        ss.ok_times.pop()
    st.rerun()

df = build_df()

st.divider()
st.subheader("Report")

if not ss.started:
    st.info("Carrega **Start** para começar.")
elif oks == 0:
    st.warning("Ainda não há OKs registados.")
else:
    times = np.array(sorted(ss.ok_times), dtype=float)
    duration_s = effective_elapsed(now())
    avg_per_min = (oks / duration_s * 60) if duration_s > 0 else 0.0

    # intervals
    intervals = np.diff(times) if times.size >= 2 else np.array([])
    median_gap = float(np.median(intervals)) if intervals.size else np.nan
    p90_gap = float(np.percentile(intervals, 90)) if intervals.size else np.nan
    min_gap = float(np.min(intervals)) if intervals.size else np.nan

    # per minute
    pm = per_minute_counts(df)
    peak_per_min = int(pm["oks"].max()) if not pm.empty else 0
    peak_minute = int(pm.loc[pm["oks"].idxmax(), "minute"]) if not pm.empty else None

    # short window peak + streak
    peak_window = sliding_window_peak(times, float(window_s))
    max_streak = max_streak_by_gap(times, float(streak_gap_s))

    # OKs per 10s buckets (extra stat)
    if not df.empty:
        bucket_10s = (df["elapsed_s"] // 10).astype(int)
        bucket_counts = bucket_10s.value_counts().sort_index()
        peak_10s_bucket = int(bucket_counts.max()) if not bucket_counts.empty else 0
    else:
        peak_10s_bucket = 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total", f"{oks}")
    k2.metric("Tempo", fmt_duration(duration_s))
    k3.metric("Média", f"{avg_per_min:.2f} OK/min")
    if peak_minute is None:
        k4.metric("Pico por minuto", f"{peak_per_min}")
    else:
        k4.metric("Pico por minuto", f"{peak_per_min}", help=f"Minuto #{peak_minute} desde o início")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric(f"Pico em {window_s}s", f"{peak_window}")
    k6.metric(f"Max streak (<= {streak_gap_s}s)", f"{max_streak}")
    k7.metric("Mediana intervalo", "-" if np.isnan(median_gap) else f"{median_gap:.2f}s")
    k8.metric("P90 intervalo", "-" if np.isnan(p90_gap) else f"{p90_gap:.2f}s")

    st.caption(
        f"Extra: maior densidade em blocos de 10s = **{peak_10s_bucket} OKs**. "
        "As métricas de taxa usam o tempo **sem pausas**."
    )

    st.divider()

    # Charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Cumulativo ao longo do tempo")
        cum = df.copy()
        cum["count"] = np.arange(1, len(cum) + 1)
        st.line_chart(cum.set_index("dt")[["count"]])

    with c2:
        st.markdown("#### OKs por minuto")
        if pm.empty:
            st.info("Sem dados suficientes.")
        else:
            st.bar_chart(pm.set_index("minute")[["oks"]])

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Distribuição de intervalos entre OKs")
        if intervals.size == 0:
            st.info("Precisas de pelo menos 2 OKs.")
        else:
            # Simple histogram via binning (em segundos)
            bins = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233], dtype=float)
            b = pd.cut(intervals, bins=bins, include_lowest=True)
            hist = b.value_counts().sort_index()
            hist_df = pd.DataFrame({"intervalo (s)": hist.index.astype(str), "contagem": hist.values}).set_index("intervalo (s)")
            st.bar_chart(hist_df)

    with c4:
        st.markdown("#### OKs dentro do minuto (segundo 0–59)")
        sec = df["dt"].dt.second
        sec_counts = sec.value_counts().sort_index()
        sec_df = pd.DataFrame({"segundo": sec_counts.index.astype(int), "oks": sec_counts.values}).set_index("segundo")
        st.bar_chart(sec_df)

    st.divider()
    st.markdown("### Exportar")
    export_df = df.copy()
    export_df["timestamp_iso"] = export_df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export_df = export_df[["timestamp_iso", "t", "elapsed_s"]]
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="ok_counter_events.csv", mime="text/csv")

    st.caption(
        "Definições: "
        f"**streak_gap={streak_gap_s}s** (intervalo máximo para continuar a streak) e "
        f"**janela_pico={window_s}s** (pico em janela deslizante)."
    )

