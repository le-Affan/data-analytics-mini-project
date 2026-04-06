import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, norm, skew, kurtosis as scipy_kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background: transparent !important;
}

/* ── Full-page dark gradient ── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%) !important;
    min-height: 100vh;
}

/* ── Hide sidebar toggle & default padding ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem 2rem !important;
    margin: 0 auto !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.4); border-radius: 3px; }

/* ─────────────────────────────────────────
   HERO HEADER
───────────────────────────────────────── */
.hero-wrap {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2.8rem 2.5rem 2.4rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    animation: fadeDown 0.7s ease both;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(102,126,234,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(118,75,162,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 0.5rem;
    letter-spacing: -0.8px;
    line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0 0 1.2rem;
    font-weight: 400;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(102,126,234,0.15);
    border: 1px solid rgba(102,126,234,0.35);
    border-radius: 50px;
    padding: 5px 16px;
    font-size: 0.78rem;
    color: #a5b4fc;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.hero-line {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 2px;
    margin: 1rem 0;
}

/* ─────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 1.4rem 1.5rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 18px 18px 0 0;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 40px rgba(102,126,234,0.2), inset 0 1px 0 rgba(255,255,255,0.08);
    border-color: rgba(102,126,234,0.3) !important;
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* ─────────────────────────────────────────
   SECTION CARDS (glassmorphism)
───────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 30px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    animation: fadeUp 0.6s ease both;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 48px rgba(102,126,234,0.12), inset 0 1px 0 rgba(255,255,255,0.08);
    border-color: rgba(102,126,234,0.18);
}

/* Section header row */
.sec-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
}
.sec-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: 0.8px;
    flex-shrink: 0;
}
.sec-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0;
    line-height: 1.3;
    letter-spacing: -0.5px;
}
.sec-desc {
    color: #cbd5e1;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* ─────────────────────────────────────────
   STAT LIST
───────────────────────────────────────── */
.stat-item {
    display: flex;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.95rem;
    color: #94a3b8;
}
.stat-item:last-child { border-bottom: none; }
.stat-val {
    color: #e2e8f0;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* ─────────────────────────────────────────
   INTERPRETATION BOX
───────────────────────────────────────── */
.interp-box {
    background: rgba(102,126,234,0.1);
    border-left: 4px solid #818cf8;
    border-radius: 0 16px 16px 0;
    padding: 1.4rem 1.6rem;
    margin-top: 1.5rem;
    font-size: 1.05rem;
    color: #f1f5f9;
    line-height: 1.7;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}
.interp-box b { color: #c084fc; font-weight: 800; }

/* ─────────────────────────────────────────
   TABLE
───────────────────────────────────────── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}
.styled-table th {
    background: rgba(102,126,234,0.15);
    color: #a5b4fc;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    font-weight: 600;
    letter-spacing: 0.3px;
}
.styled-table td {
    padding: 7px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #94a3b8;
}
.styled-table tr:hover td { background: rgba(255,255,255,0.03); }

/* ─────────────────────────────────────────
   CHART CARD WRAPPER
───────────────────────────────────────── */
.chart-shell {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1rem 0.5rem 0.5rem;
    margin-bottom: 0.5rem;
}

/* ─────────────────────────────────────────
   DIVIDER
───────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    margin: 2rem 0 !important;
}

/* ─────────────────────────────────────────
   ANIMATIONS
───────────────────────────────────────── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* staggered card animations */
.glass-card:nth-child(1) { animation-delay: 0.05s; }
.glass-card:nth-child(2) { animation-delay: 0.12s; }
.glass-card:nth-child(3) { animation-delay: 0.19s; }
.glass-card:nth-child(4) { animation-delay: 0.26s; }
.glass-card:nth-child(5) { animation-delay: 0.33s; }
.glass-card:nth-child(6) { animation-delay: 0.40s; }

/* ─────────────────────────────────────────
   METRICS ROW WRAPPER
───────────────────────────────────────── */
.metrics-row {
    animation: fadeUp 0.65s 0.1s ease both;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─── PARTICLE BACKGROUND ──────────────────────────────────────────────────────
components.html("""
<style>
  #particle-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
  }
</style>
<canvas id="particle-canvas"></canvas>
<script>
(function () {
  const canvas = document.getElementById('particle-canvas');
  const ctx    = canvas.getContext('2d');
  let W, H;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  const COUNT = 60;
  const particles = Array.from({ length: COUNT }, () => ({
    x:  Math.random() * W,
    y:  Math.random() * H,
    r:  Math.random() * 1.6 + 0.4,
    vx: (Math.random() - 0.5) * 0.22,
    vy: (Math.random() - 0.5) * 0.22,
    o:  Math.random() * 0.25 + 0.06,
  }));

  /* connect nearby particles */
  function drawLines() {
    for (let i = 0; i < COUNT; i++) {
      for (let j = i + 1; j < COUNT; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(102,126,234,${0.07 * (1 - d / 130)})`;
          ctx.lineWidth   = 0.6;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(148,163,184,${p.o})`;
      ctx.fill();
    }

    drawLines();
    requestAnimationFrame(loop);
  }

  loop();
})();
</script>
""", height=0, scrolling=False)

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.5)",
    font=dict(family="Inter", color="rgba(148,163,184,0.9)", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    colorway=["#667eea","#f59e0b","#34d399","#f87171","#a78bfa","#38bdf8"],
    hoverlabel=dict(bgcolor="#1e293b", font_color="#e2e8f0", bordercolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
)
CLUSTER_COLOR_MAP = {"0":"#667eea","1":"#f59e0b","2":"#34d399"}

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    act_path = "Data/Selected_Data/dailyActivity_merged.csv"
    slp_path = "Data/Selected_Data/sleepDay_merged.csv"
    if not os.path.exists(act_path) or not os.path.exists(slp_path):
        return None, "Data files not found."

    try:
        activity = pd.read_csv(act_path)
        sleep    = pd.read_csv(slp_path)

        activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"], format="mixed").dt.date
        sleep["ActivityDate"]    = pd.to_datetime(sleep["ActivityDate"],    format="mixed").dt.date

        df = pd.merge(activity, sleep, on=["Id","ActivityDate"], how="left")
        df["SleepMinutes"] = df["SleepMinutes"].fillna(0)

        df = df.drop_duplicates()
        df = df[df["TotalSteps"] >= 0]
        df = df[df["Calories"] > 0]
        df["TotalActiveMinutes"] = df["VeryActiveMinutes"] + df["FairlyActiveMinutes"] + df["LightlyActiveMinutes"]
        df["ActivityRatio"]      = df["TotalActiveMinutes"] / (df["SedentaryMinutes"] + 1)

        X = df[["TotalActiveMinutes","Calories"]].fillna(0)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans   = KMeans(n_clusters=3, random_state=42, n_init="auto")
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
        return df, None
    except Exception as e:
        return None, str(e)

df_raw, load_err = load_data()
if df_raw is None:
    st.error(f"Data Loading Failed: {load_err}")
    st.stop()

# No sidebar filters — use full dataset
df = df_raw.copy()
n_rows, n_cols = df.shape

# ─── HELPER: section card opener / closer ─────────────────────────────────────
def sec_open(badge, title, desc=""):
    desc_html = f"<div class='sec-desc' style='margin-bottom: 1rem;'>{desc}</div>" if desc else ""
    st.markdown(f"""
    <div class="glass-card" style="padding: 1.5rem 2rem 1.2rem;">
      <div class="sec-header" style="margin-bottom: 0.2rem;">
        <span class="sec-badge">{badge}</span>
        <span class="sec-title" style="font-size: 1.5rem;">{title}</span>
      </div>
      {desc_html}
    """, unsafe_allow_html=True)

def sec_close():
    st.markdown("</div>", unsafe_allow_html=True)

# ─── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap" style="padding: 3.5rem 2.5rem; text-align: center;">
  <p class="hero-title" style="font-size: 3.6rem; letter-spacing: -1.2px;">Fitness Analytics <span>Dashboard</span></p>
  <p class="hero-sub" style="font-size: 1.15rem; color: #cbd5e1; font-weight: 500; margin-bottom: 2rem;">Comprehensive Analysis of Activity, Calories, and Health Patterns</p>
  <div style="max-width: 850px; margin: 0 auto; color: #94a3b8; font-size: 0.98rem; line-height: 1.7;">
    This project analyzes real-world Fitbit fitness tracker data to understand activity patterns, calorie expenditure, sleep behavior, and user fitness segmentation using statistical analysis and machine learning techniques.
  </div>
</div>
""", unsafe_allow_html=True)

# ─── INFO CARDS ──────────────────────────────────────────────────────────────
info_col1, info_col2 = st.columns(2, gap="large")

with info_col1:
    st.markdown(f"""
    <div class="glass-card" style="padding: 1.8rem; height: 100%; margin-bottom: 1rem;">
      <div style="font-size: 1.1rem; font-weight: 600; color: #c7d2fe; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem;">
        <span style="background: linear-gradient(135deg, #667eea, #764ba2); width: 4px; height: 1.1rem; border-radius: 2px;"></span>
        Dataset Information
      </div>
      <div class='stat-item'><span>Dataset Name</span><span class='stat-val'>Fitbit Fitness Tracker Data</span></div>
      <div class='stat-item'><span>Source</span><span class='stat-val'><a href="https://www.kaggle.com/datasets/arashnic/fitbit" target="_blank" style="color: #818cf8; text-decoration: none;">kaggle.com/datasets/arashnic/fitbit</a></span></div>
      <div class='stat-item'><span>Records</span><span class='stat-val'>{n_rows} rows</span></div>
      <div class='stat-item'><span>Features</span><span class='stat-val'>{n_cols} columns</span></div>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="glass-card" style="padding: 1.8rem; height: 100%; margin-bottom: 1rem;">
      <div style="font-size: 1.1rem; font-weight: 600; color: #c7d2fe; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem;">
        <span style="background: linear-gradient(135deg, #667eea, #764ba2); width: 4px; height: 1.1rem; border-radius: 2px;"></span>
        Student Details
      </div>
      <div class='stat-item'><span>Name</span><span class='stat-val'>Affan Shaikh</span></div>
      <div class='stat-item'><span>Roll Number</span><span class='stat-val'>16014223006</span></div>
      <div class='stat-item'><span>Program</span><span class='stat-val'>TY BTech AI & DS</span></div>
      <div class='stat-item'><span>Batch</span><span class='stat-val'>B1</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─── DATASET DESCRIPTION ──────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="padding: 1.8rem; margin-bottom: 3rem;">
  <div style="font-size: 1.15rem; font-weight: 600; color: #c7d2fe; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: linear-gradient(135deg, #667eea, #764ba2); width: 4px; height: 1.15rem; border-radius: 2px;"></span>
    About the Dataset
  </div>
  <div style="color: #94a3b8; font-size: 0.95rem; line-height: 1.7;">
    The dataset contains activity, calorie, and sleep tracking data collected from Fitbit users. It includes daily metrics such as steps taken, activity intensity, sedentary time, and sleep duration. This data enables analysis of user behavior patterns and fitness levels.
  </div>
</div>
""", unsafe_allow_html=True)

# ─── EXP 1 — BOX PLOT ─────────────────────────────────────────────────────────
sec_open(
    "EXP 1", 
    "Calorie Distribution &amp; Outlier Analysis",
    "This experiment uses box plot analysis to understand the distribution of calorie expenditure and identify statistical outliers using quartiles and IQR."
)

q1  = df["Calories"].quantile(0.25)
med = df["Calories"].median()
q3  = df["Calories"].quantile(0.75)
iqr = q3 - q1
lb  = q1 - 1.5 * iqr
ub  = q3 + 1.5 * iqr
outliers = df[(df["Calories"] < lb) | (df["Calories"] > ub)]["Calories"]

col_l, col_r = st.columns([2, 1], gap="large")
with col_l:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    fig_box = px.box(df, y="Calories", points="outliers",
                     color_discrete_sequence=["#667eea"])
    fig_box.update_traces(marker_color="#f59e0b", marker_size=5, line_color="#667eea")
    fig_box.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown(f"""
    <div class='stat-item'><span>Lower Quartile (Q1)</span><span class='stat-val'>{q1:.1f}</span></div>
    <div class='stat-item'><span>Median</span><span class='stat-val'>{med:.1f}</span></div>
    <div class='stat-item'><span>Upper Quartile (Q3)</span><span class='stat-val'>{q3:.1f}</span></div>
    <div class='stat-item'><span>IQR</span><span class='stat-val'>{iqr:.1f}</span></div>
    <div class='stat-item'><span>Lower Bound</span><span class='stat-val'>{lb:.2f}</span></div>
    <div class='stat-item'><span>Upper Bound</span><span class='stat-val'>{ub:.2f}</span></div>
    <div class='stat-item' style='margin-top: 0.5rem;'><span style='font-weight: 700; color: #f1f5f9;'>Outlier Count</span><span class='stat-val' style='font-size: 1.15rem; color: #f87171;'>{len(outliers)}</span></div>
    
    <div class='interp-box'>
      <b>Interpretation:</b> Calorie expenditure shows notable variability with several extreme
      values, indicating diverse activity levels among users.
    </div>
    """, unsafe_allow_html=True)

sec_close()

# ─── EXP 2 — LINEAR REGRESSION ────────────────────────────────────────────────
sec_open(
    "EXP 2", 
    "Linear Regression Analysis",
    "Relationship between Total Active Minutes and Calories Burned"
)

slope_full, intercept_full, r_full, _, _ = linregress(df_raw["TotalActiveMinutes"], df_raw["Calories"])
r2_full = r_full ** 2

col_l2, col_r2 = st.columns([2, 1], gap="large")
with col_l2:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    fig_reg = px.scatter(df, x="TotalActiveMinutes", y="Calories",
                         trendline="ols", trendline_color_override="#f59e0b",
                         color_discrete_sequence=["#667eea"])
    fig_reg.update_traces(marker=dict(opacity=0.8, size=6), selector=dict(mode='markers'))
    fig_reg.update_traces(line=dict(width=4), selector=dict(mode='lines'))
    fig_reg.update_layout(**PLOTLY_LAYOUT,
                          xaxis_title="Total Active Minutes",
                          yaxis_title="Calories")
    st.plotly_chart(fig_reg, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r2:
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.4rem; margin-bottom: 1.8rem; text-align: center; font-family: 'Courier New', Courier, monospace; box-shadow: inset 0 2px 10px rgba(0,0,0,0.3);">
      <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'Inter', sans-serif;">Model Equation</div>
      <div style="color: #e2e8f0; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.5px;">
        Calories &approx; {intercept_full:.1f} &plus; ({slope_full:.2f} &times; Activity)
      </div>
    </div>
    
    <div class='stat-item'><span>Base Calories (Intercept)</span><span class='stat-val'>&#8776; {intercept_full:.1f}</span></div>
    <div class='stat-item'><span>Calories per Active Minute</span><span class='stat-val'>&#8776; {slope_full:.2f}</span></div>
    <div class='stat-item' style='border-bottom: none;'><span>Model Fit (R&sup2;)</span><span class='stat-val'>&#8776; {r2_full:.4f}</span></div>
    <div style="font-size: 0.82rem; color: #64748b; margin-top: -0.3rem; margin-bottom: 2rem; padding-bottom: 0.5rem; text-align: right;">Explains how well activity predicts calorie expenditure</div>

    <div class='interp-box' style='font-size: 1.1rem; border-left: 4px solid #fca5a5;'>
      <b>Interpretation:</b> There is a <strong style="color: #fca5a5; font-weight: 800;">weak positive relationship</strong> between total active minutes
      and calories burned, suggesting additional factors also influence calorie expenditure.
    </div>
    """, unsafe_allow_html=True)

sec_close()

# ─── EXP 3 — SAMPLING ─────────────────────────────────────────────────────────
sec_open(
    "EXP 3", 
    "Sampling Technique Comparison",
    "This experiment evaluates the representativeness of different sampling techniques against the entire population mean for calorie expenditure."
)

population = df_raw.copy()
pop_mean   = population["Calories"].mean()
sr_mean    = population.sample(n=50, random_state=42)["Calories"].mean()
k_step     = len(population) // 50
sys_mean   = population.iloc[::k_step].head(50)["Calories"].mean()

population["ActivityLevel"] = pd.qcut(population["TotalActiveMinutes"], q=3, labels=["Low","Medium","High"])
strat_sample = population.groupby("ActivityLevel", group_keys=False, observed=False).apply(
    lambda x: x.sample(15, random_state=42))
strat_mean = strat_sample["Calories"].mean()

np.random.seed(42)
random_cluster_id = np.random.choice(population["Id"].unique())
cluster_mean      = population[population["Id"] == random_cluster_id]["Calories"].mean()

sampling_df = pd.DataFrame({
    "Method":        ["Population", "Simple Random", "Systematic", "Stratified", "Cluster (single user)"],
    "Mean Calories": [pop_mean, sr_mean, sys_mean, strat_mean, cluster_mean],
})

sampling_df["Deviation"] = sampling_df["Mean Calories"] - pop_mean
sampling_df["Abs_Dev"]   = sampling_df["Deviation"].abs()

# Sort by closest to population mean (best -> worst)
sampling_df = sampling_df.sort_values("Abs_Dev").reset_index(drop=True)

def get_color(dev, method):
    if method == "Population": return "#667eea"
    if dev < 50: return "#34d399"    # Green (close)
    elif dev < 150: return "#f59e0b" # Yellow (moderate)
    else: return "#f87171"           # Red (high deviation)

sampling_df["Color"] = sampling_df.apply(lambda x: get_color(x["Abs_Dev"], x["Method"]), axis=1)
sampling_df["Label"] = sampling_df.apply(
    lambda x: "Base" if x["Method"] == "Population" else f"{'+' if x['Deviation'] > 0 else ''}{x['Deviation']:.0f}", 
    axis=1
)

col_l3, col_r3 = st.columns([2, 1], gap="large")
with col_l3:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    fig_bar = px.bar(sampling_df, x="Method", y="Mean Calories",
                     text="Label", color="Method",
                     color_discrete_map=dict(zip(sampling_df["Method"], sampling_df["Color"])))
    fig_bar.add_hline(y=pop_mean, line_dash="dash",
                      line_color="rgba(255,255,255,0.4)",
                      annotation_text=f"Population Mean: {pop_mean:.0f}",
                      annotation_font_color="white")
    fig_bar.update_traces(textposition='inside', 
                          textfont=dict(size=14, color='white'),
                          texttemplate="<b>%{text}</b>")
    fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                          xaxis_title="", yaxis_title="Mean Calories")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r3:
    rows_html = ""
    for idx, r in sampling_df.iterrows():
        if r["Method"] == "Population":
            style, dev_str = "color:#818cf8; font-weight:800;", "Base"
        elif idx == 1:  # Best method
            style, dev_str = "color:#34d399; font-weight:700;", f"({'+' if r['Deviation']>0 else ''}{r['Deviation']:.0f})"
        elif idx == len(sampling_df) - 1:  # Worst method
            style, dev_str = "color:#f87171; font-weight:700; opacity:0.9;", f"({'+' if r['Deviation']>0 else ''}{r['Deviation']:.0f})"
        else:
            style, dev_str = "color:#e2e8f0; font-weight:600;", f"({'+' if r['Deviation']>0 else ''}{r['Deviation']:.0f})"
            
        rows_html += f"<tr><td>{r['Method']}</td><td style='text-align:right;'><span style='{style}'>{r['Mean Calories']:.0f} <span style='font-size:0.75rem; opacity:0.8;'>{dev_str}</span></span></td></tr>"

    st.markdown(f"""
    <table class='styled-table'>
      <tr><th>Method</th><th style='text-align:right'>Mean Calories (Dev)</th></tr>
      {rows_html}
    </table>
    <div class='interp-box'>
      <b>Interpretation:</b> Random and systematic sampling closely approximate the population mean,
      while cluster sampling shows greater deviation due to individual variability.
    </div>
    """, unsafe_allow_html=True)

sec_close()

# ─── EXP 4 — CLUSTERING ───────────────────────────────────────────────────────
sec_open(
    "EXP 4", 
    "K-Means Clustering",
    "User Segmentation based on Activity Levels"
)

CLUSTER_LABELS = {"0":"Low Activity", "1":"Medium Activity", "2":"High Activity"}
# Higher contrast distinct colors
CLUSTER_COLOR_MAP = {"Low Activity": "#60a5fa", "Medium Activity": "#fbbf24", "High Activity": "#34d399"}

df_plot = df.copy()
df_plot["Segment"] = df_plot["Cluster"].astype(str).map(CLUSTER_LABELS)

col_l4, col_r4 = st.columns([2, 1], gap="large")
with col_l4:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    fig_clust = px.scatter(df_plot, x="TotalActiveMinutes", y="Calories",
                           color="Segment",
                           color_discrete_map=CLUSTER_COLOR_MAP,
                           opacity=0.7)
    fig_clust.update_traces(marker=dict(size=8, line=dict(width=0)))
    fig_clust.update_layout(**PLOTLY_LAYOUT,
                            xaxis_title="Total Active Minutes",
                            yaxis_title="Calories",
                            legend_title="User Segment")
    fig_clust.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig_clust.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig_clust, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r4:
    cluster_stats = df_plot.groupby("Segment").agg(
        Count=("Calories","count"),
        Avg_Cal=("Calories","mean"),
        Avg_Min=("TotalActiveMinutes","mean"),
    ).reset_index()

    rows_c = "".join(
        f"<tr>"
        f"<td><span style='color:{CLUSTER_COLOR_MAP.get(r.Segment, '#fff')}; font-weight:800; font-size:0.92rem;'>"
        f"{r.Segment}</span></td>"
        f"<td style='text-align:right'>{r.Count}</td>"
        f"<td style='text-align:right'>{r.Avg_Cal:.0f}</td>"
        f"<td style='text-align:right'>{r.Avg_Min:.0f}</td>"
        f"</tr>"
        for r in cluster_stats.itertuples()
    )
    st.markdown(f"""
    <table class='styled-table'>
      <tr>
        <th>Segment</th>
        <th style='text-align:right'>Number of Users</th>
        <th style='text-align:right'>Average Calories</th>
        <th style='text-align:right'>Average Active Minutes</th>
      </tr>
      {rows_c}
    </table>
    <div class='interp-box'>
      <b>Interpretation:</b> K-Means clustering (k=3) identifies three distinct user segments based on activity and calorie patterns, representing low, moderate, and high fitness levels.
    </div>
    """, unsafe_allow_html=True)

sec_close()

# ─── EXP 5 — PROBABILITY DISTRIBUTION ────────────────────────────────────────
sec_open(
    "EXP 5", 
    "Probability Distribution",
    "Distribution of Calorie Expenditure with Normal Fit"
)

cal_vals = df["Calories"].dropna().values
cal_mean = cal_vals.mean()
cal_std  = cal_vals.std()

col_l5, col_r5 = st.columns([2, 1], gap="large")
with col_l5:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    count, bins = np.histogram(cal_vals, bins=35, density=True)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=bins[:-1], y=count, width=np.diff(bins),
        marker_color="rgba(102,126,234,0.35)", name="Actual Distribution",
        hovertemplate="Cal: %{x:.0f}<br>Density: %{y:.5f}<extra></extra>",
    ))
    x_curve = np.linspace(cal_vals.min(), cal_vals.max(), 300)
    y_curve  = norm.pdf(x_curve, cal_mean, cal_std)
    fig_dist.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines", line=dict(color="#f59e0b", width=3.5),
        name="Normal Distribution Curve",
    ))
    fig_dist.update_layout(**PLOTLY_LAYOUT)
    fig_dist.update_layout(
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)",
            yanchor="top", y=0.99, xanchor="right", x=0.99
        )
    )
    fig_dist.update_xaxes(title_text="Calories Burned", title_font=dict(size=13))
    fig_dist.update_yaxes(title_text="Density", title_font=dict(size=13))
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r5:
    st.markdown(f"""
    <div class='stat-item'><span>Average Calories</span><span class='stat-val'>&#8776; {cal_mean:.0f}</span></div>
    <div class='stat-item'><span>Standard Deviation</span><span class='stat-val'>&#8776; {cal_std:.0f}</span></div>
    <div class='interp-box'>
      <b>Interpretation:</b> The calorie distribution approximates a normal distribution with a slight right skew, indicating that higher calorie values occur less frequently but extend the upper tail.
    </div>
    """, unsafe_allow_html=True)

sec_close()

# ─── EXP 6 — STATISTICAL ANALYSIS ────────────────────────────────────────────
sec_open(
    "EXP 6", 
    "Descriptive Statistical Analysis",
    "This experiment summarizes the core descriptive statistics of continuous health metrics to understand central tendency, dispersion, and shape."
)

cal    = df["Calories"]
mean_v = np.mean(cal)
var_v  = np.var(cal)
std_v  = np.std(cal)
skew_v = skew(cal)
kurt_v = scipy_kurtosis(cal)

# Structured stat cards instead of columns of st.metric
st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
  <!-- CARD 1 -->
  <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <div style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;">Central Tendency</div>
    <div style="color: #60a5fa; font-size: 2.2rem; font-weight: 800; line-height: 1;">Mean = {mean_v:.0f}</div>
  </div>

  <!-- CARD 2 -->
  <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1.5rem; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <div style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; text-align: center;">Dispersion</div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; padding-bottom: 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <span style="color: #cbd5e1; font-size: 1.05rem;">Variance</span>
        <span style="color: #f59e0b; font-weight: 700; font-size: 1.2rem;">{var_v:.0f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="color: #cbd5e1; font-size: 1.05rem;">Std Dev</span>
        <span style="color: #f59e0b; font-weight: 700; font-size: 1.2rem;">{std_v:.0f}</span>
    </div>
  </div>

  <!-- CARD 3 -->
  <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1.5rem; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <div style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; text-align: center;">Distribution Shape</div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; padding-bottom: 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <span style="color: #cbd5e1; font-size: 1.05rem;">Skewness</span>
        <span style="color: #34d399; font-weight: 700; font-size: 1.2rem;">{skew_v:.2f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="color: #cbd5e1; font-size: 1.05rem;">Kurtosis</span>
        <span style="color: #34d399; font-weight: 700; font-size: 1.2rem;">{kurt_v:.2f}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

fig_hist = px.histogram(df, x="Calories", nbins=50,
                        color_discrete_sequence=["#667eea"], opacity=0.8)
fig_hist.update_traces(marker_line_width=1.5, marker_line_color="rgba(15,23,42,0.8)")

fig_hist.add_vline(x=mean_v, line_dash="dash", line_width=2.5, line_color="#ef4444",
                   annotation_text="Mean", annotation_font_color="#ef4444", 
                   annotation_position="top right", annotation_font_size=13)

fig_hist.update_layout(**PLOTLY_LAYOUT,
                       xaxis_title="Calories Burned", yaxis_title="Frequency",
                       height=320)
fig_hist.update_layout(margin=dict(l=10, r=10, t=10, b=10))
fig_hist.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""
<div class='interp-box' style='margin-top:1.5rem'>
  <b>Interpretation:</b> The distribution shows moderate variability (standard deviation &#8776; 703) with slight positive skewness, indicating most users cluster around average calorie expenditure while a smaller group exhibits higher calorie burn.
</div>
""", unsafe_allow_html=True)

sec_close()

