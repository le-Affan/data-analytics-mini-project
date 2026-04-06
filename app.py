import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import linregress, norm, skew, kurtosis as scipy_kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Analytics Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Animated gradient header */
.hero-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: fadeInDown 0.6s ease;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.65);
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.8rem;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
div[data-testid="metric-container"] label { color: rgba(255,255,255,0.55) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e0e0ff !important; font-size: 1.9rem !important; font-weight: 700 !important; }

/* Section cards */
.section-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    animation: fadeIn 0.5s ease;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #a9b7ff;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-exp-num {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 8px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Interpretation box */
.interp-box {
    background: rgba(102,126,234,0.1);
    border-left: 3px solid #667eea;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.6;
}
.interp-box b { color: #a9b7ff; }

/* Stats list */
.stat-item { display:flex; justify-content:space-between; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.88rem; color: rgba(255,255,255,0.7); }
.stat-item:last-child { border-bottom: none; }
.stat-val { color: #e0e0ff; font-weight: 600; }

/* Table styling */
.styled-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.styled-table th { background:rgba(102,126,234,0.2); color:#a9b7ff; padding:8px 12px; text-align:left; border-bottom:1px solid rgba(255,255,255,0.1); }
.styled-table td { padding:7px 12px; border-bottom:1px solid rgba(255,255,255,0.04); color:rgba(255,255,255,0.75); }
.styled-table tr:hover td { background:rgba(255,255,255,0.03); }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1535 100%); }
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.65) !important; }
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background: #667eea !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeInDown { from { opacity:0; transform:translateY(-12px); } to { opacity:1; transform:translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,26,46,0.6)",
    font=dict(family="Inter", color="rgba(255,255,255,0.75)", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    colorway=["#667eea","#f5a623","#50e3c2","#e74c3c","#2ecc71","#9b59b6"],
    hoverlabel=dict(bgcolor="#2a2a3e", font_color="white", bordercolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
)
CLUSTER_COLOR_MAP = {"0":"#667eea","1":"#f5a623","2":"#50e3c2"}

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

        # Normalize dates — both CSVs already have 'ActivityDate'
        activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"], format="mixed").dt.date
        sleep["ActivityDate"]    = pd.to_datetime(sleep["ActivityDate"],    format="mixed").dt.date

        df = pd.merge(activity, sleep, on=["Id","ActivityDate"], how="left")
        df["SleepMinutes"] = df["SleepMinutes"].fillna(0)

        # Feature engineering (mirrors notebook cell 3)
        df = df.drop_duplicates()
        df = df[df["TotalSteps"] >= 0]
        df = df[df["Calories"] > 0]
        df["TotalActiveMinutes"] = df["VeryActiveMinutes"] + df["FairlyActiveMinutes"] + df["LightlyActiveMinutes"]
        df["ActivityRatio"]      = df["TotalActiveMinutes"] / (df["SedentaryMinutes"] + 1)

        # Clustering (mirrors notebook cell 9)
        X = df[["TotalActiveMinutes","Calories"]].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
        return df, None
    except Exception as e:
        return None, str(e)

df_raw, load_err = load_data()
if df_raw is None:
    st.error(f"❌ Data Loading Failed: {load_err}")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filters")
st.sidebar.markdown("---")

all_ids = sorted(df_raw["Id"].unique().tolist())
sel_ids = st.sidebar.multiselect("User ID", all_ids, default=all_ids)

min_d, max_d = df_raw["ActivityDate"].min().date(), df_raw["ActivityDate"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_d, max_d], min_value=min_d, max_value=max_d)

df = df_raw[df_raw["Id"].isin(sel_ids)].copy()
if len(date_range) == 2:
    s, e = date_range
    df = df[(df["ActivityDate"].dt.date >= s) & (df["ActivityDate"].dt.date <= e)]

if df.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

n_rows, n_cols = df.shape

# ─── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <p class="hero-title">🏃 Fitness Analytics Dashboard</p>
  <p class="hero-sub">Comprehensive Analysis of Activity, Calories &amp; Health Patterns</p>
  <span class="hero-badge">📊 Dataset: {n_rows} rows × {n_cols} columns</span>
</div>
""", unsafe_allow_html=True)

# ─── SUMMARY METRIC CARDS ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔥 Mean Calories",         f"{df['Calories'].mean():.0f} kcal")
c2.metric("😴 Avg Sleep",             f"{df['SleepMinutes'].mean():.0f} min")
c3.metric("🏃 Avg Active Minutes",    f"{df['TotalActiveMinutes'].mean():.0f} min")
c4.metric("🪑 Avg Sedentary Minutes", f"{df['SedentaryMinutes'].mean():.0f} min")

st.markdown("---")

# ─── EXP 1 — BOX PLOT ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">
    <span class="section-exp-num">EXP 1</span>
    📦 Calorie Distribution &amp; Outlier Analysis
  </div>
</div>
""", unsafe_allow_html=True)

q1  = df["Calories"].quantile(0.25)
med = df["Calories"].median()
q3  = df["Calories"].quantile(0.75)
iqr = q3 - q1
lb  = q1 - 1.5 * iqr
ub  = q3 + 1.5 * iqr
outliers = df[( df["Calories"] < lb) | (df["Calories"] > ub)]["Calories"]

col_l, col_r = st.columns([2, 1], gap="large")
with col_l:
    fig_box = px.box(df, y="Calories", points="outliers",
                     color_discrete_sequence=["#667eea"])
    fig_box.update_traces(marker_color="#f5a623", marker_size=5, line_color="#667eea")
    fig_box.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)

with col_r:
    st.markdown(f"""
    <div style='margin-top:0.5rem'>
      <div class='stat-item'><span>Q1 (25th pct)</span><span class='stat-val'>{q1:.1f}</span></div>
      <div class='stat-item'><span>Median</span><span class='stat-val'>{med:.1f}</span></div>
      <div class='stat-item'><span>Q3 (75th pct)</span><span class='stat-val'>{q3:.1f}</span></div>
      <div class='stat-item'><span>IQR</span><span class='stat-val'>{iqr:.1f}</span></div>
      <div class='stat-item'><span>Lower Bound</span><span class='stat-val'>{lb:.2f}</span></div>
      <div class='stat-item'><span>Upper Bound</span><span class='stat-val'>{ub:.2f}</span></div>
      <div class='stat-item'><span>Outlier Count</span><span class='stat-val'>{len(outliers)}</span></div>
    </div>
    <div class='interp-box'>
      <b>Interpretation:</b> Calorie expenditure shows notable variability with several extreme
      values, indicating diverse activity levels among users.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── EXP 2 — LINEAR REGRESSION ────────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">
    <span class="section-exp-num">EXP 2</span>
    📈 Linear Regression — Active Minutes vs Calories
  </div>
</div>
""", unsafe_allow_html=True)

# Full-population regression (mirrors exact notebook approach)
slope_full, intercept_full, r_full, _, _ = linregress(df_raw["TotalActiveMinutes"], df_raw["Calories"])
r2_full = r_full ** 2

# Scatter from filtered df + regression line
col_l2, col_r2 = st.columns([2, 1], gap="large")
with col_l2:
    fig_reg = px.scatter(df, x="TotalActiveMinutes", y="Calories",
                         trendline="ols", trendline_color_override="#f5a623",
                         opacity=0.55, color_discrete_sequence=["#667eea"])
    fig_reg.update_layout(**PLOTLY_LAYOUT,
                          xaxis_title="Total Active Minutes",
                          yaxis_title="Calories")
    st.plotly_chart(fig_reg, use_container_width=True)

with col_r2:
    st.markdown(f"""
    <div style='margin-top:0.5rem'>
      <div class='stat-item'><span>Intercept</span><span class='stat-val'>≈ {intercept_full:.1f}</span></div>
      <div class='stat-item'><span>Slope</span><span class='stat-val'>≈ {slope_full:.4f}</span></div>
      <div class='stat-item'><span>R² Score</span><span class='stat-val'>≈ {r2_full:.4f}</span></div>
    </div>
    <div class='interp-box'>
      <b>Interpretation:</b> There is a weak positive relationship between total active minutes
      and calories burned, suggesting that additional factors also influence calorie expenditure.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── EXP 3 — SAMPLING ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">
    <span class="section-exp-num">EXP 3</span>
    ⚖️ Sampling Technique Comparison
  </div>
</div>
""", unsafe_allow_html=True)

# Mirror notebook: use full population (df_raw after feature engineering but no filter)
population = df_raw.copy()

pop_mean = population["Calories"].mean()
sr_mean  = population.sample(n=50, random_state=42)["Calories"].mean()

k_step   = len(population) // 50
sys_mean = population.iloc[::k_step].head(50)["Calories"].mean()

population["ActivityLevel"] = pd.qcut(population["TotalActiveMinutes"], q=3, labels=["Low","Medium","High"])
strat_sample = population.groupby("ActivityLevel", group_keys=False, observed=False).apply(
    lambda x: x.sample(15, random_state=42))
strat_mean = strat_sample["Calories"].mean()

np.random.seed(42)
random_cluster_id = np.random.choice(population["Id"].unique())
cluster_mean = population[population["Id"] == random_cluster_id]["Calories"].mean()

sampling_df = pd.DataFrame({
    "Method": ["Population","Simple Random","Systematic","Stratified","Cluster (single user)"],
    "Mean Calories": [pop_mean, sr_mean, sys_mean, strat_mean, cluster_mean],
})

col_l3, col_r3 = st.columns([2, 1], gap="large")
with col_l3:
    fig_bar = px.bar(sampling_df, x="Method", y="Mean Calories",
                     text_auto=".0f",
                     color="Method",
                     color_discrete_sequence=["#667eea","#f5a623","#50e3c2","#e74c3c","#9b59b6"])
    fig_bar.add_hline(y=pop_mean, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                      annotation_text=f"Population Mean: {pop_mean:.0f}",
                      annotation_font_color="rgba(255,255,255,0.5)")
    fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                          xaxis_title="", yaxis_title="Mean Calories")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_r3:
    rows_html = "".join(
        f"<tr><td>{r['Method']}</td><td style='text-align:right;color:#e0e0ff;font-weight:600'>{r['Mean Calories']:.0f}</td></tr>"
        for _, r in sampling_df.iterrows()
    )
    st.markdown(f"""
    <table class='styled-table'>
      <tr><th>Method</th><th style='text-align:right'>Mean Calories</th></tr>
      {rows_html}
    </table>
    <div class='interp-box'>
      <b>Interpretation:</b> Random and systematic sampling closely approximate the population mean,
      while cluster sampling (single-user) shows greater deviation due to individual variability.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── EXP 4 — CLUSTERING ───────────────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">
    <span class="section-exp-num">EXP 4</span>
    🧩 K-Means Clustering — User Fitness Segments
  </div>
</div>
""", unsafe_allow_html=True)

CLUSTER_LABELS = {"0":"Low Activity","1":"Medium Activity","2":"High Activity"}

df_plot = df.copy()
df_plot["Cluster_Str"] = df_plot["Cluster"].astype(str)
df_plot["Segment"]     = df_plot["Cluster_Str"].map(CLUSTER_LABELS)

col_l4, col_r4 = st.columns([2, 1], gap="large")
with col_l4:
    fig_clust = px.scatter(df_plot, x="TotalActiveMinutes", y="Calories",
                           color="Cluster_Str",
                           hover_data={"Segment":True,"Cluster_Str":False},
                           color_discrete_map=CLUSTER_COLOR_MAP,
                           opacity=0.65)
    fig_clust.update_layout(**PLOTLY_LAYOUT,
                            xaxis_title="Total Active Minutes",
                            yaxis_title="Calories",
                            legend_title="Cluster")
    st.plotly_chart(fig_clust, use_container_width=True)

with col_r4:
    cluster_stats = df_plot.groupby("Cluster_Str").agg(
        Count=("Calories","count"),
        Avg_Cal=("Calories","mean"),
        Avg_Min=("TotalActiveMinutes","mean"),
    ).reset_index()

    rows_c = "".join(
        f"<tr><td><span style='color:{CLUSTER_COLOR_MAP[r.Cluster_Str]};font-weight:600'>{CLUSTER_LABELS.get(r.Cluster_Str,r.Cluster_Str)}</span></td>"
        f"<td style='text-align:right'>{r.Count}</td>"
        f"<td style='text-align:right'>{r.Avg_Cal:.0f}</td>"
        f"<td style='text-align:right'>{r.Avg_Min:.0f}</td></tr>"
        for r in cluster_stats.itertuples()
    )
    st.markdown(f"""
    <table class='styled-table'>
      <tr><th>Segment</th><th style='text-align:right'>Users</th><th style='text-align:right'>Avg Cal</th><th style='text-align:right'>Avg Min</th></tr>
      {rows_c}
    </table>
    <div class='interp-box'>
      <b>Interpretation:</b> K-Means clustering (k=3) reveals three distinct fitness levels
      among users — low-activity sedentary users, moderately active users, and highly active users.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── EXP 5 — PROBABILITY DISTRIBUTION ────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">
    <span class="section-exp-num">EXP 5</span>
    📊 Probability Distribution of Calories
  </div>
</div>
""", unsafe_allow_html=True)

cal_vals = df["Calories"].dropna().values
cal_mean = cal_vals.mean()
cal_std  = cal_vals.std()

col_l5, col_r5 = st.columns([2, 1], gap="large")
with col_l5:
    # histogram
    count, bins = np.histogram(cal_vals, bins=35, density=True)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=bins[:-1], y=count, width=np.diff(bins),
        marker_color="rgba(102,126,234,0.55)", name="Observed",
        hovertemplate="Cal: %{x:.0f}<br>Density: %{y:.5f}<extra></extra>",
    ))
    # normal curve
    x_curve = np.linspace(cal_vals.min(), cal_vals.max(), 300)
    y_curve  = norm.pdf(x_curve, cal_mean, cal_std)
    fig_dist.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines", line=dict(color="#f5a623", width=2.5),
        name="Normal Fit",
    ))
    fig_dist.update_layout(**PLOTLY_LAYOUT,
                           xaxis_title="Calories",
                           yaxis_title="Density")
    st.plotly_chart(fig_dist, use_container_width=True)

with col_r5:
    st.markdown(f"""
    <div style='margin-top:0.5rem'>
      <div class='stat-item'><span>Mean</span><span class='stat-val'>≈ {cal_mean:.0f}</span></div>
      <div class='stat-item'><span>Std Dev</span><span class='stat-val'>≈ {cal_std:.0f}</span></div>
    </div>
    <div class='interp-box'>
      <b>Interpretation:</b> The calorie distribution approximates a normal curve with a slight
      right skew — driven by a minority of users with unusually high calorie expenditure.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── EXP 6 — STATISTICAL ANALYSIS ────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <div class="section-title">
    <span class="section-exp-num">EXP 6</span>
    📋 Descriptive Statistical Analysis
  </div>
</div>
""", unsafe_allow_html=True)

cal = df["Calories"]
mean_v     = np.mean(cal)
var_v      = np.var(cal)
std_v      = np.std(cal)
skew_v     = skew(cal)
kurt_v     = scipy_kurtosis(cal)

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Mean",      f"{mean_v:.0f}")
s2.metric("Variance",  f"{var_v:.0f}")
s3.metric("Std Dev",   f"{std_v:.0f}")
s4.metric("Skewness",  f"{skew_v:.2f}")
s5.metric("Kurtosis",  f"{kurt_v:.2f}")

# Distribution bar
fig_hist = px.histogram(df, x="Calories", nbins=30,
                        color_discrete_sequence=["#667eea"],
                        opacity=0.75)
fig_hist.update_layout(**PLOTLY_LAYOUT,
                       xaxis_title="Calories", yaxis_title="Count",
                       height=220)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""
<div class='interp-box'>
  <b>Interpretation:</b> The data shows moderate variability (std ≈ 703) and slight positive
  skewness (≈ 0.55), consistent with a roughly normal distribution skewed by high-calorie outliers.
</div>
""", unsafe_allow_html=True)
