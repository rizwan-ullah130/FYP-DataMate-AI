import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from ydata_profiling import ProfileReport
from cache_config import *
import os
import plotly.io as pio
import io
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                    MinMaxScaler, StandardScaler,
                                        RobustScaler, MaxAbsScaler,
                                        PowerTransformer, QuantileTransformer)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                            f_classif, f_regression,
                                            mutual_info_classif, mutual_info_regression)
from scipy import stats
from sklearn.decomposition import PCA
from scipy.stats import shapiro, skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go
import warnings

# =========================================================
# === 1. PAGE CONFIGURATION & STYLING =====================
# =========================================================
st.set_page_config(
    page_title="DataMate AI - Your AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CACHED CSS STYLING ─────────────────────────────────────────────────────
@st.cache_data
def load_css_styling():
    """Cache styling to avoid re-rendering on every page load"""
    return """
<style>
    /* ================= GOOGLE FONT IMPORT ================= */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    /* ================= KEYFRAME ANIMATIONS ================= */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(52,152,219,0.4); }
        50%       { box-shadow: 0 0 0 8px rgba(52,152,219,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-6px); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.8); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes borderPulse {
        0%, 100% { border-color: #3498db; }
        50%       { border-color: #2ecc71; }
    }
    @keyframes sidebarFadeIn {
        from { opacity: 0; transform: translateX(-15px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* ================= GLOBAL FONT ================= */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2535 100%) !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(174,214,241,0.2);
        color: #cdd9e5 !important;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600 !important;
        font-size: 14px;
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
        padding: 11px 16px;
        text-align: left;
        margin-bottom: 6px;
        animation: sidebarFadeIn 0.4s ease both;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateX(6px);
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border: 1px solid #2980b9 !important;
        box-shadow: 0 4px 18px rgba(52,152,219,0.45);
    }
    section[data-testid="stSidebar"] .stButton button:active {
        transform: scale(0.97);
    }

    /* ================= MAIN HEADER ================= */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 22px 28px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 24px;
        font-family: 'Sora', sans-serif;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        animation: fadeInUp 0.6s ease both;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        animation: shimmer 3s infinite;
    }

    /* ================= SIDEBAR HEADER CARD ================= */
    .sidebar-header-card {
        background: linear-gradient(135deg, #3498db, #1a6dad) !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 12px !important;
        margin-bottom: 18px !important;
        text-align: center !important;
        box-shadow: 0 4px 20px rgba(52,152,219,0.4) !important;
        animation: pulse-glow 3s infinite;
    }

    /* ================= METRIC CARDS ================= */
    .metric-card {
        background: white;
        padding: 20px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
        margin: 10px 0;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.5s ease both;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 12px 32px rgba(52,152,219,0.18);
    }
    .metric-card p {
        animation: countUp 0.6s ease both;
    }

    /* ================= SECTION HEADER ================= */
    .section-header {
        font-family: 'Sora', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: #1a2535;
        margin: 22px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3498db;
        animation: slideInLeft 0.5s ease both;
    }

    /* ================= CONTENT FRAME ================= */
    .content-frame {
        border: 1.5px solid #dee2e6;
        border-radius: 12px;
        padding: 22px;
        margin: 12px 0;
        background: white;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        animation: fadeInUp 0.5s ease both;
    }

    /* ================= CHART CONTAINER ================= */
    .chart-container {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 12px 0;
        transition: box-shadow 0.2s;
        animation: fadeInUp 0.5s ease both;
    }
    .chart-container:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    }

    /* ================= UPLOAD SECTION ================= */
    .upload-section {
        background: #f8f9fa;
        border: 2px dashed #AED6F1;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        transition: border-color 0.3s, background 0.3s;
        animation: borderPulse 3s infinite;
    }
    .upload-section:hover {
        border-color: #3498db;
        background: #eaf4fb;
    }

    /* ================= WELCOME PAGE ================= */
    .welcome-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        animation: fadeInUp 0.7s ease both;
    }
    .welcome-title {
        font-family: 'Sora', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);   /* RESPONSIVE FONT */
        font-weight: 800;
        color: #0e1117;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }
    .welcome-brand {
        font-family: 'Sora', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);   /* RESPONSIVE FONT */
        font-weight: 800;
        background: linear-gradient(135deg, #3498db, #1abc9c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 1.2rem;
        animation: shimmer 4s infinite;
        background-size: 200% auto;
    }
    .welcome-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: clamp(1rem, 2.5vw, 1.4rem);  /* RESPONSIVE FONT */
        color: #555;
        font-weight: 400;
        margin-bottom: 2.2rem;
        line-height: 1.6;
    }

    /* ================= FEATURE CARDS ================= */
    .feature-card {
        background: white;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease both;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 14px 36px rgba(52,152,219,0.18);
        animation: float 2s ease-in-out infinite;
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
        display: block;
    }
    .feature-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 15px;
        color: #1a2535;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 13px;
        color: #7f8c8d;
        line-height: 1.5;
    }

    /* ================= ROBOT IMAGE: FLOATING ANIMATION ================= */
    .robot-container {
        animation: float 3s ease-in-out infinite;
    }

    /* ================= RESPONSIVE: MOBILE ================= */
    @media (max-width: 768px) {
        .main-header { font-size: 18px; padding: 16px; }
        .metric-card { padding: 14px 12px; }
        .content-frame { padding: 14px; }
        .welcome-title, .welcome-brand { font-size: 2rem !important; }
        .welcome-subtitle { font-size: 1rem !important; }
        .feature-card { margin-bottom: 12px; }
    }

    /* ================= GET STARTED BUTTON ================= */
    .stButton > button[kind="primary"] {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        border-radius: 50px;
        padding: 12px 32px;
        background: linear-gradient(135deg, #3498db, #1abc9c) !important;
        border: none !important;
        color: white !important;
        font-size: 16px;
        box-shadow: 0 6px 20px rgba(52,152,219,0.4);
        transition: all 0.3s ease;
        animation: pulse-glow 2.5s infinite;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 12px 32px rgba(52,152,219,0.5);
    }

    /* ================= ABOUT PAGE CARDS ================= */
    .founder-card {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 3px solid #3498db;
        transition: transform 0.25s;
        animation: fadeInUp 0.6s ease both;
    }
    .founder-card:hover { transform: translateY(-4px); }

    /* ================= DOWNLOAD BUTTON ================= */
    .stDownloadButton > button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stDownloadButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(39,174,96,0.4) !important;
    }

    /* ================= STREAMLIT NATIVE METRIC ================= */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
        border-left: 3px solid #3498db;
        animation: fadeInUp 0.5s ease both;
    }
</style>
"""

st.markdown(load_css_styling(), unsafe_allow_html=True)

# Initialize Session State
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Welcome'
if 'all_datasets' not in st.session_state:
    st.session_state.all_datasets = {}
if 'selected_file_name' not in st.session_state:
    st.session_state.selected_file_name = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'saved_charts' not in st.session_state:
    st.session_state.saved_charts = []


# =========================================================
# === 2. HELPER FUNCTIONS =================================
# =========================================================

@st.cache_data
def get_column_types(df):
    """Cache column type calculations to avoid repeated select_dtypes calls"""
    return {
        'numeric': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
    }

def load_data_with_encoding(file, file_type, user_encoding=None):
    def try_encodings(read_function, file, encodings, **kwargs):
        for enc in encodings:
            try:
                file.seek(0)
                return read_function(file, encoding=enc, **kwargs)
            except (UnicodeDecodeError, TypeError, ValueError):
                continue
        raise ValueError("Unable to decode file.")

    try:
        file_type = file_type.lower()
        if file_type == "excel":
            file_type = file.name.split('.')[-1].lower()

        encodings_to_try = []
        if user_encoding:
            encodings_to_try.append(user_encoding)
        encodings_to_try += ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        if file_type == 'csv':
            return try_encodings(pd.read_csv, file, encodings_to_try, sep=None, engine='python', on_bad_lines='skip')
        elif file_type == 'txt':
            return try_encodings(pd.read_csv, file, encodings_to_try, sep=None, engine='python', on_bad_lines='skip')
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(file)
        elif file_type == 'json':
            try:
                file.seek(0)
                return pd.read_json(file)
            except ValueError:
                return try_encodings(pd.read_json, file, encodings_to_try, lines=True)
        elif file_type == 'xml':
            return try_encodings(pd.read_xml, file, encodings_to_try)
        elif file_type == 'parquet':
            return pd.read_parquet(file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


# =========================================================
# === HTML Report Generator ================================
# =========================================================

def generate_html_report(df, saved_charts):
    # Cache expensive calculations
    missing_count = int(df.isnull().sum().sum())
    duplicates_count = int(df.duplicated().sum())
    numeric_cols_count = len(df.select_dtypes(include=['int64', 'float64']).columns)
    
    def compute_kpi_value(df, column, agg):
        if column == "__rows__":         return f"{len(df):,}"
        elif column == "__columns__":    return f"{len(df.columns):,}"
        elif column == "__numeric__":    return f"{numeric_cols_count:,}"
        elif column == "__missing__":    return f"{missing_count:,}"
        elif column == "__duplicates__": return f"{duplicates_count:,}"
        if column not in df.columns: return "N/A"
        try:
            series = pd.to_numeric(df[column], errors='coerce')
            if agg == "Sum":      return f"{series.sum():,.2f}"
            elif agg == "Count":  return f"{series.count():,}"
            elif agg == "Mean":   return f"{series.mean():,.2f}"
            elif agg == "Median": return f"{series.median():,.2f}"
            elif agg == "Max":    return f"{series.max():,.2f}"
            elif agg == "Min":    return f"{series.min():,.2f}"
            elif agg == "Std":    return f"{series.std():,.2f}"
        except Exception:
            return str(df[column].count())

    kpi_configs = st.session_state.get("kpi_configs", [
        {"label": "Total Records",   "column": "__rows__",    "agg": "Count", "color": "#3498db"},
        {"label": "Total Columns",   "column": "__columns__", "agg": "Count", "color": "#3498db"},
        {"label": "Numeric Columns", "column": "__numeric__", "agg": "Count", "color": "#3498db"},
        {"label": "Missing Values",  "column": "__missing__", "agg": "Count", "color": "#3498db"},
    ])

    kpi_cards_html = ""
    for cfg in kpi_configs:
        label  = cfg.get("label",  "KPI")
        column = cfg.get("column", "__rows__")
        agg    = cfg.get("agg",    "Count")
        color  = cfg.get("color",  "#3498db")
        val    = compute_kpi_value(df, column, agg)
        val_color = "#e74c3c" if (column == "__missing__" and int(df.isnull().sum().sum()) > 0) else color

        kpi_cards_html += f"""
        <div class="metric-card" style="border-top: 4px solid {color};">
            <span class="metric-label">{label}</span>
            <p class="metric-value" style="color: {val_color};">{val}</p>
            <span class="metric-agg">{agg}</span>
        </div>"""

    charts_html = ""
    if saved_charts:
        for chart in saved_charts:
            chart_div = pio.to_html(chart['fig'], full_html=False, include_plotlyjs='cdn')
            user_desc = chart.get('description', '').strip()
            desc_html = f"""<div class="analysis-note"><span class="note-label">📝 Analysis Note:</span>{user_desc}</div>""" if user_desc else ""
            charts_html += f"""
            <div class="chart-container">
                <h3 class="chart-title">{chart['title']}</h3>
                {chart_div}
                {desc_html}
            </div>"""
    else:
        charts_html = "<p style='color:#7f8c8d; padding:20px;'>No visualizations were pinned to the dashboard.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #2c3e50; padding: 40px 30px; }}
        .report-header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px 40px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
        .report-header h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: 1px; }}
        .section-header {{ font-size: 1.25rem; font-weight: 700; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; margin: 35px 0 20px 0; }}
        .kpi-grid {{ display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 10px; }}
        .metric-card {{ background: white; padding: 22px 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); flex: 1 1 160px; min-width: 140px; text-align: center; }}
        .metric-value {{ font-size: 2rem; font-weight: 800; margin: 10px 0 4px 0; display: block; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
        .metric-agg {{ color: #bdc3c7; font-size: 0.72rem; text-transform: uppercase; }}
        .chart-container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); margin-bottom: 35px; }}
        .chart-title {{ font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 18px; padding-bottom: 10px; border-bottom: 1px solid #ecf0f1; }}
        .analysis-note {{ background-color: #eaf4fb; border-left: 5px solid #3498db; padding: 14px 18px; margin-top: 20px; border-radius: 6px; color: #2c3e50; font-size: 0.92rem; line-height: 1.6; }}
        .note-label {{ font-weight: 700; color: #2980b9; display: block; margin-bottom: 6px; font-size: 0.88rem; }}
        @media (max-width: 600px) {{ body {{ padding: 16px; }} .metric-card {{ min-width: 100%; }} }}
    </style>
</head>
<body>
    <div class="report-header"><h1>📊 Dashboard</h1></div>
    <div class="section-header">Key Performance Indicators</div>
    <div class="kpi-grid">{kpi_cards_html}</div>
    <div class="section-header">Visualizations &amp; Analysis</div>
    {charts_html}
</body>
</html>"""

# =========================================================
# === WELCOME PAGE =========================================
# =========================================================
def welcome_page():
    # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.1rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # --------------------------------------------------

    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
        st.markdown("""
        <div class="welcome-title">Welcome to</div>
        <div class="welcome-brand">DataMate AI</div>
        <div class="welcome-subtitle">Your smart partner for<br>data insights & intelligence</div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Get Started", type="primary", use_container_width=False):
            st.session_state.current_page = "Data Upload"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        try:
            st.markdown('<div class="robot-container">', unsafe_allow_html=True)
            st.image("datamate_logo_50_2_40.png", width=350)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="robot-container" style="
                display: flex; justify-content: center; align-items: center;
                height: 380px;
                background: radial-gradient(circle at 50% 40%, #dbeeff 10%, #f0f8ff 70%);
                border-radius: 20px; border: 1px solid #cde;">
                <div style="font-size: 130px; filter: drop-shadow(0 8px 24px rgba(52,152,219,0.3));">🤖</div>
            </div>
            """, unsafe_allow_html=True)

    # st.markdown("---")

    # ── Animated Feature Cards ────────────────────────────────────────────────
    features = [
        ("📊", "Smart Analytics", "Auto-generated KPIs & real-time insights from your data"),
        ("🧹", "Auto Cleaning",   "Fix missing values, duplicates & outliers instantly"),
        ("📈", "Visualization",   "Create beautiful interactive charts in seconds"),
        ("🤖", "ML Models",       "Train AI models with zero code, zero effort"),
    ]
    f_cols = st.columns(4)
    for col, (icon, title, desc) in zip(f_cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")

# =========================================================
# === DASHBOARD PAGE ======================================
# =========================================================

def dashboard_page():
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    

    
    # -----------------------------    
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload data in the 'Data Upload' section to view the dashboard.")
        return

    df = st.session_state.df

    # Define compute_kpi_value function
    def compute_kpi_value(df, column, agg):
        if column == "__rows__":         return f"{len(df):,}"
        elif column == "__columns__":    return f"{len(df.columns):,}"
        elif column == "__numeric__":    return f"{len(df.select_dtypes(include=['int64', 'float64']).columns):,}"
        elif column == "__missing__":    return f"{int(df.isnull().sum().sum()):,}"
        elif column == "__duplicates__": return f"{int(df.duplicated().sum()):,}"
        if column not in df.columns: return "N/A"
        try:
            series = pd.to_numeric(df[column], errors='coerce')
            if agg == "Sum":      return f"{series.sum():,.2f}"
            elif agg == "Count":  return f"{series.count():,}"
            elif agg == "Mean":   return f"{series.mean():,.2f}"
            elif agg == "Median": return f"{series.median():,.2f}"
            elif agg == "Max":    return f"{series.max():,.2f}"
            elif agg == "Min":    return f"{series.min():,.2f}"
            elif agg == "Std":    return f"{series.std():,.2f}"
        except Exception:
            return str(df[column].count())

    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    all_columns = df.columns.tolist()

    if "kpi_configs" not in st.session_state:
        st.session_state.kpi_configs = [
            {"label": "Total Records",   "column": "__rows__",    "agg": "Count", "color": "#3498db"},
            {"label": "Total Columns",   "column": "__columns__", "agg": "Count", "color": "#3498db"},
            {"label": "Numeric Columns", "column": "__numeric__", "agg": "Count", "color": "#3498db"},
            {"label": "Missing Values",  "column": "__missing__", "agg": "Count", "color": "#3498db"},
        ]

    with st.expander("⚙️ Customize KPI Cards", expanded=False):
        st.caption("Set how many KPI cards to show and configure each one.")
        num_kpis = st.slider("How many KPI cards?", min_value=1, max_value=6,
                             value=len(st.session_state.kpi_configs), key="num_kpis_slider")

        while len(st.session_state.kpi_configs) < num_kpis:
            st.session_state.kpi_configs.append({
                "label": f"KPI {len(st.session_state.kpi_configs) + 1}",
                "column": all_columns[0] if all_columns else "__rows__",
                "agg": "Sum", "color": "#3498db"
            })
        st.session_state.kpi_configs = st.session_state.kpi_configs[:num_kpis]
        st.markdown("---")

        builtin_options = {
            "Total Rows": "__rows__", "Total Columns": "__columns__",
            "Numeric Columns": "__numeric__", "Missing Values": "__missing__",
            "Duplicate Rows": "__duplicates__",
        }
        agg_options = ["Sum", "Count", "Mean", "Median", "Max", "Min", "Std"]

        for idx in range(num_kpis):
            cfg = st.session_state.kpi_configs[idx]
            st.markdown(f"**Card {idx + 1}**")
            c1, c2, c3, c4 = st.columns([1.2, 1.8, 1.2, 0.8])

            with c1:
                new_label = st.text_input("Card Label", value=cfg.get("label", f"KPI {idx+1}"), key=f"kpi_label_{idx}")

            col_display_options = list(builtin_options.keys()) + all_columns
            col_value_options   = list(builtin_options.values()) + all_columns
            cur_col     = cfg.get("column", "__rows__")
            cur_col_idx = col_value_options.index(cur_col) if cur_col in col_value_options else 0

            with c2:
                selected_display = st.selectbox("Column / Metric", options=col_display_options,
                                                index=cur_col_idx, key=f"kpi_col_{idx}")
                selected_col = col_value_options[col_display_options.index(selected_display)]

            is_builtin = selected_col.startswith("__")
            with c3:
                if is_builtin:
                    st.selectbox("Aggregation", ["Count"], key=f"kpi_agg_{idx}", disabled=True)
                    selected_agg = "Count"
                else:
                    cur_agg = cfg.get("agg", "Sum")
                    agg_idx = agg_options.index(cur_agg) if cur_agg in agg_options else 0
                    selected_agg = st.selectbox("Aggregation", agg_options, index=agg_idx, key=f"kpi_agg_{idx}")

            with c4:
                selected_color = st.color_picker("Accent Color", value=cfg.get("color", "#3498db"), key=f"kpi_color_{idx}")

            st.session_state.kpi_configs[idx] = {
                "label": new_label, "column": selected_col, "agg": selected_agg, "color": selected_color
            }
            st.markdown("---")

    num_cards = len(st.session_state.kpi_configs)
    kpi_cols  = st.columns(num_cards)

    for col_ui, cfg in zip(kpi_cols, st.session_state.kpi_configs):
        val   = compute_kpi_value(df, cfg["column"], cfg["agg"])
        label = cfg["label"]
        color = cfg.get("color", "#3498db")
        val_color = "#e74c3c" if (cfg["column"] == "__missing__" and int(df.isnull().sum().sum()) > 0) else color

        with col_ui:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3 style='margin: 0; color: #2c3e50; font-family: Sora, sans-serif;'>{label}</h3>
                <p style='font-size: 26px; font-weight: 800; color: {val_color}; margin: 10px 0; font-family: Sora, sans-serif;'>{val}</p>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.saved_charts:
        st.markdown('<div class="section-header">📌 Your Pinned Visualizations</div>', unsafe_allow_html=True)
        for i, chart_obj in enumerate(st.session_state.saved_charts):
            with st.container():
                st.subheader(f"{chart_obj['title']}")
                st.plotly_chart(chart_obj['fig'], use_container_width=True, key=f"saved_chart_{i}")
                current_desc = chart_obj.get('description', '')
                new_desc = st.text_area(
                    "📝 Add Analysis Note (Optional):",
                    value=current_desc,
                    placeholder="Type your insights here...",
                    key=f"desc_input_{i}", height=100
                )
                st.session_state.saved_charts[i]['description'] = new_desc
                if st.button(f"🗑️ Remove Chart {i+1}", key=f"del_{i}"):
                    st.session_state.saved_charts.pop(i)
                    st.rerun()
            st.markdown("---")

    st.markdown("### 📥 Export Dashboard")
    st.info("Download the entire dashboard (KPIs + Charts + Your Notes) as an offline HTML file.")
    html_report = generate_html_report(df, st.session_state.saved_charts)
    st.download_button(
        label="⬇️ Download Full HTML Report",
        data=html_report,
        file_name="data_analysis_report.html",
        mime="text/html"
    )
    
# =========================================================
# === DATA UPLOAD PAGE =====================================
# =========================================================

# ── Helper: Deduplicate column names ─────────────────────────────────────────
def deduplicate_columns(columns):
    seen = {}
    result = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            result.append(col)
    return result

# ── Helper: Safe dataframe display ───────────────────────────────────────────
def safe_dataframe(df, **kwargs):
    display_df = df.copy()
    if display_df.columns.duplicated().any():
        display_df.columns = deduplicate_columns(display_df.columns.tolist())
    for col in display_df.columns:
        try:
            import pyarrow as pa
            pa.Array.from_pandas(display_df[col])
        except Exception:
            display_df[col] = display_df[col].astype(str)
    st.dataframe(display_df, **kwargs)


# ── Main Page ─────────────────────────────────────────────────────────────────
def data_upload_page():

    
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -----------------------------
    
    
    st.markdown('<div class="main-header">📑 Data Upload Management</div>',
                unsafe_allow_html=True)  

    uploaded_files = st.file_uploader(
        "Choose files",
        type=['csv', 'xlsx', 'xls', 'json', 'txt', 'xml', 'parquet'],
        accept_multiple_files=True,
        help="Supported: CSV, Excel, JSON, TXT, XML, Parquet. Auto-encoding detection enabled.",
        label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            file_key = file.name
            if file_key not in st.session_state.all_datasets:
                with st.spinner(f"Processing {file.name}..."):
                    ext = file.name.split('.')[-1].lower()
                    df = load_data_with_encoding(file, file_type=ext)
                    if df is not None:
                        st.session_state.all_datasets[file_key] = df
                        st.success(f"✅ Successfully loaded: {file.name}")
                    else:
                        st.error(f"❌ Failed to load {file.name}. Check format/encoding.")
    

    if st.session_state.all_datasets:
        st.subheader("📂 Select Active Dataset")

        file_options = list(st.session_state.all_datasets.keys())
        selected_file = st.selectbox(
            "Choose a dataset to analyze:", file_options,
            index=file_options.index(st.session_state.selected_file_name)
                  if st.session_state.selected_file_name in file_options else 0
        )

        if selected_file:
            st.session_state.selected_file_name = selected_file
            st.session_state.df = st.session_state.all_datasets[selected_file]
            current_df = st.session_state.df
            st.markdown(f"---\n**Previewing: {selected_file}**")

            # ── Metrics ───────────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(current_df):,}")
            with col2:
                st.metric("Columns", f"{len(current_df.columns):,}")
            with col3:
                mem_usage = current_df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Usage", f"{mem_usage:.2f} MB")

            # ── Rows to Preview Slider ────────────────────────────────────────
            rows_to_preview = st.slider(
                "🔢 Rows to preview",
                min_value=5,
                max_value=min(500, len(current_df)),
                value=10,
                step=5,
                help="Drag to choose how many rows to display in the preview table below."
            )
            safe_dataframe(current_df.head(rows_to_preview), use_container_width=True)

            # ── Data Types Info ───────────────────────────────────────────────
            with st.expander("🔎 View Data Types Info"):
                dtype_info = pd.DataFrame({
                    'Column': current_df.columns,
                    'Data Type': current_df.dtypes.astype(str),
                    'Non-Null Count': current_df.count(),
                    'Null Count': current_df.isnull().sum()
                })
                safe_dataframe(dtype_info, use_container_width=True)

            # ── Set Header Row ────────────────────────────────────────────────
            with st.expander("⚙️ Set Header Row"):
                st.markdown(
                    "Choose which row of the raw file should be the **column header**. "
                    "Useful when the file has title/metadata rows above the real column names."
                )

                header_source = st.radio(
                    "Header source",
                    options=["Default (Row 1)", "Row 2", "Custom row number"],
                    horizontal=True,
                    key=f"header_source_{selected_file}"
                )

                header_row_index = 0

                if header_source == "Row 2":
                    header_row_index = 1
                elif header_source == "Custom row number":
                    custom_row = st.number_input(
                        "Enter row number (1 = first row):",
                        min_value=1,
                        max_value=len(current_df),
                        value=1,
                        step=1,
                        key=f"custom_header_{selected_file}"
                    )
                    header_row_index = int(custom_row) - 1

                if st.button("✅ Apply Header Row", key=f"apply_header_{selected_file}"):
                    if header_row_index == 0:
                        st.info("ℹ️ Default header (Row 1) is already applied.")
                    elif header_row_index >= len(current_df):
                        st.error("❌ Row number exceeds dataset length.")
                    else:
                        try:
                            raw_headers  = current_df.iloc[header_row_index].astype(str).tolist()
                            new_headers  = deduplicate_columns(raw_headers)
                            duplicates_found = raw_headers != new_headers

                            new_df = current_df.iloc[header_row_index + 1:].copy()
                            new_df.columns = new_headers
                            new_df.reset_index(drop=True, inplace=True)

                            st.session_state.all_datasets[selected_file] = new_df
                            st.session_state.df = new_df
                            current_df = new_df

                            if duplicates_found:
                                st.warning(
                                    "⚠️ Duplicate column names were auto-renamed "
                                    "(e.g. `New York` → `New York_1`). Check columns below."
                                )
                            st.success(
                                f"✅ Header applied! Columns: "
                                f"{', '.join(new_headers[:5])}{'...' if len(new_headers) > 5 else ''}"
                            )
                            safe_dataframe(new_df.head(rows_to_preview), use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Failed to apply header row: {e}")

            # ── ydata-profiling ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Generate Dataset Profiling Report")

            if st.button("🔍 Generate Profiling Report for This Dataset"):
                with st.spinner("Generating profiling report… this may take a moment."):
                    try:
                        profile = ProfileReport(
                            current_df,
                            title=f"Profiling Report - {selected_file}",
                            explorative=True
                        )
                        profile.to_file("dataset_report.html")
                        with open("dataset_report.html", "rb") as f:
                            html_bytes = f.read()
                        st.success("✅ Profiling report generated successfully!")
                        st.download_button(
                            label="⬇️ Download Profiling Report",
                            data=html_bytes,
                            file_name=f"{selected_file}_report.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"❌ Could not generate report: {e}")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No datasets uploaded yet. Use the section above to add data.")

# =========================================================
# === DATA  Preprocessing Page =====================================
# =========================================================
def data_preprocessing_page():
    """Comprehensive Data Preprocessing & ML Preprocessing - DataMate AI v1"""

    # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -----------------------------



    warnings.filterwarnings("ignore")

    st.markdown('<div class="main-header">🛠 Data Preprocessing Center</div>',
                unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the 'Data Upload' section first.")
        return



    ctl1, ctl2, ctl3, ctl4 = st.columns([1.9, 1.9, 1.9, 3])
    with ctl1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with ctl2:
        if st.button("⏮️ Reset to Original", use_container_width=True):
            if st.session_state.selected_file_name in st.session_state.all_datasets:
                st.session_state.df = st.session_state.all_datasets[
                    st.session_state.selected_file_name].copy()
                st.success("✅ Reverted to original dataset.")
                st.rerun()
    with ctl3:
        if st.button("📸 Save Snapshot", use_container_width=True,
                     help="Save current state as a named version"):
            if "snapshots" not in st.session_state:
                st.session_state.snapshots = {}
            snap_name = f"Snapshot_{len(st.session_state.snapshots)+1}"
            st.session_state.snapshots[snap_name] = st.session_state.df.copy()
            st.success(f"✅ Saved: {snap_name}")

    df = st.session_state.df.copy()


    # Snapshot restore
    if "snapshots" in st.session_state and st.session_state.snapshots:
        with st.expander("🔁 Restore from Snapshot"):
            snap_choice = st.selectbox("Choose snapshot",
                                       list(st.session_state.snapshots.keys()))
            if st.button("Restore This Snapshot"):
                st.session_state.df = st.session_state.snapshots[snap_choice].copy()
                st.success(f"✅ Restored: {snap_choice}")
                st.rerun()

    # ── SECTION TABS ─────────────────────────────────────────────────────────
    (
        tab_overview, tab_missing, tab_dupes, tab_cols,
        tab_types, tab_outliers, tab_incon,
        tab_encode, tab_scale, tab_feature,
        tab_ml, tab_export
    ) = st.tabs([
        "📋 Overview",
        "🔍 Missing Values",
        "♻️ Duplicates",
        "🗂️ Column Mgmt",
        "🔁 Data Types",
        "📉 Outliers",
        "✏️ Inconsistent Data",
        "🔢 Encoding",
        "📐 Scaling",
        "🎯 Feature Engineering",
        "🤖 ML Preprocessing",
        "📤 Export"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0 – OVERVIEW / DATA HEALTH REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown('<div class="section-header">📋 Data Health Overview</div>',
                    unsafe_allow_html=True)

        numeric_cols  = df.select_dtypes(include=['int64','float64']).columns.tolist()
        cat_cols      = df.select_dtypes(include=['object','category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # KPI strip
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            ("Rows",            f"{len(df):,}",                      "#3498db"),
            ("Columns",         f"{len(df.columns)}",                "#3498db"),
            ("Numeric",         f"{len(numeric_cols)}",              "#3498db"),
            ("Categorical",     f"{len(cat_cols)}",                  "#3498db"),
            ("Missing Cells",   f"{df.isnull().sum().sum():,}",      "#3498db" if df.isnull().sum().sum() > 0 else "#3498db"),
            ("Duplicates",      f"{df.duplicated().sum():,}",        "#3498db" if df.duplicated().sum() > 0 else "#3498db"),
        ]
        for col_ui, (label, val, color) in zip([k1,k2,k3,k4,k5,k6], kpis):
            with col_ui:
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {color}; text-align:center;">
                    <p style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
                              letter-spacing:1px;margin:0;">{label}</p>
                    <p style="font-size:22px;font-weight:800;color:{color};
                              margin:6px 0 0 0;font-family:Sora,sans-serif;">{val}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Missing value heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values per Column**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            }).sort_values('Missing', ascending=False)
            missing_df_show = missing_df[missing_df['Missing'] > 0]
            if len(missing_df_show) > 0:
                fig_miss = px.bar(missing_df_show, x='Column', y='Missing %',
                                  color='Missing %',
                                  color_continuous_scale='Reds',
                                  title="Missing % by Column",
                                  template="plotly_white")
                fig_miss.update_layout(height=300, margin=dict(t=40,b=40))
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ No missing values!")

        with col2:
            st.markdown("**Data Type Distribution**")
            dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ['Type', 'Count']
            fig_dtype = px.pie(dtype_counts, names='Type', values='Count',
                               hole=0.45, title="Column Types",
                               template="plotly_white",
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig_dtype.update_layout(height=300, margin=dict(t=40,b=40))
            st.plotly_chart(fig_dtype, use_container_width=True)

        st.markdown("---")

        # Full column profile
        st.markdown("**Full Column Profile**")
        profile_rows = []
        for c in df.columns:
            row = {
                "Column": c,
                "Type": str(df[c].dtype),
                "Non-Null": df[c].count(),
                "Null": df[c].isnull().sum(),
                "Null %": f"{df[c].isnull().mean()*100:.1f}%",
                "Unique": df[c].nunique(),
            }
            if df[c].dtype in ['int64','float64']:
                row["Min"]  = f"{df[c].min():.2f}"
                row["Max"]  = f"{df[c].max():.2f}"
                row["Mean"] = f"{df[c].mean():.2f}"
                row["Std"]  = f"{df[c].std():.2f}"
                sk = skew(df[c].dropna())
                row["Skew"] = f"{sk:.2f}"
            else:
                row["Min"] = row["Max"] = row["Mean"] = row["Std"] = row["Skew"] = "—"
            profile_rows.append(row)
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – MISSING VALUES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_missing:
        st.markdown('<div class="section-header">🔍 Handle Missing Values</div>',
                    unsafe_allow_html=True)

        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
            'Dtype': df.dtypes.astype(str).values
        }).sort_values('Missing', ascending=False)
        missing_summary = missing_summary[missing_summary['Missing'] > 0]

        if len(missing_summary) > 0:
            st.dataframe(missing_summary, use_container_width=True)
            st.markdown("---")

            # Strategy selector
            strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
                "🔧 Single Column", "🔧 Bulk Fill", "🧠 Advanced (KNN / Iterative)"
            ])

            with strategy_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fill_column = st.selectbox("Select Column",
                                               missing_summary['Column'].tolist(),
                                               key="fill_col")
                    fill_method = st.radio("Fill Method", [
                        "Drop Rows", "Mean", "Median", "Mode",
                        "Forward Fill", "Backward Fill",
                        "Interpolate (Linear)", "Custom Value"
                    ], key="fill_method")
                    fill_value = st.text_input("Custom value",
                                               key="fill_value") if fill_method == "Custom Value" else None

                    if st.button("✅ Apply Fill", key="apply_fill"):
                        try:
                            if fill_method == "Drop Rows":
                                df = df.dropna(subset=[fill_column])
                            elif fill_method == "Mean":
                                df[fill_column].fillna(df[fill_column].mean(), inplace=True)
                            elif fill_method == "Median":
                                df[fill_column].fillna(df[fill_column].median(), inplace=True)
                            elif fill_method == "Mode":
                                df[fill_column].fillna(df[fill_column].mode()[0], inplace=True)
                            elif fill_method == "Forward Fill":
                                df[fill_column].fillna(method='ffill', inplace=True)
                            elif fill_method == "Backward Fill":
                                df[fill_column].fillna(method='bfill', inplace=True)
                            elif fill_method == "Interpolate (Linear)":
                                df[fill_column] = df[fill_column].interpolate(method='linear')
                            elif fill_method == "Custom Value" and fill_value:
                                val = float(fill_value) if df[fill_column].dtype in ['int64','float64'] else fill_value
                                df[fill_column].fillna(val, inplace=True)
                            st.session_state.df = df
                            st.success("✅ Applied!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

                with col2:
                    st.markdown("**Drop Columns by Threshold**")
                    threshold = st.slider("Drop if missing % >", 0, 100, 50,
                                          key="missing_threshold")
                    cols_to_drop = missing_summary[
                        missing_summary['Missing %'] > threshold]['Column'].tolist()
                    if cols_to_drop:
                        for c in cols_to_drop:
                            st.write(f"  • {c}")
                        if st.button("Drop These Columns", key="drop_missing_cols"):
                            df = df.drop(columns=cols_to_drop)
                            st.session_state.df = df
                            st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                            st.rerun()
                    else:
                        st.info("No columns exceed the threshold")

            with strategy_tab2:
                st.markdown("**Bulk Fill — Apply same strategy to multiple columns**")
                bulk_cols = st.multiselect("Select columns to fill",
                    missing_summary['Column'].tolist(), key="bulk_cols")
                bulk_method = st.selectbox("Bulk fill method",
                    ["Mean", "Median", "Mode", "Forward Fill",
                     "Backward Fill", "Drop Rows"], key="bulk_method")

                if st.button("✅ Apply Bulk Fill", key="bulk_fill_btn") and bulk_cols:
                    for c in bulk_cols:
                        try:
                            if bulk_method == "Mean" and df[c].dtype in ['int64','float64']:
                                df[c].fillna(df[c].mean(), inplace=True)
                            elif bulk_method == "Median" and df[c].dtype in ['int64','float64']:
                                df[c].fillna(df[c].median(), inplace=True)
                            elif bulk_method == "Mode":
                                df[c].fillna(df[c].mode()[0], inplace=True)
                            elif bulk_method == "Forward Fill":
                                df[c].fillna(method='ffill', inplace=True)
                            elif bulk_method == "Backward Fill":
                                df[c].fillna(method='bfill', inplace=True)
                            elif bulk_method == "Drop Rows":
                                df = df.dropna(subset=[c])
                        except Exception:
                            pass
                    st.session_state.df = df
                    st.success(f"✅ Bulk filled {len(bulk_cols)} columns")
                    st.rerun()

            with strategy_tab3:
                st.markdown("**KNN Imputer** — fills numeric missing values using K nearest neighbours")
                num_missing_cols = [
                    c for c in missing_summary['Column']
                    if df[c].dtype in ['int64','float64']
                ]
                if num_missing_cols:
                    knn_cols = st.multiselect("Select numeric columns for KNN Imputation",
                                              num_missing_cols, key="knn_cols",
                                              default=num_missing_cols)
                    k_neighbors = st.slider("K Neighbors", 2, 20, 5, key="knn_k")
                    if st.button("✅ Apply KNN Imputer", key="knn_btn") and knn_cols:
                        imputer = KNNImputer(n_neighbors=k_neighbors)
                        df[knn_cols] = imputer.fit_transform(df[knn_cols])
                        st.session_state.df = df
                        st.success(f"✅ KNN imputed {len(knn_cols)} columns")
                        st.rerun()
                else:
                    st.info("No numeric columns with missing values for KNN.")
        else:
            st.success("✅ No missing values found! Your data is clean.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – DUPLICATES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_dupes:
        st.markdown('<div class="section-header">♻️ Handle Duplicates</div>',
                    unsafe_allow_html=True)

        dup_count = df.duplicated().sum()
        st.metric("Total Duplicate Rows", f"{dup_count:,}",
                  delta="Clean" if dup_count == 0 else "Found",
                  delta_color="normal" if dup_count == 0 else "inverse")

        col1, col2 = st.columns(2)
        with col1:
            if dup_count > 0:
                st.markdown("**Preview Duplicates**")
                st.dataframe(
                    df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(30),
                    use_container_width=True)
            else:
                st.success("✅ No duplicate rows found.")

        with col2:
            st.markdown("**Remove Options**")
            keep_option = st.radio("Keep",
                ["First occurrence", "Last occurrence", "Remove all"],
                key="keep_option")
            subset_cols = st.multiselect(
                "Subset columns (empty = all columns)",
                df.columns.tolist(), key="subset_cols")
            keep_map = {
                "First occurrence": "first",
                "Last occurrence":  "last",
                "Remove all":       False
            }
            if st.button("🗑️ Remove Duplicates", key="remove_duplicates"):
                orig = len(df)
                df = df.drop_duplicates(
                    subset=subset_cols if subset_cols else None,
                    keep=keep_map[keep_option])
                st.session_state.df = df
                st.success(f"✅ Removed {orig - len(df):,} duplicate rows")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – COLUMN MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cols:
        st.markdown('<div class="section-header">🗂️ Column Management</div>',
                    unsafe_allow_html=True)

        sub1, sub2, sub3 = st.tabs(["Rename / Drop", "Reorder", "Create New Column"])

        with sub1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Rename Column**")
                col_to_rename = st.selectbox("Column", df.columns.tolist(), key="rename_col")
                new_name = st.text_input("New name", value=col_to_rename, key="new_col_name")
                if st.button("Rename", key="rename_btn"):
                    df = df.rename(columns={col_to_rename: new_name})
                    st.session_state.df = df
                    st.success(f"✅ '{col_to_rename}' → '{new_name}'")
                    st.rerun()
            with col2:
                st.markdown("**Drop Columns**")
                cols_to_drop = st.multiselect("Select columns to drop",
                                              df.columns.tolist(), key="cols_to_drop")
                if st.button("🗑️ Drop Selected", key="drop_cols_btn") and cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    st.session_state.df = df
                    st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                    st.rerun()

            st.markdown("---")
            st.dataframe(pd.DataFrame({
                'Column': df.columns, 'Type': df.dtypes.astype(str),
                'Non-Null': df.count(), 'Null': df.isnull().sum(),
                'Unique': df.nunique()
            }), use_container_width=True)

        with sub2:
            st.markdown("**Reorder Columns** — drag and rearrange")
            ordered = st.multiselect(
                "Arrange columns in desired order:",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                key="col_reorder"
            )
            if st.button("✅ Apply Order", key="reorder_btn") and ordered:
                remaining = [c for c in df.columns if c not in ordered]
                df = df[ordered + remaining]
                st.session_state.df = df
                st.success("✅ Columns reordered")
                st.rerun()

        with sub3:
            st.markdown("**Create Derived Column**")
            new_col_name = st.text_input("New column name", key="new_derived_col")
            numeric_cols_d = df.select_dtypes(include=['int64','float64']).columns.tolist()
            col_a = st.selectbox("Column A", numeric_cols_d, key="derived_a")
            operation = st.selectbox("Operation",
                ["+", "−", "×", "÷", "log(A)", "sqrt(A)",
                 "A²", "A % B", "abs(A)"], key="derived_op")
            col_b = st.selectbox("Column B (if needed)", numeric_cols_d, key="derived_b")

            if st.button("✅ Create Column", key="create_derived_btn") and new_col_name:
                try:
                    if operation == "+":       df[new_col_name] = df[col_a] + df[col_b]
                    elif operation == "−":     df[new_col_name] = df[col_a] - df[col_b]
                    elif operation == "×":     df[new_col_name] = df[col_a] * df[col_b]
                    elif operation == "÷":     df[new_col_name] = df[col_a] / df[col_b].replace(0, np.nan)
                    elif operation == "log(A)":  df[new_col_name] = np.log1p(df[col_a].clip(lower=0))
                    elif operation == "sqrt(A)": df[new_col_name] = np.sqrt(df[col_a].clip(lower=0))
                    elif operation == "A²":      df[new_col_name] = df[col_a] ** 2
                    elif operation == "A % B":   df[new_col_name] = df[col_a] % df[col_b].replace(0, np.nan)
                    elif operation == "abs(A)":  df[new_col_name] = df[col_a].abs()
                    st.session_state.df = df
                    st.success(f"✅ Created column '{new_col_name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – DATA TYPES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_types:
        st.markdown('<div class="section-header">🔁 Data Type Conversion</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Single Column Conversion**")
            col_to_convert = st.selectbox("Select Column",
                                          df.columns.tolist(), key="convert_col")
            st.caption(f"Current type: **{df[col_to_convert].dtype}** | "
                       f"Sample: `{df[col_to_convert].dropna().iloc[0] if len(df[col_to_convert].dropna()) > 0 else 'N/A'}`")
            new_type = st.selectbox("Convert To",
                ["int64","float64","string","category",
                 "datetime64[ns]","bool"], key="new_type")
            fmt_str = st.text_input("Datetime format (optional, e.g. %Y-%m-%d)",
                                    key="dt_fmt") if new_type == "datetime64[ns]" else None

            if st.button("✅ Convert", key="convert_btn"):
                try:
                    if new_type == "datetime64[ns]":
                        df[col_to_convert] = pd.to_datetime(
                            df[col_to_convert], format=fmt_str if fmt_str else None,
                            errors='coerce')
                    elif new_type == "bool":
                        df[col_to_convert] = df[col_to_convert].astype(bool)
                    else:
                        df[col_to_convert] = df[col_to_convert].astype(new_type)
                    st.session_state.df = df
                    st.success(f"✅ Converted to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

        with col2:
            st.markdown("**Bulk Type Conversion**")
            st.caption("Convert all object columns that look numeric/datetime automatically.")
            if st.button("🔍 Auto-Detect & Convert Numerics", key="auto_num"):
                converted = []
                for c in df.select_dtypes(include='object').columns:
                    converted_series = pd.to_numeric(df[c], errors='coerce')
                    if converted_series.notna().sum() / len(df) > 0.8:
                        df[c] = converted_series
                        converted.append(c)
                st.session_state.df = df
                st.success(f"✅ Converted: {converted if converted else 'None found'}")
                st.rerun()

            if st.button("🔍 Auto-Detect & Convert Datetimes", key="auto_dt"):
                converted = []
                for c in df.select_dtypes(include='object').columns:
                    try:
                        dt_series = pd.to_datetime(df[c], infer_datetime_format=True,
                                                   errors='coerce')
                        if dt_series.notna().sum() / len(df) > 0.7:
                            df[c] = dt_series
                            converted.append(c)
                    except Exception:
                        pass
                st.session_state.df = df
                st.success(f"✅ Converted: {converted if converted else 'None found'}")
                st.rerun()

            st.markdown("**Type Summary**")
            for dtype, count in df.dtypes.value_counts().items():
                st.markdown(f"""
                <div class="metric-card" style="padding:10px 14px;">
                    <span style="font-weight:700;color:#1a2535;">{dtype}</span>
                    <span style="color:#3498db;font-weight:600;float:right;">
                        {count} col(s)
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 – OUTLIERS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_outliers:
        st.markdown('<div class="section-header">📉 Outlier Detection & Treatment</div>',
                    unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns for outlier detection.")
        else:
            outlier_col = st.selectbox("Select Numeric Column", numeric_cols, key="outlier_col")

            # Distribution preview
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df[outlier_col].dropna(),
                                     name=outlier_col,
                                     marker_color='#3498db',
                                     boxmean='sd'))
            fig_box.update_layout(title=f"Distribution: {outlier_col}",
                                  template="plotly_white", height=280,
                                  margin=dict(t=40,b=20))
            st.plotly_chart(fig_box, use_container_width=True)

            # Stats
            s = df[outlier_col].dropna()
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            for col_ui, (lbl, val) in zip(
                [sc1,sc2,sc3,sc4,sc5],
                [("Mean", f"{s.mean():.2f}"),
                 ("Std",  f"{s.std():.2f}"),
                 ("Skew", f"{skew(s):.2f}"),
                 ("Kurt", f"{kurtosis(s):.2f}"),
                 ("Range",f"{s.max()-s.min():.2f}")]
            ):
                with col_ui:
                    st.metric(lbl, val)

            st.markdown("---")
            method_tab1, method_tab2, method_tab3, method_tab4 = st.tabs([
                "IQR Method", "Z-Score", "Percentile Clip", "Winsorization"
            ])

            with method_tab1:
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_mult")
                lb, ub = Q1 - multiplier * IQR, Q3 + multiplier * IQR
                mask = (df[outlier_col] < lb) | (df[outlier_col] > ub)
                outliers_iqr = df[mask]
                st.write(f"Bounds: [{lb:.2f}, {ub:.2f}] | Outliers: **{len(outliers_iqr)}**")
                if len(outliers_iqr) > 0:
                    with st.expander("Preview Outlier Rows"):
                        st.dataframe(outliers_iqr.head(20), use_container_width=True)
                action_iqr = st.radio("Treatment",
                    ["Remove Rows", "Cap to Bounds"], key="iqr_action")
                if st.button("✅ Apply IQR Treatment", key="iqr_btn"):
                    if action_iqr == "Remove Rows":
                        df = df[(df[outlier_col] >= lb) & (df[outlier_col] <= ub)]
                    else:
                        df[outlier_col] = df[outlier_col].clip(lower=lb, upper=ub)
                    st.session_state.df = df
                    st.success("✅ Applied IQR treatment")
                    st.rerun()

            with method_tab2:
                z_thresh = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z_thresh")
                z_scores = np.abs(stats.zscore(s))
                outlier_idx = s[z_scores > z_thresh].index
                outliers_z = df.loc[outlier_idx]
                st.write(f"Threshold: {z_thresh} | Outliers: **{len(outliers_z)}**")
                if len(outliers_z) > 0:
                    with st.expander("Preview Outlier Rows"):
                        st.dataframe(outliers_z.head(20), use_container_width=True)
                action_z = st.radio("Treatment",
                    ["Remove Rows", "Replace with Mean", "Replace with Median"],
                    key="z_action")
                if st.button("✅ Apply Z-Score Treatment", key="z_btn"):
                    if action_z == "Remove Rows":
                        df = df.drop(index=outlier_idx)
                    elif action_z == "Replace with Mean":
                        df.loc[outlier_idx, outlier_col] = s.mean()
                    else:
                        df.loc[outlier_idx, outlier_col] = s.median()
                    st.session_state.df = df
                    st.success("✅ Applied Z-Score treatment")
                    st.rerun()

            with method_tab3:
                p_low  = st.slider("Lower Percentile", 0, 10, 1, key="p_low")
                p_high = st.slider("Upper Percentile", 90, 100, 99, key="p_high")
                low_val  = s.quantile(p_low  / 100)
                high_val = s.quantile(p_high / 100)
                outside  = ((df[outlier_col] < low_val) | (df[outlier_col] > high_val)).sum()
                st.write(f"Clip range: [{low_val:.2f}, {high_val:.2f}] | "
                         f"Values outside: **{outside}**")
                if st.button("✅ Apply Percentile Clip", key="pct_btn"):
                    df[outlier_col] = df[outlier_col].clip(lower=low_val, upper=high_val)
                    st.session_state.df = df
                    st.success("✅ Clipped!")
                    st.rerun()

            with method_tab4:
                st.info("Winsorization replaces extreme values with the nearest non-extreme value.")
                w_low  = st.slider("Lower % to winsorize", 0.0, 10.0, 1.0, 0.5, key="w_low")
                w_high = st.slider("Upper % to winsorize", 0.0, 10.0, 1.0, 0.5, key="w_high")
                if st.button("✅ Apply Winsorization", key="wins_btn"):
                    from scipy.stats.mstats import winsorize
                    df[outlier_col] = winsorize(df[outlier_col],
                                                limits=[w_low/100, w_high/100])
                    st.session_state.df = df
                    st.success("✅ Winsorized!")
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 – INCONSISTENT DATA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_incon:
        st.markdown('<div class="section-header">✏️ Fix Inconsistent Data</div>',
                    unsafe_allow_html=True)

        text_cols = df.select_dtypes(include='object').columns.tolist()

        sub_a, sub_b, sub_c, sub_d = st.tabs([
            "Whitespace & Case", "Value Replace / Regex",
            "Category Standardize", "Date Parsing"
        ])

        with sub_a:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Trim Whitespace**")
                trim_cols = st.multiselect("Select columns", text_cols, key="trim_cols")
                if st.button("✅ Trim", key="trim_btn") and trim_cols:
                    for c in trim_cols:
                        df[c] = df[c].astype(str).str.strip()
                    st.session_state.df = df
                    st.success("✅ Trimmed")
                    st.rerun()

            with col2:
                st.markdown("**Case Conversion**")
                if text_cols:
                    case_col = st.selectbox("Column", text_cols, key="case_col")
                    case_opt = st.radio("Convert to",
                        ["lowercase","UPPERCASE","Title Case","Sentence case"],
                        key="case_opt")
                    if st.button("✅ Convert Case", key="case_btn"):
                        if case_opt == "lowercase":
                            df[case_col] = df[case_col].str.lower()
                        elif case_opt == "UPPERCASE":
                            df[case_col] = df[case_col].str.upper()
                        elif case_opt == "Title Case":
                            df[case_col] = df[case_col].str.title()
                        else:
                            df[case_col] = df[case_col].str.capitalize()
                        st.session_state.df = df
                        st.success("✅ Applied")
                        st.rerun()

        with sub_b:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Find & Replace Value**")
                rep_col  = st.selectbox("Column", df.columns.tolist(), key="rep_col")
                old_val  = st.text_input("Find value", key="rep_old")
                new_val  = st.text_input("Replace with", key="rep_new")
                if st.button("✅ Replace", key="rep_btn") and old_val:
                    df[rep_col] = df[rep_col].astype(str).str.replace(
                        old_val, new_val, regex=False)
                    st.session_state.df = df
                    st.success(f"✅ Replaced '{old_val}' → '{new_val}'")
                    st.rerun()

            with col2:
                st.markdown("**Regex Replace**")
                regex_col     = st.selectbox("Column", text_cols, key="regex_col") if text_cols else None
                regex_pattern = st.text_input("Regex pattern (e.g. [^a-zA-Z0-9])", key="regex_pat")
                regex_replace = st.text_input("Replacement string", key="regex_rep")
                if st.button("✅ Apply Regex", key="regex_btn") and regex_col and regex_pattern:
                    try:
                        df[regex_col] = df[regex_col].astype(str).str.replace(
                            regex_pattern, regex_replace, regex=True)
                        st.session_state.df = df
                        st.success("✅ Regex applied")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        with sub_c:
            st.markdown("**Category Value Mapping**")
            if text_cols:
                map_col = st.selectbox("Column", text_cols, key="map_col")
                unique_vals = df[map_col].dropna().unique().tolist()
                st.caption(f"Unique values ({len(unique_vals)}): {unique_vals[:15]}")

                st.markdown("Enter new label for each value (leave blank to keep):")
                mapping = {}
                chunk_cols = st.columns(3)
                for i, v in enumerate(unique_vals[:15]):
                    with chunk_cols[i % 3]:
                        mapped = st.text_input(f"`{v}`", key=f"map_{map_col}_{i}",
                                               placeholder=str(v))
                        if mapped.strip():
                            mapping[v] = mapped.strip()

                if st.button("✅ Apply Mapping", key="apply_map_btn") and mapping:
                    df[map_col] = df[map_col].replace(mapping)
                    st.session_state.df = df
                    st.success(f"✅ Mapped {len(mapping)} values in '{map_col}'")
                    st.rerun()

        with sub_d:
            st.markdown("**Parse & Extract Date Parts**")
            dt_cols = df.select_dtypes(include='datetime64').columns.tolist()
            obj_cols_d = df.select_dtypes(include='object').columns.tolist()
            all_date_candidates = dt_cols + obj_cols_d

            if all_date_candidates:
                date_src = st.selectbox("Date column", all_date_candidates, key="date_src")
                parts = st.multiselect("Extract parts",
                    ["Year","Month","Day","Hour","Minute",
                     "DayOfWeek","Quarter","WeekOfYear"],
                    default=["Year","Month","Day"], key="date_parts")

                if st.button("✅ Extract Date Parts", key="date_ext_btn") and parts:
                    try:
                        series = pd.to_datetime(df[date_src], errors='coerce')
                        if "Year"       in parts: df[f"{date_src}_year"]    = series.dt.year
                        if "Month"      in parts: df[f"{date_src}_month"]   = series.dt.month
                        if "Day"        in parts: df[f"{date_src}_day"]     = series.dt.day
                        if "Hour"       in parts: df[f"{date_src}_hour"]    = series.dt.hour
                        if "Minute"     in parts: df[f"{date_src}_minute"]  = series.dt.minute
                        if "DayOfWeek"  in parts: df[f"{date_src}_dow"]     = series.dt.dayofweek
                        if "Quarter"    in parts: df[f"{date_src}_quarter"] = series.dt.quarter
                        if "WeekOfYear" in parts: df[f"{date_src}_week"]    = series.dt.isocalendar().week.astype(int)
                        st.session_state.df = df
                        st.success(f"✅ Extracted {len(parts)} date features")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 – ENCODING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_encode:
        st.markdown('<div class="section-header">🔢 Categorical Encoding</div>',
                    unsafe_allow_html=True)

        cat_cols_enc = df.select_dtypes(include=['object','category']).columns.tolist()

        if not cat_cols_enc:
            st.info("No categorical columns found.")
        else:
            enc_tab1, enc_tab2, enc_tab3, enc_tab4, enc_tab5 = st.tabs([
                "Label Encoding", "One-Hot Encoding",
                "Ordinal Encoding", "Binary Encoding", "Target Encoding"
            ])

            with enc_tab1:
                le_col = st.selectbox("Column", cat_cols_enc, key="le_col")
                if st.button("✅ Apply Label Encoding", key="le_btn"):
                    le = LabelEncoder()
                    df[f"{le_col}_label_enc"] = le.fit_transform(df[le_col].astype(str))
                    st.session_state.df = df
                    st.success(f"✅ Created '{le_col}_label_enc'")
                    st.dataframe(pd.DataFrame({
                        'Original': le.classes_,
                        'Encoded': range(len(le.classes_))
                    }), use_container_width=True)
                    st.rerun()

            with enc_tab2:
                ohe_col  = st.selectbox("Column", cat_cols_enc, key="ohe_col")
                drop_first = st.checkbox("Drop first (avoid dummy trap)", value=True, key="drop_first")
                unique_count = df[ohe_col].nunique()
                st.write(f"Unique values: **{unique_count}**")
                if unique_count > 15:
                    st.warning("⚠️ High cardinality. Consider Label or Binary encoding instead.")
                if st.button("✅ Apply One-Hot Encoding", key="ohe_btn"):
                    ohe_df = pd.get_dummies(df[[ohe_col]], prefix=ohe_col,
                                            drop_first=drop_first)
                    df = pd.concat([df, ohe_df], axis=1)
                    st.session_state.df = df
                    st.success(f"✅ Created {len(ohe_df.columns)} OHE columns")
                    st.rerun()

            with enc_tab3:
                ord_col = st.selectbox("Column", cat_cols_enc, key="ord_col")
                unique_vals_ord = df[ord_col].dropna().unique().tolist()
                st.caption("Drag to set order (low → high):")
                ordered_cats = st.multiselect(
                    "Ordered categories (first = lowest rank):",
                    options=unique_vals_ord,
                    default=unique_vals_ord, key="ord_cats")
                if st.button("✅ Apply Ordinal Encoding", key="ord_btn") and ordered_cats:
                    cat_map = {v: i for i, v in enumerate(ordered_cats)}
                    df[f"{ord_col}_ordinal"] = df[ord_col].map(cat_map)
                    st.session_state.df = df
                    st.success(f"✅ Created '{ord_col}_ordinal'")
                    st.rerun()

            with enc_tab4:
                bin_col = st.selectbox("Column", cat_cols_enc, key="bin_col")
                if st.button("✅ Apply Binary Encoding", key="bin_btn"):
                    le_b = LabelEncoder()
                    le_encoded = le_b.fit_transform(df[bin_col].astype(str))
                    n_bits = max(1, int(np.ceil(np.log2(len(le_b.classes_) + 1))))
                    for bit in range(n_bits):
                        df[f"{bin_col}_bin_{bit}"] = (le_encoded >> bit) & 1
                    st.session_state.df = df
                    st.success(f"✅ Binary encoded into {n_bits} bit columns")
                    st.rerun()

            with enc_tab5:
                st.info("Target encoding replaces category with mean of the target variable.")
                tgt_enc_col = st.selectbox("Feature Column", cat_cols_enc, key="tgt_enc_col")
                numeric_for_target = df.select_dtypes(include=['int64','float64']).columns.tolist()
                if numeric_for_target:
                    target_col_enc = st.selectbox("Target Column (numeric)",
                                                   numeric_for_target, key="target_col_enc")
                    smoothing = st.slider("Smoothing factor", 0.0, 10.0, 1.0, key="te_smooth")
                    if st.button("✅ Apply Target Encoding", key="te_btn"):
                        global_mean = df[target_col_enc].mean()
                        stats_df = df.groupby(tgt_enc_col)[target_col_enc].agg(['mean','count'])
                        smoothed = (stats_df['count'] * stats_df['mean'] + smoothing * global_mean) \
                                   / (stats_df['count'] + smoothing)
                        df[f"{tgt_enc_col}_target_enc"] = df[tgt_enc_col].map(smoothed)
                        st.session_state.df = df
                        st.success(f"✅ Created '{tgt_enc_col}_target_enc'")
                        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 – SCALING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_scale:
        st.markdown('<div class="section-header">📐 Feature Scaling</div>',
                    unsafe_allow_html=True)

        numeric_cols_sc = df.select_dtypes(include=['int64','float64']).columns.tolist()

        if not numeric_cols_sc:
            st.info("No numeric columns available.")
        else:
            scale_tab1, scale_tab2, scale_tab3, scale_tab4, scale_tab5 = st.tabs([
                "Min-Max Norm.", "Standard (Z)", "Robust Scaler",
                "MaxAbs Scaler", "Power / Quantile"
            ])

            def apply_scaler(scaler_obj, cols, suffix):
                df_temp = df.copy()
                df_temp[cols] = scaler_obj.fit_transform(df_temp[cols])
                for c in cols:
                    df[f"{c}_{suffix}"] = df_temp[c]
                st.session_state.df = df
                st.success(f"✅ Applied to {len(cols)} columns (suffix: _{suffix})")
                st.rerun()

            with scale_tab1:
                st.caption("Scales each feature to [0, 1] range. Sensitive to outliers.")
                norm_cols = st.multiselect("Columns", numeric_cols_sc, key="norm_cols",
                                           default=numeric_cols_sc[:3])
                feat_range = st.slider("Feature range", 0, 1, (0, 1), key="feat_range")
                in_place = st.checkbox("Replace original columns", key="norm_inplace")
                if st.button("✅ Apply Min-Max", key="norm_btn") and norm_cols:
                    scaler = MinMaxScaler(feature_range=feat_range)
                    if in_place:
                        df[norm_cols] = scaler.fit_transform(df[norm_cols])
                        st.session_state.df = df
                        st.success("✅ Normalized in-place")
                        st.rerun()
                    else:
                        apply_scaler(scaler, norm_cols, "norm")

            with scale_tab2:
                st.caption("Zero mean, unit variance. Best for normally distributed data.")
                std_cols = st.multiselect("Columns", numeric_cols_sc, key="std_cols",
                                          default=numeric_cols_sc[:3])
                in_place_std = st.checkbox("Replace original", key="std_inplace")
                if st.button("✅ Apply Standardization", key="std_btn") and std_cols:
                    scaler = StandardScaler()
                    if in_place_std:
                        df[std_cols] = scaler.fit_transform(df[std_cols])
                        st.session_state.df = df
                        st.success("✅ Standardized in-place")
                        st.rerun()
                    else:
                        apply_scaler(scaler, std_cols, "std")

            with scale_tab3:
                st.caption("Uses median & IQR. **Robust to outliers.** Best when outliers exist.")
                rob_cols = st.multiselect("Columns", numeric_cols_sc, key="rob_cols",
                                          default=numeric_cols_sc[:3])
                q_range  = st.slider("Quantile range (%)", 5, 45, (25, 75), key="rob_q")
                if st.button("✅ Apply Robust Scaler", key="rob_btn") and rob_cols:
                    apply_scaler(RobustScaler(quantile_range=q_range), rob_cols, "robust")

            with scale_tab4:
                st.caption("Scales by max absolute value. Keeps sparsity. Range [-1, 1].")
                mabs_cols = st.multiselect("Columns", numeric_cols_sc, key="mabs_cols",
                                           default=numeric_cols_sc[:3])
                if st.button("✅ Apply MaxAbs", key="mabs_btn") and mabs_cols:
                    apply_scaler(MaxAbsScaler(), mabs_cols, "maxabs")

            with scale_tab5:
                t_type = st.radio("Transformer type",
                    ["Yeo-Johnson Power", "Box-Cox Power", "Quantile (Uniform)",
                     "Quantile (Normal)"], key="pwr_type")
                pwr_cols = st.multiselect("Columns", numeric_cols_sc, key="pwr_cols",
                                          default=numeric_cols_sc[:3])
                if st.button("✅ Apply Transformer", key="pwr_btn") and pwr_cols:
                    try:
                        if t_type == "Yeo-Johnson Power":
                            apply_scaler(PowerTransformer(method='yeo-johnson'), pwr_cols, "yeojohn")
                        elif t_type == "Box-Cox Power":
                            apply_scaler(PowerTransformer(method='box-cox'), pwr_cols, "boxcox")
                        elif t_type == "Quantile (Uniform)":
                            apply_scaler(QuantileTransformer(output_distribution='uniform',
                                                             random_state=42), pwr_cols, "q_uni")
                        elif t_type == "Quantile (Normal)":
                            apply_scaler(QuantileTransformer(output_distribution='normal',
                                                             random_state=42), pwr_cols, "q_norm")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9 – FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_feature:
        st.markdown('<div class="section-header">🎯 Feature Engineering</div>',
                    unsafe_allow_html=True)

        numeric_cols_fe = df.select_dtypes(include=['int64','float64']).columns.tolist()

        fe_tab1, fe_tab2, fe_tab3 = st.tabs([
            "Feature Selection", "Dimensionality Reduction (PCA)",
            "Interaction & Polynomial Features"
        ])

        with fe_tab1:
            st.markdown("**Variance Threshold** — remove near-zero variance features")
            if numeric_cols_fe:
                vt_threshold = st.slider("Min variance threshold", 0.0, 1.0, 0.01, 0.01,
                                         key="vt_thresh")
                sel = VarianceThreshold(threshold=vt_threshold)
                try:
                    sel.fit(df[numeric_cols_fe].fillna(0))
                    dropped_vt = [c for c, s in zip(numeric_cols_fe, sel.get_support()) if not s]
                    kept_vt    = [c for c, s in zip(numeric_cols_fe, sel.get_support()) if s]
                    st.write(f"Will drop {len(dropped_vt)} columns: {dropped_vt}")
                    st.write(f"Will keep {len(kept_vt)} columns")
                    if st.button("✅ Remove Low Variance Features", key="vt_btn") and dropped_vt:
                        df = df.drop(columns=dropped_vt)
                        st.session_state.df = df
                        st.success(f"✅ Dropped {len(dropped_vt)} features")
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

                st.markdown("---")
                st.markdown("**SelectKBest** — supervised feature selection")
                target_fs = st.selectbox("Target column", df.columns.tolist(), key="fs_target")
                k_best    = st.slider("K best features", 1,
                                      min(20, len(numeric_cols_fe)), 5, key="k_best")
                fs_type   = st.radio("Problem type", ["Regression","Classification"], key="fs_type")
                if st.button("✅ Select K Best Features", key="kbest_btn"):
                    try:
                        feature_cols_fs = [c for c in numeric_cols_fe if c != target_fs]
                        X_fs = df[feature_cols_fs].fillna(0)
                        y_fs = df[target_fs].fillna(0)
                        score_fn = f_regression if fs_type == "Regression" else f_classif
                        selector = SelectKBest(score_func=score_fn, k=k_best)
                        selector.fit(X_fs, y_fs)
                        scores_df = pd.DataFrame({
                            'Feature': feature_cols_fs,
                            'Score':   selector.scores_,
                            'P-Value': selector.pvalues_
                        }).sort_values('Score', ascending=False)
                        st.dataframe(scores_df, use_container_width=True)
                        selected = scores_df.head(k_best)['Feature'].tolist()
                        st.success(f"Top {k_best} features: {selected}")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        with fe_tab2:
            st.markdown("**PCA — Principal Component Analysis**")
            if len(numeric_cols_fe) >= 2:
                pca_cols   = st.multiselect("Select features for PCA",
                                            numeric_cols_fe,
                                            default=numeric_cols_fe[:min(10, len(numeric_cols_fe))],
                                            key="pca_cols")
                n_comp     = st.slider("Number of components", 1,
                                       min(len(pca_cols), 20) if pca_cols else 2,
                                       min(2, len(pca_cols)) if pca_cols else 2,
                                       key="pca_n")
                keep_orig  = st.checkbox("Keep original columns", value=True, key="pca_keep")

                if st.button("✅ Apply PCA", key="pca_btn") and pca_cols and n_comp <= len(pca_cols):
                    try:
                        X_pca = df[pca_cols].fillna(0)
                        X_scaled = StandardScaler().fit_transform(X_pca)
                        pca = PCA(n_components=n_comp, random_state=42)
                        pca_result = pca.fit_transform(X_scaled)
                        for i in range(n_comp):
                            df[f"PCA_{i+1}"] = pca_result[:, i]
                        if not keep_orig:
                            df = df.drop(columns=pca_cols)
                        st.session_state.df = df
                        evr = pca.explained_variance_ratio_
                        st.success(f"✅ PCA applied. Explained variance: "
                                   f"{[f'{v:.1%}' for v in evr]}")
                        # Scree plot
                        fig_scree = px.bar(x=[f"PC{i+1}" for i in range(n_comp)],
                                           y=evr * 100,
                                           labels={'x':'Component','y':'Explained Variance %'},
                                           title="PCA Scree Plot",
                                           template="plotly_white",
                                           color=evr * 100,
                                           color_continuous_scale='Blues')
                        fig_scree.update_layout(height=300, margin=dict(t=40,b=20))
                        st.plotly_chart(fig_scree, use_container_width=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("Need at least 2 numeric columns for PCA.")

        with fe_tab3:
            st.markdown("**Polynomial & Interaction Features**")
            if len(numeric_cols_fe) >= 2:
                poly_col_a = st.selectbox("Column A", numeric_cols_fe, key="poly_a")
                poly_col_b = st.selectbox("Column B", numeric_cols_fe, key="poly_b")
                poly_ops   = st.multiselect("Generate features",
                    ["A × B", "A + B", "A − B", "A²", "B²",
                     "A³", "log(A)", "log(B)", "√A", "√B",
                     "A / (B+1)", "(A−B)²"],
                    default=["A × B","A²","log(A)"], key="poly_ops")

                if st.button("✅ Generate Features", key="poly_btn") and poly_ops:
                    created = []
                    for op in poly_ops:
                        try:
                            a, b = df[poly_col_a], df[poly_col_b]
                            name_a, name_b = poly_col_a[:8], poly_col_b[:8]
                            if op == "A × B":    df[f"{name_a}x{name_b}"]   = a * b;  created.append(f"{name_a}x{name_b}")
                            elif op == "A + B":  df[f"{name_a}+{name_b}"]   = a + b;  created.append(f"{name_a}+{name_b}")
                            elif op == "A − B":  df[f"{name_a}-{name_b}"]   = a - b;  created.append(f"{name_a}-{name_b}")
                            elif op == "A²":     df[f"{name_a}_sq"]         = a ** 2; created.append(f"{name_a}_sq")
                            elif op == "B²":     df[f"{name_b}_sq"]         = b ** 2; created.append(f"{name_b}_sq")
                            elif op == "A³":     df[f"{name_a}_cu"]         = a ** 3; created.append(f"{name_a}_cu")
                            elif op == "log(A)": df[f"log_{name_a}"]        = np.log1p(a.clip(0)); created.append(f"log_{name_a}")
                            elif op == "log(B)": df[f"log_{name_b}"]        = np.log1p(b.clip(0)); created.append(f"log_{name_b}")
                            elif op == "√A":     df[f"sqrt_{name_a}"]       = np.sqrt(a.clip(0)); created.append(f"sqrt_{name_a}")
                            elif op == "√B":     df[f"sqrt_{name_b}"]       = np.sqrt(b.clip(0)); created.append(f"sqrt_{name_b}")
                            elif op == "A / (B+1)": df[f"{name_a}_div_{name_b}"] = a / (b + 1); created.append(f"{name_a}_div_{name_b}")
                            elif op == "(A−B)²": df[f"({name_a}-{name_b})_sq"]   = (a - b) ** 2; created.append(f"({name_a}-{name_b})_sq")
                        except Exception:
                            pass
                    st.session_state.df = df
                    st.success(f"✅ Created: {created}")
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 10 – ML PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ml:
        st.markdown('<div class="section-header">🤖 ML-Specific Preprocessing</div>',
                    unsafe_allow_html=True)

        ml_tab1, ml_tab2, ml_tab3, ml_tab4, ml_tab5 = st.tabs([
            "Train-Test Split", "Class Imbalance (SMOTE)",
            "Correlation Filter", "Normality Test",
            "Final ML Readiness Check"
        ])

        with ml_tab1:
            st.markdown("**Train / Validation / Test Split**")
            numeric_cols_ml = df.select_dtypes(include=['int64','float64']).columns.tolist()
            all_cols_ml     = df.columns.tolist()

            target_ml = st.selectbox("Target column (Y)", all_cols_ml, key="ml_target",
                                     index=len(all_cols_ml)-1)
            features_ml = st.multiselect("Feature columns (X)",
                [c for c in all_cols_ml if c != target_ml],
                default=[c for c in all_cols_ml if c != target_ml][:10],
                key="ml_features")

            use_val = st.checkbox("Include Validation Set", value=False, key="use_val")
            test_pct = st.slider("Test %", 5, 40, 20, key="test_pct")
            val_pct  = st.slider("Validation % (of remaining)", 5, 30, 15,
                                 key="val_pct") if use_val else 0
            random_seed = st.number_input("Random Seed", 0, 9999, 42, key="rand_seed")
            stratify_split = st.checkbox("Stratify split (classification only)",
                                         value=False, key="strat_split")

            if st.button("✅ Create Split", key="split_btn") and features_ml:
                from sklearn.model_selection import train_test_split
                try:
                    X = df[features_ml].select_dtypes(include=['int64','float64']).fillna(0)
                    y = df[target_ml]
                    strat = y if stratify_split else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_pct/100,
                        random_state=random_seed, stratify=strat)
                    if use_val:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train, test_size=val_pct/100,
                            random_state=random_seed)
                        st.session_state['ml_X_val'] = X_val
                        st.session_state['ml_y_val'] = y_val

                    st.session_state['ml_X_train'] = X_train
                    st.session_state['ml_X_test']  = X_test
                    st.session_state['ml_y_train'] = y_train
                    st.session_state['ml_y_test']  = y_test

                    total = len(X)
                    st.success("✅ Split complete!")
                    s1, s2, s3 = st.columns(3)
                    with s1: st.metric("Train", f"{len(X_train):,} ({len(X_train)/total:.0%})")
                    with s2: st.metric("Test",  f"{len(X_test):,}  ({len(X_test)/total:.0%})")
                    if use_val:
                        with s3: st.metric("Val", f"{len(X_val):,} ({len(X_val)/total:.0%})")
                except Exception as e:
                    st.error(f"❌ {str(e)}")

        with ml_tab2:
            st.markdown("**Class Imbalance Handling**")
            cat_target_cols = df.select_dtypes(include=['object','category']).columns.tolist()
            int_cols_imb    = df.select_dtypes(include=['int64']).columns.tolist()
            imb_target = st.selectbox("Target column (classification)",
                                      cat_target_cols + int_cols_imb, key="imb_target")

            # NEW CORRECTED CODE
            if imb_target:
                # 1. Get value counts and reset index
                vc = df[imb_target].value_counts().reset_index()
                
                # 2. Explicitly rename columns to ensure compatibility across Pandas versions
                # Newer Pandas uses [imb_target, 'count'], older used ['index', imb_target]
                vc.columns = ['Class', 'Count'] 
                
                st.dataframe(vc, use_container_width=True)

                # 3. Use the new explicit names ('Class' and 'Count') in Plotly
                fig_imb = px.bar(vc, 
                                x='Class', 
                                y='Count',
                                title="Class Distribution",
                                template="plotly_white",
                                color='Class',
                                color_discrete_sequence=px.colors.qualitative.Set2)
                
                fig_imb.update_layout(height=280, margin=dict(t=40,b=20))
                st.plotly_chart(fig_imb, use_container_width=True)

                imb_method = st.selectbox("Resampling method",
                    ["Random Oversampling", "Random Undersampling",
                     "SMOTE (requires imbalanced-learn)"], key="imb_method")

                if st.button("✅ Apply Resampling", key="imb_btn"):
                    try:
                        feat_cols_imb = df.select_dtypes(include=['int64','float64']).columns
                        feat_cols_imb = [c for c in feat_cols_imb if c != imb_target]
                        X_imb = df[feat_cols_imb].fillna(0)
                        y_imb = df[imb_target]

                        if imb_method == "Random Oversampling":
                            max_count = vc.max()
                            parts = [df]
                            for cls, cnt in vc.items():
                                if cnt < max_count:
                                    oversampled = df[df[imb_target] == cls].sample(
                                        max_count - cnt, replace=True, random_state=42)
                                    parts.append(oversampled)
                            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
                            st.session_state.df = df
                            st.success(f"✅ Oversampled to {len(df):,} rows")

                        elif imb_method == "Random Undersampling":
                            min_count = vc.min()
                            parts = [df[df[imb_target] == cls].sample(
                                min_count, random_state=42) for cls in vc.index]
                            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
                            st.session_state.df = df
                            st.success(f"✅ Undersampled to {len(df):,} rows")

                        elif imb_method == "SMOTE (requires imbalanced-learn)":
                            from imblearn.over_sampling import SMOTE
                            le_imb = LabelEncoder()
                            y_enc  = le_imb.fit_transform(y_imb)
                            sm     = SMOTE(random_state=42)
                            X_res, y_res = sm.fit_resample(X_imb, y_enc)
                            df_res = pd.DataFrame(X_res, columns=feat_cols_imb)
                            df_res[imb_target] = le_imb.inverse_transform(y_res)
                            st.session_state.df = df_res
                            st.success(f"✅ SMOTE applied: {len(df_res):,} rows")
                        st.rerun()
                    except ImportError:
                        st.error("❌ Install imbalanced-learn: `pip install imbalanced-learn`")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        with ml_tab3:
            st.markdown("**Correlation Filter** — remove highly correlated features")
            numeric_cols_corr = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if len(numeric_cols_corr) >= 2:
                corr_thresh = st.slider("Drop if |correlation| >", 0.5, 1.0, 0.9, 0.05,
                                        key="corr_thresh")
                corr_matrix = df[numeric_cols_corr].corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop_corr = [c for c in upper.columns if any(upper[c] > corr_thresh)]

                fig_corr = px.imshow(corr_matrix, text_auto=".2f",
                                     color_continuous_scale='RdBu_r',
                                     title="Correlation Heatmap",
                                     template="plotly_white",
                                     aspect="auto")
                fig_corr.update_layout(height=400, margin=dict(t=40,b=20))
                st.plotly_chart(fig_corr, use_container_width=True)

                if to_drop_corr:
                    st.warning(f"Highly correlated columns to drop: {to_drop_corr}")
                    if st.button("🗑️ Drop Correlated Columns", key="drop_corr_btn"):
                        df = df.drop(columns=to_drop_corr)
                        st.session_state.df = df
                        st.success(f"✅ Dropped {len(to_drop_corr)} columns")
                        st.rerun()
                else:
                    st.success("✅ No highly correlated pairs above threshold.")

        with ml_tab4:
            st.markdown("**Normality Tests** — check if features follow a normal distribution")
            numeric_cols_nt = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if numeric_cols_nt:
                nt_col = st.selectbox("Select column", numeric_cols_nt, key="nt_col")
                s_nt   = df[nt_col].dropna()

                nc1, nc2 = st.columns(2)
                with nc1:
                    fig_hist = px.histogram(df, x=nt_col, nbins=40,
                                            title=f"Distribution: {nt_col}",
                                            template="plotly_white",
                                            color_discrete_sequence=['#3498db'])
                    fig_hist.update_layout(height=260, margin=dict(t=40,b=20))
                    st.plotly_chart(fig_hist, use_container_width=True)
                with nc2:
                    fig_qq = go.Figure()
                    qq = stats.probplot(s_nt, dist="norm")
                    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1],
                                                mode='markers', name='Data',
                                                marker=dict(color='#3498db', size=4)))
                    x_line = np.array([qq[0][0].min(), qq[0][0].max()])
                    fig_qq.add_trace(go.Scatter(x=x_line,
                                                y=qq[1][1] + qq[1][0] * x_line,
                                                mode='lines', name='Normal',
                                                line=dict(color='red', dash='dash')))
                    fig_qq.update_layout(title="Q-Q Plot", template="plotly_white",
                                         height=260, margin=dict(t=40,b=20))
                    st.plotly_chart(fig_qq, use_container_width=True)

                try:
                    stat_sw, p_sw = shapiro(s_nt.sample(min(5000, len(s_nt)),
                                                         random_state=42))
                    stat_ks, p_ks = stats.kstest(s_nt, 'norm',
                                                   args=(s_nt.mean(), s_nt.std()))
                    t1, t2, t3, t4 = st.columns(4)
                    with t1: st.metric("Shapiro W",      f"{stat_sw:.4f}")
                    with t2: st.metric("Shapiro p-value", f"{p_sw:.4f}",
                                       delta="Normal" if p_sw > 0.05 else "Non-Normal",
                                       delta_color="normal" if p_sw > 0.05 else "inverse")
                    with t3: st.metric("KS Stat",        f"{stat_ks:.4f}")
                    with t4: st.metric("KS p-value",      f"{p_ks:.4f}",
                                       delta="Normal" if p_ks > 0.05 else "Non-Normal",
                                       delta_color="normal" if p_ks > 0.05 else "inverse")
                    sk_val = skew(s_nt)
                    ku_val = kurtosis(s_nt)
                    st.info(f"**Skewness:** {sk_val:.3f} "
                            f"({'Right-skewed' if sk_val > 0.5 else 'Left-skewed' if sk_val < -0.5 else 'Approx. symmetric'}) | "
                            f"**Kurtosis:** {ku_val:.3f} "
                            f"({'Leptokurtic (heavy tails)' if ku_val > 3 else 'Platykurtic (light tails)' if ku_val < 3 else 'Mesokurtic (normal)'})")
                except Exception as e:
                    st.warning(f"Could not run normality test: {str(e)}")

        with ml_tab5:
            st.markdown('<div class="section-header">✅ ML Readiness Check</div>',
                        unsafe_allow_html=True)

            checks = []
            missing_total = df.isnull().sum().sum()
            checks.append(("Missing Values",
                            "✅ None" if missing_total == 0 else f"❌ {missing_total:,} cells",
                            missing_total == 0))

            dup_total = df.duplicated().sum()
            checks.append(("Duplicate Rows",
                            "✅ None" if dup_total == 0 else f"⚠️ {dup_total:,} rows",
                            dup_total == 0))

            cat_remaining = df.select_dtypes(include='object').columns.tolist()
            checks.append(("Categorical Encoding",
                            "✅ All encoded" if not cat_remaining
                            else f"⚠️ {len(cat_remaining)} unencoded: {cat_remaining[:5]}",
                            not cat_remaining))

            numeric_cols_check = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if numeric_cols_check:
                ranges = df[numeric_cols_check].max() - df[numeric_cols_check].min()
                large_range = ranges[ranges > 1000].index.tolist()
                checks.append(("Feature Scaling",
                                "✅ Ranges OK" if not large_range
                                else f"⚠️ Large ranges in: {large_range[:5]}",
                                not large_range))
            else:
                checks.append(("Feature Scaling", "⚠️ No numeric columns", False))

            inf_count = np.isinf(df.select_dtypes(include=['int64','float64']).values).sum()
            checks.append(("Infinite Values",
                            "✅ None" if inf_count == 0 else f"❌ {inf_count} inf values",
                            inf_count == 0))

            score = sum(1 for _, _, ok in checks if ok)
            total = len(checks)

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f0f4f8,#fff);
                 padding:20px; border-radius:14px; margin-bottom:20px;
                 border-left:5px solid {'#2ecc71' if score == total else '#e67e22'};">
                <h3 style="margin:0; font-family:Sora,sans-serif; color:#1a2535;">
                    ML Readiness Score:
                    <span style="color:{'#2ecc71' if score==total else '#e67e22'};">
                        {score}/{total}
                    </span>
                </h3>
            </div>
            """, unsafe_allow_html=True)

            for name, msg, ok in checks:
                color = "#2ecc71" if ok else "#e74c3c" if "❌" in msg else "#e67e22"
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {color};
                     padding:12px 16px; margin-bottom:8px;">
                    <strong style="color:#1a2535;">{name}</strong>
                    <span style="float:right; color:{color};">{msg}</span>
                </div>
                """, unsafe_allow_html=True)

            if score == total:
                st.success("🎉 Your dataset is ML-ready! Head to the ML Algorithm page.")
            else:
                st.warning("⚠️ Address the flagged issues above before training models.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 11 – PREVIEW & EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown('<div class="section-header">📤 Preview & Export</div>',
                    unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)
        with e1: st.metric("Rows",       f"{len(df):,}")
        with e2: st.metric("Columns",    f"{len(df.columns):,}")
        with e3:
            mp = (df.isnull().sum().sum() / (len(df)*len(df.columns))*100) if len(df) > 0 else 0
            st.metric("Missing %", f"{mp:.2f}%")
        with e4: st.metric("Duplicates", f"{df.duplicated().sum()}")

        st.markdown("---")
        st.markdown("**Preview (first 50 rows)**")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("---")

        st.markdown("**Export Options**")
        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                data=csv_bytes,
                file_name=f"cleaned_{st.session_state.selected_file_name}.csv",
                mime="text/csv", use_container_width=True
            )

        with ex2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                "📥 Download Excel",
                data=excel_buf.getvalue(),
                file_name=f"cleaned_{st.session_state.selected_file_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with ex3:
            json_bytes = df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                "📥 Download JSON",
                data=json_bytes,
                file_name=f"cleaned_{st.session_state.selected_file_name}.json",
                mime="application/json", use_container_width=True
            )


# =========================================================
# === VISUALIZATION PAGE ==================================
# =========================================================

def visualization_page():
    """Advanced Visualization Expert System — DataMate AI"""
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -----------------------------
    
    import io
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    from scipy.stats import skew, kurtosis

    st.markdown('<div class="main-header">📈 Advanced Data Visualization</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the 'Data Upload' section first.")
        return

    df = st.session_state.df
    numeric_cols  = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols      = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols     = df.select_dtypes(include=['datetime64']).columns.tolist()
    all_cols      = df.columns.tolist()



    qs1, qs2, qs3, qs4, qs5 = st.columns(5)
    for col_ui, (label, val, color) in zip(
        [qs1, qs2, qs3, qs4, qs5],
        [
            ("Rows",       f"{len(df):,}",          "#3498db"),
            ("Columns",    f"{len(all_cols)}",       "#2ecc71"),
            ("Numeric",    f"{len(numeric_cols)}",   "#9b59b6"),
            ("Categorical",f"{len(cat_cols)}",       "#e67e22"),
            ("Date/Time",  f"{len(date_cols)}",      "#1abc9c"),
        ]
    ):
        with col_ui:
            st.markdown(f"""
            <div class="metric-card" style="border-left:4px solid {color}; text-align:center;">
                <p style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
                          letter-spacing:1px;margin:0;">{label}</p>
                <p style="font-size:22px;font-weight:800;color:{color};
                          margin:6px 0 0 0;font-family:Sora,sans-serif;">{val}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Helper: Add chart to dashboard ───────────────────────────────────────
    def add_to_dashboard(fig, title):
        if 'saved_charts' not in st.session_state:
            st.session_state.saved_charts = []
        st.session_state.saved_charts.append({'fig': fig, 'title': title, 'description': ''})
        st.success(f"✅ '{title}' pinned to Dashboard!")

    # ── Helper: Download button for HTML chart ────────────────────────────────
    def chart_download_btn(fig, filename, key):
        buf = io.StringIO()
        fig.write_html(buf, include_plotlyjs='cdn')
        st.download_button(
            "💾 Download HTML",
            data=buf.getvalue().encode(),
            file_name=f"{filename}.html",
            mime='text/html',
            key=key
        )

    # ── Helper: Render chart panel (plot + pin + download) ───────────────────
    def render_chart_panel(fig, section_label, chart_type_key):
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("📌 Pin to Dashboard", key=f"pin_{chart_type_key}"):
                add_to_dashboard(fig, f"{section_label}")
        with btn_col2:
            chart_download_btn(fig, chart_type_key, key=f"dl_{chart_type_key}")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # MAIN VISUALIZATION TABS
    # =========================================================================
    (
        tab_uni, tab_bi, tab_multi,
        tab_time, tab_stat, tab_geo,
        tab_custom, tab_group
    ) = st.tabs([
        "📊 Univariate",
        "📈 Bivariate",
        "💠 Multivariate",
        "🕐 Time Series",
        "🔬 Statistical",
        "🌍 Geographic",
        "🎨 Custom / Advanced",
        "📑 Grouping & Pivot"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — UNIVARIATE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_uni:
        st.markdown('<div class="section-header">📊 Single Variable Analysis</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Distribution, frequency, and shape of a single feature.")

        chart_type_uni = st.selectbox(
            "Chart Type",
            ["Histogram", "KDE / Density", "Box Plot", "Violin Plot",
             "Bar Chart (Count)", "Pie Chart", "Donut Chart", "ECDF Plot"],
            key="uni_chart_type"
        )

        settings_col, chart_col = st.columns([1, 2.5])
        fig = None

        with settings_col:
            st.markdown(f"**⚙️ Settings — {chart_type_uni}**")

            if chart_type_uni == "Histogram":
                if not numeric_cols:
                    st.warning("No numeric columns."); 
                else:
                    x_h   = st.selectbox("Feature", numeric_cols, key="uni_hist_x")
                    bins  = st.slider("Bins", 5, 150, 30, key="uni_hist_bins")
                    color = st.selectbox("Color By (Optional)", [None] + cat_cols, key="uni_hist_c")
                    norm  = st.selectbox("Normalization", ["count","percent","probability","density"], key="uni_hist_norm")
                    if x_h:
                        fig = px.histogram(df, x=x_h, color=color, nbins=bins,
                                           histnorm=None if norm == "count" else norm,
                                           title=f"Distribution of {x_h}",
                                           template="plotly_white",
                                           marginal="box",
                                           color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(bargap=0.05)
                        st.caption("📌 **When to use:** Check skewness, normality, frequency spread.")

            elif chart_type_uni == "KDE / Density":
                if not numeric_cols:
                    st.warning("No numeric columns.")
                else:
                    x_kde  = st.selectbox("Feature", numeric_cols, key="uni_kde_x")
                    color  = st.selectbox("Color By (Optional)", [None] + cat_cols, key="uni_kde_c")
                    if x_kde:
                        fig = px.histogram(df, x=x_kde, color=color,
                                           histnorm="density", nbins=50,
                                           marginal="violin",
                                           title=f"Density of {x_kde}",
                                           template="plotly_white",
                                           color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_traces(opacity=0.75)
                        st.caption("📌 **When to use:** Smooth distribution shape without bin sensitivity.")

            elif chart_type_uni == "Box Plot":
                if not numeric_cols:
                    st.warning("No numeric columns.")
                else:
                    y_box  = st.selectbox("Feature", numeric_cols, key="uni_box_y")
                    color  = st.selectbox("Group By (Optional)", [None] + cat_cols, key="uni_box_c")
                    points = st.selectbox("Show Points", ["outliers", "all", False], key="uni_box_pts")
                    if y_box:
                        fig = px.box(df, y=y_box, color=color, points=points,
                                     title=f"Box Plot — {y_box}",
                                     template="plotly_white",
                                     color_discrete_sequence=px.colors.qualitative.Set1)
                        st.caption("📌 **When to use:** Outlier detection, IQR, median comparison.")

            elif chart_type_uni == "Violin Plot":
                if not numeric_cols:
                    st.warning("No numeric columns.")
                else:
                    y_vio  = st.selectbox("Feature", numeric_cols, key="uni_vio_y")
                    color  = st.selectbox("Group By (Optional)", [None] + cat_cols, key="uni_vio_c")
                    if y_vio:
                        fig = px.violin(df, y=y_vio, color=color,
                                        box=True, points="outliers",
                                        title=f"Violin Plot — {y_vio}",
                                        template="plotly_white",
                                        color_discrete_sequence=px.colors.qualitative.Antique)
                        st.caption("📌 **When to use:** Distribution density + box plot combined.")

            elif chart_type_uni == "Bar Chart (Count)":
                if not cat_cols:
                    st.warning("No categorical columns.")
                else:
                    x_bar  = st.selectbox("Category", cat_cols, key="uni_bar_x")
                    top_n  = st.slider("Top N values", 5, 50, 15, key="uni_bar_n")
                    orient = st.radio("Orientation", ["Vertical", "Horizontal"], key="uni_bar_orient")
                    if x_bar:
                        vc = df[x_bar].value_counts().head(top_n).reset_index()
                        vc.columns = [x_bar, 'Count']
                        if orient == "Vertical":
                            fig = px.bar(vc, x=x_bar, y='Count', text='Count',
                                         title=f"Top {top_n} — {x_bar}",
                                         template="plotly_white",
                                         color='Count', color_continuous_scale='Blues')
                        else:
                            fig = px.bar(vc, x='Count', y=x_bar, text='Count',
                                         orientation='h',
                                         title=f"Top {top_n} — {x_bar}",
                                         template="plotly_white",
                                         color='Count', color_continuous_scale='Blues')
                        fig.update_traces(textposition='outside')
                        st.caption("📌 **When to use:** Frequency of categorical values.")

            elif chart_type_uni in ["Pie Chart", "Donut Chart"]:
                if not cat_cols:
                    st.warning("No categorical columns.")
                else:
                    cat_p  = st.selectbox("Category", cat_cols, key="uni_pie_c")
                    top_n  = st.slider("Top N slices", 3, 20, 8, key="uni_pie_n")
                    hole   = 0.4 if chart_type_uni == "Donut Chart" else 0
                    if cat_p:
                        vc = df[cat_p].value_counts().head(top_n).reset_index()
                        vc.columns = [cat_p, 'Count']
                        fig = px.pie(vc, names=cat_p, values='Count', hole=hole,
                                     title=f"Proportion of {cat_p} (Top {top_n})",
                                     template="plotly_white",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.caption("📌 **When to use:** Part-to-whole breakdown.")

            elif chart_type_uni == "ECDF Plot":
                if not numeric_cols:
                    st.warning("No numeric columns.")
                else:
                    x_ecdf = st.selectbox("Feature", numeric_cols, key="uni_ecdf_x")
                    color  = st.selectbox("Group By (Optional)", [None] + cat_cols, key="uni_ecdf_c")
                    if x_ecdf:
                        fig = px.ecdf(df, x=x_ecdf, color=color,
                                      title=f"ECDF — {x_ecdf}",
                                      template="plotly_white",
                                      color_discrete_sequence=px.colors.qualitative.Bold)
                        st.caption("📌 **When to use:** Cumulative distribution; no bin choice needed.")

        with chart_col:
            if fig:
                render_chart_panel(fig, f"{chart_type_uni}", f"uni_{chart_type_uni.replace(' ','_').lower()}")
            else:
                st.markdown("""
                <div style="height:380px; display:flex; align-items:center; justify-content:center;
                     border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                    ← Configure settings on the left to generate a chart
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — BIVARIATE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_bi:
        st.markdown('<div class="section-header">📈 Two Variable Analysis</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Relationships, correlations, and comparisons between two features.")

        chart_type_bi = st.selectbox(
            "Chart Type",
            ["Scatter Plot", "Line Chart", "Bar Chart (Aggregated)",
             "Grouped Bar Chart", "Area Chart", "Violin Plot (Grouped)",
             "Box Plot (Grouped)", "Strip Plot", "Hexbin / 2D Density"],
            key="bi_chart_type"
        )

        settings_col, chart_col = st.columns([1, 2.5])
        fig = None

        with settings_col:
            st.markdown(f"**⚙️ Settings — {chart_type_bi}**")

            if chart_type_bi == "Scatter Plot":
                if not numeric_cols:
                    st.warning("No numeric columns.")
                else:
                    x_s    = st.selectbox("X-Axis", numeric_cols, key="bi_scat_x")
                    y_s    = st.selectbox("Y-Axis", numeric_cols, key="bi_scat_y",
                                          index=min(1, len(numeric_cols)-1))
                    color  = st.selectbox("Color By", [None] + all_cols, key="bi_scat_c")
                    size   = st.selectbox("Size By (Numeric)", [None] + numeric_cols, key="bi_scat_sz")
                    trend  = st.checkbox("Add Trend Line (OLS)", key="bi_scat_trend")
                    if x_s and y_s:
                        fig = px.scatter(df, x=x_s, y=y_s, color=color, size=size,
                                         trendline="ols" if trend else None,
                                         title=f"{y_s} vs {x_s}",
                                         template="plotly_white",
                                         opacity=0.75,
                                         color_discrete_sequence=px.colors.qualitative.Set2)
                        r, p = stats.pearsonr(df[x_s].dropna(), df[y_s].dropna()) if len(df.dropna(subset=[x_s, y_s])) > 2 else (0, 1)
                        st.caption(f"📌 Pearson r = **{r:.3f}** | p-value = {p:.4f}")

            elif chart_type_bi == "Line Chart":
                x_opts = date_cols + numeric_cols
                if not x_opts or not numeric_cols:
                    st.warning("Need at least one date/numeric column.")
                else:
                    x_l    = st.selectbox("X-Axis", x_opts, key="bi_line_x")
                    y_l    = st.selectbox("Y-Axis", numeric_cols, key="bi_line_y")
                    color  = st.selectbox("Group By", [None] + cat_cols, key="bi_line_c")
                    smooth = st.checkbox("Smooth Line", key="bi_line_sm")
                    if x_l and y_l:
                        dfs = df.sort_values(by=x_l)
                        if smooth:
                            fig = px.scatter(dfs, x=x_l, y=y_l, color=color,
                                             trendline="lowess",
                                             title=f"Smoothed Trend: {y_l} over {x_l}",
                                             template="plotly_white")
                        else:
                            fig = px.line(dfs, x=x_l, y=y_l, color=color,
                                          title=f"Trend: {y_l} over {x_l}",
                                          template="plotly_white",
                                          markers=True)
                        st.caption("📌 **When to use:** Trend over a continuous or time axis.")

            elif chart_type_bi == "Bar Chart (Aggregated)":
                if not cat_cols or not numeric_cols:
                    st.warning("Need at least one categorical and one numeric column.")
                else:
                    x_ba   = st.selectbox("Category (X)", cat_cols, key="bi_bar_x")
                    y_ba   = st.selectbox("Numeric (Y)", numeric_cols, key="bi_bar_y")
                    agg    = st.selectbox("Aggregation", ["mean","sum","count","max","min","median"], key="bi_bar_agg")
                    sort   = st.checkbox("Sort by value", True, key="bi_bar_sort")
                    top_n  = st.slider("Top N", 5, 50, 15, key="bi_bar_n")
                    if x_ba and y_ba:
                        grp = df.groupby(x_ba)[y_ba].agg(agg).reset_index()
                        grp.columns = [x_ba, y_ba]
                        if sort:
                            grp = grp.sort_values(y_ba, ascending=False)
                        grp = grp.head(top_n)
                        fig = px.bar(grp, x=x_ba, y=y_ba, text=y_ba,
                                     title=f"{agg.title()} of {y_ba} by {x_ba}",
                                     template="plotly_white",
                                     color=y_ba, color_continuous_scale='Blues')
                        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                        st.caption("📌 **When to use:** Compare metric across categories.")

            elif chart_type_bi == "Grouped Bar Chart":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    x_gb   = st.selectbox("Group Category (X)", cat_cols, key="bi_grp_x")
                    color  = st.selectbox("Sub-Group (Color)", cat_cols, key="bi_grp_c",
                                          index=min(1, len(cat_cols)-1))
                    y_gb   = st.selectbox("Numeric (Y)", numeric_cols, key="bi_grp_y")
                    agg_gb = st.selectbox("Aggregation", ["mean","sum","count"], key="bi_grp_agg")
                    if x_gb and color and y_gb:
                        grp = df.groupby([x_gb, color])[y_gb].agg(agg_gb).reset_index()
                        fig = px.bar(grp, x=x_gb, y=y_gb, color=color,
                                     barmode='group',
                                     title=f"{agg_gb.title()} of {y_gb} by {x_gb} & {color}",
                                     template="plotly_white",
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                        st.caption("📌 **When to use:** Compare sub-groups within categories.")

            elif chart_type_bi == "Area Chart":
                x_opts = date_cols + numeric_cols
                if not x_opts or not numeric_cols:
                    st.warning("Need at least one date/numeric column.")
                else:
                    x_ar   = st.selectbox("X-Axis", x_opts, key="bi_area_x")
                    y_ar   = st.selectbox("Y-Axis", numeric_cols, key="bi_area_y")
                    color  = st.selectbox("Group By", [None] + cat_cols, key="bi_area_c")
                    stack  = st.checkbox("Stacked Area", key="bi_area_stack")
                    if x_ar and y_ar:
                        dfs = df.sort_values(by=x_ar)
                        fig = px.area(dfs, x=x_ar, y=y_ar, color=color,
                                      groupnorm='fraction' if stack else None,
                                      title=f"Area: {y_ar} over {x_ar}",
                                      template="plotly_white",
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.caption("📌 **When to use:** Volume trend and cumulative magnitude.")

            elif chart_type_bi == "Violin Plot (Grouped)":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    x_vg   = st.selectbox("Category (X)", cat_cols, key="bi_vio_x")
                    y_vg   = st.selectbox("Numeric (Y)", numeric_cols, key="bi_vio_y")
                    color  = st.selectbox("Split By", [None] + cat_cols, key="bi_vio_c")
                    if x_vg and y_vg:
                        fig = px.violin(df, x=x_vg, y=y_vg, color=color,
                                        box=True, points="outliers",
                                        title=f"Distribution of {y_vg} by {x_vg}",
                                        template="plotly_white",
                                        color_discrete_sequence=px.colors.qualitative.Antique)
                        st.caption("📌 **When to use:** Rich distribution comparison across groups.")

            elif chart_type_bi == "Box Plot (Grouped)":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    x_bg   = st.selectbox("Category (X)", cat_cols, key="bi_box_x")
                    y_bg   = st.selectbox("Numeric (Y)", numeric_cols, key="bi_box_y")
                    color  = st.selectbox("Color By", [None] + cat_cols, key="bi_box_c")
                    if x_bg and y_bg:
                        fig = px.box(df, x=x_bg, y=y_bg, color=color,
                                     notched=True, points="outliers",
                                     title=f"Box: {y_bg} by {x_bg}",
                                     template="plotly_white",
                                     color_discrete_sequence=px.colors.qualitative.Set1)
                        st.caption("📌 **When to use:** Median and IQR per category, outlier comparison.")

            elif chart_type_bi == "Strip Plot":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    x_sp   = st.selectbox("Category (X)", cat_cols, key="bi_strip_x")
                    y_sp   = st.selectbox("Numeric (Y)", numeric_cols, key="bi_strip_y")
                    color  = st.selectbox("Color By", [None] + cat_cols, key="bi_strip_c")
                    jitter = st.slider("Jitter", 0.0, 0.5, 0.2, key="bi_strip_jitter")
                    if x_sp and y_sp:
                        fig = px.strip(df, x=x_sp, y=y_sp, color=color,
                                       stripmode="overlay",
                                       title=f"Strip Plot: {y_sp} by {x_sp}",
                                       template="plotly_white",
                                       color_discrete_sequence=px.colors.qualitative.Bold)
                        st.caption("📌 **When to use:** Show all individual points per group.")

            elif chart_type_bi == "Hexbin / 2D Density":
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns.")
                else:
                    x_hex  = st.selectbox("X-Axis", numeric_cols, key="bi_hex_x")
                    y_hex  = st.selectbox("Y-Axis", numeric_cols, key="bi_hex_y",
                                          index=min(1, len(numeric_cols)-1))
                    if x_hex and y_hex:
                        fig = px.density_heatmap(df, x=x_hex, y=y_hex,
                                                 nbinsx=30, nbinsy=30,
                                                 marginal_x="histogram",
                                                 marginal_y="histogram",
                                                 color_continuous_scale="Blues",
                                                 title=f"2D Density: {y_hex} vs {x_hex}",
                                                 template="plotly_white")
                        st.caption("📌 **When to use:** Scatter with many overlapping points.")

        with chart_col:
            if fig:
                render_chart_panel(fig, f"{chart_type_bi}", f"bi_{chart_type_bi.replace(' ','_').lower()}")
            else:
                st.markdown("""
                <div style="height:380px; display:flex; align-items:center; justify-content:center;
                     border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                    ← Configure settings on the left to generate a chart
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — MULTIVARIATE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_multi:
        st.markdown('<div class="section-header">💠 Multivariate & Complex Analysis</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Interactions among 3+ variables and hierarchical structure.")

        chart_type_multi = st.selectbox(
            "Chart Type",
            ["Correlation Heatmap", "Pair Plot (Scatter Matrix)",
             "Bubble Chart", "Treemap", "Sunburst Chart",
             "Parallel Coordinates", "Parallel Categories",
             "3D Scatter Plot", "Radar / Spider Chart"],
            key="multi_chart_type"
        )

        settings_col, chart_col = st.columns([1, 2.5])
        fig = None

        with settings_col:
            st.markdown(f"**⚙️ Settings — {chart_type_multi}**")

            if chart_type_multi == "Correlation Heatmap":
                if len(numeric_cols) < 2:
                    st.warning("Need ≥ 2 numeric columns.")
                else:
                    sel_cols = st.multiselect("Select Columns", numeric_cols,
                                              default=numeric_cols[:min(10, len(numeric_cols))],
                                              key="multi_corr_cols")
                    method   = st.selectbox("Method", ["pearson","spearman","kendall"], key="multi_corr_m")
                    if sel_cols and len(sel_cols) >= 2:
                        corr = df[sel_cols].corr(method=method)
                        fig  = px.imshow(corr, text_auto=".2f", aspect="auto",
                                         color_continuous_scale='RdBu_r',
                                         title=f"{method.title()} Correlation Matrix",
                                         template="plotly_white")
                        fig.update_layout(height=500)
                        st.caption("📌 **When to use:** Feature selection, multicollinearity detection.")

            elif chart_type_multi == "Pair Plot (Scatter Matrix)":
                if len(numeric_cols) < 2:
                    st.warning("Need ≥ 2 numeric columns.")
                else:
                    sel_pp = st.multiselect("Numeric Features", numeric_cols,
                                            default=numeric_cols[:min(4, len(numeric_cols))],
                                            key="multi_pp_cols")
                    color  = st.selectbox("Color By (Optional)", [None] + cat_cols, key="multi_pp_c")
                    if sel_pp and len(sel_pp) >= 2:
                        fig = px.scatter_matrix(df, dimensions=sel_pp, color=color,
                                                title="Pair Plot — Scatter Matrix",
                                                template="plotly_white",
                                                color_discrete_sequence=px.colors.qualitative.Set2,
                                                opacity=0.6)
                        fig.update_traces(diagonal_visible=True, showupperhalf=True)
                        fig.update_layout(height=600)
                        st.caption("📌 **When to use:** Quickly see all pairwise relationships.")

            elif chart_type_multi == "Bubble Chart":
                if len(numeric_cols) < 2:
                    st.warning("Need ≥ 2 numeric columns.")
                else:
                    x_b    = st.selectbox("X-Axis", numeric_cols, key="multi_bub_x")
                    y_b    = st.selectbox("Y-Axis", numeric_cols, key="multi_bub_y",
                                          index=min(1, len(numeric_cols)-1))
                    sz_b   = st.selectbox("Size (Numeric)", numeric_cols, key="multi_bub_s",
                                          index=min(2, len(numeric_cols)-1))
                    color  = st.selectbox("Color By", [None] + all_cols, key="multi_bub_c")
                    text   = st.selectbox("Label (Optional)", [None] + cat_cols, key="multi_bub_t")
                    if x_b and y_b and sz_b:
                        fig = px.scatter(df, x=x_b, y=y_b, size=sz_b, color=color,
                                         text=text, hover_data=all_cols[:5],
                                         size_max=60,
                                         title="Bubble Chart (3 Variables)",
                                         template="plotly_white",
                                         color_discrete_sequence=px.colors.qualitative.Vivid)
                        st.caption("📌 **When to use:** X/Y relationship with magnitude context.")

            elif chart_type_multi == "Treemap":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    path   = st.multiselect("Hierarchy (Category)", cat_cols, key="multi_tree_p",
                                            default=cat_cols[:min(2, len(cat_cols))])
                    val    = st.selectbox("Size (Numeric)", numeric_cols, key="multi_tree_v")
                    color  = st.selectbox("Color (Numeric)", [None] + numeric_cols, key="multi_tree_c")
                    if path and val:
                        fig = px.treemap(df, path=path, values=val, color=color,
                                         color_continuous_scale="Blues",
                                         title="Hierarchical Treemap",
                                         template="plotly_white")
                        fig.update_layout(height=500)
                        st.caption("📌 **When to use:** Nested category breakdown with size.")

            elif chart_type_multi == "Sunburst Chart":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    path   = st.multiselect("Hierarchy (Category)", cat_cols, key="multi_sun_p",
                                            default=cat_cols[:min(2, len(cat_cols))])
                    val    = st.selectbox("Size (Numeric)", numeric_cols, key="multi_sun_v")
                    if path and val:
                        fig = px.sunburst(df, path=path, values=val,
                                          title="Multi-Level Sunburst",
                                          template="plotly_white",
                                          color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_layout(height=500)
                        st.caption("📌 **When to use:** Radial drill-down into category hierarchy.")

            elif chart_type_multi == "Parallel Coordinates":
                if len(numeric_cols) < 2:
                    st.warning("Need ≥ 2 numeric columns.")
                else:
                    sel_pc = st.multiselect("Numeric Dimensions", numeric_cols,
                                            default=numeric_cols[:min(5, len(numeric_cols))],
                                            key="multi_pc_dims")
                    color  = st.selectbox("Color By (Numeric)", numeric_cols, key="multi_pc_c")
                    if sel_pc and color:
                        fig = px.parallel_coordinates(df, dimensions=sel_pc, color=color,
                                                      color_continuous_scale='Viridis',
                                                      title="Parallel Coordinates Plot",
                                                      template="plotly_white")
                        fig.update_layout(height=480)
                        st.caption("📌 **When to use:** Explore patterns across many numeric dimensions.")

            elif chart_type_multi == "Parallel Categories":
                if not cat_cols:
                    st.warning("Need categorical columns.")
                else:
                    sel_pcat = st.multiselect("Category Dimensions", cat_cols,
                                              default=cat_cols[:min(3, len(cat_cols))],
                                              key="multi_pcat_dims")
                    color    = st.selectbox("Color By (Numeric)", [None] + numeric_cols, key="multi_pcat_c")
                    if sel_pcat:
                        fig = px.parallel_categories(df, dimensions=sel_pcat,
                                                     color=color if color else None,
                                                     color_continuous_scale="Blues",
                                                     title="Parallel Categories Flow",
                                                     template="plotly_white")
                        fig.update_layout(height=480)
                        st.caption("📌 **When to use:** Visualize flow between categorical variables.")

            elif chart_type_multi == "3D Scatter Plot":
                if len(numeric_cols) < 3:
                    st.warning("Need ≥ 3 numeric columns.")
                else:
                    x_3d   = st.selectbox("X-Axis", numeric_cols, key="multi_3d_x")
                    y_3d   = st.selectbox("Y-Axis", numeric_cols, key="multi_3d_y",
                                          index=min(1, len(numeric_cols)-1))
                    z_3d   = st.selectbox("Z-Axis", numeric_cols, key="multi_3d_z",
                                          index=min(2, len(numeric_cols)-1))
                    color  = st.selectbox("Color By", [None] + all_cols, key="multi_3d_c")
                    size   = st.selectbox("Size By (Numeric)", [None] + numeric_cols, key="multi_3d_s")
                    if x_3d and y_3d and z_3d:
                        fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                                            color=color, size=size, opacity=0.75,
                                            title=f"3D Scatter: {z_3d} vs {y_3d} vs {x_3d}",
                                            template="plotly_white",
                                            color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(height=550)
                        st.caption("📌 **When to use:** Three-dimensional relationship exploration.")

            elif chart_type_multi == "Radar / Spider Chart":
                if len(numeric_cols) < 3:
                    st.warning("Need ≥ 3 numeric columns.")
                else:
                    radar_cols = st.multiselect("Numeric Features (axes)", numeric_cols,
                                                default=numeric_cols[:min(6, len(numeric_cols))],
                                                key="multi_radar_cols")
                    group_col  = st.selectbox("Group By (Category)", [None] + cat_cols, key="multi_radar_g")
                    max_groups = st.slider("Max Groups to Show", 1, 10, 3, key="multi_radar_maxg")

                    if radar_cols and len(radar_cols) >= 3:
                        fig = go.Figure()
                        if group_col:
                            groups = df[group_col].dropna().unique()[:max_groups]
                            colors = px.colors.qualitative.Set2
                            for i, grp in enumerate(groups):
                                subset = df[df[group_col] == grp][radar_cols].mean()
                                vals   = subset.tolist() + [subset.iloc[0]]
                                cats   = radar_cols + [radar_cols[0]]
                                fig.add_trace(go.Scatterpolar(
                                    r=vals, theta=cats, fill='toself',
                                    name=str(grp),
                                    line_color=colors[i % len(colors)]
                                ))
                        else:
                            subset = df[radar_cols].mean()
                            vals   = subset.tolist() + [subset.iloc[0]]
                            cats   = radar_cols + [radar_cols[0]]
                            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name='Mean'))

                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True)),
                            title="Radar / Spider Chart — Feature Averages",
                            template="plotly_white",
                            height=480
                        )
                        st.caption("📌 **When to use:** Strength/weakness profile per group.")

        with chart_col:
            if fig:
                render_chart_panel(fig, f"{chart_type_multi}",
                                   f"multi_{chart_type_multi.replace(' ','_').lower()}")
            else:
                st.markdown("""
                <div style="height:380px; display:flex; align-items:center; justify-content:center;
                     border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                    ← Configure settings on the left to generate a chart
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — TIME SERIES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_time:
        st.markdown('<div class="section-header">🕐 Time Series Analysis</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Trends, seasonality, and patterns over time.")

        x_opts_ts = date_cols + numeric_cols
        if not x_opts_ts or not numeric_cols:
            st.warning("⚠️ Need at least one date or numeric column for the time axis.")
        else:
            chart_type_ts = st.selectbox(
                "Chart Type",
                ["Line Chart with Range Selector", "Multi-Series Line",
                 "Candlestick Chart", "Rolling Average Overlay",
                 "Seasonal Decomposition Preview"],
                key="ts_chart_type"
            )

            settings_col, chart_col = st.columns([1, 2.5])
            fig = None

            with settings_col:
                st.markdown(f"**⚙️ Settings — {chart_type_ts}**")

                if chart_type_ts == "Line Chart with Range Selector":
                    x_ts  = st.selectbox("Time / X-Axis", x_opts_ts, key="ts_line_x")
                    y_ts  = st.selectbox("Metric (Y)", numeric_cols, key="ts_line_y")
                    color = st.selectbox("Group By", [None] + cat_cols, key="ts_line_c")
                    
                    # Advanced Parameters
                    with st.expander("🛠️ Advanced Settings", expanded=False):
                        line_shape = st.selectbox("Line Shape", ["linear", "spline", "vh", "hv", "hvh", "vhv"], key="ts_line_shape")
                        show_markers = st.checkbox("Show Markers", value=False, key="ts_line_markers")
                        fill_area = st.checkbox("Fill Area Under Line", value=False, key="ts_line_fill")

                    if x_ts and y_ts:
                        dfs = df.sort_values(by=x_ts)
                        fig = px.line(dfs, x=x_ts, y=y_ts, color=color,
                                      title=f"{y_ts} over Time",
                                      template="plotly_white", markers=show_markers,
                                      line_shape=line_shape,
                                      color_discrete_sequence=px.colors.qualitative.Set1)
                        if fill_area:
                            fig.update_traces(fill='tozeroy')
                            
                        fig.update_xaxes(
                            rangeslider_visible=True,
                            rangeselector=dict(buttons=[
                                dict(count=7,  label="1W", step="day",  stepmode="backward"),
                                dict(count=1,  label="1M", step="month",stepmode="backward"),
                                dict(count=6,  label="6M", step="month",stepmode="backward"),
                                dict(count=1,  label="1Y", step="year", stepmode="backward"),
                                dict(step="all", label="All")
                            ])
                        )
                        fig.update_layout(height=460, hovermode="x unified")
                        st.caption("📌 **When to use:** Interactive time exploration with zoom.")

                elif chart_type_ts == "Multi-Series Line":
                    x_ts  = st.selectbox("Time / X-Axis", x_opts_ts, key="ts_ms_x")
                    y_cols = st.multiselect("Metrics (Y)", numeric_cols, key="ts_ms_y",
                                            default=numeric_cols[:min(3, len(numeric_cols))])
                    
                    with st.expander("🛠️ Advanced Settings"):
                        line_width = st.slider("Line Width", 1, 5, 2, key="ts_ms_lw")
                        opacity = st.slider("Line Opacity", 0.1, 1.0, 0.8, key="ts_ms_op")

                    if x_ts and y_cols:
                        dfs = df.sort_values(by=x_ts)
                        fig = go.Figure()
                        colors = px.colors.qualitative.Set2
                        for i, yc in enumerate(y_cols):
                            fig.add_trace(go.Scatter(x=dfs[x_ts], y=dfs[yc],
                                                     mode='lines+markers' if len(dfs) < 100 else 'lines', 
                                                     name=yc,
                                                     opacity=opacity,
                                                     line=dict(color=colors[i % len(colors)], width=line_width)))
                        fig.update_layout(title="Multi-Series Line Chart",
                                          template="plotly_white", height=440,
                                          hovermode="x unified",
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.caption("📌 **When to use:** Compare multiple metrics on one axis.")

                elif chart_type_ts == "Candlestick Chart":
                    st.caption("Requires OHLC columns (Open, High, Low, Close)")
                    x_c   = st.selectbox("Date", x_opts_ts, key="ts_can_x")
                    o_c   = st.selectbox("Open",  numeric_cols, key="ts_can_o")
                    h_c   = st.selectbox("High",  numeric_cols, key="ts_can_h", index=min(1, len(numeric_cols)-1))
                    l_c   = st.selectbox("Low",   numeric_cols, key="ts_can_l", index=min(2, len(numeric_cols)-1))
                    cl_c  = st.selectbox("Close", numeric_cols, key="ts_can_c", index=min(3, len(numeric_cols)-1))
                    
                    if x_c and o_c and h_c and l_c and cl_c:
                        dfs = df.sort_values(by=x_c)
                        fig = go.Figure(data=[go.Candlestick(
                            x=dfs[x_c], open=dfs[o_c], high=dfs[h_c],
                            low=dfs[l_c], close=dfs[cl_c],
                            increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
                        )])
                        fig.update_layout(
                            title="Financial Candlestick Chart",
                            template="plotly_dark", height=460, # Dark theme looks better for candlesticks
                            xaxis_rangeslider_visible=True
                        )
                        st.caption("📌 **When to use:** OHLC stock/crypto market data.")

                elif chart_type_ts == "Rolling Average Overlay":
                    x_ts  = st.selectbox("Time / X-Axis", x_opts_ts, key="ts_roll_x")
                    y_ts  = st.selectbox("Metric (Y)", numeric_cols, key="ts_roll_y")
                    window = st.slider("Rolling Window", 2, 60, 7, key="ts_roll_w")
                    
                    with st.expander("🛠️ Add Bands"):
                        show_bands = st.checkbox("Show Bollinger Bands (±2 Std Dev)", value=False)

                    if x_ts and y_ts:
                        dfs = df.sort_values(by=x_ts).copy()
                        dfs['rolling_avg'] = dfs[y_ts].rolling(window=window).mean()
                        dfs['rolling_std'] = dfs[y_ts].rolling(window=window).std()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfs[x_ts], y=dfs[y_ts],
                                                 mode='lines', name='Raw Data',
                                                 line=dict(color='#AED6F1', width=1), opacity=0.6))
                        fig.add_trace(go.Scatter(x=dfs[x_ts], y=dfs['rolling_avg'],
                                                 mode='lines', name=f'MA ({window})',
                                                 line=dict(color='#2980B9', width=2.5)))
                        
                        if show_bands:
                            fig.add_trace(go.Scatter(x=dfs[x_ts], y=dfs['rolling_avg'] + (2 * dfs['rolling_std']),
                                                     mode='lines', name='Upper Band', line=dict(color='rgba(231, 76, 60, 0.5)', width=1, dash='dash')))
                            fig.add_trace(go.Scatter(x=dfs[x_ts], y=dfs['rolling_avg'] - (2 * dfs['rolling_std']),
                                                     mode='lines', name='Lower Band', fill='tonexty', fillcolor='rgba(231, 76, 60, 0.1)',
                                                     line=dict(color='rgba(231, 76, 60, 0.5)', width=1, dash='dash')))

                        fig.update_layout(title=f"{y_ts} with {window}-Period Rolling Average",
                                          template="plotly_white", height=440, hovermode="x unified")
                        st.caption("📌 **When to use:** Smooth noisy time data and spot trend.")

                elif chart_type_ts == "Seasonal Decomposition Preview":
                    x_ts  = st.selectbox("Time / X-Axis", x_opts_ts, key="ts_seas_x")
                    y_ts  = st.selectbox("Metric (Y)", numeric_cols, key="ts_seas_y")
                    period = st.slider("Season Period", 2, 52, 12, key="ts_seas_p")
                    
                    if x_ts and y_ts:
                        try:
                            dfs = df[[x_ts, y_ts]].sort_values(x_ts).dropna()
                            series = dfs[y_ts].values
                            n = len(series)
                            if n < period * 2:
                                st.warning("Not enough data for the chosen period.")
                            else:
                                trend   = np.convolve(series, np.ones(period)/period, mode='same')
                                detrend = series - trend
                                seasonal = np.array([np.mean(detrend[i::period]) for i in range(period)] * (n // period + 1))[:n]
                                residual = series - trend - seasonal

                                fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                                    subplot_titles=["Original Data","Trend Component","Seasonal Component","Residuals"],
                                                    vertical_spacing=0.07)
                                x_axis = dfs[x_ts]
                                for row, (y_data, color_val) in enumerate([
                                    (series,   '#3498db'),
                                    (trend,    '#2ecc71'),
                                    (seasonal, '#e67e22'),
                                    (residual, '#e74c3c')
                                ], 1):
                                    fig.add_trace(go.Scatter(x=x_axis, y=y_data,
                                                             mode='lines' if row < 4 else 'markers',
                                                             line=dict(color=color_val, width=1.5) if row < 4 else None,
                                                             marker=dict(color=color_val, size=4, opacity=0.7) if row == 4 else None,
                                                             showlegend=False), row=row, col=1)
                                fig.update_layout(title="Seasonal Decomposition (Manual Approximation)",
                                                  template="plotly_white", height=650)
                                st.caption("📌 **When to use:** Reveal trend, seasonal, and noise components.")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

            with chart_col:
                if fig:
                    render_chart_panel(fig, f"{chart_type_ts}",
                                       f"ts_{chart_type_ts.replace(' ','_').lower()}")
                else:
                    st.markdown("""
                    <div style="height:380px; display:flex; align-items:center; justify-content:center;
                         border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                        ← Configure settings on the left to generate a chart
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — STATISTICAL
    # ══════════════════════════════════════════════════════════════════════════
    with tab_stat:
        st.markdown('<div class="section-header">🔬 Statistical Visualization</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Distribution analysis, statistical tests, and confidence intervals.")

        if not numeric_cols:
            st.warning("⚠️ No numeric columns available for statistical charts.")
        else:
            chart_type_stat = st.selectbox(
                "Chart Type",
                ["Q-Q Plot", "Distribution Comparison",
                 "Error Bar Chart", "Histogram + KDE Overlay",
                 "Cumulative Distribution", "Feature Importance Bar"],
                key="stat_chart_type"
            )

            settings_col, chart_col = st.columns([1, 2.5])
            fig = None

            with settings_col:
                st.markdown(f"**⚙️ Settings — {chart_type_stat}**")

                if chart_type_stat == "Q-Q Plot":
                    col_qq  = st.selectbox("Feature", numeric_cols, key="stat_qq_col")
                    dist_qq = st.selectbox("Reference Distribution",
                                           ["norm","expon","logistic","uniform"], key="stat_qq_dist")
                    
                    if col_qq:
                        s = df[col_qq].dropna()
                        qq = stats.probplot(s, dist=dist_qq)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1],
                                                 mode='markers', name='Data Points',
                                                 marker=dict(color='#3498db', size=6, opacity=0.6, line=dict(width=1, color='DarkSlateGrey'))))
                        x_line = np.array([qq[0][0].min(), qq[0][0].max()])
                        fig.add_trace(go.Scatter(x=x_line,
                                                 y=qq[1][1] + qq[1][0] * x_line,
                                                 mode='lines', name='Reference Line',
                                                 line=dict(color='#e74c3c', dash='dash', width=2.5)))
                        fig.update_layout(
                            title=f"Q-Q Plot: {col_qq} vs Theoretical {dist_qq.title()}",
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles",
                            template="plotly_white", height=420
                        )
                        # Stats panel
                        stat_sw, p_sw = stats.shapiro(s.sample(min(5000, len(s)), random_state=42))
                        st.markdown(f"""
                        <div class="metric-card" style="padding:10px 14px;">
                            <span style="font-weight:700;">Shapiro-Wilk p-value:</span>
                            <span style="color:{'#2ecc71' if p_sw > 0.05 else '#e74c3c'};float:right;font-weight:700;">
                                {p_sw:.4f} {'(Normal ✅)' if p_sw > 0.05 else '(Non-Normal ❌)'}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("📌 **When to use:** Test whether data follows a theoretical distribution.")

                elif chart_type_stat == "Distribution Comparison":
                    sel_dist = st.multiselect("Numeric Columns to Compare", numeric_cols,
                                              default=numeric_cols[:min(3, len(numeric_cols))],
                                              key="stat_dist_cols")
                    norm_each = st.checkbox("Normalize distributions (Density)", value=True, key="stat_dist_norm")
                    
                    with st.expander("🛠️ Advanced Settings"):
                        barmode = st.selectbox("Bar Mode", ["overlay", "group", "stack"], key="stat_dist_bar")
                        bins_c = st.slider("Number of Bins", 10, 100, 40, key="stat_dist_bins")

                    if sel_dist:
                        fig = go.Figure()
                        colors = px.colors.qualitative.Set1
                        for i, col_d in enumerate(sel_dist):
                            s = df[col_d].dropna()
                            fig.add_trace(go.Histogram(x=s, name=col_d,
                                                       histnorm='density' if norm_each else '', nbinsx=bins_c,
                                                       opacity=0.7 if barmode == 'overlay' else 1.0,
                                                       marker_color=colors[i % len(colors)]))
                        fig.update_layout(barmode=barmode,
                                          title="Distribution Comparison",
                                          template="plotly_white", height=420,
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.caption("📌 **When to use:** Compare shape of multiple features side-by-side.")

                elif chart_type_stat == "Error Bar Chart":
                    if not cat_cols:
                        st.warning("Need a categorical column.")
                    else:
                        x_eb  = st.selectbox("Category (X)", cat_cols, key="stat_eb_x")
                        y_eb  = st.selectbox("Numeric (Y)", numeric_cols, key="stat_eb_y")
                        ci    = st.selectbox("Error Type", ["Std Dev","Std Error","95% CI"], key="stat_eb_ci")
                        
                        if x_eb and y_eb:
                            grp   = df.groupby(x_eb)[y_eb]
                            means = grp.mean()
                            stds  = grp.std()
                            cnts  = grp.count()
                            if ci == "Std Dev":
                                err = stds
                            elif ci == "Std Error":
                                err = stds / np.sqrt(cnts)
                            else:
                                err = 1.96 * stds / np.sqrt(cnts)
                                
                            fig = go.Figure(go.Bar(
                                x=means.index.astype(str), y=means.values,
                                text=np.round(means.values, 2), textposition='auto',
                                error_y=dict(type='data', array=err.values,
                                             color='#2c3e50', thickness=2, width=4),
                                marker=dict(color='#3498db', line=dict(color='#2980B9', width=1))
                            ))
                            fig.update_layout(title=f"Mean {y_eb} by {x_eb} ± {ci}",
                                              template="plotly_white", height=420)
                            st.caption("📌 **When to use:** Mean comparison with uncertainty measure.")

                elif chart_type_stat == "Histogram + KDE Overlay":
                    col_h = st.selectbox("Feature", numeric_cols, key="stat_hkde_col")
                    color = st.selectbox("Group By (Optional)", [None] + cat_cols, key="stat_hkde_c")
                    bins  = st.slider("Bins", 10, 100, 30, key="stat_hkde_bins")
                    
                    with st.expander("🛠️ Advanced Options"):
                        marginal = st.selectbox("Marginal Plot", ["box", "violin", "rug"], key="stat_hkde_marg")

                    if col_h:
                        fig = px.histogram(df, x=col_h, color=color, nbins=bins,
                                           histnorm='density', marginal=marginal,
                                           title=f"Histogram + Density: {col_h}",
                                           template="plotly_white",
                                           color_discrete_sequence=px.colors.qualitative.Set2,
                                           opacity=0.75)
                        fig.update_layout(barmode='overlay')
                        st.caption("📌 **When to use:** Full picture of distribution shape with marginals.")

                elif chart_type_stat == "Cumulative Distribution":
                    sel_cdf = st.multiselect("Numeric Columns", numeric_cols,
                                             default=numeric_cols[:min(3, len(numeric_cols))],
                                             key="stat_cdf_cols")
                    if sel_cdf:
                        fig = go.Figure()
                        colors = px.colors.qualitative.Bold
                        for i, col_c in enumerate(sel_cdf):
                            s   = df[col_c].dropna().sort_values()
                            cdf = np.arange(1, len(s)+1) / len(s)
                            fig.add_trace(go.Scatter(x=s, y=cdf, mode='lines',
                                                     name=col_c, fill='tozeroy', fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [0.1])}",
                                                     line=dict(color=colors[i % len(colors)], width=2.5)))
                        fig.update_layout(title="Empirical CDF Comparison",
                                          xaxis_title="Value", yaxis_title="Cumulative Probability",
                                          template="plotly_white", height=420, hovermode="x unified")
                        st.caption("📌 **When to use:** Compare value spread without binning.")

                elif chart_type_stat == "Feature Importance Bar":
                    if len(numeric_cols) < 2:
                        st.warning("Need ≥ 2 numeric columns.")
                    else:
                        target_fi = st.selectbox("Target Column", numeric_cols, key="stat_fi_target")
                        feature_cols_fi = [c for c in numeric_cols if c != target_fi]
                        
                        with st.expander("🛠️ Advanced Settings"):
                            orientation = st.selectbox("Orientation", ["Horizontal (h)", "Vertical (v)"], key="stat_fi_ori")
                        
                        if target_fi and feature_cols_fi:
                            corrs = df[feature_cols_fi].corrwith(df[target_fi]).sort_values(key=abs, ascending=True if orientation.startswith('H') else False)
                            
                            fig = px.bar(
                                x=corrs.values if orientation.startswith('H') else corrs.index, 
                                y=corrs.index if orientation.startswith('H') else corrs.values,
                                orientation='h' if orientation.startswith('H') else 'v',
                                labels={'x': 'Correlation' if orientation.startswith('H') else 'Feature',
                                        'y': 'Feature' if orientation.startswith('H') else 'Correlation'},
                                title=f"Pearson Correlation with '{target_fi}'",
                                template="plotly_white",
                                text=np.round(corrs.values, 3),
                                color=corrs.values, color_continuous_scale='RdBu_r', range_color=[-1, 1]
                            )
                            fig.update_traces(textposition='outside')
                            fig.update_layout(height=460)
                            st.caption("📌 **When to use:** Quick linear feature relevance screening.")

            with chart_col:
                if fig:
                    render_chart_panel(fig, f"{chart_type_stat}",
                                       f"stat_{chart_type_stat.replace(' ','_').lower()}")
                else:
                    st.markdown("""
                    <div style="height:380px; display:flex; align-items:center; justify-content:center;
                         border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                        ← Configure settings on the left to generate a chart
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — GEOGRAPHIC
    # ══════════════════════════════════════════════════════════════════════════
    with tab_geo:
        st.markdown('<div class="section-header">🌍 Geographic Visualization</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Spatial and regional distribution of data on a map.")

        if not cat_cols and not numeric_cols:
            st.warning("⚠️ No suitable columns found.")
        else:
            chart_type_geo = st.selectbox(
                "Chart Type",
                ["Choropleth Map (Country)", "Choropleth Map (US States)",
                 "Bubble Map", "Scatter Geo"],
                key="geo_chart_type"
            )

            settings_col, chart_col = st.columns([1, 2.5])
            fig = None

            with settings_col:
                st.markdown(f"**⚙️ Settings — {chart_type_geo}**")
                st.caption("⚠️ Your location column must contain standard country names or ISO codes.")

                if chart_type_geo == "Choropleth Map (Country)":
                    loc   = st.selectbox("Location Column (Country name/ISO)", all_cols, key="geo_choro_loc")
                    val   = st.selectbox("Value Column (Numeric)", numeric_cols, key="geo_choro_val")
                    scope = st.selectbox("Scope", ["world","europe","asia","africa",
                                                   "north america","south america"], key="geo_choro_scope")
                    
                    with st.expander("🗺️ Map Projection & Style"):
                        cscale = st.selectbox("Color Scale", ["Blues","Reds","Greens","Viridis","Plasma","YlOrRd"], key="geo_choro_cs")
                        projection = st.selectbox("Projection", ["equirectangular", "mercator", "orthographic", "natural earth", "kavrayskiy7"], key="geo_choro_proj")

                    if loc and val:
                        try:
                            fig = px.choropleth(df, locations=loc, color=val,
                                                locationmode="country names",
                                                color_continuous_scale=cscale,
                                                scope=scope, projection=projection,
                                                title=f"Choropleth: {val} by Country",
                                                template="plotly_white")
                            fig.update_layout(height=550, margin={"r":0,"t":40,"l":0,"b":0})
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

                elif chart_type_geo == "Choropleth Map (US States)":
                    loc   = st.selectbox("State Code Column (2-letter abbrev.)", all_cols, key="geo_us_loc")
                    val   = st.selectbox("Value Column (Numeric)", numeric_cols, key="geo_us_val")
                    
                    with st.expander("🗺️ Style Settings"):
                        cscale = st.selectbox("Color Scale", ["Blues","Reds","Greens","Viridis","YlOrRd"], key="geo_us_cs")
                    
                    if loc and val:
                        try:
                            fig = px.choropleth(df, locations=loc, color=val,
                                                locationmode="USA-states",
                                                color_continuous_scale=cscale,
                                                scope="usa",
                                                title=f"US State Map: {val}",
                                                template="plotly_white")
                            fig.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

                elif chart_type_geo == "Bubble Map":
                    lat   = st.selectbox("Latitude Column", numeric_cols, key="geo_bub_lat")
                    lon   = st.selectbox("Longitude Column", numeric_cols, key="geo_bub_lon", index=min(1, len(numeric_cols)-1))
                    size  = st.selectbox("Size (Numeric)", numeric_cols, key="geo_bub_sz", index=min(2, len(numeric_cols)-1))
                    color = st.selectbox("Color By", [None] + all_cols, key="geo_bub_c")
                    hover = st.selectbox("Hover Name", [None] + cat_cols, key="geo_bub_h")
                    
                    with st.expander("🗺️ Projection Settings"):
                        projection = st.selectbox("Projection", ["natural earth", "orthographic", "mercator", "equirectangular"], key="geo_bub_proj")
                        max_sz = st.slider("Max Bubble Size", 10, 80, 40, key="geo_bub_maxsz")

                    if lat and lon and size:
                        try:
                            fig = px.scatter_geo(df, lat=lat, lon=lon, size=size,
                                                 color=color, hover_name=hover,
                                                 projection=projection,
                                                 size_max=max_sz, opacity=0.75,
                                                 title="Bubble Map",
                                                 template="plotly_white")
                            fig.update_layout(height=550, margin={"r":0,"t":40,"l":0,"b":0})
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

                elif chart_type_geo == "Scatter Geo":
                    lat   = st.selectbox("Latitude Column", numeric_cols, key="geo_sg_lat")
                    lon   = st.selectbox("Longitude Column", numeric_cols, key="geo_sg_lon", index=min(1, len(numeric_cols)-1))
                    color = st.selectbox("Color By", [None] + all_cols, key="geo_sg_c")
                    hover = st.selectbox("Hover Name", [None] + cat_cols, key="geo_sg_h")
                    
                    with st.expander("🗺️ Projection Settings"):
                        projection = st.selectbox("Projection", ["natural earth", "orthographic", "mercator"], key="geo_sg_proj")

                    if lat and lon:
                        try:
                            fig = px.scatter_geo(df, lat=lat, lon=lon,
                                                 color=color, hover_name=hover,
                                                 projection=projection,
                                                 title="Scatter Geo Plot",
                                                 template="plotly_white")
                            fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='white')))
                            fig.update_layout(height=550, margin={"r":0,"t":40,"l":0,"b":0})
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

            with chart_col:
                if fig:
                    render_chart_panel(fig, f"{chart_type_geo}",
                                       f"geo_{chart_type_geo.replace(' ','_').lower()}")
                else:
                    st.markdown("""
                    <div style="height:380px; display:flex; align-items:center; justify-content:center;
                         border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                        ← Configure settings on the left to generate a chart
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — CUSTOM / ADVANCED
    # ══════════════════════════════════════════════════════════════════════════
    with tab_custom:
        st.markdown('<div class="section-header">🎨 Custom & Advanced Charts</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Power-user charts for advanced analytical needs.")

        chart_type_adv = st.selectbox(
            "Chart Type",
            ["Waterfall Chart", "Funnel Chart",
             "Gantt / Timeline Chart", "Heatmap (Custom)",
             "Density Contour", "Sankey Diagram (Preview)"],
            key="adv_chart_type"
        )

        settings_col, chart_col = st.columns([1, 2.5])
        fig = None

        with settings_col:
            st.markdown(f"**⚙️ Settings — {chart_type_adv}**")

            if chart_type_adv == "Waterfall Chart":
                if not numeric_cols or not cat_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    x_wf  = st.selectbox("Categories (X)", cat_cols, key="adv_wf_x")
                    y_wf  = st.selectbox("Values (Y)", numeric_cols, key="adv_wf_y")
                    top_n = st.slider("Top N", 3, 20, 8, key="adv_wf_n")
                    
                    with st.expander("🛠️ Styling"):
                        inc_color = st.color_picker("Increasing Color", "#2ecc71")
                        dec_color = st.color_picker("Decreasing Color", "#e74c3c")
                        
                    if x_wf and y_wf:
                        grp = df.groupby(x_wf)[y_wf].sum().nlargest(top_n).reset_index()
                        grp.columns = [x_wf, y_wf]
                        measures = ['relative'] * len(grp)
                        fig = go.Figure(go.Waterfall(
                            name="Waterfall", orientation="v",
                            measure=measures,
                            x=grp[x_wf].astype(str),
                            y=grp[y_wf],
                            textposition="outside",
                            text=np.round(grp[y_wf], 2),
                            connector={"line": {"color": "#7f8c8d", "width": 1.5, "dash": "dot"}},
                            increasing={"marker": {"color": inc_color}},
                            decreasing={"marker": {"color": dec_color}},
                        ))
                        fig.update_layout(title=f"Waterfall: {y_wf} by {x_wf}",
                                          template="plotly_white", height=460,
                                          showlegend=True)
                        st.caption("📌 **When to use:** Cumulative contribution of each category.")

            elif chart_type_adv == "Funnel Chart":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    x_fn  = st.selectbox("Stage (Category)", cat_cols, key="adv_fn_x")
                    y_fn  = st.selectbox("Count / Value", numeric_cols, key="adv_fn_y")
                    
                    with st.expander("🛠️ Formatting"):
                        orientation = st.selectbox("Orientation", ["Horizontal", "Vertical"], key="adv_fn_ori")

                    if x_fn and y_fn:
                        grp = df.groupby(x_fn)[y_fn].sum().reset_index().sort_values(y_fn, ascending=False)
                        if orientation == "Horizontal":
                            fig = px.funnel(grp, x=y_fn, y=x_fn,
                                            title=f"Funnel: {y_fn} by {x_fn}", template="plotly_white",
                                            color_discrete_sequence=px.colors.qualitative.Set2)
                        else:
                            fig = px.funnel(grp, x=x_fn, y=y_fn,
                                            title=f"Funnel: {y_fn} by {x_fn}", template="plotly_white",
                                            color_discrete_sequence=px.colors.qualitative.Set2)
                            
                        fig.update_layout(height=460)
                        st.caption("📌 **When to use:** Conversion pipeline / drop-off analysis.")

            elif chart_type_adv == "Gantt / Timeline Chart":
                if not cat_cols or not date_cols:
                    st.info("💡 Gantt charts need category and date columns (datetime type).\n"
                            "Convert date columns in Data Preprocessing → Data Types tab first.")
                else:
                    task_col  = st.selectbox("Task / Resource", cat_cols, key="adv_gantt_task")
                    start_col = st.selectbox("Start Date", date_cols, key="adv_gantt_start")
                    end_col   = st.selectbox("End Date",   date_cols, key="adv_gantt_end", index=min(1, len(date_cols)-1))
                    color_col = st.selectbox("Color By", [None] + cat_cols, key="adv_gantt_c")
                    
                    if task_col and start_col and end_col:
                        try:
                            fig = px.timeline(df, x_start=start_col, x_end=end_col,
                                              y=task_col, color=color_col,
                                              title="Gantt / Timeline Chart",
                                              template="plotly_white",
                                              color_discrete_sequence=px.colors.qualitative.Set2)
                            fig.update_yaxes(autorange="reversed")
                            fig.update_layout(height=500, xaxis_title="Date")
                            st.caption("📌 **When to use:** Project schedule and timeline analysis.")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

            elif chart_type_adv == "Heatmap (Custom)":
                if len(numeric_cols) < 2 or not cat_cols:
                    st.warning("Need at least one categorical and two numeric columns.")
                else:
                    row_c  = st.selectbox("Row (Y-Axis Category)", cat_cols, key="adv_hm_row")
                    col_c  = st.selectbox("Column (X-Axis Category)", cat_cols, key="adv_hm_col", index=min(1, len(cat_cols)-1))
                    val_c  = st.selectbox("Value (Numeric)", numeric_cols, key="adv_hm_val")
                    agg_hm = st.selectbox("Aggregation", ["mean","sum","count","max","min"], key="adv_hm_agg")
                    
                    with st.expander("🛠️ Heatmap Styling"):
                        cscale = st.selectbox("Color Scale", ["Blues","Reds","Greens","Viridis","YlOrRd","RdBu_r"], key="adv_hm_cs")
                        text_auto = st.checkbox("Show Values on Heatmap", value=False)

                    if row_c and col_c and val_c:
                        try:
                            pivot = df.groupby([row_c, col_c])[val_c].agg(agg_hm).unstack(fill_value=0)
                            fig   = px.imshow(pivot, aspect="auto",
                                              text_auto=".2s" if text_auto else False,
                                              color_continuous_scale=cscale,
                                              title=f"Heatmap: {agg_hm.title()} of {val_c}",
                                              labels=dict(x=col_c, y=row_c, color=val_c),
                                              template="plotly_white")
                            fig.update_layout(height=500)
                            st.caption("📌 **When to use:** Cross-tabulation patterns at a glance.")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

            elif chart_type_adv == "Density Contour":
                if len(numeric_cols) < 2:
                    st.warning("Need ≥ 2 numeric columns.")
                else:
                    x_dc   = st.selectbox("X-Axis", numeric_cols, key="adv_dc_x")
                    y_dc   = st.selectbox("Y-Axis", numeric_cols, key="adv_dc_y", index=min(1, len(numeric_cols)-1))
                    color  = st.selectbox("Color By (Optional)", [None] + cat_cols, key="adv_dc_c")
                    
                    with st.expander("🛠️ Advanced Settings"):
                        filled = st.checkbox("Fill Contours", value=True, key="adv_dc_fill")
                        n_contours = st.slider("Number of Contours", 5, 30, 15, key="adv_dc_n")

                    if x_dc and y_dc:
                        fig = px.density_contour(df, x=x_dc, y=y_dc, color=color,
                                                 marginal_x="histogram",
                                                 marginal_y="histogram",
                                                 title=f"Density Contour: {y_dc} vs {x_dc}",
                                                 template="plotly_white")
                        if filled:
                            fig.update_traces(contours_coloring="fill",
                                              contours=dict(showlines=False),
                                              ncontours=n_contours,
                                              colorscale="Blues",
                                              selector=dict(type='histogram2dcontour'))
                        else:
                            fig.update_traces(ncontours=n_contours, selector=dict(type='histogram2dcontour'))
                            
                        fig.update_layout(height=500)
                        st.caption("📌 **When to use:** Density in 2D space; better than scatter for large data.")

            elif chart_type_adv == "Sankey Diagram (Preview)":
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    src_col = st.selectbox("Source (Category)", cat_cols, key="adv_sk_src")
                    tgt_col = st.selectbox("Target (Category)", cat_cols, key="adv_sk_tgt", index=min(1, len(cat_cols)-1))
                    val_col = st.selectbox("Flow Value (Numeric)", numeric_cols, key="adv_sk_val")
                    
                    with st.expander("🛠️ Layout Tweaks"):
                        max_n   = st.slider("Max source categories", 3, 20, 8, key="adv_sk_maxn")
                        node_pad = st.slider("Node Padding", 5, 50, 15)
                        node_thickness = st.slider("Node Thickness", 5, 40, 20)

                    if src_col and tgt_col and val_col and src_col != tgt_col:
                        try:
                            grp = df.groupby([src_col, tgt_col])[val_col].sum().reset_index()
                            top_src = grp.groupby(src_col)[val_col].sum().nlargest(max_n).index
                            grp     = grp[grp[src_col].isin(top_src)]
                            all_labels = list(pd.unique(grp[[src_col, tgt_col]].values.ravel('K')))
                            label_map  = {v: i for i, v in enumerate(all_labels)}
                            
                            # Generating some dynamic colors for nodes
                            node_colors = px.colors.qualitative.Prism * (len(all_labels) // len(px.colors.qualitative.Prism) + 1)
                            
                            fig = go.Figure(go.Sankey(
                                arrangement="snap",
                                node=dict(label=all_labels,
                                          color=node_colors[:len(all_labels)],
                                          pad=node_pad, thickness=node_thickness,
                                          line=dict(color="black", width=0.5)),
                                link=dict(source=[label_map[s] for s in grp[src_col]],
                                          target=[label_map[t] for t in grp[tgt_col]],
                                          value=grp[val_col],
                                          color='rgba(189, 195, 199, 0.4)') # light grey links
                            ))
                            fig.update_layout(title=f"Sankey: {src_col} → {tgt_col}",
                                              template="plotly_white", height=550)
                            st.caption("📌 **When to use:** Flow or conversion between two categorical stages.")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

        with chart_col:
            if fig:
                render_chart_panel(fig, f"{chart_type_adv}",
                                   f"adv_{chart_type_adv.replace(' ','_').lower()}")
            else:
                st.markdown("""
                <div style="height:380px; display:flex; align-items:center; justify-content:center;
                     border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                    ← Configure settings on the left to generate a chart
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 — GROUPING & PIVOT SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_group:
        st.markdown('<div class="section-header">📑 Grouping & Pivot Summary</div>',
                    unsafe_allow_html=True)
        st.info("**Focus:** Create custom aggregation tables and visualize them instantly.")

        sub_grp, sub_pivot = st.tabs(["📊 Group By Summary", "🔢 Pivot Table"])

        # ── Group By Summary ─────────────────────────────────────────────────
        with sub_grp:
            col1, col2 = st.columns([1, 2])
            with col1:
                if not cat_cols or not numeric_cols:
                    st.warning("Need at least one categorical and one numeric column.")
                else:
                    group_cols = st.multiselect("Group By (Categories)", cat_cols,
                                                key="grp_group_cols",
                                                default=cat_cols[:1])
                    value_col  = st.selectbox("Summarize (Metric)", numeric_cols, key="grp_val")
                    agg_func   = st.selectbox("Aggregation",
                                              ["mean","sum","count","min","max","std","median"],
                                              key="grp_agg")
                    
                    with st.expander("⚙️ Sort & Filter Options"):
                        top_n_g    = st.slider("Top N rows to show", 5, 100, 20, key="grp_topn")
                        sort_desc  = st.checkbox("Sort Descending", True, key="grp_sort")

                    chart_for_grp = st.selectbox("Visualize As",
                                                  ["Bar Chart","Horizontal Bar","Pie Chart","Treemap", "Donut Chart"],
                                                  key="grp_chart")

                    if group_cols and value_col:
                        grp_df = df.groupby(group_cols)[value_col].agg(agg_func).reset_index()
                        grp_df.columns = group_cols + [value_col]
                        grp_df = grp_df.sort_values(value_col, ascending=not sort_desc).head(top_n_g)

                        st.markdown(f"**Result: {len(grp_df)} rows**")
                        st.dataframe(grp_df, use_container_width=True)

                        csv_g = grp_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download CSV", data=csv_g,
                                           file_name="group_summary.csv", mime="text/csv")

            with col2:
                if 'grp_df' in dir() and group_cols and value_col and len(grp_df) > 0:
                    x_lbl = group_cols[0] if len(group_cols) == 1 else "_".join(group_cols[:2])
                    if chart_for_grp == "Bar Chart":
                        fig_g = px.bar(grp_df, x=x_lbl, y=value_col, text=value_col,
                                       title=f"{agg_func.title()} of {value_col}",
                                       template="plotly_white",
                                       color=value_col, color_continuous_scale='Blues')
                        fig_g.update_traces(texttemplate='%{text:.3s}', textposition='outside', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
                    
                    elif chart_for_grp == "Horizontal Bar":
                        fig_g = px.bar(grp_df, x=value_col, y=x_lbl, orientation='h',
                                       text=value_col,
                                       title=f"{agg_func.title()} of {value_col}",
                                       template="plotly_white",
                                       color=value_col, color_continuous_scale='Blues')
                        fig_g.update_traces(texttemplate='%{text:.3s}', textposition='outside', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
                    
                    elif chart_for_grp == "Pie Chart":
                        fig_g = px.pie(grp_df, names=x_lbl, values=value_col,
                                       title=f"Proportion of {value_col}",
                                       template="plotly_white",
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_g.update_traces(textposition='inside', textinfo='percent+label')
                    
                    elif chart_for_grp == "Donut Chart":
                        fig_g = px.pie(grp_df, names=x_lbl, values=value_col, hole=0.4,
                                       title=f"Proportion of {value_col}",
                                       template="plotly_white",
                                       color_discrete_sequence=px.colors.qualitative.Safe)
                        fig_g.update_traces(textposition='inside', textinfo='percent+label')
                    
                    elif chart_for_grp == "Treemap":
                        fig_g = px.treemap(grp_df, path=group_cols, values=value_col,
                                           title=f"Treemap of {value_col}",
                                           template="plotly_white",
                                           color=value_col, color_continuous_scale='Blues')
                        fig_g.update_traces(textinfo="label+value+percent root")

                    if 'fig_g' in dir():
                        fig_g.update_layout(height=480)
                        st.plotly_chart(fig_g, use_container_width=True)
                        if st.button("📌 Pin to Dashboard", key="grp_pin"):
                            add_to_dashboard(fig_g, f"Group Summary: {agg_func} of {value_col}")
                else:
                    st.markdown("""
                    <div style="height:360px; display:flex; align-items:center; justify-content:center;
                         border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                        Select columns on the left to generate a summary
                    </div>
                    """, unsafe_allow_html=True)

        # ── Pivot Table ──────────────────────────────────────────────────────
        piv_rows = piv_cols = piv_val = piv_agg = show_total = colorscale = None
        
        with sub_pivot:
            col1, col2 = st.columns([1, 2])
            with col1:
                if not cat_cols or not numeric_cols:
                    st.warning("Need categorical and numeric columns.")
                else:
                    piv_rows   = st.selectbox("Rows", cat_cols, key="piv_rows")
                    piv_cols   = st.selectbox("Columns", cat_cols, key="piv_cols", index=min(1, len(cat_cols)-1))
                    piv_val    = st.selectbox("Values", numeric_cols, key="piv_val")
                    piv_agg    = st.selectbox("Aggregation",
                                              ["mean","sum","count","min","max"],
                                              key="piv_agg")
                    
                    with st.expander("⚙️ Options & Styling"):
                        show_total = st.checkbox("Add Row/Col Totals", True, key="piv_totals")
                        colorscale = st.selectbox("Heatmap Color",
                                                  ["Blues","Reds","Greens","YlOrRd","Viridis","Cividis"],
                                                  key="piv_cs")
                        text_format = st.selectbox("Text Format", [".1f", ".2f", ".2s", ".0f"], index=2)

                    if piv_rows and piv_cols and piv_val:
                        try:
                            piv_table = df.pivot_table(index=piv_rows, columns=piv_cols,
                                                        values=piv_val, aggfunc=piv_agg,
                                                        fill_value=0)
                            if show_total:
                                piv_table.loc['Total'] = piv_table.sum()
                                piv_table['Total']     = piv_table.sum(axis=1)

                            st.markdown(f"**Pivot: {piv_rows} × {piv_cols}**")
                            st.dataframe(piv_table, use_container_width=True)

                            csv_p = piv_table.to_csv().encode('utf-8')
                            st.download_button("📥 Download Pivot CSV", data=csv_p,
                                               file_name="pivot_table.csv", mime="text/csv")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

            with col2:
                if piv_rows and piv_cols and piv_val and 'piv_table' in dir():
                    try:
                        display_piv = piv_table.drop('Total', errors='ignore').drop(columns='Total', errors='ignore')
                        fig_piv = px.imshow(display_piv,
                                            text_auto=text_format,
                                            aspect="auto",
                                            color_continuous_scale=colorscale,
                                            labels=dict(x=piv_cols, y=piv_rows, color=piv_val),
                                            title=f"Pivot Heatmap: {piv_agg.title()} of {piv_val}",
                                            template="plotly_white")
                        fig_piv.update_layout(height=480, xaxis_nticks=len(display_piv.columns), yaxis_nticks=len(display_piv.index))
                        fig_piv.update_xaxes(side="bottom") # move labels to bottom instead of top default in imshow
                        st.plotly_chart(fig_piv, use_container_width=True)
                        if st.button("📌 Pin to Dashboard", key="piv_pin"):
                            add_to_dashboard(fig_piv, f"Pivot Heatmap: {piv_val}")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
                else:
                    st.markdown("""
                    <div style="height:360px; display:flex; align-items:center; justify-content:center;
                         border:2px dashed #AED6F1; border-radius:12px; color:#7f8c8d; font-size:15px;">
                        Configure a pivot table on the left to see it here
                    </div>
                    """, unsafe_allow_html=True)

    # ==========================================
    # ML Section
    # ==========================================


def ml_algorithm_page():
    
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -----------------------------
    import pandas as pd
    import numpy as np
    import time
    import joblib
    import io
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.metrics import (
        r2_score, mean_absolute_error, mean_squared_error,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_curve, auc, classification_report
    )
    
    # Model Imports
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier, 
        GradientBoostingRegressor, GradientBoostingClassifier,
        AdaBoostClassifier
    )
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    try:
        from xgboost import XGBRegressor, XGBClassifier
    except ImportError:
        XGBRegressor, XGBClassifier = None, None

    st.markdown('<div class="main-header">🤖 Machine Learning Intelligence</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the 'Data Upload' section first.")
        return

    df = st.session_state.df.copy()
    
    # 📋 DATASET OVERVIEW BAR
    st.markdown("""<div class="section-header">📋 Dataset Overview</div>""", unsafe_allow_html=True)
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Total Rows", f"{len(df):,}")
    o2.metric("Total Columns", f"{len(df.columns)}")
    o3.metric("Numeric Cols", len(df.select_dtypes(include=np.number).columns))
    o4.metric("Categorical Cols", len(df.select_dtypes(include=['object', 'category']).columns))

    # ==========================================
    # STEP 1 & 2: TARGET, TASK & FEATURE SELECTION
    # ==========================================
    with st.expander("🎯 Step 1 & 2: Target & Feature Setup", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Select Target Column (Y)", df.columns, index=len(df.columns)-1)
            
            # Auto-detect task type
            is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
            unique_vals = df[target_col].nunique()
            default_task = "Regression" if (is_numeric_target and unique_vals > 10) else "Classification"
            
            task_type = st.radio("Task Type", ["Regression", "Classification"], 
                                 index=0 if default_task == "Regression" else 1, horizontal=True)

        with col2:
            # Smart feature selection
            potential_features = [c for c in df.columns if c != target_col]
            # Exclude likely ID or High Cardinality/Date columns by default
            default_features = [c for c in potential_features if df[c].nunique() < len(df) * 0.9]
            
            selected_features = st.multiselect("Select Feature Columns (X)", 
                                               potential_features, default=default_features)

        if not selected_features:
            st.error("Please select at least one feature.")
            return

    # ==========================================
    # STEP 4: TRAIN / TEST SPLIT
    # ==========================================
    with st.expander("⚖️ Step 4: Train / Test Split Settings"):
        ts_c1, ts_c2, ts_c3 = st.columns(3)
        test_size = ts_c1.slider("Test Size (%)", 10, 40, 20) / 100
        random_state = ts_c2.number_input("Random Seed", 0, 9999, 42)
        stratify_data = None
        if task_type == "Classification":
            if ts_c3.checkbox("Stratify Split", value=True):
                stratify_data = df[target_col]

    # ==========================================
    # STEP 5: MODEL SELECTION & HYPERPARAMETERS
    # ==========================================
    st.markdown(f'<div class="section-header">⚙️ Step 5: {task_type} Model Configuration</div>', unsafe_allow_html=True)
    
    if task_type == "Regression":
        model_choice = st.selectbox("Select Algorithm", [
            "Linear Regression", "Ridge", "Lasso", "ElasticNet", 
            "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "SVR", "KNN"
        ])
        params = {}
        if model_choice in ["Ridge", "Lasso", "ElasticNet"]:
            params['alpha'] = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0)
        if model_choice == "ElasticNet":
            params['l1_ratio'] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        if model_choice in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]:
            params['max_depth'] = st.slider("Max Depth", 1, 50, 5)
        if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            params['n_estimators'] = st.slider("n_estimators", 10, 500, 100)
        if model_choice == "KNN":
            params['n_neighbors'] = st.slider("K Neighbors", 1, 30, 5)
            
    else: # Classification
        model_choice = st.selectbox("Select Algorithm", [
            "Logistic Regression", "Decision Tree", "Random Forest", 
            "Gradient Boosting", "XGBoost", "SVC", "KNN", "Naive Bayes", "AdaBoost"
        ])
        params = {}
        if model_choice == "Logistic Regression":
            params['C'] = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0)
        if model_choice in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]:
            params['max_depth'] = st.slider("Max Depth", 1, 50, 5)
        if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost", "AdaBoost"]:
            params['n_estimators'] = st.slider("n_estimators", 10, 500, 100)
        if model_choice == "SVC":
            params['C'] = st.slider("C", 0.01, 10.0, 1.0)
            params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        if model_choice == "KNN":
            params['n_neighbors'] = st.slider("K Neighbors", 1, 30, 5)

    # ==========================================
    # STEP 3 & 6: PIPELINE BUILDING & TRAINING
    # ==========================================
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner(f"Building pipeline and training {model_choice}..."):
            start_time = time.time()
            
            # Prepare Data
            X = df[selected_features]
            y = df[target_col]
            
            # Encode Target if Classification
            label_encoder = None
            if task_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                st.session_state['ml_label_encoder'] = label_encoder

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
            )

            # Automated Preprocessing
            num_features = X.select_dtypes(include=np.number).columns.tolist()
            cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ])

            # Model Instantiation
            if task_type == "Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge": Ridge(alpha=params.get('alpha', 1.0)),
                    "Lasso": Lasso(alpha=params.get('alpha', 1.0)),
                    "ElasticNet": ElasticNet(alpha=params.get('alpha', 1.0), l1_ratio=params.get('l1_ratio', 0.5)),
                    "Decision Tree": DecisionTreeRegressor(max_depth=params.get('max_depth')),
                    "Random Forest": RandomForestRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth')),
                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth')),
                    "XGBoost": XGBRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth')) if XGBRegressor else LinearRegression(),
                    "SVR": SVR(),
                    "KNN": KNeighborsRegressor(n_neighbors=params.get('n_neighbors', 5))
                }
            else:
                models = {
                    "Logistic Regression": LogisticRegression(C=params.get('C', 1.0), max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(max_depth=params.get('max_depth')),
                    "Random Forest": RandomForestClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth')),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth')),
                    "XGBoost": XGBClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth')) if XGBClassifier else LogisticRegression(),
                    "SVC": SVC(C=params.get('C', 1.0), kernel=params.get('kernel', 'rbf'), probability=True),
                    "KNN": KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5)),
                    "Naive Bayes": GaussianNB(),
                    "AdaBoost": AdaBoostClassifier(n_estimators=params.get('n_estimators', 100))
                }

            # Final Pipeline
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', models[model_choice])
            ])

            # Fit
            model_pipeline.fit(X_train, y_train)
            
            # Save to session state
            st.session_state['trained_model'] = model_pipeline
            st.session_state['ml_results'] = {
                'X_test': X_test, 'y_test': y_test, 
                'y_pred': model_pipeline.predict(X_test),
                'task': task_type, 'model_name': model_choice,
                'train_time': time.time() - start_time,
                'features': selected_features,
                'target': target_col
            }
            st.success(f"✅ {model_choice} trained in {st.session_state.ml_results['train_time']:.2f}s")

    # ==========================================
    # STEP 7: RESULTS DASHBOARD
    # ==========================================
    if 'ml_results' in st.session_state:
        res = st.session_state.ml_results
        pipeline = st.session_state.trained_model
        
        tab_met, tab_viz, tab_feat, tab_cv = st.tabs(["📊 Metrics", "📈 Visualizations", "🔍 Importance", "🔁 Cross-Val"])

        with tab_met:
            if res['task'] == "Regression":
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2_score(res['y_test'], res['y_pred']):.4f}")
                m2.metric("MAE", f"{mean_absolute_error(res['y_test'], res['y_pred']):.2f}")
                m3.metric("RMSE", f"{np.sqrt(mean_squared_error(res['y_test'], res['y_pred'])):.2f}")
                m4.metric("Train Time", f"{res['train_time']:.2f}s")
            else:
                m1, m2, m3, m4 = st.columns(4)
                acc = accuracy_score(res['y_test'], res['y_pred'])
                f1 = f1_score(res['y_test'], res['y_pred'], average='weighted')
                m1.metric("Accuracy", f"{acc:.2%}")
                m2.metric("F1 Score", f"{f1:.4f}")
                m3.metric("Samples", len(res['y_test']))
                m4.metric("Train Time", f"{res['train_time']:.2f}s")
                
                st.markdown("**Classification Report**")
                report = classification_report(res['y_test'], res['y_pred'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        with tab_viz:
            if res['task'] == "Regression":
                fig = px.scatter(x=res['y_test'], y=res['y_pred'], labels={'x': 'Actual', 'y': 'Predicted'},
                                 title="Actual vs Predicted", template="plotly_white")
                fig.add_shape(type="line", x0=res['y_test'].min(), y0=res['y_test'].min(), 
                              x1=res['y_test'].max(), y1=res['y_test'].max(), line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                cm = confusion_matrix(res['y_test'], res['y_pred'])
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", 
                                labels=dict(x="Predicted", y="Actual"),
                                template="plotly_white", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

        with tab_feat:
            try:
                # Extract feature names from one-hot encoder
                ohe_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
                all_feature_names = res['features'] # Simpler fallback if complex mapping fails
                
                model = pipeline.named_steps['model']
                importances = None
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_.flatten())
                
                if importances is not None:
                    # Logic to align importance with raw features for simplicity
                    feat_imp = pd.DataFrame({'Feature': res['features'][:len(importances)], 'Importance': importances[:len(res['features'])]})
                    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
                    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Importance")
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            except:
                st.info("Feature importance visualization currently unavailable for this configuration.")

        with tab_cv:
            folds = st.slider("K-Fold Cross Validation", 2, 10, 5)
            if st.button("Run Cross-Validation"):
                metric = 'r2' if res['task'] == "Regression" else 'accuracy'
                scores = cross_val_score(pipeline, df[res['features']], df[res['target']], cv=folds, scoring=metric)
                st.write(f"Mean {metric.upper()}: **{scores.mean():.4f}** (± {scores.std():.4f})")
                fig_cv = px.line(x=range(1, folds+1), y=scores, markers=True, title="CV Scores per Fold")
                st.plotly_chart(fig_cv, use_container_width=True)

    # ==========================================
    # STEP 8: PREDICTION INTERFACE
    # ==========================================
    if 'trained_model' in st.session_state:
        st.markdown('<div class="section-header">🔮 Step 8: Make Predictions</div>', unsafe_allow_html=True)
        pred_mode = st.radio("Prediction Mode", ["🖊️ Manual Input", "📁 Batch Upload"], horizontal=True)
        
        if pred_mode == "🖊️ Manual Input":
            with st.form("pred_form"):
                cols = st.columns(3)
                input_data = {}
                for i, feat in enumerate(res['features']):
                    with cols[i % 3]:
                        if pd.api.types.is_numeric_dtype(df[feat]):
                            input_data[feat] = st.number_input(f"{feat}", value=float(df[feat].median()))
                        else:
                            input_data[feat] = st.selectbox(f"{feat}", df[feat].unique())
                
                if st.form_submit_button("Predict"):
                    input_df = pd.DataFrame([input_data])
                    prediction = pipeline.predict(input_df)[0]
                    
                    # Inverse transform if encoded
                    if 'ml_label_encoder' in st.session_state:
                        prediction = st.session_state.ml_label_encoder.inverse_transform([prediction])[0]
                    
                    st.markdown(f"""
                    <div style="background:#eaf4fb; padding:20px; border-radius:10px; border-left:5px solid #3498db; text-align:center;">
                        <h2 style="margin:0; color:#2c3e50;">Result: {prediction}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            batch_file = st.file_uploader("Upload CSV for prediction", type=['csv'])
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                if st.button("Generate Batch Predictions"):
                    preds = pipeline.predict(batch_df[res['features']])
                    if 'ml_label_encoder' in st.session_state:
                        preds = st.session_state.ml_label_encoder.inverse_transform(preds)
                    batch_df['Prediction'] = preds
                    st.dataframe(batch_df)
                    st.download_button("Download Predictions", batch_df.to_csv(index=False), "predictions.csv", "text/csv")

    # ==========================================
    # STEP 9: EXPORT
    # ==========================================
    if 'trained_model' in st.session_state:
        st.markdown('<div class="section-header">💾 Step 9: Export Model</div>', unsafe_allow_html=True)
        model_pkl = io.BytesIO()
        joblib.dump(st.session_state.trained_model, model_pkl)
        st.download_button(
            label="📥 Download Trained Pipeline (.pkl)",
            data=model_pkl.getvalue(),
            file_name=f"datamate_model_{res['model_name']}.pkl",
            mime="application/octet-stream"
        )



# =========================================================
# === EXPORT PAGE =========================================
# =========================================================
def export_page():
    """Professional Multi-Format Export Center — DataMate AI"""
    
    
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -----------------------------
    
    import io
    import os
    import tempfile
    import plotly.io as pio
    import pandas as pd
    import traceback
    from fpdf import FPDF
    from datetime import datetime

    # ── Embedded PDF Class ────────────────────────────────────────────────────
    class PDFReport(FPDF):
        def __init__(self):
            super().__init__()
            self._section_count = 0

        def header(self):
            self.set_fill_color(15, 32, 39)
            self.rect(0, 0, self.w, 18, 'F')
            self.set_font('Arial', 'B', 11)
            self.set_text_color(255, 255, 255)
            self.cell(0, 18, 'DataMate AI  |  Analysis Report', 0, 1, 'C')
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-14)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(127, 140, 141)
            self.cell(0, 10, f'Page {self.page_no()}  |  Generated by DataMate AI', 0, 0, 'C')
            self.set_text_color(0, 0, 0)

        def section_title(self, title):
            self._section_count += 1
            if self.get_y() > 245:
                self.add_page()
            self.ln(4)
            self.set_fill_color(52, 152, 219)
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 11)
            self.cell(0, 9, f'  {self._section_count}. {self.safe(title)}', 0, 1, 'L', True)
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def kv_row(self, key, value, shade=False):
            self.set_font('Arial', 'B', 9)
            if shade:
                self.set_fill_color(241, 246, 249)
                self.cell(65, 7, f'  {self.safe(key)}', 0, 0, 'L', True)
            else:
                self.cell(65, 7, f'  {self.safe(key)}', 0, 0)
            self.set_font('Arial', '', 9)
            val_str = str(self.safe(value))
            if len(val_str) > 70:
                val_str = val_str[:67] + "..."
            self.cell(0, 7, val_str, 0, 1)

        def body(self, text, size=9):
            self.set_font('Arial', '', size)
            self.multi_cell(0, 5.5, self.safe(text))
            self.ln(2)

        def add_chart_image(self, path, title=''):
            if self.get_y() > 175:
                self.add_page()
            if title:
                self.set_font('Arial', 'B', 10)
                self.set_text_color(44, 62, 80)
                self.cell(0, 8, self.safe(title), 0, 1)
                self.set_text_color(0, 0, 0)
            try:
                self.image(path, x=10, w=self.w - 20)
            except Exception as e:
                self.set_font('Arial', '', 8)
                self.set_text_color(200, 0, 0)
                self.multi_cell(0, 5, f'[Chart error: {str(e)[:60]}]')
                self.set_text_color(0, 0, 0)
            self.ln(4)

        def safe(self, text):
            try:
                if text is None:
                    return ''
                return str(text).encode('latin-1', 'replace').decode('latin-1')
            except Exception:
                return '[text error]'

    # ── Page Header ───────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">💾 Export Center</div>', unsafe_allow_html=True)

    if st.session_state.get('df') is None:
        st.warning("⚠️ No dataset loaded. Please upload data in the 'Data Upload' section first.")
        return

    df = st.session_state.df

    # ── Dataset Quick Stats ───────────────────────────────────────────────────
    e1, e2, e3, e4, e5 = st.columns(5)
    
    try:
        total_missing = df.isnull().sum().sum()
    except Exception:
        total_missing = 0
        
    kpi_data = [
        ("Rows",          f"{len(df):,}",                    "#3498db"),
        ("Columns",       f"{len(df.columns)}",              "#3498db"),
        ("Missing Cells", f"{total_missing:,}",              "#3c9ae7" if total_missing > 0 else "#3492d1"),
        ("Saved Charts",  f"{len(st.session_state.get('saved_charts', []))}",  "#3693d1"),
        ("Dataset Name",  str(st.session_state.get('selected_file_name', 'N/A'))[:18], "#3695cc"),
    ]
    for col_ui, (label, val, color) in zip([e1, e2, e3, e4, e5], kpi_data):
        with col_ui:
            st.markdown(f"""
            <div class="metric-card" style="border-left:4px solid {color}; text-align:center;">
                <p style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
                          letter-spacing:1px;margin:0;">{label}</p>
                <p style="font-size:20px;font-weight:800;color:{color};
                          margin:6px 0 0 0;font-family:Sora,sans-serif;">{val}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Export Tabs ───────────────────────────────────────────────────────────
    tab_pdf, tab_html, tab_data, tab_txt = st.tabs([
        "📄 PDF Report", "🌐 HTML Dashboard", "📊 Export Dataset", "📝 TXT Summary"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – PDF REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pdf:
        st.markdown('<div class="section-header">📄 Multi-Page PDF Report</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.markdown("""
            <div class="content-frame">
                <p style="font-size:14px; color:#555; margin:0;">
                    Generate a professional, print-ready PDF containing dataset overview,
                    statistics, column profiles, ML model info, and all pinned charts.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Included Sections:**")
            include_overview  = st.checkbox("Dataset Overview",   value=True,  key="pdf_overview")
            include_stats     = st.checkbox("Statistical Summary", value=True, key="pdf_stats")
            include_profile   = st.checkbox("Column Profile (Max 30)", value=True, key="pdf_profile")
            include_ml        = st.checkbox("ML Model Info",       value=True, key="pdf_ml")
            include_charts    = st.checkbox("Pinned Visualizations",value=True, key="pdf_charts")

            chart_count = len(st.session_state.get('saved_charts', []))
            if include_charts and chart_count == 0:
                st.caption("⚠️ No charts pinned yet. Visit Visualization page to pin charts.")

            gen_pdf = st.button("⚙️ Generate PDF Report", key="btn_gen_pdf",
                                use_container_width=True, type="primary")

        with col2:
            st.markdown("""
            <div class="content-frame" style="background:linear-gradient(135deg,#f8f9fa,#fff);
                 border-left:4px solid #3498db;">
                <h4 style="font-family:Sora,sans-serif; margin-top:0; color:#1a2535;">
                    📋 Report Preview
                </h4>
                <ul style="color:#555; font-size:13px; line-height:2;">
                    <li>Cover page with dataset metadata</li>
                    <li>Key statistics table (Transposed for clean fit)</li>
                    <li>Column-by-column data profile</li>
                    <li>Missing values summary</li>
                    <li>ML model performance (if trained)</li>
                    <li>All pinned Plotly charts as images</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if gen_pdf:
            with st.spinner("🔄 Building PDF — this may take a moment for large datasets..."):
                try:
                    pdf = PDFReport()
                    pdf.set_auto_page_break(auto=True, margin=20)

                    # ── PAGE 1: COVER ──
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 20)
                    pdf.set_text_color(44, 62, 80)
                    pdf.ln(10)
                    pdf.cell(0, 12, 'DataMate AI', 0, 1, 'C')
                    pdf.set_font('Arial', '', 13)
                    pdf.set_text_color(127, 140, 141)
                    pdf.cell(0, 8, 'Data Analysis Report', 0, 1, 'C')
                    pdf.ln(6)

                    pdf.set_fill_color(52, 152, 219)
                    pdf.rect(10, pdf.get_y(), pdf.w - 20, 2, 'F')
                    pdf.ln(10)

                    pdf.set_text_color(0, 0, 0)
                    file_name = st.session_state.get('selected_file_name', 'Unknown')
                    gen_time  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    try:
                        mem_usage = f"{df.memory_usage(deep=True).sum()/1024**2:.2f}"
                    except Exception:
                        mem_usage = "N/A"

                    for i, (k, v) in enumerate([
                        ("File Name",    file_name),
                        ("Generated",    gen_time),
                        ("Total Rows",   f"{len(df):,}"),
                        ("Total Columns",f"{len(df.columns)}"),
                        ("Missing Cells",f"{total_missing:,}"),
                        ("Duplicates",   f"{df.duplicated().sum():,}"),
                        ("Memory (MB)",  mem_usage),
                    ]):
                        pdf.kv_row(k, v, shade=(i % 2 == 0))

                    # ── SECTION: STATISTICAL SUMMARY (FIXED FOR LARGE COLS) ──
                    if include_stats:
                        pdf.add_page()
                        pdf.section_title("Statistical Summary (Numeric Columns)")
                        try:
                            # Using Transpose (.T) so columns become rows. Fits perfectly on PDF.
                            num_df = df.select_dtypes(include='number')
                            if not num_df.empty:
                                stats = num_df.describe().T.round(2)
                                
                                # Table Header
                                pdf.set_font('Arial', 'B', 8)
                                pdf.set_fill_color(230, 230, 230)
                                pdf.cell(50, 6, "Column Name", border=1, fill=True)
                                pdf.cell(20, 6, "Count", border=1, fill=True)
                                pdf.cell(25, 6, "Mean", border=1, fill=True)
                                pdf.cell(25, 6, "Std Dev", border=1, fill=True)
                                pdf.cell(20, 6, "Min", border=1, fill=True)
                                pdf.cell(20, 6, "Max", border=1, fill=True)
                                pdf.ln()
                                
                                # Table Body
                                pdf.set_font('Arial', '', 8)
                                for idx, row in stats.iterrows():
                                    col_name = pdf.safe(str(idx))
                                    if len(col_name) > 25: col_name = col_name[:22] + "..."
                                    pdf.cell(50, 6, col_name, border=1)
                                    pdf.cell(20, 6, str(row.get('count', 0)), border=1)
                                    pdf.cell(25, 6, str(row.get('mean', 0)), border=1)
                                    pdf.cell(25, 6, str(row.get('std', 0)), border=1)
                                    pdf.cell(20, 6, str(row.get('min', 0)), border=1)
                                    pdf.cell(20, 6, str(row.get('max', 0)), border=1)
                                    pdf.ln()
                            else:
                                pdf.body("No numeric columns available to summarize.")
                        except Exception as e:
                            pdf.body(f"Could not generate statistics table. Error: {str(e)[:50]}")

                    # ── SECTION: COLUMN PROFILE ──
                    if include_profile:
                        pdf.add_page()
                        pdf.section_title("Column Profile (Max 30)")
                        for col in df.columns[:30]:  
                            try:
                                missing_pct = df[col].isnull().mean() * 100
                                unique_vals = df[col].nunique()
                                dtype = str(df[col].dtype)
                                pdf.kv_row(col,
                                           f"Type: {dtype}  |  Unique: {unique_vals}  |  Missing: {missing_pct:.1f}%",
                                           shade=(list(df.columns).index(col) % 2 == 0))
                            except Exception:
                                pdf.kv_row(col, "Could not profile this column", shade=False)

                    # ── SECTION: ML MODEL INFO ──
                    if include_ml and 'ml_results' in st.session_state:
                        res = st.session_state.ml_results
                        pdf.add_page()
                        pdf.section_title("Machine Learning Model")
                        for i, (k, v) in enumerate([
                            ("Algorithm",     res.get('model_name',  'Unknown')),
                            ("Task Type",     res.get('task',        'Unknown')),
                            ("Target Column", res.get('target',      'Unknown')),
                            ("Features Used", str(len(res.get('features', [])))),
                            ("Training Time", f"{res.get('train_time', 0):.2f}s"),
                        ]):
                            pdf.kv_row(k, v, shade=(i % 2 == 0))

                    # ── SECTION: CHARTS ──
                    saved_charts = st.session_state.get('saved_charts', [])
                    if include_charts and saved_charts:
                        pdf.add_page()
                        pdf.section_title("Pinned Visualizations")
                        for i, chart in enumerate(saved_charts):
                            fig = chart.get('fig')
                            title = chart.get('title', f'Chart {i+1}')
                            desc  = chart.get('description', '')
                            if fig is None:
                                continue
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            try:
                                fig.write_image(tmp.name, width=750, height=420, scale=1.5)
                                pdf.add_chart_image(tmp.name, title=f"Fig {i+1}: {title}")
                                if desc:
                                    pdf.set_font('Arial', 'I', 8)
                                    pdf.set_text_color(44, 110, 160)
                                    pdf.multi_cell(0, 5, f'Analysis Note: {desc}')
                                    pdf.set_text_color(0, 0, 0)
                                    pdf.ln(2)
                            except ImportError:
                                pdf.body('[Chart skipped: Please install kaleido -> pip install kaleido]')
                            except Exception as e:
                                pdf.body(f'[Chart {i+1} error: {str(e)[:60]}]')
                            finally:
                                try:
                                    os.unlink(tmp.name)
                                except Exception:
                                    pass

                    # Save PDF Safely
                    raw = pdf.output(dest='S')
                    pdf_bytes = raw.encode('latin-1') if isinstance(raw, str) else bytes(raw)
                    st.session_state['pdf_bytes'] = pdf_bytes
                    st.success(f"✅ PDF ready — {len(pdf_bytes)/1024:.1f} KB across {pdf.page} pages.")

                except Exception as e:
                    st.error(f"❌ Critical PDF Error: {e}")
                    st.code(traceback.format_exc())

        if st.session_state.get('pdf_bytes'):
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state['pdf_bytes'],
                file_name=f"DataMate_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – HTML DASHBOARD (FIXED CHECKBOXES & GENERATION)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_html:
        st.markdown('<div class="section-header">🌐 Interactive HTML Dashboard</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="content-frame">
            <p style="font-size:14px; color:#555; margin:0;">
                Export your KPI cards and all pinned visualizations as a fully self-contained,
                offline interactive HTML file. Charts remain interactive (zoom, hover, filter).
            </p>
        </div>
        """, unsafe_allow_html=True)

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            report_title  = st.text_input("Report Title",  value="DataMate AI Dashboard",  key="html_title")
            report_author = st.text_input("Author / Team", value="DataMate Analyst",       key="html_author")
        with h_col2:
            show_kpis    = st.checkbox("Include KPI Cards",       value=True, key="html_kpis")
            show_charts  = st.checkbox("Include Pinned Charts",   value=True, key="html_charts_cb")

        if st.button("⚙️ Generate HTML Dashboard", key="btn_gen_html", use_container_width=True, type="primary"):
            with st.spinner("Building interactive HTML..."):
                try:
                    # 100% Custom HTML Builder jo exactly checkboxes ko follow karega
                    html_parts = [
                        f"<html><head><title>{report_title}</title>",
                        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
                        """
                        <style>
                            body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #f4f7f6; color: #333;}
                            .container {max-width: 1200px; margin: auto;}
                            .header {text-align: center; margin-bottom: 30px;}
                            .kpi-container {display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; margin-bottom: 30px;}
                            .kpi-card {background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; min-width: 150px; border-bottom: 4px solid #3498db;}
                            .kpi-val {font-size: 24px; font-weight: bold; color: #2c3e50; margin-top: 10px;}
                            .chart-card {background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;}
                            h2 {color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 10px;}
                        </style>
                        </head><body><div class='container'>
                        """
                    ]
                    
                    html_parts.append(f"<div class='header'><h1>📊 {report_title}</h1><p>Created by <b>{report_author}</b> on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p></div>")

                    # Checkbox logic for KPIs
                    if show_kpis:
                        html_parts.append("<h2>Data Overview</h2><div class='kpi-container'>")
                        kpis = [
                            ("Total Rows", f"{len(df):,}"),
                            ("Total Columns", str(len(df.columns))),
                            ("Missing Cells", f"{total_missing:,}")
                        ]
                        for label, val in kpis:
                            html_parts.append(f"<div class='kpi-card'><div>{label}</div><div class='kpi-val'>{val}</div></div>")
                        html_parts.append("</div>")

                    # Checkbox logic for Charts
                    if show_charts:
                        saved_charts = st.session_state.get('saved_charts', [])
                        html_parts.append("<h2>Pinned Visualizations</h2>")
                        if not saved_charts:
                            html_parts.append("<p>No charts were pinned in this session.</p>")
                        else:
                            for i, chart in enumerate(saved_charts):
                                fig = chart.get('fig')
                                if fig:
                                    # Convert plotly fig directly to HTML div
                                    fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
                                    title = chart.get('title', f'Chart {i+1}')
                                    desc = chart.get('description', '')
                                    html_parts.append(f"<div class='chart-card'><h3>{title}</h3>")
                                    if desc:
                                        html_parts.append(f"<p style='color:#7f8c8d; font-style:italic;'>{desc}</p>")
                                    html_parts.append(f"{fig_html}</div>")
                    
                    html_parts.append("</div></body></html>")
                    
                    # Finalize HTML
                    html_content = "".join(html_parts)
                    st.session_state['html_bytes'] = html_content.encode('utf-8')
                    st.success("✅ HTML dashboard ready!")
                        
                except Exception as e:
                    st.error(f"❌ Error generating HTML: {str(e)}")

        if st.session_state.get('html_bytes'):
            st.download_button(
                label="📥 Download HTML Dashboard",
                data=st.session_state['html_bytes'],
                file_name=f"DataMate_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True,
                key="dl_html"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – EXPORT DATASET
    # ══════════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown('<div class="section-header">📊 Export Cleaned Dataset</div>', unsafe_allow_html=True)

        d_col1, d_col2 = st.columns([1, 1.2])

        with d_col1:
            st.markdown("**Export Format**")
            fmt = st.radio("Choose format", ["CSV", "Excel (.xlsx)", "JSON", "Parquet"], key="data_export_fmt")
            max_rows = st.slider("Max rows to export (0 = all)", 0, min(100_000, len(df)), 0, key="data_export_rows")
            
            export_df = df if max_rows == 0 else df.head(max_rows)

            st.info(f"Exporting **{len(export_df):,}** rows × **{len(export_df.columns)}** columns")
            fname_base = f"cleaned_{st.session_state.get('selected_file_name', 'data').split('.')[0]}"

            try:
                if fmt == "CSV":
                    data_bytes = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", data=data_bytes,
                                       file_name=f"{fname_base}.csv", mime="text/csv",
                                       use_container_width=True, key="dl_csv")

                elif fmt == "Excel (.xlsx)":
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False)
                    st.download_button("📥 Download Excel", data=buf.getvalue(),
                                       file_name=f"{fname_base}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True, key="dl_xlsx")

                elif fmt == "JSON":
                    data_bytes = export_df.to_json(orient='records', indent=2).encode('utf-8')
                    st.download_button("📥 Download JSON", data=data_bytes,
                                       file_name=f"{fname_base}.json", mime="application/json",
                                       use_container_width=True, key="dl_json")

                elif fmt == "Parquet":
                    buf = io.BytesIO()
                    export_df.to_parquet(buf, index=False)
                    st.download_button("📥 Download Parquet", data=buf.getvalue(),
                                       file_name=f"{fname_base}.parquet",
                                       mime="application/octet-stream",
                                       use_container_width=True, key="dl_parquet")
            except ModuleNotFoundError as e:
                st.error(f"❌ Missing library for {fmt} export. Please install it: {e}")
            except Exception as e:
                st.error(f"❌ Export Failed: {e}")

        with d_col2:
            st.markdown("**Dataset Preview (first 10 rows)**")
            st.dataframe(export_df.head(10), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – TXT SUMMARY (FIXED FOR LARGE COLS)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_txt:
        st.markdown('<div class="section-header">📝 Plain-Text Summary Report</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="content-frame">
            <p style="font-size:14px; color:#555; margin:0;">
                Export a lightweight plain-text (.txt) report — great for logs,
                quick sharing, or attaching to emails.
            </p>
        </div>
        """, unsafe_allow_html=True)

        t_col1, t_col2 = st.columns([1, 1.5])

        with t_col1:
            include_dtypes = st.checkbox("Column Data Types",        value=True, key="txt_dtypes")
            include_desc   = st.checkbox("Descriptive Stats",        value=True, key="txt_desc")
            include_miss   = st.checkbox("Missing Values Summary",   value=True, key="txt_miss")
            include_top5   = st.checkbox("Top 5 Rows Preview",       value=True, key="txt_top5")
            include_corr   = st.checkbox("Numeric Correlation Table",value=False,key="txt_corr")

            if st.button("⚙️ Generate TXT Summary", key="btn_gen_txt", use_container_width=True, type="primary"):
                try:
                    sep = "=" * 60
                    
                    try:
                        mem_usage_txt = f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
                    except Exception:
                        mem_usage_txt = "N/A"
                        
                    lines = [
                        "DATAMATE AI — DATA SUMMARY REPORT",
                        sep,
                        f"File       : {st.session_state.get('selected_file_name', 'Unknown')}",
                        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Rows       : {len(df):,}",
                        f"Columns    : {len(df.columns)}",
                        f"Missing    : {total_missing:,}",
                        f"Duplicates : {df.duplicated().sum():,}",
                        f"Memory     : {mem_usage_txt}",
                        sep,
                    ]
                    
                    if include_dtypes:
                        lines += ["\nCOLUMN DATA TYPES", "-"*40]
                        for col in df.columns:
                            lines.append(f"  {str(col)[:28]:<30} {str(df[col].dtype)}")
                            
                    if include_desc:
                        lines += ["\nDESCRIPTIVE STATISTICS (Numeric Only)", "-"*40]
                        num_df = df.select_dtypes(include='number')
                        if not num_df.empty:
                            # Using Transpose (.T) for clean TXT output
                            lines.append(num_df.describe().T.round(3).to_string())
                        else:
                            lines.append("No numeric columns available.")
                        
                    if include_miss:
                        lines += ["\nMISSING VALUES PER COLUMN", "-"*40]
                        for col in df.columns:
                            n = df[col].isnull().sum()
                            pct = (n / len(df) * 100) if len(df) > 0 else 0
                            lines.append(f"  {str(col)[:28]:<30} {n:>6}  ({pct:.1f}%)")
                            
                    if include_top5:
                        lines += ["\nTOP 5 ROWS", "-"*40]
                        lines.append(df.head(5).to_string(index=False))
                        
                    if include_corr:
                        num_df = df.select_dtypes(include='number')
                        if not num_df.empty:
                            lines += ["\nCORRELATION MATRIX", "-"*40]
                            lines.append(num_df.corr().round(3).to_string())
                        else:
                            lines += ["\nCORRELATION MATRIX: No numeric columns found."]

                    txt_content = "\n".join(lines)
                    st.session_state['txt_bytes'] = txt_content.encode('utf-8')
                    st.success(f"✅ TXT summary ready — {len(txt_content):,} characters.")
                
                except Exception as e:
                    st.error(f"❌ Error generating TXT: {e}")

        with t_col2:
            if st.session_state.get('txt_bytes'):
                preview_text = st.session_state['txt_bytes'].decode('utf-8')[:2000]
                if len(st.session_state['txt_bytes']) > 2000:
                    preview_text += "\n\n... (truncated for preview) ..."
                st.text_area("Preview", value=preview_text, height=340, key="txt_preview")

        if st.session_state.get('txt_bytes'):
            st.download_button(
                label="📥 Download TXT Summary",
                data=st.session_state['txt_bytes'],
                file_name=f"DataMate_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_txt"
            )

# =========================================================
# === HISTORY PAGE ========================================
# =========================================================

def history_page():
    
    """Session Activity History — DataMate AI"""
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -------------------------------------------------
    
    from datetime import datetime

    st.markdown('<div class="main-header">🕒 Session History</div>', unsafe_allow_html=True)

    # ── Top Stats Bar ─────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    total_files    = len(st.session_state.get('all_datasets', {}))
    total_charts   = len(st.session_state.get('saved_charts', []))
    ml_trained     = 1 if 'trained_model' in st.session_state else 0
    active_file    = st.session_state.get('selected_file_name', 'None') or 'None'

    for col_ui, (label, val, color) in zip(
        [h1, h2, h3, h4],
        [
            ("Datasets Loaded",  str(total_files),  "#3498db"),
            ("Pinned Charts",    str(total_charts),  "#3498db"),
            ("Models Trained",   str(ml_trained), "#3498db"),
            ("Active Dataset",   active_file[:16],   "#3498db")
        ]
    ):
        with col_ui:
            st.markdown(f"""
            <div class="metric-card" style="border-left:4px solid {color}; text-align:center;">
                <p style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
                          letter-spacing:1px;margin:0;">{label}</p>
                <p style="font-size:20px;font-weight:800;color:{color};
                          margin:6px 0 0 0;font-family:Sora,sans-serif;">{val}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabbed History Sections ───────────────────────────────────────────────
    tab_files, tab_charts, tab_ml, tab_clear = st.tabs([
        "📁 Uploaded Datasets", "📌 Pinned Charts", "🤖 ML Activity", "🗑️ Manage Session"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – UPLOADED DATASETS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_files:
        st.markdown('<div class="section-header">📁 Uploaded Datasets This Session</div>',
                    unsafe_allow_html=True)

        datasets = st.session_state.get('all_datasets', {})

        if not datasets:
            st.info("📭 No datasets uploaded yet. Go to **Data Upload** to add files.")
        else:
            # Column headers
            hc1, hc2, hc3, hc4, hc5 = st.columns([3, 2, 2, 2, 1])
            for col_ui, lbl in zip([hc1, hc2, hc3, hc4, hc5],
                                   ["File Name", "Rows", "Columns", "Missing Cells", "Action"]):
                with col_ui:
                    st.markdown(f"<p style='font-weight:700;color:#2c3e50;font-size:13px;"
                                f"border-bottom:2px solid #3498db;padding-bottom:4px;"
                                f"margin:0;'>{lbl}</p>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

            for name in list(datasets.keys()):
                data = datasets[name]
                is_active = (name == st.session_state.get('selected_file_name'))
                border_color = "#3498db" if is_active else "#dee2e6"
                bg_color     = "#eaf4fb" if is_active else "#ffffff"

                rc1, rc2, rc3, rc4, rc5 = st.columns([3, 2, 2, 2, 1])

                with rc1:
                    st.markdown(f"""
                    <div style="background:{bg_color};border-left:4px solid {border_color};
                         padding:10px 14px;border-radius:8px;">
                        <p style="margin:0;font-weight:600;color:#1a2535;font-size:13px;">
                            📄 {name}
                        </p>
                        {"<span style='font-size:11px;color:#3498db;font-weight:600;'>"
                         "✅ Active</span>" if is_active else ""}
                    </div>
                    """, unsafe_allow_html=True)

                with rc2:
                    st.markdown(f"""
                    <div style="padding:10px 0;">
                        <p style="margin:0;font-size:14px;font-weight:700;color:#2c3e50;">
                            {len(data):,}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with rc3:
                    st.markdown(f"""
                    <div style="padding:10px 0;">
                        <p style="margin:0;font-size:14px;font-weight:700;color:#2c3e50;">
                            {len(data.columns)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with rc4:
                    miss = data.isnull().sum().sum()
                    miss_color = "#e74c3c" if miss > 0 else "#2ecc71"
                    st.markdown(f"""
                    <div style="padding:10px 0;">
                        <p style="margin:0;font-size:14px;font-weight:700;color:{miss_color};">
                            {miss:,}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with rc5:
                    if st.button("🗑️", key=f"del_hist_{name}",
                                 help=f"Remove '{name}' from session"):
                        del st.session_state.all_datasets[name]
                        if st.session_state.get('selected_file_name') == name:
                            st.session_state.selected_file_name = None
                            st.session_state.df = None
                        st.rerun()

                st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

            # Switch Active Dataset
            st.markdown("---")
            st.markdown("**Switch Active Dataset**")
            file_opts = list(datasets.keys())
            current_idx = file_opts.index(st.session_state.get('selected_file_name', file_opts[0])) \
                          if st.session_state.get('selected_file_name') in file_opts else 0
            switch_to = st.selectbox("Choose dataset to activate",
                                     file_opts, index=current_idx, key="hist_switch_ds")
            if st.button("✅ Set as Active Dataset", key="hist_activate_btn"):
                st.session_state.selected_file_name = switch_to
                st.session_state.df = datasets[switch_to]
                st.success(f"✅ Active dataset switched to: **{switch_to}**")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – PINNED CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_charts:
        st.markdown('<div class="section-header">📌 Pinned Visualizations</div>',
                    unsafe_allow_html=True)

        saved_charts = st.session_state.get('saved_charts', [])

        if not saved_charts:
            st.info("📭 No charts pinned yet. Visit **Visualization** and click 'Pin to Dashboard'.")
        else:
            st.caption(f"{len(saved_charts)} chart(s) currently pinned to your dashboard.")

            for i, chart in enumerate(saved_charts):
                cc1, cc2, cc3 = st.columns([3, 1.5, 0.6])
                with cc1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left:4px solid #9b59b6; padding:12px 16px;">
                        <p style="margin:0; font-weight:700; color:#1a2535; font-size:14px;">
                            📊 {chart.get('title', f'Chart {i+1}')}
                        </p>
                        <p style="margin:4px 0 0 0; font-size:12px; color:#7f8c8d;">
                            {chart.get('description', 'No description') or 'No analysis note'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with cc2:
                    st.markdown(f"""
                    <div style="padding: 8px 0;">
                        <span style="background:#eaf4fb; color:#2980b9; padding:4px 10px;
                              border-radius:20px; font-size:12px; font-weight:600;">
                            Chart #{i+1}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                with cc3:
                    if st.button("🗑️", key=f"del_chart_{i}", help="Remove this chart"):
                        st.session_state.saved_charts.pop(i)
                        st.rerun()

                st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

            st.markdown("---")
            if st.button("🗑️ Clear All Pinned Charts", key="clear_all_charts"):
                st.session_state.saved_charts = []
                st.success("✅ All pinned charts cleared.")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – ML ACTIVITY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ml:
        st.markdown('<div class="section-header">🤖 Machine Learning Activity</div>',
                    unsafe_allow_html=True)

        if 'ml_results' not in st.session_state:
            st.info("📭 No ML model has been trained yet. Visit **ML Algorithm** to train a model.")
        else:
            res = st.session_state.ml_results

            ml1, ml2, ml3, ml4 = st.columns(4)
            ml_kpis = [
                ("Algorithm",    res.get('model_name', 'N/A'),   "#3498db"),
                ("Task Type",    res.get('task', 'N/A'),         "#2ecc71"),
                ("Features Used",str(len(res.get('features', []))), "#9b59b6"),
                ("Train Time",   f"{res.get('train_time', 0):.2f}s", "#e67e22"),
            ]
            for col_ui, (label, val, color) in zip([ml1, ml2, ml3, ml4], ml_kpis):
                with col_ui:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left:4px solid {color}; text-align:center;">
                        <p style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
                                  letter-spacing:1px;margin:0;">{label}</p>
                        <p style="font-size:18px;font-weight:800;color:{color};
                                  margin:6px 0 0 0;font-family:Sora,sans-serif;">{val}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Feature Columns Used**")
            features = res.get('features', [])
            if features:
                feat_html = " ".join([
                    f"<span style='background:#eaf4fb;color:#2980b9;padding:4px 10px;"
                    f"border-radius:14px;font-size:12px;font-weight:600;margin:3px;'>{f}</span>"
                    for f in features
                ])
                st.markdown(feat_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"**Target Column:** `{res.get('target', 'N/A')}`")

            # Download model if available
            if 'trained_model' in st.session_state:
                import io as _io
                import joblib
                model_buf = _io.BytesIO()
                joblib.dump(st.session_state.trained_model, model_buf)
                st.download_button(
                    "📥 Re-download Trained Model (.pkl)",
                    data=model_buf.getvalue(),
                    file_name=f"DataMate_model_{res.get('model_name','model')}.pkl",
                    mime="application/octet-stream",
                    key="hist_dl_model"
                )
# ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – MANAGE SESSION / CLEAR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_clear:
        st.markdown('<div class="section-header">🗑️ Manage & Clear Session</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="content-frame" style="border-left:4px solid #e74c3c; background:#fff5f5;">
            <h4 style="color:#c0392b; font-family:Sora,sans-serif; margin-top:0;">
                ⚠️ Danger Zone
            </h4>
            <p style="color:#555; font-size:14px; margin:0;">
                The actions below will permanently remove data from your current session.
                This cannot be undone.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ─── ONE MASTER DIALOG FOR ALL ACTIONS ────────────────────────────────────
        @st.dialog("⚠️ Confirm Action")
        def confirm_action(action_type, warning_msg):
            st.warning(warning_msg)
            c1, c2 = st.columns(2)
            
            if c1.button("✔️ Yes, Proceed", type="primary", use_container_width=True):
                if action_type == "datasets":
                    st.session_state.all_datasets       = {}
                    st.session_state.selected_file_name = None
                    st.session_state.df                 = None
                    
                elif action_type == "charts":
                    st.session_state.saved_charts = []
                    
                elif action_type == "full_reset":
                    for key in ['all_datasets', 'selected_file_name', 'df',
                                 'saved_charts', 'kpi_configs', 'trained_model',
                                 'ml_results', 'snapshots', 'pdf_bytes',
                                 'html_bytes', 'txt_bytes']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.current_page = 'Welcome'
                
                st.rerun()
                
            if c2.button("❌ Cancel", use_container_width=True):
                st.rerun()
        # ──────────────────────────────────────────────────────────────────────────

        clear1, clear2, clear3 = st.columns(3)

        with clear1:
            st.markdown("""
            <div class="metric-card" style="border-left:4px solid #e67e22; text-align:center;">
                <p style="font-weight:700; color:#1a2535;">Clear All Datasets</p>
                <p style="font-size:12px;color:#7f8c8d;">
                    Removes all uploaded files from the session.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Clear Datasets", key="clear_datasets_btn", use_container_width=True):
                # Function ko action_type aur message pass kar diya
                confirm_action("datasets", "Are you sure you want to clear all uploaded datasets? This cannot be undone.")

        with clear2:
            st.markdown("""
            <div class="metric-card" style="border-left:4px solid #9b59b6; text-align:center;">
                <p style="font-weight:700; color:#1a2535;">Clear Pinned Charts</p>
                <p style="font-size:12px;color:#7f8c8d;">
                    Removes all pinned visualizations from the dashboard.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Clear Charts", key="clear_charts_btn", use_container_width=True):
                # Function ko action_type aur message pass kar diya
                confirm_action("charts", "Are you sure you want to clear all pinned charts? This cannot be undone.")

        with clear3:
            st.markdown("""
            <div class="metric-card" style="border-left:4px solid #e74c3c; text-align:center;">
                <p style="font-weight:700; color:#1a2535;">Full Session Reset</p>
                <p style="font-size:12px;color:#7f8c8d;">
                    Wipes everything — datasets, charts, models, KPIs.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔴 Full Reset", key="full_reset_btn", use_container_width=True):
                # Function ko action_type aur message pass kar diya
                confirm_action("full_reset", "This will wipe EVERYTHING — datasets, charts, models, and KPIs. Proceed?")

# =========================================================
# === ABOUT PAGE ==========================================
# =========================================================
def about_page():
    """About DataMate AI — Co-Founders & Feature Overview"""
    
        # --- CSS Injection to remove the top whitespace ---
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.0rem !important; /* Adjust this value to push the content up or down */
            }
        </style>
    """, unsafe_allow_html=True)
    # -----------------------------
    
    
    
    st.markdown('<div class="main-header">ℹ️ About – DataMate AI</div>', unsafe_allow_html=True)

    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="content-frame" style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
         color:white; border:none; padding:28px 32px;">
        <h2 style="font-family:Sora,sans-serif; margin-top:0; color:white; font-size:24px;">
            What is DataMate AI?
        </h2>
        <p style="color:rgba(255,255,255,0.85); font-size:15px; line-height:1.8; margin:0;">
            DataMate AI is a smart, end-to-end data analysis platform built with
            Python and Streamlit. From uploading raw files to training machine
            learning models and exporting polished reports everything lives in
            one unified, beautifully designed interface. No code required.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Feature Highlights ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚀 Platform Features</div>', unsafe_allow_html=True)

    features = [
        ("📁", "Data Upload",
         "CSV, Excel, JSON, TXT, XML, Parquet automatic encoding detection, "
         "multi-file support, and dataset management."),
        ("🔄", "Data Preprocessing",
         "12-tab preprocessing center: missing values, duplicates, outlier treatment, "
         "encoding, scaling, PCA, feature engineering, and ML readiness checks."),
        ("📈", "Visualization",
         "30+ interactive Plotly charts across 8 categories Univariate, Bivariate, "
         "Multivariate, Time Series, Statistical, Geographic, Custom, and Pivot."),
        ("📊", "Dashboard",
         "Custom KPI cards with color pickers, pinned visualizations with analysis notes, "
         "and one-click HTML export."),
        ("🤖", "Machine Learning",
         "Automated sklearn pipelines for regression and classification. "
         "10+ algorithms, hyperparameter tuning, cross-validation, and batch predictions."),
        ("💾", "Export Center",
         "Multi-page PDF reports, interactive HTML dashboards, cleaned datasets in 4 formats, "
         "and plain-text summaries."),
        ("🕒", "Session History",
         "Full activity log: manage datasets, review pinned charts, track ML models, "
         "and perform targeted or full session resets."),
        ("🔮", "Predictions",
         "Manual single-row prediction form or bulk CSV upload for batch inference "
         "with label encoding inversion support."),
    ]

    rows = [features[i:i+4] for i in range(0, len(features), 4)]
    for row in rows:
        cols = st.columns(4)
        for col_ui, (icon, title, desc) in zip(cols, row):
            with col_ui:
                # CHANGED: Replaced min-height with fixed height and added flexbox styling
                st.markdown(f"""
                <div class="feature-card" style="
                    height: 250px; 
                    display: flex; 
                    flex-direction: column; 
                    box-sizing: border-box;
                ">
                    <span class="feature-icon" style="margin-bottom: 8px;">{icon}</span>
                    <div class="feature-title" style="margin-bottom: 8px;">{title}</div>
                    <div class="feature-desc" style="flex-grow: 1; overflow: hidden;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)

    tech_cols = st.columns(5)
    techs = [
        ("🐍", "Python 3.11+",     "#3498db"),
        ("⚡", "Streamlit",        "#3498db"),
        ("📊", "Plotly",           "#3498db"),
        ("🧠", "Scikit-learn",     "#3498db"),
        ("📄", "FPDF2",            "#3498db"),
    ]
    for col_ui, (icon, name, color) in zip(tech_cols, techs):
        with col_ui:
            st.markdown(f"""
            <div class="metric-card" style="border-left:4px solid {color}; text-align:center;
                 padding:14px 10px;">
                <p style="font-size:26px; margin:0;">{icon}</p>
                <p style="font-size:13px; font-weight:700; color:#1a2535; margin:6px 0 0 0;">
                    {name}
                </p>
            </div>
            """, unsafe_allow_html=True)


    st.markdown("---")
                    # ── Co-Founders ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">👥 Co-Founders</div>', unsafe_allow_html=True)

    f_col1, f_col2 = st.columns(2)

    founder_data = [
        ("Shahbaz Junaid", "shahbazjunaid55@gmail.com",  
         "https://ui-avatars.com/api/?name=Shahbaz+Junaid&background=3498db&color=fff&size=150",
         "BS Data Analytics Student at GCUF"),
         
        ("Rizwan Ullah", "sulima9876@gmail.com",       
         "https://ui-avatars.com/api/?name=Rizwan+Ullah&background=2ecc71&color=fff&size=150",
         "BS Data Analytics Student at GCUF")
    ]

    for col_ui, (name, email, img_url, desc) in zip([f_col1, f_col2], founder_data):
        with col_ui:
            # Streamlit ka apna native container border ke sath (requires Streamlit 1.30+)
            with st.container(border=True): 
                # Image aur text ko side-by-side karne ke liye andar 2 aur columns
                img_col, text_col = st.columns([1, 2.5]) 
                
                with img_col:
                    st.image(img_url, width=100)
                
                with text_col:
                    st.subheader(name)
                    st.caption(desc) # Caption chote aur grey text ke liye hota hai
                    st.markdown(f"[📧 {email}](mailto:{email})")
                    
# ── Version & Support ─────────────────────────────────────────────────────
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        st.markdown("""
        <div class="metric-card" style="
            height: 140px;
            border-left:4px solid #2ecc71; text-align:center; padding: 20px;
            display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;
            background: #fafafa; border-radius: 0 8px 8px 0;
        ">
            <p style="font-size:10px;color:black;text-transform:uppercase;
                      letter-spacing:1.2px;margin:0; font-weight:600;">Current Version</p>
            <p style="font-size:22px;font-weight:800;color:black;
                      margin:4px 0 2px 0;font-family:Sora,sans-serif;">v1.0</p>
            <p style="font-size:10px;color:black;margin:0;">
                Production Release
            </p>
        </div>
        """, unsafe_allow_html=True)
    with v_col2:
        st.markdown("""
        <div class="content-frame" style="
            height: 140px;
            border-left:4px solid #3498db; padding: 20px;
            display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;
            background: #fafafa; border-radius: 0 8px 8px 0;
        ">
            <h4 style="font-family:Sora,sans-serif; margin-top:0; margin-bottom:6px; color:#1a2535; font-size:15px;">
                💬 Support & Feedback
            </h4>
            <p style="color:#555; font-size:12px; margin:0; line-height:1.6;">
                For bugs, feature requests, or general feedback, reach out via email. 
                We typically respond within 24–48 hours.
            </p>
        </div>
        """, unsafe_allow_html=True)
# =========================================================
# === SIDEBAR NAVIGATION ==================================
# =========================================================

with st.sidebar:
    st.markdown("""
    <div class='sidebar-header-card'>
        <h1 style='color: white; margin: 0; font-size: 22px; font-family: Sora, sans-serif; letter-spacing: 0.5px;'>
             DataMate AI
        </h1>
        <p style='color: rgba(255,255,255,0.7); margin: 6px 0 0 0; font-size: 12px;'></p>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        'Dashboard':          '📊',
        'Data Upload':        '📁',
        'Data Preprocessing': '🛠',
        'Visualization':      '📈',
        'ML Algorithm':       '🤖',
        'Export':             '💾',
        'History':            '🕒',
        'About':              'ℹ️'
    }

    for page_name, icon in pages.items():
        target = 'Welcome' if page_name == 'Home' else page_name
        if st.button(f"{icon}  {page_name}", key=page_name, use_container_width=True):
            st.session_state.current_page = target
# =========================================================
# === PAGE ROUTING ========================================
# =========================================================

def route_pages():
    if st.session_state.current_page == 'Welcome':
        welcome_page()
    elif st.session_state.current_page == 'Dashboard':
        dashboard_page()
    elif st.session_state.current_page == 'Data Upload':
        data_upload_page()
    elif st.session_state.current_page == 'Data Preprocessing':
        data_preprocessing_page()  
    # elif Visualization page is now the main page, so we route to it by default     
    elif st.session_state.current_page == 'Visualization':
        visualization_page()
    elif st.session_state.current_page == 'ML Algorithm':
        ml_algorithm_page()
    elif st.session_state.current_page == 'Export':
        export_page()
    elif st.session_state.current_page == 'History':
        history_page()        
    elif st.session_state.current_page == 'About':
        about_page()

st.markdown("<div style='text-align: center; color: #7f8c8d; padding: 20px; font-size: 12px;'></div>", unsafe_allow_html=True)

route_pages()