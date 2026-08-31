"""
╔══════════════════════════════════════════════════════════════════════════╗
║        DataMate AI — Streamlit Cache & Performance Layer  v5             ║
║        🚀 DISK-PERSISTENT CACHE — Localhost Ultra-Fast Mode              ║
║   Drop this file next to App.py and import at the TOP of App.py          ║
╚══════════════════════════════════════════════════════════════════════════╝

HOW DISK CACHE WORKS
─────────────────────
• Streamlit ka persist="disk" parameter cache ko RAM mein nahi,
  hard disk pe save karta hai → folder:  .streamlit/cache/

• Faida:
    ✅ App restart karo  → cache rehta hai (no re-compute)
    ✅ Page reload karo  → result instant milta hai
    ✅ Same data dobara upload karo → load nahi hoga, disk se ata hai
    ✅ Model train karo  → wahi model cache se milta hai (re-train nahi)
    ✅ PDF/HTML export   → bytes cached hain, instantly download hota hai

• Kab invalidate hota hai:
    ↺ DataFrame change ho (get_df_hash() different ho)
    ↺ Function arguments change hon
    ↺ TTL expire ho jaye (heavy ops = 6 h, charts = 2 h)

CACHE STRATEGY
──────────────
  persist="disk"  → ydata profiling, ML training, export bytes,
                     HTML dashboard, chart PNGs, image loader
                     (ye sab zyada time lete hain, disk pe save karo)

  persist=False   → KPI values, simple stats, preprocessing ops,
                     visualization helpers, quick stats
                     (ye fast hain, RAM cache kaafi hai)

SECTIONS
─────────
  § 0  CSS Styling Cache       (no TTL — permanent RAM cache)
  § 1  File Loading            (CSV / Excel / JSON / XML / Parquet)
  § 2  Statistics & Profiling
  § 3  ML Pipeline Training    (cached_train_pipeline + run_ml_training)
  § 4  ML Result Charts        (confusion matrix, actual-vs-predicted, importance)
  § 5  ML Metrics              (classification report, regression metrics)
  § 6  Batch Predictions
  § 7  Cross-Validation
  § 8  ydata-Profiling Report  (💾 disk persisted — very expensive)
  § 9  HTML Dashboard Report   (💾 disk persisted)
  § 10 Export Bytes            (💾 disk persisted — CSV/Excel/JSON/Parquet/TXT)
  § 11 Chart PNG Bytes         (💾 disk persisted — kaleido)
  § 12 Image / Logo Loader     (💾 disk persisted)
  § 13 Plotly Chart Helpers
  § 14 KPI Value Helper
  § 15 Utility — get_df_hash()
  § 16 Data Preprocessing Cache  ← NEW: 12-tab preprocessing center
       drop_duplicates, fill_missing, drop_missing_cols,
       detect_outliers_iqr/zscore, clip_outliers, label_encode,
       onehot_encode, minmax_scale, standard_scale, pca_transform,
       ml_readiness
  § 17 Extended Visualization Cache  ← NEW: 30+ chart types
       scatter, box, violin, bar, line, pie, area,
       heatmap_pivot, pairplot, 3d_scatter
  § 18 Quick Stats / Data Summary  ← NEW: instant metadata
       quick_stats, numeric_cols, categorical_cols,
       datetime_cols, dtype_summary

HOW TO USE
──────────
1. Place cache_config.py next to App.py
2. Top of App.py:  from cache_config import *
3. Run:            streamlit run App.py
4. Cache folder:   .streamlit/cache/   (auto-created)
5. Clear cache:    streamlit cache clear   (terminal mein)
                   ya sirf .streamlit/cache/ folder delete karo

QUICK USAGE PATTERN
───────────────────
    h  = get_df_hash(st.session_state.df)    # always first
    df = st.session_state.df

    # Preprocessing
    clean  = cached_drop_duplicates(h, df)
    filled = cached_fill_missing(h, df, strategy="median")
    audit  = cached_ml_readiness(h, df)

    # Visualization
    fig = cached_scatter(h, df, x="age", y="salary")
    fig = cached_bar_chart(h, df, col="city", top_n=15)

    # Quick stats (replaces scattered df.shape / df.dtypes calls)
    qs = cached_quick_stats(h, df)
    st.write(f"Rows: {qs['rows']}  |  Missing: {qs['missing_pct']}%")
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib


# ══════════════════════════════════════════════════════════════════════
# § 1  FILE LOADING
#      persist="disk"  — same file dobara upload karo, instantly load
#      TTL = 6 hours
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=21600, persist="disk", show_spinner="📂 Loading file…")
def cached_load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """CSV / TXT — auto encoding detection. 💾 Disk cached."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes), encoding=enc,
                sep=None, engine="python", on_bad_lines="skip",
            )
        except Exception:
            continue
    raise ValueError(f"Cannot decode {filename} with any known encoding.")


@st.cache_data(ttl=21600, persist="disk", show_spinner="📂 Loading file…")
def cached_load_excel(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Excel (.xlsx / .xls). 💾 Disk cached."""
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(ttl=21600, persist="disk", show_spinner="📂 Loading file…")
def cached_load_json(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """JSON file loader. 💾 Disk cached."""
    try:
        return pd.read_json(io.BytesIO(file_bytes))
    except ValueError:
        return pd.read_json(io.BytesIO(file_bytes), lines=True)


@st.cache_data(ttl=21600, persist="disk", show_spinner="📂 Loading file…")
def cached_load_xml(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """XML file loader. 💾 Disk cached."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_xml(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot parse {filename} as XML.")


@st.cache_data(ttl=21600, persist="disk", show_spinner="📂 Loading file…")
def cached_load_parquet(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parquet loader. 💾 Disk cached."""
    return pd.read_parquet(io.BytesIO(file_bytes))


def load_any_file(uploaded_file) -> "pd.DataFrame | None":
    """
    Universal loader — replaces load_data_with_encoding().
    Same file dobara upload → disk cache se instant load.
    """
    ext = uploaded_file.name.split(".")[-1].lower()
    raw = uploaded_file.read()
    loaders = {
        "csv":     cached_load_csv,
        "txt":     cached_load_csv,
        "xlsx":    cached_load_excel,
        "xls":     cached_load_excel,
        "json":    cached_load_json,
        "xml":     cached_load_xml,
        "parquet": cached_load_parquet,
    }
    loader = loaders.get(ext)
    if loader is None:
        st.error(f"Unsupported file type: .{ext}")
        return None
    try:
        return loader(raw, uploaded_file.name)
    except Exception as e:
        st.error(f"❌ Failed to load {uploaded_file.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# § 12  IMAGE / LOGO LOADER
#       persist="disk" — logo ek baar read, hamesha disk se serve
#       TTL = 24 hours
#
#  Usage in welcome_page():
#      img_bytes = cached_load_image("datamate_logo_50.png")
#      if img_bytes:
#          st.image(io.BytesIO(img_bytes), width=350)
#      else:
#          st.markdown('<div style="font-size:130px;">🤖</div>',
#                      unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, persist="disk", show_spinner=False)
def cached_load_image(image_path: str) -> "bytes | None":
    """
    Local image file → raw bytes. 💾 Disk cached (24 h).
    Supports PNG / JPG / JPEG / GIF / WEBP / SVG.
    Returns None if file not found.
    """
    import os
    try:
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as f:
            return f.read()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# § 2  STATISTICS & PROFILING
#      RAM cache (fast enough, no disk needed)
#      TTL = 2 hours
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_describe(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """df.describe() — cached in RAM."""
    return df.describe(include="all").round(3)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_column_profile(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Full column profile table (Overview tab) — RAM cached."""
    from scipy.stats import skew as _skew
    rows = []
    for col in df.columns:
        row = {
            "Column":   col,
            "Type":     str(df[col].dtype),
            "Non-Null": int(df[col].count()),
            "Null":     int(df[col].isnull().sum()),
            "Null %":   f"{df[col].isnull().mean() * 100:.1f}%",
            "Unique":   int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            row["Min"]  = f"{s.min():.2f}"
            row["Max"]  = f"{s.max():.2f}"
            row["Mean"] = f"{s.mean():.2f}"
            row["Std"]  = f"{s.std():.2f}"
            row["Skew"] = f"{_skew(s):.2f}" if len(s) > 2 else "—"
        else:
            row["Min"] = row["Max"] = row["Mean"] = row["Std"] = row["Skew"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_missing_summary(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Missing-values summary — RAM cached."""
    return (
        pd.DataFrame({
            "Column":    df.columns,
            "Missing":   df.isnull().sum().values,
            "Missing %": (df.isnull().sum().values / len(df) * 100).round(2),
            "Dtype":     df.dtypes.astype(str).values,
        })
        .sort_values("Missing", ascending=False)
        .query("Missing > 0")
    )


@st.cache_data(ttl=7200, show_spinner=False)
def cached_correlation(df_hash: str, df: pd.DataFrame,
                       method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix — RAM cached."""
    num = df.select_dtypes(include=["int64", "float64"])
    return num.corr(method=method).round(4)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_value_counts(df_hash: str, series_name: str,
                        df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Value counts — RAM cached."""
    vc = df[series_name].value_counts().head(top_n).reset_index()
    vc.columns = [series_name, "Count"]
    return vc


# ══════════════════════════════════════════════════════════════════════
# § 14  KPI VALUE HELPER — RAM cache (instant, no disk needed)
#
#  Usage in dashboard_page():
#      h   = get_df_hash(df)
#      val = cached_kpi_value(h, df, cfg["column"], cfg["agg"])
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_kpi_value(df_hash: str, df: pd.DataFrame,
                     column: str, agg: str) -> str:
    """KPI card value — RAM cached."""
    if column == "__rows__":       return f"{len(df):,}"
    if column == "__columns__":    return f"{len(df.columns):,}"
    if column == "__numeric__":
        return f"{len(df.select_dtypes(include=['int64','float64']).columns):,}"
    if column == "__missing__":    return f"{int(df.isnull().sum().sum()):,}"
    if column == "__duplicates__": return f"{int(df.duplicated().sum()):,}"
    if column not in df.columns:   return "N/A"
    try:
        s = pd.to_numeric(df[column], errors="coerce")
        if agg == "Sum":    return f"{s.sum():,.2f}"
        if agg == "Count":  return f"{s.count():,}"
        if agg == "Mean":   return f"{s.mean():,.2f}"
        if agg == "Median": return f"{s.median():,.2f}"
        if agg == "Max":    return f"{s.max():,.2f}"
        if agg == "Min":    return f"{s.min():,.2f}"
        if agg == "Std":    return f"{s.std():,.2f}"
    except Exception:
        pass
    return str(df[column].count())


# ══════════════════════════════════════════════════════════════════════
# § 3  ML PIPELINE — cache_resource (in-memory, no TTL)
#
#  ⚠️  NOTE: cache_resource disk persistence is not supported by
#  Streamlit yet — but cache_resource already survives page reloads
#  (model stays in RAM as long as the server is running).
#  Pipeline step names EXACTLY match ML_Section_2.py.
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🤖 Training model…")
def cached_train_pipeline(
    model_name:      str,
    task_type:       str,
    target_col:      str,
    feature_cols:    tuple,
    test_size:       float,
    random_state:    int,
    df_hash:         str,
    df_json:         str,
    stratify_target: bool = False,
    **model_kwargs,
) -> dict:
    """
    Fully cached sklearn Pipeline training.
    Model stays alive in RAM — no re-train on page reload.

    Returns: pipeline, X_test, y_test, y_pred, train_time,
             label_encoder, task, model_name, features, target
    """
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, LabelEncoder,
    )
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    )
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
        AdaBoostClassifier,
    )
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    df       = pd.read_json(io.StringIO(df_json))
    features = list(feature_cols)
    X = df[features]
    y = df[target_col]

    label_encoder = None
    if task_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    stratify_col = y if (stratify_target and task_type == "Classification") else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    num_feats = X.select_dtypes(include=np.number).columns.tolist()
    cat_feats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Step names match ML_Section_2.py exactly ──────────────────────────
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_feats),
        ("cat", cat_transformer, cat_feats),
    ])

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=model_kwargs.get("alpha", 1.0)),
        "Lasso":             Lasso(alpha=model_kwargs.get("alpha", 1.0)),
        "ElasticNet":        ElasticNet(
                                 alpha=model_kwargs.get("alpha", 1.0),
                                 l1_ratio=model_kwargs.get("l1_ratio", 0.5)),
        "Decision Tree":     DecisionTreeRegressor(
                                 max_depth=model_kwargs.get("max_depth")),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=model_kwargs.get("n_estimators", 100),
                                 max_depth=model_kwargs.get("max_depth")),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=model_kwargs.get("n_estimators", 100),
                                 max_depth=model_kwargs.get("max_depth")),
        "SVR":               SVR(C=model_kwargs.get("C", 1.0),
                                 kernel=model_kwargs.get("kernel", "rbf")),
        "KNN":               KNeighborsRegressor(
                                 n_neighbors=model_kwargs.get("n_neighbors", 5)),
    }
    clf_models = {
        "Logistic Regression": LogisticRegression(
                                   C=model_kwargs.get("C", 1.0), max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(
                                   max_depth=model_kwargs.get("max_depth")),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=model_kwargs.get("n_estimators", 100),
                                   max_depth=model_kwargs.get("max_depth")),
        "Gradient Boosting":   GradientBoostingClassifier(
                                   n_estimators=model_kwargs.get("n_estimators", 100),
                                   max_depth=model_kwargs.get("max_depth")),
        "SVC":                 SVC(C=model_kwargs.get("C", 1.0),
                                   kernel=model_kwargs.get("kernel", "rbf"),
                                   probability=True),
        "KNN":                 KNeighborsClassifier(
                                   n_neighbors=model_kwargs.get("n_neighbors", 5)),
        "Naive Bayes":         GaussianNB(),
        "AdaBoost":            AdaBoostClassifier(
                                   n_estimators=model_kwargs.get("n_estimators", 100)),
    }

    try:
        from xgboost import XGBRegressor, XGBClassifier
        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=model_kwargs.get("n_estimators", 100),
            max_depth=model_kwargs.get("max_depth", 5))
        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=model_kwargs.get("n_estimators", 100),
            max_depth=model_kwargs.get("max_depth", 5))
    except ImportError:
        pass

    model_map = reg_models if task_type == "Regression" else clf_models
    estimator = model_map.get(model_name, LinearRegression())
    pipeline  = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    return {
        "pipeline":      pipeline,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_pred":        pipeline.predict(X_test),
        "train_time":    elapsed,
        "label_encoder": label_encoder,
        "task":          task_type,
        "model_name":    model_name,
        "features":      features,
        "target":        target_col,
    }


def run_ml_training(
    df:               pd.DataFrame,
    selected_features: list,
    target_col:       str,
    task_type:        str,
    model_choice:     str,
    test_size:        float = 0.2,
    random_state:     int   = 42,
    stratify_target:  bool  = False,
    **model_kwargs,
) -> dict:
    """
    Train + auto-populate st.session_state in one call.
    Same settings = instant result from cache (no re-train).

    Usage in ml_algorithm_page():
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_choice}…"):
                result = run_ml_training(
                    df=df, selected_features=selected_features,
                    target_col=target_col, task_type=task_type,
                    model_choice=model_choice, test_size=test_size,
                    random_state=int(random_state),
                    stratify_target=stratify_data is not None,
                    **params)
                st.success(f"✅ {model_choice} trained in {result['train_time']:.2f}s")
    """
    h = get_df_hash(df)
    result = cached_train_pipeline(
        model_name=model_choice, task_type=task_type, target_col=target_col,
        feature_cols=tuple(selected_features), test_size=test_size,
        random_state=random_state, df_hash=h, df_json=df.to_json(),
        stratify_target=stratify_target, **model_kwargs,
    )
    st.session_state["trained_model"]    = result["pipeline"]
    st.session_state["ml_label_encoder"] = result["label_encoder"]
    st.session_state["ml_results"] = {
        "X_test":     result["X_test"],
        "y_test":     result["y_test"],
        "y_pred":     result["y_pred"],
        "task":       result["task"],
        "model_name": result["model_name"],
        "train_time": result["train_time"],
        "features":   result["features"],
        "target":     result["target"],
    }
    return result


# ══════════════════════════════════════════════════════════════════════
# § 4  ML RESULT CHARTS — RAM cache (TTL = 2 h)
#      Charts tab me slider click karo, chart rebuild nahi hoga
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_actual_vs_predicted(model_hash: str,
                                y_test: list, y_pred: list):
    """Actual vs Predicted scatter (Regression). RAM cached."""
    import plotly.express as px
    yt = np.array(y_test)
    yp = np.array(y_pred)
    fig = px.scatter(
        x=yt, y=yp, labels={"x": "Actual", "y": "Predicted"},
        title="📈 Actual vs Predicted",
        template="plotly_white", opacity=0.75,
        color_discrete_sequence=["#3498db"],
    )
    fig.add_shape(type="line",
                  x0=float(yt.min()), y0=float(yt.min()),
                  x1=float(yt.max()), y1=float(yt.max()),
                  line=dict(color="Red", dash="dash"))
    return fig


@st.cache_data(ttl=7200, show_spinner=False)
def cached_confusion_matrix_fig(model_hash: str,
                                 y_test: list, y_pred: list):
    """Confusion Matrix heatmap (Classification). RAM cached."""
    import plotly.express as px
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return px.imshow(cm, text_auto=True,
                     title="🔢 Confusion Matrix",
                     labels=dict(x="Predicted", y="Actual"),
                     template="plotly_white",
                     color_continuous_scale="Blues")


@st.cache_data(ttl=7200, show_spinner=False)
def cached_feature_importance_fig(model_hash: str,
                                   feature_names: list,
                                   importances: list):
    """Feature Importance horizontal bar. RAM cached."""
    import plotly.express as px
    feat_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
               .sort_values("Importance", ascending=False))
    return px.bar(feat_df, x="Importance", y="Feature",
                  orientation="h", title="🔍 Feature Importance",
                  template="plotly_white", color="Importance",
                  color_continuous_scale="Blues")


# ══════════════════════════════════════════════════════════════════════
# § 5  ML METRICS — RAM cache (TTL = 2 h)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_classification_report_df(model_hash: str,
                                     y_test: list,
                                     y_pred: list) -> pd.DataFrame:
    """Classification report DataFrame. RAM cached."""
    from sklearn.metrics import classification_report
    return pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose().round(4)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_regression_metrics(model_hash: str,
                               y_test: list,
                               y_pred: list) -> dict:
    """R², MAE, RMSE — pre-computed and cached."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    yt = np.array(y_test)
    yp = np.array(y_pred)
    return {
        "r2":   float(r2_score(yt, yp)),
        "mae":  float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
    }


# ══════════════════════════════════════════════════════════════════════
# § 6  BATCH PREDICTIONS — RAM cache (TTL = 2 h)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner="🔮 Running batch predictions…")
def cached_batch_predict(model_hash: str,
                          batch_json: str,
                          feature_cols: tuple) -> np.ndarray:
    """
    Batch CSV predictions — cached so same file nahi dobara run hoga.

    Usage:
        preds = cached_batch_predict(
                    res['model_name'] + get_df_hash(df),
                    batch_df.to_json(), tuple(res['features']))
        batch_df['Prediction'] = preds
    """
    pipeline = st.session_state.get("trained_model")
    if pipeline is None:
        raise RuntimeError("No trained model in session_state.")
    batch_df = pd.read_json(io.StringIO(batch_json))
    return pipeline.predict(batch_df[list(feature_cols)])


# ══════════════════════════════════════════════════════════════════════
# § 7  CROSS-VALIDATION — RAM cache (TTL = 2 h)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner="🔁 Running cross-validation…")
def cached_cross_val(
    df_hash: str, df_json: str, feature_cols: tuple,
    target_col: str, model_name: str, task_type: str,
    folds: int, **model_kwargs,
) -> np.ndarray:
    """Standard CV — fresh pipeline internally."""
    from sklearn.model_selection import cross_val_score
    result = cached_train_pipeline(
        model_name=model_name, task_type=task_type, target_col=target_col,
        feature_cols=feature_cols, test_size=0.2, random_state=42,
        df_hash=df_hash, df_json=df_json, **model_kwargs)
    df = pd.read_json(io.StringIO(df_json))
    metric = "r2" if task_type == "Regression" else "accuracy"
    return cross_val_score(result["pipeline"],
                           df[list(feature_cols)], df[target_col],
                           cv=folds, scoring=metric)


@st.cache_data(ttl=7200, show_spinner="🔁 Running cross-validation…")
def cached_cross_val_fast(
    df_hash: str, df_json: str, feature_cols: tuple,
    target_col: str, task_type: str, folds: int,
    model_name: str, **model_kwargs,
) -> np.ndarray:
    """
    Fast CV — reuses cached pipeline, no re-train.

    Usage in tab_cv:
        h = get_df_hash(df)
        scores = cached_cross_val_fast(
            h, df.to_json(), tuple(res['features']),
            res['target'], res['task'], folds, res['model_name'])
        st.write(f"Mean: {scores.mean():.4f}  ±{scores.std():.4f}")
    """
    from sklearn.model_selection import cross_val_score
    result = cached_train_pipeline(
        model_name=model_name, task_type=task_type, target_col=target_col,
        feature_cols=feature_cols, test_size=0.2, random_state=42,
        df_hash=df_hash, df_json=df_json, **model_kwargs)
    df = pd.read_json(io.StringIO(df_json))
    metric = "r2" if task_type == "Regression" else "accuracy"
    return cross_val_score(result["pipeline"],
                           df[list(feature_cols)], df[target_col],
                           cv=folds, scoring=metric)


# ══════════════════════════════════════════════════════════════════════
# § 8  YDATA PROFILING — 💾 DISK PERSISTED (TTL = 6 h)
#      Sabse heavy operation — ek baar chala, hamesha disk se milta hai
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=21600, persist="disk",
               show_spinner="🔍 Generating profiling report…")
def cached_profile_report(df_hash: str, df: pd.DataFrame,
                           title: str) -> bytes:
    """
    ydata-profiling HTML → bytes.
    💾 Disk persisted — app restart karo, phir bhi instant milta hai.

    Usage in data_upload_page():
        h         = get_df_hash(current_df)
        html_bytes = cached_profile_report(h, current_df,
                         f"Profiling — {selected_file}")
        st.download_button("⬇️ Download", html_bytes,
                           file_name=f"{selected_file}_report.html",
                           mime="text/html")
    """
    from ydata_profiling import ProfileReport
    import tempfile, os
    profile = ProfileReport(df, title=title, explorative=True, minimal=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    profile.to_file(tmp.name)
    with open(tmp.name, "rb") as f:
        html_bytes = f.read()
    os.unlink(tmp.name)
    return html_bytes


# ══════════════════════════════════════════════════════════════════════
# § 9  HTML DASHBOARD REPORT — 💾 DISK PERSISTED (TTL = 2 h)
#
#  Usage in export_page():
#      chart_hashes = [c['title'] for c in st.session_state.saved_charts]
#      html_bytes   = cached_html_report(
#                         get_df_hash(df), df,
#                         st.session_state.kpi_configs,
#                         st.session_state.saved_charts,
#                         chart_hashes)
#      st.download_button("📥 Download HTML", html_bytes, ...)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, persist="disk",
               show_spinner="🌐 Building HTML dashboard…")
def cached_html_report(
    df_hash: str, df: pd.DataFrame,
    kpi_configs: list, saved_charts: list,
    chart_hashes: list,
) -> bytes:
    """
    Full HTML dashboard (KPI cards + Plotly embeds) → UTF-8 bytes.
    💾 Disk persisted — no rebuild on reload unless data changes.
    """
    import plotly.io as pio

    def _kv(d, col, agg):
        if col == "__rows__":       return f"{len(d):,}"
        if col == "__columns__":    return f"{len(d.columns):,}"
        if col == "__numeric__":
            return f"{len(d.select_dtypes(include=['int64','float64']).columns):,}"
        if col == "__missing__":    return f"{int(d.isnull().sum().sum()):,}"
        if col == "__duplicates__": return f"{int(d.duplicated().sum()):,}"
        if col not in d.columns:    return "N/A"
        try:
            s = pd.to_numeric(d[col], errors="coerce")
            if agg == "Sum":    return f"{s.sum():,.2f}"
            if agg == "Count":  return f"{s.count():,}"
            if agg == "Mean":   return f"{s.mean():,.2f}"
            if agg == "Median": return f"{s.median():,.2f}"
            if agg == "Max":    return f"{s.max():,.2f}"
            if agg == "Min":    return f"{s.min():,.2f}"
            if agg == "Std":    return f"{s.std():,.2f}"
        except Exception:
            pass
        return str(d[col].count())

    kpi_html = ""
    for cfg in kpi_configs:
        label = cfg.get("label", "KPI")
        col   = cfg.get("column", "__rows__")
        agg   = cfg.get("agg",    "Count")
        color = cfg.get("color",  "#3498db")
        val   = _kv(df, col, agg)
        vc    = "#e74c3c" if (col == "__missing__" and
                               int(df.isnull().sum().sum()) > 0) else color
        kpi_html += (
            f'<div class="metric-card" style="border-top:4px solid {color};">'
            f'<span class="metric-label">{label}</span>'
            f'<p class="metric-value" style="color:{vc};">{val}</p>'
            f'<span class="metric-agg">{agg}</span></div>')

    charts_html = ""
    if saved_charts:
        for chart in saved_charts:
            cdiv  = pio.to_html(chart["fig"], full_html=False,
                                include_plotlyjs="cdn")
            desc  = chart.get("description", "").strip()
            dnote = (
                f'<div class="analysis-note">'
                f'<span class="note-label">📝 Analysis Note:</span>{desc}</div>'
            ) if desc else ""
            charts_html += (
                f'<div class="chart-container">'
                f'<h3 class="chart-title">{chart["title"]}</h3>'
                f'{cdiv}{dnote}</div>')
    else:
        charts_html = ("<p style='color:#7f8c8d;padding:20px;'>"
                       "No visualizations pinned to the dashboard.</p>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>DataMate AI Dashboard</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:'Segoe UI',sans-serif;background:#f4f6f9;
          color:#2c3e50;padding:40px 30px;}}
    .report-header{{background:linear-gradient(135deg,#2c3e50,#3498db);
          color:white;padding:30px 40px;border-radius:12px;
          margin-bottom:30px;text-align:center;}}
    .report-header h1{{font-size:2rem;font-weight:800;}}
    .section-header{{font-size:1.25rem;font-weight:700;color:#2c3e50;
          border-bottom:3px solid #3498db;padding-bottom:8px;
          margin:35px 0 20px;}}
    .kpi-grid{{display:flex;flex-wrap:wrap;gap:18px;margin-bottom:10px;}}
    .metric-card{{background:white;padding:22px 20px;border-radius:12px;
          box-shadow:0 4px 12px rgba(0,0,0,.08);flex:1 1 160px;
          min-width:140px;text-align:center;}}
    .metric-value{{font-size:2rem;font-weight:800;margin:10px 0 4px;display:block;}}
    .metric-label{{color:#7f8c8d;font-size:.78rem;text-transform:uppercase;
          letter-spacing:1px;font-weight:600;}}
    .metric-agg{{color:#bdc3c7;font-size:.72rem;text-transform:uppercase;}}
    .chart-container{{background:white;padding:30px;border-radius:12px;
          box-shadow:0 4px 15px rgba(0,0,0,.06);margin-bottom:35px;}}
    .chart-title{{font-size:1.1rem;font-weight:700;color:#2c3e50;
          margin-bottom:18px;padding-bottom:10px;
          border-bottom:1px solid #ecf0f1;}}
    .analysis-note{{background:#eaf4fb;border-left:5px solid #3498db;
          padding:14px 18px;margin-top:20px;border-radius:6px;
          color:#2c3e50;font-size:.92rem;line-height:1.6;}}
    .note-label{{font-weight:700;color:#2980b9;display:block;
          margin-bottom:6px;font-size:.88rem;}}
    @media(max-width:600px){{body{{padding:16px;}}
          .metric-card{{min-width:100%;}}}}
  </style>
</head>
<body>
  <div class="report-header"><h1>📊 DataMate AI Dashboard</h1></div>
  <div class="section-header">Key Performance Indicators</div>
  <div class="kpi-grid">{kpi_html}</div>
  <div class="section-header">Visualizations &amp; Analysis</div>
  {charts_html}
</body>
</html>"""
    return html.encode("utf-8")


# ══════════════════════════════════════════════════════════════════════
# § 10  EXPORT BYTES — 💾 DISK PERSISTED (TTL = 2 h)
#       Download button press karo — instantly file milti hai
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, persist="disk", show_spinner=False)
def cached_export_csv(df_hash: str, df: pd.DataFrame) -> bytes:
    """DataFrame → CSV bytes. 💾 Disk cached."""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(ttl=7200, persist="disk", show_spinner=False)
def cached_export_excel(df_hash: str, df: pd.DataFrame) -> bytes:
    """DataFrame → Excel (.xlsx) bytes. 💾 Disk cached."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


@st.cache_data(ttl=7200, persist="disk", show_spinner=False)
def cached_export_json(df_hash: str, df: pd.DataFrame) -> bytes:
    """DataFrame → JSON (records) bytes. 💾 Disk cached."""
    return df.to_json(orient="records", indent=2).encode("utf-8")


@st.cache_data(ttl=7200, persist="disk", show_spinner=False)
def cached_export_parquet(df_hash: str, df: pd.DataFrame) -> bytes:
    """DataFrame → Parquet bytes. 💾 Disk cached."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


@st.cache_data(ttl=7200, persist="disk", show_spinner=False)
def cached_export_txt(
    df_hash:        str,
    df:             pd.DataFrame,
    file_name:      str,
    include_dtypes: bool = True,
    include_desc:   bool = True,
    include_miss:   bool = True,
    include_top5:   bool = True,
    include_corr:   bool = False,
) -> bytes:
    """
    Plain-text summary → UTF-8 bytes. 💾 Disk cached.

    Usage in export_page() TXT tab:
        h   = get_df_hash(df)
        txt = cached_export_txt(h, df, selected_file_name,
                                include_dtypes, include_desc,
                                include_miss, include_top5, include_corr)
        st.download_button("📥 Download TXT", txt, ...)
    """
    from datetime import datetime
    sep   = "=" * 60
    lines = [
        "DATAMATE AI — DATA SUMMARY REPORT", sep,
        f"File       : {file_name}",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Rows       : {len(df):,}",
        f"Columns    : {len(df.columns)}",
        f"Missing    : {df.isnull().sum().sum():,}",
        f"Duplicates : {df.duplicated().sum():,}",
        f"Memory     : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB",
        sep,
    ]
    if include_dtypes:
        lines += ["\nCOLUMN DATA TYPES", "-"*40]
        for col in df.columns:
            lines.append(f"  {col:<30} {str(df[col].dtype)}")
    if include_desc:
        lines += ["\nDESCRIPTIVE STATISTICS", "-"*40,
                  df.describe(include="all").round(3).to_string()]
    if include_miss:
        lines += ["\nMISSING VALUES PER COLUMN", "-"*40]
        for col in df.columns:
            n   = df[col].isnull().sum()
            pct = n / len(df) * 100
            lines.append(f"  {col:<30} {n:>6}  ({pct:.1f}%)")
    if include_top5:
        lines += ["\nTOP 5 ROWS", "-"*40, df.head(5).to_string(index=False)]
    if include_corr:
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            lines += ["\nCORRELATION MATRIX", "-"*40,
                      num_df.corr().round(3).to_string()]
    return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════
# § 11  CHART PNG BYTES — 💾 DISK PERSISTED (TTL = 6 h)
#       kaleido subprocess ek baar chalta hai, phir disk se PNG milta hai
#
#  Usage in export_page() PDF loop:
#      for i, chart in enumerate(saved_charts):
#          png = cached_chart_png(chart['title'], chart['fig'])
#          if png:
#              import tempfile, os
#              t = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
#              t.write(png); t.close()
#              pdf.add_chart_image(t.name,
#                                  title=f"Fig {i+1}: {chart['title']}")
#              os.unlink(t.name)
#          else:
#              pdf.body('[Chart skipped — pip install kaleido]')
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=21600, persist="disk", show_spinner=False)
def cached_chart_png(chart_title: str, fig) -> "bytes | None":
    """
    Plotly figure → PNG bytes (800×450, scale 1.5). 💾 Disk cached.
    Returns None if kaleido is not installed.
    """
    try:
        buf = io.BytesIO()
        fig.write_image(buf, format="png", width=800, height=450, scale=1.5)
        return buf.getvalue()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# § 13  PLOTLY CHART HELPERS — RAM cache (TTL = 2 h)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_histogram(df_hash: str, df: pd.DataFrame,
                     col: str, bins: int = 30):
    import plotly.express as px
    return px.histogram(df, x=col, nbins=bins, marginal="box",
                        template="plotly_white",
                        title=f"Distribution of {col}",
                        color_discrete_sequence=px.colors.qualitative.Set2)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_corr_heatmap(df_hash: str, df: pd.DataFrame,
                        method: str = "pearson"):
    import plotly.express as px
    corr = cached_correlation(df_hash, df, method)
    return px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdBu_r",
                     title=f"{method.title()} Correlation Matrix",
                     template="plotly_white")


@st.cache_data(ttl=7200, show_spinner=False)
def cached_missing_bar(df_hash: str, df: pd.DataFrame):
    import plotly.express as px
    miss = cached_missing_summary(df_hash, df)
    if miss.empty:
        return None
    return px.bar(miss, x="Column", y="Missing %",
                  color="Missing %", color_continuous_scale="Reds",
                  title="Missing % by Column", template="plotly_white")


# ══════════════════════════════════════════════════════════════════════
# § 15  UTILITY — stable DataFrame hash
# ══════════════════════════════════════════════════════════════════════

def get_df_hash(df: pd.DataFrame) -> str:
    """
    Fast MD5 hash of a DataFrame.
    Har cached function ke first argument mein pass karo
    taaki cache sirf tab invalidate ho jab data actually change ho.

    Example:
        h = get_df_hash(st.session_state.df)
        profile = cached_column_profile(h, st.session_state.df)
    """
    try:
        raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except Exception:
        raw = df.to_csv(index=False).encode()
    return hashlib.md5(raw).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# § 0   CSS / STYLING CACHE — RAM (indefinite, no TTL)
#       App.py mein load_css_styling() ko replace karo:
#           st.markdown(cached_css_styling(), unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_css_styling() -> str:
    """
    Returns the full app CSS block as a string.
    Cached indefinitely in RAM — only computed once per server lifetime.
    No TTL: CSS never changes without a code deploy.

    Usage in App.py (replace the existing load_css_styling call):
        st.markdown(cached_css_styling(), unsafe_allow_html=True)
    """
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    @keyframes fadeInUp   { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
    @keyframes slideInLeft{ from{opacity:0;transform:translateX(-30px)} to{opacity:1;transform:translateX(0)} }
    @keyframes pulse-glow { 0%,100%{box-shadow:0 0 0 0 rgba(52,152,219,.4)} 50%{box-shadow:0 0 0 8px rgba(52,152,219,0)} }
    @keyframes shimmer    { 0%{background-position:-200% center} 100%{background-position:200% center} }
    @keyframes float      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
    @keyframes countUp    { from{opacity:0;transform:scale(.8)} to{opacity:1;transform:scale(1)} }
    @keyframes sidebarFadeIn{ from{opacity:0;transform:translateX(-15px)} to{opacity:1;transform:translateX(0)} }

    html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }

    section[data-testid="stSidebar"]{
        background:linear-gradient(180deg,#0f1923 0%,#1a2535 100%) !important;
    }
    section[data-testid="stSidebar"] .stButton button{
        background-color:rgba(255,255,255,.05);border:1px solid rgba(174,214,241,.2);
        color:#cdd9e5 !important;font-family:'DM Sans',sans-serif;font-weight:600 !important;
        font-size:14px;border-radius:10px;transition:all .3s cubic-bezier(.25,.8,.25,1);
        width:100%;padding:11px 16px;text-align:left;margin-bottom:6px;
        animation:sidebarFadeIn .4s ease both;
    }
    section[data-testid="stSidebar"] .stButton button:hover{
        transform:translateX(6px);
        background:linear-gradient(135deg,#3498db 0%,#2980b9 100%) !important;
        color:white !important;border:1px solid #2980b9 !important;
        box-shadow:0 4px 18px rgba(52,152,219,.45);
    }
    .main-header{
        background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
        color:white;padding:22px 28px;border-radius:14px;text-align:center;
        margin-bottom:24px;font-family:'Sora',sans-serif;font-size:26px;
        font-weight:800;letter-spacing:.5px;box-shadow:0 8px 32px rgba(0,0,0,.18);
        animation:fadeInUp .6s ease both;position:relative;overflow:hidden;
    }
    .main-header::after{
        content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;
        background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
        animation:shimmer 3s infinite;
    }
    .sidebar-header-card{
        background:linear-gradient(135deg,#3498db,#1a6dad) !important;
        color:white !important;padding:20px !important;border-radius:12px !important;
        margin-bottom:18px !important;text-align:center !important;
        box-shadow:0 4px 20px rgba(52,152,219,.4) !important;
        animation:pulse-glow 3s infinite;
    }
    .metric-card{
        background:white;padding:20px 18px;border-radius:14px;
        box-shadow:0 4px 16px rgba(0,0,0,.08);border-left:4px solid #3498db;
        margin:10px 0;transition:transform .25s ease,box-shadow .25s ease;
        animation:fadeInUp .5s ease both;
    }
    .metric-card:hover{ transform:translateY(-6px) scale(1.01); box-shadow:0 12px 32px rgba(52,152,219,.18); }
    .metric-card p{ animation:countUp .6s ease both; }
    .section-header{
        font-family:'Sora',sans-serif;font-size:19px;font-weight:700;color:#1a2535;
        margin:22px 0 14px;padding-bottom:10px;border-bottom:2px solid #3498db;
        animation:slideInLeft .5s ease both;
    }
    .content-frame{
        border:1.5px solid #dee2e6;border-radius:12px;padding:22px;margin:12px 0;
        background:white;box-shadow:0 2px 12px rgba(0,0,0,.06);
        animation:fadeInUp .5s ease both;
    }
    .chart-container{
        background:white;padding:18px;border-radius:12px;
        box-shadow:0 4px 16px rgba(0,0,0,.08);margin:12px 0;
        transition:box-shadow .2s;animation:fadeInUp .5s ease both;
    }
    .chart-container:hover{ box-shadow:0 8px 28px rgba(0,0,0,.12); }
    .upload-section{
        background:#f8f9fa;border:2px dashed #AED6F1;border-radius:12px;
        padding:32px;text-align:center;
    }
    .feature-card{
        background:white;border-radius:14px;padding:20px;
        box-shadow:0 4px 16px rgba(0,0,0,.07);border-top:3px solid #3498db;
        margin:8px 0;transition:transform .25s ease;animation:fadeInUp .5s ease both;
    }
    .feature-card:hover{ transform:translateY(-4px); }
    .feature-icon{ font-size:28px;display:block;margin-bottom:8px; }
    .feature-title{ font-family:'Sora',sans-serif;font-weight:700;font-size:14px;color:#1a2535;margin-bottom:6px; }
    .feature-desc{ font-size:12.5px;color:#666;line-height:1.55; }
    .founder-card{ transition:transform .25s ease; }
    .founder-card:hover{ transform:translateY(-4px); }
</style>"""


# ══════════════════════════════════════════════════════════════════════
# § 16  DATA PREPROCESSING CACHE — RAM (TTL = 2 h)
#       12-tab preprocessing center ke liye — every op cached.
#       Pass df_hash as first arg so cache invalidates only when
#       the underlying DataFrame actually changes.
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_drop_duplicates(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows. RAM cached."""
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_fill_missing(
    df_hash: str,
    df: pd.DataFrame,
    strategy: str = "mean",   # "mean" | "median" | "mode" | "constant"
    fill_value=0,             # used only when strategy == "constant"
    columns: "tuple | None" = None,
) -> pd.DataFrame:
    """
    Fill missing values. RAM cached.
    strategy: 'mean' / 'median' / 'mode' / 'constant'
    columns : tuple of col names to fill (None = all)

    Usage:
        h   = get_df_hash(df)
        out = cached_fill_missing(h, df, strategy="median")
    """
    cols = list(columns) if columns else df.columns.tolist()
    out  = df.copy()
    for col in cols:
        if out[col].isnull().sum() == 0:
            continue
        if strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
            out[col].fillna(out[col].mean(), inplace=True)
        elif strategy == "median" and pd.api.types.is_numeric_dtype(out[col]):
            out[col].fillna(out[col].median(), inplace=True)
        elif strategy == "mode":
            mode_val = out[col].mode()
            if not mode_val.empty:
                out[col].fillna(mode_val[0], inplace=True)
        elif strategy == "constant":
            out[col].fillna(fill_value, inplace=True)
    return out


@st.cache_data(ttl=7200, show_spinner=False)
def cached_drop_missing_cols(
    df_hash: str, df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Drop columns whose missing-value ratio exceeds `threshold`. RAM cached.
    threshold = 0.5 → drop columns with >50% missing.

    Usage:
        out = cached_drop_missing_cols(h, df, threshold=0.6)
    """
    missing_ratio = df.isnull().mean()
    keep = missing_ratio[missing_ratio <= threshold].index.tolist()
    return df[keep]


@st.cache_data(ttl=7200, show_spinner=False)
def cached_detect_outliers_iqr(
    df_hash: str, df: pd.DataFrame, col: str
) -> dict:
    """
    IQR-based outlier detection for a numeric column. RAM cached.
    Returns: { 'lower': float, 'upper': float,
               'n_outliers': int, 'outlier_pct': float }

    Usage:
        info = cached_detect_outliers_iqr(h, df, "salary")
        st.write(f"Outliers: {info['n_outliers']} ({info['outlier_pct']:.1f}%)")
    """
    s  = df[col].dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr    = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr
    mask   = (s < lower) | (s > upper)
    return {
        "lower":       lower,
        "upper":       upper,
        "n_outliers":  int(mask.sum()),
        "outlier_pct": float(mask.mean() * 100),
    }


@st.cache_data(ttl=7200, show_spinner=False)
def cached_detect_outliers_zscore(
    df_hash: str, df: pd.DataFrame, col: str, threshold: float = 3.0
) -> dict:
    """
    Z-score outlier detection for a numeric column. RAM cached.
    Returns: { 'n_outliers': int, 'outlier_pct': float, 'threshold': float }

    Usage:
        info = cached_detect_outliers_zscore(h, df, "age", threshold=3.0)
    """
    from scipy import stats as _stats
    s    = df[col].dropna()
    zs   = np.abs(_stats.zscore(s))
    mask = zs > threshold
    return {
        "n_outliers":  int(mask.sum()),
        "outlier_pct": float(mask.mean() * 100),
        "threshold":   threshold,
    }


@st.cache_data(ttl=7200, show_spinner=False)
def cached_clip_outliers_iqr(
    df_hash: str, df: pd.DataFrame, col: str
) -> pd.DataFrame:
    """
    Clip outliers in `col` to IQR bounds (Winsorization). RAM cached.

    Usage:
        out = cached_clip_outliers_iqr(h, df, "salary")
    """
    info = cached_detect_outliers_iqr(df_hash, df, col)
    out  = df.copy()
    out[col] = out[col].clip(lower=info["lower"], upper=info["upper"])
    return out


@st.cache_data(ttl=7200, show_spinner=False)
def cached_label_encode(
    df_hash: str, df: pd.DataFrame, columns: tuple
) -> tuple:
    """
    Label-encode categorical columns. RAM cached.
    Returns (encoded_df, {col: LabelEncoder}) — encoders needed for inversion.

    Usage:
        enc_df, encoders = cached_label_encode(h, df, tuple(cat_cols))
        original = encoders['gender'].inverse_transform(enc_df['gender'])
    """
    from sklearn.preprocessing import LabelEncoder
    out      = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le
    return out, encoders


@st.cache_data(ttl=7200, show_spinner=False)
def cached_onehot_encode(
    df_hash: str, df: pd.DataFrame, columns: tuple
) -> pd.DataFrame:
    """
    One-hot encode categorical columns (drop_first=False). RAM cached.

    Usage:
        out = cached_onehot_encode(h, df, tuple(cat_cols))
    """
    return pd.get_dummies(df, columns=list(columns), drop_first=False)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_minmax_scale(
    df_hash: str, df: pd.DataFrame, columns: tuple
) -> pd.DataFrame:
    """
    MinMax scale numeric columns to [0, 1]. RAM cached.

    Usage:
        out = cached_minmax_scale(h, df, tuple(num_cols))
    """
    from sklearn.preprocessing import MinMaxScaler
    out    = df.copy()
    scaler = MinMaxScaler()
    out[list(columns)] = scaler.fit_transform(out[list(columns)])
    return out


@st.cache_data(ttl=7200, show_spinner=False)
def cached_standard_scale(
    df_hash: str, df: pd.DataFrame, columns: tuple
) -> pd.DataFrame:
    """
    Standard-scale numeric columns (mean=0, std=1). RAM cached.

    Usage:
        out = cached_standard_scale(h, df, tuple(num_cols))
    """
    from sklearn.preprocessing import StandardScaler
    out    = df.copy()
    scaler = StandardScaler()
    out[list(columns)] = scaler.fit_transform(out[list(columns)])
    return out


@st.cache_data(ttl=7200, show_spinner=False)
def cached_pca_transform(
    df_hash: str, df: pd.DataFrame, n_components: int = 2
) -> dict:
    """
    PCA on all numeric columns. RAM cached.
    Returns {
        'df_pca':            DataFrame with PC columns,
        'explained_variance': list[float],
        'cumulative_var':     list[float],
        'loadings':           DataFrame (features × PCs),
    }

    Usage:
        result = cached_pca_transform(h, df, n_components=3)
        st.dataframe(result['df_pca'])
        st.write("Variance explained:", result['explained_variance'])
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    num_df = df.select_dtypes(include="number").dropna(axis=1)
    scaled = StandardScaler().fit_transform(num_df)
    pca    = PCA(n_components=min(n_components, num_df.shape[1]))
    comps  = pca.fit_transform(scaled)

    pc_cols  = [f"PC{i+1}" for i in range(pca.n_components_)]
    df_pca   = pd.DataFrame(comps, columns=pc_cols, index=df.index)
    ev       = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cum_ev   = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]
    loadings = pd.DataFrame(
        pca.components_.T, index=num_df.columns, columns=pc_cols
    ).round(4)
    return {
        "df_pca":            df_pca,
        "explained_variance": ev,
        "cumulative_var":     cum_ev,
        "loadings":           loadings,
    }


@st.cache_data(ttl=7200, show_spinner=False)
def cached_ml_readiness(df_hash: str, df: pd.DataFrame) -> dict:
    """
    ML readiness check — quick audit of the DataFrame. RAM cached.
    Returns:
        {
          'missing_pct':    float,          # overall % missing
          'high_missing':   list[str],      # cols > 30% missing
          'constant_cols':  list[str],      # cols with 1 unique value
          'high_cardinality': list[str],    # cat cols with >50 unique vals
          'skewed_cols':    list[str],      # numeric cols |skew| > 2
          'recommended_actions': list[str],
        }

    Usage:
        audit = cached_ml_readiness(h, df)
        for action in audit['recommended_actions']:
            st.warning(action)
    """
    from scipy.stats import skew as _skew

    missing_pct     = float(df.isnull().mean().mean() * 100)
    high_missing    = df.columns[df.isnull().mean() > 0.30].tolist()
    constant_cols   = [c for c in df.columns if df[c].nunique() <= 1]
    cat_cols        = df.select_dtypes(exclude="number").columns
    high_cardinality= [c for c in cat_cols if df[c].nunique() > 50]
    num_cols        = df.select_dtypes(include="number").columns
    skewed_cols     = [
        c for c in num_cols
        if len(df[c].dropna()) > 2 and abs(_skew(df[c].dropna())) > 2
    ]

    actions = []
    if missing_pct > 5:
        actions.append(f"⚠️ {missing_pct:.1f}% values missing — consider imputation.")
    if high_missing:
        actions.append(f"🗑️ Columns with >30% missing: {high_missing} — consider dropping.")
    if constant_cols:
        actions.append(f"📌 Constant columns (no variance): {constant_cols} — drop them.")
    if high_cardinality:
        actions.append(f"🔢 High-cardinality categoricals: {high_cardinality} — encode carefully.")
    if skewed_cols:
        actions.append(f"📐 Highly skewed columns: {skewed_cols} — consider log-transform.")
    if not actions:
        actions.append("✅ Dataset looks ML-ready!")

    return {
        "missing_pct":        missing_pct,
        "high_missing":       high_missing,
        "constant_cols":      constant_cols,
        "high_cardinality":   high_cardinality,
        "skewed_cols":        skewed_cols,
        "recommended_actions": actions,
    }


# ══════════════════════════════════════════════════════════════════════
# § 17  EXTENDED VISUALIZATION HELPERS — RAM (TTL = 2 h)
#       30+ chart types ke liye common builders — chart render
#       dobara nahi hoga jab tak data ya params same hon
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_scatter(
    df_hash: str, df: pd.DataFrame,
    x: str, y: str,
    color_col: "str | None" = None,
    size_col:  "str | None" = None,
    trendline: bool = False,
):
    """Scatter plot (optionally with trendline). RAM cached."""
    import plotly.express as px
    kwargs = dict(x=x, y=y, template="plotly_white",
                  title=f"{y} vs {x}", opacity=0.75)
    if color_col: kwargs["color"] = color_col
    if size_col:  kwargs["size"]  = size_col
    if trendline: kwargs["trendline"] = "ols"
    return px.scatter(df, **kwargs)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_box_plot(
    df_hash: str, df: pd.DataFrame,
    y_col: str, x_col: "str | None" = None,
):
    """Box plot. RAM cached."""
    import plotly.express as px
    kwargs = dict(y=y_col, template="plotly_white",
                  title=f"Box Plot — {y_col}", points="outliers")
    if x_col: kwargs["x"] = x_col
    return px.box(df, **kwargs)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_violin_plot(
    df_hash: str, df: pd.DataFrame,
    y_col: str, x_col: "str | None" = None,
):
    """Violin plot. RAM cached."""
    import plotly.express as px
    kwargs = dict(y=y_col, template="plotly_white",
                  title=f"Violin — {y_col}", box=True)
    if x_col: kwargs["x"] = x_col
    return px.violin(df, **kwargs)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_bar_chart(
    df_hash: str, df: pd.DataFrame,
    col: str, top_n: int = 20,
    orientation: str = "v",    # "v" | "h"
):
    """Top-N value-count bar chart. RAM cached."""
    import plotly.express as px
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "Count"]
    if orientation == "h":
        return px.bar(vc, x="Count", y=col, orientation="h",
                      title=f"Top {top_n} — {col}",
                      template="plotly_white",
                      color="Count", color_continuous_scale="Blues")
    return px.bar(vc, x=col, y="Count",
                  title=f"Top {top_n} — {col}",
                  template="plotly_white",
                  color="Count", color_continuous_scale="Blues")


@st.cache_data(ttl=7200, show_spinner=False)
def cached_line_chart(
    df_hash: str, df: pd.DataFrame,
    x_col: str, y_col: str,
    color_col: "str | None" = None,
):
    """Line chart (time series / trend). RAM cached."""
    import plotly.express as px
    kwargs = dict(x=x_col, y=y_col, template="plotly_white",
                  title=f"{y_col} over {x_col}", markers=True)
    if color_col: kwargs["color"] = color_col
    return px.line(df.sort_values(x_col), **kwargs)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_pie_chart(
    df_hash: str, df: pd.DataFrame,
    col: str, top_n: int = 10,
):
    """Pie chart of top-N categories. RAM cached."""
    import plotly.express as px
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "Count"]
    return px.pie(vc, names=col, values="Count",
                  title=f"Distribution — {col} (top {top_n})",
                  template="plotly_white",
                  color_discrete_sequence=px.colors.qualitative.Set2)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_area_chart(
    df_hash: str, df: pd.DataFrame,
    x_col: str, y_col: str,
    color_col: "str | None" = None,
):
    """Stacked area chart. RAM cached."""
    import plotly.express as px
    kwargs = dict(x=x_col, y=y_col, template="plotly_white",
                  title=f"{y_col} Area — {x_col}")
    if color_col: kwargs["color"] = color_col
    return px.area(df.sort_values(x_col), **kwargs)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_heatmap_pivot(
    df_hash: str, df: pd.DataFrame,
    index: str, columns: str,
    values: str, aggfunc: str = "mean",
):
    """
    Pivot heatmap (category × category → numeric). RAM cached.
    aggfunc: 'mean' | 'sum' | 'count' | 'median'

    Usage:
        fig = cached_heatmap_pivot(h, df, "region", "product", "sales")
        st.plotly_chart(fig)
    """
    import plotly.express as px
    pivot = df.pivot_table(index=index, columns=columns,
                           values=values, aggfunc=aggfunc).round(2)
    return px.imshow(pivot, text_auto=".1f", aspect="auto",
                     color_continuous_scale="RdBu_r",
                     title=f"{aggfunc.title()} of {values} by {index} × {columns}",
                     template="plotly_white")


@st.cache_data(ttl=7200, show_spinner=False)
def cached_pairplot(
    df_hash: str, df: pd.DataFrame,
    cols: tuple, color_col: "str | None" = None,
):
    """
    Scatter matrix (pair plot) for selected columns. RAM cached.
    Useful in multivariate visualization tab.
    """
    import plotly.express as px
    kwargs = dict(dimensions=list(cols), template="plotly_white",
                  title="Scatter Matrix", opacity=0.65)
    if color_col: kwargs["color"] = color_col
    return px.scatter_matrix(df, **kwargs)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_3d_scatter(
    df_hash: str, df: pd.DataFrame,
    x: str, y: str, z: str,
    color_col: "str | None" = None,
):
    """3-D scatter plot. RAM cached."""
    import plotly.express as px
    kwargs = dict(x=x, y=y, z=z, template="plotly_white",
                  title=f"3D Scatter — {x} / {y} / {z}", opacity=0.8)
    if color_col: kwargs["color"] = color_col
    return px.scatter_3d(df, **kwargs)


# ══════════════════════════════════════════════════════════════════════
# § 18  QUICK STATS / DATA SUMMARY — RAM (TTL = 2 h)
#       Sidebar ya overview cards ke liye instant metadata
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=7200, show_spinner=False)
def cached_quick_stats(df_hash: str, df: pd.DataFrame) -> dict:
    """
    Single-call dataset summary — replaces repeated df.shape / df.dtypes
    calls scattered across pages. RAM cached.

    Returns:
        {
          'rows': int, 'cols': int,
          'numeric_cols': list[str], 'cat_cols': list[str],
          'datetime_cols': list[str],
          'n_missing': int, 'missing_pct': float,
          'n_duplicates': int,
          'memory_mb': float,
        }

    Usage:
        qs = cached_quick_stats(h, df)
        st.write(f"Rows: {qs['rows']}  |  Cols: {qs['cols']}")
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()
    n_missing = int(df.isnull().sum().sum())
    return {
        "rows":          len(df),
        "cols":          len(df.columns),
        "numeric_cols":  num_cols,
        "cat_cols":      cat_cols,
        "datetime_cols": dt_cols,
        "n_missing":     n_missing,
        "missing_pct":   round(n_missing / (len(df) * len(df.columns)) * 100, 2) if df.size else 0.0,
        "n_duplicates":  int(df.duplicated().sum()),
        "memory_mb":     round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }


@st.cache_data(ttl=7200, show_spinner=False)
def cached_numeric_cols(df_hash: str, df: pd.DataFrame) -> list:
    """List of numeric column names. RAM cached."""
    return df.select_dtypes(include="number").columns.tolist()


@st.cache_data(ttl=7200, show_spinner=False)
def cached_categorical_cols(df_hash: str, df: pd.DataFrame) -> list:
    """List of categorical / object column names. RAM cached."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


@st.cache_data(ttl=7200, show_spinner=False)
def cached_datetime_cols(df_hash: str, df: pd.DataFrame) -> list:
    """List of datetime column names. RAM cached."""
    return df.select_dtypes(include="datetime").columns.tolist()


@st.cache_data(ttl=7200, show_spinner=False)
def cached_dtype_summary(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact dtype summary table for the Overview tab. RAM cached.
    Returns DataFrame with columns: [Column, Dtype, Category].
    """
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            cat = "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            cat = "DateTime"
        elif pd.api.types.is_bool_dtype(df[col]):
            cat = "Boolean"
        else:
            cat = "Categorical"
        rows.append({"Column": col, "Dtype": dtype, "Category": cat})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ══════════════════════════════════════════════════════════════════════

_GUIDE = """
┌───────────────────────────────────────────────────────────────────────────┐
│      DataMate AI — cache_config.py  v5  🚀 DISK CACHE + FULL COVERAGE    │
├──────────────────────┬────────────────────────────────────────────────────┤
│  💾 DISK PERSISTED   │ survive app restart, instant on reload             │
├──────────────────────┼────────────────────────────────────────────────────┤
│  File Loading        │ load_any_file(uploaded_file)                        │
│  Image/Logo          │ cached_load_image("datamate_logo_50.png")           │
│  ydata Profiling     │ cached_profile_report(h, df, title) → bytes        │
│  HTML Dashboard      │ cached_html_report(h, df, kpis, charts, hashes)    │
│  Export CSV          │ cached_export_csv(h, df) → bytes                   │
│  Export Excel        │ cached_export_excel(h, df) → bytes                 │
│  Export JSON         │ cached_export_json(h, df) → bytes                  │
│  Export Parquet      │ cached_export_parquet(h, df) → bytes               │
│  Export TXT          │ cached_export_txt(h, df, name, ...) → bytes        │
│  Chart PNG (PDF)     │ cached_chart_png(title, fig) → bytes               │
├──────────────────────┼────────────────────────────────────────────────────┤
│  🧠 RAM CACHED       │ fast ops — RAM cache kaafi hai                     │
├──────────────────────┼────────────────────────────────────────────────────┤
│  § 0  CSS Styling    │ cached_css_styling() → str  [no TTL, permanent]    │
│  Statistics          │ cached_describe / column_profile /                 │
│                      │ missing_summary / correlation / value_counts        │
│  KPI Value           │ cached_kpi_value(h, df, column, agg)               │
│  ML Training         │ run_ml_training(df, features, ...)  ← wrapper      │
│  Result Charts       │ cached_actual_vs_predicted(hash, yt, yp)           │
│                      │ cached_confusion_matrix_fig(hash, yt, yp)          │
│                      │ cached_feature_importance_fig(hash, names, imp)    │
│  ML Metrics          │ cached_regression_metrics(hash, yt, yp)            │
│                      │ cached_classification_report_df(hash, yt, yp)      │
│  Batch Predict       │ cached_batch_predict(hash, batch_json, cols)       │
│  Cross-Val (Fast)    │ cached_cross_val_fast(h, json, cols, ...)          │
│  Chart Helpers       │ cached_histogram / corr_heatmap / missing_bar      │
├──────────────────────┼────────────────────────────────────────────────────┤
│  § 16 PREPROCESSING  │ 12-tab preprocessing center                        │
│  Drop Dupes          │ cached_drop_duplicates(h, df)                      │
│  Fill Missing        │ cached_fill_missing(h, df, strategy, fill, cols)   │
│  Drop Miss Cols      │ cached_drop_missing_cols(h, df, threshold=0.5)     │
│  Outlier IQR         │ cached_detect_outliers_iqr(h, df, col) → dict      │
│  Outlier Z-score     │ cached_detect_outliers_zscore(h, df, col, thr)     │
│  Clip Outliers       │ cached_clip_outliers_iqr(h, df, col) → df          │
│  Label Encode        │ cached_label_encode(h, df, cols) → (df, encoders)  │
│  One-Hot Encode      │ cached_onehot_encode(h, df, cols) → df             │
│  MinMax Scale        │ cached_minmax_scale(h, df, cols) → df              │
│  Standard Scale      │ cached_standard_scale(h, df, cols) → df            │
│  PCA                 │ cached_pca_transform(h, df, n) → dict              │
│  ML Readiness        │ cached_ml_readiness(h, df) → dict                  │
├──────────────────────┼────────────────────────────────────────────────────┤
│  § 17 VISUALIZATION  │ 30+ chart types, all RAM cached                    │
│  Scatter             │ cached_scatter(h, df, x, y, color, size, trend)    │
│  Box Plot            │ cached_box_plot(h, df, y_col, x_col)               │
│  Violin              │ cached_violin_plot(h, df, y_col, x_col)            │
│  Bar Chart           │ cached_bar_chart(h, df, col, top_n, orientation)   │
│  Line / Time Series  │ cached_line_chart(h, df, x, y, color)              │
│  Pie Chart           │ cached_pie_chart(h, df, col, top_n)                │
│  Area Chart          │ cached_area_chart(h, df, x, y, color)              │
│  Heatmap Pivot       │ cached_heatmap_pivot(h, df, idx, cols, vals, agg)  │
│  Pair Plot           │ cached_pairplot(h, df, cols_tuple, color)          │
│  3D Scatter          │ cached_3d_scatter(h, df, x, y, z, color)           │
├──────────────────────┼────────────────────────────────────────────────────┤
│  § 18 QUICK STATS    │ single-call dataset metadata                       │
│  Full Stats          │ cached_quick_stats(h, df) → dict                   │
│  Numeric Cols        │ cached_numeric_cols(h, df) → list                  │
│  Categorical Cols    │ cached_categorical_cols(h, df) → list              │
│  DateTime Cols       │ cached_datetime_cols(h, df) → list                 │
│  Dtype Summary       │ cached_dtype_summary(h, df) → DataFrame            │
├──────────────────────┼────────────────────────────────────────────────────┤
│  UTILITY             │ h = get_df_hash(st.session_state.df)               │
│  CLEAR CACHE         │ terminal:  streamlit cache clear                   │
│                      │ manual:    .streamlit/cache/ folder delete karo    │
└──────────────────────┴────────────────────────────────────────────────────┘
"""

print(_GUIDE)