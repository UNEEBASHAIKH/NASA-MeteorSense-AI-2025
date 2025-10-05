# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.05)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

st.set_page_config(layout="wide", page_title="NASA MeteorSense AI")

# --- Load default dataset (if present) ---
@st.cache_data
def load_default_df(path="NASA Near-Earth ObjectCleaned(NEO).csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

df = load_default_df()

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an option:", ["EDA", "Prediction"])

st.title("NASA MeteorSense AI")
st.write("Analysis & simple hazard prediction demo for Near-Earth Objects (NEO).")

# -------------------------
# Helper utilities
# -------------------------
def safe_numeric_cols(dataframe):
    return dataframe.select_dtypes(include=[np.number]).columns.tolist()

def safe_stats_for_input(series):
    s = series.dropna()
    s = s[np.isfinite(s)]
    if s.empty:
        return 0.0, 1.0, 0.5
    lo = float(s.min())
    hi = float(s.max())
    med = float(s.median())
    if lo == hi:
        hi = lo + 1.0
    return lo, hi, med

# -------------------------
# EDA
# -------------------------
if option == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    if df is None:
        st.error("Default CSV not found in project root. Upload a CSV in the file uploader below to explore.")
        uploaded = st.file_uploader("Upload dataset for EDA", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.stop()

    st.write("Dataset Dimensions:", df.shape[0], "rows ×", df.shape[1], "columns")
    st.write("Memory footprint:", round(df.memory_usage(deep=True).sum() / 1024**2, 2), "MB")
    completeness = round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    st.write("Data completeness:", f"{completeness}%")

    st.subheader("Preview")
    st.dataframe(df.head())

    # Column-wise summary
    mem_per_col = (df.memory_usage(deep=True) / 1024**2).round(4)
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Nulls": df.isnull().sum().values,
        "Null_%": (df.isnull().sum().values / len(df) * 100).round(2),
        "Unique": [df[c].nunique() for c in df.columns],
        "Memory_MB": mem_per_col.values
    })
    st.markdown("### Column summary")
    st.dataframe(info_df)

    # Numeric features and descriptive stats
    numeric_features = safe_numeric_cols(df)
    if numeric_features:
        st.markdown("### Descriptive statistics (numeric features)")
        desc = df[numeric_features].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        desc['variance'] = df[numeric_features].var()
        desc['skew'] = df[numeric_features].skew()
        desc['kurtosis'] = df[numeric_features].kurtosis()
        desc['cv_%'] = np.where((desc['mean'].abs() > 1e-10) & np.isfinite(desc['mean']) & np.isfinite(desc['std']),
                                (desc['std'] / desc['mean'].abs() * 100).round(2), np.nan)
        st.dataframe(desc)

        # Univariate distributions
        st.markdown("### Univariate distributions")
        n = len(numeric_features)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(16, nrows * 3.5))
        gs = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.3)

        for idx, col in enumerate(numeric_features):
            r = idx // ncols
            c = idx % ncols
            ax = fig.add_subplot(gs[r, c])
            data = df[col].dropna()
            data = data[np.isfinite(data)]
            if data.empty:
                ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
                ax.set_title(col)
                continue

            bins = min(40, max(10, len(data)//10)) if len(data) > 20 else 10
            ax.hist(data, bins=bins, density=True, alpha=0.6, edgecolor='black')
            if len(data) > 1 and data.std() > 0:
                kde_x = np.linspace(data.min(), data.max(), 200)
                try:
                    kde = gaussian_kde(data)
                    ax.plot(kde_x, kde(kde_x), color='red', linewidth=1.7, label='KDE')
                except Exception:
                    pass

            mu = data.mean()
            med = data.median()
            ax.axvline(mu, color='darkgreen', linestyle='--', label=f"Mean {mu:.3f}")
            ax.axvline(med, color='darkblue', linestyle='--', label=f"Median {med:.3f}")
            ax.set_title(col)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        # Outlier summary (IQR)
        st.markdown("### Outlier summary (IQR)")
        outlier_rows = []
        for col in numeric_features:
            series = df[col].dropna()
            series = series[np.isfinite(series)]
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                outlier_rows.append({"feature": col, "mild": 0, "extreme": 0, "note": "no variance"})
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            lower_e = q1 - 3 * iqr
            upper_e = q3 + 3 * iqr
            mild = series[(series < lower) | (series > upper)]
            extreme = series[(series < lower_e) | (series > upper_e)]
            outlier_rows.append({
                "feature": col,
                "mild": int(len(mild)),
                "mild_%": f"{len(mild)/len(series)*100:.2f}%",
                "extreme": int(len(extreme)),
                "extreme_%": f"{len(extreme)/len(series)*100:.2f}%"
            })
        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows))

        # Correlation heatmap
        st.markdown("### Correlation heatmap")
        corr = df[numeric_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns detected for EDA.")

# -------------------------
# Prediction
# -------------------------
elif option == "Prediction":
    st.header("Asteroid Hazard Prediction")

    uploaded = st.file_uploader("Upload asteroid CSV for training/prediction", type=["csv"])
    if uploaded is None and df is None:
        st.warning("Please upload a CSV dataset to train & predict.")
        st.stop()

    asteroid_data = pd.read_csv(uploaded) if uploaded is not None else df.copy()

    st.write("Dataset loaded:", asteroid_data.shape)
    st.dataframe(asteroid_data.head())

    # Automatically detect potential hazard column(s)
    candidate_targets = [c for c in asteroid_data.columns if "hazard" in c.lower() or "danger" in c.lower() or "potentially" in c.lower()]

    if candidate_targets:
        target_col = st.selectbox("Select target column (auto-detected)", candidate_targets, index=0)
    else:
        st.write("No hazard-like column auto-detected.")
        # fallback: let user pick
        target_col = st.selectbox("Select the target (label) column", asteroid_data.columns.tolist())

    if not target_col:
        st.error("No target column selected.")
        st.stop()

    # Choose features: prefer known names, else numeric features
    preferred = [
        "absolute_magnitude_h",
        "estimated_diameter_min_km",
        "estimated_diameter_max_km",
        "relative_velocity_km_s",
        "miss_distance_au",
        "orbiting_body"
    ]
    features_available = [f for f in preferred if f in asteroid_data.columns]
    if not features_available:
        # fallback to numeric columns excluding target
        num_cols = safe_numeric_cols(asteroid_data)
        features_available = [c for c in num_cols if c != target_col][:6]

    st.write("Using features:", features_available)
    if "orbiting_body" in features_available:
        # label encode orbiting body
        asteroid_data["orbiting_body_encoded"] = asteroid_data["orbiting_body"].astype(str)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        asteroid_data["orbiting_body_encoded"] = le.fit_transform(asteroid_data["orbiting_body_encoded"])
        features_available = [f for f in features_available if f != "orbiting_body"] + ["orbiting_body_encoded"]

    X = asteroid_data[features_available].copy()
    y = asteroid_data[target_col].copy()

    # Clean X: impute numeric with median
    for col in X.columns:
        if X[col].dtype.kind in 'biufc':
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("missing")

    # Encode y if non-numeric
    if y.dtype.kind not in 'biufc':
        y = y.astype(str)
        from sklearn.preprocessing import LabelEncoder
        y_enc = LabelEncoder()
        y = y_enc.fit_transform(y)
    else:
        # ensure no NaN
        if y.isnull().any():
            y = y.fillna(0).astype(int)

    # Train/test split
    from sklearn.model_selection import train_test_split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Train a RandomForest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    try:
        model.fit(X_train, y_train)
        st.success("Model trained successfully!")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.metric("Test Accuracy", f"{acc:.2%}")

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature importance")
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": features_available, "importance": importances}).sort_values("importance", ascending=False)
    st.dataframe(fi)
    st.bar_chart(fi.set_index("feature"))

    # Quick prediction UI
    st.markdown("### Try a quick prediction")
    example = {}
    for feat in features_available:
        # only numeric features here (we encoded non-numeric above)
        if X[feat].dtype.kind in 'biufc':
            lo, hi, med = safe_stats_for_input(X[feat])
            example_val = st.number_input(f"{feat}", value=float(med), min_value=float(lo), max_value=float(hi))
            example[feat] = [example_val]
        else:
            # fallback to text input
            example[feat] = [st.text_input(feat, value=str(X[feat].iloc[0]))]

    if st.button("Predict"):
        ex_df = pd.DataFrame(example)
        # ensure column order match
