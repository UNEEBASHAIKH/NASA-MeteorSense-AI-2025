import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

# ----------------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
colors = sns.color_palette("husl", 12)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
# -----------------------------------------------------------------------------------
df = pd.read_csv("NASA Near-Earth ObjectCleaned(NEO).csv")
# Sidebar menu
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an option:", ["EDA", "Prediction"])

st.markdown('# NASA MeteorSense AI')
st.write('In this project, an analysis would be done on NASA MeteorSense AI dataset.')
st.write('We will be predicting whether a meteor is potentially hazardous or not.')

if option == "EDA":
    st.markdown("## EXPLORATORY DATA ANALYSIS")

    st.write("Dataset Dimensions:", df.shape[0], "observations ×", df.shape[1], "features")
    st.write("Memory Footprint:", round(df.memory_usage(deep=True).sum() / 1024**2, 2), "MB")
    st.write("Data Completeness:", round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2), "%")

    st.write("Below is a visual representation of the dataset:")
    st.dataframe(df.head())
    # -----------------------------------------------------------------------------------------------
    info_df = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes.values,
    'Null_Count': df.isnull().sum().values,
    'Null_Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
    'Unique_Values': [df[col].nunique() for col in df.columns],
    'Memory_MB': (df.memory_usage(deep=True).values[1:] / 1024**2).round(3)})
    st.markdown("### Column-wise Data Summary:")
    st.dataframe(info_df)
    # -----------------------------------------------------------------------------------------------
    # Numeric & categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### Comprehensive Descriptive Statistics")

    desc_stats = df[numeric_features].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc_stats['variance'] = df[numeric_features].var()
    desc_stats['skewness'] = df[numeric_features].skew()
    desc_stats['kurtosis'] = df[numeric_features].kurtosis()

    # Coefficient of Variation
    desc_stats['cv'] = np.where(
        (desc_stats['mean'].abs() > 1e-10) & np.isfinite(desc_stats['mean']) & np.isfinite(desc_stats['std']),
        (desc_stats['std'] / desc_stats['mean'].abs() * 100).round(2),
        np.nan
    )

    # Show table
    st.dataframe(desc_stats)
    # -----------------------------------------------------------------------------------------------


# Section header
    st.markdown("## UNIVARIATE ANALYSIS - DISTRIBUTIONS")

    # Create distribution plots
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, n_rows * 5))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    for idx, col in enumerate(numeric_features):
        row = idx // n_cols
        col_pos = idx % n_cols

        ax_main = fig.add_subplot(gs[row, col_pos])

        data = df[col].dropna()
        data = data[np.isfinite(data)]  # remove infinities

        if len(data) == 0:
            ax_main.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax_main.transAxes)
            ax_main.set_title(f'{col}', fontsize=12, fontweight='bold', pad=10)
            continue

        try:
            # Histogram
            n, bins, patches = ax_main.hist(
                data,
                bins=min(40, len(data)//10 if len(data) > 100 else 10),
                alpha=0.7, color='skyblue',
                edgecolor='black', linewidth=0.5, density=True
            )

            # KDE
            if len(data) > 1 and data.std() > 0:
                kde_x = np.linspace(data.min(), data.max(), 200)
                kde = gaussian_kde(data)
                ax_main.plot(kde_x, kde(kde_x), 'r-', linewidth=2.5, label='KDE', alpha=0.8)

            # Mean & Median
            mean_val = data.mean()
            median_val = data.median()
            ax_main.axvline(mean_val, color='darkgreen', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_val:.3f}', alpha=0.8)
            ax_main.axvline(median_val, color='darkblue', linestyle='--', linewidth=2,
                            label=f'Median: {median_val:.3f}', alpha=0.8)

            ax_main.set_title(f'{col}', fontsize=12, fontweight='bold', pad=10)
            ax_main.set_xlabel(col, fontweight='bold')
            ax_main.set_ylabel('Density', fontweight='bold')
            ax_main.legend(loc='upper right', fontsize=8)
            ax_main.grid(True, alpha=0.3)

            # Stats box
            stats_text = (f"μ = {data.mean():.4f}\n"
                        f"σ = {data.std():.4f}\n"
                        f"Skew = {data.skew():.3f}\n"
                        f"Kurt = {data.kurtosis():.3f}\n"
                        f"Range = [{data.min():.3f}, {data.max():.3f}]")

            ax_main.text(0.98, 0.97, stats_text, transform=ax_main.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'),
                        fontsize=8, family='monospace')
        except Exception as e:
            ax_main.text(0.5, 0.5, f'Error plotting: {str(e)[:30]}', ha='center', va='center',
                        transform=ax_main.transAxes, fontsize=8)
    # Show in Streamlit instead of saving
    st.pyplot(fig)

    # -----------------------------------------------------------------------------------------------
    st.markdown("## OUTLIER DETECTION & ANALYSIS")

    outlier_summary = []
    for col in numeric_features:
        try:
            data_clean = df[col].dropna()
            data_clean = data_clean[np.isfinite(data_clean)]  # remove infinite

            if len(data_clean) == 0:
                continue

            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                outlier_summary.append({
                    'Feature': col,
                    'Mild_Outliers': 0,
                    'Mild_Outliers_%': '0.00%',
                    'Extreme_Outliers': 0,
                    'Extreme_Outliers_%': '0.00%',
                    'Lower_Fence': f"{Q1:.4f}",
                    'Upper_Fence': f"{Q3:.4f}",
                    'Note': 'No variance'
                })
                continue

            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            # Extreme outliers (3*IQR)
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR

            mild_outliers = data_clean[(data_clean < lower_fence) | (data_clean > upper_fence)]
            extreme_outliers = data_clean[(data_clean < extreme_lower) | (data_clean > extreme_upper)]

            outlier_summary.append({
                'Feature': col,
                'Mild_Outliers': len(mild_outliers),
                'Mild_Outliers_%': f"{(len(mild_outliers)/len(data_clean)*100):.2f}%",
                'Extreme_Outliers': len(extreme_outliers),
                'Extreme_Outliers_%': f"{(len(extreme_outliers)/len(data_clean)*100):.2f}%",
                'Lower_Fence': f"{lower_fence:.4f}",
                'Upper_Fence': f"{upper_fence:.4f}"
            })
        except Exception as e:
            st.write(f"⚠ Warning: Could not compute outliers for {col}: {str(e)}")

    # Show summary table
    outlier_df = pd.DataFrame(outlier_summary)
    st.markdown("### Outlier Detection Summary (IQR Method)")
    st.dataframe(outlier_df)

    # Enhanced box + violin plots
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(numeric_features):
        ax = axes[idx]
        try:
            data_clean = df[col].dropna()
            data_clean = data_clean[np.isfinite(data_clean)]

            if len(data_clean) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col}', fontsize=11, fontweight='bold')
                continue

            # Violin plot
            parts = ax.violinplot([data_clean.values], positions=[0], widths=0.7,
                                showmeans=True, showextrema=True, showmedians=True)

            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)

            # Overlay box plot
            bp = ax.boxplot([data_clean.values], positions=[0], widths=0.3,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.8)

            ax.set_title(f'{col}', fontsize=11, fontweight='bold')
            ax.set_ylabel(col, fontweight='bold')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')

            # Outlier count box
            Q1, Q3 = data_clean.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_count = ((data_clean < Q1 - 1.5*IQR) | (data_clean > Q3 + 1.5*IQR)).sum()
                ax.text(0.5, 0.98, f'Outliers: {outlier_count}', transform=ax.transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=9, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)

    # Remove empty axes
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Violin & Box Plot Analysis - Outlier Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    # -----------------------------------------------------------------------------------------------
    # ==============================
    # CORRELATION & MULTICOLLINEARITY ANALYSIS
    # ==============================

    print("\n" + "="*80)
    print("CORRELATION & MULTICOLLINEARITY ANALYSIS")
    print("="*80 + "\n")

    corr_matrix = df[numeric_features].corr()

    # Create an enhanced correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Full correlation matrix (lower triangle only)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                vmin=-1, vmax=1, ax=ax1, annot_kws={'size': 8})
    ax1.set_title('Correlation Matrix (Lower Triangle)', fontsize=13, fontweight='bold', pad=15)

    # High correlation only (≥ 0.5)
    high_corr_matrix = corr_matrix.copy()
    high_corr_matrix[abs(high_corr_matrix) < 0.5] = 0
    sns.heatmap(high_corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                vmin=-1, vmax=1, ax=ax2, annot_kws={'size': 8})
    ax2.set_title('Strong Correlations Only (|r| ≥ 0.5)', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==============================
    # Correlation pairs analysis
    # ==============================
    print("\n--- Correlation Strength Classification ---\n")

    correlation_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            abs_corr = abs(corr_val)

            if abs_corr > 0.3:  # Show moderate and above
                if abs_corr >= 0.9:
                    strength = "Very Strong"
                elif abs_corr >= 0.7:
                    strength = "Strong"
                elif abs_corr >= 0.5:
                    strength = "Moderate"
                else:
                    strength = "Weak"

                correlation_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': f"{corr_val:.4f}",
                    'Abs_Correlation': f"{abs_corr:.4f}",
                    'Strength': strength,
                    'Type': 'Positive' if corr_val > 0 else 'Negative'
                })

    # if correlation_pairs:
    #     corr_df = pd.DataFrame(correlation_pairs).sort_values('Abs_Correlation', ascending=False)

    #     # Display as styled table
    #     display(corr_df.style.set_caption("Significant Correlation Pairs")
    #                     .set_table_styles([{'selector': 'caption',
    #                                         'props': [('color', 'black'),
    #                                                 ('font-size', '14px'),
    #                                                 ('font-weight', 'bold')]}]))
    # else:
    #     print("No significant correlations found (|r| > 0.3).")
    st.divider()
    # -------------------------------------------------------------------------------------------------------------------------------------------

    st.markdown("## 📊 CORRELATION & MULTICOLLINEARITY ANALYSIS")

    # Compute correlation matrix
    corr_matrix = df[numeric_features].corr()

    # --- Correlation Heatmaps ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Full correlation matrix (lower triangle only)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                vmin=-1, vmax=1, ax=ax1, annot_kws={'size': 8})
    ax1.set_title('Correlation Matrix (Lower Triangle)', fontsize=13, fontweight='bold', pad=15)

    # High correlation only (|r| ≥ 0.5)
    high_corr_matrix = corr_matrix.copy()
    high_corr_matrix[abs(high_corr_matrix) < 0.5] = 0
    sns.heatmap(high_corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                vmin=-1, vmax=1, ax=ax2, annot_kws={'size': 8})
    ax2.set_title('Strong Correlations Only (|r| ≥ 0.5)', fontsize=13, fontweight='bold', pad=15)

    st.pyplot(fig)

    # --- Correlation pairs analysis ---
    st.markdown("### 🔎 Correlation Strength Classification")

    correlation_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            abs_corr = abs(corr_val)

            if abs_corr > 0.3:  # Show moderate and above
                if abs_corr >= 0.9:
                    strength = "Very Strong"
                elif abs_corr >= 0.7:
                    strength = "Strong"
                elif abs_corr >= 0.5:
                    strength = "Moderate"
                else:
                    strength = "Weak"

                correlation_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': f"{corr_val:.4f}",
                    'Abs_Correlation': f"{abs_corr:.4f}",
                    'Strength': strength,
                    'Type': 'Positive' if corr_val > 0 else 'Negative'
                })

    if correlation_pairs:
        corr_df = pd.DataFrame(correlation_pairs).sort_values('Abs_Correlation', ascending=False)
        st.dataframe(corr_df)
    else:
        st.write("No significant correlations found.")

    st.divider()
    # -------------------------------------------------------------------------------------------------------------------------

    from scipy.stats import pearsonr

    st.subheader("📊 MULTIVARIATE RELATIONSHIP ANALYSIS")

    # Select top features by variance for pairplot
    top_variance_features = df[numeric_features].var().nlargest(min(6, len(numeric_features))).index.tolist()

    # Pairplot
    if len(top_variance_features) >= 2:
        st.write(f"Pairplot for top {len(top_variance_features)} features by variance")

        pairplot_data = df[top_variance_features].copy()
        g = sns.pairplot(pairplot_data,
                        diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.2},
                        diag_kws={'alpha': 0.7, 'linewidth': 1})

        g.fig.suptitle('Pairwise Relationships - Top Features by Variance',
                    fontsize=12, fontweight='bold', y=1.01)

        st.pyplot(g)

    # Scatter Matrix with regression lines
    if len(top_variance_features) >= 2:
        fig, axes = plt.subplots(len(top_variance_features)-1, len(top_variance_features)-1,
                                figsize=(10, 8))

        colors = sns.color_palette("Set2", len(top_variance_features))

        for i in range(len(top_variance_features)-1):
            for j in range(len(top_variance_features)-1):
                ax = axes[i, j] if len(top_variance_features) > 2 else axes

                if j < i:
                    ax.axis('off')
                elif j == i:
                    df[top_variance_features[i]].hist(bins=30, ax=ax, color=colors[i], edgecolor='black')
                    ax.set_ylabel('')
                else:
                    ax.scatter(df[top_variance_features[j+1]], df[top_variance_features[i]],
                            alpha=0.5, s=20, c=[colors[i]])

                    # Regression line
                    z = np.polyfit(df[top_variance_features[j+1]].dropna(),
                                df[top_variance_features[i]].dropna(), 1)
                    p = np.poly1d(z)
                    ax.plot(df[top_variance_features[j+1]].sort_values(),
                        p(df[top_variance_features[j+1]].sort_values()),
                        "r--", linewidth=2, alpha=0.8)

                    # Correlation
                    valid_data = df[[top_variance_features[j+1], top_variance_features[i]]].dropna()
                    if len(valid_data) > 0:
                        r, p_val = pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
                        ax.text(0.05, 0.95, f'r={r:.3f}', transform=ax.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if i == 0:
                    ax.set_title(top_variance_features[j+1] if j != i else top_variance_features[i],
                                fontsize=10, fontweight='bold')
                if j == 0 or (j == i and i > 0):
                    ax.set_ylabel(top_variance_features[i], fontsize=10, fontweight='bold')

        plt.suptitle('Scatter Matrix with Regression Lines', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        st.pyplot(fig)

    st.divider()
    # --------------------------------------------------------------------------------------------------------------------------------------
    st.subheader("⏳ TEMPORAL ANALYSIS")
    
    if 'Close Approach Date' in df.columns:
        # Convert to datetime
        df_time = df.copy()
        df_time['Close Approach Date'] = pd.to_datetime(df_time['Close Approach Date'], errors='coerce')
        df_time = df_time.dropna(subset=['Close Approach Date']).sort_values('Close Approach Date')
        
        # Extract features
        df_time['Year'] = df_time['Close Approach Date'].dt.year
        df_time['Month'] = df_time['Close Approach Date'].dt.month
        df_time['Quarter'] = df_time['Close Approach Date'].dt.quarter
        df_time['Day_of_Week'] = df_time['Close Approach Date'].dt.dayofweek
        df_time['Week'] = df_time['Close Approach Date'].dt.isocalendar().week
        
        # Monthly statistics table
        monthly_stats = df_time.groupby(df_time['Close Approach Date'].dt.to_period('M')).agg({
            'Asteroid ID': 'count',
            'Distance from Earth (AU)': ['mean', 'min', 'max'],
            'Relative Velocity (km/s)': ['mean', 'std']
        })
        monthly_stats.columns = ['Count', 'Avg_Distance', 'Min_Distance', 'Max_Distance', 'Avg_Velocity', 'Std_Velocity']
        
        st.markdown("### 📅 Monthly Temporal Statistics")
        st.dataframe(monthly_stats.head(12))
        
        # === Plot 1: Time series of asteroid counts ===
        st.markdown("#### Asteroid Close Approaches Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        monthly_counts = df_time.groupby(df_time['Close Approach Date'].dt.to_period('M')).size()
        monthly_counts.plot(kind='line', ax=ax, color='steelblue', linewidth=2.5, marker='o')
        ax.fill_between(range(len(monthly_counts)), monthly_counts.values, alpha=0.3, color='steelblue')
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Asteroids")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # === Plot 2: Seasonal pattern ===
        st.markdown("#### Seasonal Pattern - Asteroids by Month")
        fig, ax = plt.subplots(figsize=(8, 4))
        month_counts = df_time.groupby('Month').size()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        ax.bar(range(1, 13), month_counts.reindex(range(1, 13), fill_value=0),
               color=sns.color_palette("Set2", 12), edgecolor="black")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names, rotation=45)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # === Plot 3: Weekly pattern ===
        st.markdown("#### Weekly Pattern - Asteroids by Day of Week")
        fig, ax = plt.subplots(figsize=(8, 4))
        dow_counts = df_time.groupby('Day_of_Week').size()
        dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        ax.bar(range(7), dow_counts.reindex(range(7), fill_value=0),
               color=sns.color_palette("Paired", 7), edgecolor="black")
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_names)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # === Plot 4: Distance over time ===
        st.markdown("#### Distance from Earth Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        df_time_sample = df_time.sample(min(1000, len(df_time)))
        sc = ax.scatter(df_time_sample['Close Approach Date'],
                        df_time_sample['Distance from Earth (AU)'],
                        c=df_time_sample['Relative Velocity (km/s)'],
                        cmap="plasma", alpha=0.6, edgecolors="black", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="Velocity (km/s)")
        ax.set_ylabel("Distance (AU)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # === Plot 5: Cumulative count ===
        st.markdown("#### Cumulative Asteroid Approaches")
        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative = monthly_counts.cumsum()
        ax.plot(cumulative.index.astype(str), cumulative.values,
                color="darkgreen", linewidth=3, marker="o")
        ax.set_ylabel("Cumulative Count")
        ax.set_xlabel("Month")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    else:
        st.error("'Close Approach Date' column not found in the dataset!")
    st.divider()
# ---------------------------------------------------------------------------------------------------------------------------------------




elif option == "Prediction":
    st.markdown("### 🚀 Asteroid Hazard Prediction - Smart Model Training")

    # --- Step 0: Package Loading ---
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.utils import class_weight
        from sklearn.preprocessing import LabelEncoder
        import seaborn as sns
        import matplotlib.pyplot as plt
        import joblib
        st.success("✅ Basic packages loaded successfully!")
    except ImportError as e:
        st.error(f"❌ Missing package: {e}")
        st.info("Install with: pip install pandas numpy scikit-learn seaborn matplotlib joblib")

    # --- Step 1: Load Data ---
    st.subheader("📊 Step 1: Load Asteroid Data")
    uploaded_file = st.file_uploader("Upload Asteroid CSV", type=["csv"])
    if uploaded_file is not None:
        asteroid_data = pd.read_csv(uploaded_file)
        st.write(f"✅ Data loaded with **{len(asteroid_data)} rows** and **{len(asteroid_data.columns)} columns**")
        st.dataframe(asteroid_data.head())
    else:
        st.warning("Please upload a dataset to continue 🚀")
        st.stop()

    # --- Step 2: Identify Hazard Column ---
    danger_columns = [col for col in asteroid_data.columns if "hazard" in col.lower()]
    if danger_columns:
        danger_column = danger_columns[0]
        st.success(f"✅ Found hazard target column: `{danger_column}`")
    else:
        st.error("❌ Could not find hazard column automatically. Please rename your dataset properly.")
        st.stop()

    # --- Step 3: Select Features ---
    possible_features = [
        "absolute_magnitude_h",
        "estimated_diameter_min_km",
        "estimated_diameter_max_km",
        "relative_velocity_km_s",
        "miss_distance_au",
        "orbiting_body"
    ]
    features_we_have = [f for f in possible_features if f in asteroid_data.columns]

    st.write("✅ Using features:", features_we_have)

    if "orbiting_body" in features_we_have:
        encoder = LabelEncoder()
        asteroid_data["orbiting_body_encoded"] = encoder.fit_transform(asteroid_data["orbiting_body"])
        features_we_have.remove("orbiting_body")
        features_we_have.append("orbiting_body_encoded")

    X = asteroid_data[features_we_have]
    y = asteroid_data[danger_column].astype(int)

    # --- Step 4: Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write(f"📚 Training set: {len(X_train)} | 🧪 Testing set: {len(X_test)}")

    # --- Step 5: Train Model ---
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    model.fit(X_train, y_train)
    st.success("✅ Model trained successfully!")

    # --- Step 6: Evaluate Model ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Safe', 'Dangerous'],
                yticklabels=['Safe', 'Dangerous'],
                ax=ax)
    st.pyplot(fig)

    # --- Step 7: Feature Importance ---
    st.subheader("🧠 Feature Importance")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": features_we_have, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)

    st.bar_chart(feat_imp.set_index("Feature"))

    # --- Step 8: Quick Prediction ---
    st.subheader("🔮 Try a Quick Prediction")
    example = {}
    for feature in features_we_have:
        val = st.number_input(f"Enter value for {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].median()))
        example[feature] = [val]

    if st.button("Predict Asteroid Hazard 🚀"):
        example_df = pd.DataFrame(example)
        pred = model.predict(example_df)[0]
        prob = model.predict_proba(example_df)[0][1]
        if pred == 1:
            st.error(f"🚨 Dangerous asteroid detected! (Confidence {prob:.1%})")
        else:
            st.success(f"🟢 Safe asteroid (Confidence {1-prob:.1%})")
