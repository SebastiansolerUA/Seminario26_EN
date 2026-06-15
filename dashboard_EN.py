import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import numpy as np
import re

# Page configuration
st.set_page_config(
    page_title="SALiNA - Evaluation Dashboard", 
    layout="wide"
)

# Main title
st.title("📊 SALiNA - Performance Evaluation Dashboard")
st.markdown("---")

# Function to clean numeric values
def clean_numeric(value):
    """Convert various formats to numeric, handling errors gracefully"""
    if pd.isna(value):
        return np.nan
    
    # Convert to string
    val_str = str(value).strip()
    
    # If empty string
    if val_str == '':
        return np.nan
    
    # Remove common non-numeric characters
    # Keep: digits, decimal point, minus sign
    cleaned = re.sub(r'[^\d.-]', '', val_str)
    
    # Handle multiple decimal points (keep only first)
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    
    # Remove trailing decimal point
    if cleaned.endswith('.'):
        cleaned = cleaned[:-1]
    
    # Convert to float
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Excel file", 
        type=['xlsx', 'xls'],
        help="Select the file with SALiNA responses"
    )
    
    st.markdown("---")
    st.caption("""
    **Required columns:**
    - Idioma (Português, English, Español)
    - Categoría
    - Web response column
    - AI4Life response column
    - Manual response column
    """)

# Process when file is uploaded
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file)
    
    st.write("**Columns found:**", list(df.columns))
    
    # Define column names (based on your data)
    col_language = 'Idioma'
    col_category = 'Categoría'
    col_web_original = 'Respuesta extraida del chatbot https://salina.web.ua.pt/'
    col_ai4_original = 'Respuesta extraida del chatbot https://salina.ai4life.uk/'
    col_manual_original = 'Respuesta busqueda manual'
    
    # Check if columns exist
    missing_cols = []
    for col in [col_language, col_category, col_web_original, col_ai4_original, col_manual_original]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()
    
    # Show sample of original values before cleaning
    st.subheader("🔍 Original Data Sample (Before Cleaning)")
    sample_df = df[[col_web_original, col_ai4_original, col_manual_original]].head(10)
    st.dataframe(sample_df)
    
    # Clean numeric columns
    df['web_cleaned'] = df[col_web_original].apply(clean_numeric)
    df['ai4_cleaned'] = df[col_ai4_original].apply(clean_numeric)
    df['manual_cleaned'] = df[col_manual_original].apply(clean_numeric)
    
    # Show rows with invalid values
    invalid_web = df[df['web_cleaned'].isna() & df[col_web_original].notna()]
    invalid_ai4 = df[df['ai4_cleaned'].isna() & df[col_ai4_original].notna()]
    invalid_manual = df[df['manual_cleaned'].isna() & df[col_manual_original].notna()]
    
    if len(invalid_web) > 0:
        st.warning(f"⚠️ {len(invalid_web)} rows have non-numeric values in Web column")
        with st.expander("View problematic Web values"):
            st.dataframe(invalid_web[[col_web_original, 'web_cleaned']].head(20))
    
    if len(invalid_ai4) > 0:
        st.warning(f"⚠️ {len(invalid_ai4)} rows have non-numeric values in AI4Life column")
        with st.expander("View problematic AI4Life values"):
            st.dataframe(invalid_ai4[[col_ai4_original, 'ai4_cleaned']].head(20))
    
    if len(invalid_manual) > 0:
        st.warning(f"⚠️ {len(invalid_manual)} rows have non-numeric values in Manual column")
        with st.expander("View problematic Manual values"):
            st.dataframe(invalid_manual[[col_manual_original, 'manual_cleaned']].head(20))
    
    # Remove rows with NaN values
    before = len(df)
    df = df.dropna(subset=['web_cleaned', 'ai4_cleaned', 'manual_cleaned'])
    after = len(df)
    
    if after < before:
        st.warning(f"⚠️ {before - after} rows were removed due to non-numeric values")
    
    if len(df) == 0:
        st.error("❌ No valid numeric data found. Please check the problematic values shown above.")
        st.stop()
    
    # Calculate absolute errors
    df['Error_Web'] = abs(df['web_cleaned'] - df['manual_cleaned'])
    df['Error_AI4'] = abs(df['ai4_cleaned'] - df['manual_cleaned'])
    
    # Success message
    st.success(f"✅ File loaded successfully: {len(df)} valid records")
    
    # ========== DATA PREVIEW ==========
    st.subheader("📋 Data Preview (Cleaned)")
    preview_cols = [col_language, col_category, 'web_cleaned', 'ai4_cleaned', 'manual_cleaned', 'Error_Web', 'Error_AI4']
    st.dataframe(df[preview_cols].head(10))
    
    # ========== KEY METRICS ==========
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Error - Web", f"{df['Error_Web'].mean():.2f}")
    with col2:
        st.metric("Mean Error - AI4Life", f"{df['Error_AI4'].mean():.2f}")
    with col3:
        exact_hits_web = (df['Error_Web'] == 0).sum()
        st.metric("Exact Hits - Web", f"{exact_hits_web}/{len(df)} ({exact_hits_web/len(df)*100:.0f}%)")
    with col4:
        exact_hits_ai4 = (df['Error_AI4'] == 0).sum()
        st.metric("Exact Hits - AI4Life", f"{exact_hits_ai4}/{len(df)} ({exact_hits_ai4/len(df)*100:.0f}%)")
    
    # ========== HISTOGRAM ==========
    st.subheader("📈 Error Distribution")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    df['Error_Web'].hist(bins=20, alpha=0.7, label='Web System', color='#1565C0', edgecolor='black')
    df['Error_AI4'].hist(bins=20, alpha=0.7, label='AI4Life System', color='#C62828', edgecolor='black')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution by System')
    ax1.legend()
    st.pyplot(fig1)
    
    # ========== BOXPLOT ==========
    st.subheader("📊 Error Comparison")
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    box_data = [df['Error_Web'].dropna(), df['Error_AI4'].dropna()]
    bp = ax2.boxplot(box_data, labels=['Web', 'AI4Life'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#1565C0')
    bp['boxes'][1].set_facecolor('#C62828')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error Comparison Between Systems')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # ========== ANALYSIS BY LANGUAGE ==========
    if col_language in df.columns:
        st.subheader("🌐 Analysis by Language")
        
        languages = df[col_language].unique()
        
        fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Web by language
        web_data_lang = [df[df[col_language] == lang]['Error_Web'].dropna().values for lang in languages]
        axes[0].boxplot(web_data_lang, labels=languages, patch_artist=True)
        axes[0].set_title('Web System - Error by Language')
        axes[0].set_ylabel('Absolute Error')
        axes[0].grid(True, alpha=0.3)
        
        # AI4 by language
        ai4_data_lang = [df[df[col_language] == lang]['Error_AI4'].dropna().values for lang in languages]
        axes[1].boxplot(ai4_data_lang, labels=languages, patch_artist=True)
        axes[1].set_title('AI4Life System - Error by Language')
        axes[1].set_ylabel('Absolute Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Statistics table by language
        st.write("**Statistics by Language:**")
        stats_lang = []
        for lang in languages:
            sub = df[df[col_language] == lang]
            stats_lang.append({
                'Language': lang,
                'Count': len(sub),
                'Mean Error Web': round(sub['Error_Web'].mean(), 2),
                'Mean Error AI4Life': round(sub['Error_AI4'].mean(), 2),
                'Median Web': sub['Error_Web'].median(),
                'Median AI4Life': sub['Error_AI4'].median()
            })
        st.dataframe(pd.DataFrame(stats_lang))
    
    # ========== STATISTICAL TESTS ==========
    st.subheader("🔬 Statistical Tests")
    
    # Mann-Whitney U test (Web vs AI4)
    stat, p_value = mannwhitneyu(df['Error_Web'], df['Error_AI4'], alternative='two-sided')
    
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        st.write("**Mann-Whitney U Test (Web vs AI4Life):**")
        st.write(f"- U statistic: {stat:.2f}")
        st.write(f"- P-value: {p_value:.4f}")
        if p_value < 0.05:
            st.success("✅ Significant difference between systems (p < 0.05)")
        else:
            st.info("❌ No significant difference between systems (p ≥ 0.05)")
    
    # Kruskal-Wallis test by language
    if col_language in df.columns and len(languages) >= 3:
        groups_web = [df[df[col_language] == lang]['Error_Web'].dropna().values for lang in languages]
        groups_ai4 = [df[df[col_language] == lang]['Error_AI4'].dropna().values for lang in languages]
        
        h_stat_web, p_kw_web = kruskal(*groups_web)
        h_stat_ai4, p_kw_ai4 = kruskal(*groups_ai4)
        
        with col_test2:
            st.write("**Kruskal-Wallis Test (by Language):**")
            st.write(f"Web System - H: {h_stat_web:.2f}, p: {p_kw_web:.4f}")
            st.write(f"AI4Life System - H: {h_stat_ai4:.2f}, p: {p_kw_ai4:.4f}")
    
    # ========== SUMMARY TABLE ==========
    st.subheader("📋 Descriptive Statistics Summary")
    
    summary = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Deviation', 'Minimum', 'Maximum', 'Exact Hits (%)'],
        'Web': [
            f"{df['Error_Web'].mean():.2f}",
            f"{df['Error_Web'].median():.1f}",
            f"{df['Error_Web'].std():.2f}",
            f"{df['Error_Web'].min():.0f}",
            f"{df['Error_Web'].max():.0f}",
            f"{(df['Error_Web'] == 0).sum()/len(df)*100:.1f}%"
        ],
        'AI4Life': [
            f"{df['Error_AI4'].mean():.2f}",
            f"{df['Error_AI4'].median():.1f}",
            f"{df['Error_AI4'].std():.2f}",
            f"{df['Error_AI4'].min():.0f}",
            f"{df['Error_AI4'].max():.0f}",
            f"{(df['Error_AI4'] == 0).sum()/len(df)*100:.1f}%"
        ]
    })
    st.table(summary)
    
    # ========== DOWNLOAD BUTTON ==========
    st.subheader("📥 Export Results")
    
    # Prepare export dataframe
    export_df = df[[col_language, col_category, 'web_cleaned', 'ai4_cleaned', 'manual_cleaned', 'Error_Web', 'Error_AI4']].copy()
    export_df.columns = ['Language', 'Category', 'Web_Response', 'AI4Life_Response', 'Manual_Response', 'Error_Web', 'Error_AI4']
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download results as CSV",
        data=csv,
        file_name='analysis_results.csv',
        mime='text/csv',
    )

else:
    # Welcome message when no file
    st.info("👈 **Upload your Excel file** in the left panel to start the analysis")
    
    with st.expander("📖 Expected file format"):
        st.markdown("""
        **Required columns:**
        - `Idioma` (Português, English, Español)
        - `Categoría` (question type)
        - `Respuesta extraida del chatbot https://salina.web.ua.pt/` (numeric values)
        - `Respuesta extraida del chatbot https://salina.ai4life.uk/` (numeric values)
        - `Respuesta busqueda manual` (numeric values - ground truth)
        
        **Note:** The dashboard will automatically clean numeric values (removing commas, currency symbols, etc.)
        """)

# Footer
st.markdown("---")
st.caption("SALiNA Evaluation Dashboard | System Performance Analysis")
