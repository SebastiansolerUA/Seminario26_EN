import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import numpy as np

# Page configuration
st.set_page_config(
    page_title="SALiNA - Evaluation Dashboard", 
    layout="wide"
)

# Main title
st.title("📊 SALiNA - Performance Evaluation Dashboard")
st.markdown("---")

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
    - Web chatbot response (numeric)
    - AI4Life chatbot response (numeric)
    - Manual search response (numeric - ground truth)
    """)

# Function to identify numeric columns
def identify_column(df, patterns):
    for col in df.columns:
        for pattern in patterns:
            if pattern.lower() in col.lower():
                return col
    return None

# Process when file is uploaded
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file)
    
    # Show ALL column names to debug
    st.write("**Columns found:**", list(df.columns))
    
    # Identify columns (Spanish column names)
    col_language = 'Idioma' if 'Idioma' in df.columns else 'Language'
    col_category = 'Categoría' if 'Categoría' in df.columns else 'Category'
    
    # Identify numeric response columns
    col_web = identify_column(df, ['web', 'extraida', 'salina.web'])
    col_ai4 = identify_column(df, ['ai4', 'ai4life', 'salina.ai4life'])
    col_manual = identify_column(df, ['manual', 'busqueda', 'ground'])
    
    if not col_web or not col_ai4 or not col_manual:
        st.error("❌ Required columns not found")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Convert to numeric (force conversion, errors become NaN)
    df[col_web] = pd.to_numeric(df[col_web], errors='coerce')
    df[col_ai4] = pd.to_numeric(df[col_ai4], errors='coerce')
    df[col_manual] = pd.to_numeric(df[col_manual], errors='coerce')
    
    # Remove rows with NaN values
    before = len(df)
    df = df.dropna(subset=[col_web, col_ai4, col_manual])
    after = len(df)
    
    if after < before:
        st.warning(f"⚠️ {before - after} rows were removed due to non-numeric values")
    
    if len(df) == 0:
        st.error("❌ No valid numeric data found. Check your Excel file.")
        st.stop()
    
    # Calculate absolute errors
    df['Error_Web'] = abs(df[col_web] - df[col_manual])
    df['Error_AI4'] = abs(df[col_ai4] - df[col_manual])
    
    # Success message
    st.success(f"✅ File loaded successfully: {len(df)} valid records")
    
    # ========== DATA PREVIEW ==========
    st.subheader("📋 Data Preview")
    preview_cols = [col_language, col_category, col_web, col_ai4, col_manual, 'Error_Web', 'Error_AI4']
    preview_cols = [c for c in preview_cols if c in df.columns]
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
                'Mean Error Web': round(sub['Error_Web'].mean(), 2),
                'Mean Error AI4Life': round(sub['Error_AI4'].mean(), 2),
                'Median Web': sub['Error_Web'].median(),
                'Median AI4Life': sub['Error_AI4'].median(),
                'Records': len(sub)
            })
        st.dataframe(pd.DataFrame(stats_lang))
    
    # ========== ANALYSIS BY CATEGORY ==========
    if col_category in df.columns:
        st.subheader("📂 Analysis by Category")
        
        categories = df[col_category].unique()
        
        fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Web by category
        web_data_cat = [df[df[col_category] == cat]['Error_Web'].dropna().values for cat in categories]
        axes[0].boxplot(web_data_cat, labels=categories, patch_artist=True)
        axes[0].set_title('Web System - Error by Category')
        axes[0].set_ylabel('Absolute Error')
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # AI4 by category
        ai4_data_cat = [df[df[col_category] == cat]['Error_AI4'].dropna().values for cat in categories]
        axes[1].boxplot(ai4_data_cat, labels=categories, patch_artist=True)
        axes[1].set_title('AI4Life System - Error by Category')
        axes[1].set_ylabel('Absolute Error')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Statistics table by category
        st.write("**Statistics by Category:**")
        stats_cat = []
        for cat in categories:
            sub = df[df[col_category] == cat]
            stats_cat.append({
                'Category': cat,
                'Mean Error Web': round(sub['Error_Web'].mean(), 2),
                'Mean Error AI4Life': round(sub['Error_AI4'].mean(), 2),
                'Median Web': sub['Error_Web'].median(),
                'Median AI4Life': sub['Error_AI4'].median(),
                'Records': len(sub)
            })
        st.dataframe(pd.DataFrame(stats_cat))
    
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
        h_stat, p_kw = kruskal(*groups_web)
        
        with col_test2:
            st.write("**Kruskal-Wallis Test (Web System by Language):**")
            st.write(f"- H statistic: {h_stat:.2f}")
            st.write(f"- P-value: {p_kw:.4f}")
            if p_kw < 0.05:
                st.success("✅ Significant differences by language (p < 0.05)")
            else:
                st.info("❌ No significant differences by language (p ≥ 0.05)")
    
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
    
    # ========== CORRELATION ==========
    st.subheader("📈 Correlation Analysis")
    
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    ax5.scatter(df['Error_Web'], df['Error_AI4'], alpha=0.6, color='#7B1FA2', edgecolors='white', linewidth=0.5)
    ax5.plot([0, max(df['Error_Web'].max(), df['Error_AI4'].max())], 
             [0, max(df['Error_Web'].max(), df['Error_AI4'].max())], 
             'k--', alpha=0.5, label='Perfect correlation')
    ax5.set_xlabel('Web System Error')
    ax5.set_ylabel('AI4Life System Error')
    ax5.set_title('Error Correlation Between Systems')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)
    
    # Correlation coefficient
    corr = df['Error_Web'].corr(df['Error_AI4'])
    st.write(f"**Correlation coefficient:** {corr:.3f}")
    
    # ========== DOWNLOAD BUTTON ==========
    st.subheader("📥 Export Results")
    
    # Prepare export dataframe
    export_cols = [col_language, col_category, col_web, col_ai4, col_manual, 'Error_Web', 'Error_AI4']
    export_cols = [c for c in export_cols if c in df.columns]
    export_df = df[export_cols].copy()
    
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
        - `Idioma` or `Language` (Português, English, Español)
        - `Categoría` or `Category` (question type)
        - Column with web responses (must be numeric)
        - Column with AI4Life responses (must be numeric)
        - Column with manual responses (must be numeric - ground truth)
        
        **Example structure:**
        | Idioma   | Categoría | Respuesta Web | Respuesta AI4Life | Respuesta Manual |
        |----------|-----------|---------------|-------------------|------------------|
        | Português| Books     | 250           | 0                 | 151              |
        | English  | Books     | 528           | 91                | 411              |
        """)

# Footer
st.markdown("---")
st.caption("SALiNA Evaluation Dashboard | System Performance Analysis | Bilingual Support (English/Spanish)")
