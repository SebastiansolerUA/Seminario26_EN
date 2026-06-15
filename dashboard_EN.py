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
    - Language (Português, English, Español)
    - Category
    - Web chatbot response
    - AI4Life chatbot response
    - Manual search response (ground truth)
    """)

# Process when file is uploaded
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file)
    
    # Auto-detect columns
    col_web = [c for c in df.columns if 'web.ua.pt' in c.lower() or 'extraida' in c.lower() or 'web' in c.lower()][0]
    col_ai4 = [c for c in df.columns if 'ai4life' in c.lower()][0]
    col_manual = [c for c in df.columns if 'manual' in c.lower()][0]
    
    # Calculate absolute errors
    df['Error_Web'] = abs(df[col_web] - df[col_manual])
    df['Error_AI4'] = abs(df[col_ai4] - df[col_manual])
    
    # Success message
    st.success(f"✅ File loaded successfully: {len(df)} records")
    
    # ========== DATA PREVIEW ==========
    st.subheader("📋 Data Preview")
    st.dataframe(df[['Language', 'Category', col_web, col_ai4, col_manual, 'Error_Web', 'Error_AI4']].head(10))
    
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
    box_data = [df['Error_Web'], df['Error_AI4']]
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
    if 'Language' in df.columns:
        st.subheader("🌐 Analysis by Language")
        
        languages = df['Language'].unique()
        
        fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Web by language
        web_data_lang = [df[df['Language'] == lang]['Error_Web'].values for lang in languages]
        axes[0].boxplot(web_data_lang, labels=languages, patch_artist=True)
        axes[0].set_title('Web System - Error by Language')
        axes[0].set_ylabel('Absolute Error')
        axes[0].grid(True, alpha=0.3)
        
        # AI4 by language
        ai4_data_lang = [df[df['Language'] == lang]['Error_AI4'].values for lang in languages]
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
            sub = df[df['Language'] == lang]
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
    if 'Category' in df.columns:
        st.subheader("📚 Analysis by Category")
        
        categories = df['Category'].unique()
        
        fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Web by category
        web_data_cat = [df[df['Category'] == cat]['Error_Web'].values for cat in categories]
        axes[0].boxplot(web_data_cat, labels=categories, patch_artist=True)
        axes[0].set_title('Web System - Error by Category')
        axes[0].set_ylabel('Absolute Error')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # AI4 by category
        ai4_data_cat = [df[df['Category'] == cat]['Error_AI4'].values for cat in categories]
        axes[1].boxplot(ai4_data_cat, labels=categories, patch_artist=True)
        axes[1].set_title('AI4Life System - Error by Category')
        axes[1].set_ylabel('Absolute Error')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig4)
    
    # ========== STATISTICAL TESTS ==========
    st.subheader("🔬 Statistical Tests")
    
    # Mann-Whitney U test
    U, p_mw = mannwhitneyu(df['Error_Web'], df['Error_AI4'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Web vs AI4Life Comparison**")
        st.write(f"- U statistic: {U:.0f}")
        st.write(f"- p-value: {p_mw:.4f}")
        if p_mw < 0.05:
            st.success("✅ Significant difference between systems")
        else:
            st.info("❌ No significant difference")
    
    # Kruskal-Wallis by language
    if 'Language' in df.columns and len(df['Language'].unique()) > 1:
        with col2:
            st.write("**Comparison Between Languages (Web)**")
            groups = [df[df['Language'] == lang]['Error_Web'].values for lang in df['Language'].unique()]
            H, p_kw = kruskal(*groups)
            st.write(f"- H statistic: {H:.3f}")
            st.write(f"- p-value: {p_kw:.4f}")
            if p_kw < 0.05:
                st.success("✅ Significant difference between languages")
            else:
                st.info("❌ No significant difference between languages")
    
    # ========== SUMMARY TABLE ==========
    st.subheader("📋 Descriptive Statistics Summary")
    
    summary = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Deviation', 'Minimum', 'Maximum', 'Q1 (25%)', 'Q3 (75%)', 'Exact Hits (%)'],
        'Web': [
            f"{df['Error_Web'].mean():.2f}",
            f"{df['Error_Web'].median():.1f}",
            f"{df['Error_Web'].std():.2f}",
            f"{df['Error_Web'].min():.0f}",
            f"{df['Error_Web'].max():.0f}",
            f"{df['Error_Web'].quantile(0.25):.1f}",
            f"{df['Error_Web'].quantile(0.75):.1f}",
            f"{(df['Error_Web'] == 0).sum()/len(df)*100:.1f}%"
        ],
        'AI4Life': [
            f"{df['Error_AI4'].mean():.2f}",
            f"{df['Error_AI4'].median():.1f}",
            f"{df['Error_AI4'].std():.2f}",
            f"{df['Error_AI4'].min():.0f}",
            f"{df['Error_AI4'].max():.0f}",
            f"{df['Error_AI4'].quantile(0.25):.1f}",
            f"{df['Error_AI4'].quantile(0.75):.1f}",
            f"{(df['Error_AI4'] == 0).sum()/len(df)*100:.1f}%"
        ]
    })
    st.table(summary)
    
    # ========== FULL DATA ==========
    with st.expander("📊 View all data (expand)"):
        st.dataframe(df)
    
    # ========== DOWNLOAD BUTTON ==========
    csv = df.to_csv(index=False).encode('utf-8')
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
        - `Language` (Português, English, Español)
        - `Category` (question type)
        - `Respuesta extraida del chatbot https://salina.web.ua.pt/` (numeric value)
        - `Respuesta extraida del chatbot https://salina.ai4life.uk/` (numeric value)
        - `Respuesta busqueda manual` (ground truth value)
        
        **Example format:**
        | Language | Category | Web response | AI4Life response | Manual response |
        |----------|----------|--------------|------------------|-----------------|
        | Spanish | Book Search | 25 | 30 | 28 |
        """)

# Footer
st.markdown("---")
st.caption("SALiNA Evaluation Dashboard | System Performance Analysis | English Version")