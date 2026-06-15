import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, shapiro
import scikit_posthocs as sp

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SALiNA — Evaluation Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1565C0;
        margin-bottom: 0.5rem;
    }
    h1 { color: #1a237e; }
    h2 { color: #283593; border-bottom: 2px solid #e8eaf6; padding-bottom: 4px; }
    h3 { color: #3949ab; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WEB   = 'salina.web.ua.pt'
AI4   = 'salina.ai4life.uk'
LANGS = ['Português', 'English', 'Español']
CATS_QUANT = ['Búsqueda de libros', 'Búsqueda por autor']
CATS_QUAL  = ['Artículos científicos', 'Revistas y publicaciones', 'Metadatos del catálogo']
ALPHA = 0.05

SYS_PAL  = {WEB: '#1565C0', AI4: '#C62828'}
LANG_PAL = {'Português': '#4472C4', 'English': '#70AD47', 'Español': '#ED7D31'}
CAT_PAL  = {
    'Búsqueda de libros'       : '#7B1FA2',
    'Búsqueda por autor'       : '#00796B',
    'Artículos científicos'    : '#E65100',
    'Revistas y publicaciones' : '#0277BD',
    'Metadatos del catálogo'   : '#558B2F',
}

COL_EXT_WEB  = 'Respuesta extraida del chatbot https://salina.web.ua.pt/'
COL_EXT_AI4  = 'Respuesta extraida del chatbot https://salina.ai4life.uk/'
COL_RESP_WEB = 'Respuesta del chatbot https://salina.web.ua.pt/'
COL_RESP_AI4 = 'Respuesta del chatbot https://salina.ai4life.uk/'
COL_MAN      = 'Respuesta busqueda manual'
COL_Q        = 'Texto de la pregunta'

plt.rcParams.update({
    'figure.dpi'        : 120,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'font.size'         : 9,
})
sns.set_theme(style='whitegrid', font_scale=0.95)

# ── Statistical helper functions ──────────────────────────────────────────────
def epsilon_squared(H, k, n):
    return (H - k + 1) / (n - k)

def rank_biserial_r(U, n1, n2):
    return 1 - (2 * U / (n1 * n2))

def eff_eps(v):
    return 'large' if abs(v) >= 0.16 else ('medium' if abs(v) >= 0.04 else 'small')

def eff_r(v):
    return 'large' if abs(v) >= 0.5 else ('medium' if abs(v) >= 0.3 else 'small')

def sig_badge(p):
    if p < 0.001:
        return "🟢 p < 0.001 — Highly significant"
    elif p < 0.01:
        return "🟢 p < 0.01 — Significant"
    elif p < 0.05:
        return "🟡 p < 0.05 — Significant"
    else:
        return "🔴 p ≥ 0.05 — Not significant"

# ── Data loading and preparation ──────────────────────────────────────────────
@st.cache_data
def load_data(file):
    raw = pd.read_excel(file)

    # Quantitative section
    dq = raw[raw['Categoría'].isin(CATS_QUANT)].copy().reset_index(drop=True)
    dq = dq.rename(columns={
        'Idioma': 'language', 'Categoría': 'category',
        COL_EXT_WEB: 'web', COL_EXT_AI4: 'ai4', COL_MAN: 'manual',
    })
    dq['ae_web']  = (dq['web'] - dq['manual']).abs()
    dq['ae_ai4']  = (dq['ai4'] - dq['manual']).abs()
    dq['err_web'] = dq['web'] - dq['manual']
    dq['err_ai4'] = dq['ai4'] - dq['manual']

    # Qualitative section
    dc = raw[raw['Categoría'].isin(CATS_QUAL)].copy().reset_index(drop=True)
    dc = dc.rename(columns={
        'Idioma': 'language', 'Categoría': 'category',
        COL_RESP_WEB: 'resp_web', COL_RESP_AI4: 'resp_ai4', COL_Q: 'question',
    })
    NEG = [
        'não foi possível', 'não encontr', 'ocorreu um erro', 'lamentavelmente',
        'nenhum resultado', 'não há resultado', 'unable to', 'could not',
        'not found', 'no results', 'unfortunately', 'no encontr', 'no se pudo',
        'no hay resultado', 'lamentablemente', 'there are no', 'sem resultados',
    ]
    dc['neg_web'] = dc['resp_web'].apply(
        lambda t: any(p in str(t).lower() for p in NEG))
    dc['neg_ai4'] = dc['resp_ai4'].apply(
        lambda t: any(p in str(t).lower() for p in NEG))

    return dq, dc

# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        'https://i0.wp.com/cdcs.web.ua.pt/wp-content/uploads/2022/05/cropped-cropped-Picture13-1.png?w=968',
        width=220
    )
    st.markdown("### 📚 SALiNA Dashboard")
    st.markdown("*Multilingual RAG performance evaluation*")
    st.divider()

    file = st.file_uploader("📁 Upload Excel file", type=['xlsx'])

    if file:
        st.divider()
        st.markdown("### ⚙️ Filters")
        section = st.radio(
            "Analysis section",
            ["🔢 Quantitative (MAE)", "📝 Qualitative (LaBSE)", "📊 Executive Summary"],
            index=0
        )
        langs_sel = st.multiselect(
            "Languages", LANGS, default=LANGS
        )
        st.divider()
        st.caption(f"α = {ALPHA} | Non-parametric tests")

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN SCREEN
# ════════════════════════════════════════════════════════════════════════════════
if not file:
    st.title("📚 SALiNA — Multilingual Evaluation Dashboard")
    st.markdown("""
    This dashboard reproduces the statistical analysis from **Milestone 3 — Block 1**,
    evaluating the observable performance of the two SALiNA RAG system versions
    against queries in three languages.

    | System | URL |
    |---|---|
    | **salina.web** | https://salina.web.ua.pt/ |
    | **salina.ai4life** | https://salina.ai4life.uk/ |

    ---
    👈 **Upload the `Respuestas3.xlsx` file in the sidebar to begin.**
    """)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
dq, dc = load_data(file)
active_langs = [l for l in LANGS if l in langs_sel] or LANGS

# ════════════════════════════════════════════════════════════════════════════════
#  QUANTITATIVE SECTION
# ════════════════════════════════════════════════════════════════════════════════
if section == "🔢 Quantitative (MAE)":

    st.title("🔢 Quantitative Analysis — Mean Absolute Error (MAE)")
    st.markdown(
        "Categories: **Book Search** and **Author Search**. "
        "Metric: Absolute Error between the RAG response and the true catalogue value."
    )

    dq_f = dq[dq['language'].isin(active_langs)]

    # ── Global metrics ────────────────────────────────────────────────────────
    st.subheader("Global metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE salina.web",     f"{dq_f['ae_web'].mean():.2f}")
    c2.metric("MAE salina.ai4life", f"{dq_f['ae_ai4'].mean():.2f}")
    c3.metric("Exact matches web",
              f"{(dq_f['ae_web']==0).sum()}/{len(dq_f)}")
    c4.metric("Exact matches ai4life",
              f"{(dq_f['ae_ai4']==0).sum()}/{len(dq_f)}")

    st.divider()

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Descriptive stats",
        "🔵 True vs RAG",
        "📐 Bias",
        "H1 — Kruskal-Wallis",
        "H2 — Mann-Whitney"
    ])

    # ── Tab 1: Descriptive stats ──────────────────────────────────────────────
    with tab1:
        st.subheader("Descriptive statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**By language (mean MAE)**")
            tbl = dq_f.groupby('language')[['ae_web','ae_ai4']].agg(
                ['mean','median']).round(2)
            tbl.columns = ['Mean web', 'Median web', 'Mean ai4', 'Median ai4']
            st.dataframe(tbl.loc[[l for l in LANGS if l in active_langs]],
                         use_container_width=True)

        with col2:
            st.markdown("**By category (mean MAE)**")
            tbl2 = dq_f.groupby('category')[['ae_web','ae_ai4']].agg(
                ['mean','median']).round(2)
            tbl2.columns = ['Mean web', 'Median web', 'Mean ai4', 'Median ai4']
            st.dataframe(tbl2, use_container_width=True)

        st.markdown("**Absolute Error distribution**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (sn, col) in zip(axes, [(WEB,'ae_web'),(AI4,'ae_ai4')]):
            for lang in active_langs:
                sub = dq_f[dq_f['language']==lang][col]
                ax.hist(sub, bins=12, alpha=0.6, label=lang,
                        color=LANG_PAL[lang], edgecolor='white')
            ax.axvline(dq_f[col].median(), color=SYS_PAL[sn],
                       linestyle='--', linewidth=1.8,
                       label=f'Median={dq_f[col].median():.1f}')
            ax.set_xlabel('Absolute Error (AE)')
            ax.set_ylabel('Frequency')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            ax.legend(fontsize=8)
        fig.suptitle('AE distribution — right skew justifies non-parametric tests',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: True vs RAG ────────────────────────────────────────────────────
    with tab2:
        st.subheader("True catalogue value vs RAG response")
        st.caption("Points above the diagonal = overestimation | Below = underestimation")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        max_val = max(dq_f[['manual','web','ai4']].max()) * 1.05

        for ax, (sn, col) in zip(axes, [(WEB,'web'),(AI4,'ai4')]):
            for lang in active_langs:
                sub = dq_f[dq_f['language']==lang]
                ax.scatter(sub['manual'], sub[col],
                           color=LANG_PAL[lang], alpha=0.75, s=55,
                           label=lang, edgecolors='white', linewidth=0.5)
            ax.plot([0,max_val],[0,max_val],'k--',linewidth=1.2,
                    alpha=0.6, label='Perfect response')
            ax.set_xlabel('True value (manual search)')
            ax.set_ylabel(f'Response from {sn}')
            ax.set_xlim(-5, max_val); ax.set_ylim(-5, max_val)
            ax.legend(fontsize=8)

        axes[0].set_title(f'{WEB}\nMAE = {dq_f["ae_web"].mean():.1f}',
                          fontweight='bold', color=SYS_PAL[WEB])
        axes[1].set_title(f'{AI4}\nMAE = {dq_f["ae_ai4"].mean():.1f}',
                          fontweight='bold', color=SYS_PAL[AI4])

        fig.suptitle('RAG response vs true catalogue value', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Grouped bar chart
        st.markdown("**Question-by-question comparison**")
        dq_sorted = dq_f.sort_values(['category','language']).reset_index(drop=True)
        n_show = min(30, len(dq_sorted))
        sub30  = dq_sorted.iloc[:n_show]
        x = np.arange(n_show); w = 0.28

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.bar(x-w, sub30['manual'], w, label='True value', color='#37474F', alpha=0.85)
        ax.bar(x,   sub30['web'],    w, label=WEB,          color=SYS_PAL[WEB], alpha=0.8)
        ax.bar(x+w, sub30['ai4'],    w, label=AI4,          color=SYS_PAL[AI4], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['language'][:3]}" for _, r in sub30.iterrows()],
            fontsize=7
        )
        ax.set_ylabel('Number of results')
        ax.set_title(f'True value vs responses (first {n_show} records)', fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: Bias ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Over/underestimation bias")
        st.caption("Above zero line = overestimates | Below = underestimates")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (sn, err_col) in zip(axes, [(WEB,'err_web'),(AI4,'err_ai4')]):
            for lang in active_langs:
                sub = dq_f[dq_f['language']==lang]
                ax.scatter(sub['manual'], sub[err_col],
                           color=LANG_PAL[lang], alpha=0.75, s=55,
                           label=lang, edgecolors='white')
            ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
            bias = dq_f[err_col].mean()
            ax.axhline(bias, color=SYS_PAL[sn], linewidth=1.5,
                       linestyle=':', label=f'Mean bias = {bias:+.1f}')
            ax.set_xlabel('True value (manual search)')
            ax.set_ylabel('Signed error (response − true value)')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            ax.legend(fontsize=8)

        fig.suptitle('Over/underestimation bias by system', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # MAE heatmap
        st.markdown("**MAE heatmap by language and category**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        for ax, (sn, col) in zip(axes, [(WEB,'ae_web'),(AI4,'ae_ai4')]):
            pivot = dq_f.pivot_table(
                values=col, index='language', columns='category', aggfunc='mean'
            ).round(1)
            pivot = pivot.loc[[l for l in LANGS if l in pivot.index],
                               [c for c in CATS_QUANT if c in pivot.columns]]
            sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd',
                        linewidths=0.5, linecolor='#eee',
                        cbar_kws={'label': 'MAE (mean)'})
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            ax.tick_params(axis='x', rotation=15)
        fig.suptitle('Heatmap — MAE by language and category', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 4: H1 Kruskal-Wallis ──────────────────────────────────────────────
    with tab4:
        st.subheader("H1 — Does language affect the observable RAG performance?")
        st.markdown("""
        **Kruskal-Wallis** compares Absolute Error distributions across the three languages.
        A significant result indicates that at least one language produces systematically different errors.
        """)

        for sn, col in [(WEB,'ae_web'),(AI4,'ae_ai4')]:
            grupos  = [dq_f[dq_f['language']==lang][col].values
                       for lang in active_langs if lang in dq_f['language'].values]
            if len(grupos) < 2:
                continue
            n_total = sum(len(g) for g in grupos)
            k       = len(grupos)
            H, p    = kruskal(*grupos)
            eps2    = epsilon_squared(H, k, n_total)

            with st.expander(f"**{sn}** — H={H:.3f}  p={p:.4f}  ε²={eps2:.4f} ({eff_eps(eps2)})",
                             expanded=True):
                st.markdown(f"**Decision:** {sig_badge(p)}")

                cols = st.columns(len(active_langs))
                for col_ui, (lang, g) in zip(cols, zip(active_langs, grupos)):
                    col_ui.metric(lang, f"Med={np.median(g):.1f}", f"n={len(g)}")

                if p < ALPHA and len(grupos) >= 3:
                    st.markdown("**Dunn post-hoc + Bonferroni correction:**")
                    long_df = pd.concat([
                        pd.DataFrame({'val': g, 'language': lang})
                        for g, lang in zip(grupos, active_langs)
                    ], ignore_index=True)
                    dunn = sp.posthoc_dunn(long_df, val_col='val',
                                           group_col='language', p_adjust='bonferroni')
                    rows = []
                    for i, l1 in enumerate(active_langs):
                        for l2 in active_langs[i+1:]:
                            pv = dunn.loc[l1, l2]
                            rows.append({'Pair': f'{l1} vs {l2}',
                                         'Adjusted p': f'{pv:.4f}',
                                         'Significant': '✓' if pv < ALPHA else '✗'})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Boxplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (sn, col) in zip(axes, [(WEB,'ae_web'),(AI4,'ae_ai4')]):
            data = [dq_f[dq_f['language']==lang][col].values for lang in active_langs]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color='black', linewidth=2))
            for patch, lang in zip(bp['boxes'], active_langs):
                patch.set_facecolor(LANG_PAL[lang]); patch.set_alpha(0.78)
            for i, g in enumerate(data):
                ax.scatter(i+1, np.mean(g), color='white', s=55,
                           zorder=5, edgecolors='black', linewidth=1.2)
            H, p = kruskal(*data)
            eps2 = epsilon_squared(H, len(data), sum(len(g) for g in data))
            cb = '#2E7D32' if p < ALPHA else '#B71C1C'
            ax.text(0.5, 0.97, f'H={H:.2f}  p={p:.3f}  ε²={eps2:.3f}' +
                    ('  ✓' if p < ALPHA else '  ✗'),
                    transform=ax.transAxes, ha='center', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor=cb, boxstyle='round', alpha=0.9))
            ax.set_xticks(range(1, len(active_langs)+1))
            ax.set_xticklabels(active_langs)
            ax.set_ylabel('Absolute Error (AE)')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])

        patches = [mpatches.Patch(color=LANG_PAL[l], label=l) for l in active_langs]
        fig.legend(handles=patches, title='Language', loc='lower center',
                   ncol=len(active_langs), bbox_to_anchor=(0.5, -0.04))
        fig.suptitle('H1 — AE by language (Kruskal-Wallis)\nWhite circle = mean',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 5: H2 Mann-Whitney ────────────────────────────────────────────────
    with tab5:
        st.subheader("H2 — Do salina.web and salina.ai4life differ by language?")
        st.markdown("""
        **Mann-Whitney U** compares the two systems within each language.
        A significant result indicates that one system produces smaller errors in that language.
        """)

        rows = []
        for lang in active_langs:
            sub = dq_f[dq_f['language']==lang]
            g_w = sub['ae_web'].values; g_a = sub['ae_ai4'].values
            U, p = mannwhitneyu(g_w, g_a, alternative='two-sided')
            r    = rank_biserial_r(U, len(g_w), len(g_a))
            rows.append({
                'Language': lang,
                'U': f'{U:.1f}',
                'p': f'{p:.4f}',
                'r': f'{r:.3f}',
                'Effect': eff_r(r),
                'Sig.': '✓' if p < ALPHA else '✗',
                'Better (lower AE)': (WEB if np.median(g_w) < np.median(g_a) else AI4)
                                     if p < ALPHA else '—',
                'Median web': f'{np.median(g_w):.1f}',
                'Median ai4life': f'{np.median(g_a):.1f}',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Boxplots by language
        fig, axes = plt.subplots(1, len(active_langs),
                                  figsize=(5*len(active_langs), 5))
        if len(active_langs) == 1:
            axes = [axes]
        for ax, lang in zip(axes, active_langs):
            sub = dq_f[dq_f['language']==lang]
            g_w = sub['ae_web'].values; g_a = sub['ae_ai4'].values
            bp  = ax.boxplot([g_w, g_a], patch_artist=True,
                              medianprops=dict(color='black', linewidth=2.2))
            for patch, color in zip(bp['boxes'], [SYS_PAL[WEB], SYS_PAL[AI4]]):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            for i, g in enumerate([g_w, g_a]):
                ax.scatter(i+1, np.mean(g), color='white', s=55,
                           zorder=5, edgecolors='black', linewidth=1.2)
            U, p = mannwhitneyu(g_w, g_a, alternative='two-sided')
            r    = rank_biserial_r(U, len(g_w), len(g_a))
            cb   = '#2E7D32' if p < ALPHA else '#B71C1C'
            ax.text(0.5, 0.03, f'p={p:.3f}  r={r:.3f}' + ('  ✓' if p < ALPHA else '  ✗'),
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor=cb, boxstyle='round', alpha=0.9))
            ax.set_xticks([1,2])
            ax.set_xticklabels(['salina.web','salina.ai4life'], fontsize=8)
            ax.set_ylabel('Absolute Error (AE)')
            ax.set_title(lang, fontweight='bold', color=LANG_PAL[lang])

        sys_p = [mpatches.Patch(color=v, label=k) for k, v in SYS_PAL.items()]
        fig.legend(handles=sys_p, title='System', loc='lower center',
                   ncol=2, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle('H2 — salina.web vs salina.ai4life by language (Mann-Whitney U)',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Median bar chart
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(active_langs)); w = 0.35
        meds_w = [dq_f[dq_f['language']==l]['ae_web'].median() for l in active_langs]
        meds_a = [dq_f[dq_f['language']==l]['ae_ai4'].median() for l in active_langs]
        bw = ax.bar(x-w/2, meds_w, w, label=WEB, color=SYS_PAL[WEB], alpha=0.85)
        ba = ax.bar(x+w/2, meds_a, w, label=AI4, color=SYS_PAL[AI4], alpha=0.85)
        for bar in list(bw)+list(ba):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(active_langs)
        ax.set_ylabel('Median Absolute Error')
        ax.set_title('Median AE by language and system', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
#  QUALITATIVE SECTION
# ════════════════════════════════════════════════════════════════════════════════
elif section == "📝 Qualitative (LaBSE)":

    st.title("📝 Qualitative Analysis — Retrieval failures")
    st.info(
        "ℹ️ LaBSE embedding computation requires running the notebook locally "
        "(model ~500MB). This dashboard shows the analysis of **retrieval failures** "
        "(negative responses) and descriptive statistics of textual responses, "
        "which do not require the model.",
        icon="ℹ️"
    )

    dc_f = dc[dc['language'].isin(active_langs)]

    # ── Global metrics ────────────────────────────────────────────────────────
    st.subheader("Detected retrieval failures")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualitative records", len(dc_f))
    c2.metric("Failures salina.web",
              f"{dc_f['neg_web'].sum()} ({dc_f['neg_web'].mean()*100:.0f}%)")
    c3.metric("Failures salina.ai4life",
              f"{dc_f['neg_ai4'].sum()} ({dc_f['neg_ai4'].mean()*100:.0f}%)")
    c4.metric("Mean response length (web)",
              f"{dc_f['resp_web'].str.len().mean():.0f} chars")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📉 Failures by language and category",
        "📋 Response explorer",
        "📐 Response length"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Failure rate by language**")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, (sn, nc) in zip(axes, [(WEB,'neg_web'),(AI4,'neg_ai4')]):
                rates = [dc_f[dc_f['language']==l][nc].mean()*100
                         for l in active_langs]
                bars  = ax.bar(active_langs, rates,
                               color=[LANG_PAL[l] for l in active_langs],
                               alpha=0.85, edgecolor='white')
                for bar, rate in zip(bars, rates):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.5, f'{rate:.0f}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.set_ylim(0, max(rates)*1.4+5)
                ax.set_ylabel('Failure rate (%)')
                ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            fig.suptitle('Retrieval failure rate by language', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Failures by category**")
            rows = []
            for cat in CATS_QUAL:
                sub = dc_f[dc_f['category']==cat]
                rows.append({
                    'Category': cat,
                    'Failures web': f"{sub['neg_web'].sum()} ({sub['neg_web'].mean()*100:.0f}%)",
                    'Failures ai4life': f"{sub['neg_ai4'].sum()} ({sub['neg_ai4'].mean()*100:.0f}%)",
                    'n': len(sub)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("**Failures by language and category (heatmap)**")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            for ax, (sn, nc) in zip(axes, [(WEB,'neg_web'),(AI4,'neg_ai4')]):
                pivot = dc_f.pivot_table(
                    values=nc, index='language', columns='category',
                    aggfunc='mean').round(2) * 100
                pivot = pivot.loc[
                    [l for l in LANGS if l in pivot.index],
                    [c for c in CATS_QUAL if c in pivot.columns]
                ]
                sns.heatmap(pivot, ax=ax, annot=True, fmt='.0f', cmap='Reds',
                            vmin=0, vmax=100,
                            linewidths=0.5, linecolor='#eee',
                            cbar_kws={'label': 'Failure rate (%)'})
                ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
                ax.tick_params(axis='x', rotation=20)
            fig.suptitle('Failure rate (%) by language and category', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("**Response explorer**")
        cat_sel  = st.selectbox("Category", CATS_QUAL)
        lang_sel = st.selectbox("Language", active_langs)
        type_sel = st.radio("Show", ["All", "Failures only", "Successful only"],
                            horizontal=True)

        sub = dc_f[(dc_f['category']==cat_sel) & (dc_f['language']==lang_sel)]
        if type_sel == "Failures only":
            sub = sub[sub['neg_web'] | sub['neg_ai4']]
        elif type_sel == "Successful only":
            sub = sub[~sub['neg_web'] & ~sub['neg_ai4']]

        st.markdown(f"*{len(sub)} records*")

        for _, row in sub.iterrows():
            with st.expander(f"🔍 {row['question'][:80]}..."):
                c1, c2 = st.columns(2)
                with c1:
                    status = "🔴 FAILURE" if row['neg_web'] else "🟢 OK"
                    st.markdown(f"**{WEB}** {status}")
                    st.markdown(str(row['resp_web'])[:600])
                with c2:
                    status = "🔴 FAILURE" if row['neg_ai4'] else "🟢 OK"
                    st.markdown(f"**{AI4}** {status}")
                    st.markdown(str(row['resp_ai4'])[:600])

    with tab3:
        st.markdown("**Response length by language and system**")
        dc_f2 = dc_f.copy()
        dc_f2['len_web'] = dc_f2['resp_web'].str.len()
        dc_f2['len_ai4'] = dc_f2['resp_ai4'].str.len()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (sn, col) in zip(axes, [(WEB,'len_web'),(AI4,'len_ai4')]):
            data = [dc_f2[dc_f2['language']==l][col].values for l in active_langs]
            bp   = ax.boxplot(data, patch_artist=True,
                              medianprops=dict(color='black', linewidth=2))
            for patch, lang in zip(bp['boxes'], active_langs):
                patch.set_facecolor(LANG_PAL[lang]); patch.set_alpha(0.78)
            ax.set_xticks(range(1, len(active_langs)+1))
            ax.set_xticklabels(active_langs)
            ax.set_ylabel('Length (characters)')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
        fig.suptitle('Response length by language and system', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("*Longer responses do not imply higher quality, "
                    "but very short responses may indicate retrieval failures "
                    "not detected by the regex pattern matching.*")

# ════════════════════════════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
elif section == "📊 Executive Summary":

    st.title("📊 Executive Summary — Milestone 3 Block 1")
    st.markdown(f"**Systems evaluated:** {WEB} vs {AI4} | **α = {ALPHA}**")

    dq_f = dq[dq['language'].isin(active_langs)]
    dc_f = dc[dc['language'].isin(active_langs)]

    st.subheader("A. Quantitative performance (MAE)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global MAE web",
              f"{dq_f['ae_web'].mean():.2f}",
              delta=f"Med={dq_f['ae_web'].median():.1f}")
    c2.metric("Global MAE ai4life",
              f"{dq_f['ae_ai4'].mean():.2f}",
              delta=f"Med={dq_f['ae_ai4'].median():.1f}")
    c3.metric("Exact matches web",
              f"{(dq_f['ae_web']==0).mean()*100:.0f}%")
    c4.metric("Exact matches ai4life",
              f"{(dq_f['ae_ai4']==0).mean()*100:.0f}%")

    st.subheader("B. Statistical tests — summary")

    rows = []
    # H1
    for sn, col in [(WEB,'ae_web'),(AI4,'ae_ai4')]:
        grupos = [dq_f[dq_f['language']==l][col].values for l in active_langs]
        if len(grupos) >= 2:
            H, p   = kruskal(*grupos)
            eps2   = epsilon_squared(H, len(grupos), sum(len(g) for g in grupos))
            rows.append({
                'Hypothesis': 'H1', 'Test': 'Kruskal-Wallis',
                'System / Comparison': sn,
                'Statistic': f'H={H:.3f}', 'p': f'{p:.4f}',
                'Effect size': f'ε²={eps2:.4f} ({eff_eps(eps2)})',
                'Decision': '✓ Reject H₀' if p < ALPHA else '✗ Do not reject H₀'
            })

    # H2
    for lang in active_langs:
        sub = dq_f[dq_f['language']==lang]
        g_w = sub['ae_web'].values; g_a = sub['ae_ai4'].values
        U, p = mannwhitneyu(g_w, g_a, alternative='two-sided')
        r    = rank_biserial_r(U, len(g_w), len(g_a))
        rows.append({
            'Hypothesis': 'H2', 'Test': 'Mann-Whitney U',
            'System / Comparison': lang,
            'Statistic': f'U={U:.1f}', 'p': f'{p:.4f}',
            'Effect size': f'r={r:.3f} ({eff_r(r)})',
            'Decision': '✓ Reject H₀' if p < ALPHA else '✗ Do not reject H₀'
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("C. Retrieval failures (qualitative)")
    c1, c2 = st.columns(2)
    c1.metric("Failures salina.web",
              f"{dc_f['neg_web'].sum()}/{len(dc_f)} ({dc_f['neg_web'].mean()*100:.0f}%)")
    c2.metric("Failures salina.ai4life",
              f"{dc_f['neg_ai4'].sum()}/{len(dc_f)} ({dc_f['neg_ai4'].mean()*100:.0f}%)")

    st.subheader("D. Methodological limitation")
    st.warning(
        "**Black-box design:** Results reflect the observable behaviour of the complete "
        "RAG system (retriever + LLM generator). It is not possible to causally attribute "
        "the observed differences to any specific component.",
        icon="⚠️"
    )

    st.subheader("E. Full data view")
    with st.expander("Quantitative (MAE)"):
        st.dataframe(
            dq_f[['language','category','web','ai4','manual','ae_web','ae_ai4']]
                .rename(columns={'web': WEB, 'ai4': AI4,
                                 'ae_web': f'AE {WEB}', 'ae_ai4': f'AE {AI4}'}),
            use_container_width=True
        )
    with st.expander("Qualitative (responses)"):
        st.dataframe(
            dc_f[['language','category','question','neg_web','neg_ai4']]
                .rename(columns={'neg_web': f'Failure {WEB}',
                                 'neg_ai4': f'Failure {AI4}'}),
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SALiNA Evaluation Dashboard · Master in Data Science for Social Sciences · "
    "Universidade de Aveiro · Milestone 3 — Block 1"
)
