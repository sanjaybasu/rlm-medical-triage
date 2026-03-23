"""
Statistical tests and publication-quality figure generation.

Handles 4 models across 3 GPU tiers:
  - Llama-3.1-8B, Qwen3-8B (A10G)
  - Qwen3-32B (A100)
  - DeepSeek-R1-70B (A100-80GB)

Statistical tests:
  - McNemar's test for paired binary detection outcomes
  - Paired bootstrap for continuous metrics (PSR, CFS)
  - Bonferroni correction across all comparisons

Figures designed for Nature journal submission.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import chi2

PARSED_DIR = Path(__file__).parent / 'output' / 'parsed'
METRICS_DIR = Path(__file__).parent / 'output' / 'metrics'
FIG_DIR = Path(__file__).parent / 'output' / 'figures'
TABLE_DIR = Path(__file__).parent / 'output' / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration for all 4 models ---
ALL_MODELS = ['llama3.1_8b', 'qwen3_8b', 'qwen3_32b', 'deepseek-r1_70b']
MODEL_LABELS = {
    'llama3.1_8b': 'Llama-3.1-8B',
    'qwen3_8b': 'Qwen3-8B',
    'qwen3_32b': 'Qwen3-32B',
    'deepseek-r1_70b': 'DeepSeek-R1-70B',
}
MODEL_PARAMS = {
    'llama3.1_8b': 8,
    'qwen3_8b': 8,
    'qwen3_32b': 32,
    'deepseek-r1_70b': 70,
}
MODEL_COLORS = {
    'llama3.1_8b': '#FF9800',
    'qwen3_8b': '#2196F3',
    'qwen3_32b': '#4CAF50',
    'deepseek-r1_70b': '#9C27B0',
}
MODEL_MARKERS = {
    'llama3.1_8b': 'o',
    'qwen3_8b': 's',
    'qwen3_32b': 'D',
    'deepseek-r1_70b': '^',
}

ARM_ORDER = ['A_single_pass', 'B_chain_of_thought', 'C_repl_only', 'D_rlm_full']
ARM_LABELS = {
    'A_single_pass': 'Single-Pass',
    'B_chain_of_thought': 'Chain-of-Thought',
    'C_repl_only': 'REPL Only',
    'D_rlm_full': 'Full RLM',
}
ARM_SHORT = {
    'A_single_pass': 'A',
    'B_chain_of_thought': 'B',
    'C_repl_only': 'C',
    'D_rlm_full': 'D',
}


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test comparing two classifiers on the same data."""
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    b = ((correct_a == 1) & (correct_b == 0)).sum()
    c = ((correct_a == 0) & (correct_b == 1)).sum()
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p, int(b), int(c)


def paired_bootstrap_diff(vals_a, vals_b, n_boot=10000, seed=42):
    """Bootstrap 95% CI for mean(vals_b) - mean(vals_a)."""
    rng = np.random.RandomState(seed)
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return 0.0, np.array([0.0, 0.0])
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs.append(vals_b[idx].mean() - vals_a[idx].mean())
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])


def load_audit_pair(arm_a_file, arm_b_file):
    """Load two audit CSVs and align on case_name."""
    df_a = pd.read_csv(arm_a_file)
    df_b = pd.read_csv(arm_b_file)
    if len(df_a) == len(df_b):
        df_a = df_a.sort_values('case_idx').reset_index(drop=True)
        df_b = df_b.sort_values('case_idx').reset_index(drop=True)
    else:
        merged = df_a.merge(df_b, on='case_name', suffixes=('_a', '_b'))
        df_a = merged[[c + '_a' if c + '_a' in merged.columns else c for c in df_a.columns]].copy()
        df_b = merged[[c + '_b' if c + '_b' in merged.columns else c for c in df_b.columns]].copy()
        df_a.columns = [c.replace('_a', '') for c in df_a.columns]
        df_b.columns = [c.replace('_b', '') for c in df_b.columns]
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)
        print(f"  Paired on case_name: {len(df_a)} matched cases "
              f"(from {len(pd.read_csv(arm_a_file))} x {len(pd.read_csv(arm_b_file))})")
    return df_a, df_b


def run_statistical_tests():
    """Run all pairwise comparisons across all models with Bonferroni correction."""
    audit_files = {f.stem.replace('_audit', ''): f for f in sorted(PARSED_DIR.glob('*_audit.csv'))}

    comparisons = []
    for model in ALL_MODELS:
        for ds in ['physician', 'realworld']:
            key_a = f'A_single_pass_{model}_{ds}'
            key_b = f'B_chain_of_thought_{model}_{ds}'
            key_c = f'C_repl_only_{model}_{ds}'
            key_d = f'D_rlm_full_{model}_{ds}'

            pairs = [
                (key_a, key_b, 'B_vs_A'),
                (key_a, key_c, 'C_vs_A'),
                (key_a, key_d, 'D_vs_A'),
                (key_c, key_d, 'D_vs_C'),
            ]

            for k1, k2, label in pairs:
                if k1 not in audit_files or k2 not in audit_files:
                    continue
                df1, df2 = load_audit_pair(audit_files[k1], audit_files[k2])
                y_true = df1['detection_truth'].values.astype(int)

                stat, p, b, c = mcnemar_test(
                    y_true,
                    df1['detection_pred'].values.astype(int),
                    df2['detection_pred'].values.astype(int),
                )

                psr1 = df1['phantom_symptom_rate'].fillna(0).values
                psr2 = df2['phantom_symptom_rate'].fillna(0).values
                psr_diff, psr_ci = paired_bootstrap_diff(psr1, psr2)

                cfs1 = df1['citation_fidelity_score'].fillna(0).values
                cfs2 = df2['citation_fidelity_score'].fillna(0).values
                cfs_diff, cfs_ci = paired_bootstrap_diff(cfs1, cfs2)

                comparisons.append({
                    'comparison': label,
                    'model': model,
                    'dataset': ds,
                    'n_paired': len(df1),
                    'mcnemar_stat': round(stat, 3),
                    'mcnemar_p': round(p, 6),
                    'discordant_b': b,
                    'discordant_c': c,
                    'psr_diff_mean': round(psr_diff, 4),
                    'psr_diff_ci_lo': round(psr_ci[0], 4),
                    'psr_diff_ci_hi': round(psr_ci[1], 4),
                    'cfs_diff_mean': round(cfs_diff, 4),
                    'cfs_diff_ci_lo': round(cfs_ci[0], 4),
                    'cfs_diff_ci_hi': round(cfs_ci[1], 4),
                })

    comp_df = pd.DataFrame(comparisons)
    n_tests = len(comp_df)
    if n_tests > 0:
        comp_df['bonferroni_threshold'] = round(0.05 / n_tests, 6)
        comp_df['significant'] = comp_df['mcnemar_p'] < (0.05 / n_tests)

    comp_df.to_csv(TABLE_DIR / 'statistical_tests.csv', index=False)
    print(f"Statistical tests saved ({len(comp_df)} comparisons, "
          f"Bonferroni threshold = {0.05/max(n_tests,1):.4f})")
    return comp_df


def _setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def generate_figures():
    """Generate publication-quality figures for Nature journal."""
    metrics_file = METRICS_DIR / 'all_metrics.csv'
    if not metrics_file.exists():
        print("No metrics file found. Run 04_evaluate.py first.")
        return

    _setup_style()
    df = pd.read_csv(metrics_file)
    # Deduplicate: keep highest-N row per model+arm+dataset (partial batches create duplicates)
    df = (df.sort_values('n_cases', ascending=False)
            .drop_duplicates(subset=['model', 'arm', 'dataset'])
            .reset_index(drop=True))
    phys = df[df['dataset'] == 'physician'].copy()

    # Determine which models have data
    available_models = [m for m in ALL_MODELS if m in df['model'].values]
    print(f"Available models: {[MODEL_LABELS[m] for m in available_models]}")

    # =========================================================================
    # Figure 1: Main results — Sensitivity and PSR across arms by model
    # (Physician test set, 2-panel: sensitivity left, PSR right)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    _arm_idx = {a: i for i, a in enumerate(ARM_ORDER)}
    for model in available_models:
        sub = phys[phys['model'] == model].copy()
        sub = sub[sub['arm'].isin(ARM_ORDER)].copy()
        sub['_ao'] = sub['arm'].map(_arm_idx)
        sub = sub.sort_values('_ao').dropna(subset=['sensitivity'])
        if len(sub) == 0:
            continue
        x = sub['_ao'].tolist()
        lbl = MODEL_LABELS[model]
        clr = MODEL_COLORS[model]
        mkr = MODEL_MARKERS[model]

        yerr_lo = (sub['sensitivity'] - sub['sensitivity_ci_lo']).fillna(0).clip(lower=0)
        yerr_hi = (sub['sensitivity_ci_hi'] - sub['sensitivity']).fillna(0).clip(lower=0)
        axes[0].errorbar(
            x, sub['sensitivity'], yerr=[yerr_lo, yerr_hi],
            fmt=f'{mkr}-', label=lbl, color=clr, capsize=3, markersize=5,
            linewidth=1.2, markeredgecolor='black', markeredgewidth=0.3,
        )
        psr_sub = sub.dropna(subset=['phantom_symptom_rate'])
        if len(psr_sub) > 0:
            x_psr = psr_sub['_ao'].tolist()
            yerr_lo_p = (psr_sub['phantom_symptom_rate'] - psr_sub['psr_ci_lo']).fillna(0).clip(lower=0)
            yerr_hi_p = (psr_sub['psr_ci_hi'] - psr_sub['phantom_symptom_rate']).fillna(0).clip(lower=0)
            axes[1].errorbar(
                x_psr, psr_sub['phantom_symptom_rate'], yerr=[yerr_lo_p, yerr_hi_p],
                fmt=f'{mkr}-', label=lbl, color=clr, capsize=3, markersize=5,
                linewidth=1.2, markeredgecolor='black', markeredgewidth=0.3,
            )

    for i, ax in enumerate(axes):
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], rotation=30, ha='right')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel('Sensitivity')
    axes[0].set_title('a', loc='left', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Phantom Symptom Rate')
    axes[1].set_title('b', loc='left', fontweight='bold')
    axes[1].set_ylim(-0.02, 1.05)

    fig.tight_layout(w_pad=2)
    fig.savefig(FIG_DIR / 'figure1_sensitivity_psr.pdf')
    fig.savefig(FIG_DIR / 'figure1_sensitivity_psr.png')
    plt.close(fig)
    print("Figure 1 saved: Sensitivity and PSR by arm (physician)")

    # =========================================================================
    # Figure 2: Scaling analysis — Sensitivity vs Model Size by arm
    # Shows how each arm's performance changes with model scale
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    arm_colors = {
        'A_single_pass': '#1565C0',
        'B_chain_of_thought': '#2E7D32',
        'C_repl_only': '#E65100',
        'D_rlm_full': '#C62828',
    }

    # Use categorical x-axis ordered by parameter count to avoid log-scale crowding
    MODEL_ORDER = ['llama3.1_8b', 'qwen3_8b', 'qwen3_32b', 'deepseek-r1_70b']
    MODEL_XPOS = {m: i for i, m in enumerate(MODEL_ORDER)}
    MODEL_XLABELS = ['Llama-3.1\n8B', 'Qwen3\n8B', 'Qwen3\n32B', 'DeepSeek-R1\n70B']

    phys_complete = phys[phys['arm'].isin(ARM_ORDER)].copy()
    for arm in ARM_ORDER:
        sub = phys_complete[phys_complete['arm'] == arm].copy()
        if len(sub) == 0:
            continue
        sub['xpos'] = sub['model'].map(MODEL_XPOS)
        sub = sub.dropna(subset=['xpos']).sort_values('xpos')
        clr = arm_colors[arm]

        sens_sub = sub.dropna(subset=['sensitivity'])
        if len(sens_sub) > 0:
            yerr_lo = (sens_sub['sensitivity'] - sens_sub['sensitivity_ci_lo']).fillna(0).clip(lower=0)
            yerr_hi = (sens_sub['sensitivity_ci_hi'] - sens_sub['sensitivity']).fillna(0).clip(lower=0)
            axes[0].errorbar(sens_sub['xpos'], sens_sub['sensitivity'],
                             yerr=[yerr_lo, yerr_hi], fmt='o-',
                             label=ARM_LABELS[arm], color=clr, capsize=3, markersize=5,
                             linewidth=1.5, markeredgecolor='black', markeredgewidth=0.3)
        if sub['phantom_symptom_rate'].notna().any():
            psr_sub = sub.dropna(subset=['phantom_symptom_rate'])
            yerr_lo_p = (psr_sub['phantom_symptom_rate'] - psr_sub['psr_ci_lo']).fillna(0).clip(lower=0)
            yerr_hi_p = (psr_sub['psr_ci_hi'] - psr_sub['phantom_symptom_rate']).fillna(0).clip(lower=0)
            axes[1].errorbar(psr_sub['xpos'], psr_sub['phantom_symptom_rate'],
                             yerr=[yerr_lo_p, yerr_hi_p], fmt='s-',
                             label=ARM_LABELS[arm], color=clr, capsize=3, markersize=5,
                             linewidth=1.5, markeredgecolor='black', markeredgewidth=0.3)

    for ax in axes:
        ax.set_xlabel('Model')
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_XLABELS, fontsize=7.5)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_xlim(-0.4, len(MODEL_ORDER) - 0.6)

    axes[0].set_ylabel('Sensitivity')
    axes[0].set_title('a', loc='left', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Phantom Symptom Rate')
    axes[1].set_title('b', loc='left', fontweight='bold')
    axes[1].set_ylim(-0.02, 1.05)

    # Shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, -0.12, 'PSR lines end where parse rate = 0% (undefined PSR; DeepSeek-R1-70B REPL-only).',
             ha='center', fontsize=6, color='#666666', style='italic')
    fig.tight_layout(w_pad=2, rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / 'figure2_scaling_analysis.pdf')
    fig.savefig(FIG_DIR / 'figure2_scaling_analysis.png')
    plt.close(fig)
    print("Figure 2 saved: Scaling analysis")

    # =========================================================================
    # Figure 3: Sensitivity vs PSR scatterplot (the money plot)
    # Ideal is top-left (high sensitivity, low PSR)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(5, 4.5))

    phys_primary = phys[phys['arm'].isin(ARM_ORDER)].copy()
    for _, row in phys_primary.iterrows():
        if pd.isna(row.get('phantom_symptom_rate')):
            continue
        model = row['model']
        arm = row['arm']
        clr = MODEL_COLORS.get(model, 'gray')
        mkr = MODEL_MARKERS.get(model, 'o')
        ax.scatter(row['phantom_symptom_rate'], row['sensitivity'],
                   c=clr, marker=mkr, s=80, edgecolors='black', linewidth=0.5, zorder=5)
        arm_letter = ARM_SHORT.get(arm, arm)
        ax.annotate(arm_letter,
                    (row['phantom_symptom_rate'], row['sensitivity']),
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    xytext=(8, 8), textcoords='offset points',
                    color=clr)

    # Add "ideal" annotation
    ax.annotate('IDEAL', xy=(0, 1), fontsize=8, fontweight='bold', color='green',
                alpha=0.6, ha='left', va='top')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Phantom Symptom Rate (lower is better)')
    ax.set_ylabel('Sensitivity (higher is better)')


    # Legend for models
    from matplotlib.lines import Line2D
    legend_elements = []
    for model in available_models:
        legend_elements.append(
            Line2D([0], [0], marker=MODEL_MARKERS[model], color='w',
                   markerfacecolor=MODEL_COLORS[model], markersize=7,
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=MODEL_LABELS[model])
        )
    # Add arm letter legend
    legend_elements.append(Line2D([0], [0], marker='None', color='w', label=''))
    for arm in ARM_ORDER:
        legend_elements.append(
            Line2D([0], [0], marker='None', color='w',
                   label=f'{ARM_SHORT[arm]} = {ARM_LABELS[arm]}')
        )
    ax.legend(handles=legend_elements, fontsize=6.5, loc='upper right',
              framealpha=0.9, handletextpad=0.5)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, max(0.7, phys['phantom_symptom_rate'].max() * 1.1))
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'sfigure3_sensitivity_vs_psr.pdf')
    fig.savefig(FIG_DIR / 'sfigure3_sensitivity_vs_psr.png')
    plt.close(fig)
    print("Supplementary Figure 3 saved: Sensitivity vs PSR scatterplot")

    # =========================================================================
    # Figure 4: CoT benefit heatmap — Delta sensitivity (B-A) and Delta PSR (B-A)
    # Shows scale-dependent CoT effect across models and datasets
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    delta_data = []
    for model in available_models:
        for ds in ['physician', 'realworld']:
            row_a = df[(df['model'] == model) & (df['arm'] == 'A_single_pass') & (df['dataset'] == ds)]
            row_b = df[(df['model'] == model) & (df['arm'] == 'B_chain_of_thought') & (df['dataset'] == ds)]
            if len(row_a) == 0 or len(row_b) == 0:
                continue
            delta_data.append({
                'model': MODEL_LABELS[model],
                'params': MODEL_PARAMS[model],
                'dataset': ds.title(),
                'delta_sens': row_b.iloc[0]['sensitivity'] - row_a.iloc[0]['sensitivity'],
                'delta_psr': row_b.iloc[0].get('phantom_symptom_rate', 0) - row_a.iloc[0].get('phantom_symptom_rate', 0),
            })

    if delta_data:
        delta_df = pd.DataFrame(delta_data)

        # Categorical x-axis ordered by parameter count (avoids vertical lines from shared 8B params)
        MODEL_ORDER_F3 = ['Llama-3.1-8B', 'Qwen3-8B', 'Qwen3-32B', 'DeepSeek-R1-70B']
        MODEL_XLABELS_F3 = ['Llama-3.1\n8B', 'Qwen3\n8B', 'Qwen3\n32B', 'DeepSeek-R1\n70B']
        delta_df['xpos'] = delta_df['model'].map({m: i for i, m in enumerate(MODEL_ORDER_F3)})
        n_models_f3 = len(MODEL_ORDER_F3)

        # Panel a: Delta sensitivity
        for ds, mkr in [('Physician', 'o'), ('Realworld', '^')]:
            sub = delta_df[delta_df['dataset'] == ds].dropna(subset=['xpos']).sort_values('xpos')
            if len(sub) > 0:
                axes[0].plot(sub['xpos'], sub['delta_sens'], f'{mkr}-',
                             label=ds, markersize=6, linewidth=1.5,
                             markeredgecolor='black', markeredgewidth=0.3)

        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].fill_between([-0.5, n_models_f3 - 0.5], 0, 0.5, alpha=0.05, color='green')
        axes[0].fill_between([-0.5, n_models_f3 - 0.5], -0.5, 0, alpha=0.05, color='red')
        axes[0].set_ylabel('$\\Delta$ Sensitivity (CoT $-$ Single-Pass)')
        axes[0].set_title('a', loc='left', fontweight='bold')
        axes[0].text(n_models_f3 - 1.1, 0.02, 'CoT helps', fontsize=7, color='green', alpha=0.7)
        axes[0].text(n_models_f3 - 1.1, -0.08, 'CoT hurts', fontsize=7, color='red', alpha=0.7)

        # Panel b: Delta PSR
        for ds, mkr in [('Physician', 'o'), ('Realworld', '^')]:
            sub = delta_df[delta_df['dataset'] == ds].dropna(subset=['xpos']).sort_values('xpos')
            if len(sub) > 0:
                axes[1].plot(sub['xpos'], sub['delta_psr'], f'{mkr}-',
                             label=ds, markersize=6, linewidth=1.5,
                             markeredgecolor='black', markeredgewidth=0.3)

        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].fill_between([-0.5, n_models_f3 - 0.5], -0.2, 0, alpha=0.05, color='green')
        axes[1].fill_between([-0.5, n_models_f3 - 0.5], 0, 0.2, alpha=0.05, color='red')
        axes[1].set_ylabel('$\\Delta$ PSR (CoT $-$ Single-Pass)')
        axes[1].set_title('b', loc='left', fontweight='bold')
        axes[1].text(n_models_f3 - 1.1, -0.02, 'Less hallucination', fontsize=7, color='green', alpha=0.7)
        axes[1].text(n_models_f3 - 1.1, 0.02, 'More hallucination', fontsize=7, color='red', alpha=0.7)

        for ax in axes:
            ax.set_xlabel('Model')
            ax.set_xticks(range(n_models_f3))
            ax.set_xticklabels(MODEL_XLABELS_F3, fontsize=7.5)
            ax.set_xlim(-0.4, n_models_f3 - 0.6)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout(w_pad=2)
    fig.savefig(FIG_DIR / 'figure3_cot_scaling_effect.pdf')
    fig.savefig(FIG_DIR / 'figure3_cot_scaling_effect.png')
    plt.close(fig)
    print("Figure 3 saved: CoT scaling effect")

    # =========================================================================
    # Supplementary Figure: Real-world validation grouped bar chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    rw = df[df['dataset'] == 'realworld'].copy()
    rw = rw[rw['arm'].isin(ARM_ORDER)]

    n_models = len(available_models)
    bar_width = 0.8 / max(n_models, 1)

    for j, model in enumerate(available_models):
        sub = rw[rw['model'] == model].copy()
        if len(sub) == 0:
            continue
        sub['_ao'] = sub['arm'].map(_arm_idx)
        sub = sub.dropna(subset=['_ao', 'sensitivity']).sort_values('_ao')
        if len(sub) == 0:
            continue
        x = sub['_ao'].values + j * bar_width
        yerr_lo = (sub['sensitivity'] - sub['sensitivity_ci_lo']).fillna(0).clip(lower=0)
        yerr_hi = (sub['sensitivity_ci_hi'] - sub['sensitivity']).fillna(0).clip(lower=0)
        ax.bar(x, sub['sensitivity'], bar_width,
               yerr=[yerr_lo.values, yerr_hi.values],
               label=MODEL_LABELS[model], color=MODEL_COLORS[model],
               capsize=2, edgecolor='black', linewidth=0.3)

    ax.set_xticks(np.arange(len(ARM_ORDER)) + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.set_ylabel('Sensitivity')

    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax.set_ylim(0, 0.55)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'sfigure1_realworld_sensitivity.pdf')
    fig.savefig(FIG_DIR / 'sfigure1_realworld_sensitivity.png')
    plt.close(fig)
    print("Supplementary Figure 1 saved: Real-world sensitivity")

    # =========================================================================
    # Supplementary Figure: Action accuracy and critical under-triage
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    for model in available_models:
        sub = phys[phys['model'] == model].copy()
        sub = sub[sub['arm'].isin(ARM_ORDER)].copy()
        sub['_ao'] = sub['arm'].map(_arm_idx)
        sub = sub.sort_values('_ao').dropna(subset=['action_accuracy'])
        if len(sub) == 0:
            continue
        x = sub['_ao'].tolist()
        lbl = MODEL_LABELS[model]
        clr = MODEL_COLORS[model]
        mkr = MODEL_MARKERS[model]

        yerr_lo = (sub['action_accuracy'] - sub['action_accuracy_ci_lo']).fillna(0).clip(lower=0)
        yerr_hi = (sub['action_accuracy_ci_hi'] - sub['action_accuracy']).fillna(0).clip(lower=0)
        axes[0].errorbar(
            x, sub['action_accuracy'], yerr=[yerr_lo, yerr_hi],
            fmt=f'{mkr}-', label=lbl, color=clr, capsize=3, markersize=5,
            linewidth=1.2, markeredgecolor='black', markeredgewidth=0.3,
        )
        cut_sub = sub.dropna(subset=['critical_under_triage'])
        if len(cut_sub) > 0:
            x_cut = cut_sub['_ao'].tolist()
            yerr_lo_c = (cut_sub['critical_under_triage'] - cut_sub['cut_ci_lo']).fillna(0).clip(lower=0)
            yerr_hi_c = (cut_sub['cut_ci_hi'] - cut_sub['critical_under_triage']).fillna(0).clip(lower=0)
            axes[1].errorbar(
                x_cut, cut_sub['critical_under_triage'], yerr=[yerr_lo_c, yerr_hi_c],
                fmt=f'{mkr}-', label=lbl, color=clr, capsize=3, markersize=5,
                linewidth=1.2, markeredgecolor='black', markeredgewidth=0.3,
            )

    for ax in axes:
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], rotation=30, ha='right')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel('Action Accuracy')
    axes[0].set_title('a', loc='left', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Critical Under-Triage Rate')
    axes[1].set_title('b', loc='left', fontweight='bold')
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout(w_pad=2)
    fig.savefig(FIG_DIR / 'sfigure2_action_accuracy.pdf')
    fig.savefig(FIG_DIR / 'sfigure2_action_accuracy.png')
    plt.close(fig)
    print("Supplementary Figure 2 saved: Action accuracy and under-triage")


def generate_tables():
    """Generate manuscript-ready tables."""
    metrics_file = METRICS_DIR / 'all_metrics.csv'
    if not metrics_file.exists():
        return

    df = pd.read_csv(metrics_file)

    def fmt_ci(val, lo, hi):
        if pd.isna(val):
            return '--'
        return f"{val:.3f} ({lo:.3f}-{hi:.3f})"

    def fmt_pct_ci(val, lo, hi):
        if pd.isna(val):
            return '--'
        return f"{val*100:.1f} ({lo*100:.1f}-{hi*100:.1f})"

    # ---- Table 1: Primary outcomes (physician set) ----
    phys = df[df['dataset'] == 'physician'].copy()
    rows = []
    for model in ALL_MODELS:
        for arm in ARM_ORDER:
            r = phys[(phys['model'] == model) & (phys['arm'] == arm)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append({
                'Model': MODEL_LABELS.get(model, model),
                'Parameters': f"{MODEL_PARAMS.get(model, '?')}B",
                'Arm': ARM_LABELS.get(arm, arm),
                'N': int(r['n_cases']),
                'Sensitivity (95% CI)': fmt_pct_ci(r['sensitivity'], r['sensitivity_ci_lo'], r['sensitivity_ci_hi']),
                'Specificity (95% CI)': fmt_pct_ci(r['specificity'], r['specificity_ci_lo'], r['specificity_ci_hi']),
                'MCC (95% CI)': fmt_ci(r['mcc'], r['mcc_ci_lo'], r['mcc_ci_hi']),
                'PSR (95% CI)': fmt_pct_ci(r.get('phantom_symptom_rate', np.nan),
                                           r.get('psr_ci_lo', np.nan), r.get('psr_ci_hi', np.nan)),
                'CFS (95% CI)': fmt_pct_ci(r.get('citation_fidelity', np.nan),
                                           r.get('cfs_ci_lo', np.nan), r.get('cfs_ci_hi', np.nan)),
                'Action Accuracy': fmt_pct_ci(r['action_accuracy'],
                                              r['action_accuracy_ci_lo'], r['action_accuracy_ci_hi']),
                'Critical Under-Triage': fmt_pct_ci(r['critical_under_triage'],
                                                     r['cut_ci_lo'], r['cut_ci_hi']),
                'Parse Rate': f"{r['parse_success_rate']*100:.1f}%" if pd.notna(r.get('parse_success_rate')) else '--',
            })

    table1 = pd.DataFrame(rows)
    table1.to_csv(TABLE_DIR / 'table1_physician_outcomes.csv', index=False)
    print(f"Table 1 saved: {len(table1)} rows (physician outcomes)")

    # ---- Table 2: Real-world outcomes ----
    rw = df[df['dataset'] == 'realworld'].copy()
    rows = []
    for model in ALL_MODELS:
        for arm in ARM_ORDER:
            r = rw[(rw['model'] == model) & (rw['arm'] == arm)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append({
                'Model': MODEL_LABELS.get(model, model),
                'Parameters': f"{MODEL_PARAMS.get(model, '?')}B",
                'Arm': ARM_LABELS.get(arm, arm),
                'N': int(r['n_cases']),
                'Sensitivity (95% CI)': fmt_pct_ci(r['sensitivity'], r['sensitivity_ci_lo'], r['sensitivity_ci_hi']),
                'Specificity (95% CI)': fmt_pct_ci(r['specificity'], r['specificity_ci_lo'], r['specificity_ci_hi']),
                'PSR (95% CI)': fmt_pct_ci(r.get('phantom_symptom_rate', np.nan),
                                           r.get('psr_ci_lo', np.nan), r.get('psr_ci_hi', np.nan)),
                'Action Accuracy': fmt_pct_ci(r['action_accuracy'],
                                              r['action_accuracy_ci_lo'], r['action_accuracy_ci_hi']),
                'Parse Rate': f"{r['parse_success_rate']*100:.1f}%" if pd.notna(r.get('parse_success_rate')) else '--',
            })

    table2 = pd.DataFrame(rows)
    table2.to_csv(TABLE_DIR / 'table2_realworld_outcomes.csv', index=False)
    print(f"Table 2 saved: {len(table2)} rows (real-world outcomes)")

    # ---- Table 3: All metrics (complete) ----
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'Arm': ARM_LABELS.get(r['arm'], r['arm']),
            'Model': MODEL_LABELS.get(r['model'], r['model']),
            'Dataset': r['dataset'].title(),
            'N': int(r['n_cases']),
            'Sensitivity (95% CI)': fmt_ci(r['sensitivity'], r['sensitivity_ci_lo'], r['sensitivity_ci_hi']),
            'Specificity (95% CI)': fmt_ci(r['specificity'], r['specificity_ci_lo'], r['specificity_ci_hi']),
            'MCC (95% CI)': fmt_ci(r['mcc'], r['mcc_ci_lo'], r['mcc_ci_hi']),
            'Action Accuracy (95% CI)': fmt_ci(r['action_accuracy'], r['action_accuracy_ci_lo'], r['action_accuracy_ci_hi']),
            'Critical Under-Triage (95% CI)': fmt_ci(r['critical_under_triage'], r['cut_ci_lo'], r['cut_ci_hi']),
            'PSR (95% CI)': fmt_ci(r.get('phantom_symptom_rate', np.nan),
                                   r.get('psr_ci_lo', np.nan), r.get('psr_ci_hi', np.nan)),
            'CFS (95% CI)': fmt_ci(r.get('citation_fidelity', np.nan),
                                   r.get('cfs_ci_lo', np.nan), r.get('cfs_ci_hi', np.nan)),
            'Parse Rate': f"{r['parse_success_rate']:.3f}" if pd.notna(r.get('parse_success_rate')) else '--',
            'Median Latency (s)': f"{r['latency_median_sec']:.1f}" if pd.notna(r.get('latency_median_sec')) else '--',
        })

    table3 = pd.DataFrame(rows)
    table3.to_csv(TABLE_DIR / 'table3_all_metrics.csv', index=False)
    print(f"Table 3 saved: {len(table3)} rows (all metrics)")


def print_summary():
    """Print a concise results summary for manuscript writing."""
    metrics_file = METRICS_DIR / 'all_metrics.csv'
    if not metrics_file.exists():
        return

    df = pd.read_csv(metrics_file)
    phys = df[df['dataset'] == 'physician']

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — Physician Test Set")
    print("=" * 70)

    header = f"{'Model':<22} {'Arm':<18} {'N':>5} {'Sens':>7} {'PSR':>7} {'CFS':>7} {'ActAcc':>7}"
    print(header)
    print("-" * len(header))

    for model in ALL_MODELS:
        for arm in ARM_ORDER:
            r = phys[(phys['model'] == model) & (phys['arm'] == arm)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            psr = f"{r['phantom_symptom_rate']:.3f}" if pd.notna(r.get('phantom_symptom_rate')) else '  --'
            cfs = f"{r['citation_fidelity']:.3f}" if pd.notna(r.get('citation_fidelity')) else '  --'
            print(f"{MODEL_LABELS.get(model, model):<22} "
                  f"{ARM_LABELS.get(arm, arm):<18} "
                  f"{int(r['n_cases']):>5} "
                  f"{r['sensitivity']:>7.3f} "
                  f"{psr:>7} "
                  f"{cfs:>7} "
                  f"{r['action_accuracy']:>7.3f}")
        print()

    # Key contrasts
    print("\nKEY CONTRASTS (physician set):")
    print("-" * 50)
    for model in ALL_MODELS:
        a = phys[(phys['model'] == model) & (phys['arm'] == 'A_single_pass')]
        b = phys[(phys['model'] == model) & (phys['arm'] == 'B_chain_of_thought')]
        if len(a) > 0 and len(b) > 0:
            ds = b.iloc[0]['sensitivity'] - a.iloc[0]['sensitivity']
            dp = (b.iloc[0].get('phantom_symptom_rate', 0) or 0) - (a.iloc[0].get('phantom_symptom_rate', 0) or 0)
            print(f"  {MODEL_LABELS[model]}: CoT vs SP: "
                  f"delta_sens={ds:+.3f}, delta_PSR={dp:+.3f}")


def main():
    comp_df = run_statistical_tests()
    print()
    if len(comp_df) > 0:
        sig = comp_df[comp_df['significant'] == True]
        print(f"Significant comparisons: {len(sig)} / {len(comp_df)}")
        if len(sig) > 0:
            print(sig[['comparison', 'model', 'dataset', 'mcnemar_p',
                       'psr_diff_mean']].to_string())
    print()
    generate_figures()
    print()
    generate_tables()
    print()
    print_summary()


if __name__ == '__main__':
    main()
