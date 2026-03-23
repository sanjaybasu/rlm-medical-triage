"""
Generate Nature-quality heatmap of physician and real-world set outcomes.
Converts main text Tables 2 and 3 into publication-quality annotated heatmaps.
Output: output/figures/figure4_heatmap_physician.png/.pdf
        output/figures/figure5_heatmap_realworld.png/.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nature-style typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colorblind-safe diverging palette anchored per metric direction
# Each metric: (display_name, higher_is_better, format_str, vmin, vmax)
PHYSICIAN_METRICS = [
    ("Sensitivity\n(%)",    True,  "{:.0f}",  0,   100),
    ("Specificity\n(%)",    True,  "{:.0f}",  50,  100),
    ("MCC",                 True,  "{:.2f}", -0.1,   1.0),
    ("PSR\n(%)",            False, "{:.0f}",  0,   100),
    ("CFS\n(%)",            True,  "{:.0f}",  0,   100),
    ("Action\nAcc (%)",     True,  "{:.0f}",  0,   100),
    ("CUT\n(%)",            False, "{:.0f}",  0,   100),
]

REALWORLD_METRICS = [
    ("Sensitivity\n(%)",    True,  "{:.0f}",  0,   100),
    ("Specificity\n(%)",    True,  "{:.0f}",  50,  100),
    ("PSR\n(%)",            False, "{:.0f}",  0,   100),
    ("CFS\n(%)",            True,  "{:.0f}",  0,   100),
    ("Action\nAcc (%)",     True,  "{:.0f}",  0,   100),
    ("CUT\n(%)",            False, "{:.0f}",  0,   100),
]

# Row order: model x arm (readable labels)
PHYSICIAN_ROWS = [
    # (model_key, arm_key, display_label, row_group)
    ("llama3.1_8b",     "A_single_pass",       "Llama-3.1-8B\nSingle-pass",     "Llama"),
    ("llama3.1_8b",     "B_chain_of_thought",  "Llama-3.1-8B\nChain-of-thought","Llama"),
    ("llama3.1_8b",     "C_repl_only",         "Llama-3.1-8B\nREPL only",        "Llama"),
    ("llama3.1_8b",     "D_rlm_full",          "Llama-3.1-8B\nFull RLM",         "Llama"),
    ("qwen3_8b",        "A_single_pass",       "Qwen3-8B\nSingle-pass",          "Qwen3-8B"),
    ("qwen3_8b",        "B_chain_of_thought",  "Qwen3-8B\nChain-of-thought",     "Qwen3-8B"),
    ("qwen3_8b",        "C_repl_only",         "Qwen3-8B\nREPL only",            "Qwen3-8B"),
    ("qwen3_8b",        "D_rlm_full",          "Qwen3-8B\nFull RLM",             "Qwen3-8B"),
    ("qwen3_32b",       "A_single_pass",       "Qwen3-32B\nSingle-pass",         "Qwen3-32B"),
    ("qwen3_32b",       "B_chain_of_thought",  "Qwen3-32B\nChain-of-thought",    "Qwen3-32B"),
    ("qwen3_32b",       "C_repl_only",         "Qwen3-32B\nREPL only",           "Qwen3-32B"),
    ("qwen3_32b",       "D_rlm_full",          "Qwen3-32B\nFull RLM",            "Qwen3-32B"),
    ("deepseek-r1_70b", "A_single_pass",       "DeepSeek-R1-70B\nSingle-pass",   "DeepSeek"),
    ("deepseek-r1_70b", "B_chain_of_thought",  "DeepSeek-R1-70B\nChain-of-thought","DeepSeek"),
    ("deepseek-r1_70b", "C_repl_only",         "DeepSeek-R1-70B\nREPL only",      "DeepSeek"),
    ("deepseek-r1_70b", "D_rlm_full",          "DeepSeek-R1-70B\nFull RLM",       "DeepSeek"),
]

REALWORLD_ROWS = [
    ("llama3.1_8b",     "A_single_pass",       "Llama-3.1-8B\nSingle-pass",     "Llama"),
    ("llama3.1_8b",     "B_chain_of_thought",  "Llama-3.1-8B\nChain-of-thought","Llama"),
    ("llama3.1_8b",     "C_repl_only",         "Llama-3.1-8B\nREPL only",        "Llama"),
    ("llama3.1_8b",     "D_rlm_full",          "Llama-3.1-8B\nFull RLM",         "Llama"),
    ("qwen3_8b",        "A_single_pass",       "Qwen3-8B\nSingle-pass",          "Qwen3-8B"),
    ("qwen3_8b",        "B_chain_of_thought",  "Qwen3-8B\nChain-of-thought",     "Qwen3-8B"),
    ("qwen3_8b",        "C_repl_only",         "Qwen3-8B\nREPL only",            "Qwen3-8B"),
    ("qwen3_8b",        "D_rlm_full",          "Qwen3-8B\nFull RLM",             "Qwen3-8B"),
    ("qwen3_32b",       "A_single_pass",       "Qwen3-32B\nSingle-pass",         "Qwen3-32B"),
    ("qwen3_32b",       "B_chain_of_thought",  "Qwen3-32B\nChain-of-thought",    "Qwen3-32B"),
    ("qwen3_32b",       "C_repl_only",         "Qwen3-32B\nREPL only",           "Qwen3-32B"),
    ("qwen3_32b",       "D_rlm_full",          "Qwen3-32B\nFull RLM",            "Qwen3-32B"),
    ("deepseek-r1_70b", "A_single_pass",       "DeepSeek-R1-70B\nSingle-pass",   "DeepSeek"),
    ("deepseek-r1_70b", "B_chain_of_thought",  "DeepSeek-R1-70B\nChain-of-thought","DeepSeek"),
    ("deepseek-r1_70b", "C_repl_only",         "DeepSeek-R1-70B\nREPL only",      "DeepSeek"),
    ("deepseek-r1_70b", "D_rlm_full",          "DeepSeek-R1-70B\nFull RLM",       "DeepSeek"),
]


def load_metrics(csv_path):
    df = pd.read_csv(csv_path)
    return df


def get_value(df, model, arm, dataset, metric_col):
    mask = (
        (df["model"].str.lower().str.replace(" ", "_") == model.lower().replace("-", "_"))
        & (df["arm"].str.lower().str.replace(" ", "_").str.replace("/", "_") == arm.lower().replace(" ", "_"))
        & (df["dataset"].str.lower() == dataset.lower())
    )
    subset = df[mask]
    if subset.empty or metric_col not in subset.columns:
        return np.nan
    val = subset[metric_col].values[0]
    return float(val) if pd.notna(val) else np.nan


def build_matrix(df, rows, metrics, dataset):
    """Build numeric matrix (n_rows x n_metrics) scaled 0-100 where applicable."""
    metric_cols = {
        "Sensitivity\n(%)": "sensitivity",
        "Specificity\n(%)": "specificity",
        "MCC": "mcc",
        "PSR\n(%)": "phantom_symptom_rate",
        "CFS\n(%)": "citation_fidelity",
        "Action\nAcc (%)": "action_accuracy",
        "CUT\n(%)": "critical_under_triage",
    }
    data = np.full((len(rows), len(metrics)), np.nan)
    for i, (model_key, arm_key, _, _) in enumerate(rows):
        # normalize arm key: strip prefix
        arm_lookup = arm_key
        for j, (metric_display, _, _, _, _) in enumerate(metrics):
            col = metric_cols.get(metric_display)
            if col is None:
                continue
            val = get_value(df, model_key, arm_lookup, dataset, col)
            if not np.isnan(val):
                # these columns are stored as 0-1 fractions in the CSV; convert to percent
                pct_cols = {
                    "sensitivity", "specificity",
                    "phantom_symptom_rate", "citation_fidelity",
                    "action_accuracy", "critical_under_triage",
                }
                if col in pct_cols:
                    val = val * 100.0
            data[i, j] = val
    return data


def make_cell_color(value, vmin, vmax, higher_is_better):
    """Map a value to a color. Green = good, red = bad, white = middle."""
    if np.isnan(value):
        return (0.85, 0.85, 0.85, 1.0)
    # normalize to [0,1]
    norm = (value - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    if not higher_is_better:
        norm = 1.0 - norm
    # RdYlGn colormap, lightened toward center for readability
    cmap = plt.cm.RdYlGn
    return cmap(norm)


def luminance(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def draw_heatmap(data, rows, metrics, title, outpath, group_separators=True):
    n_rows = len(rows)
    n_cols = len(metrics)

    cell_w = 1.3
    row_h = 0.55
    left_margin = 2.8
    top_margin = 0.8
    right_margin = 0.55
    bottom_margin = 0.5

    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin + n_rows * row_h + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("auto")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    # Draw cells
    for i in range(n_rows):
        for j, (metric_display, higher_is_better, fmt, vmin, vmax) in enumerate(metrics):
            val = data[i, j]
            color = make_cell_color(val, vmin, vmax, higher_is_better)
            rect = plt.Rectangle([j, i], 1.0, 1.0, color=color, linewidth=0.3,
                                  edgecolor="white")
            ax.add_patch(rect)
            if not np.isnan(val):
                text = fmt.format(val)
                # Choose text color based on cell luminance
                lum = luminance(color)
                text_color = "black" if lum > 0.35 else "white"
                ax.text(j + 0.5, i + 0.5, text,
                        ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="normal")
            else:
                ax.text(j + 0.5, i + 0.5, "—",
                        ha="center", va="center",
                        fontsize=7, color="#888888")

    # Group separator lines
    if group_separators:
        seen_groups = []
        for i, (_, _, _, group) in enumerate(rows):
            if group not in seen_groups:
                if seen_groups:
                    ax.axhline(y=i, color="#555555", linewidth=0.8, zorder=10)
                seen_groups.append(group)

    # Row labels
    row_labels = [r[2] for r in rows]
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(row_labels, fontsize=7.5, va="center")

    # Column labels
    col_labels = [m[0] for m in metrics]
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=7.5, ha="center", va="bottom")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Arm annotation band on left (A/B/C/D)
    arm_colors = {
        "A_single_pass": "#4477AA",
        "B_chain_of_thought": "#228833",
        "C_repl_only": "#CC3311",
        "D_rlm_full": "#AA3377",
    }
    arm_labels = {
        "A_single_pass": "A",
        "B_chain_of_thought": "B",
        "C_repl_only": "C",
        "D_rlm_full": "D",
    }
    for i, (_, arm_key, _, _) in enumerate(rows):
        color = arm_colors.get(arm_key, "#888888")
        label = arm_labels.get(arm_key, "?")
        ax.text(n_cols + 0.18, i + 0.5, label,
                ha="left", va="center",
                fontsize=7, color=color, fontweight="bold",
                transform=ax.transData, clip_on=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, top=False, length=0)

    # Colorbar legend
    sm_good = plt.cm.ScalarMappable(cmap="RdYlGn",
                                    norm=mcolors.Normalize(vmin=0, vmax=1))
    sm_good.set_array([])

    legend_text = (
        "Color scale: green = better (high sensitivity, high specificity, low PSR, low CUT); "
        "red = worse. Grey = not evaluated. PSR = Phantom Symptom Rate; "
        "CUT = Critical Under-Triage rate."
    )
    fig.text(0.02, 0.005, legend_text, fontsize=6, color="#555555",
             ha="left", va="bottom", wrap=True,
             style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outpath + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}.png / .pdf")


def main():
    metrics_csv = os.path.join(os.path.dirname(__file__),
                               "output", "metrics", "all_metrics.csv")
    df = load_metrics(metrics_csv)

    # Normalize column names for lookup
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Normalize model and arm columns
    if "model" in df.columns:
        df["model"] = df["model"].str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    if "arm" in df.columns:
        df["arm"] = df["arm"].str.strip().str.lower().str.replace(" ", "_")
    # Deduplicate: keep the ITT (max n_cases) row per model+arm+dataset
    if "n_cases" in df.columns:
        df = (df.sort_values("n_cases", ascending=False)
                .drop_duplicates(subset=["model", "arm", "dataset"])
                .reset_index(drop=True))

    # Physician heatmap
    phys_data = build_matrix(df, PHYSICIAN_ROWS, PHYSICIAN_METRICS, "physician")
    draw_heatmap(
        phys_data,
        PHYSICIAN_ROWS,
        PHYSICIAN_METRICS,
        title="Figure 4 | Physician test set outcomes across inference strategies (N=450)",
        outpath=os.path.join(OUTPUT_DIR, "figure4_heatmap_physician"),
    )

    # Real-world heatmap
    rw_data = build_matrix(df, REALWORLD_ROWS, REALWORLD_METRICS, "realworld")
    draw_heatmap(
        rw_data,
        REALWORLD_ROWS,
        REALWORLD_METRICS,
        title="Figure 5 | Real-world validation set outcomes across inference strategies",
        outpath=os.path.join(OUTPUT_DIR, "figure5_heatmap_realworld"),
    )


if __name__ == "__main__":
    main()
