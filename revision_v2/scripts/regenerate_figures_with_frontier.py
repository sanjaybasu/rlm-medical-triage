"""Regenerate Figures 1, 3 and Supplementary Figures S1, S2, S3 with frontier models added.

Combines open-source data from output/metrics/all_metrics.csv with frontier-model
metrics computed in this revision (sup_table_1d_frontier_metrics.csv for physician;
realworld_n300_frontier_metrics.csv for stratified real-world).

Outputs to revision_v2/figures/. Figure 2 is left unchanged (scaling analysis is
open-source-specific).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REVISION = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1])))
PACKAGING = Path(os.environ.get("RLM_PACKAGING_DIR",
                                os.environ.get("RLM_PACKAGING_DIR", str(Path(__file__).resolve().parents[2]))))
FIG_DIR = REVISION / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OPEN_SOURCE = ["llama3.1_8b", "qwen3_8b", "qwen3_32b", "deepseek-r1_70b"]
FRONTIER = ["Claude Opus 4.7", "GPT-5.5", "Gemini 3.1 Pro"]
FRONTIER_DISPLAY = {"Claude Opus 4.7": "Claude Opus 4.7", "GPT-5.5": "GPT-5.5",
                    "Gemini 3.1 Pro": "Gemini 3.1 Pro Preview"}
OS_LABELS = {
    "llama3.1_8b": "Llama-3.1-8B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_32b": "Qwen3-32B",
    "deepseek-r1_70b": "DeepSeek-R1-70B",
}
OS_COLORS = {
    "llama3.1_8b": "#FF9800",
    "qwen3_8b": "#2196F3",
    "qwen3_32b": "#4CAF50",
    "deepseek-r1_70b": "#9C27B0",
}
OS_MARKERS = {
    "llama3.1_8b": "o", "qwen3_8b": "s", "qwen3_32b": "D", "deepseek-r1_70b": "^",
}
FRONTIER_COLORS = {
    "Claude Opus 4.7": "#E91E63",
    "GPT-5.5": "#009688",
    "Gemini 3.1 Pro": "#FF5722",
}
FRONTIER_MARKERS = {
    "Claude Opus 4.7": "P",
    "GPT-5.5": "X",
    "Gemini 3.1 Pro": "*",
}
ARM_ORDER = ["A_single_pass", "B_chain_of_thought", "C_repl_only", "D_rlm_full"]
ARM_LABELS = {
    "A_single_pass": "Single-Pass",
    "B_chain_of_thought": "Chain-of-Thought",
    "C_repl_only": "REPL Only",
    "D_rlm_full": "Full RLM",
}
ARM_SHORT = {"A_single_pass": "A", "B_chain_of_thought": "B",
             "C_repl_only": "C", "D_rlm_full": "D"}
# Frontier arms in the same plotting order. We do not have D for frontier (Arm D was stubbed).
FRONTIER_ARM_TO_INDEX = {"Single-Pass": 0, "CoT": 1, "REPL Only": 2}


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load_open_source(dataset: str) -> pd.DataFrame:
    """Load open-source metrics from canonical all_metrics.csv."""
    df = pd.read_csv(PACKAGING / "output" / "metrics" / "all_metrics.csv")
    df = (df.sort_values("n_cases", ascending=False)
            .drop_duplicates(subset=["model", "arm", "dataset", "analysis_type"]))
    df = df[(df["dataset"] == dataset) & (df["analysis_type"] == "itt")].copy()
    df = df[df["arm"].isin(ARM_ORDER)].copy()
    df = df[df["model"].isin(OPEN_SOURCE)].copy()
    return df


def load_frontier_physician() -> pd.DataFrame:
    """Load frontier metrics on physician set from already-computed CSV."""
    p = REVISION / "frontier_runs" / "sup_table_1d_frontier_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df


def load_frontier_realworld() -> pd.DataFrame:
    p = REVISION / "frontier_runs" / "realworld_n300_frontier_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Normalise model name to match physician CSV
    df["model"] = df["model"].replace({"Gemini 3.1 Pro Preview": "Gemini 3.1 Pro"})
    return df


# ============================================================================
# Figure 1 — Sens + PSR by arm (physician)
# ============================================================================
def figure1():
    os_df = load_open_source("physician")
    fr_df = load_frontier_physician()
    arm_idx = {a: i for i, a in enumerate(ARM_ORDER)}
    # Frontier arms map: Single-Pass→A, CoT→B, REPL Only→C, Extraction→(not plotted)
    fr_arm_idx = {"Single-Pass": 0, "CoT": 1, "REPL Only": 2}

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for model in OPEN_SOURCE:
        sub = os_df[os_df["model"] == model].copy()
        sub["_ao"] = sub["arm"].map(arm_idx)
        sub = sub.sort_values("_ao").dropna(subset=["sensitivity"])
        if len(sub) == 0:
            continue
        x = sub["_ao"].tolist()
        lbl = OS_LABELS[model]; clr = OS_COLORS[model]; mkr = OS_MARKERS[model]
        sens_lo = (sub["sensitivity"] - sub["sensitivity_ci_lo"]).clip(lower=0)
        sens_hi = (sub["sensitivity_ci_hi"] - sub["sensitivity"]).clip(lower=0)
        axes[0].errorbar(x, sub["sensitivity"], yerr=[sens_lo, sens_hi],
                         fmt=f"{mkr}-", label=lbl, color=clr, capsize=2.5,
                         markersize=5, linewidth=1.1, alpha=0.85,
                         markeredgecolor="black", markeredgewidth=0.3)
        psr_sub = sub.dropna(subset=["phantom_symptom_rate"])
        if len(psr_sub) > 0:
            xp = psr_sub["_ao"].tolist()
            plo = (psr_sub["phantom_symptom_rate"] - psr_sub["psr_ci_lo"]).clip(lower=0)
            phi = (psr_sub["psr_ci_hi"] - psr_sub["phantom_symptom_rate"]).clip(lower=0)
            axes[1].errorbar(xp, psr_sub["phantom_symptom_rate"], yerr=[plo, phi],
                             fmt=f"{mkr}-", label=lbl, color=clr, capsize=2.5,
                             markersize=5, linewidth=1.1, alpha=0.85,
                             markeredgecolor="black", markeredgewidth=0.3)

    # Frontier overlay
    if not fr_df.empty:
        for model in FRONTIER:
            sub = fr_df[fr_df["model"] == model].copy()
            sub["_ao"] = sub["arm"].map(fr_arm_idx)
            sub = sub.dropna(subset=["_ao"]).sort_values("_ao")
            if len(sub) == 0: continue
            x = sub["_ao"].tolist()
            lbl = FRONTIER_DISPLAY[model]; clr = FRONTIER_COLORS[model]; mkr = FRONTIER_MARKERS[model]
            sens_lo = (sub["sens"] - sub["sens_lo"]).clip(lower=0)
            sens_hi = (sub["sens_hi"] - sub["sens"]).clip(lower=0)
            axes[0].errorbar(x, sub["sens"], yerr=[sens_lo, sens_hi],
                             fmt=f"{mkr}--", label=lbl, color=clr, capsize=2.5,
                             markersize=7, linewidth=1.3,
                             markeredgecolor="black", markeredgewidth=0.4)
            psr_lo = (sub["psr"] - sub["psr_lo"]).clip(lower=0)
            psr_hi = (sub["psr_hi"] - sub["psr"]).clip(lower=0)
            axes[1].errorbar(x, sub["psr"], yerr=[psr_lo, psr_hi],
                             fmt=f"{mkr}--", label=lbl, color=clr, capsize=2.5,
                             markersize=7, linewidth=1.3,
                             markeredgecolor="black", markeredgewidth=0.4)

    for ax in axes:
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel("Sensitivity"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("a", loc="left", fontweight="bold")
    axes[1].set_ylabel("Phantom Symptom Rate"); axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_title("b", loc="left", fontweight="bold")
    # Combined legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13),
               framealpha=0.95, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figure1_sensitivity_psr.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 saved")


# ============================================================================
# Figure 3 — Heterogeneous CoT effect (open-source + frontier, two panels)
# ============================================================================
def figure3():
    os_df_phys = load_open_source("physician")
    os_df_rw = load_open_source("realworld")
    fr_df = load_frontier_physician()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    # Panel a (open-source on parameter-count axis)
    OS_ORDER = ["llama3.1_8b", "qwen3_8b", "qwen3_32b", "deepseek-r1_70b"]
    OS_XLABELS = ["Llama-3.1\n8B", "Qwen3\n8B", "Qwen3\n32B", "DeepSeek-R1\n70B"]
    xpos_os = {m: i for i, m in enumerate(OS_ORDER)}
    for ds_name, ds_df, mkr in [("Physician", os_df_phys, "o"),
                                ("Real-world", os_df_rw, "^")]:
        xs, ys = [], []
        for m in OS_ORDER:
            a = ds_df[(ds_df["model"] == m) & (ds_df["arm"] == "A_single_pass")]
            b = ds_df[(ds_df["model"] == m) & (ds_df["arm"] == "B_chain_of_thought")]
            if len(a) and len(b):
                xs.append(xpos_os[m])
                ys.append(b.iloc[0]["sensitivity"] - a.iloc[0]["sensitivity"])
        if xs:
            axes[0].plot(xs, ys, f"{mkr}-", label=ds_name, markersize=6, linewidth=1.5,
                         markeredgecolor="black", markeredgewidth=0.3)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].fill_between([-0.5, len(OS_ORDER)-0.5], 0, 0.5, alpha=0.05, color="green")
    axes[0].fill_between([-0.5, len(OS_ORDER)-0.5], -0.5, 0, alpha=0.05, color="red")
    axes[0].set_xticks(range(len(OS_ORDER)))
    axes[0].set_xticklabels(OS_XLABELS, fontsize=7.5)
    axes[0].set_xlim(-0.4, len(OS_ORDER)-0.6)
    axes[0].set_ylabel("$\\Delta$ Sensitivity (CoT $-$ Single-Pass)")
    axes[0].set_xlabel("Open-source model")
    axes[0].set_title("a   Open-source models", loc="left", fontweight="bold", fontsize=9.5)
    axes[0].legend(loc="best", framealpha=0.9)

    # Panel b (frontier models, bar chart of ΔSens and ΔPSR on physician set)
    if not fr_df.empty:
        fr_xpos = {"Claude Opus 4.7": 0, "GPT-5.5": 1, "Gemini 3.1 Pro": 2}
        delta_sens, delta_psr = [], []
        for m in FRONTIER:
            a = fr_df[(fr_df["model"] == m) & (fr_df["arm"] == "Single-Pass")]
            b = fr_df[(fr_df["model"] == m) & (fr_df["arm"] == "CoT")]
            if len(a) and len(b):
                delta_sens.append(b.iloc[0]["sens"] - a.iloc[0]["sens"])
                delta_psr.append(b.iloc[0]["psr"] - a.iloc[0]["psr"])
            else:
                delta_sens.append(np.nan); delta_psr.append(np.nan)
        x = np.arange(len(FRONTIER))
        w = 0.35
        b1 = axes[1].bar(x - w/2, delta_sens, w, label="$\\Delta$ Sensitivity",
                         color="#1976D2", edgecolor="black", linewidth=0.4)
        b2 = axes[1].bar(x + w/2, delta_psr, w, label="$\\Delta$ PSR",
                         color="#E64A19", edgecolor="black", linewidth=0.4)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["Claude\nOpus 4.7", "GPT-5.5", "Gemini 3.1\nPro Preview"], fontsize=7.5)
        axes[1].set_ylabel("CoT effect (CoT $-$ Single-Pass)")
        axes[1].set_xlabel("Frontier reasoning model")
        axes[1].set_title("b   Frontier reasoning models (physician set)",
                          loc="left", fontweight="bold", fontsize=9.5)
        axes[1].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figure3_cot_scaling_effect.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 saved")


# ============================================================================
# Sup Fig S1 — Real-world sensitivity bar chart (open-source + frontier N=300)
# ============================================================================
def sfigure1():
    os_df = load_open_source("realworld")
    fr_df = load_frontier_realworld()
    arms = ["A_single_pass", "B_chain_of_thought", "C_repl_only", "D_rlm_full"]
    arm_lbl = {"A_single_pass": "A", "B_chain_of_thought": "B",
               "C_repl_only": "C", "D_rlm_full": "D"}
    fig, ax = plt.subplots(figsize=(9, 4))
    all_models = OPEN_SOURCE + FRONTIER
    n_models = len(all_models)
    w = 0.18
    for i, arm in enumerate(arms):
        vals, errs_lo, errs_hi, xs = [], [], [], []
        for j, m in enumerate(all_models):
            if m in OPEN_SOURCE:
                r = os_df[(os_df["model"] == m) & (os_df["arm"] == arm)]
                if len(r):
                    vals.append(r.iloc[0]["sensitivity"])
                    errs_lo.append(r.iloc[0]["sensitivity"] - r.iloc[0]["sensitivity_ci_lo"])
                    errs_hi.append(r.iloc[0]["sensitivity_ci_hi"] - r.iloc[0]["sensitivity"])
                    xs.append(j + (i - 1.5) * w)
            else:
                # frontier: only A and C plotted (we evaluated those on real-world)
                frontier_arm = {"A_single_pass": "Single-Pass",
                                "C_repl_only": "REPL Only"}.get(arm)
                if frontier_arm is None:
                    continue
                r = fr_df[(fr_df["model"] == m) & (fr_df["arm"] == frontier_arm)]
                if len(r):
                    vals.append(r.iloc[0]["sens"])
                    errs_lo.append(r.iloc[0]["sens"] - r.iloc[0]["sens_lo"])
                    errs_hi.append(r.iloc[0]["sens_hi"] - r.iloc[0]["sens"])
                    xs.append(j + (i - 1.5) * w)
        if vals:
            ax.bar(xs, vals, w, yerr=[errs_lo, errs_hi], label=f"{arm_lbl[arm]} — {arm.split('_',1)[1].replace('_',' ').title()}",
                   capsize=2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([OS_LABELS[m] if m in OPEN_SOURCE else FRONTIER_DISPLAY.get(m, m) for m in all_models],
                       rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Sensitivity (real-world set)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.text(n_models - 0.5, 0.81, "0.80 reference", fontsize=7, color="green", alpha=0.7, ha="right")
    ax.axvline(len(OPEN_SOURCE) - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(len(OPEN_SOURCE) - 0.55, 0.95, "open-source ←", fontsize=8, color="gray", ha="right")
    ax.text(len(OPEN_SOURCE) - 0.45, 0.95, "→ frontier (N=300)", fontsize=8, color="gray", ha="left")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"sfigure1_realworld_sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Sup Fig S1 saved")


# ============================================================================
# Sup Fig S2 — Action accuracy + CUT bar chart (physician, open-source + frontier)
# ============================================================================
def sfigure2():
    os_df = load_open_source("physician")
    fr_df = load_frontier_physician()
    arms = ["A_single_pass", "B_chain_of_thought", "C_repl_only", "D_rlm_full"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    all_models = OPEN_SOURCE + FRONTIER
    n_models = len(all_models)
    w = 0.18
    fr_arm_map = {"A_single_pass": "Single-Pass", "B_chain_of_thought": "CoT",
                  "C_repl_only": "REPL Only"}
    for i, arm in enumerate(arms):
        for metric_idx, (col_os, col_fr, lo_os, lo_fr, hi_os, hi_fr) in enumerate([
            ("action_accuracy", "act_acc", "action_accuracy_ci_lo", None,
             "action_accuracy_ci_hi", None),
            ("critical_under_triage", "cut", "cut_ci_lo", "cut_lo",
             "cut_ci_hi", "cut_hi"),
        ]):
            vals, lo, hi, xs = [], [], [], []
            for j, m in enumerate(all_models):
                if m in OPEN_SOURCE:
                    r = os_df[(os_df["model"] == m) & (os_df["arm"] == arm)]
                    if len(r) and pd.notna(r.iloc[0].get(col_os)):
                        v = r.iloc[0][col_os]
                        l = r.iloc[0].get(lo_os, v); h = r.iloc[0].get(hi_os, v)
                        vals.append(v); lo.append(v - l); hi.append(h - v); xs.append(j + (i - 1.5) * w)
                else:
                    frontier_arm = fr_arm_map.get(arm)
                    if frontier_arm is None: continue
                    r = fr_df[(fr_df["model"] == m) & (fr_df["arm"] == frontier_arm)]
                    if len(r):
                        v = r.iloc[0][col_fr]
                        if lo_fr is None:
                            l = v; h = v  # no bootstrap CI for frontier action_accuracy
                        else:
                            l = r.iloc[0][lo_fr]; h = r.iloc[0][hi_fr]
                        vals.append(v); lo.append(max(0, v - l)); hi.append(max(0, h - v)); xs.append(j + (i - 1.5) * w)
            if vals:
                axes[metric_idx].bar(xs, vals, w, yerr=[lo, hi],
                                      label=ARM_SHORT[arm], capsize=2,
                                      edgecolor="black", linewidth=0.3)
    titles = ["a   Action Accuracy", "b   Critical Under-Triage"]
    ylabs = ["Action accuracy", "Critical under-triage rate"]
    for ax, t, yl in zip(axes, titles, ylabs):
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([OS_LABELS[m] if m in OPEN_SOURCE else m for m in all_models],
                           rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel(yl)
        ax.set_ylim(0, 1.0)
        ax.set_title(t, loc="left", fontweight="bold", fontsize=9.5)
        ax.axvline(len(OPEN_SOURCE) - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7, title="Arm")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"sfigure2_action_accuracy.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Sup Fig S2 saved")


# ============================================================================
# Sup Fig S3 — Sens vs PSR scatter (physician, open-source + frontier)
# ============================================================================
def sfigure3():
    """Two-panel scatter: left = full range (open-source + frontier);
    right = zoomed-in upper-left IDEAL region with per-model label offsets
    to avoid overlap."""
    os_df = load_open_source("physician")
    fr_df = load_frontier_physician()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fr_arm_short = {"Single-Pass": "A", "CoT": "B", "REPL Only": "C", "Extraction Prompt": "E"}
    # Per-model label-offset direction so co-located labels don't overlap
    os_offset = {"llama3.1_8b": (8, 8), "qwen3_8b": (-12, 8),
                 "qwen3_32b": (8, -10), "deepseek-r1_70b": (-12, -10)}
    fr_offset = {"Claude Opus 4.7": (10, 10), "GPT-5.5": (10, -12),
                 "Gemini 3.1 Pro": (-14, 10)}

    def _plot(ax, zoom=False):
        for _, r in os_df.iterrows():
            if pd.isna(r.get("phantom_symptom_rate")): continue
            m = r["model"]; arm = r["arm"]
            if zoom and (r["phantom_symptom_rate"] > 0.15 or r["sensitivity"] < 0.65):
                continue
            ax.scatter(r["phantom_symptom_rate"], r["sensitivity"],
                       c=OS_COLORS[m], marker=OS_MARKERS[m], s=80,
                       edgecolors="black", linewidth=0.5, zorder=5, alpha=0.85)
            dx, dy = os_offset[m]
            ax.annotate(ARM_SHORT[arm], (r["phantom_symptom_rate"], r["sensitivity"]),
                        fontsize=7, fontweight="bold", ha="center", va="center",
                        xytext=(dx, dy), textcoords="offset points", color=OS_COLORS[m])
        if not fr_df.empty:
            for _, r in fr_df.iterrows():
                m = r["model"]; arm = r["arm"]
                if pd.isna(r.get("psr")): continue
                if zoom and (r["psr"] > 0.15 or r["sens"] < 0.65):
                    continue
                ax.scatter(r["psr"], r["sens"], c=FRONTIER_COLORS[m],
                           marker=FRONTIER_MARKERS[m], s=110,
                           edgecolors="black", linewidth=0.5, zorder=6)
                dx, dy = fr_offset[m]
                ax.annotate(fr_arm_short.get(arm, arm), (r["psr"], r["sens"]),
                            fontsize=7, fontweight="bold", ha="center", va="center",
                            xytext=(dx, dy), textcoords="offset points",
                            color=FRONTIER_COLORS[m])

    # Left panel: full range
    ax = axes[0]
    _plot(ax, zoom=False)
    ax.annotate("IDEAL", xy=(0, 1), fontsize=9, fontweight="bold", color="green",
                alpha=0.6, ha="left", va="top")
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(0.1, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
    # Indicate the zoom region
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0.65), 0.15, 0.40, fill=False,
                            edgecolor="gray", linestyle=":", linewidth=0.8))
    ax.set_xlabel("Phantom Symptom Rate (lower is better)")
    ax.set_ylabel("Sensitivity (higher is better)")
    ax.set_title("a   Full range", loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(0, 1.05)

    # Right panel: zoom on IDEAL upper-left
    ax = axes[1]
    _plot(ax, zoom=True)
    ax.annotate("IDEAL", xy=(0, 1.0), fontsize=10, fontweight="bold", color="green",
                alpha=0.6, ha="left", va="top")
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(0.1, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Phantom Symptom Rate")
    ax.set_ylabel("Sensitivity")
    ax.set_title("b   Zoom on IDEAL region", loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.005, 0.15); ax.set_ylim(0.65, 1.02)

    # Build legend with model markers + arm letter legend
    legend_elements = []
    for m in OPEN_SOURCE:
        legend_elements.append(Line2D([0],[0], marker=OS_MARKERS[m], color="w",
                                       markerfacecolor=OS_COLORS[m], markersize=7,
                                       markeredgecolor="black", markeredgewidth=0.5,
                                       label=OS_LABELS[m]))
    for m in FRONTIER:
        legend_elements.append(Line2D([0],[0], marker=FRONTIER_MARKERS[m], color="w",
                                       markerfacecolor=FRONTIER_COLORS[m], markersize=8,
                                       markeredgecolor="black", markeredgewidth=0.5,
                                       label=FRONTIER_DISPLAY[m]))
    legend_elements.append(Line2D([0],[0], marker="None", color="w", label=""))
    for arm in ARM_ORDER:
        legend_elements.append(Line2D([0],[0], marker="None", color="w",
                                       label=f"{ARM_SHORT[arm]} = {ARM_LABELS[arm]}"))
    legend_elements.append(Line2D([0],[0], marker="None", color="w",
                                   label="E = Extraction Prompt"))
    fig.legend(handles=legend_elements, fontsize=6.5, loc="lower center",
               framealpha=0.9, handletextpad=0.5, ncol=6, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"sfigure3_sensitivity_vs_psr.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Sup Fig S3 saved")


def main():
    _setup_style()
    figure1()
    figure3()
    sfigure1()
    sfigure2()
    sfigure3()


if __name__ == "__main__":
    main()
