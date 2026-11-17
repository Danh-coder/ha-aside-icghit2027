"""
Publication-Quality Figure Generator for HA-ASIDE Paper.
Generates:
1. Fig 1: Architecture schematic (vector-like style) comparing ASIDE vs HA-ASIDE
2. Fig 2: Layer-wise probing accuracy & cosine similarity drift
3. Fig 3: Multi-turn leakage across Turns 1, 2, 3, 4 (LLaMA-3, Qwen-2.5, Mistral)
4. Fig 4: Multi-Model Benchmark Comparison (Direct ASR, Indirect ASR, Utility)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Set clean academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "ha-aside-paper", "paper", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "ha-aside-paper", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette: Academic publication compliant
COLOR_VANILLA = "#7f7f7f"   # Gray
COLOR_ASIDE = "#d95f02"     # Orange-Red (ICLR 2026 baseline)
COLOR_HAASIDE = "#1b9e77"   # Teal / Emerald Green (Ours)
COLOR_ACCENT = "#7570b3"    # Purple

def draw_axes_helper(ax, x_len=9.5, y_len=9.5):
    """Draw clean L-shaped axes with arrowheads and clear labels."""
    ax.annotate('', xy=(x_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.8, mutation_scale=14))
    ax.annotate('', xy=(0, y_len), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.8, mutation_scale=14))
    ax.text(x_len / 2.0, -1.0, "1-dim (illustrative in 2D)", 
            ha='center', va='top', fontsize=10, color='#222222')
    ax.text(-1.1, y_len / 2.0, "2-dim (illustrative in 2D)", 
            ha='center', va='bottom', rotation=90, fontsize=10, color='#222222')
    ax.set_xlim(-2.2, 10.8)
    ax.set_ylim(-1.8, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')

def scatter_segment_helper(p_start, p_end, n_pts, noise=0.14, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 1, n_pts)
    x = p_start[0] + t * (p_end[0] - p_start[0]) + np.random.normal(0, noise, n_pts)
    y = p_start[1] + t * (p_end[1] - p_start[1]) + np.random.normal(0, noise, n_pts)
    return x, y

def plot_fig1_architecture():
    """Generates Figure 1: Geometric representation comparison (Standard LM vs ASIDE vs HA-ASIDE)."""
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6), dpi=300)
    
    # (a) Standard LM
    ax1 = axes[0]
    draw_axes_helper(ax1)
    ax1.set_title("Standard Language Model\n(no separation of functional roles)", 
                  fontsize=12, fontweight='bold', pad=18, linespacing=1.4)
    x_norm, y_norm = scatter_segment_helper((2.2, 1.8), (8.2, 7.2), 32, noise=0.18, seed=101)
    ax1.scatter(x_norm[6:], y_norm[6:], s=42, color='#1e1e1e', edgecolors='#000000', lw=0.8, zorder=3)
    x_inj = x_norm[:6] + np.array([-0.1, 0.12, -0.12, 0.08, -0.05, 0.10])
    y_inj = y_norm[:6] + np.array([0.12, -0.08, 0.10, -0.12, 0.06, -0.04])
    ax1.scatter(x_inj, y_inj, s=55, facecolors='#f3e5f5', edgecolors='#7b1fa2', lw=2.0, zorder=4)
    ax1.scatter(x_inj, y_inj, s=18, color='#7b1fa2', zorder=4)

    ax1.annotate('"solve" token\n(instruction)', xy=(x_norm[-3], y_norm[-3]), xytext=(x_norm[-3]-1.6, y_norm[-3]+1.6),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2), fontsize=8.5, ha='center', fontweight='medium')
    ax1.annotate('"2 + 2" token\n(data payload)', xy=(x_norm[14], y_norm[14]), xytext=(x_norm[14]+1.6, y_norm[14]-1.4),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2), fontsize=8.5, ha='left', fontweight='medium')
    ax1.annotate('prompt injection tokens\n(indistinguishable in 2D)', xy=(x_inj[2], y_inj[2]), xytext=(x_inj[2]+1.6, y_inj[2]-1.4),
                 arrowprops=dict(arrowstyle="->", color='#7b1fa2', lw=1.3), fontsize=8.5, ha='left', color='#6a1b9a', fontweight='semibold')
    leg1 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1e1e1e', markeredgecolor='#000', markersize=6.5, label='Token embeddings (any role)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f3e5f5', markeredgecolor='#7b1fa2', markeredgewidth=1.8, markersize=7.0, label='Prompt injection tokens')
    ]
    ax1.legend(handles=leg1, loc='upper left', bbox_to_anchor=(0.08, 0.98), 
               frameon=True, framealpha=0.95, facecolor='#fafafa', edgecolor='#d0d7de', fontsize=8.0)

    # (b) ASIDE
    ax2 = axes[1]
    draw_axes_helper(ax2)
    ax2.set_title("ASIDE (ICLR 2026)\n(binary 90° planar rotation for data)", 
                  fontsize=12, fontweight='bold', pad=18, linespacing=1.4)
    x_inst, y_inst = scatter_segment_helper((4.6, 3.8), (8.4, 7.0), 20, noise=0.16, seed=202)
    ax2.scatter(x_inst, y_inst, s=44, color='#3498db', edgecolors='#1b4f72', lw=0.9, zorder=3)
    x_ghost, y_ghost = scatter_segment_helper((2.2, 1.8), (4.4, 3.6), 12, noise=0.15, seed=303)
    ax2.scatter(x_ghost, y_ghost, s=38, facecolors='none', edgecolors='#95a5a6', linestyle='--', lw=1.3, zorder=2)
    x_data, y_data = scatter_segment_helper((1.4, 3.4), (2.5, 6.8), 12, noise=0.14, seed=303)
    ax2.scatter(x_data, y_data, s=44, color='#f39c12', edgecolors='#b9770e', lw=0.9, zorder=4)

    arc = patches.Arc((3.2, 2.5), 3.8, 3.8, angle=0, theta1=65, theta2=155, color='#7f8c8d', linestyle='--', lw=1.5, zorder=2)
    ax2.add_patch(arc)
    ax2.annotate('', xy=(x_data[2], y_data[2]), xytext=(x_data[2]+0.6, y_data[2]-0.4),
                 arrowprops=dict(arrowstyle="-|>", color='#7f8c8d', lw=1.5, mutation_scale=12))
    ax2.text(2.5, 4.3, "90° rotation", fontsize=8.5, style='italic', color='#444444', ha='center', rotation=22)

    ax2.annotate('"solve" token\n(instruction)', xy=(x_inst[-2], y_inst[-2]), xytext=(x_inst[-2]-0.5, y_inst[-2]+1.6),
                 arrowprops=dict(arrowstyle="->", color='#1b4f72', lw=1.1), fontsize=8.5, ha='center', color='#154360', fontweight='medium')
    ax2.annotate('"2 + 2" token\n(instruction)', xy=(x_inst[2], y_inst[2]), xytext=(x_inst[2]+1.8, y_inst[2]-1.4),
                 arrowprops=dict(arrowstyle="->", color='#1b4f72', lw=1.1), fontsize=8.5, ha='left', color='#154360', fontweight='medium')
    ax2.annotate('"2 + 2" token\n(as data)', xy=(x_data[5], y_data[5]), xytext=(x_data[5]+1.2, y_data[5]+1.4),
                 arrowprops=dict(arrowstyle="->", color='#b9770e', lw=1.1), fontsize=8.5, ha='left', color='#935116', fontweight='medium')
    ax2.annotate('Vulnerability: System & User\ncollapsed on same 0° axis!', xy=(x_ghost[3], y_ghost[3]), xytext=(x_ghost[3]+2.2, y_ghost[3]-1.6),
                 arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.3), fontsize=8.2, ha='left', color='#922b21', fontweight='semibold',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2e9", ec="#f5b7b1", lw=0.8))
    leg2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markeredgecolor='#1b4f72', markersize=6.5, label='Instruction token embeddings'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markeredgecolor='#b9770e', markersize=6.5, label='Data token embeddings (90°)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#95a5a6', linestyle='--', markeredgewidth=1.3, markersize=6.0, label='Original data position')
    ]
    ax2.legend(handles=leg2, loc='upper left', bbox_to_anchor=(0.08, 0.98), 
               frameon=True, framealpha=0.95, facecolor='#fafafa', edgecolor='#d0d7de', fontsize=7.8)

    # (c) HA-ASIDE
    ax3 = axes[2]
    draw_axes_helper(ax3)
    ax3.set_title("HA-ASIDE (Ours)\n(hierarchical 4-tier rotation + RGAG gating)", 
                  fontsize=12, fontweight='bold', pad=18, linespacing=1.4)
    x_sys, y_sys = scatter_segment_helper((3.0, 0.8), (8.6, 1.8), 14, noise=0.13, seed=404)
    ax3.scatter(x_sys, y_sys, s=44, color='#2c3e50', edgecolors='#1a252f', lw=0.9, zorder=3)
    x_usr, y_usr = scatter_segment_helper((2.4, 2.4), (7.4, 6.8), 14, noise=0.15, seed=505)
    ax3.scatter(x_usr, y_usr, s=44, color='#1b9e77', edgecolors='#0e664e', lw=0.9, zorder=4)
    x_dat, y_dat = scatter_segment_helper((1.2, 3.2), (2.0, 6.8), 14, noise=0.14, seed=606)
    ax3.scatter(x_dat, y_dat, s=44, color='#e67e22', edgecolors='#b95e04', lw=0.9, zorder=4)

    arc_u = patches.Arc((0, 0), 7.8, 7.8, angle=0, theta1=14, theta2=45, color='#1b9e77', linestyle='--', lw=1.5, zorder=2)
    ax3.add_patch(arc_u)
    ax3.annotate('', xy=(x_usr[4], y_usr[4]), xytext=(x_usr[4]+0.5, y_usr[4]-0.5),
                 arrowprops=dict(arrowstyle="-|>", color='#1b9e77', lw=1.5, mutation_scale=11))
    ax3.text(3.8, 2.0, "45° rotation", fontsize=8.0, color='#0e664e', fontweight='semibold', rotation=28)

    arc_d = patches.Arc((0, 0), 9.6, 9.6, angle=0, theta1=15, theta2=75, color='#e67e22', linestyle='--', lw=1.5, zorder=2)
    ax3.add_patch(arc_d)
    ax3.annotate('', xy=(x_dat[5], y_dat[5]), xytext=(x_dat[5]+0.6, y_dat[5]-0.4),
                 arrowprops=dict(arrowstyle="-|>", color='#e67e22', lw=1.5, mutation_scale=11))
    ax3.text(2.3, 4.6, "90° rotation", fontsize=8.0, color='#935116', fontweight='semibold', rotation=45)

    ax3.annotate('System Prompt (0°)\n[Developer Security Anchor]', xy=(x_sys[-2], y_sys[-2]), xytext=(x_sys[-2]+0.6, y_sys[-2]+1.6),
                 arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=1.1), fontsize=8.0, ha='center', color='#1a252f', fontweight='semibold')
    ax3.annotate('User Instruction (45°)\n[Direct Jailbreak Blocked]', xy=(x_usr[-3], y_usr[-3]), xytext=(x_usr[-3]+1.6, y_usr[-3]+0.6),
                 arrowprops=dict(arrowstyle="->", color='#0e664e', lw=1.1), fontsize=8.0, ha='left', color='#0e664e', fontweight='semibold')
    ax3.annotate('Untrusted Data (90°)\n[Indirect Injection Blocked]', xy=(x_dat[-2], y_dat[-2]), xytext=(x_dat[-2]+1.6, y_dat[-2]+1.2),
                 arrowprops=dict(arrowstyle="->", color='#b95e04', lw=1.1), fontsize=8.0, ha='left', color='#935116', fontweight='semibold')

    leg3 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c3e50', markeredgecolor='#1a252f', markersize=6.5, label='System / Assistant (0° Anchor)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1b9e77', markeredgecolor='#0e664e', markersize=6.5, label='User Instructions (45° Rotation)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', markeredgecolor='#b95e04', markersize=6.5, label='Untrusted Data (90° Rotation)')
    ]
    ax3.legend(handles=leg3, loc='upper left', bbox_to_anchor=(0.08, 0.98), 
               frameon=True, framealpha=0.95, facecolor='#fafafa', edgecolor='#d0d7de', fontsize=7.8)

    plt.subplots_adjust(wspace=0.18, left=0.04, right=0.98, top=0.88, bottom=0.09)
    out_file = os.path.join(FIGURES_DIR, "fig1_ha_aside_architecture.png")
    fig.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    root_fig = os.path.join(PROJECT_ROOT, "figures", "fig1_ha_aside_architecture.png")
    if os.path.exists(os.path.dirname(root_fig)):
        fig.savefig(root_fig, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"[+] Saved Fig 1: {out_file}")

def plot_fig2_layerwise():
    """Generates Figure 2: Layer-wise probing accuracy and subspace drift."""
    probing_path = os.path.join(RESULTS_DIR, "linear_probing_results.json")
    if not os.path.exists(probing_path):
        raise FileNotFoundError(
            f"Probing results file not found at '{probing_path}'. "
            "Please run analysis/linear_probing.py first to produce empirical results."
        )

    with open(probing_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        layers = list(range(len(data["Vanilla"]["probe_accuracies"])))
        vanilla_acc = data["Vanilla"]["probe_accuracies"]
        aside_acc = data["ASIDE (ICLR 2026)"]["probe_accuracies"]
        ha_aside_acc = data["HA-ASIDE (Ours)"]["probe_accuracies"]

        vanilla_cos = data["Vanilla"]["cosine_similarities"]
        aside_cos = data["ASIDE (ICLR 2026)"]["cosine_similarities"]
        ha_aside_cos = data["HA-ASIDE (Ours)"]["cosine_similarities"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Probing Accuracy
    ax1.plot(layers, vanilla_acc, 's--', color=COLOR_VANILLA, label="Vanilla LM", alpha=0.8)
    ax1.plot(layers, aside_acc, 'o-', color=COLOR_ASIDE, label="ASIDE (ICLR 2026)", alpha=0.9)
    ax1.plot(layers, ha_aside_acc, 'D-', color=COLOR_HAASIDE, label="HA-ASIDE (Ours)", lw=2.5)
    ax1.set_xlabel("Transformer Layer Index")
    ax1.set_ylabel("Linear Probe Role Accuracy")
    ax1.set_title("(a) Layer-wise Role Separability")
    ax1.set_ylim(0.2, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.9)

    # (b) Cosine Similarity (Subspace Drift)
    ax2.plot(layers, vanilla_cos, 's--', color=COLOR_VANILLA, label="Vanilla LM", alpha=0.8)
    ax2.plot(layers, aside_cos, 'o-', color=COLOR_ASIDE, label="ASIDE (ICLR 2026)", alpha=0.9)
    ax2.plot(layers, ha_aside_cos, 'D-', color=COLOR_HAASIDE, label="HA-ASIDE (Ours)", lw=2.5)
    ax2.set_xlabel("Transformer Layer Index")
    ax2.set_ylabel("Cosine Similarity: cos(h_instruction, h_data)")
    ax2.set_title("(b) Instruction-Data Subspace Drift")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, "fig2_layerwise_probing.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved Fig 2: {out_file}")

def plot_fig3_multiturn():
    """Generates Figure 3: Cumulative Multi-Turn Attention Leakage."""
    bench_path = os.path.join(RESULTS_DIR, "benchmarks_3models.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    turns = [2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    models = [("llama3", "LLaMA-3.1"), ("qwen25", "Qwen-2.5"), ("mistral", "Mistral-7B")]

    for idx, (m_key, m_name) in enumerate(models):
        ax = axes[idx]
        m_data = bench_data[m_key]

        v_leak = [m_data["vanilla"][f"turn_{t}_leakage"] for t in turns]
        a_leak = [m_data["aside_orig"][f"turn_{t}_leakage"] for t in turns]
        h_leak = [m_data["ha_aside"][f"turn_{t}_leakage"] for t in turns]

        ax.plot(turns, v_leak, 's--', color=COLOR_VANILLA, label="Vanilla", alpha=0.8)
        ax.plot(turns, a_leak, 'o-', color=COLOR_ASIDE, label="ASIDE (ICLR 2026)", alpha=0.9)
        ax.plot(turns, h_leak, 'D-', color=COLOR_HAASIDE, label="HA-ASIDE (Ours)", lw=2.5)

        ax.set_title(f"{m_name}")
        ax.set_xlabel("Dialogue Turn")
        ax.set_xticks(turns)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Attention Leakage to Injection (%)")
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Multi-Turn Cumulative Cross-Attention Leakage Across Dialogue Turns 2, 3, 4", y=1.02, weight='bold')
    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, "fig3_multiturn_leakage.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved Fig 3: {out_file}")

def plot_fig4_benchmarks():
    """Generates Figure 4: Grouped Bar Chart of Direct ASR, Indirect ASR, and Utility across 3 models."""
    bench_path = os.path.join(RESULTS_DIR, "benchmarks_3models.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    models = ["LLaMA-3.1", "Qwen-2.5", "Mistral-7B"]
    m_keys = ["llama3", "qwen25", "mistral"]
    x = np.arange(len(models))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Direct Injection Attack Success Rate (Jailbreak)
    v_direct = [bench_data[k]["vanilla"]["direct_asr"] for k in m_keys]
    a_direct = [bench_data[k]["aside_orig"]["direct_asr"] for k in m_keys]
    h_direct = [bench_data[k]["ha_aside"]["direct_asr"] for k in m_keys]

    rects1 = ax1.bar(x - width, v_direct, width, label='Vanilla', color=COLOR_VANILLA, alpha=0.8)
    rects2 = ax1.bar(x, a_direct, width, label='ASIDE (ICLR 2026)', color=COLOR_ASIDE, alpha=0.9)
    rects3 = ax1.bar(x + width, h_direct, width, label='HA-ASIDE (Ours)', color=COLOR_HAASIDE)

    ax1.set_ylabel('Attack Success Rate (ASR %)')
    ax1.set_title('(a) Direct Prompt Injection / Jailbreak ASR (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    # (b) Task Utility on Standard Benchmarks
    v_util = [bench_data[k]["vanilla"]["utility_score"] for k in m_keys]
    a_util = [bench_data[k]["aside_orig"]["utility_score"] for k in m_keys]
    h_util = [bench_data[k]["ha_aside"]["utility_score"] for k in m_keys]

    rects4 = ax2.bar(x - width, v_util, width, label='Vanilla', color=COLOR_VANILLA, alpha=0.8)
    rects5 = ax2.bar(x, a_util, width, label='ASIDE (ICLR 2026)', color=COLOR_ASIDE, alpha=0.9)
    rects6 = ax2.bar(x + width, h_util, width, label='HA-ASIDE (Ours)', color=COLOR_HAASIDE)

    ax2.set_ylabel('Task Utility Score (%)')
    ax2.set_title('(b) Downstream Task Utility (Higher is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, "fig4_direct_indirect_asr.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved Fig 4: {out_file}")

def plot_fig_rgag_mechanism():
    """Generates schematic for Role-Guided Attention Gating (RGAG)."""
    from make_fig_rgag import create_rgag_figure
    create_rgag_figure()

def main():
    print("[*] Generating all publication figures...")
    plot_fig1_architecture()
    plot_fig_rgag_mechanism()
    plot_fig2_layerwise()
    plot_fig3_multiturn()
    plot_fig4_benchmarks()
    print("\n[SUCCESS] All publication figures successfully generated in ha-aside-paper/paper/figures/!")

if __name__ == "__main__":
    main()
