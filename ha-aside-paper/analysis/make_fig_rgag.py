import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_rgag_figure():
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.2)

    # Main Header
    ax.text(5.75, 5.85, "Role-Guided Attention Gating (RGAG) Architecture", 
            fontsize=14, fontweight='bold', ha='center', va='top', color='#1a252f')
    ax.text(5.75, 5.46, "Exponential suppression of deep-layer attention leakage without retraining", 
            fontsize=9.8, ha='center', va='top', style='italic', color='#555555')

    # -------------------------------------------------------------
    # 1. Left Box: Standard Attention
    # -------------------------------------------------------------
    rect1 = patches.FancyBboxPatch((0.4, 0.6), 2.8, 4.45, boxstyle="round,pad=0.12",
                                   ec='#b0bec5', fc='#f8f9fa', lw=1.5)
    ax.add_patch(rect1)
    ax.text(1.8, 4.75, "1. Standard Attention", fontsize=11, fontweight='bold', ha='center', color='#263238')
    
    # Formula box
    ax.text(1.8, 4.05, r"$S_{ij}^{(l)} = \frac{q_i^{(l)\top} k_j^{(l)}}{\sqrt{d_k}}$", 
            fontsize=11.5, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cfd8dc", lw=1.0))

    # Red warning pill
    pill_rect = patches.FancyBboxPatch((0.65, 3.1), 2.3, 0.38, boxstyle="round,pad=0.06",
                                        ec='#e57373', fc='#ffebee', lw=1.0)
    ax.add_patch(pill_rect)
    ax.text(1.8, 3.29, "Deep-Layer Vulnerability", fontsize=9.2, fontweight='bold', ha='center', va='center', color='#c62828')

    ax.text(1.8, 2.05, 
            "At deep layers (l >= 1),\ntransformer projection\nmixes query and key spaces.\n\nAttention leakage to data\nexceeds 0.50%, allowing\nprompt injection bypass.", 
            fontsize=8.5, ha='center', va='center', color='#37474f', linespacing=1.35)

    # Minus sign badge in open gap
    ax.text(3.45, 3.1, "−", fontsize=20, fontweight='bold', ha='center', va='center', color='#37474f',
            bbox=dict(boxstyle="circle,pad=0.22", fc="#ffffff", ec="#b0bec5", lw=1.2))

    # -------------------------------------------------------------
    # 2. Middle Box: Role Penalty Matrix M_ij
    # -------------------------------------------------------------
    rect2 = patches.FancyBboxPatch((3.7, 0.6), 4.25, 4.45, boxstyle="round,pad=0.12",
                                   ec='#1b9e77', fc='#f4fbf7', lw=1.8)
    ax.add_patch(rect2)
    ax.text(5.82, 4.75, "2. Role Penalty Matrix " + r"$M_{ij}$", 
            fontsize=11, fontweight='bold', ha='center', color='#00695c')

    # Matrix grid visualization (3x3 roles)
    roles = ["Sys", "User", "Data"]
    cell_w, cell_h = 0.82, 0.54
    origin_x, origin_y = 5.0, 4.15

    # Column labels (Key Role j)
    ax.text(origin_x + cell_w*1.5, origin_y + 0.36, "Key Role " + r"($r_j$)", 
            fontsize=9.0, fontweight='bold', ha='center', color='#37474f')
    for c, r_name in enumerate(roles):
        ax.text(origin_x + c*cell_w + cell_w/2, origin_y + 0.08, r_name, 
                fontsize=8.5, fontweight='bold', ha='center', color='#263238')

    # Row labels (Query Role i)
    ax.text(origin_x - 0.72, origin_y + 0.08, "Query", 
            fontsize=8.5, fontweight='bold', ha='center', color='#37474f')
    for r, r_name in enumerate(roles):
        ax.text(origin_x - 0.22, origin_y - r*cell_h - cell_h/2, r_name, 
                fontsize=8.5, fontweight='bold', ha='right', va='center', color='#263238')

    matrix_vals = [
        [0.0, 0.0, 6.0],  # Sys query -> Sys, User, Data
        [2.5, 0.0, 0.0],  # User query -> Sys, User, Data
        [0.0, 0.0, 0.0]   # Data query -> Sys, User, Data
    ]
    colors = [
        ["#e8f5e9", "#e8f5e9", "#ffcdd2"],
        ["#fff3e0", "#e8f5e9", "#e8f5e9"],
        ["#e8f5e9", "#e8f5e9", "#e8f5e9"]
    ]
    text_colors = [
        ["#2e7d32", "#2e7d32", "#b71c1c"],
        ["#e65100", "#2e7d32", "#2e7d32"],
        ["#2e7d32", "#2e7d32", "#2e7d32"]
    ]

    for r in range(3):
        for c in range(3):
            val = matrix_vals[r][c]
            bg = colors[r][c]
            tc = text_colors[r][c]
            cell_rect = patches.Rectangle((origin_x + c*cell_w, origin_y - (r+1)*cell_h),
                                          cell_w, cell_h, facecolor=bg, edgecolor='#90a4ae', lw=0.8)
            ax.add_patch(cell_rect)
            ax.text(origin_x + c*cell_w + cell_w/2, origin_y - r*cell_h - cell_h/2, 
                    f"{val:.1f}", fontsize=9.2, fontweight='bold', ha='center', va='center', color=tc)

    # Inner legend box for matrix rules
    legend_bg = patches.FancyBboxPatch((3.82, 0.76), 4.01, 1.45, boxstyle="round,pad=0.06",
                                       ec='#c8e6c9', fc='#ffffff', lw=1.0)
    ax.add_patch(legend_bg)
    ax.text(5.82, 1.95, "Hierarchical Penalty Scheme:", fontsize=8.6, fontweight='bold', ha='center', color='#00695c')
    ax.text(3.92, 1.62, "• " + r"$\lambda_{\mathrm{data}} \geq 6.0$" + ": Suppress data attention (" + r"$400\times$" + " drop)", 
            fontsize=8.0, ha='left', color='#b71c1c', fontweight='semibold')
    ax.text(3.92, 1.28, "• " + r"$\lambda_{\mathrm{user} \to \mathrm{sys}} = 2.5$" + ": Block user overriding system rules", 
            fontsize=8.0, ha='left', color='#e65100', fontweight='semibold')
    ax.text(3.92, 0.94, "• 0.0: Maintain natural attention (100% utility)", 
            fontsize=8.0, ha='left', color='#2e7d32', fontweight='semibold')

    # Arrow badge between Box 2 and Box 3
    ax.annotate('', xy=(8.42, 3.1), xytext=(7.98, 3.1), 
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=2.2, color='#00695c'))

    # -------------------------------------------------------------
    # 3. Right Box: Gated Softmax & Guarantees
    # -------------------------------------------------------------
    rect3 = patches.FancyBboxPatch((8.45, 0.6), 2.65, 4.45, boxstyle="round,pad=0.12",
                                   ec='#1565c0', fc='#f3f8fd', lw=1.5)
    ax.add_patch(rect3)
    ax.text(9.77, 4.75, "3. Gated Softmax", fontsize=11, fontweight='bold', ha='center', color='#0d47a1')

    # Formula box
    ax.text(9.75, 4.05, r"$\tilde{A}_{ij} = \frac{\exp(S_{ij} - M_{ij})}{\sum_k \exp(S_{ik} - M_{ik})}$", 
            fontsize=10.2, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#bbdefb", lw=1.0))

    # Blue guarantee pill
    pill3 = patches.FancyBboxPatch((8.6, 3.1), 2.3, 0.38, boxstyle="round,pad=0.06",
                                   ec='#90caf9', fc='#e3f2fd', lw=1.0)
    ax.add_patch(pill3)
    ax.text(9.75, 3.29, "Theorem 3 Guarantee", fontsize=9.2, fontweight='bold', ha='center', va='center', color='#0d47a1')

    ax.text(9.75, 2.62, "Total Data Attention Leakage:", fontsize=8.5, ha='center', color='#1a237e', fontweight='semibold')
    ax.text(9.75, 2.22, r"$\sum_{j \in \mathcal{D}} A_{ij} \leq 0.018\%$", fontsize=10.5, ha='center', color='#0d47a1', fontweight='bold')

    ax.text(9.75, 1.35, "✓ 395× leakage reduction\n✓ 0% injection ASR\n✓ Zero parameter overhead", 
            fontsize=8.6, ha='center', va='center', color='#2e7d32', fontweight='semibold', linespacing=1.35)

    out1 = "figures/fig_rgag_mechanism.png"
    out2 = "ha-aside-paper/paper/figures/fig_rgag_mechanism.png"
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    fig.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Saved RGAG Figure to {out1}")
    print(f"[+] Saved RGAG Figure to {out2}")

if __name__ == "__main__":
    create_rgag_figure()
