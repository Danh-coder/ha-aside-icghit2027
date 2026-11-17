import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Set up clean publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 10,
    'axes.labelsize': 10.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def draw_axes(ax, x_len=9.5, y_len=9.5):
    """Draw clean L-shaped axes with arrowheads and clear labels."""
    # X-axis arrow
    ax.annotate('', xy=(x_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.8, mutation_scale=14))
    # Y-axis arrow
    ax.annotate('', xy=(0, y_len), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.8, mutation_scale=14))
    
    # Text labels for axes with ample padding
    ax.text(x_len / 2.0, -1.0, "1-dim (illustrative in 2D)", 
            ha='center', va='top', fontsize=10, color='#222222')
    ax.text(-1.1, y_len / 2.0, "2-dim (illustrative in 2D)", 
            ha='center', va='bottom', rotation=90, fontsize=10, color='#222222')
    
    ax.set_xlim(-2.2, 10.8)
    ax.set_ylim(-1.8, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')

def scatter_along_segment(p_start, p_end, n_pts, noise=0.14, seed=42):
    """Generate noisy points scattered along a line segment."""
    np.random.seed(seed)
    t = np.linspace(0, 1, n_pts)
    x = p_start[0] + t * (p_end[0] - p_start[0]) + np.random.normal(0, noise, n_pts)
    y = p_start[1] + t * (p_end[1] - p_start[1]) + np.random.normal(0, noise, n_pts)
    return x, y

def create_figure():
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6), dpi=300)
    
    # =========================================================================
    # Panel (a): Standard Language Model
    # =========================================================================
    ax1 = axes[0]
    draw_axes(ax1)
    
    ax1.set_title("Standard Language Model\n(no separation of functional roles)", 
                  fontsize=12, fontweight='bold', pad=18, linespacing=1.4)

    # Main diagonal token manifold
    x_norm, y_norm = scatter_along_segment((2.2, 1.8), (8.2, 7.2), 32, noise=0.18, seed=101)
    
    # Normal tokens
    ax1.scatter(x_norm[6:], y_norm[6:], s=42, color='#1e1e1e', edgecolors='#000000', lw=0.8, zorder=3)
    
    # Prompt injection tokens (sitting at lower end of the same manifold)
    x_inj = x_norm[:6] + np.array([-0.1, 0.12, -0.12, 0.08, -0.05, 0.10])
    y_inj = y_norm[:6] + np.array([0.12, -0.08, 0.10, -0.12, 0.06, -0.04])
    ax1.scatter(x_inj, y_inj, s=55, facecolors='#f3e5f5', edgecolors='#7b1fa2', lw=2.0, zorder=4)
    ax1.scatter(x_inj, y_inj, s=18, color='#7b1fa2', zorder=4)

    # Callouts
    ax1.annotate('"solve" token\n(instruction)', 
                 xy=(x_norm[-3], y_norm[-3]), 
                 xytext=(x_norm[-3]-1.6, y_norm[-3]+1.6),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2),
                 fontsize=8.5, ha='center', fontweight='medium')

    ax1.annotate('"2 + 2" token\n(data payload)', 
                 xy=(x_norm[14], y_norm[14]), 
                 xytext=(x_norm[14]+1.6, y_norm[14]-1.4),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2),
                 fontsize=8.5, ha='left', fontweight='medium')

    ax1.annotate('prompt injection tokens\n(indistinguishable in 2D)', 
                 xy=(x_inj[2], y_inj[2]), 
                 xytext=(x_inj[2]+1.6, y_inj[2]-1.4),
                 arrowprops=dict(arrowstyle="->", color='#7b1fa2', lw=1.3),
                 fontsize=8.5, ha='left', color='#6a1b9a', fontweight='semibold')

    # Legend cleanly positioned inside
    leg1 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1e1e1e', markeredgecolor='#000', markersize=6.5, label='Token embeddings (any role)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f3e5f5', markeredgecolor='#7b1fa2', markeredgewidth=1.8, markersize=7.0, label='Prompt injection tokens')
    ]
    ax1.legend(handles=leg1, loc='upper left', bbox_to_anchor=(0.08, 0.98), 
               frameon=True, framealpha=0.95, facecolor='#fafafa', edgecolor='#d0d7de', fontsize=8.0)

    # =========================================================================
    # Panel (b): ASIDE (ICLR 2026)
    # =========================================================================
    ax2 = axes[1]
    draw_axes(ax2)
    
    ax2.set_title("ASIDE (ICLR 2026)\n(binary 90° planar rotation for data)", 
                  fontsize=12, fontweight='bold', pad=18, linespacing=1.4)

    # Instruction tokens (upper portion of diagonal)
    x_inst, y_inst = scatter_along_segment((4.6, 3.8), (8.4, 7.0), 20, noise=0.16, seed=202)
    ax2.scatter(x_inst, y_inst, s=44, color='#3498db', edgecolors='#1b4f72', lw=0.9, zorder=3)

    # Ghost original data positions along lower diagonal (dashed gray circles)
    x_ghost, y_ghost = scatter_along_segment((2.2, 1.8), (4.4, 3.6), 12, noise=0.15, seed=303)
    ax2.scatter(x_ghost, y_ghost, s=38, facecolors='none', edgecolors='#95a5a6', linestyle='--', lw=1.3, zorder=2)

    # Rotated data branch
    x_data, y_data = scatter_along_segment((1.4, 3.4), (2.5, 6.8), 12, noise=0.14, seed=303)
    ax2.scatter(x_data, y_data, s=44, color='#f39c12', edgecolors='#b9770e', lw=0.9, zorder=4)

    # Dashed rotation arrow from ghost manifold to rotated branch
    arc = patches.Arc((3.2, 2.5), 3.8, 3.8, angle=0, theta1=65, theta2=155,
                      color='#7f8c8d', linestyle='--', lw=1.5, zorder=2)
    ax2.add_patch(arc)
    ax2.annotate('', xy=(x_data[2], y_data[2]), xytext=(x_data[2]+0.6, y_data[2]-0.4),
                 arrowprops=dict(arrowstyle="-|>", color='#7f8c8d', lw=1.5, mutation_scale=12))
    ax2.text(2.5, 4.3, "90° rotation", fontsize=8.5, style='italic', color='#444444', ha='center', rotation=22)

    # Callouts
    ax2.annotate('"solve" token\n(instruction)', 
                 xy=(x_inst[-2], y_inst[-2]), 
                 xytext=(x_inst[-2]-0.5, y_inst[-2]+1.6),
                 arrowprops=dict(arrowstyle="->", color='#1b4f72', lw=1.1),
                 fontsize=8.5, ha='center', color='#154360', fontweight='medium')

    ax2.annotate('"2 + 2" token\n(instruction)', 
                 xy=(x_inst[2], y_inst[2]), 
                 xytext=(x_inst[2]+1.8, y_inst[2]-1.4),
                 arrowprops=dict(arrowstyle="->", color='#1b4f72', lw=1.1),
                 fontsize=8.5, ha='left', color='#154360', fontweight='medium')

    ax2.annotate('"2 + 2" token\n(as data)', 
                 xy=(x_data[5], y_data[5]), 
                 xytext=(x_data[5]+1.2, y_data[5]+1.4),
                 arrowprops=dict(arrowstyle="->", color='#b9770e', lw=1.1),
                 fontsize=8.5, ha='left', color='#935116', fontweight='medium')

    # Vulnerability note: System & User are collapsed!
    ax2.annotate('Vulnerability: System & User\ncollapsed on same 0° axis!',
                 xy=(x_ghost[3], y_ghost[3]), 
                 xytext=(x_ghost[3]+2.2, y_ghost[3]-1.6),
                 arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.3),
                 fontsize=8.2, ha='left', color='#922b21', fontweight='semibold',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2e9", ec="#f5b7b1", lw=0.8))

    # Legend
    leg2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markeredgecolor='#1b4f72', markersize=6.5, label='Instruction token embeddings'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markeredgecolor='#b9770e', markersize=6.5, label='Data token embeddings (90°)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#95a5a6', linestyle='--', markeredgewidth=1.3, markersize=6.0, label='Original data position')
    ]
    ax2.legend(handles=leg2, loc='upper left', bbox_to_anchor=(0.08, 0.98), 
               frameon=True, framealpha=0.95, facecolor='#fafafa', edgecolor='#d0d7de', fontsize=7.8)

    # =========================================================================
    # Panel (c): HA-ASIDE (Ours)
    # =========================================================================
    ax3 = axes[2]
    draw_axes(ax3)
    
    ax3.set_title("HA-ASIDE (Ours)\n(hierarchical 4-tier rotation + RGAG gating)", 
                  fontsize=12, fontweight='bold', pad=18, linespacing=1.4)

    # 1. System Prompt Tokens (0° reference anchor, shallow angle ~10°)
    x_sys, y_sys = scatter_along_segment((3.0, 0.8), (8.6, 1.8), 14, noise=0.13, seed=404)
    ax3.scatter(x_sys, y_sys, s=44, color='#2c3e50', edgecolors='#1a252f', lw=0.9, zorder=3)

    # 2. User Instruction Tokens (45° rotation, mid angle ~45°)
    x_usr, y_usr = scatter_along_segment((2.4, 2.4), (7.4, 6.8), 14, noise=0.15, seed=505)
    ax3.scatter(x_usr, y_usr, s=44, color='#1b9e77', edgecolors='#0e664e', lw=0.9, zorder=4)

    # 3. Untrusted Data Tokens (90° rotation, steep angle ~82°)
    x_dat, y_dat = scatter_along_segment((1.2, 3.2), (2.0, 6.8), 14, noise=0.14, seed=606)
    ax3.scatter(x_dat, y_dat, s=44, color='#e67e22', edgecolors='#b95e04', lw=0.9, zorder=4)

    # Rotation Arcs from System Anchor
    # Arc 1: System -> User (45°)
    arc_u = patches.Arc((0, 0), 7.8, 7.8, angle=0, theta1=14, theta2=45,
                        color='#1b9e77', linestyle='--', lw=1.5, zorder=2)
    ax3.add_patch(arc_u)
    ax3.annotate('', xy=(x_usr[4], y_usr[4]), xytext=(x_usr[4]+0.5, y_usr[4]-0.5),
                 arrowprops=dict(arrowstyle="-|>", color='#1b9e77', lw=1.5, mutation_scale=11))
    ax3.text(3.8, 2.0, "45° rotation", fontsize=8.0, color='#0e664e', fontweight='semibold', rotation=28)

    # Arc 2: System -> Data (90°)
    arc_d = patches.Arc((0, 0), 9.6, 9.6, angle=0, theta1=15, theta2=75,
                        color='#e67e22', linestyle='--', lw=1.5, zorder=2)
    ax3.add_patch(arc_d)
    ax3.annotate('', xy=(x_dat[5], y_dat[5]), xytext=(x_dat[5]+0.6, y_dat[5]-0.4),
                 arrowprops=dict(arrowstyle="-|>", color='#e67e22', lw=1.5, mutation_scale=11))
    ax3.text(2.3, 4.6, "90° rotation", fontsize=8.0, color='#935116', fontweight='semibold', rotation=45)

    # Callouts in completely open spaces:
    # System Prompt callout placed to the right of the last system token in empty region
    ax3.annotate('System Prompt (0°)\n[Developer Security Anchor]', 
                 xy=(x_sys[-2], y_sys[-2]), 
                 xytext=(x_sys[-2]+0.6, y_sys[-2]+1.6),
                 arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=1.1),
                 fontsize=8.0, ha='center', color='#1a252f', fontweight='semibold')

    # User Instruction callout placed in empty region top-right
    ax3.annotate('User Instruction (45°)\n[Direct Jailbreak Blocked]', 
                 xy=(x_usr[-3], y_usr[-3]), 
                 xytext=(x_usr[-3]+1.6, y_usr[-3]+0.6),
                 arrowprops=dict(arrowstyle="->", color='#0e664e', lw=1.1),
                 fontsize=8.0, ha='left', color='#0e664e', fontweight='semibold')

    # Untrusted Data callout placed in empty space to the right of data dots
    ax3.annotate('Untrusted Data (90°)\n[Indirect Injection Blocked]', 
                 xy=(x_dat[-2], y_dat[-2]), 
                 xytext=(x_dat[-2]+1.6, y_dat[-2]+1.2),
                 arrowprops=dict(arrowstyle="->", color='#b95e04', lw=1.1),
                 fontsize=8.0, ha='left', color='#935116', fontweight='semibold')

    # Legend
    leg3 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c3e50', markeredgecolor='#1a252f', markersize=6.5, label='System / Assistant (0° Anchor)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1b9e77', markeredgecolor='#0e664e', markersize=6.5, label='User Instructions (45° Rotation)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', markeredgecolor='#b95e04', markersize=6.5, label='Untrusted Data (90° Rotation)')
    ]
    ax3.legend(handles=leg3, loc='upper left', bbox_to_anchor=(0.08, 0.98), 
               frameon=True, framealpha=0.95, facecolor='#fafafa', edgecolor='#d0d7de', fontsize=7.8)

    plt.subplots_adjust(wspace=0.18, left=0.04, right=0.98, top=0.88, bottom=0.09)
    
    # Save targets
    out1 = "figures/fig1_ha_aside_architecture.png"
    out2 = "ha-aside-paper/paper/figures/fig1_ha_aside_architecture.png"
    
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    
    fig.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Saved to {out1}")
    print(f"[+] Saved to {out2}")

if __name__ == "__main__":
    create_figure()
