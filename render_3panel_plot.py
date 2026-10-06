import json
import os
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\3e1f8e6b-d38f-462f-a6ae-29020340b04b"

# 1. Load transient relaxation curves (X1 and X3 out to 3.0s)
with open(os.path.join(CURRENT_DIR, "transient_curves_x1_x3.json"), "r", encoding="utf-8") as f:
    transient_data = json.load(f)

# 2. Load 12-regime data for X7 - X12
with open(os.path.join(CURRENT_DIR, "unified_sciml_7seed_results.json"), "r", encoding="utf-8") as f:
    seed_data = json.load(f)

seeds = seed_data["seeds"]
cases_osc = ["X8", "X9", "X10", "X11", "X12"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6.2), dpi=220, gridspec_kw={'width_ratios': [1.25, 1.25, 0.8]})
plt.subplots_adjust(wspace=0.28)

# -----------------------------------------------------------------------------
# Panel A: Transient Relaxation vs. Ground Truth RK4 (X1 & X3)
# -----------------------------------------------------------------------------
# X1: slow relaxation (tau = 2.79s); X3: fast relaxation (tau = 1.16s)
t_x1 = np.array(transient_data["X1"]["t"])
ystar_x1 = transient_data["X1"]["y_star"]
rk4_x1_offset = np.abs(np.array(transient_data["X1"]["rk4"]) - ystar_x1)

t_x3 = np.array(transient_data["X3"]["t"])
ystar_x3 = transient_data["X3"]["y_star"]
rk4_x3_offset = np.abs(np.array(transient_data["X3"]["rk4"]) - ystar_x3)

# Plot RK4 ground truth curves
ax1.plot(t_x1, rk4_x1_offset, color='#1b4965', lw=2.8, label=r"Ground Truth RK4: $X_1$ ($\tau_{\mathrm{phys}} = 2.79\,\mathrm{s}$)")
ax1.plot(t_x3, rk4_x3_offset, color='#b7094c', lw=2.8, label=r"Ground Truth RK4: $X_3$ ($\tau_{\mathrm{phys}} = 1.16\,\mathrm{s}$)")

# Plot Model individual seeds and ensemble mean for X1
x1_seed_trajs = []
for s_str, s_traj in transient_data["X1"]["seeds"].items():
    s_offset = np.abs(np.array(s_traj) - ystar_x1)
    x1_seed_trajs.append(s_offset)
    ax1.plot(t_x1, s_offset, color='#62b6cb', lw=1.0, alpha=0.6, linestyle=':')
x1_mean_offset = np.mean(x1_seed_trajs, axis=0)
ax1.plot(t_x1, x1_mean_offset, color='#006466', lw=2.2, linestyle='--', label=r"SciML Model Ensemble: $X_1$")

# Plot Model individual seeds and ensemble mean for X3
x3_seed_trajs = []
for s_str, s_traj in transient_data["X3"]["seeds"].items():
    s_offset = np.abs(np.array(s_traj) - ystar_x3)
    x3_seed_trajs.append(s_offset)
    ax1.plot(t_x3, s_offset, color='#ff758f', lw=1.0, alpha=0.6, linestyle=':')
x3_mean_offset = np.mean(x3_seed_trajs, axis=0)
ax1.plot(t_x3, x3_mean_offset, color='#890620', lw=2.2, linestyle='--', label=r"SciML Model Ensemble: $X_3$")

# Inset / annotations showing continuous contraction and sub-0.12 mM tracking gap
ax1.annotate(r"$X_1$ @ 3.0s: RK4 = 18.94 mM | Model = 18.89±0.08 mM" + "\n" + r"(Tracking Gap to RK4: 0.05 mM | $\tau_{\mathrm{phys}} = 2.79\,\mathrm{s}$)",
             xy=(0.75, 24.5), fontsize=8.2, fontweight='bold', color='#1b4965',
             bbox=dict(boxstyle="round,pad=0.3", fc='#eaf4f4', ec='#1b4965', alpha=0.92))

ax1.annotate(r"$X_3$ @ 3.0s: RK4 = 2.54 mM | Model = 2.50±0.04 mM" + "\n" + r"(Tracking Gap to RK4: 0.04 mM | $\tau_{\mathrm{phys}} = 1.16\,\mathrm{s}$)",
             xy=(0.85, 7.5), fontsize=8.2, fontweight='bold', color='#b7094c',
             bbox=dict(boxstyle="round,pad=0.3", fc='#fdf0d5', ec='#b7094c', alpha=0.92))

ax1.set_xlabel("Elapsed Time $t$ (seconds)", fontsize=11, fontweight='bold')
ax1.set_ylabel(r"Dynamic Offset from Steady State $|m_1(t) - y^*|$ (mM)", fontsize=11, fontweight='bold')
ax1.set_title("Panel A: Transient Relaxation vs. Ground Truth RK4 ($X_1$ & $X_3$)\n[Active Contraction Towards $y^*$; Neural Tracking Gap $< 0.12$ mM]", fontsize=11.5, fontweight='bold')
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.set_xlim(0, 3.0)
ax1.set_ylim(0, 48.0)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

# -----------------------------------------------------------------------------
# Panel B: Standard Limit Cycles (X8 - X12) — Granular Per-Seed Breakdown
# -----------------------------------------------------------------------------
# 5 cases, 7 seeds per case = grouped bars
n_cases = len(cases_osc)
n_seeds = len(seeds)
bar_width = 0.11
group_positions = np.arange(n_cases)

# Palette for seeds
seed_colors = ['#457b9d', '#2a9d8f', '#e76f51', '#f4a261', '#9d4edd', '#1d3557', '#e63946']

for s_idx, s in enumerate(seeds):
    s_errs = [seed_data["per_seed"][str(s)][cid]["error"] for cid in cases_osc]
    offsets = group_positions + (s_idx - (n_seeds - 1) / 2) * bar_width
    
    # Custom color: if Seed 5555 in X11, highlight in vivid crimson
    bar_cols = []
    for c_idx, e in enumerate(s_errs):
        if cases_osc[c_idx] == "X11" and s == 5555:
            bar_cols.append('#d00000') # Outlier highlight
        else:
            bar_cols.append(seed_colors[s_idx])
            
    bars = ax2.bar(offsets, s_errs, width=bar_width * 0.9, color=bar_cols,
                   edgecolor='black', lw=0.6, alpha=0.88, label=f"Seed {s}")

# Highlight X11 Seed 5555 outlier specifically
x11_idx = cases_osc.index("X11")
x11_s5555_pos = group_positions[x11_idx] + (6 - (n_seeds - 1) / 2) * bar_width
ax2.annotate("Seed 5555:\n0.343%", xy=(x11_s5555_pos, 0.343), xytext=(x11_s5555_pos - 0.35, 0.38),
             arrowprops=dict(arrowstyle="->", color='#d00000', lw=1.5),
             fontsize=8.5, fontweight='bold', color='#d00000', ha='center',
             bbox=dict(boxstyle="round,pad=0.2", fc="#ffebee", ec="#d00000", alpha=0.9))

# Add reference threshold line at 0.20%
ax2.axhline(0.20, color='red', linestyle='--', lw=1.4, alpha=0.8, label="Previous 0.20% Claim")

ax2.set_xticks(group_positions)
ax2.set_xticklabels(cases_osc, fontweight='bold', fontsize=10)
ax2.set_xlabel("Limit-Cycle Regime ($X_8 - X_{12}$)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Peak Amplitude Error (%)", fontsize=11, fontweight='bold')
ax2.set_title("Panel B: Limit Cycles ($X_8 - X_{12}$) Per-Seed Audit\n[Seed 5555 Outlier Reaches 0.343% in $X_{11}$; Others Sub-0.20%]", fontsize=11.5, fontweight='bold')
ax2.grid(True, linestyle=":", alpha=0.5, axis='y')
ax2.set_ylim(0, 0.44)
# Combine seed legend and threshold line
handles, labels = ax2.get_legend_handles_labels()
# Place legend outside or compact
ax2.legend(handles, labels, loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)

# -----------------------------------------------------------------------------
# Panel C: Stiff Relaxation Regime (X7) — Granular Per-Seed Scatter / Bimodality
# -----------------------------------------------------------------------------
x7_errs = [seed_data["per_seed"][str(s)]["X7"]["error"] for s in seeds]

colors_x7 = []
for e in x7_errs:
    if e > 150.0:
        colors_x7.append('#7570b3') # Diverged/Exploded
    elif e >= 99.0:
        colors_x7.append('#e7298a') # Collapsed to 0
    else:
        colors_x7.append('#1b9e77') # Bounded

ax3.bar(np.arange(7), x7_errs, color=colors_x7, alpha=0.85, edgecolor='black', lw=1.2)
for i, (e, s) in enumerate(zip(x7_errs, seeds)):
    lbl = f"{e:.1f}%"
    if e >= 99.0 and e <= 100.0:
        lbl += "\n(Damped 0)"
    elif e > 150.0:
        lbl += "\n(Exploded)"
    ax3.text(i, e + 4.0, lbl, ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3.axhline(np.mean(x7_errs), color='black', linestyle='--', lw=1.5, label=f"Mean: {np.mean(x7_errs):.1f}%\n(Misleading Aggregate)")
ax3.set_xticks(np.arange(7))
ax3.set_xticklabels([str(s) for s in seeds], rotation=35, fontweight='bold', fontsize=9)
ax3.set_xlabel("Random Seed", fontsize=11, fontweight='bold')
ax3.set_ylabel("Amplitude Error (%)", fontsize=11, fontweight='bold')
ax3.set_title("Panel C: Stiff Relaxation ($X_7$)\n[43% Catastrophic Bimodality Across 7 Seeds]", fontsize=11.5, fontweight='bold')
ax3.grid(True, linestyle=":", alpha=0.5, axis='y')
ax3.set_ylim(0, 260.0)
ax3.legend(loc="upper left", fontsize=8.5)

fig.suptitle("Unified Continuous SciML: Rigorous Empirical Auditing Across Fixed Points, Limit Cycles & Stiff Bifurcations\n(Panel A: Ground-Truth Dynamical Relaxation vs RK4; Panel B: Per-Seed Limit Cycles Exposing X11 Outlier; Panel C: X7 Bimodality)", fontsize=13, fontweight='bold', y=1.02)

out_png1 = os.path.join(CURRENT_DIR, "unified_sciml_7seed_replication_rebuilt.png")
out_png2 = os.path.join(BRAIN_DIR, "unified_sciml_7seed_replication_rebuilt.png")
plt.savefig(out_png1, bbox_inches="tight", dpi=220)
plt.savefig(out_png2, bbox_inches="tight", dpi=220)
plt.close()
print(f"Saved rebuilt 3-panel figure to:\n  - {out_png1}\n  - {out_png2}")

